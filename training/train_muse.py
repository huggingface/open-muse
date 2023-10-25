# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import math
import os
import random
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Any, List, Tuple, Union

import numpy as np
import plotly.express as px
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
from data import ClassificationDataset, Text2ImageDataset
from omegaconf import DictConfig, ListConfig, OmegaConf
from optimizer import Lion
from PIL import Image
from torch.optim import AdamW  # why is shampoo not available in PT :(
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5Tokenizer,
)

import muse
import muse.training_utils
from muse import (
    MOVQ,
    EMAModel,
    MaskGitTransformer,
    MaskGiTUViT,
    MaskGitVQGAN,
    PaellaVQModel,
    VQGANModel,
    get_mask_chedule,
)
from muse.lr_schedulers import get_scheduler

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")


def get_config():
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


def flatten_omega_conf(cfg: Any, resolve: bool = False) -> List[Tuple[str, Any]]:
    ret = []

    def handle_dict(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{k1}", v1) for k1, v1 in flatten_omega_conf(value, resolve=resolve)]

    def handle_list(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{idx}", v1) for idx, v1 in flatten_omega_conf(value, resolve=resolve)]

    if isinstance(cfg, DictConfig):
        for k, v in cfg.items_ex(resolve=resolve):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(k, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(k, v, resolve=resolve))
            else:
                ret.append((str(k), v))
    elif isinstance(cfg, ListConfig):
        for idx, v in enumerate(cfg._iter_ex(resolve=resolve)):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(idx, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(idx, v, resolve=resolve))
            else:
                ret.append((str(idx), v))
    else:
        assert False

    return ret


def get_vq_model_class(model_type):
    if model_type == "vqgan":
        return VQGANModel
    elif model_type == "movq":
        return MOVQ
    elif model_type == "maskgit_vqgan":
        return MaskGitVQGAN
    elif model_type == "paella_vq":
        return PaellaVQModel
    else:
        raise ValueError(f"model_type {model_type} not supported for VQGAN")


def soft_target_cross_entropy(logits, targets, soft_targets):
    # ignore the first token from logits and targets (class id token)
    logits = logits[:, 1:]
    targets = targets[:, 1:]

    logits = logits[..., : soft_targets.shape[-1]]

    log_probs = F.log_softmax(logits, dim=-1)
    padding_mask = targets.eq(-100)

    loss = torch.sum(-soft_targets * log_probs, dim=-1)
    loss.masked_fill_(padding_mask, 0.0)

    # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
    num_active_elements = padding_mask.numel() - padding_mask.long().sum()
    loss = loss.sum() / num_active_elements
    return loss


def get_loss_weight(t, mask, min_val=0.3):
    return 1 - (1 - mask) * ((1 - t) * (1 - min_val))[:, None]


def mask_or_random_replace_tokens(image_tokens, mask_id, config, mask_schedule, is_train=True):
    batch_size, seq_len = image_tokens.shape

    if not is_train and config.training.get("eval_mask_ratios", None):
        mask_prob = random.choices(config.training.eval_mask_ratios, k=batch_size)
        mask_prob = torch.tensor(mask_prob, device=image_tokens.device)
    else:
        # Sample a random timestep for each image
        timesteps = torch.rand(batch_size, device=image_tokens.device)
        # Sample a random mask probability for each image using timestep and cosine schedule
        mask_prob = mask_schedule(timesteps)
        mask_prob = mask_prob.clip(config.training.min_masking_rate)

    # creat a random mask for each image
    num_token_masked = (seq_len * mask_prob).round().clamp(min=1)

    mask_contiguous_region_prob = config.training.get("mask_contiguous_region_prob", None)

    if mask_contiguous_region_prob is None:
        mask_contiguous_region = False
    else:
        mask_contiguous_region = random.random() < mask_contiguous_region_prob

    if not mask_contiguous_region:
        batch_randperm = torch.rand(batch_size, seq_len, device=image_tokens.device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
    else:
        resolution = int(seq_len**0.5)
        mask = torch.zeros((batch_size, resolution, resolution), device=image_tokens.device)

        # TODO - would be nice to vectorize
        for batch_idx, num_token_masked_ in enumerate(num_token_masked):
            num_token_masked_ = int(num_token_masked_.item())

            # NOTE: a bit handwavy with the bounds but gets a rectangle of ~num_token_masked_
            num_token_masked_height = random.randint(
                math.ceil(num_token_masked_ / resolution), min(resolution, num_token_masked_)
            )
            num_token_masked_height = min(num_token_masked_height, resolution)

            num_token_masked_width = math.ceil(num_token_masked_ / num_token_masked_height)
            num_token_masked_width = min(num_token_masked_width, resolution)

            start_idx_height = random.randint(0, resolution - num_token_masked_height)
            start_idx_width = random.randint(0, resolution - num_token_masked_width)

            mask[
                batch_idx,
                start_idx_height : start_idx_height + num_token_masked_height,
                start_idx_width : start_idx_width + num_token_masked_width,
            ] = 1

        mask = mask.reshape(batch_size, seq_len)
        mask = mask.to(torch.bool)

    # mask images and create input and labels
    if config.training.get("noise_type", "mask"):
        input_ids = torch.where(mask, mask_id, image_tokens)
    elif config.training.get("noise_type", "random_replace"):
        # sample random tokens from the vocabulary
        random_tokens = torch.randint_like(
            image_tokens, low=0, high=config.model.codebook_size, device=image_tokens.device
        )
        input_ids = torch.where(mask, random_tokens, image_tokens)
    else:
        raise ValueError(f"noise_type {config.training.noise_type} not supported")

    if (
        config.training.get("predict_all_tokens", False)
        or config.training.get("noise_type", "mask") == "random_replace"
    ):
        labels = image_tokens
        loss_weight = get_loss_weight(mask_prob, mask.long())
    else:
        labels = torch.where(mask, image_tokens, -100)
        loss_weight = None

    return input_ids, labels, loss_weight, mask_prob


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,  # It's important to set this to True when using webdataset to get the right number of steps for lr scheduling. If set to False, the number of steps will be devide by the number of processes assuming batches are multiplied by the number of processes
    )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            config.training.batch_size
        )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        muse.logging.set_verbosity_info()
    else:
        muse.logging.set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        resume_wandb_run = config.experiment.resume_from_checkpoint
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint")
        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    is_pre_encode = config.training.get("pre_encode", False)
    if not is_pre_encode:
        if config.model.text_encoder.type == "clip":
            text_encoder_cls = (
                CLIPTextModelWithProjection
                if config.model.transformer.get("add_cond_embeds", False)
                else CLIPTextModel
            )
            text_encoder = text_encoder_cls.from_pretrained(config.model.text_encoder.pretrained, projection_dim=768)
            tokenizer = CLIPTokenizer.from_pretrained(config.model.text_encoder.pretrained)
            if config.model.text_encoder.get("pad_token_id", None):
                tokenizer.pad_token_id = config.model.text_encoder.pad_token_id
        elif config.model.text_encoder.type == "t5":
            text_encoder = T5EncoderModel.from_pretrained(config.model.text_encoder.pretrained)
            tokenizer = T5Tokenizer.from_pretrained(config.model.text_encoder.pretrained)
        else:
            raise ValueError(f"Unknown text model type: {config.model.text_encoder.type}")

        vq_class = get_vq_model_class(config.model.vq_model.type)
        vq_model = vq_class.from_pretrained(config.model.vq_model.pretrained)

        # Freeze the text model and VQGAN
        text_encoder.requires_grad_(False)
        vq_model.requires_grad_(False)
    else:
        text_encoder = None
        tokenizer = None
        vq_model = None

    model_cls = MaskGitTransformer if config.model.get("architecture", "transformer") == "transformer" else MaskGiTUViT
    if config.model.get("pretrained_model_path", None) is not None:
        model = model_cls.from_pretrained(config.model.pretrained_model_path)
    else:
        model = model_cls(**config.model.transformer)
    mask_id = model.config.mask_token_id
    output_size = model.output_size

    # Create EMA
    if config.training.get("use_ema", False):
        ema = EMAModel(
            model.parameters(),
            decay=config.training.ema_decay,
            update_after_step=config.training.ema_update_after_step,
            update_every=config.training.ema_update_every,
            model_cls=model_cls,
            model_config=model.config,
        )

        # Create custom saving and loading hooks so that `accelerator.save_state(...)` serializes in a nice format.
        def load_model_hook(models, input_dir):
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "ema_model"), model_cls=model_cls)
            ema.load_state_dict(load_model.state_dict())
            ema.to(accelerator.device)
            del load_model

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                ema.save_pretrained(os.path.join(output_dir, "ema_model"))

        accelerator.register_load_state_pre_hook(load_model_hook)
        accelerator.register_save_state_pre_hook(save_model_hook)

    # Enable flash attention if asked
    if config.model.enable_xformers_memory_efficient_attention:
        model.enable_xformers_memory_efficient_attention()

    optimizer_config = config.optimizer.params
    learning_rate = optimizer_config.learning_rate
    if optimizer_config.scale_lr:
        learning_rate = (
            learning_rate
            * config.training.batch_size
            * accelerator.num_processes
            * config.training.gradient_accumulation_steps
        )

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer_cls = AdamW
    elif optimizer_type == "fused_adamw":
        if is_apex_available:
            optimizer_cls = apex.optimizers.FusedAdam
        else:
            raise ImportError("Please install apex to use fused_adam")
    elif optimizer_type == "8bit_adamw":
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_cls = bnb.optim.AdamW8bit
    elif optimizer_type == "lion":
        optimizer_cls = Lion
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optimizer_cls(
        optimizer_grouped_parameters,
        lr=optimizer_config.learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        weight_decay=optimizer_config.weight_decay,
        eps=optimizer_config.epsilon,
    )

    # Cretae mask scheduler
    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_chedule(schedule, **args)
    else:
        mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

    ##################################
    # DATLOADER and LR-SCHEDULER     #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    total_batch_size_without_accum = config.training.batch_size * accelerator.num_processes
    total_batch_size = (
        config.training.batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps
    )

    # DataLoaders creation:
    # We use webdataset for data loading. The dataloaders are created with sampling with replacement.
    # We don't do dataset resuming here, instead we resample the shards and buffer each time. The sampling is stochastic.
    # This means that the dataloading is not deterministic, but it's fast and efficient.
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    if config.dataset.type == "classification":
        dataset_cls = partial(
            ClassificationDataset,
            return_text=True,
            imagenet_class_mapping_path=dataset_config.imagenet_class_mapping_path,
        )
    else:
        dataset_cls = Text2ImageDataset

    dataset = dataset_cls(
        train_shards_path_or_url=dataset_config.train_shards_path_or_url,
        eval_shards_path_or_url=dataset_config.eval_shards_path_or_url,
        tokenizer=tokenizer,
        max_seq_length=preproc_config.max_seq_length,
        num_train_examples=config.experiment.max_train_examples,
        per_gpu_batch_size=config.training.batch_size,
        global_batch_size=total_batch_size_without_accum,
        num_workers=dataset_config.num_workers,
        resolution=preproc_config.resolution,
        center_crop=preproc_config.center_crop,
        random_flip=preproc_config.random_flip,
        shuffle_buffer_size=dataset_config.shuffle_buffer_size,
        pin_memory=dataset_config.pin_memory,
        persistent_workers=dataset_config.persistent_workers,
        is_pre_encoded=is_pre_encode,
        vae_checkpoint=config.model.vq_model.pretrained,
        text_encoder_checkpoint=config.model.text_encoder.pretrained,
        use_filtered_dataset=dataset_config.get("use_filtered_dataset", False),
        require_marked_as_ok_by_spawning=dataset_config.get("require_marked_as_ok_by_spawning", False),
        require_marked_as_not_getty=dataset_config.get("require_marked_as_not_getty", False),
        max_pnsfw=dataset_config.get("max_pnsfw", None),
        max_pwatermark=dataset_config.get("max_pwatermark", 0.5),
        min_aesthetic_score=dataset_config.get("min_aesthetic_score", 4.75),
        min_size=dataset_config.get("min_size", 256),
        is_sdxl_synthetic_dataset=dataset_config.get("is_sdxl_synthetic_dataset", False),
        is_ds_clean_upscaled=dataset_config.get("is_ds_clean_upscaled", False),
        is_ds_clean=dataset_config.get("is_ds_clean", False),
    )
    train_dataloader, eval_dataloader = dataset.train_dataloader, dataset.eval_dataloader

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
    )

    # Prepare everything with accelerator
    logger.info("Preparing model, optimizer and dataloaders")
    # The dataloader are already aware of distributed training, so we don't need to prepare them.
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    # TODO: make this configurable
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if not is_pre_encode:
        text_encoder.to(device=accelerator.device, dtype=weight_dtype)
        vq_model.to(device=accelerator.device)
    if config.training.get("use_ema", False):
        ema.to(accelerator.device)

    if not is_pre_encode and config.model.transformer.get("use_empty_embeds_for_uncond", False):
        empty_input = tokenizer("", padding="max_length", return_tensors="pt").input_ids.to(accelerator.device)
        outputs = text_encoder(empty_input, output_hidden_states=True)
        if config.model.transformer.get("add_cond_embeds", False):
            empty_embeds = outputs.hidden_states[-2]
            empty_clip_embeds = outputs[0]
        else:
            empty_embeds = outputs.last_hidden_state
            empty_clip_embeds = None
    else:
        empty_embeds = None
        empty_clip_embeds = None

    if config.training.overfit_one_batch:
        train_dataloader = [next(iter(train_dataloader))]

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(train_dataloader.num_batches / config.training.gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs.
    # Note: We are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = { config.training.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    resume_from_checkpoint = config.experiment.resume_from_checkpoint
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(config.experiment.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
            if path is not None:
                path = os.path.join(config.experiment.output_dir, path)

        if path is None:
            accelerator.print(f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run.")
            resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")

            resume_lr_scheduler = config.experiment.get("resume_lr_scheduler", True)
            dont_resume_optimizer = config.experiment.get("dont_resume_optimizer", False)
            if not resume_lr_scheduler:
                logger.info("Not resuming the lr scheduler.")
                accelerator._schedulers = []  # very hacky, but we don't want to resume the lr scheduler
            if dont_resume_optimizer:
                logger.info("Not resuming the optimizer.")
                accelerator._optimizers = []  # very hacky, but we don't want to resume the optimizer
                grad_scaler = accelerator.scaler
                accelerator.scaler = None

            accelerator.load_state(path)
            if not resume_lr_scheduler:
                accelerator._schedulers = [lr_scheduler]
            if dont_resume_optimizer:
                accelerator._optimizers = [optimizer]
                accelerator.scaler = grad_scaler

            global_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    @torch.no_grad()
    def prepare_inputs_and_labels(
        pixel_values_or_image_ids: Union[torch.FloatTensor, torch.LongTensor],
        text_input_ids_or_embeds: Union[torch.LongTensor, torch.LongTensor],
        min_masking_rate: float = 0.0,
        batch: Any = None,
        is_train: bool = True,
    ):
        if is_pre_encode:
            image_tokens = pixel_values_or_image_ids
            soft_targets = None
        else:
            if config.training.use_soft_code_target and is_train:
                soft_targets, image_tokens = vq_model.get_soft_code(
                    pixel_values_or_image_ids, temp=config.training.soft_code_temp, stochastic=config.training.use_stochastic_code
                )
            else:
                soft_targets = None

                if config.training.get("split_vae_encode", False):
                    split_batch_size = config.training.split_vae_encode
                    # Use a batch of at most split_vae_encode images to encode and then concat the results
                    batch_size = pixel_values_or_image_ids.shape[0]
                    num_splits = math.ceil(batch_size / split_batch_size)
                    image_tokens = []
                    for i in range(num_splits):
                        start_idx = i * split_batch_size
                        end_idx = min((i + 1) * split_batch_size, batch_size)
                        image_tokens.append(vq_model.get_code(pixel_values_or_image_ids[start_idx:end_idx]))
                    image_tokens = torch.cat(image_tokens, dim=0)
                else:
                    image_tokens = vq_model.get_code(pixel_values_or_image_ids)

        if not is_pre_encode:
            if config.model.transformer.get("add_cond_embeds", False):
                outputs = text_encoder(text_input_ids_or_embeds, return_dict=True, output_hidden_states=True)
                encoder_hidden_states = outputs.hidden_states[-2]
                clip_embeds = outputs[0]
            else:
                encoder_hidden_states = text_encoder(text_input_ids_or_embeds)[0]
                clip_embeds = None

            if config.model.transformer.get("add_micro_cond_embeds", False):
                original_sizes = list(map(list, zip(*batch["orig_size"])))
                crop_coords = list(map(list, zip(*batch["crop_coords"])))
                aesthetic_scores = batch["aesthetic_score"]
                micro_conds = torch.cat(
                    [torch.tensor(original_sizes), torch.tensor(crop_coords), aesthetic_scores.unsqueeze(-1)], dim=-1
                )
                micro_conds = micro_conds.to(
                    encoder_hidden_states.device, dtype=encoder_hidden_states.dtype, non_blocking=True
                )
            else:
                micro_conds = None
        else:
            encoder_hidden_states = text_input_ids_or_embeds
            clip_embeds = None

        # create MLM mask and labels
        input_ids, labels, loss_weight, mask_prob = mask_or_random_replace_tokens(
            image_tokens,
            mask_id,
            config,
            mask_schedule=mask_schedule,
            is_train=is_train,
        )
        return input_ids, encoder_hidden_states, labels, soft_targets, mask_prob, loss_weight, clip_embeds, micro_conds

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    # As stated above, we are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        for batch in train_dataloader:
            # TODO(Patrick) - We could definitely pre-compute the image tokens for faster training on larger datasets
            if is_pre_encode:
                pixel_values, input_ids = batch["image_input_ids"], batch["encoder_hidden_states"]
            else:
                pixel_values, input_ids = batch["image"], batch["input_ids"]

            pixel_values = pixel_values.to(accelerator.device, non_blocking=True)
            input_ids = input_ids.to(accelerator.device, non_blocking=True)
            data_time_m.update(time.time() - end)

            # encode images to image tokens, mask them and create input and labels
            (
                input_ids,
                encoder_hidden_states,
                labels,
                soft_targets,
                mask_prob,
                loss_weight,
                clip_embeds,
                micro_conds,
            ) = prepare_inputs_and_labels(pixel_values, input_ids, config.training.min_masking_rate, batch=batch)

            # log the inputs for the first step of the first epoch
            if global_step == 0 and epoch == 0:
                logger.info("Input ids: {}".format(input_ids))
                logger.info("Labels: {}".format(labels))

            if config.training.cond_dropout_prob > 0.0:
                assert encoder_hidden_states is not None

                batch_size = encoder_hidden_states.shape[0]

                mask = (
                    torch.zeros((batch_size, 1, 1), device=encoder_hidden_states.device).float().uniform_(0, 1)
                    < config.training.cond_dropout_prob
                )

                empty_embeds_ = empty_embeds.expand(batch_size, -1, -1)
                encoder_hidden_states = torch.where(
                    (encoder_hidden_states * mask).bool(), encoder_hidden_states, empty_embeds_
                )

                empty_clip_embeds_ = empty_clip_embeds.expand(batch_size, -1)
                cond_embeds = torch.where((clip_embeds * mask.squeeze(-1)).bool(), clip_embeds, empty_clip_embeds_)

            # Train Step
            with accelerator.accumulate(model):
                if config.training.use_soft_code_target:
                    logits = model(
                        input_ids=input_ids,
                        encoder_hidden_states=encoder_hidden_states,
                    )
                    loss = soft_target_cross_entropy(logits, labels, soft_targets)
                else:
                    logits, loss = model(
                        input_ids=input_ids,
                        encoder_hidden_states=encoder_hidden_states,
                        labels=labels,
                        label_smoothing=config.training.label_smoothing,
                        cond_embeds=cond_embeds,
                        loss_weight=loss_weight,
                        micro_conds=micro_conds,
                    )

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(config.training.batch_size)).mean()
                avg_masking_rate = accelerator.gather(mask_prob.repeat(config.training.batch_size)).mean()

                accelerator.backward(loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

                # log gradient norm before zeroing it
                if (
                    accelerator.sync_gradients
                    and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                    and accelerator.is_main_process
                ):
                    log_grad_norm(model, accelerator, global_step + 1)

                if optimizer_type == "fused_adamw":
                    optimizer.zero_grad()
                else:
                    optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if config.training.get("use_ema", False):
                    ema.step(model.parameters())

                batch_time_m.update(time.time() - end)
                end = time.time()

                # Log metrics
                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                        config.training.gradient_accumulation_steps * config.training.batch_size / batch_time_m.val
                    )
                    logs = {
                        "step_loss": avg_loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "avg_masking_rate": avg_masking_rate.item(),
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
                    accelerator.log(logs, step=global_step + 1)

                    logger.info(
                        f"Step: {global_step + 1} "
                        f"Loss: {avg_loss.item():0.4f} "
                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                    )

                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()

                if (
                    ("log_pixel_entropy_every" in config.experiment)
                    and ((global_step + 1) % config.experiment.log_pixel_entropy_every == 0)
                    and accelerator.is_main_process
                ):
                    log_pixel_entropy(logits, input_ids, mask_id, accelerator, global_step + 1)

                if (
                    ("log_image_entropy_every" in config.experiment)
                    and ((global_step + 1) % config.experiment.log_image_entropy_every == 0)
                    and accelerator.is_main_process
                ):
                    log_image_entropy(logits, input_ids, mask_id, accelerator, global_step + 1)

                if (
                    ("log_cross_entropy_every" in config.experiment)
                    and ((global_step + 1) % config.experiment.log_cross_entropy_every == 0)
                    and accelerator.is_main_process
                ):
                    log_cross_entropy(
                        logits,
                        labels,
                        input_ids,
                        mask_id,
                        output_size,
                        config.training.label_smoothing,
                        accelerator,
                        global_step + 1,
                    )

                if (
                    ("log_token_probability_distributions_every" in config.experiment)
                    and ((global_step + 1) % config.experiment.log_token_probability_distributions_every == 0)
                    and accelerator.is_main_process
                ):
                    log_token_probability_distributions(logits, input_ids, mask_id, accelerator, global_step + 1)

                # Save model checkpoint
                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step + 1)

                # Evaluate model on main process
                if (global_step + 1) % config.experiment.eval_every == 0 and accelerator.is_main_process:
                    # Store the model parameters temporarily and load the EMA parameters to perform inference.
                    if config.training.get("use_ema", False):
                        ema.store(model.parameters())
                        ema.copy_to(model.parameters())

                    validate_model(
                        model,
                        eval_dataloader,
                        accelerator,
                        global_step + 1,
                        prepare_inputs_and_labels,
                        config.experiment.get("max_eval_examples", None),
                    )

                    if config.training.get("use_ema", False):
                        # Switch back to the original model parameters for training.
                        ema.restore(model.parameters())

                # Generate images
                if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                    # Store the model parameters temporarily and load the EMA parameters to perform inference.
                    if config.training.get("use_ema", False):
                        ema.store(model.parameters())
                        ema.copy_to(model.parameters())

                    generate_images(
                        model,
                        vq_model,
                        text_encoder,
                        tokenizer,
                        accelerator,
                        config,
                        global_step + 1,
                        mask_schedule=mask_schedule,
                        empty_embeds=empty_embeds,
                        empty_clip_embeds=empty_clip_embeds,
                    )

                    generate_inpainting_images(
                        model,
                        vq_model,
                        text_encoder,
                        tokenizer,
                        accelerator,
                        config,
                        global_step + 1,
                        mask_schedule=mask_schedule,
                        empty_embeds=empty_embeds,
                        empty_clip_embeds=empty_clip_embeds,
                    )

                    if config.training.get("use_ema", False):
                        # Switch back to the original model parameters for training.
                        ema.restore(model.parameters())

                global_step += 1
                # TODO: Add generation

            # Stop training if max steps is reached
            if global_step >= config.training.max_train_steps:
                break
        # End for

    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    if accelerator.is_main_process:
        validate_model(
            model,
            eval_dataloader,
            accelerator,
            global_step,
            prepare_inputs_and_labels,
            config.experiment.get("max_eval_examples", None),
        )
    save_checkpoint(model, config, accelerator, global_step)

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if config.training.get("use_ema", False):
            ema.copy_to(model.parameters())
        model.save_pretrained(config.experiment.output_dir)

    accelerator.end_training()


@torch.no_grad()
def validate_model(
    model,
    eval_dataloader,
    accelerator,
    global_step,
    prepare_inputs_and_labels,
    max_eval_examples=None,
):
    logger.info("Evaluating...")
    model.eval()
    eval_loss = 0
    now = time.time()

    samples_taken = 0

    for i, batch in enumerate(eval_dataloader):
        pixel_values, input_ids = batch["image"], batch["input_ids"]
        pixel_values = pixel_values.to(accelerator.device, non_blocking=True)
        input_ids = input_ids.to(accelerator.device, non_blocking=True)
        (
            input_ids,
            encoder_hidden_states,
            labels,
            _,
            _,
            loss_weight,
            clip_embeds,
            micro_conds,
        ) = prepare_inputs_and_labels(pixel_values, input_ids, batch=batch, is_train=False)
        _, loss = model(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            labels=labels,
            cond_embeds=clip_embeds,
            loss_weight=loss_weight,
            micro_conds=micro_conds,
        )
        eval_loss += loss.mean()

        samples_taken += input_ids.shape[0]

        if max_eval_examples is not None and samples_taken >= max_eval_examples:
            break

    eval_loss = eval_loss / (i + 1)
    eval_time = time.time() - now

    logger.info(f"Step: {global_step} Eval Loss: {eval_loss.item():0.4f} Eval time: {eval_time:0.2f} s")
    accelerator.log({"eval_loss": eval_loss.item()}, step=global_step)
    model.train()


@torch.no_grad()
def generate_images(
    model,
    vq_model,
    text_encoder,
    tokenizer,
    accelerator,
    config,
    global_step,
    mask_schedule,
    empty_embeds=None,
    empty_clip_embeds=None,
):
    logger.info("Generating images...")
    model.eval()
    # fmt: off
    imagenet_class_names = ['jay', 'castle', 'coffee mug', 'desk', 'Eskimo dog,  husky', 'valley,  vale', 'red wine', 'coral reef', 'mixing bowl', 'cleaver,  meat cleaver,  chopper', 'vine snake', 'bloodhound,  sleuthhound', 'barbershop', 'ski', 'otter', 'snowmobile']
    # fmt: on

    # read validation prompts from file
    if config.dataset.params.validation_prompts_file is not None:
        with open(config.dataset.params.validation_prompts_file, "r") as f:
            validation_prompts = f.read().splitlines()
    else:
        validation_prompts = imagenet_class_names

    if config.training.get("pre_encode", False):
        if config.model.text_encoder.type == "clip":
            text_encoder = CLIPTextModel.from_pretrained(config.model.text_encoder.pretrained)
            tokenizer = CLIPTokenizer.from_pretrained(config.model.text_encoder.pretrained)
        elif config.model.text_encoder.type == "t5":
            text_encoder = T5EncoderModel.from_pretrained(config.model.text_encoder.pretrained)
            tokenizer = T5Tokenizer.from_pretrained(config.model.text_encoder.pretrained)
        else:
            raise ValueError(f"Unknown text model type: {config.model.text_encoder.type}")

        vq_class = get_vq_model_class(config.model.vq_model.type)
        vq_model = vq_class.from_pretrained(config.model.vq_model.pretrained)

        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        text_encoder.to(device=accelerator.device, dtype=weight_dtype)
        vq_model.to(accelerator.device)

    input_ids = tokenizer(
        validation_prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config.dataset.preprocessing.max_seq_length,
    ).input_ids

    if config.model.transformer.get("add_cond_embeds", False):
        outputs = text_encoder(input_ids.to(accelerator.device), return_dict=True, output_hidden_states=True)
        encoder_hidden_states = outputs.hidden_states[-2]
        clip_embeds = outputs[0]
    else:
        encoder_hidden_states = text_encoder(input_ids.to(accelerator.device)).last_hidden_state
        clip_embeds = None

    if config.model.transformer.get("add_micro_cond_embeds", False):
        resolution = config.dataset.preprocessing.resolution
        micro_conds = torch.tensor(
            [resolution, resolution, 0, 0, 6], device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype
        )
        micro_conds = micro_conds.unsqueeze(0).repeat(encoder_hidden_states.shape[0], 1)

    if config.training.get("pre_encode", False):
        del text_encoder

    with torch.autocast("cuda", dtype=encoder_hidden_states.dtype, enabled=accelerator.mixed_precision != "no"):
        # Generate images
        gen_token_ids = accelerator.unwrap_model(model).generate2(
            encoder_hidden_states=encoder_hidden_states,
            cond_embeds=clip_embeds,
            empty_embeds=empty_embeds,
            empty_cond_embeds=empty_clip_embeds,
            micro_conds=micro_conds,
            guidance_scale=config.training.guidance_scale,
            temperature=config.training.get("generation_temperature", 1.0),
            timesteps=config.training.generation_timesteps,
            noise_schedule=mask_schedule,
            noise_type=config.training.get("noise_type", "mask"),
            predict_all_tokens=config.training.get("predict_all_tokens", False),
            seq_len=config.model.transformer.num_vq_tokens,
        )
    # In the beginning of training, the model is not fully trained and the generated token ids can be out of range
    # so we clamp them to the correct range.
    gen_token_ids = torch.clamp(gen_token_ids, max=accelerator.unwrap_model(model).config.codebook_size - 1)
    
    if config.training.get("split_vae_encode", False):
        split_batch_size = config.training.split_vae_encode
        # Use a batch of at most split_vae_encode images to encode and then concat the results
        batch_size = gen_token_ids.shape[0]
        num_splits = math.ceil(batch_size / split_batch_size)
        images = []
        for i in range(num_splits):
            start_idx = i * split_batch_size
            end_idx = min((i + 1) * split_batch_size, batch_size)
            images.append(vq_model.decode_code(gen_token_ids[start_idx:end_idx]))
        images = torch.cat(images, dim=0)
    else:
        images = vq_model.decode_code(gen_token_ids)
    
    model.train()

    if config.training.get("pre_encode", False):
        del vq_model

    # Convert to PIL images
    images = 2.0 * images - 1.0
    images = torch.clamp(images, -1.0, 1.0)
    images = (images + 1.0) / 2.0
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]

    # Log images
    wandb_images = [wandb.Image(image, caption=validation_prompts[i]) for i, image in enumerate(pil_images)]
    wandb.log({"generated_images": wandb_images}, step=global_step)


@torch.no_grad()
def generate_inpainting_images(
    model,
    vq_model,
    text_encoder,
    tokenizer,
    accelerator,
    config,
    global_step,
    mask_schedule,
    empty_embeds=None,
    empty_clip_embeds=None,
):
    assert not config.training.get("pre_encode", False)

    model.eval()

    mask_token_id = config.model.transformer.vocab_size - 1

    validation_prompts, validation_images, validation_masks = inpainting_validation_data()

    validation_masks = validation_masks_to_latent_tensors(validation_masks).to(accelerator.device)

    validation_images = torch.stack([TF.to_tensor(x) for x in validation_images])
    validation_images = validation_images.to(accelerator.device)
    _, validation_images = vq_model.encode(validation_images)
    validation_images[validation_masks] = mask_token_id

    token_input_ids = tokenizer(
        validation_prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config.dataset.preprocessing.max_seq_length,
    ).input_ids

    if config.model.transformer.get("add_cond_embeds", False):
        outputs = text_encoder(token_input_ids.to(accelerator.device), return_dict=True, output_hidden_states=True)
        encoder_hidden_states = outputs.hidden_states[-2]
        clip_embeds = outputs[0]
    else:
        encoder_hidden_states = text_encoder(token_input_ids.to(accelerator.device)).last_hidden_state
        clip_embeds = None

    if config.model.transformer.get("add_micro_cond_embeds", False):
        resolution = config.dataset.preprocessing.resolution
        micro_conds = torch.tensor(
            [resolution, resolution, 0, 0, 6], device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype
        )
        micro_conds = micro_conds.unsqueeze(0).repeat(encoder_hidden_states.shape[0], 1)

    with torch.autocast("cuda", dtype=encoder_hidden_states.dtype, enabled=accelerator.mixed_precision != "no"):
        # Generate images
        gen_token_ids = accelerator.unwrap_model(model).generate2(
            input_ids=validation_images,
            encoder_hidden_states=encoder_hidden_states,
            cond_embeds=clip_embeds,
            empty_embeds=empty_embeds,
            empty_cond_embeds=empty_clip_embeds,
            micro_conds=micro_conds,
            guidance_scale=config.training.guidance_scale,
            temperature=config.training.get("generation_temperature", 1.0),
            timesteps=config.training.generation_timesteps,
            noise_schedule=mask_schedule,
            noise_type=config.training.get("noise_type", "mask"),
            predict_all_tokens=config.training.get("predict_all_tokens", False),
        )
    # In the beginning of training, the model is not fully trained and the generated token ids can be out of range
    # so we clamp them to the correct range.
    gen_token_ids = torch.clamp(gen_token_ids, max=accelerator.unwrap_model(model).config.codebook_size - 1)

    if config.training.get("split_vae_encode", False):
        split_batch_size = config.training.split_vae_encode
        # Use a batch of at most split_vae_encode images to decode and then concat the results
        batch_size = gen_token_ids.shape[0]
        num_splits = math.ceil(batch_size / split_batch_size)
        images = []
        for i in range(num_splits):
            start_idx = i * split_batch_size
            end_idx = min((i + 1) * split_batch_size, batch_size)
            images.append(vq_model.decode_code(gen_token_ids[start_idx:end_idx]))
        images = torch.cat(images, dim=0)
    else:
        images = vq_model.decode_code(gen_token_ids)

    # Convert to PIL images
    images = 2.0 * images - 1.0
    images = torch.clamp(images, -1.0, 1.0)
    images = (images + 1.0) / 2.0
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]

    # Log images
    wandb_images = [wandb.Image(image, caption=validation_prompts[i]) for i, image in enumerate(pil_images)]
    wandb.log({"generated_inpainting_images": wandb_images}, step=global_step)

    model.train()


def inpainting_validation_data():
    validation_prompts = []
    validation_images = []
    validation_masks = []

    for folder_name in os.listdir("./inpainting_validation"):
        validation_prompts.append(folder_name)

        image = None
        mask = None

        for file_name in os.listdir(f"./inpainting_validation/{folder_name}"):
            if file_name.startswith("image"):
                image = Image.open(f"./inpainting_validation/{folder_name}/{file_name}")

            if file_name.startswith("mask"):
                mask = Image.open(f"./inpainting_validation/{folder_name}/{file_name}").convert("L")

        assert image is not None, f"could not find inpainting validation image under {folder_name}"
        assert mask is not None, f"could not find inpainting validation mask under {folder_name}"

        validation_images.append(image)
        validation_masks.append(mask)

    return validation_prompts, validation_images, validation_masks


def validation_masks_to_latent_tensors(validation_masks):
    validation_masks_ = []

    for mask in validation_masks:
        mask = mask.resize((mask.height // 16, mask.width // 16))
        mask = np.array(mask)
        mask = mask / 255
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = mask.reshape(-1)
        mask = mask.astype(bool)
        validation_masks_.append(mask)

    validation_masks_ = np.stack(validation_masks_)

    return torch.from_numpy(validation_masks_)


def save_checkpoint(model, config, accelerator, global_step):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    # retrieve the model on all processes for deepspeed stage 3 to work then save on one process (we are not using stage 3 yet)
    # XXX: could also make this conditional on deepspeed
    state_dict = accelerator.get_state_dict(model)

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")

    accelerator.save_state(save_path)
    logger.info(f"Saved state to {save_path}")


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


@torch.no_grad()
def log_pixel_entropy(logits, input_ids, mask_id, accelerator, global_step):
    pixel_entropy_per_percent_masked_bucket = muse.training_utils.pixel_entropy_per_percent_masked_bucket(
        logits, input_ids, mask_id
    )

    entropy_log = {}

    for bucket, bucket_entropy in enumerate(pixel_entropy_per_percent_masked_bucket):
        bucket_entropy = bucket_entropy.item()
        if bucket_entropy != 0:
            entropy_log[f"bucket {bucket}"] = bucket_entropy

    accelerator.log({"pixel_entropy/stats": entropy_log}, step=global_step)


@torch.no_grad()
def log_image_entropy(logits, input_ids, mask_id, accelerator, global_step):
    image_entropy_per_percent_masked_bucket = muse.training_utils.image_entropy_per_percent_masked_bucket(
        logits, input_ids, mask_id
    )

    entropy_log = {}

    for bucket, bucket_entropy in enumerate(image_entropy_per_percent_masked_bucket):
        bucket_entropy = bucket_entropy.item()
        if bucket_entropy != 0:
            entropy_log[f"bucket {bucket}"] = bucket_entropy

    accelerator.log({"image_entropy/stats": entropy_log}, step=global_step)


@torch.no_grad()
def log_cross_entropy(logits, labels, input_ids, mask_id, output_size, label_smoothing, accelerator, global_step):
    cross_entropy_per_percent_masked_bucket = muse.training_utils.cross_entropy_per_percent_masked_bucket(
        logits, labels, input_ids, mask_id, output_size, label_smoothing
    )

    cross_entropy_log = {}

    for bucket, bucket_cross_entropy in enumerate(cross_entropy_per_percent_masked_bucket):
        bucket_cross_entropy = bucket_cross_entropy.item()
        if bucket_cross_entropy != 0:
            cross_entropy_log[f"bucket {bucket}"] = bucket_cross_entropy

    accelerator.log({"cross entropy/strats": cross_entropy_log}, step=global_step)


@torch.no_grad()
def log_token_probability_distributions(logits, input_ids, mask_id, accelerator, global_step):
    token_probability_distributions = muse.training_utils.token_probability_distributions_per_percent_masked_bucket(
        logits, input_ids, mask_id
    )

    token_probability_distributions_fig = px.histogram(
        token_probability_distributions,
        x="masked_pixel_prob",
        color="bucket",
        color_discrete_sequence=px.colors.qualitative.Plotly,
        marginal="rug",
    )

    accelerator.log({"token_probability_distributions/stats": token_probability_distributions_fig}, step=global_step)


if __name__ == "__main__":
    main()
