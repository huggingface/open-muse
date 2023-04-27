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
import time
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
from data import ClassificationDataset
from omegaconf import DictConfig, ListConfig, OmegaConf
from optimizer import Lion
from PIL import Image
from torch.optim import AdamW  # why is shampoo not available in PT :(

import muse
from muse import MOVQ, MaskGitTransformer, MaskGitVQGAN
from muse.lr_schedulers import get_scheduler
from muse.sampling import cosine_schedule

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
    if model_type == "movq":
        return MOVQ
    elif model_type == "maskgit_vqgan":
        return MaskGitVQGAN
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
        logging_dir=config.experiment.logging_dir,
        split_batches=True,  # It's important to set this to True when using webdataset to get the right number of steps for lr scheduling. If set to False, the number of steps will be devide by the number of processes assuming batches are multiplied by the number of processes.
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

    vq_class = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_class.from_pretrained(config.model.vq_model.pretrained)
    model = MaskGitTransformer(**config.model.transformer)
    mask_id = model.config.mask_token_id

    # Freeze the VQGAN
    vq_model.requires_grad_(False)

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
    elif optimizer_type == "lion":
        optimizer_cls = Lion
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    optimizer = optimizer_cls(
        model.parameters(),
        lr=optimizer_config.learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        weight_decay=optimizer_config.weight_decay,
        eps=optimizer_config.epsilon,
    )

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
    dataset = ClassificationDataset(
        train_shards_path_or_url=dataset_config.train_shards_path_or_url,
        eval_shards_path_or_url=dataset_config.eval_shards_path_or_url,
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
    vq_model.to(accelerator.device)

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
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Instantaneous batch size per device = { config.training.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
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
            path = os.path.join(config.experiment.output_dir, path)

        if path is None:
            accelerator.print(f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run.")
            resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")

            resume_lr_scheduler = config.experiment.get("resume_lr_scheduler", True)
            if not resume_lr_scheduler:
                logger.info("Not resuming the lr scheduler.")
                accelerator._schedulers = []  # very hacky, but we don't want to resume the lr scheduler
            accelerator.load_state(path)
            accelerator.wait_for_everyone()
            if not resume_lr_scheduler:
                accelerator._schedulers = [lr_scheduler]
            global_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    @torch.no_grad()
    def prepare_inputs_and_labels(
        pixel_values: torch.FloatTensor,
        class_ids: torch.LongTensor,
        min_masking_rate: float = 0.0,
        is_train: bool = True,
    ):
        if config.training.use_soft_code_target and is_train:
            soft_targets, image_tokens = vq_model.get_soft_code(
                pixel_values, temp=config.training.soft_code_temp, stochastic=config.training.use_stochastic_code
            )
        else:
            image_tokens = vq_model.encode(pixel_values)[1]
            soft_targets = None

        batch_size, seq_len = image_tokens.shape
        # TODO(Patrick) - I don't think that's how the timesteps are sampled in maskgit or MUSE
        # Sample a random timestep for each image
        timesteps = torch.rand(batch_size, device=image_tokens.device)
        # Sample a random mask probability for each image using timestep and cosine schedule
        mask_prob = cosine_schedule(timesteps)
        mask_prob = mask_prob.clip(min_masking_rate)
        # creat a random mask for each image
        num_token_masked = (seq_len * mask_prob).round().clamp(min=1)
        batch_randperm = torch.rand(batch_size, seq_len, device=image_tokens.device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
        # mask images and create input and labels
        input_ids = torch.where(mask, mask_id, image_tokens)
        labels = torch.where(mask, image_tokens, -100)

        # shift the class ids by codebook size
        class_ids = class_ids + vq_model.num_embeddings
        # prepend the class ids to the image tokens
        input_ids = torch.cat([class_ids.unsqueeze(-1), input_ids], dim=-1)
        # prepend -100 to the labels as we don't want to predict the class ids
        labels_mask = torch.ones_like(class_ids, device=image_tokens.device).unsqueeze(-1).fill_(-100)
        labels = torch.cat([labels_mask, labels], dim=-1)
        return input_ids, labels, soft_targets, mask_prob

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    # As stated above, we are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        for batch in train_dataloader:
            # TODO(Patrick) - We could definitely pre-compute the image tokens for faster training on larger datasets
            pixel_values, class_ids = batch
            pixel_values = pixel_values.to(accelerator.device, non_blocking=True)
            class_ids = class_ids.to(accelerator.device, non_blocking=True)
            data_time_m.update(time.time() - end)

            # encode images to image tokens, mask them and create input and labels
            input_ids, labels, soft_targets, mask_prob = prepare_inputs_and_labels(
                pixel_values, class_ids, config.training.min_masking_rate
            )

            # log the inputs for the first step of the first epoch
            if global_step == 0 and epoch == 0:
                logger.info("Input ids: {}".format(input_ids))
                logger.info("Labels: {}".format(labels))

            # Train Step
            with accelerator.accumulate(model):
                if config.training.use_soft_code_target:
                    logits = model(input_ids=input_ids)
                    loss = soft_target_cross_entropy(logits, labels, soft_targets)
                else:
                    _, loss = model(
                        input_ids=input_ids, labels=labels, label_smoothing=config.training.label_smoothing
                    )
                # Gather thexd losses across all processes for logging (if we use distributed training).
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

                # Evaluate model on main process
                if (global_step + 1) % config.experiment.eval_every == 0 and accelerator.is_main_process:
                    validate_model(model, eval_dataloader, accelerator, global_step + 1, prepare_inputs_and_labels)

                # Save model checkpoint
                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step + 1)

                # Generate images
                if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                    generate_images(model, vq_model, accelerator, global_step + 1)

                global_step += 1
                # TODO: Add generation

            # Stop training if max steps is reached
            if global_step >= config.training.max_train_steps:
                break
        # End for

    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    if accelerator.is_main_process:
        validate_model(model, eval_dataloader, accelerator, global_step, prepare_inputs_and_labels)
    save_checkpoint(model, config, accelerator, global_step)

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(config.experiment.output_dir)

    accelerator.end_training()


@torch.no_grad()
def validate_model(model, eval_dataloader, accelerator, global_step, prepare_inputs_and_labels):
    logger.info("Evaluating...")
    model.eval()
    eval_loss = 0
    now = time.time()
    for i, batch in enumerate(eval_dataloader):
        pixel_values, class_ids = batch
        pixel_values = pixel_values.to(accelerator.device, non_blocking=True)
        class_ids = class_ids.to(accelerator.device, non_blocking=True)
        input_ids, labels, _, _ = prepare_inputs_and_labels(pixel_values, class_ids, is_train=False)
        _, loss = model(input_ids=input_ids, labels=labels)
        eval_loss += loss.mean()
    eval_loss = eval_loss / (i + 1)
    eval_time = time.time() - now

    logger.info(f"Step: {global_step} Eval Loss: {eval_loss.item():0.4f} Eval time: {eval_time:0.2f} s")
    accelerator.log({"eval_loss": eval_loss.item()}, step=global_step)
    model.train()


@torch.no_grad()
def generate_images(model, vq_model, accelerator, global_step):
    logger.info("Generating images...")
    # fmt: off
    imagenet_class_names = ["Jay", "Castle", "coffee mug", "desk", "Husky", "Valley", "Red wine", "Coral reef", "Mixing bowl", "Cleaver", "Vine Snake", "Bloodhound", "Barbershop", "Ski", "Otter", "Snowmobile"]
    # fmt: on
    imagenet_class_ids = torch.tensor(
        [17, 483, 504, 526, 248, 979, 966, 973, 659, 499, 59, 163, 424, 795, 360, 802],
        device=accelerator.device,
        dtype=torch.long,
    )

    # Generate images
    model.eval()
    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    with torch.autocast("cuda", dtype=dtype, enabled=accelerator.mixed_precision != "no"):
        gen_token_ids = accelerator.unwrap_model(model).generate2(imagenet_class_ids, timesteps=8)
    # In the beginning of training, the model is not fully trained and the generated token ids can be out of range
    # so we clamp them to the correct range.
    gen_token_ids = torch.clamp(gen_token_ids, max=accelerator.unwrap_model(model).config.codebook_size - 1)
    images = vq_model.decode_code(gen_token_ids)
    model.train()

    # Convert to PIL images
    images = 2.0 * images - 1.0
    images = torch.clamp(images, -1.0, 1.0)
    images = (images + 1.0) / 2.0
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]

    # Log images
    wandb_images = [wandb.Image(image, caption=imagenet_class_names[i]) for i, image in enumerate(pil_images)]
    wandb.log({"generated_images": wandb_images}, step=global_step)


def save_checkpoint(model, config, accelerator, global_step):
    save_path = Path(config.experiment.output_dir) / f"checkpoint-{global_step}"

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


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


if __name__ == "__main__":
    main()
