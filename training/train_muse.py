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
from functools import partial
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from data import ClassificationDataset, Text2ImageDataset
from omegaconf import DictConfig, ListConfig, OmegaConf
from optimizer import Lion
from PIL import Image
from torch.optim import AdamW  # why is shampoo not available in PT :(
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

import muse
from muse import MaskGitTransformer, MaskGitVQGAN
from muse.lr_schedulers import get_scheduler
from muse.sampling import cosine_schedule

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


# create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
def save_model_hook(models, weights, output_dir):
    for model in models:
        model.save_pretrained(output_dir)
        # make sure to pop weight so that corresponding model is not saved again
        weights.pop()


def load_model_hook(models, input_dir):
    while len(models) > 0:
        # pop models so that they are not loaded again
        model = models.pop()

        # load muse style into model
        load_model = MaskGitTransformer.from_pretrained(input_dir)
        model.register_to_config(**load_model.config)

        model.load_state_dict(load_model.state_dict())
        del load_model


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

    config.experiment.logging_dir = Path(config.experiment.output_dir) / "logs"
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        logging_dir=config.experiment.logging_dir,
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
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.name,
            id=run_id,
            resume=config.experiment.resume_from_checkpoint,
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

    if config.model.text_model.type == "clip":
        text_model = CLIPTextModel.from_pretrained(config.model.text_model.pretrained)
        tokenizer = CLIPTokenizer.from_pretrained(config.model.text_model.pretrained)
    elif config.model.text_model.type == "t5":
        text_model = T5EncoderModel.from_pretrained(config.model.text_model.pretrained)
        tokenizer = T5Tokenizer.from_pretrained(config.model.text_model.pretrained)
    else:
        raise ValueError(f"Unknown text model type: {config.model.text_model.type}")

    vq_model = MaskGitVQGAN.from_pretrained(config.model.vq_model.pretrained)
    model = MaskGitTransformer(**config.model.transformer)
    mask_id = model.config.mask_token_id

    # Freeze the text model and VQGAN
    text_model.requires_grad_(False)
    vq_model.requires_grad_(False)

    # Enable flash attention if asked
    if config.model.enable_xformers_memory_efficient_attention:
        model.enable_xformers_memory_efficient_attention()

    # Create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

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

    if config.dataset.type == "claasification":
        dataset_cls = partial(ClassificationDataset, return_text=True)
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
    )
    train_dataloader, eval_dataloader = dataset.train_dataloader, dataset.eval_dataloader

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps * config.training.gradient_accumulation_steps,
    )

    # Prepare everything with accelerator
    logger.info("Preparing model, optimizer and dataloaders")
    # The dataloader are already aware of distributed training, so we don't need to prepare them.
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    vq_model.to(accelerator.device)
    text_model.to(accelerator.device)

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
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(config.experiment.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run.")
            resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(config.experiment.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    @torch.no_grad()
    def prepare_inputs_and_labels(
        pixel_values: torch.FloatTensor, input_ids: torch.LongTensor, min_masking_rate: float = 0.0
    ):
        image_tokens = vq_model.encode(pixel_values)[1]
        encoder_hidden_states = text_model(input_ids)[0]

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

        return input_ids, encoder_hidden_states, labels, mask_prob

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    # As stated above, we are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        for batch in train_dataloader:
            # TODO(Patrick) - We could definitely pre-compute the image tokens for faster training on larger datasets
            pixel_values, input_ids = batch
            pixel_values = pixel_values.to(accelerator.device, non_blocking=True)
            input_ids = input_ids.to(accelerator.device, non_blocking=True)
            data_time_m.update(time.time() - end)

            # encode images to image tokens, mask them and create input and labels
            input_ids, labels, mask_prob = prepare_inputs_and_labels(
                pixel_values, input_ids, config.training.min_masking_rate
            )

            # log the inputs for the first step of the first epoch
            if global_step == 0 and epoch == 0:
                logger.info("Input ids: {}".format(input_ids))
                logger.info("Labels: {}".format(labels))

            # Train Step
            with accelerator.accumulate(model):
                _, loss = model(input_ids=input_ids, labels=labels)
                # Gather thexd losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(config.training.batch_size)).mean()
                avg_masking_rate = accelerator.gather(mask_prob.repeat(config.training.batch_size)).mean()

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
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
                if (global_step + 1) % config.experiment.save_every == 0 and accelerator.is_main_process:
                    save_checkpoint(config, accelerator, global_step + 1)

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
        save_checkpoint(config, accelerator, global_step)

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
        input_ids, labels, _ = prepare_inputs_and_labels(pixel_values, class_ids)
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
    gen_token_ids = accelerator.unwrap_model(model).generate(imagenet_class_ids, timesteps=4)
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


def save_checkpoint(config, accelerator, global_step):
    save_path = Path(config.experiment.output_dir) / f"checkpoint-{global_step}"
    accelerator.save_state(save_path)
    json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
    logger.info(f"Saved state to {save_path}")


if __name__ == "__main__":
    main()
