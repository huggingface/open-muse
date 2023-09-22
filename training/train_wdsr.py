"""Training script for VQGAN."""
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
from data import Text2ImageDataset
from omegaconf import DictConfig, ListConfig, OmegaConf
from optimizer import Lion
from PIL import Image
from torch.optim import AdamW  # why is shampoo not available in PT :(

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

from vqgan_train_utils.discriminator import NLayerDiscriminator, weights_init
from vqgan_train_utils.vq_perceptual_loss import VQLPIPSWithDiscriminator

import muse
from muse import EMAModel, VQGANModel, WDSR
from muse.lr_schedulers import get_scheduler

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

    vqgan = VQGANModel.from_pretrained("openMUSE/vqgan-f16-8192-laion")
    # freeze the vqgan
    vqgan.requires_grad_(False)

    model = WDSR(
        num_channels=3,
        num_blocks=12,
        num_residual_units=128,
        width_multiplier=4,
        image_mean=0.5,
        scale=1,
        temporal_size=None,
    )

    # Create the EMA model
    if config.training.use_ema:
        ema_model = EMAModel(model.parameters(), model_cls=WDSR, model_config=model.config)

        # Create custom saving and loading hooks so that `accelerator.save_state(...)` serializes in a nice format.
        def load_model_hook(models, input_dir):
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "ema_model"), model_cls=WDSR)
            ema_model.load_state_dict(load_model.state_dict())
            ema_model.to(accelerator.device)
            del load_model

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                ema_model.save_pretrained(os.path.join(output_dir, "ema_model"))

        accelerator.register_load_state_pre_hook(load_model_hook)
        accelerator.register_save_state_pre_hook(save_model_hook)

    discriminator = NLayerDiscriminator(
        input_nc=3,
        n_layers=3,
        ndf=64,
    ).apply(weights_init)

    loss_module = VQLPIPSWithDiscriminator(
        disc_start=config.training.disc_start,
        disc_factor=config.training.disc_factor,
        disc_weight=config.training.disc_weight,
        perceptual_weight=config.training.perceptual_weight,
        codebook_weight=0.0,
        disc_loss="hinge",
    )

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
        list(model.parameters()),
        lr=optimizer_config.learning_rate,
        betas=(0.5, 0.9),
        weight_decay=optimizer_config.weight_decay,
        eps=optimizer_config.epsilon,
    )
    discr_optimizer = optimizer_cls(
        list(discriminator.parameters()),
        lr=optimizer_config.discr_learning_rate,
        betas=(0.5, 0.9),
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
    dataset = Text2ImageDataset(
        train_shards_path_or_url=dataset_config.train_shards_path_or_url,
        eval_shards_path_or_url=dataset_config.eval_shards_path_or_url,
        tokenizer=None,
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
        only_return_images=True,
    )
    train_dataloader, eval_dataloader = dataset.train_dataloader, dataset.eval_dataloader

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
    )
    discr_lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=discr_optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
    )

    # Prepare everything with accelerator
    logger.info("Preparing model, optimizer and dataloaders")
    # The dataloader are already aware of distributed training, so we don't need to prepare them.
    model, discriminator, optimizer, discr_optimizer, lr_scheduler, discr_lr_scheduler = accelerator.prepare(
        model, discriminator, optimizer, discr_optimizer, lr_scheduler, discr_lr_scheduler
    )
    loss_module.to(accelerator.device)
    vqgan.to(accelerator.device)
    if config.training.use_ema:
        ema_model.to(accelerator.device)

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

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    # As stated above, we are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        for batch in train_dataloader:
            pixel_values = batch["image"].to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
            data_time_m.update(time.time() - end)

            # encode images to the latent space and get the commit loss from vq tokenization
            # Return commit loss
            # encode with batch size 16, vq_images, _, _ = vqgan(pixel_values, return_loss=False)
            vq_images = []
            for i in range(0, pixel_values.shape[0], 32):
                vq_image, _, _ = vqgan(pixel_values[i:i+32], return_loss=False)
                vq_images.append(vq_image)
            vq_images = torch.cat(vq_images, dim=0)
            
            
            vq_images = 2.0 * vq_images - 1.0
            vq_images = torch.clamp(vq_images, -1.0, 1.0)
            vq_images = (vq_images + 1.0) / 2.0
            vq_images = (255.0 * vq_images) / 255.0
            
            with accelerator.accumulate(model):
                reconstructions = model(vq_images)

                # ########################
                # autoencode
                # ########################
                aeloss, log_dict_ae = loss_module(
                    discriminator,
                    None,
                    pixel_values,
                    reconstructions,
                    0,
                    global_step,
                    last_layer=accelerator.unwrap_model(model).get_last_layer(),
                    split="train",
                )

                # Gather the losses across all processes for logging (if we use distributed training).
                ae_logs = {}
                for k, v in log_dict_ae.items():
                    if k in ["train/disc_factor", "train/g_loss", "train/d_weight", "train/p_loss"]:
                        ae_logs[k] = v
                    else:
                        ae_logs[k] = accelerator.gather(v.repeat(config.training.batch_size)).mean().item()

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                accelerator.backward(aeloss)

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

                # ########################
                # discriminator loss
                # ########################
                discr_loss, log_dict_discr = loss_module(
                    discriminator,
                    None,
                    pixel_values,
                    reconstructions,
                    1,
                    global_step,
                    last_layer=accelerator.unwrap_model(model).get_last_layer(),
                    split="train",
                )

                # Gather the losses across all processes for logging (if we use distributed training).
                discr_logs = {}
                for k, v in log_dict_discr.items():
                    if k in ["train/logits_real", "train/logits_fake"]:
                        discr_logs[k] = v
                    else:
                        discr_logs[k] = accelerator.gather(v.repeat(config.training.batch_size)).mean().item()

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(discriminator.parameters(), config.training.max_grad_norm)

                accelerator.backward(discr_loss)

                discr_optimizer.step()
                discr_lr_scheduler.step()

                # log gradient norm before zeroing it
                if (
                    accelerator.sync_gradients
                    and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                    and accelerator.is_main_process
                ):
                    log_grad_norm(discriminator, accelerator, global_step + 1)

                if optimizer_type == "fused_adamw":
                    discr_optimizer.zero_grad()
                else:
                    discr_optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if config.training.use_ema:
                    ema_model.step(model.parameters())

                # wait for both generator and discriminator to settle
                batch_time_m.update(time.time() - end)
                end = time.time()

                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                        config.training.gradient_accumulation_steps * config.training.batch_size / batch_time_m.val
                    )
                    logger.info(
                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f} "
                        f"Step: {global_step + 1} "
                        f"AE Loss: {ae_logs['train/total_loss']:0.4f} "
                        f"Discr Loss: {discr_logs['train/disc_loss']:0.4f} "
                    )
                    logs = {
                        "lr": lr_scheduler.get_last_lr()[0],
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
                    logs.update(ae_logs)
                    logs.update(discr_logs)
                    accelerator.log(logs, step=global_step + 1)

                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()

                # Save model checkpoint
                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, discriminator, config, accelerator, global_step + 1)

                # Generate images
                if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                    # Store the model parameters temporarily and load the EMA parameters to perform inference.
                    if config.training.get("use_ema", False):
                        ema_model.store(model.parameters())
                        ema_model.copy_to(model.parameters())
                    
                    generate_images(
                        model, vqgan, pixel_values[: config.training.num_validation_log], accelerator, global_step + 1
                    )

                    if config.training.get("use_ema", False):
                        # Switch back to the original model parameters for training.
                        ema_model.restore(model.parameters())

                global_step += 1

            # Stop training if max steps is reached
            if global_step >= config.training.max_train_steps:
                break
        # End for

    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(model, discriminator, config, accelerator, global_step)

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if config.training.use_ema:
            ema_model.copy_to(model.parameters())
        model.save_pretrained(config.experiment.output_dir)

    accelerator.end_training()


@torch.no_grad()
def generate_images(model, vqgan, original_images, accelerator, global_step):
    logger.info("Generating images...")
    original_images = torch.clone(original_images)
    # Generate images
    model.eval()
    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    with torch.autocast("cuda", dtype=dtype, enabled=accelerator.mixed_precision != "no"):
        vq_image, _, _ = vqgan(original_images, return_loss=False)
        vq_image = 2.0 * vq_image - 1.0
        vq_image = torch.clamp(vq_image, -1.0, 1.0)
        vq_image = (vq_image + 1.0) / 2.0
        vq_image = (255.0 * vq_image) / 255.0
    
    images = model(vq_image)

    model.train()

    # Convert to PIL images
    vq_image *= 255.0
    vq_image = vq_image.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    images *= 255.0
    images = torch.clamp(images, 0.0, 255.0)
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    
    images = np.concatenate([vq_image, images], axis=2)
    pil_images = [Image.fromarray(image) for image in images]

    # Log images
    wandb_images = [wandb.Image(image, caption="Original, Generated") for image in pil_images]
    wandb.log({"vae_images": wandb_images}, step=global_step)


def save_checkpoint(model, discriminator, config, accelerator, global_step):
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
