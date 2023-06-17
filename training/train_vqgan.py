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
from ema import EMAModel
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
from muse import MOVQ, MaskGitTransformer, MaskGitVQGAN, VQGANModel
from muse.lr_schedulers import get_scheduler
from muse.sampling import cosine_schedule
from training.discriminator import Discriminator
try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False
import timm
from einops import repeat, rearrange
from tqdm import tqdm


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
    elif model_type == "taming_vqgan":
        return VQGANModel
    else:
        raise ValueError(f"model_type {model_type} not supported for VQGAN")


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

def _map_layer_to_idx(backbone, layers, offset=0):
    """Maps set of layer names to indices of model. Ported from anomalib

    Returns:
        Feature map extracted from the CNN
    """
    idx = []
    features = timm.create_model(
        backbone,
        pretrained=False,
        features_only=False,
        exportable=True,
    )
    for i in layers:
        try:
            idx.append(list(dict(features.named_children()).keys()).index(i)-offset)
        except ValueError:
            raise ValueError(
                f"Layer {i} not found in model {backbone}. Select layer from {list(dict(features.named_children()).keys())}. The network architecture is {features}"
            )
    return idx

# From https://arxiv.org/abs/2111.01007v1 Projected Gan where instead of giving the discriminator/generator the input image, we give hierarchical features
# from a timm model
class MultiLayerTimmModel(torch.nn.Module):
    def __init__(self, model, input_shape=(3, 224, 224)):
        super().__init__()
        self.model = model
        self.input_shape = input_shape
        self.image_sizes = []
        self.max_feats = self.get_layer_widths(input_shape)
        self.max_feature_sizes = (self.max_feats, self.max_feats)
    def get_layer_widths(self, shape=(3, 224, 224)):
        output = []
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feats = self.model(input)
        for output_feat in output_feats:
            output.append(output_feat.shape[-1])
        max_feats = max(output)
        return max_feats
    def forward(self, images):
        features = self.model(images)
        output_features = []
        for feature in features:
            if feature.shape[-1] == self.max_feats:
                output_features.append(feature)
            else:
                output_features.append(F.interpolate(feature, size=self.max_feature_sizes))
        return torch.cat(output_features, dim=1)

def get_perceptual_loss(pixel_values, fmap, timm_discriminator):
    img_timm_discriminator_input = pixel_values
    fmap_timm_discriminator_input = fmap

    if pixel_values.shape[1] == 1:
        # handle grayscale for timm_discriminator
        img_timm_discriminator_input, fmap_timm_discriminator_input = map(
            lambda t: repeat(t, "b 1 ... -> b c ...", c=3),
            (img_timm_discriminator_input, fmap_timm_discriminator_input),
        )

    img_timm_discriminator_feats = timm_discriminator(
        img_timm_discriminator_input
    )
    recon_timm_discriminator_feats = timm_discriminator(
        fmap_timm_discriminator_input
    )
    perceptual_loss = F.mse_loss(
        img_timm_discriminator_feats[0], recon_timm_discriminator_feats[0]
    )
    for i in range(1, len(img_timm_discriminator_feats)):
        perceptual_loss += F.mse_loss(
            img_timm_discriminator_feats[i], recon_timm_discriminator_feats[i]
        )
    perceptual_loss /= len(img_timm_discriminator_feats)
    return perceptual_loss

def grad_layer_wrt_loss(loss, layer):
    return torch.autograd.grad(
        outputs=loss,
        inputs=layer,
        grad_outputs=torch.ones_like(loss),
        retain_graph=True,
    )[0].detach()

def gradient_penalty(images, output, weight=10):
    gradients = torch.autograd.grad(
        outputs=output,
        inputs=images,
        grad_outputs=torch.ones(output.size(), device=images.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = rearrange(gradients, "b ... -> b (...)")
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

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
    model = vq_class.from_pretrained(config.model.vq_model.pretrained)
    if config.training.use_ema:
        ema_model = EMAModel(model.parameters(), model_cls=vq_class, model_config=model.config)

    discriminator = Discriminator(config)
    # TODO: Add timm_discriminator_backend to config.training. Set default to vgg16
    idx = _map_layer_to_idx(config.training.timm_discriminator_backend,\
                            config.training.timm_disc_layers.split("|"), config.training.timm_discr_offset)

    timm_discriminator = timm.create_model(
        config.training.timm_discriminator_backend,
        pretrained=True,
        features_only=True,
        exportable=True,
        out_indices=idx,
    )
    timm_discriminator = timm_discriminator.to(accelerator.device)
    timm_discriminator.requires_grad = False
    timm_discriminator.eval()
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
        list(model.parameters()),
        lr=optimizer_config.learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        weight_decay=optimizer_config.weight_decay,
        eps=optimizer_config.epsilon,
    )
    discr_optimizer = optimizer_cls(
        list(discriminator.parameters()),
        lr=optimizer_config.discr_learning_rate,
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
    discr_lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=discr_optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
    )


    # Prepare everything with accelerator
    logger.info("Preparing model, optimizer and dataloaders")
    # The dataloader are already aware of distributed training, so we don't need to prepare them.
    model, discriminator, optimizer, discr_optimizer, lr_scheduler, discr_lr_scheduler = accelerator.prepare(model, discriminator, optimizer, discr_optimizer, lr_scheduler, discr_lr_scheduler)

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

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    # As stated above, we are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    avg_gen_loss, avg_discr_loss = None, None
    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        for i, batch in tqdm(enumerate(train_dataloader)):
            pixel_values, _ = batch
            pixel_values = pixel_values.to(accelerator.device, non_blocking=True)
            data_time_m.update(time.time() - end)
            generator_step = ((i // config.training.gradient_accumulation_steps) % 2) == 0 and i > config.training.discriminator_warmup
            # TODO:
            # Add entropy to maximize codebook usage
            # Train Step
            # The behavior of accelerator.accumulate is to 
            # 1. Check if gradients are synced(reached gradient-accumulation_steps)
            # 2. If so sync gradients by stopping the not syncing process
            if generator_step:
                if optimizer_type == "fused_adamw":
                    optimizer.zero_grad()
                else:
                    optimizer.zero_grad(set_to_none=True)
            else:
                if optimizer_type == "fused_adamw":
                    discr_optimizer.zero_grad()
                else:
                    discr_optimizer.zero_grad(set_to_none=True)
            # encode images to the latent space and get the commit loss from vq tokenization
            # Return commit loss
            fmap, _, _, commit_loss = model(pixel_values, return_loss=True)

            if generator_step:
                with accelerator.accumulate(model):
                    # reconstruction loss. Pixel level differences between input vs output
                    if config.training.vae_loss == "l2":
                        loss = F.mse_loss(pixel_values, fmap)
                    else:
                        loss = F.l1_loss(pixel_values, fmap)
                    # perceptual loss. The high level feature mean squared error loss
                    perceptual_loss = get_perceptual_loss(pixel_values, fmap, timm_discriminator)
                    # generator loss
                    gen_loss = -discriminator(fmap).mean()
                    last_dec_layer = accelerator.unwrap_model(model).decoder.conv_out.weight
                    norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(perceptual_loss, last_dec_layer).norm(p=2)
                    norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p=2)

                    adaptive_weight = norm_grad_wrt_perceptual_loss/norm_grad_wrt_gen_loss.clamp(min=1e-8)
                    adaptive_weight = adaptive_weight.clamp(max=1e4)
                    loss += commit_loss
                    loss += perceptual_loss
                    loss += adaptive_weight*gen_loss
                    # Gather thexd losses across all processes for logging (if we use distributed training).
                    avg_gen_loss = accelerator.gather(loss.repeat(config.training.batch_size)).float().mean()
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
            else:
                # Return discriminator loss
                with accelerator.accumulate(discriminator):
                    fmap.detach_()
                    pixel_values.requires_grad_()
                    real = discriminator(pixel_values)
                    fake = discriminator(fmap)
                    loss = (F.relu(1 + fake) + F.relu(1 - real)).mean()
                    gp = gradient_penalty(pixel_values, real)
                    loss += gp
                    avg_discr_loss = accelerator.gather(loss.repeat(config.training.batch_size)).mean()
                    accelerator.backward(loss)

                    if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(discriminator.parameters(), config.training.max_grad_norm)

                    discr_optimizer.step()
                    discr_lr_scheduler.step()
                    if (
                        accelerator.sync_gradients
                        and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                        and accelerator.is_main_process
                    ):
                        log_grad_norm(discriminator, accelerator, global_step + 1)
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients and not generator_step:
                if config.training.use_ema:
                    ema_model.step(model.parameters())
                # wait for both generator and discriminator to settle
                batch_time_m.update(time.time() - end)
                end = time.time()
                # Log metrics
                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                        config.training.gradient_accumulation_steps * config.training.batch_size / batch_time_m.val
                    )
                    logs = {
                        "step_discr_loss": avg_discr_loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
                    if avg_gen_loss is not None:
                        logs["step_gen_loss"] = avg_gen_loss.item()
                    accelerator.log(logs, step=global_step + 1)
                    logger.info(
                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f} "
                        f"Step: {global_step + 1} "
                        f"Discriminator Loss: {avg_discr_loss.item():0.4f} "
                    )
                    if avg_gen_loss is not None:
                        logger.info(f"Generator Loss: {avg_gen_loss.item():0.4f} ")

                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()
                # Save model checkpoint
                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, discriminator, config, accelerator, global_step + 1)

                # Generate images
                if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                    generate_images(model, pixel_values[:config.training.num_validation_log], accelerator, global_step + 1)

                global_step += 1
                # TODO: Add generation

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
def generate_images(model, original_images, accelerator, global_step):
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
        _, enc_token_ids = accelerator.unwrap_model(model).encode(original_images)
    # In the beginning of training, the model is not fully trained and the generated token ids can be out of range
    # so we clamp them to the correct range.
    enc_token_ids = torch.clamp(enc_token_ids, max=accelerator.unwrap_model(model).config.num_embeddings - 1)
    images = accelerator.unwrap_model(model).decode_code(enc_token_ids)
    model.train()

    # Convert to PIL images
    images = 2.0 * images - 1.0
    original_images = 2.0 * original_images - 1.0
    images = torch.clamp(images, -1.0, 1.0)
    original_images = torch.clamp(original_images, -1.0, 1.0)
    images = (images + 1.0) / 2.0
    original_images = (original_images + 1.0) / 2.0
    images *= 255.0
    original_images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    original_images = original_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    images = np.concatenate([original_images, images], axis=2)
    pil_images = [Image.fromarray(image) for image in images]

    # Log images
    wandb_images = [wandb.Image(image, caption="Original, Generated") for image in pil_images]
    wandb.log({"vae_images": wandb_images}, step=global_step)


def save_checkpoint(model, discriminator, config, accelerator, global_step):
    save_path = Path(config.experiment.output_dir) / f"checkpoint-{global_step}"

    # retrieve the model on all processes for deepspeed stage 3 to work then save on one process (we are not using stage 3 yet)
    # XXX: could also make this conditional on deepspeed
    state_dict = accelerator.get_state_dict(model)
    discr_state_dict = accelerator.get_state_dict(discriminator)

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
        )
        torch.save(discr_state_dict, save_path / "unwrapped_discriminator")
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
