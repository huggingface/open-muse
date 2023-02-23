import logging
import os
import time
from pathlib import Path
from typing import Any, List, Tuple

import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import logging
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.optim import AdamW  # why is shampoo not available in PT :(
from torchvision import transforms
from tqdm import tqdm

import muse
from muse import MaskGitTransformer, MaskGitVQGAN
from muse.lr_schedulers import get_scheduler
from muse.sampling import cosine_schedule
from muse.training_utils import EMA

logger = get_logger(__name__, log_level=logging.INFO)


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


def main():
    config = get_config()

    config.experiment.logging_dir = Path(config.experiment.output_dir) / "logs"
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        logging_dir=config.experiment.logging_dir,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        muse.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        muse.logging.set_verbosity_error()

    if accelerator.is_main_process():
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
    vq_model = MaskGitVQGAN.from_pretrained(config.model.vq_model.pretrained)
    model = MaskGitTransformer(**config.model.transformer)
    mask_id = model.mask_id

    # Freeze the VQGAN
    vq_model.requires_grad_(False)

    optimizer_config = config.optimizer.params
    optimizer = AdamW(
        model.parameters(),
        lr=optimizer_config.learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        weight_decay=optimizer_config.weight_decay,
        eps=optimizer_config.epsilon,
    )

    ##################################
    # DATSET LOADING & PREPROCESSING #
    #################################
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    datasets_config = config.dataset.params
    dataset = datasets.load_dataset(datasets_config.path, streaming=datasets_config.streaming)

    # Preprocessing the datasets.
    preproc_config = config.preprocessing
    train_transforms = transforms.Compose(
        [
            transforms.Resize(preproc_config.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            (
                transforms.CenterCrop(preproc_config.resolution)
                if preproc_config.center_crop
                else transforms.RandomCrop(preproc_config.resolution)
            ),
            transforms.RandomHorizontalFlip() if preproc_config.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ]
    )

    def preprocess(examples):
        examples["image"] = [train_transforms(image.convert("RGB")) for image in examples["image"]]
        return examples

    with accelerator.main_process_first():
        # Set the training transforms
        train_dataset = dataset["train"].map(preprocess, batched=True)
        eval_dataset = dataset["validation"].map(preprocess, batched=True)

        # We need to shuffle early when using streaming datasets
        train_dataset = train_dataset.shuffle(
            buffer_size=datasets_config.shuffle_buffer_size, seed=config.training.seed
        )

    def collate_fn(examples):
        pixel_values = torch.stack([example["image"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        class_id = torch.stack([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "class_id": class_id}

    ##################################
    # DATLOADER and LR-SCHEDULER     #
    #################################
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=config.training.batch_size,
        num_workers=datasets_config.workers,
    )

    total_batch_size = (
        config.training.batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps
    )
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps * config.training.gradient_accumulation_steps,
    )

    # Prepare everything with accelerator
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(
            config.experiment.project, config={k: v for k, v in flatten_omega_conf(config, resolve=True)}
        )

    # TODO: Add checkpoint loading. We need to check how to resume datasets in streaming mode.

    # Train!
    global_step = 0
    first_epoch = 0
    examples_since_last_logged = 0

    now = time.time()
    for epoch in range(first_epoch, config.training.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                image_tokens = vq_model.encode(batch["pixel_values"])[1]
                batch_size, seq_len = image_tokens.shape

                # Sample a random timestep for each image
                timesteps = torch.rand(batch_size, device=image_tokens.device)
                # Sample a random mask probability for each image using timestep and cosine schedule
                mask_prob = cosine_schedule(timesteps)
                # creat a random mask for each image
                num_token_masked = (seq_len * mask_prob).round().clamp(min=1)
                batch_randperm = torch.rand((batch, seq_len), device=image_tokens.device).argsort(dim=-1)
                mask = batch_randperm < num_token_masked.unsqueeze(-1)
                # mask images and create input and labels
                inout_ids = torch.where(mask, mask_id, image_tokens)
                labels = torch.where(mask, image_tokens, -100)

                # shift the class ids by codebook size
                class_ids = batch["class_id"] + vq_model.num_embeddings
                # prepend the class ids to the image tokens
                inout_ids = torch.cat([class_ids.unsqueeze(-1), inout_ids], dim=-1)
                # prepend -100 to the labels as we don't want to predict the class ids
                labels = torch.cat([-100 * torch.ones_like(class_ids).unsqueeze(-1), labels], dim=-1)

                _, loss = model(input_ids=inout_ids, labels=labels)

                # Gather thexd losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(config.training.batch_size)).mean()

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                global_step += 1
                examples_since_last_logged += batch["input_ids"].shape[0] * config.training.gradient_accumulation_steps

                if global_step % config.experiment.log_every:
                    images_per_second_per_gpu = examples_since_last_logged / (time.time() - now)
                    # Log metrics
                    logs = {
                        "step_loss": avg_loss.item(),
                        "lr": lr_scheduler.current_lr(),
                        "images/sec/gpu": images_per_second_per_gpu,
                    }
                    accelerator.log(logs, step=global_step)

                    logger.info(
                        f" Step: {step} Loss: {avg_loss.item():0.4f} im/s/GPU: {images_per_second_per_gpu:0.2f}"
                    )

                if global_step % config.experiment.save_every == 0:
                    if accelerator.is_main_process:
                        save_path = Path(config.training.output_dir) / f"checkpoint-{global_step}"
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    # TODO: Add generation

    # Save the final trained checkpoint
    accelerator.wait_for_everyone()
    if accelerator.is_main_process():
        model = accelerator.unwrap_model()
        model.save_pretrained(config.training.output_dir)

    accelerator.end_training()
