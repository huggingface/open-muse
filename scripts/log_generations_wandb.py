import json
from argparse import ArgumentParser
from itertools import islice

import torch
import wandb

from muse import PipelineMuse


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def generate_and_log(args):
    run_name = f"{args.transformer} samples at checkpoint {args.checkpoint}"
    wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_name,
        notes=(
            f"Samples from {args.run_id} at checkpoint {args.checkpoint} with timesteps={args.timesteps},"
            f" guidance_scale={args.guidance_scale}, temperature={args.temperature}"
        ),
    )

    pipe = PipelineMuse.from_pretrained(
        text_encoder_path=args.text_encoder,
        vae_path=args.vae,
        transformer_path=args.transformer,
    ).to(device=args.device)
    pipe.transformer.enable_xformers_memory_efficient_attention()

    # open args.prompts_file_path and read prompts in a list
    with open(args.prompts_file_path, "r") as f:
        prompts = f.readlines()

    # divide the prompts into batches of size args.batch_size
    prompts = list(chunk(prompts, args.batch_size))

    # generate images and log in wandb table
    table = wandb.Table(columns=["prompt"] + [f"image {i}" for i in range(args.num_generations)])
    for batch in prompts:
        images = pipe(
            batch,
            timesteps=args.timesteps,
            guidance_scale=args.guidance_scale,
            temperature=args.temperature,
            num_images_per_prompt=args.num_generations,
            use_maskgit_generate=True,
            use_fp16=True,
        )

        # create rows like this: [prompt, image 1, image 2, ...]
        # where each image is a wandb.Image
        # and log in wandb table
        images = list(chunk(images, args.num_generations))
        for prompt, gen_images in zip(batch, images):
            row = [prompt]
            for image in gen_images:
                row.append(wandb.Image(image))
            table.add_data(*row)

    wandb.log({"samples": table})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--project", type=str, default="muse")
    parser.add_argument("--entity", type=str, default="psuraj")
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--timesteps", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=8)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text_encoder", type=str, default="google/t5-v1_1-large")
    parser.add_argument("--vae", type=str, default="openMUSE/maskgit-vqgan-imagenet-f16-256")
    parser.add_argument("--transformer", type=str, required=True)
    parser.add_argument("--prompts_file_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    generate_and_log(args)
