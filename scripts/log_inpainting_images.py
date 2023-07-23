import copy
import json
import os
from argparse import ArgumentParser
from itertools import islice

import numpy as np
import torch
import wandb
from PIL import Image

from muse import PipelineMuseInpainting


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def generate_and_log(args):
    os.makedirs(args.output_dir, exist_ok=True)
    vae_scaling_factor = args.vae_scaling_factor
    pipe = PipelineMuseInpainting.from_pretrained(
        model_name_or_path=args.model_name_or_path,
        is_class_conditioned=args.is_class_conditioned,
    ).to(device=args.device)
    pipe.transformer.enable_xformers_memory_efficient_attention()

    if args.is_class_conditioned:
        imagenet_class_ids = [args.imagenet_class_id]
        class_ids = torch.tensor(imagenet_class_ids).to(device=args.device, dtype=torch.long)
        inputs = {"class_ids": class_ids}
    else:
        inputs = {"text": args.text}

    mask = np.zeros((args.image_size // vae_scaling_factor, args.image_size // vae_scaling_factor))
    mask[args.mask_start_x : args.mask_end_x, args.mask_start_y : args.mask_end_y] = 1
    mask = mask.reshape(-1)
    mask = torch.tensor(mask).to(args.device, dtype=torch.bool)

    image = Image.open(args.input_image).resize((args.image_size, args.image_size))

    masked_image = copy.deepcopy(np.array(image))
    masked_image[
        args.mask_start_x * vae_scaling_factor : args.mask_end_x * vae_scaling_factor,
        args.mask_start_y * vae_scaling_factor : args.mask_end_y * vae_scaling_factor,
    ] = 0
    masked_image = Image.fromarray(masked_image)
    masked_image.save(os.path.join(args.output_dir, "segmented.jpg"))
    images = pipe(
        image=image,
        mask=mask,
        **inputs,
        timesteps=args.timesteps,
        guidance_scale=args.guidance_scale,
        temperature=args.temperature,
        use_maskgit_generate=not args.not_maskgit_generate,
        num_images_per_prompt=args.num_generations,
        image_size=args.image_size,
    )

    if args.is_class_conditioned:
        images = list(chunk(images, args.num_generations))
        for class_id, class_images in zip(imagenet_class_ids, images):
            for i, image in enumerate(class_images):
                image.save(os.path.join(args.output_dir, f"output_{class_id}_{i}.jpg"))
    else:
        for i, image in enumerate(images):
            image.save(os.path.join(args.output_dir, f"output_{i}.jpg"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--is_class_conditioned", action="store_true")
    parser.add_argument("--timesteps", type=int, default=18)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--not_maskgit_generate", action="store_true")
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--model_name_or_path", type=str, default="openMUSE/muse-laiona6-uvit-clip-220k")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--imagenet_class_id", type=int, default=248)
    parser.add_argument("--text", type=str, default="a picture of a dog")
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--mask_start_x", type=int, default=4)
    parser.add_argument("--mask_start_y", type=int, default=4)
    parser.add_argument("--mask_end_x", type=int, default=12)
    parser.add_argument("--mask_end_y", type=int, default=12)
    parser.add_argument("--vae_scaling_factor", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="generated")
    args = parser.parse_args()
    generate_and_log(args)
