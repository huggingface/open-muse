import json
from argparse import ArgumentParser
from itertools import islice

import torch
import wandb

from muse import PipelineMuseInpainting


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def generate_and_log(args):
    pipe = PipelineMuseInpainting.from_pretrained(
        text_encoder_path=args.text_encoder,
        vae_path=args.vae,
        transformer_path=args.transformer,
        is_class_conditioned=args.is_class_conditioned,
    ).to(device=args.device)
    pipe.transformer.enable_xformers_memory_efficient_attention()

    imagenet_class_ids = [args.imagenet_class_id]
    class_ids = torch.tensor(imagenet_class_ids).to(device=args.device, dtype=torch.long)

    if args.is_class_conditioned:
        inputs = {"class_ids": class_ids}
    else:
        raise Exception("Not implemented")

    images = pipe(
        **inputs,
        timesteps=args.timesteps,
        guidance_scale=args.guidance_scale,
        temperature=args.temperature,
        use_maskgit_generate=args.use_maskgit_generate,
        num_images_per_prompt=args.num_generations,
    )

    images = list(chunk(images, args.num_generations))
    for class_id, class_images in zip(imagenet_class_ids, images):
        for i, image in enumerate(class_images):
            image.save(f"output_{i}.jpg")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--is_class_conditioned", action="store_true")
    parser.add_argument("--timesteps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--use_maskgit_generate", action="store_true")
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--text_encoder", type=str, default="google/t5-v1_1-large")
    parser.add_argument("--vae", type=str, default="openMUSE/maskgit-vqgan-imagenet-f16-256")
    parser.add_argument("--transformer", type=str, required=True)
    parser.add_argument("--imagenet_class_mapping_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--imagenet_class_id", type=int, default=4)

    args = parser.parse_args()
    generate_and_log(args)
