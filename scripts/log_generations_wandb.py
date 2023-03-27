import json
from argparse import ArgumentParser
from itertools import islice

import wandb

from muse import PipelineMuse


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def generate_and_log(args):
    run_name = f"samples-{args.run_id}-{args.checkpoint}-t={args.timesteps}-g={args.guidance_scale}-temp={args.temperature}-maskgit={args.use_maskgit_generate}"
    wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_name,
        notes=(
            f"Samples from {args.run_id} at checkpoint {args.checkpoint} with timesteps={args.timesteps},"
            f" guidance_scale={args.guidance_scale}, temperature={args.temperature},"
            f" use_maskgit_generate={args.use_maskgit_generate}"
        ),
    )

    pipe = PipelineMuse.from_pretrained(
        text_encoder_path=args.text_encoder,
        vae_path=args.vae,
        transformer_path=args.transformer,
    ).to(device=args.device)
    pipe.transformer.enable_xformers_memory_efficient_attention()

    imagenet_class_ids = list(range(1000))
    with open(args.imagenet_class_mapping_path) as f:
        imagenet_class_mapping = json.load(f)
    imagenet_class_ids = list(chunk(imagenet_class_ids, args.batch_size // args.num_generations))

    # generate images and log in wandb table
    table = wandb.Table(columns=["class name"] + [f"image {i}" for i in range(args.num_generations)])
    for imagenet_class_id in imagenet_class_ids:
        imagenet_class_names = [imagenet_class_mapping[str(i)] for i in imagenet_class_id]
        images = pipe(
            imagenet_class_names,
            timesteps=args.timesteps,
            guidance_scale=args.guidance_scale,
            temperature=args.temperature,
            use_maskgit_generate=args.use_maskgit_generate,
            num_images_per_prompt=args.num_generations,
        )

        # create rows like this: [class name, image 1, image 2, ...]
        # where each image is a wandb.Image
        # and log in wandb table
        images = list(chunk(images, args.num_generations))
        for imagenet_class_name, class_images in zip(imagenet_class_names, images):
            row = [imagenet_class_name]
            for image in class_images:
                row.append(wandb.Image(image))
            table.add_data(*row)

    wandb.log({"samples": table})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--project", type=str, default="muse")
    parser.add_argument("--entity", type=str, default="psuraj")
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--timesteps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--use_maskgit_generate", action="store_true")
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--text_encoder", type=str, default="google/t5-v1_1-large")
    parser.add_argument("--vae", type=str, default="openMUSE/maskgit-vqgan-imagenet-f16-256")
    parser.add_argument("--transformer", type=str, required=True)
    parser.add_argument("--imagenet_class_mapping_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    generate_and_log(args)
