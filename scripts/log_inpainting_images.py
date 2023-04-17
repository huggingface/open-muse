import json
from argparse import ArgumentParser
from itertools import islice

import torch
import wandb
import numpy as np
from muse import PipelineMuseInpainting
from PIL import Image
import copy
def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def generate_and_log(args):
    pipe = PipelineMuseInpainting.from_pretrained(
        model_name_or_path=args.model_name_or_path,
        is_class_conditioned=args.is_class_conditioned,
    ).to(device=args.device)
    pipe.transformer.enable_xformers_memory_efficient_attention()

    imagenet_class_ids = [args.imagenet_class_id]
    class_ids = torch.tensor(imagenet_class_ids).to(device=args.device, dtype=torch.long)

    if args.is_class_conditioned:
        inputs = {"class_ids": class_ids}
    else:
        raise NotImplementedError("Not implemented")

    mask = np.ones((args.image_size // 16, args.image_size // 16))
    mask[args.mask_start_x:args.mask_end_x, args.mask_start_y:args.mask_end_y] = 0
    mask = mask.reshape(-1)
    mask = torch.tensor(mask).to(args.device, dtype=torch.bool)

    image = Image.open(args.input_image).resize((args.image_size, args.image_size))

    masked_image = copy.deepcopy(np.array(image))
    masked_image[args.mask_start_x*16:args.mask_end_x*16, args.mask_start_y*16:args.mask_end_y*16] = 0
    masked_image = Image.fromarray(masked_image)
    masked_image.save("segmented.jpg")

    images = pipe(
        **inputs,
        image=image,
        mask=mask,
        timesteps=args.timesteps,
        guidance_scale=args.guidance_scale,
        temperature=args.temperature,
        use_maskgit_generate=args.use_maskgit_generate,
        num_images_per_prompt=args.num_generations,
    )

    images = list(chunk(images, args.num_generations))
    for class_id, class_images in zip(imagenet_class_ids, images):
        for i, image in enumerate(class_images):
            image.save(f"output_{class_id}_{i}.jpg")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--is_class_conditioned", default=True, action="store_false")
    parser.add_argument("--timesteps", type=int, default=18)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--use_maskgit_generate", action="store_true")
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--model_name_or_path", type=str, default="openMUSE/maskgit-large-imagenet")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--imagenet_class_id", type=int, default=248)
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--mask_start_x", type=int, default=4)
    parser.add_argument("--mask_start_y", type=int, default=4)
    parser.add_argument("--mask_end_x", type=int, default=12)
    parser.add_argument("--mask_end_y", type=int, default=12)

    args = parser.parse_args()
    generate_and_log(args)
