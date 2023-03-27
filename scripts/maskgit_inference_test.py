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


# !pip install ml_collections
# !wget https://storage.googleapis.com/maskgit-public/checkpoints/tokenizer_imagenet256_checkpoint
import torch
from muse import MaskGitTransformer, MaskGitVQGAN
from muse.sampling import cosine_schedule
import argparse
from accelerate import Accelerator
import numpy as np
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Simple loading script for maskgit.")
    parser.add_argument(
        "--pytorch_dump_folder_path",
        type=str,
        default=None,
        required=True,
        help="Path to maskgit model.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--imagenet_class_idx",
        type=int,
        default=4,
        help="The index of the imagenet class id to do inference on. Chosen from imagenet_class_names below."
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=4,
        help="The number of timesteps used to generate."
    )
    return parser.parse_args()

if __name__ == "__main__":
    with torch.no_grad():
        args = parse_args()
        accelerator = Accelerator(
            mixed_precision=args.mixed_precision
        )

        precision = torch.float32
        if args.mixed_precision == "fp16":
            precision = torch.float16
        elif args.mixed_precision == "bf16":
            precision =  torch.bfloat16

        vq_model = MaskGitVQGAN.from_pretrained("openMUSE/maskgit-vqgan-imagenet-f16-256").to(accelerator.device, dtype=precision)
        model = MaskGitTransformer.from_pretrained(args.pytorch_dump_folder_path)
        model = model.to(accelerator.device, dtype=precision)
        model.eval()

        # Initialize the MaskGitTransformer model
        mask_id = model.config.mask_token_id
        imagenet_class_idx=args.imagenet_class_idx
        imagenet_class_names = ["Jay", "Castle", "coffee mug", "desk", "Husky", "Valley", "Red wine", "Coral reef", "Mixing bowl", "Cleaver", "Vine Snake", "Bloodhound", "Barbershop", "Ski", "Otter", "Snowmobile"]
        imagenet_class_ids_list = [17, 483, 504, 526, 248, 979, 966, 973, 659, 499, 59, 163, 424, 795, 360, 802]
        # fmt: on
        imagenet_class_ids = torch.tensor(
            [imagenet_class_ids_list[imagenet_class_idx]],
            device=accelerator.device,
            dtype=torch.long,
        )
        print(f"Generating image of {imagenet_class_names[imagenet_class_idx]}")
        gen_token_ids = model.generate(imagenet_class_ids, timesteps=args.num_timesteps)
        # In the beginning of training, the model is not fully trained and the generated token ids can be out of range
        # so we clamp them to the correct range.
        gen_token_ids = torch.clamp(gen_token_ids, max=model.config.codebook_size - 1)
        images = vq_model.decode_code(gen_token_ids)

        # Convert to PIL images
        images = 2.0 * images - 1.0
        images = torch.clamp(images, -1.0, 1.0)
        images = (images + 1.0) / 2.0
        images *= 255.0
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        pil_images = [Image.fromarray(image) for image in images]
        for i, pil_image in enumerate(pil_images):
            pil_image.save(f"output-{i}.png")
