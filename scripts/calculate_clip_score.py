import os
from argparse import ArgumentParser

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, AutoProcessor

from muse import PipelineMuse
import warnings
import PIL
import numpy as np

class Flickr8kDataset(Dataset):
    def __init__(self, root_dir, captions_file):
        self.root_dir = root_dir
        self.captions_file = captions_file

        df = pd.read_csv(captions_file, sep="\t", names=["image_name", "caption"])
        df["image_name"] = df["image_name"].apply(lambda name: name.split("#")[0])

        self.images = df["image_name"].unique().tolist()
        self.captions = [df[df["image_name"] == name]["caption"].tolist()[0] for name in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.captions[idx]

class DatasetWithGeneratedImages(Dataset):
    def __init__(self, real_images, generated_images, captions, processor):
        self.real_images = real_images
        self.generated_images = generated_images
        self.captions = captions
        self.processor = processor
    def __len__(self):
        return len(self.real_images)
    def __getitem__(self, index):
        real_image = PIL.Image.open(self.real_images[index])
        generated_image = PIL.Image.open(self.generated_images[index])
        caption = self.captions[index]
        text_inputs = self.processor(text=[caption], return_tensors="pt", padding=True)
        real_image_inputs = self.processor(images=[real_image], return_tensors="pt", padding=True)
        generated_image_inputs = self.processor(images=[generated_image], return_tensors="pt", padding=True)
        return text_inputs, real_image_inputs, generated_image_inputs


def generate_and_save_images(args):
    """
    Generate images from captions and save them to disk.
    """
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    os.makedirs(args.save_path, exist_ok=True)

    print("Loading pipe")
    pipeline = PipelineMuse.from_pretrained(args.model_name_or_path).to(args.device, dtype=weight_dtype)
    if args.enable_memory_efficient_attention:
        pipeline.transformer.enable_xformers_memory_efficient_attention()

    print("Loading data")
    dataset = Flickr8kDataset(args.dataset_root, args.dataset_captions_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    generator = torch.Generator(args.device, dtype=weight_dtype).manual_seed(args.seed)
    generated_image_paths = []
    print("Generating images")
    for batch in tqdm(dataloader):
        image_names = batch[0]
        text = batch[1]

        images = pipeline(
            text,
            timesteps=args.timesteps,
            guidance_scale=args.guidance_scale,
            temperature=args.temperature,
            generator=generator,
        )

        for image_name, image, image_caption in zip(image_names, images, text):
            generated_image_path = os.path.join(args.save_path, f"{image_name}")
            image.save(generated_image_path)
            generated_image_paths.append(generated_image_path)
    return dataset.captions, dataset.images, generated_image_paths
def get_clip_scores(args, captions, real_image_names, generated_image_names):
    # This code is based on https://arxiv.org/abs/2104.08718 and it's implementation
    clip_model_name = "openai/clip-vit-base-patch16"
    # In the clip score paper they scaling the textual alignment by 2.5
    w = 2.5
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(args.device)
    if args.device == 'cpu':
        warnings.warn(
            'CLIP runs in full float32 on CPU. Results in paper were computed on GPU, which uses float16. '
            'If you\'re reporting results on CPU, please note this when you report.')
    else:
        clip_model = clip_model.to(dtype=torch.float16)
    clip_model.eval()
    processor = AutoProcessor.from_pretrained(clip_model_name)
    dataset = DatasetWithGeneratedImages(real_images=real_image_names, generated_images=generated_image_names, captions=captions, processor=processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    clip_text_scores = []
    clip_image_scores = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if args.device == 'cpu':
                text_inputs, real_inputs, generated_inputs = batch.to(args.device)
            else:
                text_inputs, real_inputs, generated_inputs = batch.to(args.device, dtype=torch.float16)

            text_embeds = clip_model.get_text_features(**text_inputs)
            real_embeds = clip_model.get_image_features(**real_inputs)
            generated_embeds = clip_model.get_image_features(**generated_inputs)

            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            real_embeds = real_embeds / real_embeds.norm(p=2, dim=-1, keepdim=True)
            generated_embeds = generated_embeds / generated_embeds.norm(p=2, dim=-1, keepdim=True)
            clip_text_alignment = torch.clip(torch.matmul(text_embeds, generated_embeds.t()) * w, 0)
            clip_image_alignment = torch.clip(torch.matmul(real_embeds, generated_embeds.t()), 0)
            clip_text_alignment = clip_text_alignment.cpu().detach().numpy()
            clip_image_alignment = clip_image_alignment.cpu().detach().numpy()
            clip_text_scores.append(clip_text_alignment)
            clip_image_scores.append(clip_image_alignment)
    clip_text_scores = np.mean(np.concatenate(clip_text_scores, axis=0))
    clip_image_scores = np.mean(np.concatenate(clip_image_scores, axis=0))
    return clip_text_scores, clip_image_scores

def main(args):
    captions, real_image_names, generated_image_names = generate_and_save_images(args)
    print("computing Image and Text CLIP Score")
    clip_text_score, clip_image_score = get_clip_scores(args, captions, real_image_names, generated_image_names)
    print(f"clip text score is {clip_text_score:.3f} clip image score is {clip_image_score:.3f}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--dataset_captions_file", type=str, default=None)
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
    parser.add_argument("--enable_memory_efficient_attention", action='store_true')
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--timesteps", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=8.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2028)

    args = parser.parse_args()
    main(args)
