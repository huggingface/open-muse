import os
from argparse import ArgumentParser

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from cleanfid import fid
except:
    raise ImportError("Please install cleanfid: pip install clean_fid")


from muse import PipelineMuse


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


def generate_and_save_images(args):
    """
    Generate images from captions and save them to disk.
    """
    os.makedirs(args.save_path, exist_ok=True)

    print("Loading pipe")
    pipeline = PipelineMuse.from_pretrained(args.model_name_or_path).to(args.device)
    pipeline.transformer.enable_xformers_memory_efficient_attention()

    print("Loading data")
    dataset = Flickr8kDataset(args.dataset_root, args.dataset_captions_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    generator = torch.Generator(args.device).manual_seed(args.seed)

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

        for image_name, image in zip(image_names, images):
            image.save(os.path.join(args.save_path, f"{image_name}"))


def main(args):
    generate_and_save_images(args)
    real_images = args.dataset_root
    generated_images = args.save_path
    print("computing FiD")
    score_clean = fid.compute_fid(real_images, generated_images, mode="clean", num_workers=0)
    print(f"clean-fid score is {score_clean:.3f}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--dataset_captions_file", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--timesteps", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=8.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2028)

    args = parser.parse_args()
    main(args)
