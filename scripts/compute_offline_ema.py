import argparse
import os
from pathlib import Path

import torch

from muse import EMAModel, MaskGitTransformer, MaskGiTUViT


def offline_ema(args):
    checkpoint_dir_path = args.checkpoint_dir_path
    ema_save_path = args.ema_save_path
    ema_decay = args.ema_decay
    checkpoint_interval = args.checkpoint_interval

    dirs = os.listdir(checkpoint_dir_path)
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    dirs = [Path(checkpoint_dir_path) / dir_ for dir_ in dirs]

    transformer_config = MaskGitTransformer.load_config(dirs[0] / "unwrapped_model")
    if transformer_config["_class_name"] == "MaskGitTransformer":
        model_cls = MaskGitTransformer
    elif transformer_config["_class_name"] == "MaskGiTUViT":
        model_cls = MaskGiTUViT

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    model = model_cls.from_pretrained(dirs[0] / "unwrapped_model").to(device)
    ema_model = EMAModel(parameters=model.parameters(), decay=ema_decay, update_every=checkpoint_interval)
    ema_model.to(device)

    end_step = int(str(dirs[-1]).split("-")[-1])
    for step in range(0, end_step):
        if (step + 1) % checkpoint_interval == 0:
            print(f"Loading checkpoint {step + 1}...")
            model = model_cls.from_pretrained(Path(checkpoint_dir_path) / f"checkpoint-{step + 1}" / "unwrapped_model")
            model.to(device)

        ema_model.step(model.parameters())

    ema_model.copy_to(model.parameters())
    model.save_pretrained(ema_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir_path", type=str, default=None, required=True)
    parser.add_argument("--ema_save_path", type=str, default=None, required=True)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)

    args = parser.parse_args()
    offline_ema(args)
