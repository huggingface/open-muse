# Adapted from https://github.com/webdataset/webdataset-imagenet/blob/main/convert-imagenet.py

import argparse
import os
import sys
import time

import webdataset as wds
from datasets import load_dataset


def convert_imagenet_to_wds(output_dir, max_train_samples_per_shard, max_val_samples_per_shard):
    assert not os.path.exists(os.path.join(output_dir, "imagenet-train-000000.tar"))
    assert not os.path.exists(os.path.join(output_dir, "imagenet-val-000000.tar"))

    opat = os.path.join(output_dir, "imagenet-train-%06d.tar")
    output = wds.ShardWriter(opat, maxcount=max_train_samples_per_shard)
    dataset = load_dataset("imagenet-1k", streaming=True, split="train", use_auth_token=True)
    now = time.time()
    for i, example in enumerate(dataset):
        if i % max_train_samples_per_shard == 0:
            print(i, file=sys.stderr)
        img, label = example["image"], example["label"]
        output.write({"__key__": "%08d" % i, "jpg": img.convert("RGB"), "cls": label})
    output.close()
    time_taken = time.time() - now
    print(f"Wrote {i+1} train examples in {time_taken // 3600} hours.")

    opat = os.path.join(output_dir, "imagenet-val-%06d.tar")
    output = wds.ShardWriter(opat, maxcount=max_val_samples_per_shard)
    dataset = load_dataset("imagenet-1k", streaming=True, split="validation", use_auth_token=True)
    now = time.time()
    for i, example in enumerate(dataset):
        if i % max_val_samples_per_shard == 0:
            print(i, file=sys.stderr)
        img, label = example["image"], example["label"]
        output.write({"__key__": "%08d" % i, "jpg": img.convert("RGB"), "cls": label})
    output.close()
    time_taken = time.time() - now
    print(f"Wrote {i+1} val examples in {time_taken // 60} min.")


if __name__ == "__main__":
    # create parase object
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--max_train_samples_per_shard", type=int, default=4000, help="Path to the output directory.")
    parser.add_argument("--max_val_samples_per_shard", type=int, default=1000, help="Path to the output directory.")
    args = parser.parse_args()

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    convert_imagenet_to_wds(args.output_dir, args.max_train_samples_per_shard, args.max_val_samples_per_shard)
