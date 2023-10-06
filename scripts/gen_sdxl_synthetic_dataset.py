import logging
import time
import os
import argparse

import pyarrow.parquet as pq
import torch
import webdataset as wds
from diffusers import DiffusionPipeline
from huggingface_hub import HfFileSystem
from transformers import CLIPModel, CLIPProcessor

torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)

logger = logging.getLogger(__name__)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--slurm", action="store_true")
    args.add_argument("--n_shards_to_write", required=True, type=int)
    args = args.parse_args()

    if args.slurm:
        slurm_procid = int(os.environ["SLURM_PROCID"])
        # `1 +` because we already used caption shard 0 while doing testing
        caption_shard_n = 1 + slurm_procid
    else:
        caption_shard_n = 0

    device = "cuda"

    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)
    clip.to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16"
    )
    pipe.load_lora_weights(
        "stabilityai/stable-diffusion-xl-base-1.0",
        weight_name="sd_xl_offset_example-lora_1.0.safetensors",
    )
    pipe.to(device)
    pipe.fuse_lora(lora_scale=0.4)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.vae.enable_slicing()

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")

    captions = get_captions(caption_shard_n)
    captions = take_up_to(captions, 500)

    for shard_n, captions_ in enumerate(captions):
        t0 = time.perf_counter()

        logger.warning(f"shard_n {shard_n}")

        writer = wds.TarWriter(
            "pipe:aws s3 cp -"
            f" s3://muse-datasets/sdxl-synthetic-dataset/{caption_shard_n}/{format_shard_number(shard_n)}.tar"
        )

        key = 0

        for caption_batch_idx, captions__ in enumerate(split_list(captions_, 8)):
            logger.warning(f"caption_batch_idx {caption_batch_idx}")

            num_inference_steps = 35
            num_images_per_prompt = 4
            proportion_base_model = 0.8

            images = pipe(
                prompt=captions__,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                denoising_end=proportion_base_model,
                output_type="latent",
            ).images
            images = refiner(
                prompt=captions__,
                num_inference_steps=num_inference_steps,
                denoising_start=proportion_base_model,
                image=images,
                num_images_per_prompt=num_images_per_prompt,
            ).images

            for caption, images_ in zip(captions__, split_list(images, 4)):
                # TODO - can we avoid syncing images to cpu
                input = clip_processor(text=caption, images=images_, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
                input["pixel_values"] = input["pixel_values"].to(dtype=torch.float16, device=device)
                input["input_ids"] = input["input_ids"].to(device)
                input["attention_mask"] = input["attention_mask"].to(device)

                clip_scores = clip(**input).logits_per_image.flatten().tolist()
                clip_scores = [str(x) for x in clip_scores]
                clip_scores = ",".join(clip_scores)

                logger.warning(f"__key__ {key}")

                writer.write(
                    {
                        "__key__": format_shard_number(key),
                        "0.png": images_[0],
                        "1.png": images_[1],
                        "2.png": images_[2],
                        "3.png": images_[3],
                        "txt": caption,
                        "clip_scores.txt": clip_scores,
                    }
                )

                key += 1

        writer.close()

        logger.warning(f"shard_n {shard_n} {time.perf_counter() - t0}")

        if shard_n + 1 > args.n_shards_to_write:
            break


def format_shard_number(shard_n: int):
    return "{:0>{}}".format(shard_n, 5)


# A shard has on the order of a million captions, so it's sufficient to just get captions from
# a single shard for a single job. Each job should pass in a unique caption_shard_n
def get_captions(caption_shard_n):
    fs = HfFileSystem()

    shards = fs.ls("datasets/laion/laion-coco", detail=False)

    shard_ctr = 0
    found_shard = None

    for shard in shards:
        if not shard.endswith(".parquet"):
            continue

        if shard_ctr == caption_shard_n:
            found_shard = shard
            break

        shard_ctr += 1

    assert found_shard is not None

    with fs.open(found_shard, "rb") as f:
        table = pq.read_table(f)

    for i in range(len(table[0])):
        caption = table[2][i]
        yield caption.as_py()


def take_up_to(iterator, n):
    iterator = iter(iterator)

    iterator_has_elements = True

    while iterator_has_elements:
        items = []

        for _ in range(n):
            try:
                items.append(next(iterator))
            except StopIteration:
                iterator_has_elements = False

        yield items


def split_list(input_list, chunk_size):
    return [input_list[i : i + chunk_size] for i in range(0, len(input_list), chunk_size)]


if __name__ == "__main__":
    main()
