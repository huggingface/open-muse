import json
import logging
import os
import time
from argparse import ArgumentParser

import pandas as pd
import torch
import webdataset as wds
from torch.utils.data import DataLoader, Dataset

try:
    from cleanfid import fid
except:
    raise ImportError("Please install cleanfid: pip install clean_fid")


from muse import PipelineMuse

logger = logging.getLogger(__name__)


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


def generate_and_save_images_flickr_8k(args):
    """
    Generate images from captions and save them to disk.
    """
    os.makedirs(args.save_path, exist_ok=True)

    logger.warning("Loading pipe")
    pipeline = PipelineMuse.from_pretrained(args.model_name_or_path).to(args.device)
    pipeline.transformer.enable_xformers_memory_efficient_attention()

    logger.warning("Loading data")
    dataset = Flickr8kDataset(args.dataset_root, args.dataset_captions_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    generator = torch.Generator(args.device).manual_seed(args.seed)

    logger.warning("Generating images")
    for batch in dataloader:
        image_names = batch[0]
        text = batch[1]

        images = pipeline(
            text,
            timesteps=args.timesteps,
            guidance_scale=args.guidance_scale,
            temperature=args.temperature,
            generator=generator,
            use_tqdm=False,
        )

        for image_name, image in zip(image_names, images):
            image.save(os.path.join(args.save_path, f"{image_name}"))


def distribute_shards(start_shard_all, end_shard_all, slurm_ntasks):
    total_shards = end_shard_all - start_shard_all + 1
    shards_per_task = total_shards // slurm_ntasks
    shards_per_task = [shards_per_task] * slurm_ntasks

    # to distribute the remainder of tasks for non-evenly divisible number of shards
    left_over_shards = total_shards % slurm_ntasks

    for slurm_procid in range(left_over_shards):
        shards_per_task[slurm_procid] += 1

    assert sum(shards_per_task) == total_shards

    distributed_shards = []

    for slurm_procid in range(len(shards_per_task)):
        if slurm_procid == 0:
            start_shard = start_shard_all
        else:
            start_shard = distributed_shards[slurm_procid - 1][1] + 1

        end_shard = start_shard + shards_per_task[slurm_procid] - 1
        distributed_shards.append((start_shard, end_shard))

    assert sum([end_shard - start_shard + 1 for start_shard, end_shard in distributed_shards]) == total_shards

    return distributed_shards


def format_shard_number(shard_n: int):
    return "{:0>{}}".format(shard_n, 5)


def generate_and_save_images_coco(args):
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.dataset_root, exist_ok=True)

    logger.warning("Loading pipe")
    pipeline = PipelineMuse.from_pretrained(args.model_name_or_path).to(args.device)
    pipeline.transformer.enable_xformers_memory_efficient_attention()

    logger.warning("Loading data")

    # 20 shards is safe range to get 30k images
    start_shard = 0
    end_shard = 20
    num_images_to_generate = 30_000

    if args.slurm:
        slurm_ntasks = int(os.environ["SLURM_NTASKS"])
        slurm_procid = int(os.environ["SLURM_PROCID"])

        distributed_shards = distribute_shards(start_shard, end_shard, slurm_ntasks)

        start_shard, end_shard = distributed_shards[slurm_procid]
        num_images_to_generate = round(num_images_to_generate / slurm_ntasks)

        logger.warning("************")
        logger.warning("Running as slurm task")
        logger.warning(f"SLURM_NTASKS: {slurm_ntasks}")
        logger.warning(f"SLURM_PROCID: {slurm_procid}")
        logger.warning(f"start_shard: {start_shard}, end_shard: {end_shard}")
        logger.warning("************")
        logger.warning(f"all slurm processes")
        for slurm_proc_id_, (proc_start_shard, proc_end_shard) in enumerate(distributed_shards):
            logger.warning(
                f"slurm process: {slurm_proc_id_}, start_shard: {proc_start_shard}, end_shard: {proc_end_shard}"
            )
        logger.warning("************")

    shard_range = "{" + format_shard_number(start_shard) + ".." + format_shard_number(end_shard) + "}"
    download_shards = f"pipe:aws s3 cp s3://muse-datasets/coco/2017/train/{shard_range}.tar -"

    logger.warning(f"downloading shards {download_shards}")

    dataset = (
        wds.WebDataset(download_shards)
        .decode("pil", handler=wds.warn_and_continue)
        .rename(image="jpg;png;jpeg;webp", metadata="json")
        .map(
            lambda dict: {
                "__key__": dict["__key__"],
                "image": dict["image"],
                "metadata": dict["metadata"],
            }
        )
        .to_tuple("__key__", "image", "metadata")
        .batched(args.batch_size)
    )
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=0,
    )

    generator = torch.Generator(args.device).manual_seed(args.seed)

    logger.warning("Generating images")

    num_images_generated = 0

    for __key__, real_image, metadata in dataloader:
        logger.warning(f"Creating {len(__key__)} images: {__key__[0]} {__key__[-1]}")
        num_images_generated += len(__key__)

        text = [json.loads(x["annotations"])[0]["caption"] for x in metadata]

        t0 = time.perf_counter()

        generated_image = pipeline(
            text,
            timesteps=args.timesteps,
            guidance_scale=args.guidance_scale,
            temperature=args.temperature,
            generator=generator,
            use_tqdm=False,
        )

        logger.warning(f"Generation time {time.perf_counter() - t0}")

        for __key__, generated_image, real_image in zip(__key__, generated_image, real_image):
            real_image.save(os.path.join(args.dataset_root, f"{__key__}.png"))
            generated_image.save(os.path.join(args.save_path, f"{__key__}.png"))

        logger.warning(f"Generated {num_images_generated}/{num_images_to_generate}")

        if num_images_generated >= num_images_to_generate:
            logger.warning("done")
            break


def main(args):
    if args.do in ["full", "generate_and_save_images"]:
        if args.dataset == "flickr_8k":
            generate_and_save_images_flickr_8k(args)
        elif args.dataset == "coco":
            generate_and_save_images_coco(args)
        else:
            assert False

    if args.do in ["full", "compute_fid"]:
        real_images = args.dataset_root
        generated_images = args.save_path
        logger.warning("computing FiD")
        score_clean = fid.compute_fid(real_images, generated_images, mode="clean", num_workers=0)
        logger.warning(f"clean-fid score is {score_clean:.3f}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=False)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--dataset_captions_file", type=str, required=False)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--timesteps", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=8.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--dataset", type=str, default="flickr_8k", choices=("flickr_8k", "coco"))
    parser.add_argument("--do", type=str, default="full", choices=("full", "generate_and_save_images", "compute_fid"))
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Set when running as a slurm job to distribute coco image generation among multiple GPUs",
    )

    args = parser.parse_args()

    if args.do in ["full", "generated_and_save_images"]:
        if args.dataset == "flickr_8k" and args.dataset_captions_file is None:
            raise ValueError("`--dataset=flickr_8k` requires setting `--dataset_captions_file`")

        if args.model_name_or_path is None:
            raise ValueError("`--do=full|generate_and_save_images` requires setting `--model_name_or_path`")

    if args.do == "full":
        logger.warning("generating images and calculating fid")
    elif args.do == "generate_and_save_images":
        logger.warning("just generating and saving images")
    elif args.do == "compute_fid":
        logger.warning("just computing fid")
    else:
        assert False

    main(args)
