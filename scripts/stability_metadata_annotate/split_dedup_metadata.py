"""
Step 1: 
$ mkdir -p /scratch/muse/laiocov2
$ mkdir -p /scratch/muse/laiocov2-splitted/{1,2,3,4}
$ cd /scratch/muse/laiocov2
$ aws s3 sync s3://muse-datasets/laiocov2/ .
$ rm _SUCCESS

Step 2:
$ cd /fsx/william/open-muse
$ time python -u scripts/m4_annotate/split_dedup_metadata.py --mode dryrun
$ time python -u scripts/m4_annotate/split_dedup_metadata.py --mode move_files
"""

import argparse
import logging
import os

logger = logging.Logger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dryrun", "move_files"], default="dryrun", required=False)
    args = parser.parse_args()

    initial_dedup_metadata_downloaded_to = "/scratch/muse/laiocov2"
    splitted_dedup_metadata_dir = "/scratch/muse/laiocov2-splitted"
    splitted_dedup_metadata_subdirs = [
        f"{splitted_dedup_metadata_dir}/1",
        f"{splitted_dedup_metadata_dir}/2",
        f"{splitted_dedup_metadata_dir}/3",
        f"{splitted_dedup_metadata_dir}/4",
    ]

    if args.mode == "dryrun":
        logger.warning("Running dry run. Will not actually move files")
    elif args.mode == "move_files":
        logger.warning(
            f"Running for real. Will move files from {initial_dedup_metadata_downloaded_to} to"
            f" {splitted_dedup_metadata_subdirs}"
        )
    else:
        assert False

    filepaths = os.listdir(initial_dedup_metadata_downloaded_to)
    total_num_files = len(filepaths)
    num_files_per_split = total_num_files // 4

    current_split_index = 0
    current_split_count = 0

    for filepath in filepaths:
        from_ = f"{initial_dedup_metadata_downloaded_to}/{filepath}"
        to = f"{splitted_dedup_metadata_subdirs[current_split_index]}/{filepath}"

        if args.mode == "dryrun":
            logger.warning(f"{from_}\t{to}")
        elif args.mode == "move_files":
            os.rename(from_, to)
        else:
            assert False

        current_split_count += 1

        if (
            current_split_count >= num_files_per_split
            and current_split_index < len(splitted_dedup_metadata_subdirs) - 1
        ):
            current_split_index += 1
            current_split_count = 0
