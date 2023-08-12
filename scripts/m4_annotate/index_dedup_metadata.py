"""
Step 3:
$ mkdir -p /scratch/muse/laicov2-url-indexed

$ time python -u scripts/m4_annotate/index_dedup_metadata --sub_dir_name 1
$ time python -u scripts/m4_annotate/index_dedup_metadata --sub_dir_name 2
$ time python -u scripts/m4_annotate/index_dedup_metadata --sub_dir_name 3
$ time python -u scripts/m4_annotate/index_dedup_metadata --sub_dir_name 4

$ aws s3 sync /scratch/muse/laiocov2-url-indexed s3://muse-datasets/laiocov2-url-indexed/
"""

import argparse
import logging
from time import perf_counter

import dask
from dask import dataframe

logger = logging.Logger(__name__)

if __name__ == "__main__":
    dask.config.set({"temporary_directory": "/scratch/dask_tmp"})

    parser = argparse.ArgumentParser()
    parser.add_argument("--sub_dir_name", choices=["1", "2", "3", "4"], required=True, type=str)
    args = parser.parse_args()

    splitted_dedup_metadata_dir = "/scratch/muse/laiocov2-url-splitted"
    url_indexed_dedup_metadata_dir = "/scratch/muse/laiocov2-url-indexed"

    read_from = f"{splitted_dedup_metadata_dir}/{args.sub_dir_name}/*.parquet"
    write_to = f"{url_indexed_dedup_metadata_dir}/{args.sub_dir_name}"

    logger.warning(f"reading from {read_from}")
    logger.warning(f"writing to {write_to}")

    df = dataframe.read_parquet(read_from)

    npartitions = df.npartitions

    t0 = perf_counter()
    df = df.repartition(npartitions=500)
    logger.warning(f"repartition {perf_counter() - t0}")

    t0 = perf_counter()
    df = df.set_index("url")
    logger.warning(f"set_index {perf_counter() - t0}")

    t0 = perf_counter()
    df = df.repartition(npartitions=npartitions)
    logger.warning(f"repartition {perf_counter() - t0}")

    t0 = perf_counter()
    df.to_parquet(
        write_to,
        name_function=lambda i: "{:0>{}}".format(i, 5) + ".parquet",
    )
    logger.warning(f"to_parquet {perf_counter() - t0}")
