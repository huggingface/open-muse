import argparse
import concurrent.futures
import io
import json
import multiprocessing as mp
import os
import re
import time
from logging import Logger

import dask
import dask.dataframe as dd
import dask.dataframe.core
import dask.dataframe.multi
import dask.utils
import pandas as pd
import pyarrow as pa
import s3fs
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from PIL import Image
from webdataset import TarWriter
from webdataset.writer import add_handlers, make_handlers

LAION_COYO_DEDUP_METADATA_URL_INDEXED_ROOT_DIR = "/scratch/muse/laicov2-url-indexed"

M4_FILE_N_REGEX = r"/(\d+)/data-(\d+)-of-\d+\.arrow"

# We manually add the `_stability_metadata` suffix to make it easier
# to move them into their own subdict separate from the existing metadata
COLS_FROM_STABILITY_METADATA_RENAMES = {
    "SSCD_85": "SSCD_85_stability_metadata",
    "SSCD_75": "SSCD_75_stability_metadata",
    "SSCD_65": "SSCD_65_stability_metadata",
    "SSCD_50": "SSCD_50_stability_metadata",
    "is_spawning": "is_spawning_stability_metadata",
    "is_getty": "is_getty_stability_metadata",
    "p_watermarkdf": "p_watermarkdf_stability_metadata",
    "p_nsfwdf": "p_nsfwdf_stability_metadata",
    "p_bumble": "p_bumble_stability_metadata",
    "gnt_drawings": "gnt_drawings_stability_metadata",
    "gnt_hentai": "gnt_hentai_stability_metadata",
    "gnt_neutral": "gnt_neutral_stability_metadata",
    "gnt_porn": "gnt_porn_stability_metadata",
    "gnt_sexy": "gnt_sexy_stability_metadata",
}

COLS_FROM_STABILITY_METADATA_DROPS = [
    "is_coyo",
    "is_laion",
    "Id",
    "caption",
    "key",
    "status",
    "error_message",
    "width",
    "height",
    "original_width",
    "original_height",
    "exif",
    "sha256",
]

COLS_FROM_STABILITY_METADATA = COLS_FROM_STABILITY_METADATA_RENAMES.values()


logger = Logger(__name__)


def cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_shard",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--end_shard",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
    )
    parser.add_argument(
        "--skip_upload",
        action="store_true",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
    )

    cli_args = parser.parse_args()

    if cli_args.slurm and cli_args.end_shard is None:
        raise ValueError("`--end_shard` must be set when `--slurm` is set")

    if cli_args.end_shard is None:
        cli_args.end_shard = cli_args.start_shard

    if cli_args.end_shard < cli_args.start_shard:
        raise ValueError("`--end_shard` must be >= `--start_shard`")

    if cli_args.slurm:
        slurm_procid = int(os.environ["SLURM_PROCID"])
        slurm_ntasks = int(os.environ["SLURM_NTASKS"])

        distributed_shards = distribute_shards(cli_args.start_shard, cli_args.end_shard, slurm_ntasks)

        start_shard_task, end_shard_task = distributed_shards[slurm_procid]

        cli_args.start_shard = start_shard_task
        cli_args.end_shard = end_shard_task

        logger.warning("************")
        logger.warning("Running as slurm task")
        logger.warning(f"SLURM_NTASKS: {slurm_ntasks}")
        logger.warning(f"SLURM_PROCID: {slurm_procid}")
        logger.warning(f"start_shard: {start_shard_task}, end_shard: {end_shard_task}")
        logger.warning("************")
        logger.warning(f"all slurm processes")
        for slurm_proc_id_, (start_shard, end_shard) in enumerate(distributed_shards):
            logger.warning(f"slurm process: {slurm_proc_id_}, start_shard: {start_shard}, end_shard: {end_shard}")
        logger.warning("************")

    return cli_args


def main(args):
    temp_dir = "/scratch/dask_tmp"
    os.makedirs(temp_dir, exist_ok=True)
    dask.config.set({"temporary_directory": temp_dir})

    os.makedirs(LAION_COYO_DEDUP_METADATA_URL_INDEXED_ROOT_DIR, exist_ok=True)
    os.system(f"aws s3 sync s3://muse-datasets/laiocov2-url-indexed/ {LAION_COYO_DEDUP_METADATA_URL_INDEXED_ROOT_DIR}")

    if args.n_workers == 1:
        single_process_main(args, 0)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            for _ in pool.map(single_process_main, [args] * args.n_workers, range(args.n_workers)):
                ...


def single_process_main(args, process_idx):
    # process pool executor restricts the cpu affinity for workers
    os.sched_setaffinity(os.getpid(), range(os.cpu_count()))

    start_shard, end_shard = distribute_shards(args.start_shard, args.end_shard, args.n_workers)[process_idx]
    args.start_shard = start_shard
    args.end_shard = end_shard

    logger.warning(f"worker process: {process_idx} loading stability metadata 1 of 4")
    stability_metadata_dfs_1 = dd.read_parquet(
        f"{LAION_COYO_DEDUP_METADATA_URL_INDEXED_ROOT_DIR}/1/*.parquet",
        index="url",
        calculate_divisions=True,
    )

    logger.warning(f"worker process: {process_idx} loading stability metadata 2 of 4")
    stability_metadata_dfs_2 = dd.read_parquet(
        f"{LAION_COYO_DEDUP_METADATA_URL_INDEXED_ROOT_DIR}/2/*.parquet",
        index="url",
        calculate_divisions=True,
    )

    logger.warning(f"worker process: {process_idx} loading stability metadata 3 of 4")
    stability_metadata_dfs_3 = dd.read_parquet(
        f"{LAION_COYO_DEDUP_METADATA_URL_INDEXED_ROOT_DIR}/3/*.parquet",
        index="url",
        calculate_divisions=True,
    )

    logger.warning(f"worker process: {process_idx} loading stability metadata 4 of 4")
    stability_metadata_dfs_4 = dd.read_parquet(
        f"{LAION_COYO_DEDUP_METADATA_URL_INDEXED_ROOT_DIR}/4/*.parquet",
        index="url",
        calculate_divisions=True,
    )

    s3 = s3fs.S3FileSystem()

    with open("/fsx/william/open-muse/shards.txt", "r") as f:
        shard_urls = f.readlines()

    # Runs with 97 total processes (one more than total processors) -
    #
    # 1 launcher + 4 "main" subprocesses + 88 worker subsubprocesses + 4 additional processes that I assume are being launched by
    # the dask runtime to manage the worker pool.
    #
    # It's likely those additional processes aren't doing heavy work and it would be ok to have a slightly larger pool,
    # but I'm just being safe because I worry there's some additional process switching that would occur.
    # I also took some very rough measurements on writing a total of 2 shards per worker process and
    # it was marginally faster.
    with concurrent.futures.ProcessPoolExecutor(max_workers=22, mp_context=mp.get_context("spawn")) as process_pool:
        upload_futures = []

        with dask.config.set(pool=process_pool):
            for shard_url_idx in range(args.start_shard, args.end_shard + 1):
                shard_url = shard_urls[shard_url_idx].strip()

                logger.warning(f"[{args.start_shard}..{shard_url_idx}..{args.end_shard}] shard_url: {shard_url}")

                shard_df = read_shard_and_create_data_frame(s3, shard_url)

                if shard_df is None:
                    continue

                t0 = time.perf_counter()

                t00 = time.perf_counter()
                meta = make_optimized_left_join_meta(shard_df, stability_metadata_dfs_1)
                shard_df_1 = optimized_left_join(shard_df, stability_metadata_dfs_1, meta, 1)
                shard_df_1 = shard_df_1.compute()
                shard_df_1 = shard_df_1[~shard_df_1.index.duplicated(keep="first")]
                logger.warning(
                    f"[{args.start_shard}..{shard_url_idx}..{args.end_shard}] time for merge 1"
                    f" {time.perf_counter() - t00}"
                )

                t00 = time.perf_counter()
                meta = make_optimized_left_join_meta(shard_df, stability_metadata_dfs_2)
                shard_df_2 = optimized_left_join(shard_df, stability_metadata_dfs_2, meta, 2)
                shard_df_2 = shard_df_2.compute()
                shard_df_2 = shard_df_2[~shard_df_2.index.duplicated(keep="first")]
                logger.warning(
                    f"[{args.start_shard}..{shard_url_idx}..{args.end_shard}] time for merge 2"
                    f" {time.perf_counter() - t00}"
                )

                t00 = time.perf_counter()
                meta = make_optimized_left_join_meta(shard_df, stability_metadata_dfs_3)
                shard_df_3 = optimized_left_join(shard_df, stability_metadata_dfs_3, meta, 3)
                shard_df_3 = shard_df_3.compute()
                shard_df_3 = shard_df_3[~shard_df_3.index.duplicated(keep="first")]
                logger.warning(
                    f"[{args.start_shard}..{shard_url_idx}..{args.end_shard}] time for merge 3"
                    f" {time.perf_counter() - t00}"
                )

                t00 = time.perf_counter()
                meta = make_optimized_left_join_meta(shard_df, stability_metadata_dfs_4)
                shard_df_4 = optimized_left_join(shard_df, stability_metadata_dfs_4, meta, 4)
                shard_df_4 = shard_df_4.compute()
                shard_df_4 = shard_df_4[~shard_df_4.index.duplicated(keep="first")]
                logger.warning(
                    f"[{args.start_shard}..{shard_url_idx}..{args.end_shard}] time for merge 4"
                    f" {time.perf_counter() - t00}"
                )

                shard_df_joined = (
                    shard_df_1.combine_first(shard_df_2).combine_first(shard_df_3).combine_first(shard_df_4)
                )

                logger.warning(
                    f"[{args.start_shard}..{shard_url_idx}..{args.end_shard}] time for total merge"
                    f" {time.perf_counter() - t0}"
                )

                if not args.skip_upload:
                    upload_future = process_pool.submit(
                        write_joined_data_to_new_s3_bucket_as_wds,
                        shard_df_joined,
                        shard_url,
                        args.start_shard,
                        args.end_shard,
                        shard_url_idx,
                    )

                    upload_futures.append(upload_future)

                    for i in range(len(upload_futures) - 1, -1, -1):
                        upload_future = upload_futures[i]

                        if upload_future.done():
                            upload_future.result()  # To raise exception if occurred
                            del upload_futures[i]

        concurrent.futures.wait(upload_futures)


def distribute_shards(start_shard_all, end_shard_all, ntasks):
    total_shards = end_shard_all - start_shard_all + 1
    shards_per_task = total_shards // ntasks
    shards_per_task = [shards_per_task] * ntasks

    # to distribute the remainder of tasks for non-evenly divisible number of shards
    left_over_shards = total_shards % ntasks

    for task_idx in range(left_over_shards):
        shards_per_task[task_idx] += 1

    assert sum(shards_per_task) == total_shards

    distributed_shards = []

    for task_idx in range(len(shards_per_task)):
        if task_idx == 0:
            start_shard = start_shard_all
        else:
            start_shard = distributed_shards[task_idx - 1][1] + 1

        end_shard = start_shard + shards_per_task[task_idx] - 1
        distributed_shards.append((start_shard, end_shard))

    assert sum([end_shard - start_shard + 1 for start_shard, end_shard in distributed_shards]) == total_shards

    return distributed_shards


def read_shard_and_create_data_frame(s3, shard_url):
    with s3.open(shard_url, "rb") as f:
        in_memory_stream = pa.input_stream(f)
        try:
            opened_stream = pa.ipc.open_stream(in_memory_stream)
        except pa.lib.ArrowInvalid as e:
            logger.warning(str(e))
            return None
        pa_table = opened_stream.read_all()

    shard_df = pa_table.to_pandas()

    def parse_url(x):
        try:
            x = json.loads(x)
        except:
            return pd.NA

        return x["url"]

    shard_df["url"] = shard_df["meta"].apply(parse_url)

    shard_df = shard_df.set_index("url")

    shard_df.sort_index(inplace=True)

    return shard_df


def make_optimized_left_join_meta(shard_df, stability_metadata_df):
    return dd.from_pandas(shard_df, npartitions=1)._meta_nonempty.merge(
        stability_metadata_df._meta_nonempty,
        left_index=True,
        right_index=True,
    )


def optimized_left_join(lhs, rhs, meta, stability_metadata_dir_idx):
    name = "optimized-left-join-" + tokenize(lhs, rhs)

    dsk = dict()
    dependencies = []
    divisions = []

    lhs_idx = 0

    for partition_idx in range(rhs.npartitions):
        if lhs_idx >= len(lhs):
            break

        start = rhs.divisions[partition_idx]
        end = rhs.divisions[partition_idx + 1]

        start_lhs_idx = lhs_idx

        while lhs_idx < len(lhs):
            lhs_index = lhs.index[lhs_idx]

            # For the last partition, the end divisions is inclusive
            lhs_index_is_in_rhs_partition = (
                partition_idx == rhs.npartitions - 1 and start <= lhs_index and lhs_index <= end
            )

            # For all other partitions, the end division is exclusive
            lhs_index_is_in_rhs_partition = lhs_index_is_in_rhs_partition or start <= lhs_index and lhs_index < end

            if lhs_index_is_in_rhs_partition:
                lhs_idx += 1
            else:
                break

        if start_lhs_idx == lhs_idx:
            # There was no overlap between lhs and the existing rhs partition, do not need to
            # search in the current rhs partition
            continue

        divisions.append(lhs.index[start_lhs_idx])
        lhs_for_single_rhs_partition = lhs.iloc[start_lhs_idx:lhs_idx]
        dsk[(name, len(dsk))] = (
            dask.utils.apply,
            optimized_left_join_merge_with_one_partition,
            [lhs_for_single_rhs_partition, partition_idx, stability_metadata_dir_idx],
        )

    divisions.append(lhs.index[-1])

    graph = HighLevelGraph.from_collections(name, dsk, dependencies=dependencies)

    divisions = tuple(divisions)

    df = dask.dataframe.core.new_dd_object(graph, name, meta, divisions)

    return df


def optimized_left_join_merge_with_one_partition(
    lhs_for_single_rhs_partition: pd.DataFrame, rhs_partition_idx, stability_metadata_dir_idx
):
    rhs_partition_idx = format_shard_number(rhs_partition_idx)

    rhs_partition = pd.read_parquet(
        f"{LAION_COYO_DEDUP_METADATA_URL_INDEXED_ROOT_DIR}/{stability_metadata_dir_idx}/{rhs_partition_idx}.parquet"
    )

    rhs_partition.drop(labels=COLS_FROM_STABILITY_METADATA_DROPS, axis="columns", inplace=True)

    rhs_partition.rename(columns=COLS_FROM_STABILITY_METADATA_RENAMES, inplace=True)

    single_partition_merged = lhs_for_single_rhs_partition.join(rhs_partition)

    return single_partition_merged


def write_joined_data_to_new_s3_bucket_as_wds(shard_df, shard_url, start_shard, end_shard, shard_url_idx):
    # process pool executor restricts the cpu affinity for workers
    os.sched_setaffinity(os.getpid(), range(os.cpu_count()))

    t0 = time.perf_counter()

    file_n_match = re.search(M4_FILE_N_REGEX, shard_url)

    assert file_n_match

    split_n = file_n_match.group(1)
    file_n = file_n_match.group(2)

    write_to = f"s3://muse-datasets/m4-datasets-laion-dataset-filtered-dedup-joined-with-stability-metadata-laicov2/{split_n}/{file_n}.tar"

    logger.warning(f"[{start_shard}..{shard_url_idx}..{end_shard}] write_to: {write_to}")

    tar_writer = TarWriter(f"pipe:aws s3 cp - {write_to}", encoder=TAR_WRITER_ENCODER)

    for count, (url, row) in enumerate(shard_df.iterrows()):
        image = row["image"]["bytes"]
        image = io.BytesIO(image)
        image = Image.open(image)
        try:
            image.load()
        except OSError:
            continue

        __key__ = format_shard_number(count)

        txt = row["text"]

        json_ = json.loads(row["meta"])
        # put `source` in the metadata dict so it doesn't have to be saved as an additional file
        json_["source"] = row["source"]

        row_stability_metadata = {}

        for col in COLS_FROM_STABILITY_METADATA:
            val = row[col]

            if pd.isna(val):
                continue

            col_ = col.replace("_stability_metadata", "")

            row_stability_metadata[col_] = val

        json_["stability_metadata"] = row_stability_metadata

        tar_writer.write({"__key__": __key__, "json": json_, "txt": txt, "jpg": image})

    tar_writer.close()

    logger.warning(f"[{start_shard}..{shard_url_idx}..{end_shard}] time for write {time.perf_counter() - t0}")


def format_shard_number(shard_n: int):
    return "{:0>{}}".format(shard_n, 5)


# default webdatasets jpeg encoder sets quality to 100 which makes the archives
# twice as large as the m4 arrows
def jpeg_encoder(image: Image.Image):
    with io.BytesIO() as result:
        image.save(result, format="JPEG")
        return result.getvalue()


TAR_WRITER_ENCODER = make_handlers()
add_handlers(TAR_WRITER_ENCODER, "jpg jpeg img image", jpeg_encoder)


if __name__ == "__main__":
    main(cli_args())
