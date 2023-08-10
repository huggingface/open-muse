import dask.dataframe as dd
import pandas as pd
import s3fs
import pyarrow as pa
from logging import Logger
import json
from webdataset import TarWriter
import re
import io
from PIL import Image
import os
import argparse
import time
import dask
from dask.diagnostics import ProgressBar

dask.config.set({"temporary_directory": "/scratch/dask_tmp"})

logger = Logger(__name__)


def cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_shard",
        type=int,
        help="The starting shard to pre-encode.",
        required=True,
    )
    parser.add_argument(
        "--end_shard",
        type=int,
        help="The ending shard to pre-encode, inclusive. If not given, defaults to `--start_shard`.",
        required=False,
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help=(
            "If set, this process is running under a batch of slurm tasks."
            "`--start_shard` and `--end_shard` must be set for the entirety of shards over all slurm tasks."
            " The shards that will be encoded in each instance of the task will be determined via"
            " the env vars `$SLURM_NTASKS` and `$SLURM_PROCID`."
        ),
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

        distributed_shards = distribute_shards(
            cli_args.start_shard, cli_args.end_shard, slurm_ntasks
        )

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
            logger.warning(
                f"slurm process: {slurm_proc_id_}, start_shard: {start_shard}, end_shard: {end_shard}"
            )
        logger.warning("************")

    return cli_args


def main(args):
    stability_metadata_dfs = []

    print("loading df 1 of 4")
    stability_metadata_dfs.append(
        dd.read_parquet(
            "/scratch/muse/laion-coyo-dedup-metadata-url-indexed/1/*.parquet",
            index="url",
            calculate_divisions=True,
        )
    )

    print("loading df 2 of 4")
    stability_metadata_dfs.append(
        dd.read_parquet(
            "/scratch/muse/laion-coyo-dedup-metadata-url-indexed/2/*.parquet",
            index="url",
            calculate_divisions=True,
        )
    )

    print("loading df 3 of 4")
    stability_metadata_dfs.append(
        dd.read_parquet(
            "/scratch/muse/laion-coyo-dedup-metadata-url-indexed/3/*.parquet",
            index="url",
            calculate_divisions=True,
        )
    )

    print("loading df 4 of 4")
    stability_metadata_dfs.append(
        dd.read_parquet(
            "/scratch/muse/laion-coyo-dedup-metadata-url-indexed/4/*.parquet",
            index="url",
            calculate_divisions=True,
        )
    )

    with open("/fsx/william/m4_annotate/shards.txt", "r") as f:
        shard_urls = f.readlines()

    s3 = s3fs.S3FileSystem()

    num_rows_found_metadata = 0
    num_rows = 0

    m4_file_n_regex = r"/(\d+)/data-(\d+)-of-\d+\.arrow"

    for shard_url_idx in range(args.start_shard, args.end_shard + 1):
        shard_url = shard_urls[shard_url_idx].strip()

        logger.warning(f"shard_url: {shard_url}")

        file_n_match = re.search(m4_file_n_regex, shard_url)

        assert file_n_match

        split_n = file_n_match.group(1)
        file_n = file_n_match.group(2)

        write_to = f"s3://muse-datasets/m4-datasets-laion-dataset-filtered-dedup-stability-metadata/{split_n}/{file_n}.tar"

        logger.warning(f"write_to: {write_to}")

        tar_writer = TarWriter(f"pipe:aws s3 cp - {write_to}")

        with s3.open(shard_url, "rb") as f:
            in_memory_stream = pa.input_stream(f)
            try:
                opened_stream = pa.ipc.open_stream(in_memory_stream)
            except pa.lib.ArrowInvalid as e:
                logger.warning(str(e))
                continue
            pa_table = opened_stream.read_all()

        table = pa_table.to_pydict()

        total_num_rows = len(table["text"])

        for i in range(total_num_rows):
            t0 = time.perf_counter()

            meta = json.loads(table["meta"][i])
            url = meta["url"]

            for df in stability_metadata_dfs:
                additional_metadata = df.loc[url].compute()

                if len(additional_metadata) >= 1:
                    # assume that all are same and just use the first
                    additional_metadata = additional_metadata.iloc[0].to_dict()

                    for k, v in additional_metadata.items():
                        if pd.isna(v):
                            additional_metadata[k] = None

                    meta["stability_metadata"] = additional_metadata

                    num_rows_found_metadata += 1

                    break

            # put `source` in the metadata dict so it doesn't have to be saved as an additional file
            meta["source"] = table["source"][i]

            image = table["image"][i]["bytes"]
            image = io.BytesIO(image)
            image = Image.open(image)

            tar_writer.write(
                {"__key__": file_n, "json": meta, "txt": table["text"][i], "jpg": image}
            )

            num_rows += 1
            logger.warning(
                f"{num_rows}/{total_num_rows} proportion found: {num_rows_found_metadata / num_rows} time: {time.perf_counter() - t0}"
            )

        tar_writer.close()

        break


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

    assert (
        sum(
            [
                end_shard - start_shard + 1
                for start_shard, end_shard in distributed_shards
            ]
        )
        == total_shards
    )

    return distributed_shards


def main(args):
    stability_metadata_dfs = []

    print("loading df 1 of 4")
    stability_metadata_dfs.append(
        dd.read_parquet(
            "/scratch/muse/laion-coyo-dedup-metadata-url-indexed/1/*.parquet",
            index="url",
            calculate_divisions=True,
        )
    )

    # print("loading df 2 of 4")
    # stability_metadata_dfs.append(
    #     dd.read_parquet(
    #         "/scratch/muse/laion-coyo-dedup-metadata-url-indexed/2/*.parquet",
    #         index="url",
    #         calculate_divisions=True,
    #     )
    # )

    # print("loading df 3 of 4")
    # stability_metadata_dfs.append(
    #     dd.read_parquet(
    #         "/scratch/muse/laion-coyo-dedup-metadata-url-indexed/3/*.parquet",
    #         index="url",
    #         calculate_divisions=True,
    #     )
    # )

    # print("loading df 4 of 4")
    # stability_metadata_dfs.append(
    #     dd.read_parquet(
    #         "/scratch/muse/laion-coyo-dedup-metadata-url-indexed/4/*.parquet",
    #         index="url",
    #         calculate_divisions=True,
    #     )
    # )

    m4_file_n_regex = r"/(\d+)/data-(\d+)-of-\d+\.arrow"

    s3 = s3fs.S3FileSystem()

    with open("/fsx/william/m4_annotate/shards.txt", "r") as f:
        shard_urls = f.readlines()

    for shard_url_idx in range(args.start_shard, args.end_shard + 1):
        shard_url = shard_urls[shard_url_idx].strip()

        file_n_match = re.search(m4_file_n_regex, shard_url)

        assert file_n_match

        split_n = file_n_match.group(1)
        file_n = file_n_match.group(2)

        logger.warning(f"shard_url: {shard_url}")

        with s3.open(shard_url, "rb") as f:
            in_memory_stream = pa.input_stream(f)
            try:
                opened_stream = pa.ipc.open_stream(in_memory_stream)
            except pa.lib.ArrowInvalid as e:
                logger.warning(str(e))
                continue
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

        # shard_saved_to = f"/scratch/muse/downloaded/{split_n}-{file_n}.parquet"
        # shard_df.to_parquet(shard_saved_to)

        t0 = time.perf_counter()

        meta = dd.from_pandas(shard_df, npartitions=1)._meta_nonempty.merge(
            stability_metadata_dfs[0]._meta_nonempty,
            left_index=True,
            right_index=True,
        )
        outp = optimized_left_join_filename(shard_df, stability_metadata_dfs[0], meta)

        with ProgressBar():
            outp = outp.compute(scheduler="processes", num_workers=24)
            # outp = outp.compute(scheduler='threads', num_workers=1)
        print(f"time for merge {time.perf_counter() - t0}")

        import ipdb

        ipdb.set_trace()

        # t0 = time.perf_counter()

        # for stability_metadata_df in stability_metadata_dfs:
        #     shard_df = shard_df.merge(stability_metadata_df, how="left", left_index=True, right_index=True)

        # shard_df = shard_df.persist()
        # print(f"time for merge {time.perf_counter() - t0}")

        # import ipdb; ipdb.set_trace()

        # TODO: turn pa_table into parquet and join


import dask.utils
import dask.dataframe.multi
from dask.highlevelgraph import HighLevelGraph
import dask.dataframe.core
from dask.base import tokenize


def optimized_left_join(lhs, rhs, meta):
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
                partition_idx == rhs.npartitions - 1
                and start <= lhs_index
                and lhs_index <= end
            )

            # For all other partitions, the end division is exclusive
            lhs_index_is_in_rhs_partition = (
                lhs_index_is_in_rhs_partition or start <= lhs_index and lhs_index < end
            )

            if lhs_index_is_in_rhs_partition:
                lhs_idx += 1
            else:
                break

        if start_lhs_idx == lhs_idx:
            # There was no overlap between lhs and the existing rhs partition, do not need to
            # search in the current rhs partition
            continue

        divisions.append(lhs.index[start_lhs_idx])
        lhs_ = lhs.iloc[start_lhs_idx:lhs_idx]
        rhs_ = rhs.get_partition(partition_idx)
        dsk[(name, len(dsk))] = (
            dask.utils.apply,
            optimized_left_join_merge_helper,
            [lhs_, rhs_],
        )
        dependencies.append(rhs_)

    divisions.append(lhs.index[-1])

    print(f"len(divisions): {len(divisions)}")

    graph = HighLevelGraph.from_collections(name, dsk, dependencies=dependencies)

    divisions = tuple(divisions)

    df = dask.dataframe.core.new_dd_object(graph, name, meta, divisions)

    return df


ctr = 0
times = 0


def optimized_left_join_merge_helper(lhs, rhs):
    # global ctr, times
    # t0 = time.perf_counter()
    rhs = rhs.compute()
    merged = lhs.merge(rhs, how="left", left_index=True, right_index=True)
    # ctr += 1
    # times += time.perf_counter() - t0
    # print(f"{ctr}: {times / ctr}")
    return merged


def optimized_left_join_filename(lhs, rhs, meta):
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
                partition_idx == rhs.npartitions - 1
                and start <= lhs_index
                and lhs_index <= end
            )

            # For all other partitions, the end division is exclusive
            lhs_index_is_in_rhs_partition = (
                lhs_index_is_in_rhs_partition or start <= lhs_index and lhs_index < end
            )

            if lhs_index_is_in_rhs_partition:
                lhs_idx += 1
            else:
                break

        if start_lhs_idx == lhs_idx:
            # There was no overlap between lhs and the existing rhs partition, do not need to
            # search in the current rhs partition
            continue

        divisions.append(lhs.index[start_lhs_idx])
        lhs_ = lhs.iloc[start_lhs_idx:lhs_idx]
        dsk[(name, len(dsk))] = (
            dask.utils.apply,
            optimized_left_join_merge_helper_filename,
            [lhs_, partition_idx],
        )

    divisions.append(lhs.index[-1])

    print(f"len(divisions): {len(divisions)}")

    graph = HighLevelGraph.from_collections(name, dsk, dependencies=dependencies)

    divisions = tuple(divisions)

    df = dask.dataframe.core.new_dd_object(graph, name, meta, divisions)

    return df


def format_shard_number(shard_n: int):
    return "{:0>{}}".format(shard_n, 5)


def optimized_left_join_merge_helper_filename(lhs, rhs_partition_idx):
    rhs_partition_idx = format_shard_number(rhs_partition_idx)
    rhs = pd.read_parquet(
        f"/scratch/muse/laion-coyo-dedup-metadata-url-indexed/1/{rhs_partition_idx}.parquet"
    )
    merged = lhs.merge(rhs, how="left", left_index=True, right_index=True)
    return merged


if __name__ == "__main__":
    main(cli_args())
