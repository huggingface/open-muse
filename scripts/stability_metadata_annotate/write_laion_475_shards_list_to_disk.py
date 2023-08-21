from typing import List

import s3fs

LAION_475_S3_URL = "s3://muse-datasets/laion-aesthetic-475-min-1024"
LAION_475_WRITE_SHARDS_TO = "/fsx/william/open-muse/shards_laion_475.txt"

LAION_475_S3_MAX_1024_URL = "s3://muse-datasets/laion-aesthetic-475-max-1024"
LAION_475_MAX_1024_WRITE_SHARDS_TO = "/fsx/william/open-muse/shards_laion_475_resized_to_max_1024.txt"


if __name__ == "__main__":
    s3 = s3fs.S3FileSystem()

    subdirs = range(1, 10)
    dataset = "laion_475_resized_to_max_1024"

    if dataset == "laion_475":
        s3_url = LAION_475_S3_URL
        write_shards_to = LAION_475_WRITE_SHARDS_TO
    elif dataset == "laion_475_resized_to_max_1024":
        s3_url = LAION_475_S3_MAX_1024_URL
        write_shards_to = LAION_475_MAX_1024_WRITE_SHARDS_TO
    else:
        assert False

    shard_urls: List[str] = []

    for subdir in subdirs:
        shard_urls += s3.ls(f"{s3_url}/{subdir}")

    with open(write_shards_to, "w") as f:
        for shard_url in shard_urls:
            if shard_url.endswith(".tar"):
                f.write("s3://" + shard_url + "\n")
