from typing import List

import s3fs

# the original data is downloaded from
LAION_475_S3_URL = "s3://muse-datasets/laion-aesthetic-475-min-1024"

if __name__ == "__main__":
    s3 = s3fs.S3FileSystem()

    shard_urls: List[str] = s3.ls(f"{LAION_475_S3_URL}/1")
    shard_urls += s3.ls(f"{LAION_475_S3_URL}/2")
    shard_urls += s3.ls(f"{LAION_475_S3_URL}/3")
    # TODO - enable these once they're ready
    # shard_urls += s3.ls(f"{LAION_475_S3_URL}/4")
    # shard_urls += s3.ls(f"{LAION_475_S3_URL}/5")

    with open("/fsx/william/open-muse/shards_laion_475.txt", "w") as f:
        for shard_url in shard_urls:
            if shard_url.endswith(".tar"):
                f.write("s3://" + shard_url + "\n")
