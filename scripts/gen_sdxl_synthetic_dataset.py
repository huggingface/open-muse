# load sdxl pipeline
# load offset noise lora with 0.4 lora alpha
#
# For each prompt in laion coco:
#     Generate 4 sdxl images: 
#         - use either 35 steps or 20 steps
#         - use maybe moe with refiner
#     for each sdxl image: 
#         generate clip score for image
#     
from huggingface_hub import HfFileSystem
import pyarrow as pa
import logging

logger = logging.getLogger(__name__)

def main():
    fs = HfFileSystem()
    
    shards = fs.ls("datasets/laion/laion-coco", detail=False)

    for shard in shards:
        if not shard.endswith(".parquet"):
            continue

        with fs.open(shard, "r") as f:
            in_memory_stream = pa.input_stream(f)
            try:
                opened_stream = pa.ipc.open_stream(in_memory_stream)
            except pa.lib.ArrowInvalid as e:
                logger.warning(str(e))
                continue
            pa_table = opened_stream.read_all()

        table = pa_table.to_pydict()

        import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    main()
