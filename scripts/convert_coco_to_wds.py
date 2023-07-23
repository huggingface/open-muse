# To download the 2017 train split of coco
# $ wget http://images.cocodataset.org/zips/train2017.zip
# $ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
#
# script assumes they're downloaded and unzipped into ../data
#
# we write into ../data/coco-2017-train/ because shard writer doesn't support piped
# uploads
#
# after writing to disk, run
# $ aws s3 cp ../data/coco-2017-train/ s3://muse-datasets/coco/2017/train/ --recursive

# Needed for `PIL.Image` to work in wds :/

import json

import webdataset as wds
from cv2 import COLOR_BGR2RGB, cvtColor, imread


def main():
    with open("../data/annotations/captions_train2017.json") as f:
        annotations = json.load(f)["annotations"]

    annotations_by_image_id = {}

    for annotation in annotations:
        image_id = annotation["image_id"]

        if image_id not in annotations_by_image_id:
            annotations_by_image_id[image_id] = []

        annotations_by_image_id[image_id].append(annotation)

    shard_writer = wds.ShardWriter("../data/coco-2017-train/%05d.tar", maxsize=5e8)

    for image_id, annotations in annotations_by_image_id.items():
        print(f"writing {image_id}")
        image = imread("../data/train2017/%012d.jpg" % image_id)
        image = cvtColor(image, COLOR_BGR2RGB)

        annotations_metadata = []

        for annotation in annotations:
            annotations_metadata.append({"id": annotation["id"], "caption": annotation["caption"]})

        metadata = {"annotations": json.dumps(annotations_metadata)}

        shard_writer.write({"__key__": str(image_id), "json": metadata, "jpg": image})

    shard_writer.close()


if __name__ == "__main__":
    main()
