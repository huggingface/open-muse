# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is heavily inspired by https://github.com/mlfoundations/open_clip/blob/main/src/training/data.py

import io
import itertools
import json
import logging
import math
import os
import random
import re
import time
from typing import List, Optional, Union

import pyarrow as pa
import s3fs
import webdataset as wds
from braceexpand import braceexpand
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, default_collate
from torchvision import transforms
from transformers import PreTrainedTokenizer
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)

logger = logging.Logger(__name__)

person_token = ["a person", "someone", "somebody"]


def replace_person_token(t):
    "Used for CC12M"
    t = re.sub("<person>([,\s]*(and)*[,\s]*<person>)+", " people ", t)
    while "<person>" in t:
        t = t.replace("<person>", f" {random.choices(person_token)} ", 1)
    return t


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=wds.warn_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


class ImageNetTransform:
    def __init__(self, resolution, center_crop=True, random_flip=False):
        self.train_transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                (transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution)),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
            ]
        )
        self.eval_transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
            ]
        )


class ClassificationDataset:
    def __init__(
        self,
        train_shards_path_or_url: Union[str, List[str]],
        eval_shards_path_or_url: Union[str, List[str]],
        num_train_examples: int,
        per_gpu_batch_size: int,
        global_batch_size: int,
        num_workers: int,
        resolution: int = 256,
        return_text: bool = False,
        tokenizer: PreTrainedTokenizer = None,
        max_seq_length: int = 16,
        center_crop: bool = True,
        random_flip: bool = False,
        imagenet_class_mapping_path=None,
        shuffle_buffer_size: int = 1000,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        **kwargs,
    ):
        transform = ImageNetTransform(resolution, center_crop, random_flip)

        if return_text:
            if imagenet_class_mapping_path is None:
                raise ValueError("imagenet_class_mapping_path must be provided when return_text is True")

            with open(imagenet_class_mapping_path, "r") as f:
                self.class_mapping = json.load(f)

            def tokenize(imagenet_class_id):
                text = self.class_mapping[str(imagenet_class_id)]
                input_ids = tokenizer(
                    text, max_length=max_seq_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                return input_ids[0]

            processing_pipeline = [
                wds.rename(
                    image="jpg;png;jpeg;webp",
                    input_ids="cls",
                    text_raw="cls",
                    class_id="cls",
                    handler=wds.warn_and_continue,
                ),
                wds.map(filter_keys(set(["image", "input_ids", "text_raw", "class_idx"]))),
                wds.map_dict(
                    image=transform.train_transform,
                    input_ids=tokenize,
                    text_raw=lambda class_idx: self.class_mapping[str(class_idx)],
                ),
                wds.to_tuple("image", "input_ids"),
            ]
        else:
            processing_pipeline = [
                wds.rename(image="jpg;png;jpeg;webp", class_id="cls", handler=wds.warn_and_continue),
                wds.map(filter_keys(set(["image", "class_id"]))),
                wds.map_dict(image=transform.train_transform, class_id=lambda x: int(x)),
                wds.to_tuple("image", "class_id"),
            ]

        # Create train dataset and loader
        pipeline = [
            wds.ResampledShards(train_shards_path_or_url),
            wds.tarfile_to_samples(handler=wds.ignore_and_continue),
            wds.shuffle(shuffle_buffer_size),
            wds.decode("pil", handler=wds.ignore_and_continue),
            *processing_pipeline,
            wds.batched(per_gpu_batch_size, partial=False, collation_fn=default_collate),
        ]

        num_batches = math.ceil(num_train_examples / global_batch_size)
        num_worker_batches = math.ceil(num_train_examples / (global_batch_size * num_workers))  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size

        # each worker is iterating over this
        self._train_dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
        self._train_dataloader = wds.WebLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        # add meta-data to dataloader instance for convenience
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

        # Create eval dataset and loader
        pipeline = [
            wds.SimpleShardList(eval_shards_path_or_url),
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.ignore_and_continue),
            wds.decode("pil", handler=wds.ignore_and_continue),
            *processing_pipeline,
            wds.batched(per_gpu_batch_size, partial=True, collation_fn=default_collate),
        ]
        self._eval_dataset = wds.DataPipeline(*pipeline)
        self._eval_dataloader = wds.WebLoader(
            self._eval_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader

    @property
    def eval_dataset(self):
        return self._eval_dataset

    @property
    def eval_dataloader(self):
        return self._eval_dataloader


# taken from https://github.com/dome272/Paella/blob/main/src_distributed/utils.py#L20
class WebdatasetFilter:
    def __init__(self, min_size=256, max_pwatermark=0.5, aesthetic_threshold=4.9):
        self.min_size = min_size
        self.max_pwatermark = max_pwatermark
        self.aesthetic_threshold = aesthetic_threshold

    def __call__(self, x):
        try:
            if "json" in x:
                x_json = json.loads(x["json"])
                filter_size = (x_json.get("original_width", 0.0) or 0.0) >= self.min_size and x_json.get(
                    "original_height", 0
                ) >= self.min_size
                filter_watermark = (x_json.get("pwatermark", 1.0) or 1.0) <= self.max_pwatermark
                filter_watermark_coyo = (x_json.get("watermark_score", 1.0) or 1.0) <= self.max_pwatermark
                filter_aesthetic_a = (x_json.get("aesthetic", 0.0) or 0.0) >= self.aesthetic_threshold
                filter_aesthetic_b = (x_json.get("AESTHETIC_SCORE", 0.0) or 0.0) >= self.aesthetic_threshold
                filter_aesthetic_coyo = (
                    x_json.get("aesthetic_score_laion_v2", 0.0) or 0.0
                ) >= self.aesthetic_threshold
                return (
                    filter_size
                    and (filter_watermark or filter_watermark_coyo)
                    and (filter_aesthetic_a or filter_aesthetic_b or filter_aesthetic_coyo)
                )
            else:
                return False
        except:
            return False


class Text2ImageDataset:
    def __init__(
        self,
        train_shards_path_or_url: Union[str, List[str]],
        eval_shards_path_or_url: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int,
        num_train_examples: int,
        per_gpu_batch_size: int,
        global_batch_size: int,
        num_workers: int,
        resolution: int = 256,
        center_crop: bool = True,
        random_flip: bool = False,
        shuffle_buffer_size: int = 1000,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        is_pre_encoded: bool = False,
        vae_checkpoint: Optional[str] = None,
        text_encoder_checkpoint: Optional[str] = None,
        use_filtered_dataset: bool = False,
        use_m4_laion_text_2_image_dataset: bool = False,
    ):
        num_batches = math.ceil(num_train_examples / global_batch_size)
        num_worker_batches = math.ceil(num_train_examples / (global_batch_size * num_workers))  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size

        transform = ImageNetTransform(resolution, center_crop, random_flip)

        def tokenize(text):
            text = replace_person_token(text)
            input_ids = tokenizer(
                text, max_length=max_seq_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            return input_ids[0]

        if not isinstance(train_shards_path_or_url, str):
            train_shards_path_or_url = [list(braceexpand(urls)) for urls in train_shards_path_or_url]
            # flatten list using itertools
            train_shards_path_or_url = list(itertools.chain.from_iterable(train_shards_path_or_url))

        if not isinstance(eval_shards_path_or_url, str):
            eval_shards_path_or_url = [list(braceexpand(urls)) for urls in eval_shards_path_or_url]
            # flatten list using itertools
            eval_shards_path_or_url = list(itertools.chain.from_iterable(eval_shards_path_or_url))

        if not use_m4_laion_text_2_image_dataset:
            if not is_pre_encoded:
                processing_pipeline = [
                    wds.decode("pil", handler=wds.ignore_and_continue),
                    wds.rename(image="jpg;png;jpeg;webp", input_ids="text;txt;caption", handler=wds.warn_and_continue),
                    wds.map(filter_keys(set(["image", "input_ids"]))),
                    wds.map_dict(image=transform.train_transform, input_ids=tokenize),
                    wds.to_tuple("image", "input_ids"),
                ]
            else:
                # lowercase and replace / with .
                vae_checkpoint = vae_checkpoint.lower().replace("/", ".")
                text_encoder_checkpoint = text_encoder_checkpoint.lower().replace("/", ".")
                processing_pipeline = [
                    wds.decode(
                        wds.handle_extension("pth", wds.autodecode.torch_loads), handler=wds.ignore_and_continue
                    ),
                    wds.rename(
                        input_ids=f"{vae_checkpoint}.pth",
                        encoder_hidden_states=f"{text_encoder_checkpoint}.pth",
                        handler=wds.warn_and_continue,
                    ),
                    wds.map(filter_keys(set(["input_ids", "encoder_hidden_states"]))),
                    wds.to_tuple("input_ids", "encoder_hidden_states"),
                ]

            # Create train dataset and loader
            pipeline = [
                wds.ResampledShards(train_shards_path_or_url),
                tarfile_to_samples_nothrow,
                wds.select(
                    WebdatasetFilter(min_size=256, max_pwatermark=0.5, aesthetic_threshold=4.9)
                    if use_filtered_dataset
                    else lambda x: True
                ),
                wds.shuffle(shuffle_buffer_size),
                *processing_pipeline,
                wds.batched(per_gpu_batch_size, partial=False, collation_fn=default_collate),
            ]

            # each worker is iterating over this
            train_dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)

            eval_dataset_pipeline = [
                wds.SimpleShardList(eval_shards_path_or_url),
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=wds.ignore_and_continue),
                *processing_pipeline,
                wds.batched(per_gpu_batch_size, partial=False, collation_fn=default_collate),
            ]
            eval_dataset = wds.DataPipeline(*eval_dataset_pipeline)
        else:
            train_shards_path_or_url = wds.ResampledShards(train_shards_path_or_url)

            train_dataset = M4LaionDatasetStream(train_shards_path_or_url)

            if use_filtered_dataset:
                train_dataset = M4LaionDatasetFilter(train_dataset, min_size=256)

            train_dataset = M4LaionDatasetShuffle(train_dataset, shuffle_buffer_size)
            train_dataset = M4LaionDatasetProcessingPipeline(train_dataset, transform.train_transform, tokenize)
            train_dataset = M4LaionDatasetBatched(
                train_dataset, per_gpu_batch_size, partial=False, collation_fn=default_collate
            )
            train_dataset = M4LaionDatasetWithEpoch(train_dataset, num_worker_batches)

            eval_dataset = M4LaionDatasetStream(eval_shards_path_or_url)
            eval_dataset = wds.split_by_worker(eval_dataset)
            eval_dataset = M4LaionDatasetProcessingPipeline(eval_dataset, transform.train_transform, tokenize)
            eval_dataset = M4LaionDatasetBatched(
                eval_dataset, per_gpu_batch_size, partial=False, collation_fn=default_collate
            )

        self._train_dataset = train_dataset
        self._train_dataloader = DataLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        # add meta-data to dataloader instance for convenience
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

        # Create eval dataset and loader
        self._eval_dataset = eval_dataset
        self._eval_dataloader = DataLoader(
            self._eval_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader

    @property
    def eval_dataset(self):
        return self._eval_dataset

    @property
    def eval_dataloader(self):
        return self._eval_dataloader


def write_m4_laion_shard_urls():
    s3 = s3fs.S3FileSystem()

    split_urls = braceexpand("s3://m4-datasets/LAION_data/laion_dataset_filtered_dedup/{0..199}")

    shard_urls = []

    for split_url in split_urls:
        for shard_url in s3.ls(split_url):
            shard_urls.append(shard_url)

    shard_urls = "\n".join(shard_urls)

    with open("./configs/m4_laion_shard_urls.txt", "w") as shard_urls_file:
        shard_urls_file.write(shard_urls)


def load_m4_laion_shard_urls():
    with open("./configs/m4_laion_shard_urls.txt", "r") as shard_urls_file:
        for filename in shard_urls_file.readlines():
            yield filename.strip()


class M4LaionDatasetStream(IterableDataset):
    def __init__(self, shard_urls):
        self.shard_urls = shard_urls

    def __iter__(self):
        # s3 handle is not fork safe, just create a new handle when the stream is created
        s3 = s3fs.S3FileSystem()

        for shard_url in self.shard_urls:
            # HACK dealing with wds.ResampledShards wrapping result in a dict
            if isinstance(shard_url, dict):
                shard_url = shard_url["url"]

            with s3.open(shard_url, "rb") as f:
                in_memory_stream = pa.input_stream(f)
                try:
                    opened_stream = pa.ipc.open_stream(in_memory_stream)
                except pa.lib.ArrowInvalid as e:
                    logger.warning(str(e))
                    continue
                pa_table = opened_stream.read_all()

            table = pa_table.to_pydict()

            for i in range(len(table["text"])):
                image_bytes = table["image"][i]["bytes"]
                image_bytes = io.BytesIO(image_bytes)
                try:
                    image = Image.open(image_bytes)
                    image = image.convert("RGB")
                except Exception as e:
                    logger.warning(str(e))
                    continue

                text = table["text"][i]

                meta = table["meta"][i]

                yield {"image": image, "text": text, "meta": meta}


class M4LaionDatasetFilter(IterableDataset):
    def __init__(self, iterable, min_size=256):
        self.iterable = iterable
        self.min_size = min_size

    def __iter__(self):
        for sample in self.iterable:
            original_width = sample["meta"].get("original_width", 0.0) or 0.0
            original_height = sample["meta"].get("original_height", 0.0) or 0.0

            filter_size = original_width >= self.min_size and original_height >= self.min_size

            if filter_size:
                yield sample


class M4LaionDatasetShuffle(IterableDataset):
    def __init__(self, iterable, bufsize=1000, initial=100):
        self.iterable = iterable
        self.bufsize = bufsize
        self.initial = initial

    def __iter__(self):
        data = iter(self.iterable)

        rng = random.Random(int((os.getpid() + time.time()) * 1e9))
        initial = min(self.initial, self.bufsize)
        buf = []
        for sample in data:
            buf.append(sample)
            if len(buf) < self.bufsize:
                try:
                    buf.append(next(data))  # skipcq: PYL-R1708
                except StopIteration:
                    pass
            if len(buf) >= initial:
                yield pick_random(buf, rng)
        while len(buf) > 0:
            yield pick_random(buf, rng)


class M4LaionDatasetProcessingPipeline(IterableDataset):
    def __init__(self, iterable, transform, tokenize):
        self.iterable = iterable
        self.transform = transform
        self.tokenize = tokenize

    def __iter__(self):
        for sample in self.iterable:
            image = sample["image"]
            text = sample["text"]

            image = self.transform(image)
            text = self.tokenize(text)

            yield image, text


class M4LaionDatasetBatched(IterableDataset):
    def __init__(self, iterable, batchsize=20, collation_fn=default_collate, partial=True):
        self.iterable = iterable
        self.batchsize = batchsize
        self.collation_fn = collation_fn
        self.partial = partial

    def __iter__(self):
        batch = []
        for sample in self.iterable:
            if len(batch) >= self.batchsize:
                if self.collation_fn is not None:
                    batch = self.collation_fn(batch)
                yield batch
                batch = []
            batch.append(sample)
        if len(batch) == 0:
            return
        elif len(batch) == self.batchsize or self.partial:
            if self.collation_fn is not None:
                batch = self.collation_fn(batch)
            yield batch


class M4LaionDatasetWithEpoch(IterableDataset):
    def __init__(self, iterable, num_epochs):
        self.iterable = iterable
        self.num_epochs = num_epochs

    def __iter__(self):
        for _ in range(self.num_epochs):
            for sample in self.iterable:
                yield sample


def pick_random(buf, rng):
    k = rng.randint(0, len(buf) - 1)
    sample = buf[k]
    buf[k] = buf[-1]
    buf.pop()
    return sample
