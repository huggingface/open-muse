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

import itertools
import json
import math
import random
import re
from typing import List, Union

import webdataset as wds
from braceexpand import braceexpand
from torch.utils.data import default_collate
from torchvision import transforms
from transformers import PreTrainedTokenizer

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
    ):
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

        # Create train dataset and loader
        pipeline = [
            wds.ResampledShards(train_shards_path_or_url),
            wds.tarfile_to_samples(handler=wds.ignore_and_continue),
            wds.shuffle(shuffle_buffer_size),
            wds.decode("pil", handler=wds.ignore_and_continue),
            wds.rename(image="jpg;png;jpeg;webp", input_ids="text;txt;caption", handler=wds.warn_and_continue),
            wds.map(filter_keys(set(["image", "input_ids"]))),
            wds.map_dict(image=transform.train_transform, input_ids=tokenize),
            wds.to_tuple("image", "input_ids"),
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
            wds.rename(image="jpg;png;jpeg;webp", input_ids="text;txt;caption", handler=wds.warn_and_continue),
            wds.map(filter_keys(set(["image", "input_ids"]))),
            wds.map_dict(image=transform.train_transform, input_ids=tokenize),
            wds.to_tuple("image", "input_ids"),
            wds.batched(per_gpu_batch_size, partial=False, collation_fn=default_collate),
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
