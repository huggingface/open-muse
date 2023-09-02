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
from functools import partial
from typing import List, Optional, Union

import webdataset as wds
import yaml
from braceexpand import braceexpand
from torch.utils.data import default_collate
from torchvision import transforms
from transformers import PreTrainedTokenizer
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)

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


def get_orig_size(json):
    return (int(json.get("original_width", 0.0)), int(json.get("original_height", 0.0)))


def get_aesthetic_score(json):
    if "aesthetic" in json:
        a = json["aesthetic"]
    elif "AESTHETIC_SCORE" in json:
        a = json["AESTHETIC_SCORE"]
    elif "aesthetic_score_laion_v2" in json:
        a = json["aesthetic_score_laion_v2"]
    elif "stability_metadata" in json and "aes_scorelv2" in json["stability_metadata"]:
        a = json["stability_metadata"]["aes_scorelv2"]
    else:
        a = 0.0

    a = float(a)

    return a


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


def image_transform(example, resolution=256):
    image = example["image"]
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)(image)
    # get crop coordinates
    c_top, c_left, _, _ = transforms.RandomCrop.get_params(image, output_size=(resolution, resolution))
    image = transforms.functional.crop(image, c_top, c_left, resolution, resolution)
    image = transforms.ToTensor()(image)
    example["image"] = image
    example["crop_coords"] = (c_top, c_left)
    return example


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


class WebdatasetSelect:
    def __init__(
        self,
        min_size=256,
        max_pwatermark=0.5,
        min_aesthetic_score=4.9,
        require_marked_as_ok_by_spawning=False,
        require_marked_as_not_getty=False,
        max_pnsfw=None,
    ):
        self.min_size = min_size
        self.max_pwatermark = max_pwatermark
        self.min_aesthetic_score = min_aesthetic_score
        self.require_marked_as_ok_by_spawning = require_marked_as_ok_by_spawning
        self.require_marked_as_not_getty = require_marked_as_not_getty
        self.max_pnsfw = max_pnsfw

    def __call__(self, x):
        if "json" not in x:
            return False
        try:
            x_json = json.loads(x["json"])
        except:
            return False

        # For all requirements, if the necessary key(s) are not present, we assume
        # the requirement does not hold. Note that many checks are done on different keys
        # which is due to different datasets being used with different metadata dicts.

        # size

        if "original_width" not in x_json or "original_height" not in x_json:
            return False

        original_width = x_json["original_width"]
        original_height = x_json["original_height"]

        is_less_than_min_size = original_width < self.min_size or original_height < self.min_size

        if is_less_than_min_size:
            return False

        # watermark

        if (
            "pwatermark" not in x_json
            and "watermark_score" not in x_json
            and ("stability_metadata" not in x_json or "p_watermarkdf" not in x_json["stability_metadata"])
        ):
            return False

        if "pwatermark" in x_json:
            is_watermarked = x_json["pwatermark"] > self.max_pwatermark

            if is_watermarked:
                return False

        if "watermark_score" in x_json:
            is_watermarked_coyo = x_json["watermark_score"] > self.max_pwatermark

            if is_watermarked_coyo:
                return False

        if "stability_metadata" in x_json and "p_watermarkdf" in x_json["stability_metadata"]:
            is_watermarked_stability_metadata = x_json["stability_metadata"]["p_watermarkdf"] > self.max_pwatermark

            if is_watermarked_stability_metadata:
                return False

        # aesthetic

        if (
            "aesthetic" not in x_json
            and "AESTHETIC_SCORE" not in x_json
            and "aesthetic_score_laion_v2" not in x_json
            and ("stability_metadata" not in x_json or "aes_scorelv2" not in x_json["stability_metadata"])
        ):
            return False

        if "aesthetic" in x_json:
            is_under_min_aesthetic_threshold = x_json["aesthetic"] < self.min_aesthetic_score

            if is_under_min_aesthetic_threshold:
                return False

        if "AESTHETIC_SCORE" in x_json:
            is_under_min_aesthetic_threshold_b = x_json["AESTHETIC_SCORE"] < self.min_aesthetic_score

            if is_under_min_aesthetic_threshold_b:
                return False

        if "aesthetic_score_laion_v2" in x_json:
            is_under_min_aesthetic_threshold_coyo = x_json["aesthetic_score_laion_v2"] < self.min_aesthetic_score

            if is_under_min_aesthetic_threshold_coyo:
                return False

        if "stability_metadata" in x_json and "aes_scorelv2" in x_json["stability_metadata"]:
            is_under_min_aesthetic_threshold_stability_metadata = (
                x_json["stability_metadata"]["aes_scorelv2"] < self.min_aesthetic_score
            )

            if is_under_min_aesthetic_threshold_stability_metadata:
                return False

        # spawning

        if self.require_marked_as_ok_by_spawning:
            if "stability_metadata" not in x_json or "is_spawning" not in x_json["stability_metadata"]:
                return False

            is_marked_as_not_ok_by_spawning = x_json["stability_metadata"]["is_spawning"]

            if is_marked_as_not_ok_by_spawning:
                return False

        # getty

        if self.require_marked_as_not_getty:
            if "stability_metadata" not in x_json or "is_getty" not in x_json["stability_metadata"]:
                return False

            is_marked_as_getty = x_json["stability_metadata"]["is_getty"]

            if is_marked_as_getty:
                return False

        # nsfw

        if self.max_pnsfw is not None:
            if "stability_metadata" not in x_json or "p_nsfwdf" not in x_json["stability_metadata"]:
                return False

            is_above_max_nsfw = x_json["stability_metadata"]["p_nsfwdf"] > self.max_pnsfw

            if is_above_max_nsfw:
                return False

        return True


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
        require_marked_as_ok_by_spawning: bool = False,
        require_marked_as_not_getty: bool = False,
        max_pnsfw: Optional[float] = None,
    ):
        yaml_serialized_shard_paths = [
            "m4_shards",
            "laion-aesthetic-475-max-1024-joined-with-stability-metadata-laicov2_shards",
        ]
        if train_shards_path_or_url in yaml_serialized_shard_paths:
            with open(f"./configs/{train_shards_path_or_url}.yaml") as f:
                train_shards_path_or_url = yaml.safe_load(f)

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

        if not is_pre_encoded:
            processing_pipeline = [
                wds.decode("pil", handler=wds.ignore_and_continue),
                wds.rename(
                    image="jpg;png;jpeg;webp",
                    input_ids="text;txt;caption",
                    orig_size="json",
                    aesthetic_score="json",
                    handler=wds.warn_and_continue,
                ),
                wds.map(filter_keys(set(["image", "input_ids", "orig_size", "aesthetic_score"]))),
                wds.map(partial(image_transform, resolution=resolution), handler=wds.warn_and_continue),
                wds.map_dict(
                    input_ids=tokenize,
                    orig_size=get_orig_size,
                    aesthetic_score=get_aesthetic_score,
                    handler=wds.warn_and_continue,
                ),
            ]
        else:
            # lowercase and replace / with .
            vae_checkpoint = vae_checkpoint.lower().replace("/", ".")
            text_encoder_checkpoint = text_encoder_checkpoint.lower().replace("/", ".")
            processing_pipeline = [
                wds.decode(wds.handle_extension("pth", wds.autodecode.torch_loads), handler=wds.ignore_and_continue),
                wds.rename(
                    image_input_ids=f"{vae_checkpoint}.pth",
                    encoder_hidden_states=f"{text_encoder_checkpoint}.pth",
                    handler=wds.warn_and_continue,
                ),
                wds.map(filter_keys(set(["image_input_ids", "encoder_hidden_states"]))),
            ]

        if use_filtered_dataset:
            select = wds.select(
                WebdatasetSelect(
                    require_marked_as_ok_by_spawning=require_marked_as_ok_by_spawning,
                    require_marked_as_not_getty=require_marked_as_not_getty,
                    max_pnsfw=max_pnsfw,
                )
            )
        else:
            select = None

        # Create train dataset and loader
        pipeline = [
            wds.ResampledShards(train_shards_path_or_url),
            tarfile_to_samples_nothrow,
            *([select] if select is not None else []),
            wds.shuffle(shuffle_buffer_size),
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
            *processing_pipeline,
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
