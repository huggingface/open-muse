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


# !pip install ml_collections
# !wget https://storage.googleapis.com/maskgit-public/checkpoints/tokenizer_imagenet256_checkpoint
import logging

import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict
import argparse

from muse import MaskGitTransformer
import tensorflow as tf
import flax

def restore_from_path(path):
  with tf.io.gfile.GFile(path, "rb") as f:
    state = flax.serialization.from_bytes(None, f.read())
  return state

logger = logging.getLogger(__name__)


def rename_flax_dict(params):
    keys = list(params.keys())

    for key in keys:
        new_key = ".".join(key)
        params[new_key] = params.pop(key)
    keys = list(params.keys())

    for key in keys:
        new_key = key.replace("Embed_0", "embed")
        for i in range(24):
            new_key = new_key.replace(f"TransformerLayer_{i}.", f"transformer_layers.{i}.")
        new_key = new_key.replace("Attention_0.attention_output_ln", "attn_layer_norm")
        new_key = new_key.replace("Attention_0.self_attention", "attention")
        new_key = new_key.replace("MlmLayer_0", "mlm_layer")
        new_key = new_key.replace("mlm_bias", "to_logits")
        new_key = new_key.replace("Mlp_0", "ffn")
        new_key = new_key.replace("layer_output_ln", "layer_norm")
        new_key = new_key.replace("scale", "weight")
        new_key = new_key.replace("embeddings.embedding", "embeddings.weight")
        new_key = new_key.replace("kernel", "weight")
        params[new_key] = params.pop(key)
    params['mlm_layer.to_logits.weight'] = params['embed.word_embeddings.weight']
    return params


def load_flax_weights_in_pytorch_model(pt_model, flax_state):
    """Load flax checkpoints in a PyTorch model"""

    try:
        import torch  # noqa: F401
    except ImportError:
        logger.error(
            "Loading a Flax weights in PyTorch, requires both PyTorch and Flax to be installed. Please see"
            " https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for installation"
            " instructions."
        )
        raise

    pt_model_dict = pt_model.state_dict()

    # keep track of unexpected & missing keys
    unexpected_keys = []
    missing_keys = set(pt_model_dict.keys())

    for flax_key, flax_tensor in flax_state.items():
        flax_key_tuple = tuple(flax_key.split("."))
        flax_tensor = torch.from_numpy(np.array(flax_tensor))
        if flax_key_tuple[0] == "transformer_layers" and len(flax_tensor.shape) == 3:
            if flax_tensor.shape[0] == 768:
                flax_tensor = flax_tensor.permute(1, 2, 0)
            elif flax_tensor.shape[-1] == 768:
                flax_tensor = flax_tensor.permute(2, 0, 1)
        if flax_key_tuple[0] == "transformer_layers" and len(flax_tensor.shape) == 2:
            if not 'bias' == flax_key_tuple[-1]:
                flax_tensor = flax_tensor.permute(1, 0)
        if flax_key_tuple[0] == "mlm_layer" and len(flax_tensor.shape) == 2:
            if not flax_key_tuple[1] == "to_logits":
                flax_tensor = flax_tensor.permute(1, 0)

        if flax_key_tuple[0] == "transformer_layers" and 'bias' == flax_key_tuple[-1] and len(flax_tensor.shape) == 2:
            flax_tensor = flax_tensor.reshape(-1)
        if flax_key_tuple[0] == "transformer_layers" and 'weight' == flax_key_tuple[-1] and len(flax_tensor.shape) == 3:
            if flax_tensor.shape[0] == 768:
                flax_tensor = flax_tensor.reshape(flax_tensor.shape[0], -1)
            elif flax_tensor.shape[-1] == 768:
                flax_tensor = flax_tensor.reshape(-1, flax_tensor.shape[-1])

        flax_key = ".".join(flax_key_tuple)
        if flax_key in pt_model_dict:
            if flax_tensor.shape != pt_model_dict[flax_key].shape:
                raise ValueError(
                    f"Flax checkpoint seems to be incorrect. Weight {flax_key_tuple} was expected "
                    f"to be of shape {pt_model_dict[flax_key].shape}, but is {flax_tensor.shape}."
                )
            else:
                # add weight to pytorch dict
                flax_tensor = np.asarray(flax_tensor) if not isinstance(flax_tensor, np.ndarray) else flax_tensor
                pt_model_dict[flax_key] = torch.from_numpy(flax_tensor)
                # remove from missing keys
                missing_keys.remove(flax_key)
        else:
            # weight is not expected by PyTorch model
            unexpected_keys.append(flax_key)

    pt_model.load_state_dict(pt_model_dict)

    # re-transform missing_keys to list
    missing_keys = list(missing_keys)

    if len(unexpected_keys) > 0:
        logger.warning(
            "Some weights of the Flax model were not used when initializing the PyTorch model"
            f" {pt_model.__class__.__name__}: {unexpected_keys}."
        )
    else:
        logger.warning(f"All Flax model weights were used when initializing {pt_model.__class__.__name__}.\n")
    if len(missing_keys) > 0:
        logger.warning(
            f"Some weights of {pt_model.__class__.__name__} were not initialized from the Flax model and are newly"
            f" initialized: {missing_keys}\nYou should probably TRAIN this model on a down-stream task to be able to"
            " use it for predictions and inference."
        )
    else:
        logger.warning(f"All the weights of {pt_model.__class__.__name__} were initialized from the Flax model.")

    return pt_model


def convert(flax_model_path, pytorch_dump_folder_path):
    params = restore_from_path(flax_model_path)["params"]
    params = flatten_dict(params)
    params = rename_flax_dict(params)

    pt_model = MaskGitTransformer(
        vocab_size=2025, #(1024 + 1000 + 1 = 2025 -> Vq_tokens + Imagenet class ids + <mask>)
        max_position_embeddings=257, # 256 + 1 for class token
        hidden_size=768,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=3072,
        codebook_size=1024,
        num_vq_tokens=256,
        num_classes=1000,
        layer_norm_eps=1e-12,
        layer_norm_embeddings=True,
        use_bias=True,
        use_encoder_layernorm=False,
        use_maskgit_mlp=True,
        use_normformer=False
    )

    pt_model = load_flax_weights_in_pytorch_model(pt_model, params)
    pt_model.save_pretrained(pytorch_dump_folder_path)
    return pt_model

def parse_args():
    parser = argparse.ArgumentParser(description="Simple loading script for maskgit.")
    parser.add_argument(
        "--flax_model_path",
        type=str,
        default=None,
        required=True,
        help="Path to flax maskgit transformer.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        type=str,
        default=None,
        required=True,
        help="Path to dump pytorch model.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    convert(args.flax_model_path, args.pytorch_dump_folder_path)