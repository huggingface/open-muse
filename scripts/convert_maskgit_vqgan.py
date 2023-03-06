# !pip install ml_collections
# !wget https://storage.googleapis.com/maskgit-public/checkpoints/tokenizer_imagenet256_checkpoint
import logging

import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict
from maskgit.utils import restore_from_path

from muse import MaskGitVQGAN

logger = logging.get_logger(__name__)


def rename_flax_dict(params):
    keys = list(params.keys())

    for key in keys:
        new_key = ".".join(key)
        params[new_key] = params.pop(key)
    keys = list(params.keys())

    block_map = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, 1),
        4: (2, 0),
        5: (2, 1),
        6: (3, 0),
        7: (3, 1),
        8: (4, 0),
        9: (4, 1),
    }

    encoder_keys = [key for key in keys if "encoder.ResBlock" in key]
    for key in encoder_keys:
        if "ResBlock_10" in key:
            new_key = key.replace("ResBlock_10", "mid.0")
            new_key = new_key.replace("Conv_0", "conv1")
            new_key = new_key.replace("Conv_1", "conv2")
            new_key = new_key.replace("GroupNorm_0", "norm1")
            new_key = new_key.replace("GroupNorm_1", "norm2")
            params[new_key] = params.pop(key)
        elif "ResBlock_11" in key:
            new_key = key.replace("ResBlock_11", "mid.1")
            new_key = new_key.replace("Conv_0", "conv1")
            new_key = new_key.replace("Conv_1", "conv2")
            new_key = new_key.replace("GroupNorm_0", "norm1")
            new_key = new_key.replace("GroupNorm_1", "norm2")
            params[new_key] = params.pop(key)
    keys = list(params.keys())

    encoder_keys = [key for key in keys if "encoder.ResBlock" in key]
    for key in encoder_keys:
        name = key.split(".")[1]
        res_name, idx = name.split("_")
        idx1, idx2 = block_map[int(idx)]
        new_key = key.replace(name, f"down.{idx1}.block.{idx2}")
        new_key = new_key.replace("Conv_0", "conv1")
        new_key = new_key.replace("Conv_1", "conv2")
        new_key = new_key.replace("Conv_2", "nin_shortcut")
        new_key = new_key.replace("GroupNorm_0", "norm1")
        new_key = new_key.replace("GroupNorm_1", "norm2")
        params[new_key] = params.pop(key)
    keys = list(params.keys())

    decoder_keys = [key for key in keys if "decoder.ResBlock" in key]
    for key in decoder_keys:
        if "ResBlock_0" in key:
            new_key = key.replace("ResBlock_0", "mid.0")
            new_key = new_key.replace("Conv_0", "conv1")
            new_key = new_key.replace("Conv_1", "conv2")
            new_key = new_key.replace("GroupNorm_0", "norm1")
            new_key = new_key.replace("GroupNorm_1", "norm2")
            params[new_key] = params.pop(key)
        elif "ResBlock_1." in key:
            new_key = key.replace("ResBlock_1", "mid.1")
            new_key = new_key.replace("Conv_0", "conv1")
            new_key = new_key.replace("Conv_1", "conv2")
            new_key = new_key.replace("GroupNorm_0", "norm1")
            new_key = new_key.replace("GroupNorm_1", "norm2")
            params[new_key] = params.pop(key)
    keys = list(params.keys())

    decoder_keys = [key for key in keys if "decoder.ResBlock" in key]
    for key in decoder_keys:
        name = key.split(".")[1]
        res_name, idx = name.split("_")
        idx = int(idx) - 2
        idx1, idx2 = block_map[int(idx)]
        new_key = key.replace(name, f"up.{idx1}.block.{idx2}")
        new_key = new_key.replace("Conv_0", "conv1")
        new_key = new_key.replace("Conv_1", "conv2")
        new_key = new_key.replace("Conv_2", "nin_shortcut")
        new_key = new_key.replace("GroupNorm_0", "norm1")
        new_key = new_key.replace("GroupNorm_1", "norm2")
        params[new_key] = params.pop(key)
    keys = list(params.keys())

    for i in range(1, 5):
        w = f"decoder.Conv_{i}.kernel"
        b = f"decoder.Conv_{i}.bias"
        new_w = f"decoder.up.{i}.upsample_conv.kernel"
        new_b = f"decoder.up.{i}.upsample_conv.bias"
        params[new_w] = params.pop(w)
        params[new_b] = params.pop(b)
    keys = list(params.keys())

    for key in keys:
        if "Conv_" in key:
            new_key = key.replace("Conv_0", "conv_in")
            new_key = new_key.replace("Conv_5", "conv_out")
            new_key = new_key.replace("Conv_1", "conv_out")
            params[new_key] = params.pop(key)
        elif "GroupNorm" in key:
            new_key = key.replace("GroupNorm_0", "norm_out")
            params[new_key] = params.pop(key)
    params["quantize.embedding.embedding"] = params.pop("quantizer.codebook")

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
        flax_key_tuple = flax_key.split(".")

        # rename flax weights to PyTorch format
        if flax_key_tuple[-1] == "kernel" and flax_tensor.ndim == 4 and ".".join(flax_key_tuple) not in pt_model_dict:
            # conv layer
            flax_key_tuple = flax_key_tuple[:-1] + ("weight",)
            flax_tensor = jnp.transpose(flax_tensor, (3, 2, 0, 1))
        elif flax_key_tuple[-1] == "kernel" and ".".join(flax_key_tuple) not in pt_model_dict:
            # linear layer
            flax_key_tuple = flax_key_tuple[:-1] + ("weight",)
            flax_tensor = flax_tensor.T
        elif flax_key_tuple[-1] in ["scale", "embedding"]:
            flax_key_tuple = flax_key_tuple[:-1] + ("weight",)

        flax_key = ".".join(flax_key_tuple)

        if "in_proj.weight" in flax_key:
            flax_key = flax_key.replace("in_proj.weight", "in_proj_weight")

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

    pt_model = MaskGitVQGAN()

    pt_model = load_flax_weights_in_pytorch_model(pt_model, params)
    pt_model.save_pretrained(pytorch_dump_folder_path)
