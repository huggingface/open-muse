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

import functools
import inspect
import json
import os
from collections import OrderedDict
from functools import partial
from pathlib import PosixPath
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import accelerate
import numpy as np
import torch
from accelerate.utils import set_module_tensor_to_device
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from requests import HTTPError
from torch import Tensor, device

from . import __version__, logging

logger = logging.get_logger(__name__)


hf_cache_home = os.path.expanduser(
    os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
)
default_cache_path = os.path.join(hf_cache_home, "muse")


CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
SAFETENSORS_WEIGHTS_NAME = "pytorch_model.safetensors"
HUGGINGFACE_CO_RESOLVE_ENDPOINT = "https://huggingface.co"
MUSE_CACHE = default_cache_path
MUSE_DYNAMIC_MODULE_NAME = "myse_modules"
HF_MODULES_CACHE = os.getenv("HF_MODULES_CACHE", os.path.join(hf_cache_home, "modules"))


_LOW_CPU_MEM_USAGE_DEFAULT = True


def get_parameter_device(parameter: torch.nn.Module):
    try:
        return next(parameter.parameters()).device
    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device


def get_parameter_dtype(parameter: torch.nn.Module):
    try:
        return next(parameter.parameters()).dtype
    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype


def load_state_dict(checkpoint_file: Union[str, os.PathLike]):
    """
    Reads a checkpoint file, returning properly formatted errors if they arise.
    """
    try:
        if os.path.basename(checkpoint_file) == WEIGHTS_NAME:
            return torch.load(checkpoint_file, map_location="cpu")
    except Exception as e:
        try:
            with open(checkpoint_file) as f:
                if f.read().startswith("version"):
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please install "
                        "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                        "you cloned."
                    )
                else:
                    raise ValueError(
                        f"Unable to locate the file {checkpoint_file} which is necessary to load this pretrained "
                        "model. Make sure you have saved the model properly."
                    ) from e
        except (UnicodeDecodeError, ValueError):
            raise OSError(
                f"Unable to load weights from checkpoint file for '{checkpoint_file}' "
                f"at '{checkpoint_file}'. "
                "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True."
            )


def _load_state_dict_into_model(model_to_load, state_dict):
    # Convert old format to new format if needed from a PyTorch state_dict
    # copy state_dict so _load_from_state_dict can modify it
    state_dict = state_dict.copy()
    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: torch.nn.Module, prefix=""):
        args = (state_dict, prefix, {}, True, [], [], error_msgs)
        module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model_to_load)

    return error_msgs


def _get_model_file(
    pretrained_model_name_or_path,
    *,
    weights_name,
    subfolder,
    cache_dir,
    force_download,
    proxies,
    resume_download,
    local_files_only,
    use_auth_token,
    user_agent,
    revision,
):
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    if os.path.isfile(pretrained_model_name_or_path):
        return pretrained_model_name_or_path
    elif os.path.isdir(pretrained_model_name_or_path):
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, weights_name)):
            # Load from a PyTorch checkpoint
            model_file = os.path.join(pretrained_model_name_or_path, weights_name)
            return model_file
        elif subfolder is not None and os.path.isfile(
            os.path.join(pretrained_model_name_or_path, subfolder, weights_name)
        ):
            model_file = os.path.join(pretrained_model_name_or_path, subfolder, weights_name)
            return model_file
        else:
            raise EnvironmentError(
                f"Error no file named {weights_name} found in directory {pretrained_model_name_or_path}."
            )
    else:
        try:
            # Load from URL or cache if already cached
            model_file = hf_hub_download(
                pretrained_model_name_or_path,
                filename=weights_name,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                user_agent=user_agent,
                subfolder=subfolder,
                revision=revision,
            )
            return model_file

        except RepositoryNotFoundError:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier "
                "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a "
                "token having permission to this repo with `use_auth_token` or log in with `huggingface-cli "
                "login`."
            )
        except RevisionNotFoundError:
            raise EnvironmentError(
                f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for "
                "this model name. Check the model page at "
                f"'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
            )
        except EntryNotFoundError:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} does not appear to have a file named {weights_name}."
            )
        except HTTPError as err:
            raise EnvironmentError(
                f"There was a specific connection error when trying to load {pretrained_model_name_or_path}:\n{err}"
            )
        except ValueError:
            raise EnvironmentError(
                f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it"
                f" in the cached files and it looks like {pretrained_model_name_or_path} is not the path to a"
                f" directory containing a file named {weights_name} or"
                " \nCheckout your internet connection or see how to run the library in"
                " offline mode at 'https://huggingface.co/docs/diffusers/installation#offline-mode'."
            )
        except EnvironmentError:
            raise EnvironmentError(
                f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                f"containing a file named {weights_name}"
            )


class ModelMixin(torch.nn.Module):
    r"""
    Base class for all models.

    [`ModelMixin`] takes care of storing the configuration of the models and handles methods for loading, downloading
    and saving models.

        - **config_name** ([`str`]) -- A filename under which the model should be stored when calling
          [`~models.ModelMixin.save_pretrained`].
    """
    config_name = CONFIG_NAME
    _automatically_saved_args = ["_version", "_class_name", "_name_or_path"]
    _supports_gradient_checkpointing = False

    def __init__(self):
        super().__init__()

    @property
    def is_gradient_checkpointing(self) -> bool:
        """
        Whether gradient checkpointing is activated for this model or not.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        return any(hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing for m in self.modules())

    def enable_gradient_checkpointing(self):
        """
        Activates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if not self._supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")
        self.apply(partial(self._set_gradient_checkpointing, value=True))

    def disable_gradient_checkpointing(self):
        """
        Deactivates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if self._supports_gradient_checkpointing:
            self.apply(partial(self._set_gradient_checkpointing, value=False))

    def set_use_memory_efficient_attention_xformers(
        self, valid: bool, attention_op: Optional[Callable] = None
    ) -> None:
        # Recursively walk through all the children.
        # Any children which exposes the set_use_memory_efficient_attention_xformers method
        # gets the message
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid, attention_op)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        for module in self.children():
            if isinstance(module, torch.nn.Module):
                fn_recursive_set_mem_eff(module)

    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None):
        r"""
        Enable memory efficient attention as implemented in xformers.

        When this option is enabled, you should observe lower GPU memory usage and a potential speed up at inference
        time. Speed up at training time is not guaranteed.

        Warning: When Memory Efficient Attention and Sliced attention are both enabled, the Memory Efficient Attention
        is used.

        Parameters:
            attention_op (`Callable`, *optional*):
                Override the default `None` operator for use as `op` argument to the
                [`memory_efficient_attention()`](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention)
                function of xFormers.

        Examples:

        ```py
        >>> import torch
        >>> from diffusers import UNet2DConditionModel
        >>> from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

        >>> model = UNet2DConditionModel.from_pretrained(
        ...     "stabilityai/stable-diffusion-2-1", subfolder="unet", torch_dtype=torch.float16
        ... )
        >>> model = model.to("cuda")
        >>> model.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        ```
        """
        self.set_use_memory_efficient_attention_xformers(True, attention_op)

    def disable_xformers_memory_efficient_attention(self):
        r"""
        Disable memory efficient attention as implemented in xformers.
        """
        self.set_use_memory_efficient_attention_xformers(False)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Callable = None,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `[`~models.ModelMixin.from_pretrained`]` class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace `torch.save` by another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
            state_dict (`Dict[str, torch.Tensor]`, *optional*):
                The state dictionary to save. If `None`, the model's state dictionary will be saved.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if save_function is None:
            save_function = torch.save

        os.makedirs(save_directory, exist_ok=True)

        model_to_save = self

        # Attach architecture to the config
        # Save the config
        if is_main_process:
            model_to_save.save_config(save_directory)

        # Save the model
        if state_dict is None:
            state_dict = model_to_save.state_dict()

        weights_name = WEIGHTS_NAME

        # Save the model
        save_function(state_dict, os.path.join(save_directory, weights_name))

        logger.info(f"Model weights saved in {os.path.join(save_directory, weights_name)}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        r"""
        Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you should first set it back in training mode with `model.train()`.

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids should have an organization name, like `google/ddpm-celebahq-256`.
                    - A path to a *directory* containing model weights saved using [`~ModelMixin.save_config`], e.g.,
                      `./my_model_directory/`.

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model under this dtype. If `"auto"` is passed the dtype
                will be automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `diffusers-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            from_flax (`bool`, *optional*, defaults to `False`):
                Load the model weights from a Flax checkpoint save file.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo (either remote in
                huggingface.co or downloaded locally), you can specify the folder name here.

            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be refined to each
                parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
                same device.

                To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading by not initializing the weights and only loading the pre-trained weights. This
                also tries to not use more than 1x model size in CPU memory (including peak memory) while loading the
                model. This is only supported when torch version >= 1.9.0. If you are using an older version of torch,
                setting this argument to `True` will raise an error.

        <Tip>

         It is required to be logged in (`huggingface-cli login`) when you want to use private or [gated
         models](https://huggingface.co/docs/hub/models-gated#gated-models).

        </Tip>

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use
        this method in a firewalled environment.

        </Tip>

        """
        cache_dir = kwargs.pop("cache_dir", MUSE_CACHE)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)  # TODO
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        subfolder = kwargs.pop("subfolder", None)
        device_map = kwargs.pop("device_map", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)

        if low_cpu_mem_usage is False and device_map is not None:
            raise ValueError(
                f"You cannot set `low_cpu_mem_usage` to `False` while using device_map={device_map} for loading and"
                " dispatching. Please make sure to set `low_cpu_mem_usage=True`."
            )

        user_agent = {
            "diffusers": __version__,
            "file_type": "model",
            "framework": "pytorch",
        }

        # Load config if we don't provide a configuration
        config_path = pretrained_model_name_or_path

        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # Load model

        model_file = None

        if model_file is None:
            model_file = _get_model_file(
                pretrained_model_name_or_path,
                weights_name=WEIGHTS_NAME,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )

        if low_cpu_mem_usage:
            # Instantiate model with empty weights
            with accelerate.init_empty_weights():
                config, unused_kwargs = cls.load_config(
                    config_path,
                    cache_dir=cache_dir,
                    return_unused_kwargs=True,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    subfolder=subfolder,
                    device_map=device_map,
                    **kwargs,
                )
                model = cls.from_config(config, **unused_kwargs)

            # if device_map is None, load the state dict and move the params from meta device to the cpu
            if device_map is None:
                param_device = "cpu"
                state_dict = load_state_dict(model_file)
                # move the params from meta device to cpu
                missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
                if len(missing_keys) > 0:
                    raise ValueError(
                        f"Cannot load {cls} from {pretrained_model_name_or_path} because the following keys are"
                        f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"
                        " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomely initialize"
                        " those weights or else make sure your checkpoint file is correct."
                    )

                for param_name, param in state_dict.items():
                    accepts_dtype = "dtype" in set(inspect.signature(set_module_tensor_to_device).parameters.keys())
                    if accepts_dtype:
                        set_module_tensor_to_device(model, param_name, param_device, value=param, dtype=torch_dtype)
                    else:
                        set_module_tensor_to_device(model, param_name, param_device, value=param)
            else:  # else let accelerate handle loading and dispatching.
                # Load weights and dispatch according to the device_map
                # by deafult the device_map is None and the weights are loaded on the CPU
                accelerate.load_checkpoint_and_dispatch(model, model_file, device_map, dtype=torch_dtype)

            loading_info = {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }
        else:
            config, unused_kwargs = cls.load_config(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                subfolder=subfolder,
                device_map=device_map,
                **kwargs,
            )
            model = cls.from_config(config, **unused_kwargs)

            state_dict = load_state_dict(model_file)

            model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
                model,
                state_dict,
                model_file,
                pretrained_model_name_or_path,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
            )

            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
                "error_msgs": error_msgs,
            }

        if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
            raise ValueError(
                f"{torch_dtype} needs to be of type `torch.dtype`, e.g. `torch.float16`, but is {type(torch_dtype)}."
            )
        elif torch_dtype is not None:
            model = model.to(torch_dtype)

        model.register_to_config(_name_or_path=pretrained_model_name_or_path)

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()
        if output_loading_info:
            return model, loading_info

        return model

    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        state_dict,
        resolved_archive_file,
        pretrained_model_name_or_path,
        ignore_mismatched_sizes=False,
    ):
        # Retrieve missing & unexpected_keys
        model_state_dict = model.state_dict()
        loaded_keys = [k for k in state_dict.keys()]

        expected_keys = list(model_state_dict.keys())

        original_loaded_keys = loaded_keys

        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))

        # Make sure we are able to load base models as well as derived models (with heads)
        model_to_load = model

        def _find_mismatched_keys(
            state_dict,
            model_state_dict,
            loaded_keys,
            ignore_mismatched_sizes,
        ):
            mismatched_keys = []
            if ignore_mismatched_sizes:
                for checkpoint_key in loaded_keys:
                    model_key = checkpoint_key

                    if (
                        model_key in model_state_dict
                        and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                    ):
                        mismatched_keys.append(
                            (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                        )
                        del state_dict[checkpoint_key]
            return mismatched_keys

        if state_dict is not None:
            # Whole checkpoint
            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                original_loaded_keys,
                ignore_mismatched_sizes,
            )
            error_msgs = _load_state_dict_into_model(model_to_load, state_dict)

        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if "size mismatch" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task"
                " or with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly"
                " identical (initializing a BertForSequenceClassification model from a"
                " BertForSequenceClassification model)."
            )
        else:
            logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the"
                f" checkpoint was trained on, you can already use {model.__class__.__name__} for predictions"
                " without further training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be"
                " able to use it for predictions and inference."
            )

        return model, missing_keys, unexpected_keys, mismatched_keys, error_msgs

    @property
    def device(self) -> device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """
        Get number of (optionally, trainable or non-embeddings) parameters in the module.

        Args:
            only_trainable (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of trainable parameters

            exclude_embeddings (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of non-embeddings parameters

        Returns:
            `int`: The number of parameters.
        """

        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight"
                for name, module_type in self.named_modules()
                if isinstance(module_type, torch.nn.Embedding)
            ]
            non_embedding_parameters = [
                parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
            ]
            return sum(p.numel() for p in non_embedding_parameters if p.requires_grad or not only_trainable)
        else:
            return sum(p.numel() for p in self.parameters() if p.requires_grad or not only_trainable)


""" ConfigMixin base class and utilities."""


class FrozenDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for key, value in self.items():
            setattr(self, key, value)

        self.__frozen = True

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __setattr__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance.")
        super().__setattr__(name, value)

    def __setitem__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance.")
        super().__setitem__(name, value)


class ConfigMixin:
    r"""
    Base class for all configuration classes. Stores all configuration parameters under `self.config` Also handles all
    methods for loading/downloading/saving classes inheriting from [`ConfigMixin`] with
        - [`~ConfigMixin.from_config`]
        - [`~ConfigMixin.save_config`]

    Class attributes:
        - **config_name** (`str`) -- A filename under which the config should stored when calling
          [`~ConfigMixin.save_config`] (should be overridden by parent class).
        - **ignore_for_config** (`List[str]`) -- A list of attributes that should not be saved in the config (should be
          overridden by subclass).
        - **has_compatibles** (`bool`) -- Whether the class has compatible classes (should be overridden by subclass).
        - **_deprecated_kwargs** (`List[str]`) -- Keyword arguments that are deprecated. Note that the init function
          should only have a `kwargs` argument if at least one argument is deprecated (should be overridden by
          subclass).
    """
    config_name = None
    ignore_for_config = []
    has_compatibles = False

    _deprecated_kwargs = []

    def register_to_config(self, **kwargs):
        if self.config_name is None:
            raise NotImplementedError(f"Make sure that {self.__class__} has defined a class name `config_name`")
        # Special case for `kwargs` used in deprecation warning added to schedulers
        # TODO: remove this when we remove the deprecation warning, and the `kwargs` argument,
        # or solve in a more general way.
        kwargs.pop("kwargs", None)
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

        if not hasattr(self, "_internal_dict"):
            internal_dict = kwargs
        else:
            previous_dict = dict(self._internal_dict)
            internal_dict = {**self._internal_dict, **kwargs}
            logger.debug(f"Updating config from {previous_dict} to {internal_dict}")

        self._internal_dict = FrozenDict(internal_dict)

    def save_config(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~ConfigMixin.from_config`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        # If we save using the predefined names, we can load using `from_config`
        output_config_file = os.path.join(save_directory, self.config_name)

        self.to_json_file(output_config_file)
        logger.info(f"Configuration saved in {output_config_file}")

    @classmethod
    def from_config(cls, config: Union[FrozenDict, Dict[str, Any]] = None, **kwargs):
        r"""
        Instantiate a Python class from a config dictionary

        Parameters:
            config (`Dict[str, Any]`):
                A config dictionary from which the Python class will be instantiated. Make sure to only load
                configuration files of compatible classes.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                Whether kwargs that are not consumed by the Python class should be returned or not.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the Python class.
                `**kwargs` will be directly passed to the underlying scheduler/model's `__init__` method and eventually
                overwrite same named arguments of `config`.

        Examples:

        ```python
        >>> from diffusers import DDPMScheduler, DDIMScheduler, PNDMScheduler

        >>> # Download scheduler from huggingface.co and cache.
        >>> scheduler = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")

        >>> # Instantiate DDIM scheduler class with same config as DDPM
        >>> scheduler = DDIMScheduler.from_config(scheduler.config)

        >>> # Instantiate PNDM scheduler class with same config as DDPM
        >>> scheduler = PNDMScheduler.from_config(scheduler.config)
        ```
        """
        # <===== TO BE REMOVED WITH DEPRECATION
        # TODO(Patrick) - make sure to remove the following lines when config=="model_path" is deprecated
        if "pretrained_model_name_or_path" in kwargs:
            config = kwargs.pop("pretrained_model_name_or_path")

        if config is None:
            raise ValueError("Please make sure to provide a config as the first positional argument.")
        # ======>

        # Return model and optionally state and/or unused_kwargs
        model = cls(**config)
        return model

    @classmethod
    def load_config(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], return_unused_kwargs=False, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        r"""
        Instantiate a Python class from a config dictionary

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* of a model repo on huggingface.co. Valid model ids should have an
                      organization name, like `google/ddpm-celebahq-256`.
                    - A path to a *directory* containing model weights saved using [`~ConfigMixin.save_config`], e.g.,
                      `./my_model_directory/`.

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `transformers-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo (either remote in
                huggingface.co or downloaded locally), you can specify the folder name here.

        <Tip>

         It is required to be logged in (`huggingface-cli login`) when you want to use private or [gated
         models](https://huggingface.co/docs/hub/models-gated#gated-models).

        </Tip>

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to
        use this method in a firewalled environment.

        </Tip>
        """
        cache_dir = kwargs.pop("cache_dir", MUSE_CACHE)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        _ = kwargs.pop("mirror", None)
        subfolder = kwargs.pop("subfolder", None)

        user_agent = {"file_type": "config"}

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        if cls.config_name is None:
            raise ValueError(
                "`self.config_name` is not defined. Note that one should not load a config from "
                "`ConfigMixin`. Please make sure to define `config_name` in a class inheriting from `ConfigMixin`"
            )

        if os.path.isfile(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        elif os.path.isdir(pretrained_model_name_or_path):
            if os.path.isfile(os.path.join(pretrained_model_name_or_path, cls.config_name)):
                # Load from a PyTorch checkpoint
                config_file = os.path.join(pretrained_model_name_or_path, cls.config_name)
            elif subfolder is not None and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, cls.config_name)
            ):
                config_file = os.path.join(pretrained_model_name_or_path, subfolder, cls.config_name)
            else:
                raise EnvironmentError(
                    f"Error no file named {cls.config_name} found in directory {pretrained_model_name_or_path}."
                )
        else:
            try:
                # Load from URL or cache if already cached
                config_file = hf_hub_download(
                    pretrained_model_name_or_path,
                    filename=cls.config_name,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    user_agent=user_agent,
                    subfolder=subfolder,
                    revision=revision,
                )

            except RepositoryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier"
                    " listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a"
                    " token having permission to this repo with `use_auth_token` or log in with `huggingface-cli"
                    " login`."
                )
            except RevisionNotFoundError:
                raise EnvironmentError(
                    f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for"
                    " this model name. Check the model page at"
                    f" 'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
                )
            except EntryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} does not appear to have a file named {cls.config_name}."
                )
            except HTTPError as err:
                raise EnvironmentError(
                    "There was a specific connection error when trying to load"
                    f" {pretrained_model_name_or_path}:\n{err}"
                )
            except ValueError:
                raise EnvironmentError(
                    f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it"
                    f" in the cached files and it looks like {pretrained_model_name_or_path} is not the path to a"
                    f" directory containing a {cls.config_name} file.\nCheckout your internet connection or see how to"
                    " run the library in offline mode at"
                    " 'https://huggingface.co/docs/diffusers/installation#offline-mode'."
                )
            except EnvironmentError:
                raise EnvironmentError(
                    f"Can't load config for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                    "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                    f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                    f"containing a {cls.config_name} file"
                )

        try:
            # Load config dict
            config_dict = cls._dict_from_json_file(config_file)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(f"It looks like the config file at '{config_file}' is not a valid JSON file.")

        if return_unused_kwargs:
            return config_dict, kwargs

        return config_dict

    @staticmethod
    def _get_init_keys(cls):
        return set(dict(inspect.signature(cls.__init__).parameters).keys())

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    @property
    def config(self) -> Dict[str, Any]:
        """
        Returns the config of the class as a frozen dictionary

        Returns:
            `Dict[str, Any]`: Config of the class.
        """
        return self._internal_dict

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        config_dict = self._internal_dict if hasattr(self, "_internal_dict") else {}
        config_dict["_class_name"] = self.__class__.__name__
        config_dict["_version"] = __version__

        def to_json_saveable(value):
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, PosixPath):
                value = str(value)
            return value

        config_dict = {k: to_json_saveable(v) for k, v in config_dict.items()}
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())


def register_to_config(init):
    r"""
    Decorator to apply on the init of classes inheriting from [`ConfigMixin`] so that all the arguments are
    automatically sent to `self.register_for_config`. To ignore a specific argument accepted by the init but that
    shouldn't be registered in the config, use the `ignore_for_config` class variable

    Warning: Once decorated, all private arguments (beginning with an underscore) are trashed and not sent to the init!
    """

    @functools.wraps(init)
    def inner_init(self, *args, **kwargs):
        # Ignore private kwargs in the init.
        init_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        config_init_kwargs = {k: v for k, v in kwargs.items() if k.startswith("_")}
        if not isinstance(self, ConfigMixin):
            raise RuntimeError(
                f"`@register_for_config` was applied to {self.__class__.__name__} init method, but this class does "
                "not inherit from `ConfigMixin`."
            )

        ignore = getattr(self, "ignore_for_config", [])
        # Get positional arguments aligned with kwargs
        new_kwargs = {}
        signature = inspect.signature(init)
        parameters = {
            name: p.default for i, (name, p) in enumerate(signature.parameters.items()) if i > 0 and name not in ignore
        }
        for arg, name in zip(args, parameters.keys()):
            new_kwargs[name] = arg

        # Then add all kwargs
        new_kwargs.update(
            {
                k: init_kwargs.get(k, default)
                for k, default in parameters.items()
                if k not in ignore and k not in new_kwargs
            }
        )
        new_kwargs = {**config_init_kwargs, **new_kwargs}
        getattr(self, "register_to_config")(**new_kwargs)
        init(self, *args, **init_kwargs)

    return inner_init
