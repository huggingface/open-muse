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

__version__ = "0.0.1"

from .modeling_ema import EMAModel
from .modeling_maskgit_vqgan import MaskGitVQGAN
from .modeling_movq import MOVQ
from .modeling_paella_vq import PaellaVQModel
from .modeling_taming_vqgan import VQGANModel
from .modeling_transformer import MaskGitTransformer, MaskGiTUViT
from .pipeline_muse import PipelineMuse
from .sampling import get_mask_chedule
