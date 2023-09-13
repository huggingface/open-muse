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

# This file is heavily inspired by the original implementation from https://github.com/lucidrains/muse-maskgit-pytorch

import dataclasses
import math
import numbers
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .modeling_utils import ConfigMixin, ModelMixin
from .sampling import cosine_schedule, mask_by_random_topk

try:
    import xformers.ops as xops

    is_xformers_available = True
except ImportError:
    is_xformers_available = False

try:
    from flash_attn.ops.rms_norm import dropout_add_rms_norm
except ImportError:
    dropout_add_rms_norm = None

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm = None

try:
    from flash_attn.ops.fused_dense import fused_mlp_func
except ImportError:
    fused_mlp_func = None


def sinusoidal_encode(features, embedding_dim, max_positions=10000):
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / half_dim
    emb = (
        torch.arange(
            0,
            half_dim,
            device=features.device,
            dtype=torch.float32,
        )
        .mul(-emb)
        .exp()
    )
    emb = features[:, None] * emb[None, :]
    emb = torch.cat([emb.cos(), emb.sin()], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = nn.functional.pad(emb, (0, 1), mode="constant")
    return emb


@dataclass
class MaskGiTUViT_v2Config:
    # global config
    hidden_size: int = 1024
    use_bias: bool = False
    hidden_dropout: float = 0.0

    # conditioning dimensions
    cond_embed_dim: int = 768
    micro_cond_encode_dim: int = 256
    micro_cond_embed_dim: int = 1280
    encoder_hidden_size: int = 768

    # num tokens
    vocab_size: int = 8256  # codebook_size + 1 (for the mask token) rounded
    mask_token_id: int = 8255
    codebook_size: int = 8192

    # `DownsampleBlock` and `UpsampleBlock`
    in_channels: int = 768
    block_out_channels: Tuple[int] = (768,)
    num_res_blocks: int = 3
    force_down_up_sample: bool = False
    block_num_heads: int = 12

    # `TransformerLayer`
    num_hidden_layers: int = 22
    num_attention_heads: int = 16

    # `Attention`
    attention_dropout: float = 0.0

    # `FeedForward`
    intermediate_size: int = 2816
    use_fused_mlp: bool = False

    # `Norm`
    norm_type: str = "rmsnorm"
    layer_norm_eps: float = 1e-6
    ln_elementwise_affine: bool = True
    use_fused_residual_norm: bool = False

    # Legacy: kept for compatibility with pipeline
    add_cond_embeds: bool = True
    add_micro_cond_embeds: bool = True


def config_from_legacy_kwargs(**kwargs):
    if "block_num_heads" in kwargs:
        if isinstance(kwargs["block_num_heads"], (tuple, list)):
            assert len(kwargs["block_num_heads"]) == 1
            kwargs["block_num_heads"] = kwargs["block_num_heads"][0]
        elif isinstance(kwargs["block_num_heads"], int):
            ...
        else:
            assert False

    config = {}

    # select only values that are expected to be in the config
    for field in dataclasses.fields(MaskGiTUViT_v2Config):
        if field.name in kwargs:
            config[field.name] = kwargs[field.name]

    # set default config values
    config = MaskGiTUViT_v2Config(**config)
    config.block_out_channels = list(config.block_out_channels)

    return config


class MaskGiTUViT_v2(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    def __init__(self, **kwargs):
        super().__init__()

        config = config_from_legacy_kwargs(**kwargs)
        self.register_to_config(**dataclasses.asdict(config))
        self.register_to_config(mask_token_id=self.config.vocab_size - 1)

        assert len(self.config.block_out_channels) == 1

        # Legacy: kept for compatibility with pipeline
        self.output_size = self.config.codebook_size

        self.encoder_proj = nn.Linear(
            self.config.encoder_hidden_size, self.config.hidden_size, bias=self.config.use_bias
        )
        self.encoder_proj_layer_norm = Norm(self.config.hidden_size, self.config)

        self.embed = ConvEmbed(self.config)

        self.cond_embed = nn.Sequential(
            nn.Linear(
                self.config.micro_cond_embed_dim + self.config.cond_embed_dim,
                self.config.hidden_size,
                bias=self.config.use_bias,
            ),
            nn.SiLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=self.config.use_bias),
        )

        self.down_blocks = nn.ModuleList([DownsampleBlock(self.config.block_out_channels[0], self.config)])

        self.project_to_hidden_norm = Norm(self.config.block_out_channels[-1], self.config)
        self.project_to_hidden = nn.Linear(
            self.config.block_out_channels[-1], self.config.hidden_size, bias=self.config.use_bias
        )

        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(self.config) for _ in range(self.config.num_hidden_layers)]
        )

        self.project_from_hidden_norm = Norm(self.config.hidden_size, self.config)
        self.project_from_hidden = nn.Linear(
            self.config.hidden_size, self.config.block_out_channels[-1], bias=self.config.use_bias
        )

        self.up_blocks = nn.ModuleList([UpsampleBlock(self.config.block_out_channels[0], self.config)])

        self.mlm_layer = ConvMlmLayer(self.config)

        self.gradient_checkpointing = False

        # --- WEIGHT INIT ---
        self.apply(self._init_weights)  # General init
        nn.init.xavier_uniform_(self.embed.conv.weight, 0.02)  # inputs
        nn.init.normal_(self.embed.embeddings.weight, std=np.sqrt(1 / self.config.vocab_size))
        nn.init.constant_(self.mlm_layer.conv1.weight, 0)  # output
        self.mlm_layer.conv2.weight.data = self.embed.embeddings.weight.data[
            : self.config.codebook_size, :, None, None
        ].clone()

        # init AdaLNModulation.mapper layers to 0
        for m in self.modules():
            if isinstance(m, AdaLNModulation):
                nn.init.constant_(m.mapper.weight, 0)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _init_weights(self, module):
        """
        Initialize the weights according to the original implementation.
        https://github.com/google-research/maskgit/blob/main/maskgit/nets/maskgit_transformer.py#L37
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
        elif isinstance(module, (LayerNorm, RMSNorm)):
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        cond_embeds,
        micro_conds,
        labels=None,
        label_smoothing=0.0,
        loss_weight=None,
    ):
        encoder_hidden_states = self.encoder_proj(encoder_hidden_states)
        encoder_hidden_states, _ = self.encoder_proj_layer_norm(encoder_hidden_states)

        micro_cond_embeds = sinusoidal_encode(micro_conds.flatten(), self.config.micro_cond_encode_dim)
        micro_cond_embeds = micro_cond_embeds.reshape((input_ids.shape[0], -1))

        cond_embeds = torch.cat([cond_embeds, micro_cond_embeds], dim=1)
        cond_embeds = cond_embeds.to(dtype=self.dtype)
        cond_embeds = self.cond_embed(cond_embeds).to(encoder_hidden_states.dtype)

        hidden_states = self.embed(input_ids)

        hidden_states = self.down_blocks[0](
            hidden_states, cond_embeds=cond_embeds, encoder_hidden_states=encoder_hidden_states
        )

        batch_size, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)

        hidden_states, _ = self.project_to_hidden_norm(hidden_states)
        hidden_states = self.project_to_hidden(hidden_states)

        transformer_residual = None

        for layer in self.transformer_layers:
            if self.training and self.gradient_checkpointing:
                layer_ = lambda *args: checkpoint(layer, *args)
            else:
                layer_ = layer

            hidden_states, transformer_residual = layer_(
                hidden_states,
                encoder_hidden_states,
                cond_embeds,
                transformer_residual,
            )

        hidden_states = hidden_states + transformer_residual

        hidden_states, _ = self.project_from_hidden_norm(hidden_states)
        hidden_states = self.project_from_hidden(hidden_states)

        hidden_states = hidden_states.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)

        assert len(self.up_blocks) == 1
        hidden_states = self.up_blocks[0](
            hidden_states, cond_embeds=cond_embeds, encoder_hidden_states=encoder_hidden_states
        )

        batch_size, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        logits = self.mlm_layer(hidden_states)

        if labels is not None:
            reduction = "none" if loss_weight is not None else "mean"
            loss = F.cross_entropy(
                logits.view(-1, self.codebook_size),
                labels.view(-1),
                ignore_index=-100,
                label_smoothing=label_smoothing,
                reduction=reduction,
            )
            if loss_weight is not None:
                loss_weight = loss_weight.view(-1)
                loss = ((loss * loss_weight).sum(dim=-1) / loss_weight.sum(dim=-1)).mean()
            return logits, loss

        return logits

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value
        if isinstance(module, (DownsampleBlock, UpsampleBlock)):
            module.gradient_checkpointing = value

    # Legacy: kept for compatibility with pipeline
    def generate(self):
        assert False

    def generate2(
        self,
        encoder_hidden_states: torch.FloatTensor,
        cond_embeds: torch.FloatTensor,
        micro_conds: torch.FloatTensor,
        empty_embeds: torch.FloatTensor,
        empty_cond_embeds: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        negative_embeds: torch.FloatTensor = None,
        negative_cond_embeds: torch.FloatTensor = None,
        temperature=1.0,
        timesteps=18,  # ideal number of steps is 18 in maskgit paper
        guidance_scale=0,
        guidance_schedule=None,
        noise_schedule=cosine_schedule,
        generator: torch.Generator = None,
        return_intermediate=False,
        seq_len=None,
        use_tqdm=None,
        # Legacy: kept for compatibility with pipeline
        topk_filter_thres=None,
        noise_type=None,
        predict_all_tokens=None,
    ):
        batch_size = encoder_hidden_states.shape[0]

        if seq_len is None:
            seq_len = 256

        shape = (batch_size, seq_len)

        if isinstance(temperature, tuple):
            temperatures = torch.linspace(temperature[0], temperature[1], timesteps)
        else:
            temperatures = torch.linspace(temperature, 0.01, timesteps)

        if input_ids is None:
            input_ids = torch.ones(shape, dtype=torch.long, device=self.device) * self.config.mask_token_id

        if return_intermediate:
            intermediate = []

        if guidance_schedule == "linear":
            guidance_scales = torch.linspace(0, guidance_scale, timesteps)
        elif guidance_schedule == "cosine":
            guidance_scales = []
            for step in range(timesteps):
                ratio = 1.0 * (step + 1) / timesteps
                scale = cosine_schedule(torch.tensor(1 - ratio)) * guidance_scale
                guidance_scales.append(scale.floor())
            guidance_scales = torch.tensor(guidance_scales)
        else:
            guidance_scales = torch.ones(timesteps) * guidance_scale

        if micro_conds.shape[0] == 1:
            micro_conds = micro_conds.repeat(batch_size, 1).to(input_ids.device)

        if guidance_scale > 0:
            # encoder_hidden_states

            if negative_embeds is None:
                uncond_encoder_states = empty_embeds
            else:
                uncond_encoder_states = negative_embeds

            if uncond_encoder_states.shape[0] == 1:
                uncond_encoder_states = uncond_encoder_states.expand(batch_size, -1, -1)

            encoder_hidden_states = torch.cat([encoder_hidden_states, uncond_encoder_states])

            # cond_embeds

            if negative_cond_embeds is None:
                uncond_embeds = empty_cond_embeds
            else:
                uncond_embeds = negative_cond_embeds

            if uncond_embeds.shape[0] == 1:
                uncond_embeds = uncond_embeds.expand(batch_size, -1)

            cond_embeds = torch.cat([cond_embeds, uncond_embeds])

            # micro_conds

            micro_conds = torch.cat([micro_conds, micro_conds], dim=0)

        if use_tqdm:
            from tqdm import tqdm
            timesteps_iter = tqdm(range(timesteps))
        else:
            timesteps_iter = range(timesteps)
        
        for step in timesteps_iter:
            if guidance_scale > 0:
                model_input = torch.cat([input_ids] * 2)

            model_output = self(
                model_input,
                micro_conds=micro_conds,
                cond_embeds=cond_embeds,
                encoder_hidden_states=encoder_hidden_states,
            )

            if guidance_scale > 0:
                cond_logits, uncond_logits = model_output.chunk(2)
                cond_logits = cond_logits[..., : self.config.codebook_size]
                uncond_logits = uncond_logits[..., : self.config.codebook_size]
                logits = uncond_logits + guidance_scales[step] * (cond_logits - uncond_logits)
            else:
                logits = model_output

            logits = logits[..., : self.config.codebook_size]

            probs = logits.softmax(dim=-1)

            sampled = probs.reshape(-1, logits.size(-1))
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])

            if return_intermediate:
                intermediate.append(sampled_ids)

            unknown_map = input_ids == self.config.mask_token_id
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))

            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (seq_len * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device),
                torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len),
            )

            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)
            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            temperature = temperatures[step]
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            input_ids = torch.where(masking, self.config.mask_token_id, sampled_ids)

        if return_intermediate:
            return sampled_ids, intermediate

        return sampled_ids


# embedding blocks


class ConvEmbed(nn.Module):
    def __init__(self, config: MaskGiTUViT_v2Config):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.in_channels)
        self.layer_norm = Norm(config.in_channels, config)
        self.conv = nn.Conv2d(config.in_channels, config.block_out_channels[0], kernel_size=1, bias=config.use_bias)

    def forward(self, input_ids):
        batch_size, seq_length = input_ids.shape
        height, width = int(seq_length**0.5), int(seq_length**0.5)
        input_ids = input_ids.view(-1, height, width)
        embeddings = self.embeddings(input_ids)
        embeddings, _ = self.layer_norm(embeddings)
        embeddings = embeddings.permute(0, 3, 1, 2)
        embeddings = self.conv(embeddings)
        return embeddings


# down/upsample blocks


class DownsampleBlock(nn.Module):
    def __init__(self, channels, config: MaskGiTUViT_v2Config):
        super().__init__()

        if config.force_down_up_sample:
            self.downsample = nn.Sequential(
                Norm2D(channels, config),
                nn.Conv2d(channels, channels, kernel_size=2, stride=2, bias=config.use_bias),
            )
        else:
            self.downsample = None

        self.res_blocks = nn.ModuleList([ResBlock(channels, config) for _ in range(config.num_res_blocks)])

        self.attention_blocks = nn.ModuleList(
            [AttentionBlock2D(channels, config) for _ in range(config.num_res_blocks)]
        )

        self.gradient_checkpointing = False

    def forward(self, x, cond_embeds, encoder_hidden_states):
        if self.downsample is not None:
            x = self.downsample(x)

        for res_block, attention_block in zip(self.res_blocks, self.attention_blocks):
            if self.training and self.gradient_checkpointing:
                res_block_ = lambda *args: checkpoint(res_block, *args)
                attention_block_ = lambda *args: checkpoint(attention_block, *args)
            else:
                res_block_ = res_block
                attention_block_ = attention_block

            x = res_block_(x, cond_embeds)
            x = attention_block_(x, encoder_hidden_states)

        return x


class UpsampleBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        config: MaskGiTUViT_v2Config,
    ):
        super().__init__()

        self.res_blocks = nn.ModuleList([ResBlock(channels, config) for i in range(config.num_res_blocks)])

        self.attention_blocks = nn.ModuleList(
            [AttentionBlock2D(channels, config) for _ in range(config.num_res_blocks)]
        )

        if config.force_down_up_sample:
            self.upsample = nn.Sequential(
                Norm2D(channels, config),
                nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2, bias=config.use_bias),
            )
        else:
            self.upsample = None

        self.gradient_checkpointing = False

    def forward(self, x, cond_embeds, encoder_hidden_states):
        for res_block, attention_block in zip(self.res_blocks, self.attention_blocks):
            if self.training and self.gradient_checkpointing:
                res_block_ = lambda *args: checkpoint(res_block, *args)
                attention_block_ = lambda *args: checkpoint(attention_block, *args)
            else:
                res_block_ = res_block
                attention_block_ = attention_block

            x = res_block_(x, cond_embeds)
            x = attention_block_(x, encoder_hidden_states)

        if self.upsample is not None:
            x = self.upsample(x)

        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
        config: MaskGiTUViT_v2Config,
        res_ffn_factor=4,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=config.use_bias,
        )
        self.norm = Norm2D(channels, config)
        self.channelwise = nn.Sequential(
            nn.Linear(channels, int(channels * res_ffn_factor), bias=config.use_bias),
            nn.GELU(),
            GlobalResponseNorm(int(channels * res_ffn_factor)),
            nn.Dropout(config.hidden_dropout),
            nn.Linear(int(channels * res_ffn_factor), channels, bias=config.use_bias),
        )
        self.adaLN_modulation = AdaLNModulation(channels, config)

    def forward(self, x, cond_embeds):
        x_res = x
        x = self.norm(self.depthwise(x)).permute(0, 2, 3, 1)
        x = self.channelwise(x).permute(0, 3, 1, 2)
        x = x + x_res
        x = self.adaLN_modulation(x, cond_embeds)
        return x


# norm blocks


class Norm2D(nn.Module):
    def __init__(self, dim, config: MaskGiTUViT_v2Config):
        super().__init__()
        self.norm = Norm(dim, config)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x, _ = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


def Norm(dim, config: MaskGiTUViT_v2Config):
    if config.norm_type == "layernorm":
        return LayerNorm(dim, config)
    elif config.norm_type == "rmsnorm":
        return RMSNorm(dim, config)
    else:
        assert False


class RMSNorm(nn.Module):
    def __init__(self, dim, config: MaskGiTUViT_v2Config):
        super().__init__()

        self.config = config

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)

        if self.config.ln_elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, input, residual=None):
        if self.config.use_fused_residual_norm:
            if dropout_add_rms_norm is None:
                raise ImportError("Please install flash_attn to use fused rms norm")

            return dropout_add_rms_norm(
                input, residual, self.weight, None, dropout_p=0.0, epsilon=self.config.layer_norm_eps, prenorm=True
            )
        else:
            return unfused_rms_norm(input, residual, self.weight, self.config.layer_norm_eps)


def unfused_rms_norm(input, residual, weight, eps):
    if residual is not None:
        input = input + residual

    prenorm_residual = input

    input_dtype = input.dtype
    variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)
    input = input * torch.rsqrt(variance + eps)

    if weight is not None:
        # convert into half-precision if necessary
        if weight.dtype in [torch.float16, torch.bfloat16]:
            input = input.to(weight.dtype)
        input = input * weight
    else:
        input = input.to(input_dtype)

    return input, prenorm_residual


class LayerNorm(nn.Module):
    def __init__(self, dim, config: MaskGiTUViT_v2Config):
        super().__init__()

        self.config = config

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)

        if self.config.ln_elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim)) if self.config.use_bias else None
        else:
            self.weight = None
            self.bias = None

    def forward(self, input, residual=None):
        if self.config.use_fused_residual_norm:
            if dropout_add_layer_norm is None:
                raise ImportError("Please install flash_attn to use fused layer norm")

            return dropout_add_layer_norm(
                x0=input,
                residual=residual,
                weight=self.weight,
                bias=self.bias,
                epsilon=self.config.layer_norm_eps,
                dropout_p=0.0,
                prenorm=True,
            )
        else:
            return unfused_layer_norm(input, residual, self.dim, self.weight, self.bias, self.config.layer_norm_eps)


def unfused_layer_norm(input, residual, dim, weight, bias, eps):
    if residual is not None:
        input = input + residual

    prenorm_residual = input

    input = F.layer_norm(input, dim, weight, bias, eps)

    return input, prenorm_residual


class GlobalResponseNorm(nn.Module):
    # Taken from https://github.com/facebookresearch/ConvNeXt-V2/blob/3608f67cc1dae164790c5d0aead7bf2d73d9719b/models/utils.py#L105
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# attention/transformer blocks


class TransformerLayer(nn.Module):
    def __init__(self, config: MaskGiTUViT_v2Config):
        super().__init__()

        self.attn_layer_norm = Norm(config.hidden_size, config)

        self.self_attn_adaLN_modulation = AdaLNModulation(config.hidden_size, config)
        self.attention = Attention(config.hidden_size, config.hidden_size, config.num_attention_heads, config)

        self.crossattn_layer_norm = Norm(config.hidden_size, config)
        self.crossattention = Attention(
            config.hidden_size, config.hidden_size, config.num_attention_heads, config
        )
        self.cross_attn_adaLN_modulation = AdaLNModulation(config.hidden_size, config)

        self.ffn = FeedForward(config)

    def forward(self, hidden_states, encoder_hidden_states, cond_embeds, residual=None):
        hidden_states, residual = self.attn_layer_norm(hidden_states, residual=residual)

        hidden_states = self.self_attn_adaLN_modulation(hidden_states, cond_embeds)

        hidden_states = self.attention(hidden_states, hidden_states)

        hidden_states, residual = self.crossattn_layer_norm(hidden_states, residual=residual)

        hidden_states = self.cross_attn_adaLN_modulation(hidden_states, cond_embeds)

        hidden_states = self.crossattention(
            hidden_states,
            encoder_hidden_states,
        )

        hidden_states, residual = self.ffn(hidden_states, cond_embeds=cond_embeds, residual=residual)

        return hidden_states, residual


class AttentionBlock2D(nn.Module):
    def __init__(self, hidden_size: int, config: MaskGiTUViT_v2Config):
        super().__init__()

        if config.hidden_size != hidden_size:
            self.kv_mapper = nn.Linear(config.hidden_size, hidden_size, bias=config.use_bias)
        else:
            self.kv_mapper = None

        encoder_hidden_size = hidden_size

        # NOTE: this is actually a cross attention layer, but keeping the naming from v1 to
        # keep the state dicts compatible
        self.attn_layer_norm = Norm(hidden_size, config)
        self.attention = Attention(hidden_size, encoder_hidden_size, config.block_num_heads, config)

        self.crossattn_layer_norm = Norm(hidden_size, config)
        self.crossattention = Attention(hidden_size, encoder_hidden_size, config.block_num_heads, config)

    def forward(self, hidden_states, encoder_hidden_states):
        batch_size, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channels, height * width).permute(0, 2, 1)

        if self.kv_mapper is not None:
            encoder_hidden_states = self.kv_mapper(F.silu(encoder_hidden_states))

        # NOTE: This is actually a cross attention layer
        hidden_states, residual = self.attn_layer_norm(hidden_states)
        hidden_states = self.attention(hidden_states, encoder_hidden_states)

        hidden_states, residual = self.crossattn_layer_norm(hidden_states, residual)
        hidden_states = self.crossattention(hidden_states, encoder_hidden_states)
        hidden_states = hidden_states + residual

        hidden_states = hidden_states.permute(0, 2, 1).view(batch_size, channels, height, width)

        return hidden_states


class Attention(nn.Module):
    def __init__(self, hidden_size: int, context_dim: int, num_heads: int, config: MaskGiTUViT_v2Config):
        super().__init__()

        self.config = config
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // num_heads

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"self.hidden_size: {self.hidden_size} must be divisible by self.num_heads: {self.num_heads}"
            )

        self.scale_attn = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())

        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=self.config.use_bias)

        self.key = nn.Linear(context_dim, self.hidden_size, bias=self.config.use_bias)
        self.value = nn.Linear(context_dim, self.hidden_size, bias=self.config.use_bias)

        self.out = nn.Linear(self.hidden_size, self.hidden_size, bias=self.config.use_bias)
        self.dropout = nn.Dropout(self.config.attention_dropout)

        self.use_memory_efficient_attention_xformers = False
        self.xformers_attention_op = None

    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
    ):
        if use_memory_efficient_attention_xformers and not is_xformers_available:
            raise ImportError("Please install xformers to use memory efficient attention")
        self.use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
        self.xformers_attention_op = attention_op

    def forward(self, hidden_states, context):
        batch, q_seq_len, _ = hidden_states.shape
        kv_seq_len = context.shape[1]

        query = self.query(hidden_states)
        key = self.key(context)
        value = self.value(context)

        query = query.view(batch, q_seq_len, self.num_heads, self.head_dim)  # (B, T, nh, hs)
        key = key.view(batch, kv_seq_len, self.num_heads, self.head_dim)  # (B, T, nh, hs)
        value = value.view(batch, kv_seq_len, self.num_heads, self.head_dim)  # (B, T, nh, hs)

        if self.use_memory_efficient_attention_xformers:
            attn_output = xops.memory_efficient_attention(
                query,
                key,
                value,
                op=self.xformers_attention_op,
                p=self.config.attention_dropout if self.training else 0.0,
            )
            attn_output = attn_output.view(batch, q_seq_len, self.hidden_size)
        else:
            attn_output = self.attention(query, key, value)

        attn_output = self.out(attn_output)
        return attn_output

    def attention(self, query, key, value, attention_mask=None):
        batch, seq_len = query.shape[:2]
        kv_seq_len = key.shape[1]
        query, key, value = map(lambda t: t.transpose(1, 2).contiguous(), (query, key, value))  # (B, nh, T, hs)

        attn_weights = torch.baddbmm(
            input=torch.zeros(batch * self.num_heads, seq_len, kv_seq_len, dtype=query.dtype, device=query.device),
            batch1=query.view(batch * self.num_heads, seq_len, self.head_dim),
            batch2=key.view(batch * self.num_heads, kv_seq_len, self.head_dim).transpose(1, 2),
            alpha=1 / self.scale_attn,
        )
        attn_weights = attn_weights.view(batch, self.num_heads, seq_len, kv_seq_len)  # -1 is kv_seq_len
        # Apply the attention mask
        if attention_mask is not None:
            attn_weights = torch.masked_fill(attn_weights, attention_mask, torch.finfo(query.dtype).min)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        return attn_output


def FeedForward(config: MaskGiTUViT_v2Config):
    if config.use_fused_mlp:
        return FusedGeLUFeedForward(config)
    else:
        return GLUFeedForward(config)


class GLUFeedForward(nn.Module):
    def __init__(self, config: MaskGiTUViT_v2Config):
        super().__init__()
        self.pre_mlp_layer_norm = LayerNorm(config.hidden_size, config)
        self.adaLN_modulation = AdaLNModulation(config.hidden_size, config)
        self.wi_0 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.use_bias)
        self.wi_1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.use_bias)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.use_bias)

    def forward(self, hidden_states, cond_embeds, residual=None):
        hidden_states, residual = self.pre_mlp_layer_norm(hidden_states, residual=residual)

        hidden_states = self.adaLN_modulation(hidden_states, cond_embeds)

        hidden_gelu = F.gelu(self.wi_0(hidden_states))

        hidden_linear = self.wi_1(hidden_states)

        hidden_states = hidden_gelu * hidden_linear

        hidden_states = self.dropout(hidden_states)

        hidden_states = self.wo(hidden_states)

        return hidden_states, residual


class FusedGeLUFeedForward(nn.Module):
    def __init__(self, config: MaskGiTUViT_v2Config):
        super().__init__()
        self.pre_mlp_layer_norm = LayerNorm(config.hidden_size, config)
        self.adaLN_modulation = AdaLNModulation(config.hidden_size, config)
        self.wi_0 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.use_bias)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.use_bias)

    def forward(self, hidden_states, cond_embeds, residual=None):
        if fused_mlp_func is None:
            raise ImportError("Please install flash_attn to use fused mlp")

        hidden_states, residual = self.pre_mlp_layer_norm(hidden_states, residual=residual)

        hidden_states = self.adaLN_modulation(hidden_states, cond_embeds)

        dtype = hidden_states.dtype if not torch.is_autocast_enabled() else torch.get_autocast_gpu_dtype()
        cuda_ver = tuple(map(int, torch.version.cuda.split(".")))

        if torch.cuda.get_device_capability("cuda") == (9, 0):
            heuristic = -1
        elif cuda_ver >= (11, 8):
            heuristic = 0
        elif dtype == torch.float16:
            heuristic = 1
        else:
            heuristic = -1

        hidden_states = fused_mlp_func(
            hidden_states,
            self.wi_0.weight,
            self.wo.weight,
            self.wi_0.bias,
            self.wo.bias,
            activation="gelu_approx",
            save_pre_act=self.training,
            return_residual=False,
            checkpoint_lvl=0,
            heuristic=heuristic,
        )

        return hidden_states, residual


# misc blocks


class ConvMlmLayer(nn.Module):
    def __init__(self, config: MaskGiTUViT_v2Config):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv2d(
            self.config.block_out_channels[0], self.config.in_channels, kernel_size=1, bias=self.config.use_bias
        )
        self.layer_norm = Norm2D(self.config.in_channels, config)
        self.conv2 = nn.Conv2d(
            self.config.in_channels, self.config.codebook_size, kernel_size=1, bias=self.config.use_bias
        )

    def forward(self, hidden_states):
        batch_size, seq_length, hidden_size = hidden_states.shape
        resolution = int(seq_length**0.5)
        hidden_states = hidden_states.view(batch_size, resolution, resolution, hidden_size).permute(0, 3, 1, 2)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = self.conv2(hidden_states)
        logits = logits.permute(0, 2, 3, 1).view(batch_size, -1, self.config.codebook_size)
        return logits


class AdaLNModulation(nn.Module):
    def __init__(self, hidden_size: int, config: MaskGiTUViT_v2Config):
        super().__init__()
        self.mapper = nn.Linear(config.hidden_size, hidden_size * 2, bias=config.use_bias)

    def forward(self, hidden_states, cond_embeds):
        cond_embeds = F.silu(cond_embeds)
        scale, shift = self.mapper(cond_embeds).chunk(2, dim=1)
        if hidden_states.dim() > 3:
            scale, shift = scale[:, :, None, None], shift[:, :, None, None]
        else:
            scale, shift = scale[:, None], shift[:, None]
        return hidden_states * (1 + scale) + shift
