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

from functools import partial
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
from .sampling import cosine_schedule, gumbel_sample, mask_by_random_topk, top_k

try:
    import xformers.ops as xops

    is_xformers_available = True
except ImportError:
    is_xformers_available = False


# classifier free guidance functions


def uniform(shape, min=0, max=1, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def prob_mask_like(shape, prob, device=None):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return uniform(shape, device=device) < prob


def make_attention_mask(
    query_input: torch.Tensor, key_input: torch.Tensor, pairwise_fn: Callable = torch.mul
) -> torch.Tensor:
    # [batch, len_q, len_kv]
    mask = pairwise_fn(
        # [batch, len_q] -> [batch, len_q, 1]
        torch.unsqueeze(query_input, axis=-1),
        # [batch, len_q] -> [batch, 1, len_kv]
        torch.unsqueeze(key_input, axis=-2),
    )
    # [batch, 1, len_q, len_kv]. This creates the head dim.
    mask = torch.unsqueeze(mask, axis=-3)
    return (1.0 - mask).type(torch.bool)


try:
    from apex.normalization import FusedRMSNorm as RMSNorm  # noqa
except Exception:

    class RMSNorm(nn.Module):
        def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
            super().__init__()
            self.elementwise_affine = elementwise_affine
            if elementwise_affine:
                self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.variance_epsilon = eps

        def forward(self, input):
            input_dtype = input.dtype
            variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)
            input = input * torch.rsqrt(variance + self.variance_epsilon)

            if self.elementwise_affine:
                # convert into half-precision if necessary
                if self.weight.dtype in [torch.float16, torch.bfloat16]:
                    input = input.to(self.weight.dtype)
                input = input * self.weight
            else:
                input = input.to(input_dtype)

            return input


# layer norm without bias
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, use_bias=False, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim)) if use_bias else None
        else:
            self.weight = None
            self.bias = None
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, (self.dim,), self.weight, self.bias, self.eps)


class AdaLNModulation(nn.Module):
    def __init__(self, cond_embed_dim, hidden_size, use_bias=False):
        super().__init__()
        self.mapper = nn.Linear(cond_embed_dim, hidden_size * 2, bias=use_bias)

    def forward(self, hidden_states, cond_embeds):
        cond_embeds = F.silu(cond_embeds)
        scale, shift = self.mapper(cond_embeds).chunk(2, dim=1)
        if hidden_states.dim() > 3:
            scale, shift = scale[:, :, None, None], shift[:, :, None, None]
        else:
            scale, shift = scale[:, None], shift[:, None]
        return hidden_states * (1 + scale) + shift


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, encoder_hidden_size=None, attention_dropout=0.0, use_bias=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attention_dropout = attention_dropout
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.hidden_size} and"
                f" `num_heads`: {self.num_heads})."
            )
        self.scale_attn = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())

        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)

        kv_hidden_size = self.hidden_size if encoder_hidden_size is None else encoder_hidden_size
        self.key = nn.Linear(kv_hidden_size, self.hidden_size, bias=use_bias)
        self.value = nn.Linear(kv_hidden_size, self.hidden_size, bias=use_bias)

        self.out = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)
        self.dropout = nn.Dropout(attention_dropout)

        self.use_memory_efficient_attention_xformers = False
        self.xformers_attention_op = None

    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
    ):
        if use_memory_efficient_attention_xformers and not is_xformers_available:
            raise ImportError("Please install xformers to use memory efficient attention")
        self.use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
        self.xformers_attention_op = attention_op

    def forward(self, hidden_states, encoder_hidden_states=None, encoder_attention_mask=None):
        if encoder_attention_mask is not None and self.use_memory_efficient_attention_xformers:
            raise ValueError("Memory efficient attention does not yet support encoder attention mask")

        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        batch, q_seq_len, _ = hidden_states.shape
        kv_seq_len = q_seq_len if encoder_hidden_states is None else encoder_hidden_states.shape[1]

        query = self.query(hidden_states)
        key = self.key(context)
        value = self.value(context)

        query = query.view(batch, q_seq_len, self.num_heads, self.head_dim)  # (B, T, nh, hs)
        key = key.view(batch, kv_seq_len, self.num_heads, self.head_dim)  # (B, T, nh, hs)
        value = value.view(batch, kv_seq_len, self.num_heads, self.head_dim)  # (B, T, nh, hs)

        if self.use_memory_efficient_attention_xformers:
            attn_output = xops.memory_efficient_attention(
                query, key, value, op=self.xformers_attention_op, p=self.attention_dropout if self.training else 0.0
            )
            attn_output = attn_output.view(batch, q_seq_len, self.hidden_size)
        else:
            attention_mask = None
            if encoder_attention_mask is not None:
                src_attn_mask = torch.ones(batch, q_seq_len, dtype=torch.long, device=query.device)
                attention_mask = make_attention_mask(src_attn_mask, encoder_attention_mask, dtype=query.dtype)
            attn_output = self.attention(query, key, value, attention_mask)

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


# U-ViT blocks
# Adpated from https://github.com/dome272/Paella/blob/main/src_distributed/modules.py


class AttentionBlock2D(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        encoder_hidden_size,
        attention_dropout=0.0,
        norm_type="layernorm",
        layer_norm_eps=1e-6,
        ln_elementwise_affine=True,
        use_bias=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        norm_cls = partial(LayerNorm, use_bias=use_bias) if norm_type == "layernorm" else RMSNorm
        self.attn_layer_norm = norm_cls(self.hidden_size, eps=layer_norm_eps, elementwise_affine=ln_elementwise_affine)
        self.attention = Attention(hidden_size, num_heads, attention_dropout=attention_dropout, use_bias=use_bias)
        self.crossattn_layer_norm = norm_cls(hidden_size, eps=layer_norm_eps, elementwise_affine=ln_elementwise_affine)
        self.crossattention = Attention(hidden_size, num_heads, attention_dropout=attention_dropout, use_bias=use_bias)

        if encoder_hidden_size != hidden_size:
            self.kv_mapper = nn.Linear(encoder_hidden_size, hidden_size, bias=use_bias)
        else:
            self.kv_mapper = None

    def forward(self, hidden_states, encoder_hidden_states, encoder_attention_mask=None):
        # hidden_states -> (bs, hidden_size, height, width)
        # reshape to (bs, height * width, hidden_size)
        batch_size, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channels, height * width).permute(0, 2, 1)

        # map encoder hidden states to hidden size of current layer
        if self.kv_mapper is not None:
            encoder_hidden_states = self.kv_mapper(F.silu(encoder_hidden_states))

        # self attention
        residual = hidden_states
        hidden_states = self.attn_layer_norm(hidden_states)
        hidden_states = self.attention(hidden_states, encoder_hidden_states, encoder_attention_mask)
        hidden_states = hidden_states + residual

        # cross attention
        residual = hidden_states
        hidden_states = self.crossattn_layer_norm(hidden_states)
        hidden_states = self.crossattention(hidden_states, encoder_hidden_states, encoder_attention_mask)
        hidden_states = hidden_states + residual

        # reshape back to (bs, hidden_size, height, width)
        hidden_states = hidden_states.permute(0, 2, 1).view(batch_size, channels, height, width)

        return hidden_states


class Norm2D(nn.Module):
    def __init__(self, dim, eps=1e-5, use_bias=False, norm_type="layernorm", elementwise_affine=True):
        super().__init__()
        if norm_type == "layernorm":
            self.norm = LayerNorm(dim, eps, use_bias, elementwise_affine=elementwise_affine)
        elif norm_type == "rmsnorm":
            self.norm = RMSNorm(dim, eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class GlobalResponseNorm(nn.Module):
    "Taken from https://github.com/facebookresearch/ConvNeXt-V2/blob/3608f67cc1dae164790c5d0aead7bf2d73d9719b/models/utils.py#L105"

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels=None,
        kernel_size=3,
        dropout=0.0,
        norm_type="layernorm",
        ln_elementwise_affine=True,
        add_cond_embeds=False,
        cond_embed_dim=None,
        use_bias=False,
        **kwargs,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels + skip_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=use_bias,
        )
        self.norm = Norm2D(
            in_channels, eps=1e-6, norm_type=norm_type, use_bias=use_bias, elementwise_affine=ln_elementwise_affine
        )
        self.channelwise = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4, bias=use_bias),
            nn.GELU(),
            GlobalResponseNorm(in_channels * 4),
            nn.Dropout(dropout),
            nn.Linear(in_channels * 4, in_channels, bias=use_bias),
        )

        if add_cond_embeds:
            self.adaLN_modulation = AdaLNModulation(
                cond_embed_dim=cond_embed_dim, hidden_size=in_channels, use_bias=use_bias
            )

    def forward(self, x, x_skip=None, cond_embeds=None):
        x_res = x
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.norm(self.depthwise(x)).permute(0, 2, 3, 1)
        x = self.channelwise(x).permute(0, 3, 1, 2)
        x = x + x_res
        if cond_embeds is not None:
            x = self.adaLN_modulation(x, cond_embeds)
        return x


class ResnetBlockVanilla(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, use_bias=False, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias)

        self.norm2 = torch.nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=use_bias
                )

    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)

        return residual + hidden_states


class DownsampleBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels=None,
        skip_channels=None,
        num_res_blocks=4,
        kernel_size=3,
        dropout=0.0,
        norm_type="layernorm",
        ln_elementwise_affine=True,
        add_downsample=True,
        add_cond_embeds=False,
        cond_embed_dim=None,
        has_attention=False,
        num_heads=None,
        encoder_hidden_size=None,
        use_bias=False,
        **kwargs,
    ):
        super().__init__()
        self.add_downsample = add_downsample
        self.has_attention = has_attention
        if add_downsample:
            self.downsample = nn.Sequential(
                Norm2D(
                    input_channels,
                    eps=1e-6,
                    use_bias=use_bias,
                    norm_type=norm_type,
                    elementwise_affine=ln_elementwise_affine,
                ),
                nn.Conv2d(input_channels, output_channels, kernel_size=2, stride=2, bias=use_bias),
            )
            self.input_channels = output_channels
        else:
            self.input_channels = input_channels

        self.res_blocks = nn.ModuleList(
            [
                ResBlock(
                    self.input_channels,
                    skip_channels=skip_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    norm_type=norm_type,
                    ln_elementwise_affine=ln_elementwise_affine,
                    add_cond_embeds=add_cond_embeds,
                    cond_embed_dim=cond_embed_dim,
                    use_bias=use_bias,
                )
                for _ in range(num_res_blocks)
            ]
        )

        if has_attention:
            self.attention_blocks = nn.ModuleList(
                [
                    AttentionBlock2D(
                        hidden_size=self.input_channels,
                        num_heads=num_heads,
                        encoder_hidden_size=encoder_hidden_size,
                        attention_dropout=dropout,
                        norm_type=norm_type,
                        ln_elementwise_affine=ln_elementwise_affine,
                        use_bias=use_bias,
                    )
                    for _ in range(num_res_blocks)
                ]
            )

        self.gradient_checkpointing = False

    def forward(self, x, x_skip=None, cond_embeds=None, encoder_hidden_states=None, **kwargs):
        if self.add_downsample:
            x = self.downsample(x)

        output_states = ()
        for i, res_block in enumerate(self.res_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                x = torch.utils.checkpoint.checkpoint(create_custom_forward(res_block), x, x_skip)
                if self.has_attention:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.attention_blocks[i]), x, encoder_hidden_states
                    )
            else:
                x = res_block(x, x_skip, cond_embeds=cond_embeds)
                if self.has_attention:
                    x = self.attention_blocks[i](x, encoder_hidden_states)

            output_states += (x,)
        return x, output_states


class UpsampleBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels=None,
        skip_channels=None,
        num_res_blocks=4,
        kernel_size=3,
        dropout=0.0,
        norm_type="layernorm",
        ln_elementwise_affine=True,
        add_upsample=True,
        add_cond_embeds=False,
        cond_embed_dim=None,
        has_attention=False,
        num_heads=None,
        encoder_hidden_size=None,
        use_bias=False,
        **kwargs,
    ):
        super().__init__()
        self.add_upsample = add_upsample
        self.has_attention = has_attention
        self.input_channels = input_channels
        self.output_channels = output_channels if output_channels is not None else input_channels

        self.res_blocks = nn.ModuleList(
            [
                ResBlock(
                    self.input_channels,
                    skip_channels=skip_channels if i == 0 else 0,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    norm_type=norm_type,
                    ln_elementwise_affine=ln_elementwise_affine,
                    add_cond_embeds=add_cond_embeds,
                    cond_embed_dim=cond_embed_dim,
                    use_bias=use_bias,
                )
                for i in range(num_res_blocks)
            ]
        )

        if has_attention:
            self.attention_blocks = nn.ModuleList(
                [
                    AttentionBlock2D(
                        hidden_size=self.input_channels,
                        num_heads=num_heads,
                        encoder_hidden_size=encoder_hidden_size,
                        attention_dropout=dropout,
                        norm_type=norm_type,
                        ln_elementwise_affine=ln_elementwise_affine,
                        use_bias=use_bias,
                    )
                    for _ in range(num_res_blocks)
                ]
            )

        if add_upsample:
            self.upsample = nn.Sequential(
                Norm2D(
                    self.input_channels,
                    eps=1e-6,
                    norm_type=norm_type,
                    use_bias=use_bias,
                    elementwise_affine=ln_elementwise_affine,
                ),
                nn.ConvTranspose2d(self.input_channels, self.output_channels, kernel_size=2, stride=2, bias=use_bias),
            )

        self.gradient_checkpointing = False

    def forward(self, x, x_skip=None, cond_embeds=None, encoder_hidden_states=None, **kwargs):
        for i, res_block in enumerate(self.res_blocks):
            x_res = x_skip[0] if i == 0 and x_skip is not None else None

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                x = torch.utils.checkpoint.checkpoint(create_custom_forward(res_block), x, x_res)
                if self.has_attention:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.attention_blocks[i]), x, encoder_hidden_states
                    )
            else:
                x = res_block(x, x_res, cond_embeds=cond_embeds)
                if self.has_attention:
                    x = self.attention_blocks[i](x, encoder_hidden_states)

        if self.add_upsample:
            x = self.upsample(x)
        return x


class DownsampleBlockVanilla(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels=None,
        num_res_blocks=4,
        dropout=0.0,
        add_downsample=True,
        use_bias=False,
        **kwargs,
    ):
        super().__init__()
        self.add_downsample = add_downsample

        res_blocks = []
        for i in range(num_res_blocks):
            in_channels = input_channels if i == 0 else output_channels
            res_blocks.append(
                ResnetBlockVanilla(
                    in_channels=in_channels, out_channels=output_channels, dropout=dropout, use_bias=use_bias
                )
            )
        self.res_blocks = nn.ModuleList(res_blocks)

        if add_downsample:
            self.downsample_conv = nn.Conv2d(output_channels, output_channels, 3, stride=2, bias=use_bias)

        self.gradient_checkpointing = False

    def forward(self, x, **kwargs):
        output_states = ()
        for res_block in self.res_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                x = torch.utils.checkpoint.checkpoint(create_custom_forward(res_block), x)
            else:
                x = res_block(x)

            output_states = output_states + (x,)

        if self.add_downsample:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.downsample_conv(x)
            output_states = output_states + (x,)

        return x, output_states


class UpsampleBlockVanilla(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        skip_channels=None,
        num_res_blocks=4,
        dropout=0.0,
        add_upsample=True,
        use_bias=False,
        **kwargs,
    ):
        super().__init__()
        self.add_upsample = add_upsample
        res_blocks = []
        for i in range(num_res_blocks):
            res_skip_channels = input_channels if (i == num_res_blocks - 1) else output_channels
            resnet_in_channels = skip_channels if i == 0 else output_channels

            res_blocks.append(
                ResnetBlockVanilla(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=output_channels,
                    dropout=dropout,
                )
            )
        self.res_blocks = nn.ModuleList(res_blocks)

        if add_upsample:
            self.upsample_conv = nn.Conv2d(output_channels, output_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x, x_skip, **kwargs):
        for res_block in self.res_blocks:
            # pop res hidden states
            res_hidden_states = x_skip[-1]
            x_skip = x_skip[:-1]
            x = torch.cat([x, res_hidden_states], dim=1)
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                x = torch.utils.checkpoint.checkpoint(create_custom_forward(res_block), x)
            else:
                x = res_block(x)

        if self.add_upsample:
            if x.shape[0] >= 64:
                x = x.contiguous()
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            x = self.upsample_conv(x)

        return x


# End U-ViT blocks


# Normformer style GLU FeedForward
class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        hidden_dropout=0.0,
        norm_type="layernorm",
        layer_norm_eps=1e-5,
        ln_elementwise_affine=True,
        use_normformer=True,
        add_cond_embeds=False,
        cond_embed_dim=None,
        use_bias=False,
        ffn_type="glu",  # glu or vanilla
    ):
        super().__init__()
        self.use_normformer = use_normformer
        self.ffn_type = ffn_type
        self.pre_mlp_layer_norm = LayerNorm(
            hidden_size, eps=layer_norm_eps, use_bias=use_bias, elementwise_affine=ln_elementwise_affine
        )
        self.wi_0 = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        if ffn_type == "glu":
            self.wi_1 = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        if use_normformer:
            norm_cls = partial(LayerNorm, use_bias=use_bias) if norm_type == "layernorm" else RMSNorm
            self.mid_mlp_layer_norm = norm_cls(
                intermediate_size, eps=layer_norm_eps, elementwise_affine=ln_elementwise_affine
            )
        self.wo = nn.Linear(intermediate_size, hidden_size, bias=use_bias)
        self.dropout = nn.Dropout(hidden_dropout)
        if add_cond_embeds:
            self.adaLN_modulation = AdaLNModulation(
                cond_embed_dim=cond_embed_dim, hidden_size=hidden_size, use_bias=use_bias
            )

    def forward(self, hidden_states: torch.FloatTensor, cond_embeds=None) -> torch.FloatTensor:
        hidden_states = self.pre_mlp_layer_norm(hidden_states)
        if cond_embeds is not None:
            hidden_states = self.adaLN_modulation(hidden_states, cond_embeds)
        hidden_gelu = F.gelu(self.wi_0(hidden_states))
        if self.ffn_type == "glu":
            hidden_linear = self.wi_1(hidden_states)
            hidden_states = hidden_gelu * hidden_linear
        else:
            hidden_states = hidden_gelu
        if self.use_normformer:
            hidden_states = self.mid_mlp_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


# PreLN Transformer layer
class TransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        encoder_hidden_size=1024,
        add_cross_attention=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        norm_type="layernorm",
        layer_norm_eps=1e-5,
        ln_elementwise_affine=True,
        use_normformer=True,
        add_cond_embeds=False,
        cond_embed_dim=None,
        ffn_type="glu",
        use_bias=False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.use_normformer = use_normformer

        norm_cls = partial(LayerNorm, use_bias=use_bias) if norm_type == "layernorm" else RMSNorm
        self.attn_layer_norm = norm_cls(self.hidden_size, eps=layer_norm_eps, elementwise_affine=ln_elementwise_affine)
        self.attention = Attention(
            self.hidden_size, self.num_attention_heads, attention_dropout=attention_dropout, use_bias=use_bias
        )
        if use_normformer:
            self.post_attn_layer_norm = norm_cls(
                self.hidden_size, eps=layer_norm_eps, elementwise_affine=ln_elementwise_affine
            )
        self.ffn = FeedForward(
            self.hidden_size,
            self.intermediate_size,
            hidden_dropout,
            norm_type,
            layer_norm_eps,
            ln_elementwise_affine,
            use_normformer,
            add_cond_embeds,
            cond_embed_dim,
            use_bias,
            ffn_type,
        )

        if add_cross_attention:
            self.crossattn_layer_norm = norm_cls(
                self.hidden_size, eps=layer_norm_eps, elementwise_affine=ln_elementwise_affine
            )
            self.crossattention = Attention(
                self.hidden_size, self.num_attention_heads, encoder_hidden_size, attention_dropout, use_bias
            )
            if use_normformer:
                self.post_crossattn_layer_norm = norm_cls(
                    self.hidden_size, eps=layer_norm_eps, elementwise_affine=ln_elementwise_affine
                )

        if add_cond_embeds:
            self.self_attn_adaLN_modulation = AdaLNModulation(
                cond_embed_dim=cond_embed_dim, hidden_size=hidden_size, use_bias=use_bias
            )
            if add_cross_attention:
                self.cross_attn_adaLN_modulation = AdaLNModulation(
                    cond_embed_dim=cond_embed_dim,
                    hidden_size=hidden_size,
                    use_bias=use_bias,
                )

    def forward(self, hidden_states, encoder_hidden_states=None, encoder_attention_mask=None, cond_embeds=None):
        residual = hidden_states

        hidden_states = self.attn_layer_norm(hidden_states)
        if cond_embeds is not None:
            hidden_states = self.self_attn_adaLN_modulation(hidden_states, cond_embeds)
        attention_output = self.attention(hidden_states)
        if self.use_normformer:
            attention_output = self.post_attn_layer_norm(attention_output)
        hidden_states = residual + attention_output

        if encoder_hidden_states is not None:
            residual = hidden_states
            # TODO: should norm be applied to encoder_hidden_states as well?
            hidden_states = self.crossattn_layer_norm(hidden_states)
            if cond_embeds is not None:
                hidden_states = self.cross_attn_adaLN_modulation(hidden_states, cond_embeds)
            attention_output = self.crossattention(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
            if self.use_normformer:
                attention_output = self.post_crossattn_layer_norm(attention_output)
            hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = self.ffn(hidden_states, cond_embeds=cond_embeds)
        hidden_states = residual + hidden_states
        return hidden_states


class Embed(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size,
        hidden_size,
        hidden_dropout=0.0,
        max_position_embeddings=512,
        norm_type="layernorm",
        layer_norm_eps=1e-5,
        use_bias=False,
        layer_norm_embedddings=False,
        use_embeddings_project=False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.hidden_dropout = hidden_dropout
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_embedddings = layer_norm_embedddings
        self.use_embeddings_project = use_embeddings_project

        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.position_embeddings = nn.Embedding(self.max_position_embeddings, self.embedding_size)
        self.dropout = nn.Dropout(self.hidden_dropout)

        if layer_norm_embedddings:
            norm_cls = partial(LayerNorm, use_bias=use_bias) if norm_type == "layernorm" else RMSNorm
            self.embeddings_ln = norm_cls(self.embedding_size, eps=layer_norm_eps)

        if use_embeddings_project:
            self.embedding_hidden_mapping = nn.Linear(self.embedding_size, self.hidden_size, bias=use_bias)

    def forward(self, input_ids):
        seq_length = input_ids.shape[-1]
        position_ids = torch.arange(seq_length)[None, :].to(input_ids.device)

        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        input_embeddings = word_embeddings + position_embeddings

        if self.layer_norm_embedddings:
            input_embeddings = self.embeddings_ln(input_embeddings)

        if self.use_embeddings_project:
            input_embeddings = self.embedding_hidden_mapping(input_embeddings)

        input_embeddings = self.dropout(input_embeddings)
        return input_embeddings


class MlmLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        vocab_size,
        norm_type="layernorm",
        layer_norm_eps=1e-5,
        use_mlm_layernorm=True,
        use_bias=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_mlm_layernorm = use_mlm_layernorm
        self.mlm_dense = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)
        if use_mlm_layernorm:
            norm_cls = partial(LayerNorm, use_bias=use_bias) if norm_type == "layernorm" else RMSNorm
            self.mlm_ln = norm_cls(self.hidden_size, eps=layer_norm_eps)
        self.to_logits = nn.Linear(self.hidden_size, vocab_size, bias=use_bias)

    def forward(self, hidden_states):
        hidden_states = self.mlm_dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        if self.use_mlm_layernorm:
            hidden_states = self.mlm_ln(hidden_states)
        logits = self.to_logits(hidden_states)
        return logits


class ConvEmbed(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size,
        hidden_size,
        patch_size=2,
        max_position_embeddings=256,
        norm_type="layernorm",
        ln_elementwise_affine=True,
        layer_norm_embedddings=False,
        layer_norm_eps=1e-5,
        use_position_embeddings=True,
        use_quant_embeds=False,
        quant_embed_dim=None,
        use_bias=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.max_position_embeddings = max_position_embeddings
        self.use_position_embeddings = use_position_embeddings
        self.layer_norm_embedddings = layer_norm_embedddings

        norm_cls = partial(LayerNorm, use_bias=use_bias) if norm_type == "layernorm" else RMSNorm

        if not use_quant_embeds:
            self.embeddings = nn.Embedding(vocab_size, embedding_size)
            self.layer_norm = norm_cls(embedding_size, eps=layer_norm_eps, elementwise_affine=ln_elementwise_affine)
        else:
            self.quant_embeds_proj = nn.Conv2d(quant_embed_dim, embedding_size, kernel_size=1, bias=use_bias)
            self.layer_norm = Norm2D(
                embedding_size, eps=layer_norm_eps, norm_type=norm_type, elementwise_affine=ln_elementwise_affine
            )

        if patch_size > 1:
            self.pixel_unshuffle = nn.PixelUnshuffle(patch_size)

        self.conv = nn.Conv2d(embedding_size * (patch_size**2), hidden_size, kernel_size=1, bias=use_bias)

        if use_position_embeddings:
            self.position_embeddings = nn.Embedding(self.max_position_embeddings, hidden_size)

        if self.layer_norm_embedddings:
            self.embeddings_ln = Norm2D(
                hidden_size, eps=layer_norm_eps, norm_type=norm_type, elementwise_affine=ln_elementwise_affine
            )

    def forward(self, input_ids, quant_states=None):
        if quant_states is None:
            batch_size, seq_length = input_ids.shape
            height, width = int(seq_length**0.5), int(seq_length**0.5)
            input_ids = input_ids.view(-1, height, width)
            embeddings = self.embeddings(input_ids)
            embeddings = self.layer_norm(embeddings)
            embeddings = embeddings.permute(0, 3, 1, 2)
        else:
            embeddings = self.quant_embeds_proj(quant_states)
            embeddings = self.layer_norm(embeddings)
        if self.patch_size > 1:
            embeddings = self.pixel_unshuffle(embeddings)
        embeddings = self.conv(embeddings)
        if self.use_position_embeddings:
            embeddings = embeddings.permute(0, 2, 3, 1).view(batch_size, -1, self.hidden_size)
            position_ids = torch.arange(embeddings.shape[1])[None, :].to(input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        if self.layer_norm_embedddings:
            embeddings = self.embeddings_ln(embeddings)
        return embeddings


class ConvMlmLayer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size,
        hidden_size,
        patch_size=2,
        norm_type="layernorm",
        ln_elementwise_affine=True,
        layer_norm_eps=1e-5,
        use_bias=False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(hidden_size, embedding_size * (patch_size**2), kernel_size=1, bias=use_bias)
        if patch_size > 1:
            self.pixel_shuffle = nn.PixelShuffle(patch_size)
        self.layer_norm = Norm2D(
            embedding_size,
            norm_type=norm_type,
            eps=layer_norm_eps,
            use_bias=use_bias,
            elementwise_affine=ln_elementwise_affine,
        )
        self.conv2 = nn.Conv2d(embedding_size, vocab_size, kernel_size=1, bias=use_bias)

    def forward(self, hidden_states):
        batch_size, seq_length, hidden_size = hidden_states.shape
        height, width = int(seq_length**0.5), int(seq_length**0.5)
        hidden_states = hidden_states.view(batch_size, height, width, hidden_size).permute(0, 3, 1, 2)
        hidden_states = self.conv1(hidden_states)
        if self.patch_size > 1:
            hidden_states = self.pixel_shuffle(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = self.conv2(hidden_states)
        logits = logits.permute(0, 2, 3, 1).view(batch_size, -1, self.vocab_size)
        return logits


class MaskGitTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        vocab_size,  # codebook_size + 1 (for the mask token), for class-conditioned generation it'll be codebook_size + num_classes + 1
        hidden_size=768,
        embedding_size=None,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=256,  # for clas-conditioned generation it'll be 256 + 1 (for the class token)
        add_cross_attention=False,
        encoder_hidden_size=1024,  # T5-large
        project_encoder_hidden_states=False,
        initializer_range=0.02,
        norm_type="layernorm",  # or rmsnorm
        layer_norm_eps=1e-5,
        use_normformer=True,
        use_encoder_layernorm=True,
        use_mlm_layer=True,
        use_mlm_layernorm=True,
        use_bias=False,
        codebook_size=1024,
        num_vq_tokens=256,
        num_classes=None,  # set for class-conditioned generation
        use_codebook_size_for_output=False,
        use_conv_in_out=False,
        patch_size=1,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.embedding_size = embedding_size or hidden_size
        self.register_to_config(mask_token_id=vocab_size - 1)

        norm_cls = partial(LayerNorm, use_bias=use_bias) if norm_type == "layernorm" else RMSNorm

        if use_conv_in_out:
            self.embed = ConvEmbed(
                vocab_size,
                embedding_size,
                hidden_size,
                patch_size=patch_size,
                norm_type=norm_type,
                layer_norm_eps=layer_norm_eps,
                use_bias=use_bias,
            )
        else:
            self.embed = Embed(
                self.vocab_size,
                self.hidden_size,
                self.hidden_size,
                self.hidden_dropout,
                self.max_position_embeddings,
                use_bias,
                norm_type,
                layer_norm_eps,
            )

        if add_cross_attention is not None and project_encoder_hidden_states:  # Cross attention
            self.encoder_proj = nn.Linear(encoder_hidden_size, hidden_size, bias=use_bias)
            self.encoder_proj_layer_norm = norm_cls(hidden_size, eps=layer_norm_eps)
            encoder_hidden_size = hidden_size

        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(
                    hidden_size=self.hidden_size,
                    intermediate_size=self.intermediate_size,
                    num_attention_heads=self.num_attention_heads,
                    encoder_hidden_size=encoder_hidden_size,
                    add_cross_attention=add_cross_attention,
                    hidden_dropout=self.hidden_dropout,
                    attention_dropout=self.attention_dropout,
                    norm_type=norm_type,
                    layer_norm_eps=layer_norm_eps,
                    use_normformer=use_normformer,
                    use_bias=use_bias,
                )
                for _ in range(self.num_hidden_layers)
            ]
        )
        if use_encoder_layernorm:
            self.encoder_layer_norm = norm_cls(self.hidden_size, eps=layer_norm_eps)

        self.output_size = codebook_size if use_codebook_size_for_output else self.vocab_size
        if use_mlm_layer:
            if use_conv_in_out:
                self.mlm_layer = ConvMlmLayer(
                    self.output_size,
                    embedding_size,
                    hidden_size,
                    patch_size=patch_size,
                    norm_type=norm_type,
                    layer_norm_eps=layer_norm_eps,
                    use_bias=use_bias,
                )
            else:
                self.mlm_layer = MlmLayer(
                    self.hidden_size, self.output_size, norm_type, layer_norm_eps, use_mlm_layernorm, use_bias
                )
        else:
            self.to_logits = nn.Linear(self.hidden_size, self.output_size, bias=use_bias)

        self.gradient_checkpointing = False

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize the weights according to the original implementation.
        https://github.com/google-research/maskgit/blob/main/maskgit/nets/maskgit_transformer.py#L37
        """
        # TODO: make this configurable
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True

    def forward(
        self,
        input_ids,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        label_smoothing=0.0,
        cond_dropout_prob=0.0,
        **kwargs,
    ):
        if self.config.add_cross_attention and encoder_hidden_states is None:
            raise ValueError("If `add_cross_attention` is True, `encoder_hidden_states` should be provided.")

        hidden_states = self.embed(input_ids)

        if encoder_hidden_states is not None and self.config.project_encoder_hidden_states:
            encoder_hidden_states = self.encoder_proj(encoder_hidden_states)
            encoder_hidden_states = self.encoder_proj_layer_norm(encoder_hidden_states)

        # condition dropout for classifier free guidance
        if encoder_hidden_states is not None and self.training and cond_dropout_prob > 0.0:
            batch_size = encoder_hidden_states.shape[0]
            mask = prob_mask_like((batch_size, 1, 1), 1.0 - cond_dropout_prob, encoder_hidden_states.device)
            encoder_hidden_states = encoder_hidden_states * mask

        for layer in self.transformer_layers:
            if self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = checkpoint(
                    create_custom_forward(layer), hidden_states, encoder_hidden_states, encoder_attention_mask
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                )

        if self.config.use_encoder_layernorm:
            hidden_states = self.encoder_layer_norm(hidden_states)

        if self.config.use_mlm_layer:
            logits = self.mlm_layer(hidden_states)
        else:
            logits = self.to_logits(hidden_states)

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.output_size), labels.view(-1), ignore_index=-100, label_smoothing=label_smoothing
            )
            return logits, loss
        return logits

    def generate(
        self,
        input_ids: torch.LongTensor = None,
        class_ids: torch.LongTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        temperature=1.0,
        topk_filter_thres=0.9,
        can_remask_prev_masked=False,  # TODO: implement this
        timesteps=18,  # ideal number of steps is 18 in maskgit paper
        guidance_scale=3,
        noise_schedule: Callable = cosine_schedule,
        use_tqdm=True,
    ):
        # begin with all image token ids masked
        mask_token_id = self.config.mask_token_id
        seq_len = self.config.num_vq_tokens

        batch_size = len(class_ids) if class_ids is not None else encoder_hidden_states.shape[0]
        shape = (batch_size, seq_len)

        # shift the class ids by the codebook size
        if class_ids is not None:
            class_ids += self.config.codebook_size
        # initialize with all image tokens masked
        if input_ids is not None:
            input_ids = torch.ones(shape, dtype=torch.long, device=self.device) * mask_token_id
        scores = torch.zeros(shape, dtype=torch.float32, device=self.device)

        starting_temperature = temperature

        iterate_over = zip(torch.linspace(0, 1, timesteps, device=self.device), reversed(range(timesteps)))

        if use_tqdm:
            iterate_over = tqdm(iterate_over, total=timesteps)

        for timestep, steps_until_x0 in iterate_over:
            rand_mask_prob = noise_schedule(timestep)
            num_token_masked = max(int((rand_mask_prob * seq_len).item()), 1)

            masked_indices = scores.topk(num_token_masked, dim=-1).indices
            input_ids = input_ids.scatter(1, masked_indices, mask_token_id)

            # prepend class token to input_ids
            if class_ids is not None:
                input_ids = torch.cat([class_ids[:, None], input_ids], dim=1)

            # classifier free guidance
            if encoder_hidden_states is not None and guidance_scale > 0:
                uncond_encoder_states = torch.zeros_like(encoder_hidden_states)
                model_input = torch.cat([input_ids] * 2)
                condition = torch.cat([encoder_hidden_states, uncond_encoder_states])
                cond_logits, uncond_logits = self(model_input, encoder_hidden_states=condition).chunk(2)
                cond_logits = cond_logits[..., : self.config.codebook_size]
                uncond_logits = uncond_logits[..., : self.config.codebook_size]
                logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
            else:
                logits = self(input_ids, encoder_hidden_states=encoder_hidden_states)
                logits = logits[..., : self.config.codebook_size]

            # remove class token
            if class_ids is not None:
                input_ids = input_ids[:, 1:]
                logits = logits[:, 1:]

            filtered_logits = top_k(logits, topk_filter_thres)

            temperature = starting_temperature * (steps_until_x0 / timesteps)  # temperature is annealed

            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

            is_mask = input_ids == mask_token_id

            input_ids = torch.where(is_mask, pred_ids, input_ids)

            probs_without_temperature = F.softmax(logits, dim=-1)

            scores = 1 - probs_without_temperature.gather(2, pred_ids[..., None])
            scores = rearrange(scores, "... 1 -> ...")  # TODO: use torch
        return input_ids

    def generate2(
        self,
        input_ids: torch.LongTensor = None,
        class_ids: torch.LongTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        negative_embeds: torch.FloatTensor = None,
        temperature=1.0,
        timesteps=18,  # ideal number of steps is 18 in maskgit paper
        guidance_scale=0,
        noise_schedule=cosine_schedule,
        generator: torch.Generator = None,
        **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """
        # begin with all image token ids masked
        mask_token_id = self.config.mask_token_id
        seq_len = self.config.num_vq_tokens

        batch_size = len(class_ids) if class_ids is not None else encoder_hidden_states.shape[0]
        shape = (batch_size, seq_len)

        # shift the class ids by the codebook size
        if class_ids is not None:
            class_ids += self.config.codebook_size

        # initialize with all image tokens masked
        if input_ids is None:
            input_ids = torch.ones(shape, dtype=torch.long, device=self.device) * mask_token_id

        # classifier free guidance
        if encoder_hidden_states is not None and guidance_scale > 0:
            if negative_embeds is None:
                uncond_encoder_states = torch.zeros_like(encoder_hidden_states)
            else:
                uncond_encoder_states = negative_embeds
            condition = torch.cat([encoder_hidden_states, uncond_encoder_states])
            model_conds = {"encoder_hidden_states": condition}

        for step in range(timesteps):
            # prepend class token to input_ids
            if class_ids is not None:
                input_ids = torch.cat([class_ids[:, None], input_ids], dim=1)

            if encoder_hidden_states is not None and guidance_scale > 0:
                model_input = torch.cat([input_ids] * 2)
                cond_logits, uncond_logits = self(model_input, **model_conds).chunk(2)
                cond_logits = cond_logits[..., : self.config.codebook_size]
                uncond_logits = uncond_logits[..., : self.config.codebook_size]
                logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
            else:
                logits = self(input_ids, encoder_hidden_states=encoder_hidden_states)
                logits = logits[..., : self.config.codebook_size]

            # remove class token
            if class_ids is not None:
                input_ids = input_ids[:, 1:]
                logits = logits[:, 1:]

            # Samples the ids using categorical sampling: [batch_size, seq_length].
            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])

            # Just updates the masked tokens.
            unknown_map = input_ids == mask_token_id
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (seq_len * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )

            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            input_ids = torch.where(masking, mask_token_id, sampled_ids)

        return sampled_ids


class MaskGiTUViT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        vocab_size,  # codebook_size + 1 (for the mask token), for class-conditioned generation it'll be codebook_size + num_classes + 1
        hidden_size=768,
        in_channels=384,
        block_out_channels=(768, 768),
        block_has_attention=None,
        block_num_heads=None,
        num_res_blocks=2,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=256,  # for clas-conditioned generation it'll be 256 + 1 (for the class token)
        add_cross_attention=False,
        encoder_hidden_size=1024,  # T5-large
        project_encoder_hidden_states=False,
        initializer_range=0.02,
        norm_type="layernorm",  # or rmsnorm
        ln_elementwise_affine=True,
        layer_norm_eps=1e-5,
        use_normformer=False,
        use_encoder_layernorm=True,
        use_bias=False,
        codebook_size=1024,
        num_vq_tokens=256,
        num_classes=None,  # set for class-conditioned generation
        use_position_embeddings=False,
        use_codebook_size_for_output=False,
        patch_size=1,
        layer_norm_before_mlm=False,
        layer_norm_embedddings=False,
        add_cond_embeds=False,
        cond_embed_dim=None,
        xavier_init_embed=True,
        use_empty_embeds_for_uncond=False,
        learn_uncond_embeds=False,
        use_vannilla_resblock=False,
        ffn_type="glu",
        use_quant_embeds=False,
        quant_embed_dim=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.use_projection = block_out_channels[-1] != hidden_size
        self.register_to_config(mask_token_id=vocab_size - 1)
        self.register_to_config(block_out_channels=tuple(block_out_channels))

        norm_cls = partial(LayerNorm, use_bias=use_bias) if norm_type == "layernorm" else RMSNorm

        if block_has_attention is None:
            block_has_attention = [False] * len(block_out_channels)
        elif isinstance(block_has_attention, bool):
            block_has_attention = [block_has_attention] * len(block_out_channels)

        if block_num_heads is None:
            block_num_heads = [None] * len(block_out_channels)
        elif isinstance(block_num_heads, int):
            block_num_heads = [block_num_heads] * len(block_out_channels)

        self.register_to_config(block_has_attention=tuple(block_has_attention))
        self.register_to_config(block_num_heads=tuple(block_num_heads))

        self.register_to_config(block_has_attention=tuple(block_has_attention))
        self.register_to_config(block_num_heads=tuple(block_num_heads))

        if learn_uncond_embeds:
            self.uncond_embeds = nn.Parameter(torch.randn(size=(77, encoder_hidden_size), requires_grad=True))
            nn.init.normal_(self.uncond_embeds, std=0.02)

        if add_cross_attention is not None and project_encoder_hidden_states:  # Cross attention
            self.encoder_proj = nn.Linear(encoder_hidden_size, hidden_size, bias=use_bias)
            self.encoder_proj_layer_norm = norm_cls(
                hidden_size, eps=layer_norm_eps, elementwise_affine=ln_elementwise_affine
            )
            encoder_hidden_size = hidden_size

        # Embeddings
        self.embed = ConvEmbed(
            vocab_size,
            in_channels,
            block_out_channels[0],
            patch_size=patch_size,
            norm_type=norm_type,
            layer_norm_embedddings=layer_norm_embedddings,
            layer_norm_eps=layer_norm_eps,
            ln_elementwise_affine=ln_elementwise_affine,
            use_position_embeddings=use_position_embeddings,
            use_quant_embeds=use_quant_embeds,
            quant_embed_dim=quant_embed_dim,
            use_bias=use_bias,
        )

        if use_quant_embeds:
            self.mask_embeddings = nn.Parameter(torch.randn(size=(quant_embed_dim,), requires_grad=True))
            nn.init.normal_(self.mask_embeddings, std=0.02)

        # Condition embeddings
        if add_cond_embeds:
            self.cond_embed = nn.Sequential(
                nn.Linear(cond_embed_dim, hidden_size, bias=use_bias),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=use_bias),
            )
            cond_embed_dim = hidden_size

        # Downsample
        DownBlock = DownsampleBlockVanilla if use_vannilla_resblock else DownsampleBlock
        output_channels = block_out_channels[0]
        self.down_blocks = nn.ModuleList([])
        for i in range(len(block_out_channels)):
            is_first_block = i == 0
            is_final_block = i == len(block_out_channels) - 1
            input_channels = output_channels
            output_channels = block_out_channels[i]

            if use_vannilla_resblock:
                add_downsample = not is_final_block
            else:
                add_downsample = not is_first_block

            self.down_blocks.append(
                DownBlock(
                    input_channels=input_channels,
                    output_channels=output_channels,
                    skip_channels=0,
                    num_res_blocks=num_res_blocks,
                    kernel_size=3,
                    dropout=hidden_dropout if i == 0 else 0.0,
                    norm_type=norm_type,
                    ln_elementwise_affine=ln_elementwise_affine,
                    add_downsample=add_downsample,
                    add_cond_embeds=add_cond_embeds,
                    cond_embed_dim=cond_embed_dim,
                    has_attention=block_has_attention[i],
                    num_heads=block_num_heads[i],
                    encoder_hidden_size=encoder_hidden_size,
                    use_bias=use_bias,
                )
            )

        if self.use_projection:
            self.project_to_hidden_norm = norm_cls(
                block_out_channels[-1], eps=layer_norm_eps, elementwise_affine=ln_elementwise_affine
            )
            self.project_to_hidden = nn.Linear(block_out_channels[-1], hidden_size, bias=use_bias)

        # Mid Transformer
        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(
                    hidden_size=self.hidden_size,
                    intermediate_size=self.intermediate_size,
                    num_attention_heads=self.num_attention_heads,
                    encoder_hidden_size=encoder_hidden_size,
                    add_cross_attention=add_cross_attention,
                    hidden_dropout=self.hidden_dropout,
                    attention_dropout=self.attention_dropout,
                    norm_type=norm_type,
                    ln_elementwise_affine=ln_elementwise_affine,
                    layer_norm_eps=layer_norm_eps,
                    use_normformer=use_normformer,
                    add_cond_embeds=add_cond_embeds,
                    cond_embed_dim=cond_embed_dim,
                    ffn_type=ffn_type,
                    use_bias=use_bias,
                )
                for _ in range(self.num_hidden_layers)
            ]
        )
        if use_encoder_layernorm:
            self.encoder_layer_norm = norm_cls(
                self.hidden_size, eps=layer_norm_eps, elementwise_affine=ln_elementwise_affine
            )

        if self.use_projection:
            self.project_from_hidden_norm = norm_cls(
                hidden_size, eps=layer_norm_eps, elementwise_affine=ln_elementwise_affine
            )
            self.project_from_hidden = nn.Linear(hidden_size, block_out_channels[-1], bias=use_bias)

        # Up sample
        UpBlock = UpsampleBlockVanilla if use_vannilla_resblock else UpsampleBlock
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_block_has_attention = list(reversed(block_has_attention))
        reversed_block_num_heads = list(reversed(block_num_heads))
        output_channels = reversed_block_out_channels[0]
        self.up_blocks = nn.ModuleList([])
        for i in range(len(reversed_block_out_channels)):
            is_final_block = i == len(block_out_channels) - 1

            if use_vannilla_resblock:
                prev_output_channels = output_channels
                output_channels = reversed_block_out_channels[i]
                input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
            else:
                input_channel = reversed_block_out_channels[i]
                output_channels = reversed_block_out_channels[i + 1] if not is_final_block else output_channels
                prev_output_channels = input_channel if i != 0 else 0

            self.up_blocks.append(
                UpBlock(
                    input_channels=input_channel,
                    skip_channels=prev_output_channels,
                    output_channels=output_channels,
                    num_res_blocks=num_res_blocks + 1 if use_vannilla_resblock else num_res_blocks,
                    kernel_size=3,
                    dropout=hidden_dropout if i == 0 else 0.0,
                    norm_type=norm_type,
                    ln_elementwise_affine=ln_elementwise_affine,
                    add_upsample=not is_final_block,
                    add_cond_embeds=add_cond_embeds,
                    cond_embed_dim=cond_embed_dim,
                    has_attention=reversed_block_has_attention[i],
                    num_heads=reversed_block_num_heads[i],
                    encoder_hidden_size=encoder_hidden_size,
                    use_bias=use_bias,
                )
            )

        if layer_norm_before_mlm:
            self.layer_norm_before_mlm = Norm2D(
                block_out_channels[0],
                norm_type=norm_type,
                eps=layer_norm_eps,
                elementwise_affine=ln_elementwise_affine,
            )

        # Output
        self.output_size = codebook_size if use_codebook_size_for_output else self.vocab_size
        self.mlm_layer = ConvMlmLayer(
            self.output_size,
            in_channels,
            block_out_channels[0],
            patch_size=patch_size,
            norm_type=norm_type,
            ln_elementwise_affine=ln_elementwise_affine,
            layer_norm_eps=layer_norm_eps,
            use_bias=use_bias,
        )
        self.gradient_checkpointing = False

        # --- WEIGHT INIT ---
        self.apply(self._init_weights)  # General init
        if xavier_init_embed and not use_quant_embeds:
            nn.init.xavier_uniform_(self.embed.conv.weight, 0.02)  # inputs
        nn.init.constant_(self.mlm_layer.conv1.weight, 0)  # output
        if not use_quant_embeds:
            nn.init.normal_(self.embed.embeddings.weight, std=np.sqrt(1 / vocab_size))
            self.mlm_layer.conv2.weight.data = self.embed.embeddings.weight.data[:codebook_size, :, None, None].clone()

        # init AdaLNModulation.mapper layers to 0
        if add_cond_embeds:
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
        # TODO: make this configurable
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        label_smoothing=0.0,
        cond_dropout_prob=0.0,
        cond_embeds=None,
        loss_weight=None,
        empty_embeds=None,
        quant_embeds=None,
        quant_embeds_mask=None,
    ):
        if input_ids is None and quant_embeds is None:
            raise ValueError("Either `input_ids` or `quant_embeds` should be provided.")
        elif input_ids is not None and quant_embeds is not None:
            raise ValueError("Only one of `input_ids` or `quant_embeds` should be provided.")

        if quant_embeds is not None and quant_embeds_mask is None:
            raise ValueError("If `quant_embeds` is provided, `quant_embeds_mask` should be provided.")

        if self.config.add_cross_attention and encoder_hidden_states is None:
            raise ValueError("If `add_cross_attention` is True, `encoder_hidden_states` should be provided.")

        if self.training and self.config.use_empty_embeds_for_uncond and empty_embeds is None:
            raise ValueError("If `use_empty_embeds_for_uncond` is True, `empty_embeds` should be provided.")

        # condition dropout for classifier free guidance
        if encoder_hidden_states is not None and self.training and cond_dropout_prob > 0.0:
            batch_size = encoder_hidden_states.shape[0]
            mask = prob_mask_like((batch_size, 1, 1), 1.0 - cond_dropout_prob, encoder_hidden_states.device)

            if self.config.use_empty_embeds_for_uncond:
                # empty embeds is of shape (1, seq, hidden_size) expand it to batch size
                empty_embeds = empty_embeds.expand(batch_size, -1, -1)
                encoder_hidden_states = torch.where(
                    (encoder_hidden_states * mask).bool(), encoder_hidden_states, empty_embeds
                )
            elif self.config.learn_uncond_embeds:
                uncond_embeds = self.uncond_embeds.unsqueeze(0).expand(batch_size, -1, -1)
                encoder_hidden_states = torch.where(
                    (encoder_hidden_states * mask).bool(), encoder_hidden_states, uncond_embeds
                )
            else:
                encoder_hidden_states = encoder_hidden_states * mask
            if cond_embeds is not None:
                cond_embeds = cond_embeds * mask.squeeze(-1)

        if encoder_hidden_states is not None and self.config.project_encoder_hidden_states:
            encoder_hidden_states = self.encoder_proj(encoder_hidden_states)
            encoder_hidden_states = self.encoder_proj_layer_norm(encoder_hidden_states)

        if cond_embeds is not None:
            cond_embeds = self.cond_embed(cond_embeds)

        if quant_embeds is not None:
            # add mask_embeddings in quant_embeds using mask
            # mask is of shape (batch_size, seq_len)
            # quant_embeds is of shape (batch_size, quant_embed_dim, height, width)
            # mask_embeddings is of shape (quant_embed_dim,)
            batch_size, seq_len = quant_embeds_mask.shape
            quant_embeds_mask = quant_embeds_mask.view(batch_size, int(seq_len**0.5), int(seq_len**0.5))
            mask_embeddings = self.mask_embeddings.view(1, -1, 1, 1).expand_as(quant_embeds)
            quant_embeds = torch.where(quant_embeds_mask.unsqueeze(1).bool(), mask_embeddings, quant_embeds)

        hidden_states = self.embed(input_ids, quant_states=quant_embeds)

        down_block_res_samples = (hidden_states,)
        for down_block in self.down_blocks:
            hidden_states, res_samples = down_block(
                hidden_states, cond_embeds=cond_embeds, encoder_hidden_states=encoder_hidden_states
            )
            down_block_res_samples += res_samples

        batch_size, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)

        if self.use_projection:
            hidden_states = self.project_to_hidden_norm(hidden_states)
            hidden_states = self.project_to_hidden(hidden_states)

        for layer in self.transformer_layers:
            if self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    cond_embeds,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    cond_embeds=cond_embeds,
                )

        if self.config.use_encoder_layernorm:
            hidden_states = self.encoder_layer_norm(hidden_states)

        if self.use_projection:
            hidden_states = self.project_from_hidden_norm(hidden_states)
            hidden_states = self.project_from_hidden(hidden_states)

        hidden_states = hidden_states.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)

        for i, up_block in enumerate(self.up_blocks):
            num_up_blocks = len(up_block.res_blocks)
            res_samples = down_block_res_samples[-num_up_blocks:]
            down_block_res_samples = down_block_res_samples[:-num_up_blocks]

            if self.config.use_vannilla_resblock:
                x_skip = res_samples
            else:
                x_skip = res_samples if i > 0 else None

            hidden_states = up_block(
                hidden_states, x_skip=x_skip, cond_embeds=cond_embeds, encoder_hidden_states=encoder_hidden_states
            )

        if self.config.layer_norm_before_mlm:
            hidden_states = self.layer_norm_before_mlm(hidden_states)

        batch_size, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        logits = self.mlm_layer(hidden_states)

        if labels is not None:
            reduction = "none" if loss_weight is not None else "mean"
            loss = F.cross_entropy(
                logits.view(-1, self.output_size),
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
        self.gradient_checkpointing = True
        if isinstance(module, (DownsampleBlock, UpsampleBlock)):
            module.gradient_checkpointing = value

    def generate(
        self,
        input_ids: torch.LongTensor = None,
        class_ids: torch.LongTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        negative_embeds: torch.FloatTensor = None,
        empty_embeds: torch.FloatTensor = None,
        temperature=1.0,
        topk_filter_thres=0.9,
        can_remask_prev_masked=False,  # TODO: implement this
        timesteps=18,  # ideal number of steps is 18 in maskgit paper
        guidance_scale=8,
        guidance_schedule=None,
        noise_schedule: Callable = cosine_schedule,
        use_tqdm=True,
        **kwargs,
    ):
        # begin with all image token ids masked
        mask_token_id = self.config.mask_token_id
        seq_len = self.config.num_vq_tokens

        batch_size = len(class_ids) if class_ids is not None else encoder_hidden_states.shape[0]
        shape = (batch_size, seq_len)

        # shift the class ids by the codebook size
        if class_ids is not None:
            class_ids += self.config.codebook_size
        # initialize with all image tokens masked
        if input_ids is not None:
            input_ids = torch.ones(shape, dtype=torch.long, device=self.device) * mask_token_id

        if isinstance(temperature, tuple):
            temperatures = torch.linspace(temperature[0], temperature[1], timesteps)
        else:
            temperatures = torch.linspace(temperature, 0.01, timesteps)
        # reverse the temperatures
        temperatures = temperatures.flip(0)

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
        # reverse the guidance scales
        guidance_scales = guidance_scales.flip(0)

        # classifier free guidance
        if encoder_hidden_states is not None and guidance_scale > 0:
            if negative_embeds is None:
                if self.config.use_empty_embeds_for_uncond:
                    uncond_encoder_states = empty_embeds.expand(batch_size, -1, -1)
                elif self.config.learn_uncond_embeds:
                    uncond_encoder_states = self.uncond_embeds.unsqueeze(0).expand(batch_size, -1, -1)
                else:
                    uncond_encoder_states = torch.zeros_like(encoder_hidden_states)
            else:
                uncond_encoder_states = negative_embeds
            condition = torch.cat([encoder_hidden_states, uncond_encoder_states])
            model_conds = {"encoder_hidden_states": condition}

            if cond_embeds is not None:
                uncond_embeds = torch.zeros_like(cond_embeds)
                cond_embeds = torch.cat([cond_embeds, uncond_embeds])
                model_conds["cond_embeds"] = cond_embeds

        scores = torch.zeros(shape, dtype=torch.float32, device=self.device)

        starting_temperature = temperature

        iterate_over = zip(torch.linspace(0, 1, timesteps, device=self.device), reversed(range(timesteps)))

        if use_tqdm:
            iterate_over = tqdm(iterate_over, total=timesteps)

        for timestep, steps_until_x0 in iterate_over:
            rand_mask_prob = noise_schedule(timestep)
            num_token_masked = max(int((rand_mask_prob * seq_len).item()), 1)

            masked_indices = scores.topk(num_token_masked, dim=-1).indices
            input_ids = input_ids.scatter(1, masked_indices, mask_token_id)

            # prepend class token to input_ids
            if class_ids is not None:
                input_ids = torch.cat([class_ids[:, None], input_ids], dim=1)

            # classifier free guidance
            if encoder_hidden_states is not None and guidance_scale > 0:
                model_input = torch.cat([input_ids] * 2)
                cond_logits, uncond_logits = self(model_input, **model_conds).chunk(2)
                cond_logits = cond_logits[..., : self.config.codebook_size]
                uncond_logits = uncond_logits[..., : self.config.codebook_size]
                logits = uncond_logits + guidance_scales[steps_until_x0] * (cond_logits - uncond_logits)
            else:
                logits = self(input_ids, encoder_hidden_states=encoder_hidden_states)
                logits = logits[..., : self.config.codebook_size]

            # remove class token
            if class_ids is not None:
                input_ids = input_ids[:, 1:]
                logits = logits[:, 1:]

            filtered_logits = top_k(logits, topk_filter_thres)

            temperature = temperatures[steps_until_x0]

            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

            is_mask = input_ids == mask_token_id

            input_ids = torch.where(is_mask, pred_ids, input_ids)

            probs_without_temperature = F.softmax(logits, dim=-1)

            scores = 1 - probs_without_temperature.gather(2, pred_ids[..., None])
            scores = rearrange(scores, "... 1 -> ...")  # TODO: use torch
        return input_ids

    def generate2(
        self,
        input_ids: torch.LongTensor = None,
        class_ids: torch.LongTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        cond_embeds: torch.FloatTensor = None,
        empty_embeds: torch.FloatTensor = None,
        negative_embeds: torch.FloatTensor = None,
        temperature=1.0,
        timesteps=18,  # ideal number of steps is 18 in maskgit paper
        guidance_scale=0,
        guidance_schedule=None,
        noise_schedule=cosine_schedule,
        noise_type="mask",  # can be "mask" or "random_replace"
        predict_all_tokens=False,
        get_quant_embeds: Callable = None,
        generator: torch.Generator = None,
        return_intermediate=False,
        **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """
        # begin with all image token ids masked
        mask_token_id = self.config.mask_token_id
        seq_len = self.config.num_vq_tokens

        batch_size = len(class_ids) if class_ids is not None else encoder_hidden_states.shape[0]
        shape = (batch_size, seq_len)

        if isinstance(temperature, tuple):
            temperatures = torch.linspace(temperature[0], temperature[1], timesteps)
        else:
            temperatures = torch.linspace(temperature, 0.01, timesteps)

        # shift the class ids by the codebook size
        if class_ids is not None:
            class_ids += self.config.codebook_size

        if input_ids is None:
            # initialize with all image tokens masked
            if noise_type == "mask":
                input_ids = torch.ones(shape, dtype=torch.long, device=self.device) * mask_token_id
            elif noise_type == "random_replace":
                input_ids = torch.randint(0, self.config.codebook_size, shape, device=self.device)
            else:
                raise ValueError(f"noise_type {noise_type} not recognized")

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

        # classifier free guidance
        if encoder_hidden_states is not None and guidance_scale > 0:
            if negative_embeds is None:
                if self.config.use_empty_embeds_for_uncond:
                    uncond_encoder_states = empty_embeds.expand(batch_size, -1, -1)
                elif self.config.learn_uncond_embeds:
                    uncond_encoder_states = self.uncond_embeds.unsqueeze(0).expand(batch_size, -1, -1)
                else:
                    uncond_encoder_states = torch.zeros_like(encoder_hidden_states)
            else:
                uncond_encoder_states = negative_embeds
            condition = torch.cat([encoder_hidden_states, uncond_encoder_states])
            model_conds = {"encoder_hidden_states": condition}

            if cond_embeds is not None:
                uncond_embeds = torch.zeros_like(cond_embeds)
                cond_embeds = torch.cat([cond_embeds, uncond_embeds])
                model_conds["cond_embeds"] = cond_embeds

        if self.config.use_quant_embeds:
            if get_quant_embeds is None:
                raise ValueError("If `use_quant_embeds` is True, `get_quant_embeds` should be provided.")
            size = (batch_size, self.config.quant_embed_dim, int(seq_len**0.5), int(seq_len**0.5))
            quant_embeds = torch.randn(size, device=self.device)
            input_ids_or_quant_embeds = quant_embeds
            mask = torch.zeros(batch_size, seq_len, device=self.device)
        else:
            input_ids_or_quant_embeds = input_ids

        for step in range(timesteps):
            # prepend class token to input_ids
            if class_ids is not None:
                input_ids = torch.cat([class_ids[:, None], input_ids], dim=1)

            # classifier free guidance
            if encoder_hidden_states is not None and guidance_scale > 0:
                model_input = torch.cat([input_ids_or_quant_embeds] * 2)
                if self.config.use_quant_embeds:
                    quant_embeds_mask = torch.cat([mask] * 2)
                    cond_logits, uncond_logits = self(quant_embeds=model_input, quant_embeds_mask=quant_embeds_mask, **model_conds).chunk(2)
                else:
                    cond_logits, uncond_logits = self(model_input, **model_conds).chunk(2)
                cond_logits = cond_logits[..., : self.config.codebook_size]
                uncond_logits = uncond_logits[..., : self.config.codebook_size]
                logits = uncond_logits + guidance_scales[step] * (cond_logits - uncond_logits)
            else:
                logits = self(input_ids, encoder_hidden_states=encoder_hidden_states)
                logits = logits[..., : self.config.codebook_size]

            # remove class token
            if class_ids is not None:
                input_ids = input_ids[:, 1:]
                logits = logits[:, 1:]

            if noise_type == "mask":
                # Samples the ids using categorical sampling: [batch_size, seq_length].
                if predict_all_tokens:
                    probs = logits.div(temperatures[step]).softmax(dim=-1)
                else:
                    probs = logits.softmax(dim=-1)

                sampled = probs.reshape(-1, logits.size(-1))
                sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])

                if return_intermediate:
                    intermediate.append(sampled_ids)

                # Just updates the masked tokens.
                unknown_map = input_ids == mask_token_id
                prev_sampled_ids = sampled_ids.clone()
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

                # Adds noise for randomness
                if not predict_all_tokens:
                    selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
                    selected_probs = selected_probs.squeeze(-1)
                    # Ignores the tokens given in the input by overwriting their confidence.
                    selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
                    temperature = temperatures[step]
                    masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
                    # Masks tokens with lower confidence.
                    if self.config.use_quant_embeds:
                        input_ids_or_quant_embeds = get_quant_embeds(prev_sampled_ids)
                        mask = masking
                        input_ids = torch.where(masking, mask_token_id, sampled_ids)
                    else:
                        input_ids = torch.where(masking, mask_token_id, sampled_ids)
                        input_ids_or_quant_embeds = input_ids
                else:
                    batch_size, seq_len = input_ids.shape
                    batch_randperm = torch.rand(batch_size, seq_len, device=input_ids.device).argsort(dim=-1)
                    mask = batch_randperm < mask_len
                    # mask images and create input and labels
                    input_ids = torch.where(mask, mask_token_id, sampled_ids)
            else:
                # Samples the ids using categorical sampling: [batch_size, seq_length].
                probs = logits.div(temperatures[step]).softmax(dim=-1)
                # probs = logits.softmax(dim=-1)
                sampled = probs.reshape(-1, logits.size(-1))
                sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])

                if return_intermediate:
                    intermediate.append(sampled_ids)

                # Defines the mask ratio for the next round. The number to mask out is
                # determined by mask_ratio * unknown_number_in_the_beginning.
                ratio = 1.0 * (step + 1) / timesteps
                mask_ratio = noise_schedule(torch.tensor(ratio))

                # Adds noise for randomness
                num_token_masked = (seq_len * mask_ratio).round().clamp(min=1).to(sampled_ids.device)
                batch_randperm = torch.rand(batch_size, seq_len, device=sampled_ids.device).argsort(dim=-1)
                mask = batch_randperm < num_token_masked.unsqueeze(-1)
                # sample random tokens from the vocabulary
                random_tokens = torch.randint_like(
                    input_ids, low=0, high=self.config.codebook_size, device=input_ids.device
                )
                input_ids = torch.where(mask, random_tokens, sampled_ids)

        if return_intermediate:
            return sampled_ids, intermediate
        return sampled_ids
