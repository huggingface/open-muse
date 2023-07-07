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
        def __init__(self, normalized_shape, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.variance_epsilon = eps

        def forward(self, input):
            variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)
            input = input * torch.rsqrt(variance + self.variance_epsilon)

            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                input = input.to(self.weight.dtype)

            return self.weight * input


# layer norm without bias
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, use_bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if use_bias else None
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


# U-ViT blocks
# Adpated from https://github.com/dome272/Paella/blob/main/src_distributed/modules.py
class Norm2D(nn.Module):
    def __init__(self, dim, eps=1e-5, use_bias=False, norm_type="layernorm"):
        super().__init__()
        if norm_type == "layernorm":
            self.norm = LayerNorm(dim, eps, use_bias)
        elif norm_type == "rmsnorm":
            self.norm = RMSNorm(dim, eps)

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
        self, channels, skip_channels=None, kernel_size=3, dropout=0.0, norm_type="layernorm", use_bias=False
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels + skip_channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=use_bias,
        )
        self.norm = Norm2D(channels, eps=1e-6, norm_type=norm_type, use_bias=use_bias)
        self.channelwise = nn.Sequential(
            nn.Linear(channels, channels * 4, bias=use_bias),
            nn.GELU(),
            GlobalResponseNorm(channels * 4),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels, bias=use_bias),
        )

    def forward(self, x, x_skip=None):
        x_res = x
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.norm(self.depthwise(x)).permute(0, 2, 3, 1)
        x = self.channelwise(x).permute(0, 3, 1, 2)
        return x + x_res


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
        add_downsample=True,
        use_bias=False,
    ):
        super().__init__()
        self.add_downsample = add_downsample
        if add_downsample:
            self.downsample = nn.Sequential(
                Norm2D(input_channels, eps=1e-6, use_bias=use_bias, norm_type=norm_type),
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
                    use_bias=use_bias,
                )
                for _ in range(num_res_blocks)
            ]
        )

        self.gradient_checkpointing = False

    def forward(self, x, x_skip=None):
        if self.add_downsample:
            x = self.downsample(x)

        output_states = ()
        for res_block in self.res_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                x = torch.utils.checkpoint.checkpoint(create_custom_forward(res_block), x, x_skip)
            else:
                x = res_block(x, x_skip)

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
        add_upsample=True,
        use_bias=False,
    ):
        super().__init__()
        self.add_upsample = add_upsample
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
                    use_bias=use_bias,
                )
                for i in range(num_res_blocks)
            ]
        )

        if add_upsample:
            self.upsample = nn.Sequential(
                Norm2D(self.input_channels, eps=1e-6, norm_type=norm_type, use_bias=use_bias),
                nn.ConvTranspose2d(self.input_channels, self.output_channels, kernel_size=2, stride=2, bias=use_bias),
            )

        self.gradient_checkpointing = False

    def forward(self, x, x_skip=None):
        for i, res_block in enumerate(self.res_blocks):
            x_res = x_skip[0] if i == 0 and x_skip is not None else None

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                x = torch.utils.checkpoint.checkpoint(create_custom_forward(res_block), x, x_res)
            else:
                x = res_block(x, x_res)

        if self.add_upsample:
            x = self.upsample(x)
        return x


# End U-ViT blocks


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, encoder_hidden_size=None, attention_dropout=0.0, use_bias=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
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
            attn_output = xops.memory_efficient_attention(query, key, value, op=self.xformers_attention_op)
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


# Normformer style GLU FeedForward
class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        hidden_dropout=0.0,
        norm_type="layernorm",
        layer_norm_eps=1e-5,
        use_normformer=True,
        use_bias=False,
    ):
        super().__init__()
        self.use_normformer = use_normformer
        self.pre_mlp_layer_norm = LayerNorm(hidden_size, eps=layer_norm_eps, use_bias=use_bias)
        self.wi_0 = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        self.wi_1 = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        if use_normformer:
            norm_cls = partial(LayerNorm, use_bias=use_bias) if norm_type == "layernorm" else RMSNorm
            self.mid_mlp_layer_norm = norm_cls(intermediate_size, eps=layer_norm_eps)
        self.wo = nn.Linear(intermediate_size, hidden_size, bias=use_bias)
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.pre_mlp_layer_norm(hidden_states)

        hidden_gelu = F.gelu(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
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
        use_normformer=True,
        use_bias=False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.use_normformer = use_normformer

        norm_cls = partial(LayerNorm, use_bias=use_bias) if norm_type == "layernorm" else RMSNorm
        self.attn_layer_norm = norm_cls(self.hidden_size, eps=layer_norm_eps)
        self.attention = Attention(
            self.hidden_size, self.num_attention_heads, attention_dropout=attention_dropout, use_bias=use_bias
        )
        if use_normformer:
            self.post_attn_layer_norm = norm_cls(self.hidden_size, eps=layer_norm_eps)
        self.ffn = FeedForward(
            self.hidden_size,
            self.intermediate_size,
            hidden_dropout,
            norm_type,
            layer_norm_eps,
            use_normformer,
            use_bias,
        )

        if add_cross_attention:
            self.crossattn_layer_norm = norm_cls(self.hidden_size, eps=layer_norm_eps)
            self.crossattention = Attention(
                self.hidden_size, self.num_attention_heads, encoder_hidden_size, attention_dropout, use_bias
            )
            if use_normformer:
                self.post_crossattn_layer_norm = norm_cls(self.hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states, encoder_hidden_states=None, encoder_attention_mask=None):
        residual = hidden_states

        hidden_states = self.attn_layer_norm(hidden_states)
        attention_output = self.attention(hidden_states)
        if self.use_normformer:
            attention_output = self.post_attn_layer_norm(attention_output)
        hidden_states = residual + attention_output

        if encoder_hidden_states is not None:
            residual = hidden_states
            # TODO: should norm be applied to encoder_hidden_states as well?
            hidden_states = self.crossattn_layer_norm(hidden_states)
            attention_output = self.crossattention(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
            if self.use_normformer:
                attention_output = self.post_crossattn_layer_norm(attention_output)
            hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = self.ffn(hidden_states)
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
        layer_norm_eps=1e-5,
        use_position_embeddings=True,
        use_bias=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.max_position_embeddings = max_position_embeddings
        self.use_position_embeddings = use_position_embeddings

        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        norm_cls = partial(LayerNorm, use_bias=use_bias) if norm_type == "layernorm" else RMSNorm
        self.layer_norm = norm_cls(embedding_size, eps=layer_norm_eps)
        if patch_size > 1:
            self.pixel_unshuffle = nn.PixelUnshuffle(patch_size)
        self.conv = nn.Conv2d(embedding_size * (patch_size**2), hidden_size, kernel_size=1, bias=use_bias)
        if use_position_embeddings:
            self.position_embeddings = nn.Embedding(self.max_position_embeddings, hidden_size)

    def forward(self, input_ids):
        batch_size, seq_length = input_ids.shape
        height, width = int(seq_length**0.5), int(seq_length**0.5)
        input_ids = input_ids.view(-1, height, width)
        embeddings = self.embeddings(input_ids)
        embeddings = self.layer_norm(embeddings)
        embeddings = embeddings.permute(0, 3, 1, 2)
        if self.patch_size > 1:
            embeddings = self.pixel_unshuffle(embeddings)
        embeddings = self.conv(embeddings)
        if self.use_position_embeddings:
            embeddings = embeddings.permute(0, 2, 3, 1).view(batch_size, -1, self.hidden_size)
            position_ids = torch.arange(embeddings.shape[1])[None, :].to(input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        return embeddings


class ConvMlmLayer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size,
        hidden_size,
        patch_size=2,
        norm_type="layernorm",
        layer_norm_eps=1e-5,
        use_bias=False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(hidden_size, embedding_size * (patch_size**2), kernel_size=1, bias=use_bias)
        if patch_size > 1:
            self.pixel_shuffle = nn.PixelShuffle(patch_size)
        self.layer_norm = Norm2D(embedding_size, norm_type=norm_type, eps=layer_norm_eps, use_bias=use_bias)
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
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            if module.bias is not None:
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
        if input_ids is not None:
            input_ids = torch.ones(shape, dtype=torch.long, device=self.device) * mask_token_id

        for step in range(timesteps):
            # prepend class token to input_ids
            if class_ids is not None:
                input_ids = torch.cat([class_ids[:, None], input_ids], dim=1)

            # classifier free guidance
            if encoder_hidden_states is not None and guidance_scale > 0:
                if negative_embeds is None:
                    uncond_encoder_states = torch.zeros_like(encoder_hidden_states)
                else:
                    uncond_encoder_states = negative_embeds

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

            # Samples the ids using categorical sampling: [batch_size, seq_length].
            # sampled_ids = torch.multinomial(logits.softmax(dim=-1), 1)
            sampled_ids = torch.stack(
                [torch.multinomial(l.softmax(dim=-1), 1, generator=generator).squeeze(1) for l in logits]
            )

            # Just updates the masked tokens.
            unknown_map = input_ids == mask_token_id
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            probs = logits.softmax(dim=-1)
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
        self.register_to_config(mask_token_id=vocab_size - 1)
        self.register_to_config(block_out_channels=tuple(block_out_channels))

        norm_cls = partial(LayerNorm, use_bias=use_bias) if norm_type == "layernorm" else RMSNorm

        if add_cross_attention is not None and project_encoder_hidden_states:  # Cross attention
            self.encoder_proj = nn.Linear(encoder_hidden_size, hidden_size, bias=use_bias)
            self.encoder_proj_layer_norm = norm_cls(hidden_size, eps=layer_norm_eps)
            encoder_hidden_size = hidden_size

        # Embeddings
        self.embed = ConvEmbed(
            vocab_size,
            in_channels,
            block_out_channels[0],
            patch_size=patch_size,
            norm_type=norm_type,
            layer_norm_eps=layer_norm_eps,
            use_position_embeddings=use_position_embeddings,
            use_bias=use_bias,
        )

        # Downsample
        output_channels = block_out_channels[0]
        self.down_blocks = nn.ModuleList([])
        for i in range(len(block_out_channels)):
            is_first_block = i == 0
            input_channels = output_channels
            output_channels = block_out_channels[i]
            self.down_blocks.append(
                DownsampleBlock(
                    input_channels=input_channels,
                    output_channels=output_channels,
                    skip_channels=0,
                    num_res_blocks=num_res_blocks,
                    kernel_size=3,
                    dropout=hidden_dropout,
                    norm_type=norm_type,
                    add_downsample=not is_first_block,
                    use_bias=use_bias,
                )
            )

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
                    layer_norm_eps=layer_norm_eps,
                    use_normformer=use_normformer,
                    use_bias=use_bias,
                )
                for _ in range(self.num_hidden_layers)
            ]
        )
        if use_encoder_layernorm:
            self.encoder_layer_norm = norm_cls(self.hidden_size, eps=layer_norm_eps)

        # Up sample
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channels = reversed_block_out_channels[0]
        self.up_blocks = nn.ModuleList([])
        for i in range(len(reversed_block_out_channels)):
            is_final_block = i == len(block_out_channels) - 1
            input_channel = reversed_block_out_channels[i]
            output_channels = reversed_block_out_channels[i + 1] if not is_final_block else output_channels
            prev_output_channels = input_channel if i != 0 else 0
            self.up_blocks.append(
                UpsampleBlock(
                    input_channels=input_channel,
                    skip_channels=prev_output_channels,
                    output_channels=output_channels,
                    num_res_blocks=num_res_blocks,
                    kernel_size=3,
                    dropout=hidden_dropout,
                    norm_type=norm_type,
                    add_upsample=not is_final_block,
                    use_bias=use_bias,
                )
            )

        # Output
        self.output_size = codebook_size if use_codebook_size_for_output else self.vocab_size
        self.mlm_layer = ConvMlmLayer(
            self.output_size,
            in_channels,
            block_out_channels[0],
            patch_size=patch_size,
            norm_type=norm_type,
            layer_norm_eps=layer_norm_eps,
            use_bias=use_bias,
        )
        self.gradient_checkpointing = False

        # --- WEIGHT INIT ---
        self.apply(self._init_weights)  # General init
        nn.init.constant_(self.mlm_layer.conv1.weight, 0)
        nn.init.normal_(self.embed.embeddings.weight, std=np.sqrt(1 / vocab_size))
        nn.init.normal_(self.mlm_layer.conv2.weight, std=np.sqrt(1 / codebook_size))

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
            module.weight.data.fill_(1.0)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        label_smoothing=0.0,
        cond_dropout_prob=0.0,
        loss_weight=None,
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

        down_block_res_samples = ()
        for down_block in self.down_blocks:
            hidden_states, res_samples = down_block(hidden_states)
            down_block_res_samples += res_samples

        batch_size, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)

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

        hidden_states = hidden_states.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)

        for i, up_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-self.config.num_res_blocks :]
            down_block_res_samples = down_block_res_samples[: -self.config.num_res_blocks]
            hidden_states = up_block(hidden_states, x_skip=res_samples if i > 0 else None)

        batch_size, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        logits = self.mlm_layer(hidden_states)

        if labels is not None:
            cross_entropy_per_image = F.cross_entropy(
                logits.view(-1, self.output_size),
                labels.view(-1),
                ignore_index=-100,
                label_smoothing=label_smoothing,
                reduction="none",
            )

            if loss_weight is None:
                loss = cross_entropy_per_image.mean()
            else:
                loss_weight = loss_weight.view(-1)
                loss = ((cross_entropy_per_image * loss_weight).sum(dim=-1) / loss_weight.sum(dim=-1)).mean()

            return logits, loss, cross_entropy_per_image
        return logits

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True
        if isinstance(module, (DownsampleBlock, UpsampleBlock)):
            module.gradient_checkpointing = value

    def generate(self):
        pass

    def generate2(
        self,
        input_ids: torch.LongTensor = None,
        class_ids: torch.LongTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        negative_embeds: torch.FloatTensor = None,
        temperature=1.0,
        timesteps=18,  # ideal number of steps is 18 in maskgit paper
        guidance_scale=0,
        guidance_schedule=None,
        noise_schedule=cosine_schedule,
        noise_type="mask",  # can be "mask" or "random_replace"
        predict_all_tokens=False,
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

        for step in range(timesteps):
            # prepend class token to input_ids
            if class_ids is not None:
                input_ids = torch.cat([class_ids[:, None], input_ids], dim=1)

            # classifier free guidance
            if encoder_hidden_states is not None and guidance_scale > 0:
                if negative_embeds is None:
                    uncond_encoder_states = torch.zeros_like(encoder_hidden_states)
                else:
                    uncond_encoder_states = negative_embeds

                model_input = torch.cat([input_ids] * 2)
                condition = torch.cat([encoder_hidden_states, uncond_encoder_states])
                cond_logits, uncond_logits = self(model_input, encoder_hidden_states=condition).chunk(2)
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
                    input_ids = torch.where(masking, mask_token_id, sampled_ids)
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
