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
import warnings
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .modeling_transformer_v2 import MaskGiTUViT_v2
from .modeling_utils import ConfigMixin, ModelMixin
from .sampling import cosine_schedule, mask_by_random_topk

try:
    import xformers.ops as xops

    is_xformers_available = True
except ImportError:
    is_xformers_available = False

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm = None

try:
    from flash_attn.ops.fused_dense import fused_mlp_func
except ImportError:
    fused_mlp_func = None

warnings.simplefilter("once", UserWarning)


MaskGiTUViT = MaskGiTUViT_v2


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


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, lewei_scale=1.0, base_size=16):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / lewei_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / lewei_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


@dataclasses.dataclass
class MaskGitTransformerConfig:
    # global config
    hidden_size: int = 1024
    in_channels: int = 256
    fmap_size: int = 16
    patch_size: int = 2
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

    # `TransformerLayer`
    num_hidden_layers: int = 24
    num_attention_heads: int = 16

    # `Attention`
    attention_dropout: float = 0.0

    # `FeedForward`
    intermediate_size: int = 4096
    use_fused_mlp: bool = False

    # `Norm`
    layer_norm_eps: float = 1e-6
    use_fused_residual_norm: bool = False

    # Legacy: kept for compatibility with pipeline
    add_cond_embeds: bool = True
    add_micro_cond_embeds: bool = True


def config_from_legacy_kwargs(**kwargs):
    config = {}

    # select only values that are expected to be in the config
    for field in dataclasses.fields(MaskGitTransformerConfig):
        if field.name in kwargs:
            config[field.name] = kwargs[field.name]

    # set default config values
    config = MaskGitTransformerConfig(**config)

    return config


class MaskGitTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    def __init__(self, **kwargs):
        super().__init__()

        config = config_from_legacy_kwargs(**kwargs)
        self.register_to_config(**dataclasses.asdict(config))
        self.register_to_config(mask_token_id=self.config.vocab_size - 1)

        # TODO: Allow enabling fused norm using a function (like we do for xformers attention)
        if self.config.use_fused_residual_norm and dropout_add_layer_norm is None:
            warnings.warn(
                "Cannot use fused layer norm. Please install flash_attn. Falling back to unfused layer norm",
                UserWarning,
            )
            self.register_to_config(use_fused_residual_norm=False)

        if self.config.use_fused_mlp and fused_mlp_func is None:
            warnings.warn(
                "Cannot use fused MLP. Please install flash_attn. Falling back to unfused MLP",
                UserWarning,
            )
            self.register_to_config(use_fused_mlp=False)

        # Legacy: kept for compatibility with pipeline
        self.output_size = self.config.codebook_size

        self.encoder_proj = nn.Linear(
            self.config.encoder_hidden_size, self.config.hidden_size, bias=self.config.use_bias
        )
        self.encoder_proj_layer_norm = LayerNorm(config.hidden_size, config=config, elementwise_affine=True)

        self.in_mapper = nn.Sequential(
            nn.Embedding(config.vocab_size, self.config.in_channels),
            LayerNorm(config.in_channels, config=config, elementwise_affine=True),
        )
        self.embed = PatchEmbed(config=self.config)

        self.cond_embed = nn.Sequential(
            nn.Linear(
                self.config.micro_cond_embed_dim + self.config.cond_embed_dim,
                self.config.hidden_size,
                bias=self.config.use_bias,
            ),
            nn.SiLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=self.config.use_bias),
        )

        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(self.config) for _ in range(self.config.num_hidden_layers)]
        )

        self.mlm_layer = FinalLayer(self.config)

        self.gradient_checkpointing = False

        # --- WEIGHT INIT ---
        self._init_weights()

    def _init_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize in_mapper
        nn.init.normal_(self.in_mapper[0].weight, std=np.sqrt(1 / self.config.vocab_size))  # out mapper

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.embed.proj.bias is not None:
            nn.init.constant_(self.embed.proj.bias, 0)

        # init AdaLNModulation.mapper layers to 0
        for m in self.modules():
            if isinstance(m, AdaLNModulation):
                nn.init.constant_(m.mapper.weight, 0)
                if m.mapper.bias is not None:
                    nn.init.constant_(m.mapper.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.mlm_layer.linear1.weight, 0)
        self.mlm_layer.linear2.weight.data = self.in_mapper[0].weight.data[: self.config.codebook_size, :].clone()

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        cond_embeds,
        micro_conds,
        labels=None,
        label_smoothing=0.0,
        loss_weight=None,
        **kwargs,
    ):
        encoder_hidden_states = self.encoder_proj(encoder_hidden_states)
        encoder_hidden_states, _ = self.encoder_proj_layer_norm(encoder_hidden_states)

        micro_cond_embeds = sinusoidal_encode(micro_conds.flatten(), self.config.micro_cond_encode_dim)
        micro_cond_embeds = micro_cond_embeds.reshape((input_ids.shape[0], -1))

        cond_embeds = torch.cat([cond_embeds, micro_cond_embeds], dim=1)
        cond_embeds = cond_embeds.to(dtype=self.dtype)
        cond_embeds = self.cond_embed(cond_embeds).to(encoder_hidden_states.dtype)

        hidden_states, _ = self.in_mapper(input_ids)
        hidden_states = self.embed(hidden_states)

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

        logits = self.mlm_layer(hidden_states, cond_embeds)

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
        **kwargs,
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
            from tqdm.auto import tqdm

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

            if not predict_all_tokens:
                selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
                selected_probs = selected_probs.squeeze(-1)
                # Ignores the tokens given in the input by overwriting their confidence.
                # selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
                temperature = temperatures[step]
                masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
                # Masks tokens with lower confidence.
                input_ids = torch.where(masking, self.config.mask_token_id, sampled_ids)
            else:
                batch_size, seq_len = input_ids.shape
                batch_randperm = torch.rand(batch_size, seq_len, device=input_ids.device).argsort(dim=-1)
                mask = batch_randperm < mask_len
                # mask images and create input and labels
                input_ids = torch.where(mask, self.config.mask_token_id, sampled_ids)

        if return_intermediate:
            return sampled_ids, intermediate

        return sampled_ids


# embedding blocks


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding + Position Embedding
    """

    def __init__(self, config: MaskGitTransformerConfig):
        super().__init__()
        self.config = config
        self.img_size = config.fmap_size
        self.patch_size = config.patch_size

        grid_size = (config.fmap_size // config.patch_size, config.fmap_size // config.patch_size)
        
        self.proj = nn.Conv2d(
            config.in_channels, config.hidden_size, kernel_size=config.patch_size, stride=(config.patch_size, config.patch_size), bias=config.use_bias
        )
        pos_embed = (
            torch.from_numpy(get_2d_sincos_pos_embed(config.hidden_size, grid_size, base_size=grid_size[0]))
            .float()
            .unsqueeze(0)
        )
        self.register_buffer("pos_embed", pos_embed)

    def forward(self, x):
        batch, seq_len, channels = x.shape
        x = x.reshape(batch, int(seq_len**0.5), int(seq_len**0.5), channels).permute(
            0, 3, 1, 2
        )  # B L C -> B C H W

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # B C H W -> B C L
        x = x + self.pos_embed
        return x


# norm blocks


class LayerNorm(nn.Module):
    def __init__(self, dim, config: MaskGitTransformerConfig, elementwise_affine=True):
        super().__init__()
        self.config = config
        self.dim = torch.Size((dim,))

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim)) if config.use_bias else None
        else:
            # fused layer norm does not support elementwise_affine=False
            # so we initialize weight to 1 and freeze it
            if self.config.use_fused_residual_norm:
                self.weight = nn.Parameter(torch.ones(dim), requires_grad=False)
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


# attention/transformer blocks


class TransformerLayer(nn.Module):
    def __init__(self, config: MaskGitTransformerConfig):
        super().__init__()

        self.attn_layer_norm = LayerNorm(config.hidden_size, config, elementwise_affine=False)
        self.self_attn_adaLN_modulation = AdaLNModulation(config.hidden_size, config)
        self.attention = Attention(config.hidden_size, config.hidden_size, config.num_attention_heads, config)

        self.crossattn_layer_norm = LayerNorm(config.hidden_size, config, elementwise_affine=True)
        self.crossattention = Attention(config.hidden_size, config.hidden_size, config.num_attention_heads, config)

        self.pre_mlp_layer_norm = LayerNorm(config.hidden_size, config, elementwise_affine=False)
        self.mlp_adaln_modulation = AdaLNModulation(config.hidden_size, config)
        self.mlp = FeedForward(config)

    def forward(self, hidden_states, encoder_hidden_states, cond_embeds, residual=None):
        hidden_states, residual = self.attn_layer_norm(hidden_states, residual=residual)
        hidden_states = self.self_attn_adaLN_modulation(hidden_states, cond_embeds)
        hidden_states = self.attention(hidden_states, hidden_states)

        hidden_states, residual = self.crossattn_layer_norm(hidden_states, residual=residual)
        hidden_states = self.crossattention(hidden_states, encoder_hidden_states)

        hidden_states, residual = self.pre_mlp_layer_norm(hidden_states, residual=residual)
        hidden_states = self.mlp_adaln_modulation(hidden_states, cond_embeds)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Attention(nn.Module):
    def __init__(self, hidden_size: int, context_dim: int, num_heads: int, config: MaskGitTransformerConfig):
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


def FeedForward(config: MaskGitTransformerConfig):
    if config.use_fused_mlp:
        return FusedMLP(config)
    else:
        return MLP(config)


class MLP(nn.Module):
    def __init__(self, config: MaskGitTransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.use_bias)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.use_bias)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class FusedMLP(nn.Module):
    def __init__(self, config: MaskGitTransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.use_bias)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.use_bias)

    def forward(self, hidden_states):
        if fused_mlp_func is None:
            raise ImportError("Please install flash_attn to use fused mlp")

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
            self.fc1.weight,
            self.fc2.weight,
            self.fc1.bias,
            self.fc2.bias,
            activation="gelu_approx",
            save_pre_act=self.training,
            return_residual=False,
            checkpoint_lvl=0,
            heuristic=heuristic,
        )

        return hidden_states


# misc blocks


class FinalLayer(nn.Module):
    def __init__(self, config: MaskGitTransformerConfig):
        super().__init__()
        self.config = config
        
        self.adaLN_modulation = AdaLNModulation(config.hidden_size, use_bias=config.use_bias)
        self.norm1 = LayerNorm(config.hidden_size, config=config, elementwise_affine=False)
        self.linear1 = nn.Linear(
            config.hidden_size, config.patch_size * config.patch_size * config.in_channels, bias=True
        )
        
        self.norm2 = LayerNorm(config.in_channels, config=config)
        self.linear2 = nn.Linear(config.in_channels, config.codebook_size, bias=False)

    def forward(self, hidden_states, cond_embeds):
        hidden_states, _ = self.norm1(hidden_states)
        hidden_states = self.adaLN_modulation(hidden_states, cond_embeds)
        hidden_states = self.linear1(hidden_states)

        hidden_states = self.unpatchify(hidden_states)
        hidden_states, _ = self.norm2(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H*W, C)
        """
        c = self.config.in_channels
        p = self.config.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nhpwqc", x)
        x = x.reshape(shape=(x.shape[0], h * p, h * p, c))
        x = x.reshape(shape=(x.shape[0], -1, c))
        return x


class AdaLNModulation(nn.Module):
    def __init__(self, hidden_size: int, use_bias: bool = False):
        super().__init__()
        self.mapper = nn.Linear(hidden_size, hidden_size * 2, bias=use_bias)

    def forward(self, hidden_states, cond_embeds):
        cond_embeds = F.silu(cond_embeds)
        scale, shift = self.mapper(cond_embeds).chunk(2, dim=1)
        if hidden_states.dim() > 3:
            scale, shift = scale[:, :, None, None], shift[:, :, None, None]
        else:
            scale, shift = scale[:, None], shift[:, None]
        return hidden_states * (1 + scale) + shift
