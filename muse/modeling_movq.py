# Taken from https://github.com/ai-forever/Kandinsky-2/blob/main/kandinsky2/vqgan/movq_modules.py
# pytorch_diffusion + derived encoder decoder

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modeling_utils import ConfigMixin, ModelMixin, register_to_config

try:
    import xformers.ops as xops

    is_xformers_available = True
except ImportError:
    is_xformers_available = False


class SpatialNorm(nn.Module):
    def __init__(
        self,
        f_channels,
        zq_channels,
        norm_layer=nn.GroupNorm,
        freeze_norm_layer=False,
        add_conv=False,
        **norm_layer_params,
    ):
        super().__init__()
        self.norm_layer = norm_layer(num_channels=f_channels, **norm_layer_params)
        if freeze_norm_layer:
            for p in self.norm_layer.parameters:
                p.requires_grad = False
        self.add_conv = add_conv
        if self.add_conv:
            self.conv = nn.Conv2d(zq_channels, zq_channels, kernel_size=3, stride=1, padding=1)
        self.conv_y = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, f, zq):
        f_size = f.shape[-2:]
        zq = F.interpolate(zq, size=f_size, mode="nearest")
        if self.add_conv:
            zq = self.conv(zq)
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f


def Normalize(in_channels, zq_ch, add_conv):
    return SpatialNorm(
        in_channels,
        zq_ch,
        norm_layer=nn.GroupNorm,
        freeze_norm_layer=False,
        add_conv=add_conv,
        num_groups=32,
        eps=1e-6,
        affine=True,
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        zq_ch=None,
        add_conv=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        if zq_ch:
            self.norm1 = Normalize(in_channels, zq_ch, add_conv=add_conv)
        else:
            self.norm1 = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if zq_ch:
            self.norm2 = Normalize(out_channels, zq_ch, add_conv=add_conv)
        else:
            self.norm2 = torch.nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)

        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, zq=None):
        residual = hidden_states
        if zq is not None:
            hidden_states = self.norm1(hidden_states, zq)
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if zq is not None:
            hidden_states = self.norm2(hidden_states, zq)
        else:
            hidden_states = self.norm2(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)

        return hidden_states + residual


class AttnBlock(nn.Module):
    def __init__(self, in_channels, zq_ch=None, add_conv=False):
        super().__init__()
        self.in_channels = in_channels
        if zq_ch:
            self.norm = Normalize(in_channels, zq_ch, add_conv=add_conv)
        else:
            self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Linear(in_channels, in_channels)
        self.k = nn.Linear(in_channels, in_channels)
        self.v = nn.Linear(in_channels, in_channels)
        self.proj_out = nn.Linear(in_channels, in_channels)

        self.use_memory_efficient_attention_xformers = False
        self.xformers_attention_op = None

    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
    ):
        if use_memory_efficient_attention_xformers and not is_xformers_available:
            raise ImportError("Please install xformers to use memory efficient attention")
        self.use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
        self.xformers_attention_op = attention_op

    def forward(self, hidden_states, zq=None):
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape
        if zq is not None:
            hidden_states = self.norm(hidden_states, zq)
        else:
            hidden_states = self.norm(hidden_states)

        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)
        scale = 1.0 / torch.sqrt(torch.tensor(channel, dtype=hidden_states.dtype, device=hidden_states.device))

        query = self.q(hidden_states)
        key = self.k(hidden_states)
        value = self.v(hidden_states)

        if self.use_memory_efficient_attention_xformers:
            # Memory efficient attention
            hidden_states = xops.memory_efficient_attention(
                query, key, value, attn_bias=None, op=self.xformers_attention_op
            )
        else:
            attention_scores = torch.baddbmm(
                torch.empty(
                    query.shape[0],
                    query.shape[1],
                    key.shape[1],
                    dtype=query.dtype,
                    device=query.device,
                ),
                query,
                key.transpose(-1, -2),
                beta=0,
                alpha=scale,
            )
            attention_probs = torch.softmax(attention_scores.float(), dim=-1).type(attention_scores.dtype)
            hidden_states = torch.bmm(attention_probs, value)

        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).view(batch, channel, height, width)

        return hidden_states + residual


class UpsamplingBlock(nn.Module):
    def __init__(self, config, curr_res: int, block_idx: int, zq_ch: int):
        super().__init__()

        self.config = config
        self.block_idx = block_idx
        self.curr_res = curr_res

        if self.block_idx == self.config.num_resolutions - 1:
            block_in = self.config.hidden_channels * self.config.channel_mult[-1]
        else:
            block_in = self.config.hidden_channels * self.config.channel_mult[self.block_idx + 1]

        block_out = self.config.hidden_channels * self.config.channel_mult[self.block_idx]

        res_blocks = []
        attn_blocks = []
        for _ in range(self.config.num_res_blocks + 1):
            res_blocks.append(ResnetBlock(block_in, block_out, zq_ch=zq_ch, dropout=self.config.dropout))
            block_in = block_out
            if self.curr_res in self.config.attn_resolutions:
                attn_blocks.append(AttnBlock(block_in, zq_ch=zq_ch))

        self.block = nn.ModuleList(res_blocks)
        self.attn = nn.ModuleList(attn_blocks)

        self.upsample = None
        if self.block_idx != 0:
            self.upsample = Upsample(block_in, self.config.resample_with_conv)

    def forward(self, hidden_states, zq):
        for i, res_block in enumerate(self.block):
            hidden_states = res_block(hidden_states, zq)
            if len(self.attn) > 1:
                hidden_states = self.attn[i](hidden_states, zq)

        if self.upsample is not None:
            hidden_states = self.upsample(hidden_states)

        return hidden_states


class DownsamplingBlock(nn.Module):
    def __init__(self, config, curr_res: int, block_idx: int):
        super().__init__()

        self.config = config
        self.curr_res = curr_res
        self.block_idx = block_idx

        in_channel_mult = (1,) + tuple(self.config.channel_mult)
        block_in = self.config.hidden_channels * in_channel_mult[self.block_idx]
        block_out = self.config.hidden_channels * self.config.channel_mult[self.block_idx]

        res_blocks = nn.ModuleList()
        attn_blocks = nn.ModuleList()
        for _ in range(self.config.num_res_blocks):
            res_blocks.append(ResnetBlock(block_in, block_out, dropout=self.config.dropout))
            block_in = block_out
            if self.curr_res in self.config.attn_resolutions:
                attn_blocks.append(AttnBlock(block_in))

        self.block = res_blocks
        self.attn = attn_blocks

        self.downsample = None
        if self.block_idx != self.config.num_resolutions - 1:
            self.downsample = Downsample(block_in, self.config.resample_with_conv)

    def forward(self, hidden_states):
        for i, res_block in enumerate(self.block):
            hidden_states = res_block(hidden_states)
            if len(self.attn) > 1:
                hidden_states = self.attn[i](hidden_states)

        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)

        return hidden_states


class MidBlock(nn.Module):
    def __init__(self, config, in_channels: int, zq_ch=None, dropout: float = 0.0):
        super().__init__()

        self.config = config
        self.in_channels = in_channels
        self.dropout = dropout

        self.block_1 = ResnetBlock(
            self.in_channels,
            self.in_channels,
            dropout=self.dropout,
            zq_ch=zq_ch,
        )
        self.attn_1 = AttnBlock(self.in_channels, zq_ch=zq_ch)
        self.block_2 = ResnetBlock(
            self.in_channels,
            self.in_channels,
            dropout=self.dropout,
            zq_ch=zq_ch,
        )

    def forward(self, hidden_states, zq=None):
        hidden_states = self.block_1(hidden_states, zq)
        hidden_states = self.attn_1(hidden_states, zq)
        hidden_states = self.block_2(hidden_states, zq)
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # downsampling
        self.conv_in = nn.Conv2d(
            self.config.num_channels,
            self.config.hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        curr_res = self.config.resolution
        downsample_blocks = []
        for i_level in range(self.config.num_resolutions):
            downsample_blocks.append(DownsamplingBlock(self.config, curr_res, block_idx=i_level))

            if i_level != self.config.num_resolutions - 1:
                curr_res = curr_res // 2
        self.down = nn.ModuleList(downsample_blocks)

        # middle
        mid_channels = self.config.hidden_channels * self.config.channel_mult[-1]
        self.mid = MidBlock(config, mid_channels, dropout=self.config.dropout)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=mid_channels, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(
            mid_channels,
            self.config.z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, pixel_values):
        # downsampling
        hidden_states = self.conv_in(pixel_values)
        for block in self.down:
            hidden_states = block(hidden_states)

        # middle
        hidden_states = self.mid(hidden_states)

        # end
        hidden_states = self.norm_out(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class MoVQDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # compute in_channel_mult, block_in and curr_res at lowest res
        block_in = self.config.hidden_channels * self.config.channel_mult[self.config.num_resolutions - 1]
        curr_res = self.config.resolution // 2 ** (self.config.num_resolutions - 1)
        self.z_shape = (1, self.config.z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(
            self.config.z_channels,
            block_in,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # middle
        self.mid = MidBlock(config, block_in, zq_ch=self.config.quantized_embed_dim, dropout=self.config.dropout)

        # upsampling
        upsample_blocks = []
        for i_level in reversed(range(self.config.num_resolutions)):
            upsample_blocks.append(
                UpsamplingBlock(self.config, curr_res, block_idx=i_level, zq_ch=self.config.quantized_embed_dim)
            )
            if i_level != 0:
                curr_res = curr_res * 2
        self.up = nn.ModuleList(list(reversed(upsample_blocks)))  # reverse to get consistent order

        # end
        block_out = self.config.hidden_channels * self.config.channel_mult[0]
        self.norm_out = Normalize(block_out, self.config.quantized_embed_dim, False)
        self.conv_out = nn.Conv2d(
            block_out,
            self.config.num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, hidden_states, zq):
        # z to block_in
        hidden_states = self.conv_in(hidden_states)

        # middle
        hidden_states = self.mid(hidden_states, zq)

        # upsampling
        for block in reversed(self.up):
            hidden_states = block(hidden_states, zq)

        # end
        hidden_states = self.norm_out(hidden_states, zq)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, legacy=True):
        r"""
        Args:
            num_embeddings: number of vectors in the quantized space.
            embedding_dim: dimensionaity of the tensors in the quantized space.
                Inputs to the modules must be in this format as well.
            commitment_cost: scalar which controls the weighting of the loss terms
                (see equation 4 in the paper https://arxiv.org/abs/1711.00937 - this variable is Beta).
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.legacy = legacy

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, hidden_states, return_loss=False):
        # reshape z -> (batch, height, width, channel) and flatten
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous()

        distances = self.compute_distances(hidden_states)
        min_encoding_indices = torch.argmin(distances, axis=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.num_embeddings).to(hidden_states)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(hidden_states.shape)

        # reshape to (batch, num_tokens)
        min_encoding_indices = min_encoding_indices.reshape(hidden_states.shape[0], -1)

        # compute loss for embedding
        loss = None
        if return_loss:
            if not self.legacy:
                loss = self.beta * torch.mean((z_q.detach() - hidden_states) ** 2) + torch.mean(
                    (z_q - hidden_states.detach()) ** 2
                )
            else:
                loss = torch.mean((z_q.detach() - hidden_states) ** 2) + self.beta * torch.mean(
                    (z_q - hidden_states.detach()) ** 2
                )

            # preserve gradients
            z_q = hidden_states + (z_q - hidden_states).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, min_encoding_indices, loss

    def compute_distances(self, hidden_states):
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        hidden_states_flattended = hidden_states.reshape((-1, self.embedding_dim))
        return torch.cdist(hidden_states_flattended, self.embedding.weight)

    def get_codebook_entry(self, indices):
        # indices are expected to be of shape (batch, num_tokens)
        # get quantized latent vectors
        batch, num_tokens = indices.shape
        z_q = self.embedding(indices)
        z_q = z_q.reshape(batch, int(math.sqrt(num_tokens)), int(math.sqrt(num_tokens)), -1).permute(0, 3, 1, 2)
        return z_q

    def get_soft_code(self, hidden_states, temp=1.0, stochastic=False):
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous()  # (batch, height, width, channel)
        distances = self.compute_distances(hidden_states)  # (batch * height * width, num_embeddings)

        soft_code = F.softmax(-distances / temp, dim=-1)  # (batch * height * width, num_embeddings)
        if stochastic:
            code = torch.multinomial(soft_code, 1)  # (batch * height * width, 1)
        else:
            code = distances.argmin(dim=-1)  # (batch * height * width)

        code = code.reshape(hidden_states.shape[0], -1)  # (batch, height * width)
        batch, num_tokens = code.shape
        soft_code = soft_code.reshape(batch, num_tokens, -1)  # (batch, height * width, num_embeddings)
        return soft_code, code

    def get_code(self, hidden_states):
        # reshape z -> (batch, height, width, channel)
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous()
        distances = self.compute_distances(hidden_states)
        indices = torch.argmin(distances, axis=1).unsqueeze(1)
        indices = indices.reshape(hidden_states.shape[0], -1)
        return indices


class MOVQ(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        resolution: int = 256,
        num_channels=3,
        out_channels=3,
        hidden_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(32,),
        z_channels=4,
        double_z=False,
        num_embeddings=16384,
        quantized_embed_dim=4,
        dropout=0.0,
        resample_with_conv: bool = True,
        commitment_cost: float = 0.25,
    ):
        super().__init__()

        self.config.num_resolutions = len(channel_mult)
        self.config.reduction_factor = 2 ** (self.config.num_resolutions - 1)
        self.config.latent_size = resolution // self.config.reduction_factor

        self.encoder = Encoder(self.config)
        self.decoder = MoVQDecoder(self.config)
        self.quantize = VectorQuantizer(num_embeddings, quantized_embed_dim, commitment_cost=commitment_cost)
        self.quant_conv = torch.nn.Conv2d(z_channels, quantized_embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(quantized_embed_dim, z_channels, 1)

    def encode(self, pixel_values, return_loss=False):
        hidden_states = self.encoder(pixel_values)
        hidden_states = self.quant_conv(hidden_states)
        quantized_states, codebook_indices, codebook_loss = self.quantize(hidden_states, return_loss)
        output = (quantized_states, codebook_indices)
        if return_loss:
            output = output + (codebook_loss,)
        return output

    def decode(self, quant):
        quant2 = self.post_quant_conv(quant)
        dec = self.decoder(quant2, quant)
        return dec

    def decode_code(self, codebook_indices):
        quantized_states = self.quantize.get_codebook_entry(codebook_indices)
        reconstructed_pixel_values = self.decode(quantized_states)
        return reconstructed_pixel_values

    def get_code(self, pixel_values):
        hidden_states = self.encoder(pixel_values)
        hidden_states = self.quant_conv(hidden_states)
        codebook_indices = self.quantize.get_code(hidden_states)
        return codebook_indices

    def forward(self, pixel_values, return_loss=False):
        hidden_states = self.encoder(pixel_values)
        hidden_states = self.quant_conv(hidden_states)
        quantized_states, codebook_indices, codebook_loss = self.quantize(hidden_states, return_loss)
        reconstructed_pixel_values = self.decode(quantized_states)
        output = (reconstructed_pixel_values, codebook_indices)
        if return_loss:
            output = output + (codebook_loss,)
        return output
