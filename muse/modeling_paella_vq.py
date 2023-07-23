# VQGAN taken from https://github.com/dome272/Paella/

import math

import torch
import torch.nn.functional as F
from torch import nn

from .modeling_utils import ConfigMixin, ModelMixin, register_to_config

# TODO: This model only supports inference, not training. Make it trainable.


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    Discretization bottleneck part of the VQ-VAE.
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        r"""
        Args:
            num_embeddings: number of vectors in the quantized space.
            embedding_dim: dimensionality of the tensors in the quantized space.
                Inputs to the modules must be in this format as well.
            commitment_cost: scalar which controls the weighting of the loss terms
                (see equation 4 in the paper https://arxiv.org/abs/1711.00937 - this variable is Beta).
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.codebook_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, hidden_states, return_loss=False):
        """
        Inputs the output of the encoder network z and maps it to a discrete one-hot vector that is the index of the
        closest embedding vector e_j z (continuous) -> z_q (discrete) z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous()

        distances = self.compute_distances(hidden_states)
        min_encoding_indices = torch.argmin(distances, axis=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.num_embeddings).to(hidden_states)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.codebook.weight).view(hidden_states.shape)

        # reshape to (batch, num_tokens)
        min_encoding_indices = min_encoding_indices.reshape(hidden_states.shape[0], -1)

        # compute loss for embedding
        loss = None
        if return_loss:
            loss = torch.mean((z_q.detach() - hidden_states) ** 2) + self.commitment_cost * torch.mean(
                (z_q - hidden_states.detach()) ** 2
            )
            # preserve gradients
            z_q = hidden_states + (z_q - hidden_states).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, min_encoding_indices, loss

    def compute_distances(self, hidden_states):
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        hidden_states_flattended = hidden_states.reshape((-1, self.codebook_dim))
        return torch.cdist(hidden_states_flattended, self.codebook.weight)

    def get_codebook_entry(self, indices):
        # indices are expected to be of shape (batch, num_tokens)
        # get quantized latent vectors
        batch, num_tokens = indices.shape
        z_q = self.codebook(indices)
        z_q = z_q.reshape(batch, int(math.sqrt(num_tokens)), int(math.sqrt(num_tokens)), -1).permute(0, 3, 1, 2)
        return z_q

    # adapted from https://github.com/kakaobrain/rq-vae-transformer/blob/main/rqvae/models/rqvae/quantizations.py#L372
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


class ResBlock(nn.Module):
    def __init__(self, c, c_hidden):
        super().__init__()
        # depthwise/attention
        self.norm1 = nn.LayerNorm(c, elementwise_affine=False, eps=1e-6)
        self.depthwise = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(c, c, kernel_size=3, groups=c))

        self.norm2 = nn.LayerNorm(c, elementwise_affine=False, eps=1e-6)
        self.channelwise = nn.Sequential(
            nn.Linear(c, c_hidden),
            nn.GELU(),
            nn.Linear(c_hidden, c),
        )

        self.gammas = nn.Parameter(torch.zeros(6), requires_grad=True)

        def _basic_init(module):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def _norm(self, x, norm):
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, x):
        mods = self.gammas
        x_temp = self._norm(x, self.norm1) * (1 + mods[0]) + mods[1]
        x = x + self.depthwise(x_temp) * mods[2]
        x_temp = self._norm(x, self.norm2) * (1 + mods[3]) + mods[4]
        x = x + self.channelwise(x_temp.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) * mods[5]
        return x


class PaellaVQModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self, levels=2, bottleneck_blocks=12, c_hidden=384, c_latent=4, codebook_size=8192, scale_factor=0.3764
    ):  # 1.0
        super().__init__()
        self.c_latent = c_latent
        self.scale_factor = scale_factor
        c_levels = [c_hidden // (2**i) for i in reversed(range(levels))]

        # Encoder blocks
        self.in_block = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(3 * 4, c_levels[0], kernel_size=1))
        down_blocks = []
        for i in range(levels):
            if i > 0:
                down_blocks.append(nn.Conv2d(c_levels[i - 1], c_levels[i], kernel_size=4, stride=2, padding=1))
            block = ResBlock(c_levels[i], c_levels[i] * 4)
            down_blocks.append(block)
        down_blocks.append(
            nn.Sequential(
                nn.Conv2d(c_levels[-1], c_latent, kernel_size=1, bias=False),
                nn.BatchNorm2d(c_latent),  # then normalize them to have mean 0 and std 1
            )
        )
        self.down_blocks = nn.Sequential(*down_blocks)

        self.codebook_size = codebook_size
        self.vquantizer = VectorQuantizer(codebook_size, c_latent)

        # Decoder blocks
        up_blocks = [nn.Sequential(nn.Conv2d(c_latent, c_levels[-1], kernel_size=1))]
        for i in range(levels):
            for j in range(bottleneck_blocks if i == 0 else 1):
                block = ResBlock(c_levels[levels - 1 - i], c_levels[levels - 1 - i] * 4)
                up_blocks.append(block)
            if i < levels - 1:
                up_blocks.append(
                    nn.ConvTranspose2d(
                        c_levels[levels - 1 - i], c_levels[levels - 2 - i], kernel_size=4, stride=2, padding=1
                    )
                )
        self.up_blocks = nn.Sequential(*up_blocks)
        self.out_block = nn.Sequential(
            nn.Conv2d(c_levels[0], 3 * 4, kernel_size=1),
            nn.PixelShuffle(2),
        )

    def encode(self, x):
        x = self.in_block(x)
        x = self.down_blocks(x)
        # qe, (vq_loss, commit_loss), indices = self.vquantizer(x, dim=1)
        # return qe / self.scale_factor, x / self.scale_factor, indices, vq_loss + commit_loss * 0.25
        quantized_states, codebook_indices, codebook_loss = self.vquantizer(x)
        quantized_states = quantized_states / self.scale_factor
        output = (quantized_states, codebook_indices, codebook_loss)
        return output

    def decode(self, x):
        x = x * self.scale_factor
        x = self.up_blocks(x)
        x = self.out_block(x)
        return x

    def decode_code(self, codebook_indices):
        x = self.vquantizer.get_codebook_entry(codebook_indices)
        x = self.up_blocks(x)
        x = self.out_block(x)
        return x

    def get_code(self, pixel_values):
        x = self.in_block(pixel_values)
        x = self.down_blocks(x)
        return self.vquantizer.get_code(x)

    def forward(self, x, quantize=False):
        qe = self.encode(x)[0]
        x = self.decode(qe)
        return x
