from einops import rearrange, reduce, einsum
import math
from functools import partial
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from .modeling_utils import ConfigMixin, ModelMixin, register_to_config

def log(t, eps = 1e-5):
    return t.clamp(min = eps).log()

def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)

class LFQ(nn.Module):
    def __init__(
        self, quantized_embed_dim, entropy_cost = 0.1, commitment_cost = 0.25, diversity_gamma=1, codebook_dim=16
    ):
        super().__init__()

        codebook_size = 2**codebook_dim
        has_projections = quantized_embed_dim != codebook_dim
        self.project_in = nn.Linear(quantized_embed_dim, codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, quantized_embed_dim) if has_projections else nn.Identity()
        self.has_projections = has_projections


        self.codebook_dim = codebook_dim
        self.entropy_cost = entropy_cost
        self.commitment_cost = commitment_cost
        self.diversity_gamma = diversity_gamma

        # for no auxiliary loss, during inference

        self.register_buffer('mask', 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # codes

        all_codes = torch.arange(codebook_size)
        bits = ((all_codes[..., None].int() & self.mask) != 0).float()
        codebook = bits*2-1

        self.register_buffer('codebook', codebook, persistent = False)
    @property
    def dtype(self):
        return self.codebook.dtype
    def get_code(self, hidden_states):
        # reshape z -> (batch, height, width, channel)
        batch, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous()
        hidden_states = self.project_in(hidden_states)
        flattened_hidden_state = hidden_states.reshape((-1, self.codebook_dim))

        codebook_value = torch.ones_like(flattened_hidden_state)
        quantized = torch.where(flattened_hidden_state > 0, codebook_value, -codebook_value)
        z_q = hidden_states
        z_q = z_q + (quantized.reshape((batch, height, width, channel)) - z_q).detach()
        z_q = self.project_out(z_q)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        # calculate indices
        # (B*H*W,C) * (C)
        indices = (quantized > 0).int() * self.mask.int()
        return indices
    def get_codebook_entry(self, indices):
        # indices are expected to be of shape (batch, num_tokens)
        # get quantized latent vectors
        batch, num_tokens = indices.shape
        bits = ((indices[..., None].int() & self.mask) != 0).to(self.dtype)
        z_q = bits * 2 - 1
        z_q = z_q.reshape(batch, int(math.sqrt(num_tokens)), int(math.sqrt(num_tokens)), -1)
        z_q = self.project_out(z_q)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q
    def forward(
        self,
        hidden_states,
        return_loss = True,
        inv_temperature = 100.,
    ):
        batch, _, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous()
        hidden_states = self.project_in(hidden_states)
        flattened_hidden_state = hidden_states.reshape((-1, self.codebook_dim))

        codebook_value = torch.ones_like(flattened_hidden_state)
        quantized = torch.where(flattened_hidden_state > 0, codebook_value, -codebook_value)
        z_q = hidden_states
        z_q = z_q + (quantized.reshape((batch, height, width, self.codebook_dim)) - z_q).detach()
        z_q = self.project_out(z_q)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        # calculate indices
        # (B*H*W,C) * (C)
        indices = (quantized > 0).int() * self.mask.int()
        loss = None
        if return_loss:

            # entropy aux loss

            if self.training:
                # the same as euclidean distance up to a constant
                distance = flattened_hidden_state
                distance = -2 * einsum(distance, self.codebook, 'bc, jc -> bj')

                prob = (-distance * inv_temperature).softmax(dim = -1)

                per_sample_entropy = entropy(prob).mean()

                # distribution over all available tokens in the batch

                avg_prob = reduce(prob, 'bc -> c', 'mean')
                codebook_entropy = entropy(avg_prob).mean()

                # 1. entropy will be nudged to be low for each code, to encourage the network to output confident predictions
                # 2. codebook entropy will be nudged to be high, to encourage all codes to be uniformly used within the batch

                entropy_aux_loss = per_sample_entropy - self.diversity_gamma * codebook_entropy
            else:
                # if not training, just return dummy 0
                entropy_aux_loss = per_sample_entropy = codebook_entropy = self.zero

            # commit loss

            if self.training:
                commit_loss = F.mse_loss(flattened_hidden_state, quantized.detach(), reduction = 'none')


                commit_loss = commit_loss.mean()
            else:
                commit_loss = self.zero
            loss = entropy_aux_loss * self.entropy_cost + commit_loss * self.commitment_cost


        return z_q, indices, loss