from typing import Callable

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
from .sampling import cosine_schedule, gumbel_sample, top_k


# layer norm without bias
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, use_bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if use_bias else None
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


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

        self.kv_input_dim = self.hidden_size if encoder_hidden_size is None else encoder_hidden_size
        self.key = nn.Linear(self.kv_input_dim, self.hidden_size, bias=use_bias)
        self.value = nn.Linear(self.kv_input_dim, self.hidden_size, bias=use_bias)

        self.out = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch, seq_len, _ = hidden_states.shape

        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states

        query = self.query(hidden_states)
        key = self.key(context)
        value = self.value(context)

        query = query.view(batch, seq_len, self.num_heads, self.head_dim)  # (B, T, nh, hs)
        key = key.view(batch, seq_len, self.num_heads, self.head_dim)  # (B, T, nh, hs)
        value = value.view(batch, seq_len, self.num_heads, self.head_dim)  # (B, T, nh, hs)

        query, key, value = map(lambda t: t.transpose(1, 2), (query, key, value))  # (B, nh, T, hs)

        query = query / self.scale_attn
        attn_weights = torch.matmul(query, key.transpose(-2, -1))
        # Apply the attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        )  # re-assemble all head outputs side by side
        attn_output = self.out(attn_output)
        return attn_output


# Normformer style GLU FeedForward
class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_dropout=0.0, layer_norm_eps=1e-5, use_bias=False):
        super().__init__()
        self.pre_mlp_layer_norm = LayerNorm(hidden_size, eps=layer_norm_eps, use_bias=use_bias)
        self.wi_0 = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        self.wi_1 = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        self.mid_mlp_layer_norm = LayerNorm(intermediate_size, eps=layer_norm_eps, use_bias=use_bias)
        self.wo = nn.Linear(intermediate_size, hidden_size, bias=use_bias)
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.pre_mlp_layer_norm(hidden_states)

        hidden_gelu = F.gelu(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear

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
        layer_norm_eps=1e-5,
        use_bias=False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads

        self.attn_layer_norm = LayerNorm(self.hidden_size, eps=layer_norm_eps, use_bias=use_bias)
        self.attention = Attention(
            self.hidden_size, self.num_attention_heads, attention_dropout=attention_dropout, use_bias=use_bias
        )
        self.ffn = FeedForward(self.hidden_size, self.intermediate_size, hidden_dropout, layer_norm_eps, use_bias)

        if add_cross_attention:
            self.crossattn_layer_norm = LayerNorm(self.hidden_size, eps=layer_norm_eps, use_bias=use_bias)
            self.crossattention = Attention(
                self.hidden_size, self.num_attention_heads, encoder_hidden_size, attention_dropout, use_bias
            )

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        residual = hidden_states

        hidden_states = self.attn_layer_norm(hidden_states)
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = residual + attention_output

        if encoder_hidden_states is not None:
            residual = hidden_states
            # TODO: should norm be applied to encoder_hidden_states as well?
            hidden_states = self.crossattn_layer_norm(hidden_states)
            attention_output = self.crossattention(hidden_states, encoder_hidden_states=encoder_hidden_states)
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
            self.embeddings_ln = LayerNorm(self.embedding_size, eps=layer_norm_eps, use_bias=use_bias)

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
    def __init__(self, hidden_size, vocab_size, layer_norm_eps=1e-5, use_bias=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlm_dense = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)
        self.mlm_ln = LayerNorm(self.hidden_size, eps=layer_norm_eps, use_bias=use_bias)
        self.to_logits = nn.Linear(self.hidden_size, vocab_size, bias=use_bias)

    def forward(self, hidden_states):
        hidden_states = self.mlm_dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.mlm_ln(hidden_states)
        logits = self.to_logits(hidden_states)
        return logits


class MaskGitTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        vocab_size,  # codebook_size + 1 (for the mask token), for class-conditioned generation it'll be codebook_size + num_classes + 1
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=256,  # for clas-conditioned generation it'll be 256 + 1 (for the class token)
        encoder_hidden_size=1024,  # T5-large
        add_cross_attention=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_bias=False,
        codebook_size=1024,
        num_classes=None,  # set for class-conditioned generation
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
        self.config.mask_token_id = vocab_size - 1

        self.embed = Embed(
            self.vocab_size,
            self.hidden_size,
            self.hidden_size,
            self.hidden_dropout,
            self.max_position_embeddings,
            use_bias,
            layer_norm_eps,
        )
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
                    layer_norm_eps=layer_norm_eps,
                    use_bias=use_bias,
                )
                for _ in range(self.num_hidden_layers)
            ]
        )
        self.encoder_layer_norm = LayerNorm(self.hidden_size, eps=layer_norm_eps, use_bias=use_bias)
        self.mlm_layer = MlmLayer(self.hidden_size, self.vocab_size, layer_norm_eps, use_bias)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True

    def forward(self, input_ids, encoder_hidden_states=None, labels=None):
        hidden_states = self.embed(input_ids)

        for layer in self.transformer_layers:
            if self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = checkpoint(create_custom_forward(layer), hidden_states, encoder_hidden_states)
            else:
                hidden_states = layer(hidden_states, encoder_hidden_states=encoder_hidden_states)

        hidden_states = self.encoder_layer_norm(hidden_states)
        logits = self.mlm_layer(hidden_states)
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1), ignore_index=-100)
            return logits, loss
        return logits

    def generate(
        self,
        class_ids: torch.LongTensor,
        temperature=1.0,
        topk_filter_thres=0.9,
        can_remask_prev_masked=False,  # TODO: implement this
        timesteps=18,  # ideal number of steps is 18 in maskgit paper
        cond_scale=3,  # TODO: implement this
        noise_schedule: Callable = cosine_schedule,
    ):
        # begin with all image token ids masked
        mask_token_id = self.config.mask_token_id
        device = next(self.parameters()).device
        seq_len = self.max_position_embeddings - 1  # 256 image tokens + 1 class token, hardcode for now

        batch_size = len(class_ids)
        shape = (batch_size, seq_len)

        # shift the class ids by the codebook size
        class_ids += self.config.codebook_size

        # initialize with all image tokens masked
        input_ids = torch.ones((1, seq_len), dtype=torch.long, device=device) * mask_token_id
        scores = torch.zeros(shape, dtype=torch.float32, device=device)

        starting_temperature = temperature

        for timestep, steps_until_x0 in tqdm(
            zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))), total=timesteps
        ):
            rand_mask_prob = noise_schedule(timestep)
            num_token_masked = max(int((rand_mask_prob * seq_len).item()), 1)

            masked_indices = scores.topk(num_token_masked, dim=-1).indices
            input_ids = input_ids.scatter(1, masked_indices, mask_token_id)

            # prepend class token to input_ids
            input_ids = torch.cat([class_ids[:, None], input_ids], dim=1)

            logits = self(input_ids)

            # remove class token
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
