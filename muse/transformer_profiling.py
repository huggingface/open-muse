import torch
import torch.nn.functional as F
from apex.normalization import FusedRMSNorm as RMSNorm  # noqa
from torch import nn


class MaskGitUVit(nn.Module):
    def __init__(
        self,
        embedding_size=512,
        num_res_blocks=3,
        hidden_size=1024,
        encoder_hidden_size=768,
        input_vocab_size=8256,  # (8192 + 1 for <mask> = 8193 but 8256 is the next multiple of 8)
        output_vocab_size=8192,
        num_transformer_layers=22,
        ln_elementwise_affine=True,
        layer_norm_eps=1e-6,
        num_attention_heads=16,
        dropout_p=0.0,
        use_bias=False,
    ):
        super().__init__()

        self.embed = nn.ModuleDict(
            dict(
                embeddings=nn.Embedding(input_vocab_size, embedding_size),
                layer_norm=RMSNorm(embedding_size, eps=layer_norm_eps, elementwise_affine=ln_elementwise_affine),
                conv=nn.Conv2d(embedding_size, hidden_size, kernel_size=1, bias=use_bias),
            )
        )

        self.down_blocks = nn.Sequential(
            *[
                ResBlock(
                    hidden_size,
                    use_bias=use_bias,
                    layer_norm_eps=layer_norm_eps,
                    ln_elementwise_affine=ln_elementwise_affine,
                    dropout_p=dropout_p,
                )
                for _ in range(num_res_blocks)
            ],
        )

        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(
                    hidden_size=hidden_size,
                    encoder_hidden_size=encoder_hidden_size,
                    use_bias=use_bias,
                    layer_norm_eps=layer_norm_eps,
                    ln_elementwise_affine=ln_elementwise_affine,
                    num_heads=num_attention_heads,
                    dropout_p=dropout_p,
                )
                for _ in range(num_transformer_layers)
            ]
        )
        self.encoder_layer_norm = RMSNorm(hidden_size, eps=layer_norm_eps, elementwise_affine=ln_elementwise_affine)

        self.up_blocks = nn.Sequential(
            *[
                ResBlock(
                    hidden_size,
                    use_bias=use_bias,
                    layer_norm_eps=layer_norm_eps,
                    ln_elementwise_affine=ln_elementwise_affine,
                    dropout_p=dropout_p,
                )
                for _ in range(num_res_blocks)
            ],
        )

        self.out = nn.ModuleDict(
            dict(
                conv1=nn.Conv2d(hidden_size, embedding_size, kernel_size=1, bias=use_bias),
                norm=RMSNorm(embedding_size, eps=layer_norm_eps, elementwise_affine=ln_elementwise_affine),
                conv2=nn.Conv2d(embedding_size, output_vocab_size, kernel_size=1, bias=use_bias),
            )
        )

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_attention_mask=None,
    ):
        batch_size, seq_length = input_ids.shape

        height = width = int(seq_length**0.5)
        input_ids = input_ids.view(-1, height, width)

        hidden_states = self.embed["embeddings"](input_ids)
        hidden_states = self.embed["layer_norm"](hidden_states)

        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.embed["conv"](hidden_states)

        hidden_states = self.down_blocks(hidden_states)

        batch_size, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)

        for layer in self.transformer_layers:
            hidden_states = layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )

        hidden_states = self.encoder_layer_norm(hidden_states)

        hidden_states = hidden_states.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)

        hidden_states = self.up_blocks(hidden_states)

        hidden_states = self.out["conv1"](hidden_states)
        hidden_states = hidden_states.permute(0, 2, 3, 1)
        hidden_states = self.out["norm"](hidden_states)
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        logits = self.out["conv2"](hidden_states)

        return logits


class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
        use_bias,
        layer_norm_eps,
        ln_elementwise_affine,
        dropout_p,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=use_bias,
        )
        self.norm = RMSNorm(channels, eps=layer_norm_eps, elementwise_affine=ln_elementwise_affine)
        self.channelwise = nn.Sequential(
            nn.Linear(channels, channels * 4, bias=False),
            nn.GELU(),
            GlobalResponseNorm(channels * 4),
            nn.Dropout(dropout_p),
            nn.Linear(channels * 4, channels, bias=False),
        )

    def forward(self, x):
        x_res = x
        x = self.depthwise(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.channelwise(x)
        x = x.permute(0, 3, 1, 2)
        x = x + x_res
        return x


class TransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        encoder_hidden_size,
        use_bias,
        layer_norm_eps,
        ln_elementwise_affine,
        num_heads,
        dropout_p,
    ):
        super().__init__()

        self.self_attention_layer_norm = RMSNorm(
            hidden_size, eps=layer_norm_eps, elementwise_affine=ln_elementwise_affine
        )
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout_p,
            bias=use_bias,
            batch_first=True,
        )
        self.ffn = nn.ModuleDict(
            dict(
                ln=LayerNormOptionalBias(
                    hidden_size, eps=layer_norm_eps, use_bias=use_bias, elementwise_affine=ln_elementwise_affine
                ),
                wi_0=nn.Linear(hidden_size, hidden_size * 4, bias=use_bias),
                wi_1=nn.Linear(hidden_size, hidden_size * 4, bias=use_bias),
                dropout=nn.Dropout(dropout_p),
                wo=nn.Linear(hidden_size * 4, hidden_size, bias=use_bias),
            )
        )

        self.cross_attention_layer_norm = RMSNorm(
            hidden_size, eps=layer_norm_eps, elementwise_affine=ln_elementwise_affine
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            kdim=encoder_hidden_size,
            vdim=encoder_hidden_size,
            num_heads=num_heads,
            dropout=dropout_p,
            bias=use_bias,
            batch_first=True,
        )

    def forward(self, hidden_states, encoder_hidden_states, encoder_attention_mask=None):
        residual = hidden_states
        normed_hidden_states = self.self_attention_layer_norm(hidden_states)
        hidden_states = self.attention(
            normed_hidden_states,
            normed_hidden_states,
            normed_hidden_states,
            need_weights=False,
        )[0]
        hidden_states = hidden_states + residual

        residual = hidden_states
        normed_hidden_states = self.cross_attention_layer_norm(hidden_states)
        hidden_states = self.cross_attention(
            normed_hidden_states,
            encoder_hidden_states,
            encoder_hidden_states,
        )[0]
        hidden_states = hidden_states + residual

        residual = hidden_states

        hidden_states = self.ffn["ln"](hidden_states)

        hidden_gelu = F.gelu(self.ffn["wi_0"](hidden_states))
        hidden_linear = self.ffn["wi_1"](hidden_states)

        hidden_states = hidden_gelu * hidden_linear

        hidden_states = self.ffn["dropout"](hidden_states)
        hidden_states = self.ffn["wo"](hidden_states)

        hidden_states = hidden_states + residual

        return hidden_states


class LayerNormOptionalBias(nn.Module):
    def __init__(self, dim, eps, use_bias, elementwise_affine):
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


def convert_state_dict(sd):
    sd_ = {}

    # embed
    sd_.update(
        {
            "embed.embeddings.weight": sd.pop("embed.embeddings.weight"),
            "embed.layer_norm.weight": sd.pop("embed.layer_norm.weight"),
            "embed.conv.weight": sd.pop("embed.conv.weight"),
        }
    )

    # in res blocks
    for res_block_idx in range(3):
        sd_.update(
            {
                f"down_blocks.{res_block_idx}.depthwise.weight": sd.pop(
                    f"down_blocks.0.res_blocks.{res_block_idx}.depthwise.weight"
                ),
                f"down_blocks.{res_block_idx}.norm.weight": sd.pop(
                    f"down_blocks.0.res_blocks.{res_block_idx}.norm.norm.weight"
                ),
                f"down_blocks.{res_block_idx}.channelwise.0.weight": sd.pop(
                    f"down_blocks.0.res_blocks.{res_block_idx}.channelwise.0.weight"
                ),
                f"down_blocks.{res_block_idx}.channelwise.2.gamma": sd.pop(
                    f"down_blocks.0.res_blocks.{res_block_idx}.channelwise.2.gamma"
                ),
                f"down_blocks.{res_block_idx}.channelwise.2.beta": sd.pop(
                    f"down_blocks.0.res_blocks.{res_block_idx}.channelwise.2.beta"
                ),
                f"down_blocks.{res_block_idx}.channelwise.4.weight": sd.pop(
                    f"down_blocks.0.res_blocks.{res_block_idx}.channelwise.4.weight"
                ),
            }
        )

    for transformer_layer_idx in range(22):
        in_proj_weight = torch.cat(
            (
                sd.pop(f"transformer_layers.{transformer_layer_idx}.attention.query.weight"),
                sd.pop(f"transformer_layers.{transformer_layer_idx}.attention.key.weight"),
                sd.pop(f"transformer_layers.{transformer_layer_idx}.attention.value.weight"),
            ),
            dim=0,
        )

        sd_.update(
            {
                # self attention
                f"transformer_layers.{transformer_layer_idx}.self_attention_layer_norm.weight": sd.pop(
                    f"transformer_layers.{transformer_layer_idx}.attn_layer_norm.weight"
                ),
                f"transformer_layers.{transformer_layer_idx}.self_attention.in_proj_weight": in_proj_weight,
                f"transformer_layers.{transformer_layer_idx}.self_attention.out_proj.weight": sd.pop(
                    f"transformer_layers.{transformer_layer_idx}.attention.out.weight"
                ),
                # cross attention
                f"transformer_layers.{transformer_layer_idx}.cross_attention_layer_norm.weight": sd.pop(
                    f"transformer_layers.{transformer_layer_idx}.crossattn_layer_norm.weight"
                ),
                f"transformer_layers.{transformer_layer_idx}.cross_attention.q_proj_weight": sd.pop(
                    f"transformer_layers.{transformer_layer_idx}.crossattention.query.weight"
                ),
                f"transformer_layers.{transformer_layer_idx}.cross_attention.k_proj_weight": sd.pop(
                    f"transformer_layers.{transformer_layer_idx}.crossattention.key.weight"
                ),
                f"transformer_layers.{transformer_layer_idx}.cross_attention.v_proj_weight": sd.pop(
                    f"transformer_layers.{transformer_layer_idx}.crossattention.value.weight"
                ),
                f"transformer_layers.{transformer_layer_idx}.cross_attention.out_proj.weight": sd.pop(
                    f"transformer_layers.{transformer_layer_idx}.crossattention.out.weight"
                ),
                # ffn
                f"transformer_layers.{transformer_layer_idx}.ffn.ln.weight": sd.pop(
                    f"transformer_layers.{transformer_layer_idx}.ffn.pre_mlp_layer_norm.weight"
                ),
                f"transformer_layers.{transformer_layer_idx}.ffn.wi_0.weight": sd.pop(
                    f"transformer_layers.{transformer_layer_idx}.ffn.wi_0.weight"
                ),
                f"transformer_layers.{transformer_layer_idx}.ffn.wi_1.weight": sd.pop(
                    f"transformer_layers.{transformer_layer_idx}.ffn.wi_1.weight"
                ),
                f"transformer_layers.{transformer_layer_idx}.ffn.wo.weight": sd.pop(
                    f"transformer_layers.{transformer_layer_idx}.ffn.wo.weight"
                ),
            }
        )

    # encoder layer norm
    sd_.update({"encoder_layer_norm.weight": sd.pop("encoder_layer_norm.weight")})

    # out res blocks
    for res_block_idx in range(3):
        sd_.update(
            {
                f"up_blocks.{res_block_idx}.depthwise.weight": sd.pop(
                    f"up_blocks.0.res_blocks.{res_block_idx}.depthwise.weight"
                ),
                f"up_blocks.{res_block_idx}.norm.weight": sd.pop(
                    f"up_blocks.0.res_blocks.{res_block_idx}.norm.norm.weight"
                ),
                f"up_blocks.{res_block_idx}.channelwise.0.weight": sd.pop(
                    f"up_blocks.0.res_blocks.{res_block_idx}.channelwise.0.weight"
                ),
                f"up_blocks.{res_block_idx}.channelwise.2.gamma": sd.pop(
                    f"up_blocks.0.res_blocks.{res_block_idx}.channelwise.2.gamma"
                ),
                f"up_blocks.{res_block_idx}.channelwise.2.beta": sd.pop(
                    f"up_blocks.0.res_blocks.{res_block_idx}.channelwise.2.beta"
                ),
                f"up_blocks.{res_block_idx}.channelwise.4.weight": sd.pop(
                    f"up_blocks.0.res_blocks.{res_block_idx}.channelwise.4.weight"
                ),
            }
        )

    # mlm_layer -> out
    sd_.update(
        {
            "out.conv1.weight": sd.pop("mlm_layer.conv1.weight"),
            "out.norm.weight": sd.pop("mlm_layer.layer_norm.norm.weight"),
            "out.conv2.weight": sd.pop("mlm_layer.conv2.weight"),
        }
    )

    assert len(sd) == 0

    return sd_
