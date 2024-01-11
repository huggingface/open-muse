import json
from argparse import ArgumentParser
from muse import VQGANModel
import torch


def add_spectral_norm_to_vae(args):
    vae = VQGANModel.from_pretrained(args.vae)
    vae_with_spectral = VQGANModel(vae.resolution,
                                   no_attn_mid_block=args.no_attn_mid_block,
                                   z_channels=args.z_channels,
                                   channel_mult=vae.channel_mult,
                                   quantized_embed_dim=args.quantized_embed_dim,
                                   num_embeddings=args.num_embeddings,
                                   attn_resolutions=() if len(args.attn_resolutions) == 0 else [int(resolution) for resolution in args.attn_resolutions.split('|')],
                                   use_z_channels=True
                                )
    original_state_dict = vae.state_dict()
    output_dict = {}
    for key in original_state_dict:
        if "decoder" in key and "norm" in key:
            weight_or_bias = key.split(".")[-1]
            new_key = ".".join(key.split(".")[:-1])+".norm_layer."+weight_or_bias
            output_dict[new_key] = original_state_dict[key]
        else:
            output_dict[key] = original_state_dict[key]
    vae_with_spectral.load_state_dict(output_dict, strict=False)
    print(f"Saving to {args.movq_vae_output_path}")
    vae_with_spectral.save_pretrained(args.movq_vae_output_path)

    # print(vae_with_spectral.decoder)
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--vae", type=str, default="openMUSE/vqgan-f16-8192-laion")
    parser.add_argument("--movq_vae_output_path", type=str, default="vqgan-f16-8192-laion-movq")
    parser.add_argument("--no_attn_mid_block", action="store_false", default=True)
    parser.add_argument("--z_channels", type=int, default=64)
    parser.add_argument("--attn_resolutions", type=str, default="", help="Attention resolutions split by |")
    parser.add_argument("--quantized_embed_dim", type=int, default=64)
    parser.add_argument("--num_embeddings", type=int, default=8192)
    args = parser.parse_args()
    add_spectral_norm_to_vae(args)
