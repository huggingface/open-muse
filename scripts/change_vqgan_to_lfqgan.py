import json
from argparse import ArgumentParser
from muse import VQGANModel
import torch
from muse.modeling_lfq import LFQ

def switch_to_lfq(args):
    vae = VQGANModel.from_pretrained(args.vae)
    vae.config["use_lfq"] = True
    vae.config["commitment_cost"] = args.commitment_cost
    vae.config["entropy_cost"] = args.entropy_cost
    vae.config["diversity_gamma"] = args.diversity_gamma
    vae.config["codebook_dim"] = args.codebook_dim


    vae_with_lfq = VQGANModel.from_config(
        vae.config,
        use_lfq=True,
        commitment_cost=args.commitment_cost,
        entropy_cost=args.entropy_cost,
        diversity_gamma=args.diversity_gamma,
        codebook_dim=args.codebook_dim,
    )
    vae_with_lfq.load_state_dict(vae.state_dict(), strict=False)
    print(f"Saving to {args.lfq_vae_output_path}")
    vae_with_lfq.save_pretrained(args.lfq_vae_output_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--vae", type=str, default="amused/vqgan-f16-8192-laion")
    parser.add_argument("--lfq_vae_output_path", type=str, default="checkpoints/lfqgan-f16-65536-laion")
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument("--entropy_cost", type=float, default=0.1)
    parser.add_argument("--diversity_gamma", type=float, default=1)
    parser.add_argument("--codebook_dim", type=int, default=16)
    args = parser.parse_args()
    switch_to_lfq(args)
