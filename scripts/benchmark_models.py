import argparse
from functools import partial

import torch
import torch.utils.benchmark as benchmark

from muse import MaskGitTransformer, MaskGiTUViT


def benchmark_torch_function(f, *args, **kwargs):
    t0 = benchmark.Timer(stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f})
    return round(t0.blocked_autorange(min_run_time=1).mean, 2)


def create_model_and_benchmark(args):
    if args.model_type == "transformer":
        config = MaskGitTransformer.load_config(args.config_path)
        model = MaskGitTransformer.from_config(config).to(args.device)
    elif args.model_type == "uvit":
        config = MaskGiTUViT.load_config(args.config_path)
        model = MaskGiTUViT.from_config(config).to(args.device)

    model.eval()

    print("Running benchmark for vanilla attention in FP32 ...")
    encoder_hidden_states = torch.randn(
        args.batch_size, args.text_length, model.config.encoder_hidden_size, device=args.device, dtype=torch.float32
    )
    f = lambda: model.generate2(encoder_hidden_states=encoder_hidden_states, timesteps=args.time_steps)
    time_vanilla = benchmark_torch_function(f)

    print("Running benchmark for vanilla attention in FP16 ...")
    encoder_hidden_states = encoder_hidden_states.half()
    model = model.half()
    f = lambda: model.generate2(encoder_hidden_states=encoder_hidden_states, timesteps=args.time_steps)
    time_vanilla_fp16 = benchmark_torch_function(f)

    print("Running benchmark for efficient attention in FP16 ...")
    model.enable_xformers_memory_efficient_attention()
    f = lambda: model.generate2(encoder_hidden_states=encoder_hidden_states, timesteps=args.time_steps)
    time_efficient_fp16 = benchmark_torch_function(f)

    # print results with nice formatting
    print(f"Vanilla attention in FP32: {time_vanilla} ms")
    print(f"Vanilla attention in FP16: {time_vanilla_fp16} ms")
    print(f"Efficient attention in FP16: {time_efficient_fp16} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "uvit"])
    parser.add_argument("--text_length", type=int, default=96)
    parser.add_argument("--time_steps", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    create_model_and_benchmark(args)
