import csv
import dataclasses
from argparse import ArgumentParser

import torch
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from torch.utils.benchmark import Compare, Timer
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from muse import MaskGiTUViT, PipelineMuse, VQGANModel

torch.manual_seed(0)
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")

num_threads = torch.get_num_threads()
prompt = "A high tech solarpunk utopia in the Amazon rainforest"


def main():
    args = ArgumentParser()
    args.add_argument("--device", choices=["4090", "a100"], required=True)

    args = args.parse_args()
    csv_data = []

    for batch_size in [1, 8]:
        for timesteps in [12, 20]:
            for resolution in [256, 512]:
                for use_xformers in [False, True]:
                    out, mem_bytes = sd_benchmark(
                        resolution=resolution, batch_size=batch_size, timesteps=timesteps, use_xformers=use_xformers
                    )

                    Compare([out]).print()
                    print("*******")

                    csv_data.append(
                        [
                            batch_size,
                            "stable_diffusion_1_5",
                            out.median * 1000,
                            args.device,
                            timesteps,
                            mem_bytes,
                            resolution,
                            use_xformers,
                            None,
                            None,
                            None,
                        ]
                    )

                    for force_down_up_sample in [False, True]:
                        for use_fused_mlp in [False, True]:
                            for use_fused_residual_norm in [False, True]:
                                out, mem_bytes = muse_benchmark(
                                    resolution=resolution,
                                    batch_size=batch_size,
                                    timesteps=timesteps,
                                    use_xformers=use_xformers,
                                    force_down_up_sample=force_down_up_sample,
                                    use_fused_mlp=use_fused_mlp,
                                    use_fused_residual_norm=use_fused_residual_norm,
                                )

                                Compare([out]).print()
                                print("*******")

                                csv_data.append(
                                    [
                                        batch_size,
                                        "muse",
                                        out.median * 1000,
                                        args.device,
                                        timesteps,
                                        mem_bytes,
                                        resolution,
                                        use_xformers,
                                        force_down_up_sample,
                                        use_fused_mlp,
                                        use_fused_residual_norm,
                                    ]
                                )

    with open("benchmark/artifacts/all.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)


def muse_benchmark(
    resolution, batch_size, timesteps, use_xformers, force_down_up_sample, use_fused_mlp, use_fused_residual_norm
):
    model = "williamberman/muse_research_run_benchmarking_512_output"
    device = "cuda"
    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model, subfolder="text_encoder")

    text_encoder = CLIPTextModelWithProjection.from_pretrained(model, subfolder="text_encoder")
    text_encoder.to(device=device, dtype=dtype)

    vae = VQGANModel.from_pretrained(model, subfolder="vae")
    vae.to(device=device, dtype=dtype)

    transformer = MaskGiTUViT(
        use_fused_mlp=use_fused_mlp,
        use_fused_residual_norm=use_fused_residual_norm,
        force_down_up_sample=force_down_up_sample,
    )
    transformer = transformer.to(device=device, dtype=dtype)
    transformer.eval()

    if use_xformers:
        transformer.enable_xformers_memory_efficient_attention()

    pipe = PipelineMuse(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
    )
    pipe.device = device
    pipe.dtype = dtype

    seq_len = (resolution // 16) ** 2

    def benchmark_fn():
        pipe(prompt, num_images_per_prompt=batch_size, timesteps=timesteps, transformer_seq_len=seq_len)

    pipe(prompt, num_images_per_prompt=batch_size, timesteps=2, transformer_seq_len=seq_len)

    def fn():
        return Timer(
            stmt="benchmark_fn()",
            globals={"benchmark_fn": benchmark_fn},
            num_threads=num_threads,
            label=(
                f"batch_size: {batch_size}, dtype: {dtype}, timesteps {timesteps}, resolution: {resolution},"
                f" use_xformers: {use_xformers}, use_fused_mlp: {use_fused_mlp}, use_fused_residual_norm:"
                f" {use_fused_residual_norm}, force_down_up_sample: {force_down_up_sample}"
            ),
            description=model,
        ).blocked_autorange(min_run_time=1)

    return measure_max_memory_allocated(fn)


def sd_benchmark(resolution, batch_size, timesteps, use_xformers):
    model = "runwayml/stable-diffusion-v1-5"
    device = "cuda"
    dtype = torch.float16

    tokenizer = CLIPTokenizer.from_pretrained(model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model, subfolder="text_encoder")
    text_encoder.to(device=device, dtype=dtype)

    vae = AutoencoderKL.from_pretrained(model, subfolder="vae")
    vae = vae.to(device=device, dtype=dtype)

    unet = UNet2DConditionModel.from_pretrained(model, subfolder="unet")
    unet = unet.to(device=device, dtype=dtype)

    pipe = StableDiffusionPipeline.from_pretrained(
        model,
        vae=vae,
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        safety_checker=None,
    )

    if use_xformers:
        pipe.enable_xformers_memory_efficient_attention()

    def benchmark_fn():
        pipe(
            prompt,
            num_images_per_prompt=batch_size,
            num_inference_steps=timesteps,
            height=resolution,
            width=resolution,
        )

    pipe(prompt, num_images_per_prompt=batch_size, num_inference_steps=2, height=resolution, width=resolution)

    def fn():
        return Timer(
            stmt="benchmark_fn()",
            globals={"benchmark_fn": benchmark_fn},
            num_threads=num_threads,
            label=(
                f"batch_size: {batch_size}, dtype: {dtype}, timesteps {timesteps}, resolution: {resolution},"
                f" use_xformers: {use_xformers}"
            ),
            description=model,
        ).blocked_autorange(min_run_time=1)

    return measure_max_memory_allocated(fn)


def measure_max_memory_allocated(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    rv = fn()

    mem_bytes = torch.cuda.max_memory_allocated()

    return rv, mem_bytes


if __name__ == "__main__":
    main()
