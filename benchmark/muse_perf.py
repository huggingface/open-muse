import csv
from argparse import ArgumentParser

import torch
from diffusers import (
    AutoencoderKL,
    AutoPipelineForText2Image,
    LatentConsistencyModelPipeline,
    LCMScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS
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
            for use_xformers in [False, True]:
                out, mem_bytes = sd_benchmark(batch_size=batch_size, timesteps=timesteps, use_xformers=use_xformers)

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
                        512,
                        use_xformers,
                        None,
                    ]
                )

                out, mem_bytes = sdxl_benchmark(
                    batch_size=batch_size, timesteps=timesteps, use_xformers=use_xformers, gpu_type=args.device
                )

                Compare([out]).print()
                print("*******")

                csv_data.append(
                    [
                        batch_size,
                        "sdxl",
                        out.median * 1000,
                        args.device,
                        timesteps,
                        mem_bytes,
                        1024,
                        use_xformers,
                        None,
                    ]
                )

                out, mem_bytes = ssd_1b_benchmark(
                    batch_size=batch_size, timesteps=timesteps, use_xformers=use_xformers, gpu_type=args.device
                )

                Compare([out]).print()
                print("*******")

                csv_data.append(
                    [
                        batch_size,
                        "ssd_1b",
                        out.median * 1000,
                        args.device,
                        timesteps,
                        mem_bytes,
                        1024,
                        use_xformers,
                        None,
                    ]
                )

                for resolution in [256, 512]:
                    for use_fused_residual_norm in [False, True]:
                        out, mem_bytes = muse_benchmark(
                            resolution=resolution,
                            batch_size=batch_size,
                            timesteps=timesteps,
                            use_xformers=use_xformers,
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
                                use_fused_residual_norm,
                            ]
                        )

        for use_xformers in [False, True]:
            out, mem_bytes = wurst_benchmark(batch_size, use_xformers)

            Compare([out]).print()
            print("*******")

            csv_data.append(
                [
                    batch_size,
                    "wurst",
                    out.median * 1000,
                    args.device,
                    "default",
                    mem_bytes,
                    1024,
                    use_xformers,
                    None,
                ]
            )

        for timesteps in [4, 8]:
            for use_xformers in [False, True]:
                out, mem_bytes = lcm_benchmark(batch_size, timesteps, use_xformers)

                Compare([out]).print()
                print("*******")

                csv_data.append(
                    [
                        batch_size,
                        "lcm",
                        out.median * 1000,
                        args.device,
                        timesteps,
                        mem_bytes,
                        1024,
                        use_xformers,
                        None,
                    ]
                )

    with open("benchmark/artifacts/all.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)


def muse_benchmark(resolution, batch_size, timesteps, use_xformers, use_fused_residual_norm):
    model = "williamberman/muse_research_run_benchmarking_512_output"
    device = "cuda"
    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model, subfolder="text_encoder")

    text_encoder = CLIPTextModelWithProjection.from_pretrained(model, subfolder="text_encoder")
    text_encoder.to(device=device, dtype=dtype)

    vae = VQGANModel.from_pretrained(model, subfolder="vae")
    vae.to(device=device, dtype=dtype)

    transformer = MaskGiTUViT(
        use_fused_mlp=False,
        use_fused_residual_norm=use_fused_residual_norm,
        force_down_up_sample=resolution == 512,
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
                f" use_xformers: {use_xformers}, use_fused_residual_norm: {use_fused_residual_norm}"
            ),
            description=model,
        ).blocked_autorange(min_run_time=1)

    return measure_max_memory_allocated(fn)


def wurst_benchmark(batch_size, use_xformers):
    model = "warp-ai/wuerstchen"
    device = "cuda"
    dtype = torch.float16

    pipe = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=dtype).to(device)

    if use_xformers:
        pipe.enable_xformers_memory_efficient_attention()

    def benchmark_fn():
        pipe(
            prompt,
            height=1024,
            width=1024,
            prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
            prior_guidance_scale=4.0,
            num_images_per_prompt=batch_size,
        )

    # warmup
    benchmark_fn()

    def fn():
        return Timer(
            stmt="benchmark_fn()",
            globals={"benchmark_fn": benchmark_fn},
            num_threads=num_threads,
            label=f"batch_size: {batch_size}, dtype: {dtype}, use_xformers: {use_xformers}",
            description=model,
        ).blocked_autorange(min_run_time=1)

    return measure_max_memory_allocated(fn)


def sdxl_benchmark(batch_size, timesteps, use_xformers, gpu_type):
    model = "stabilityai/stable-diffusion-xl-base-1.0"
    device = "cuda"
    dtype = torch.float16

    pipe = StableDiffusionXLPipeline.from_pretrained(model, torch_dtype=dtype)
    pipe = pipe.to(device)

    if use_xformers:
        pipe.enable_xformers_memory_efficient_attention()

    if gpu_type == "4090" and batch_size == 8:
        output_type = "latent"
    else:
        output_type = "pil"

    def benchmark_fn():
        pipe(prompt, num_inference_steps=timesteps, num_images_per_prompt=batch_size, output_type=output_type)

    pipe(prompt, num_inference_steps=2, num_images_per_prompt=batch_size, output_type=output_type)

    def fn():
        return Timer(
            stmt="benchmark_fn()",
            globals={"benchmark_fn": benchmark_fn},
            num_threads=num_threads,
            label=f"batch_size: {batch_size}, dtype: {dtype}, timesteps {timesteps}, use_xformers: {use_xformers}",
            description=model,
        ).blocked_autorange(min_run_time=1)

    return measure_max_memory_allocated(fn)


def lcm_benchmark(batch_size, timesteps, use_xformers):
    model = "SimianLuo/LCM_Dreamshaper_v7"
    device = "cuda"
    dtype = torch.float16

    scheduler = LCMScheduler.from_pretrained(model, subfolder="scheduler")

    pipe = LatentConsistencyModelPipeline.from_pretrained(model, torch_dtype=dtype, scheduler=scheduler)
    pipe.to(device)

    if use_xformers:
        pipe.enable_xformers_memory_efficient_attention()

    def benchmark_fn():
        pipe(prompt, num_inference_steps=timesteps, num_images_per_prompt=batch_size)

    pipe(prompt, num_inference_steps=2, num_images_per_prompt=batch_size)

    def fn():
        return Timer(
            stmt="benchmark_fn()",
            globals={"benchmark_fn": benchmark_fn},
            num_threads=num_threads,
            label=f"batch_size: {batch_size}, dtype: {dtype}, timesteps {timesteps}, use_xformers: {use_xformers}",
            description=model,
        ).blocked_autorange(min_run_time=1)

    return measure_max_memory_allocated(fn)


def ssd_1b_benchmark(batch_size, timesteps, use_xformers, gpu_type):
    model = "segmind/SSD-1B"
    device = "cuda"
    dtype = torch.float16

    pipe = StableDiffusionXLPipeline.from_pretrained(model, torch_dtype=dtype, use_safetensors=True, variant="fp16")
    pipe.to(device)

    if use_xformers:
        pipe.enable_xformers_memory_efficient_attention()

    if gpu_type == "4090" and batch_size == 8:
        output_type = "latent"
    else:
        output_type = "pil"

    def benchmark_fn():
        pipe(prompt, num_inference_steps=timesteps, num_images_per_prompt=batch_size, output_type=output_type)

    pipe(prompt, num_inference_steps=2, num_images_per_prompt=batch_size, output_type=output_type)

    def fn():
        return Timer(
            stmt="benchmark_fn()",
            globals={"benchmark_fn": benchmark_fn},
            num_threads=num_threads,
            label=f"batch_size: {batch_size}, dtype: {dtype}, timesteps {timesteps}, use_xformers: {use_xformers}",
            description=model,
        ).blocked_autorange(min_run_time=1)

    return measure_max_memory_allocated(fn)


def sd_benchmark(batch_size, timesteps, use_xformers):
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
        )

    pipe(prompt, num_images_per_prompt=batch_size, num_inference_steps=2)

    def fn():
        return Timer(
            stmt="benchmark_fn()",
            globals={"benchmark_fn": benchmark_fn},
            num_threads=num_threads,
            label=f"batch_size: {batch_size}, dtype: {dtype}, timesteps {timesteps}, use_xformers: {use_xformers}",
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
