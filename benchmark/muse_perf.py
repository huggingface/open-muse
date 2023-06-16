import torch
from torch.utils.benchmark import Timer, Compare
from muse.modeling_taming_vqgan import VQGANModel
from muse.modeling_transformer import MaskGiTUViT
from muse import PipelineMuse, PaellaVQModel
import multiprocessing
import traceback
from argparse import ArgumentParser
import csv
from diffusers import UNet2DConditionModel, AutoencoderKL, StableDiffusionPipeline

from transformers import CLIPTextModel, AutoTokenizer, CLIPTokenizer

torch.manual_seed(0)
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")

num_threads = torch.get_num_threads()
prompt = "A high tech solarpunk utopia in the Amazon rainforest"


all_models = [
    "openMUSE/muse-laiona6-uvit-clip-220k",
    "runwayml/stable-diffusion-v1-5",
    "williamberman/laiona6plus_uvit_clip_f8",
]

all_batch_sizes = [1, 2, 4, 8, 16, 32]

all_compiled = [None, "default", "reduce-overhead"]

all_components = ["backbone", "vae", "full"]

all_devices = ["4090", "t4", "a100", "cpu"]

all_timesteps = [12, 20]

skip = [
    # 4090 backbone
    ("4090", "runwayml/stable-diffusion-v1-5", "backbone", 32, "reduce-overhead"),
    # 4090 full
    ("4090", "runwayml/stable-diffusion-v1-5", "full", 8, "reduce-overhead"),
    ("4090", "runwayml/stable-diffusion-v1-5", "full", 16, "reduce-overhead"),
    ("4090", "runwayml/stable-diffusion-v1-5", "full", 32, "default"),
    ("4090", "runwayml/stable-diffusion-v1-5", "full", 32, "reduce-overhead"),
    # t4 backbone
    ("t4", "runwayml/stable-diffusion-v1-5", "backbone", 8, "reduce-overhead"),
    ("t4", "runwayml/stable-diffusion-v1-5", "backbone", 16, "reduce-overhead"),
    ("t4", "runwayml/stable-diffusion-v1-5", "backbone", 32, "reduce-overhead"),
    # t4 vae
    ("t4", "runwayml/stable-diffusion-v1-5", "vae", 32),
    # t4 full
    ("t4", "runwayml/stable-diffusion-v1-5", "full", 4, "reduce-overhead"),
    ("t4", "runwayml/stable-diffusion-v1-5", "full", 8, "reduce-overhead"),
    ("t4", "runwayml/stable-diffusion-v1-5", "full", 16, "reduce-overhead"),
    ("t4", "runwayml/stable-diffusion-v1-5", "full", 32),
    # cpu full
    ("cpu", "runwayml/stable-diffusion-v1-5", "full", 2),
    ("cpu", "runwayml/stable-diffusion-v1-5", "full", 4),
    ("cpu", "runwayml/stable-diffusion-v1-5", "full", 8),
    ("cpu", "runwayml/stable-diffusion-v1-5", "full", 16),
    ("cpu", "runwayml/stable-diffusion-v1-5", "full", 32),
]


def main():
    args = ArgumentParser()
    args.add_argument("--device", required=True)

    args = args.parse_args()

    assert args.device in all_devices

    if args.device in ["4090", "a100", "t4"]:
        dtype = torch.float16
        torch_device = "cuda"
    elif args.device in ["cpu"]:
        dtype = torch.float32
        torch_device = "cpu"
    else:
        assert False

    csv_data = []

    for model in all_models:
        if (args.device, model) in skip:
            continue

        for component in all_components:
            if (args.device, model, component) in skip:
                continue

            for batch_size in all_batch_sizes:
                if (args.device, model, component, batch_size) in skip:
                    continue

                for compiled in all_compiled:
                    if (args.device, model, component, batch_size, compiled) in skip:
                        continue

                    if component == "full":
                        all_timesteps_ = all_timesteps
                    else:
                        all_timesteps_ = [None]

                    for timesteps in all_timesteps_:
                        if component in ["backbone", "vae"]:
                            label = f"single pass {component}"
                        elif component == "full":
                            label = "full pipeline"
                        else:
                            assert False

                        label = f"{label}, batch_size: {batch_size}, dtype: {dtype}, timesteps {timesteps}"
                        description = f"{model}, compiled {compiled}"

                        print(label)
                        print(description)

                        inputs = [
                            torch_device,
                            dtype,
                            compiled,
                            batch_size,
                            model,
                            label,
                            description,
                            timesteps,
                        ]

                        fn = model_config[model][component]["fn"]

                        out, mem_bytes = run_in_subprocess(fn, inputs=inputs)

                        median = out.median * 1000

                        mean = out.mean * 1000

                        iqr = out.iqr * 1000

                        csv_data.append(
                            [
                                batch_size,
                                model,
                                str(compiled),
                                median,
                                mean,
                                args.device,
                                component,
                                timesteps,
                                mem_bytes,
                                iqr
                            ]
                        )

                        Compare([out]).print()
                        print("*******")

    with open("artifacts/all.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)


def muse_benchmark_transformer_backbone(in_queue, out_queue, timeout):
    wrap_subprocess_fn(
        in_queue, out_queue, timeout, _muse_benchmark_transformer_backbone
    )


def _muse_benchmark_transformer_backbone(
    device, dtype, compiled, batch_size, model, label, description, timesteps
):
    text_encoder = CLIPTextModel.from_pretrained(model, subfolder="text_encoder").to(
        device
    )

    tokenizer = AutoTokenizer.from_pretrained(model, subfolder="text_encoder")

    text_tokens = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).input_ids
    text_tokens = text_tokens.to(device)

    encoder_hidden_states = text_encoder(text_tokens).last_hidden_state

    encoder_hidden_states = encoder_hidden_states.expand(batch_size, -1, -1)

    encoder_hidden_states = encoder_hidden_states.to(dtype)

    transformer = MaskGiTUViT.from_pretrained(model, subfolder="transformer")

    transformer = transformer.to(device=device, dtype=dtype)

    if compiled is not None:
        transformer = torch.compile(transformer, mode=compiled)

    image_tokens = torch.full(
        (batch_size, 256), fill_value=5, dtype=torch.long, device=device
    )

    def benchmark_fn():
        transformer(image_tokens, encoder_hidden_states=encoder_hidden_states)

    benchmark_fn()

    def fn():
        return Timer(
            stmt="benchmark_fn()",
            globals={"benchmark_fn": benchmark_fn},
            num_threads=num_threads,
            label=label,
            description=description,
        ).blocked_autorange(min_run_time=1)

    if device == "cuda":
        return measure_max_memory_allocated(fn)
    else:
        return fn(), None


def sd_benchmark_unet_backbone(in_queue, out_queue, timeout):
    wrap_subprocess_fn(in_queue, out_queue, timeout, _sd_benchmark_unet_backbone)


def _sd_benchmark_unet_backbone(
    device, dtype, compiled, batch_size, model, label, description, timesteps
):
    unet = UNet2DConditionModel.from_pretrained(model, subfolder="unet")

    unet = unet.to(device=device, dtype=dtype)

    if compiled is not None:
        unet = torch.compile(unet, mode=compiled)

    text_encoder = CLIPTextModel.from_pretrained(model, subfolder="text_encoder").to(
        device
    )

    tokenizer = CLIPTokenizer.from_pretrained(model, subfolder="tokenizer")

    text_tokens = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).input_ids
    text_tokens = text_tokens.to(device)

    encoder_hidden_states = text_encoder(text_tokens).last_hidden_state

    encoder_hidden_states = encoder_hidden_states.expand(batch_size, -1, -1)

    encoder_hidden_states = encoder_hidden_states.to(dtype)

    latent_image = torch.randn((batch_size, 4, 64, 64), dtype=dtype, device=device)

    t = torch.randint(1, 999, (batch_size,), dtype=dtype, device=device)

    def benchmark_fn():
        unet(latent_image, timestep=t, encoder_hidden_states=encoder_hidden_states)

    benchmark_fn()

    def fn():
        return Timer(
            stmt="benchmark_fn()",
            globals={"benchmark_fn": benchmark_fn},
            num_threads=num_threads,
            label=label,
            description=description,
        ).blocked_autorange(min_run_time=1)

    if device == "cuda":
        return measure_max_memory_allocated(fn)
    else:
        return fn(), None


def muse_benchmark_vae(in_queue, out_queue, timeout):
    wrap_subprocess_fn(in_queue, out_queue, timeout, _muse_benchmark_vae)


def _muse_benchmark_vae(
    device, dtype, compiled, batch_size, model, label, description, timesteps
):
    vae_cls = model_config[model]["vae"]["cls"]
    vae = vae_cls.from_pretrained(model, subfolder="vae")

    vae = vae.to(device=device, dtype=dtype)

    if compiled is not None:
        vae = torch.compile(vae, mode=compiled)

    image_tokens = torch.full(
        (batch_size, 256), fill_value=5, dtype=torch.long, device=device
    )

    def benchmark_fn():
        vae.decode_code(image_tokens)

    benchmark_fn()

    def fn():
        return Timer(
            stmt="benchmark_fn()",
            globals={"benchmark_fn": benchmark_fn},
            num_threads=num_threads,
            label=label,
            description=description,
        ).blocked_autorange(min_run_time=1)

    if device == "cuda":
        return measure_max_memory_allocated(fn)
    else:
        return fn(), None


def sd_benchmark_vae(in_queue, out_queue, timeout):
    wrap_subprocess_fn(in_queue, out_queue, timeout, _sd_benchmark_vae)


def _sd_benchmark_vae(
    device, dtype, compiled, batch_size, model, label, description, timesteps
):
    vae = AutoencoderKL.from_pretrained(model, subfolder="vae")

    vae = vae.to(device=device, dtype=dtype)

    if compiled is not None:
        vae = torch.compile(vae, mode=compiled)

    latent_image = torch.randn((batch_size, 4, 64, 64), dtype=dtype, device=device)

    def benchmark_fn():
        vae.decode(latent_image)

    benchmark_fn()

    def fn():
        return Timer(
            stmt="benchmark_fn()",
            globals={"benchmark_fn": benchmark_fn},
            num_threads=num_threads,
            label=label,
            description=description,
        ).blocked_autorange(min_run_time=1)

    if device == "cuda":
        return measure_max_memory_allocated(fn)
    else:
        return fn(), None


def muse_benchmark_full(in_queue, out_queue, timeout):
    wrap_subprocess_fn(in_queue, out_queue, timeout, _muse_benchmark_full)


def _muse_benchmark_full(
    device, dtype, compiled, batch_size, model, label, description, timesteps
):
    tokenizer = AutoTokenizer.from_pretrained(model, subfolder="text_encoder")

    text_encoder = CLIPTextModel.from_pretrained(model, subfolder="text_encoder").to(
        device=device, dtype=dtype
    )

    vae_cls = model_config[model]["vae"]["cls"]
    vae = vae_cls.from_pretrained(model, subfolder="vae")

    vae = vae.to(device=device, dtype=dtype)

    transformer = MaskGiTUViT.from_pretrained(model, subfolder="transformer")

    transformer = transformer.to(device=device, dtype=dtype)

    if compiled is not None:
        vae = torch.compile(vae, mode=compiled)
        transformer = torch.compile(transformer, mode=compiled)

    pipe = PipelineMuse(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
    )
    pipe.device = device
    pipe.dtype = dtype

    def benchmark_fn():
        pipe(prompt, num_images_per_prompt=batch_size, timesteps=timesteps)

    pipe(prompt, num_images_per_prompt=batch_size, timesteps=2)

    def fn():
        return Timer(
            stmt="benchmark_fn()",
            globals={"benchmark_fn": benchmark_fn},
            num_threads=num_threads,
            label=label,
            description=description,
        ).blocked_autorange(min_run_time=1)

    if device == "cuda":
        return measure_max_memory_allocated(fn)
    else:
        return fn(), None


def sd_benchmark_full(in_queue, out_queue, timeout):
    wrap_subprocess_fn(in_queue, out_queue, timeout, _sd_benchmark_full)


def _sd_benchmark_full(
    device, dtype, compiled, batch_size, model, label, description, timesteps
):
    tokenizer = CLIPTokenizer.from_pretrained(model, subfolder="tokenizer")

    text_encoder = CLIPTextModel.from_pretrained(model, subfolder="text_encoder").to(
        device=device, dtype=dtype
    )

    vae = AutoencoderKL.from_pretrained(model, subfolder="vae")

    vae = vae.to(device=device, dtype=dtype)

    unet = UNet2DConditionModel.from_pretrained(model, subfolder="unet")

    unet = unet.to(device=device, dtype=dtype)

    if compiled is not None:
        vae = torch.compile(vae, mode=compiled)
        unet = torch.compile(unet, mode=compiled)

    pipe = StableDiffusionPipeline.from_pretrained(
        model,
        vae=vae,
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        safety_checker=None,
    )

    def benchmark_fn():
        pipe(prompt, num_images_per_prompt=batch_size, num_inference_steps=timesteps)

    pipe(prompt, num_images_per_prompt=batch_size, num_inference_steps=2)

    def fn():
        return Timer(
            stmt="benchmark_fn()",
            globals={"benchmark_fn": benchmark_fn},
            num_threads=num_threads,
            label=label,
            description=description,
        ).blocked_autorange(min_run_time=1)

    if device == "cuda":
        return measure_max_memory_allocated(fn)
    else:
        return fn(), None


def measure_max_memory_allocated(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    rv = fn()

    mem_bytes = torch.cuda.max_memory_allocated()

    return rv, mem_bytes


def wrap_subprocess_fn(in_queue, out_queue, timeout, fn):
    error = None
    out = None

    try:
        args = in_queue.get(timeout=timeout)
        out = fn(*args)

    except Exception:
        error = f"{traceback.format_exc()}"

    results = {"error": error, "out": out}
    out_queue.put(results, timeout=timeout)
    out_queue.join()


def run_in_subprocess(target_func, inputs=None):
    timeout = None

    ctx = multiprocessing.get_context("spawn")

    input_queue = ctx.Queue(1)
    output_queue = ctx.JoinableQueue(1)

    # We can't send `unittest.TestCase` to the child, otherwise we get issues regarding pickle.
    input_queue.put(inputs, timeout=timeout)

    process = ctx.Process(target=target_func, args=(input_queue, output_queue, timeout))
    process.start()
    # Kill the child process if we can't get outputs from it in time: otherwise, the hanging subprocess prevents
    # the test to exit properly.
    try:
        results = output_queue.get(timeout=timeout)
        output_queue.task_done()
    except Exception as e:
        process.terminate()
        raise e
    process.join(timeout=timeout)

    if results["error"] is not None:
        raise Exception(results["error"])

    return results["out"]


model_config = {
    "openMUSE/muse-laiona6-uvit-clip-220k": {
        "backbone": {
            "fn": muse_benchmark_transformer_backbone,
        },
        "vae": {
            "fn": muse_benchmark_vae,
            "cls": VQGANModel,
        },
        "full": {"fn": muse_benchmark_full},
    },
    "runwayml/stable-diffusion-v1-5": {
        "backbone": {
            "fn": sd_benchmark_unet_backbone,
        },
        "vae": {
            "fn": sd_benchmark_vae,
        },
        "full": {"fn": sd_benchmark_full},
    },
    "williamberman/laiona6plus_uvit_clip_f8": {
        "backbone": {
            "fn": muse_benchmark_transformer_backbone,
        },
        "vae": {
            "fn": muse_benchmark_vae,
            "cls": PaellaVQModel,
        },
        "full": {"fn": muse_benchmark_full},
    },
}


if __name__ == "__main__":
    main()
