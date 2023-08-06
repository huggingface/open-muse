import torch

from muse import MaskGitTransformer, MaskGiTUViT, VQGANModel, MaskGitUVit

# codebook_size = 1024
# num_classes = 1000
# model = MaskGitTransformer(
#     vocab_size=codebook_size + num_classes + 1,
#     hidden_size=32,
#     num_hidden_layers=2,
#     num_attention_heads=4,
#     intermediate_size=64,
#     max_position_embeddings=4 + 1,  # +1 for the class token
#     num_vq_tokens=4,
#     codebook_size=codebook_size,
#     num_classes=num_classes,
# )

# input_ids = torch.randint(0, 100, (1, 4))
# output = model(input_ids)
# assert output.shape == (1, 4, model.config.vocab_size)

# class_ids = torch.tensor([1], dtype=torch.long)
# gen_ids = model.generate(class_ids, timesteps=4)
# assert gen_ids.shape == (1, 4)
# print(gen_ids)

# model.enable_gradient_checkpointing()
# output = model(input_ids)
# assert output.shape == (1, 4, model.config.vocab_size)


# model = MaskGitTransformer(
#     vocab_size=100,
#     hidden_size=32,
#     num_hidden_layers=2,
#     num_attention_heads=4,
#     intermediate_size=64,
#     max_position_embeddings=4,
#     add_cross_attention=True,
#     encoder_hidden_size=64,
# )

# input_ids = torch.randint(0, 100, (1, 4))
# encoder_hidden_states = torch.randn(1, 4, 64)
# output = model(input_ids, encoder_hidden_states=encoder_hidden_states)
# assert output.shape == (1, 4, 100)

# vq = VQGANModel(
#     resolution=32,
#     num_channels=3,
#     hidden_channels=32,
#     channel_mult=(1, 2),
#     num_res_blocks=2,
#     attn_resolutions=tuple(),
#     z_channels=64,
#     num_embeddings=512,
#     quantized_embed_dim=64,
# )
# image = torch.randn(1, 3, 32, 32)
# out = vq(image)[0]
# assert out.shape == (1, 3, 32, 32)

#### TORCH PROFILING
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler, record_function
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer
import datetime
import time
num_vq_tokens = 1024
codebook_size = 8192
# encoder_hidden_size = 768
encoder_hidden_size = num_vq_tokens
profiler_export_path = "./tb_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
batch_size = 4
# transformer = MaskGiTUViT(
#     vocab_size=codebook_size+1,
#     hidden_size=768,
#     in_channels=384,
#     block_out_channels=(384, 512),
#     encoder_hidden_size=encoder_hidden_size,
#     add_cross_attention=True,
#     num_res_blocks=1,
#     block_has_attention=(False, True),
#     block_num_heads=8,
#     num_hidden_layers=1,
#     use_encoder_layernorm=False,
#     codebook_size=codebook_size,
#     num_vq_tokens=num_vq_tokens,
#     use_vannilla_resblock=False,
#     use_codebook_size_for_output=True,
# ).eval()
transformer = MaskGitUVit(
    embedding_size=512,
    num_res_blocks=3,
    hidden_size=1024,
    encoder_hidden_size=encoder_hidden_size,
    input_vocab_size=8256,  # (8192 + 1 for <mask> = 8193 but 8256 is the next multiple of 8)
    output_vocab_size=8192,
    num_transformer_layers=22,
    ln_elementwise_affine=True,
    layer_norm_eps=1e-6,
    num_attention_heads=16,
    dropout_p=0.0,
    use_bias=False,
).eval()
device = "cuda"
dtype = torch.float16

# transformer = MaskGiTUViT.from_pretrained(model, subfolder="transformer")

transformer = transformer.to(device=device, dtype=dtype)
transformer = transformer.to(memory_format=torch.channels_last)

# transformer = torch.compile(transformer, mode="reduce-overhead")

    
with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1, skip_first=1),
            on_trace_ready=tensorboard_trace_handler(profiler_export_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
    
    for _ in range(6):
        prof.step()

        with record_function("model_inference"):
            input_ids = torch.randint(0, codebook_size, (batch_size, num_vq_tokens), device=device)
            enc = torch.randn(batch_size, 4, encoder_hidden_size, device=device, dtype=dtype)
            output = transformer(input_ids, encoder_hidden_states=enc)
            # assert output.shape == (batch_size, num_vq_tokens, codebook_size)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
torch.cuda.synchronize()
start_time = time.time()
input_ids = torch.randint(0, codebook_size, (batch_size, num_vq_tokens), device=device)
enc = torch.randn(batch_size, 4, encoder_hidden_size, device=device, dtype=dtype)
output = transformer(input_ids, encoder_hidden_states=enc)
torch.cuda.synchronize()
print("Time taken: ", time.time() - start_time)

# Self CPU time total: 143.205ms
# Self CUDA time total: 100.064ms

# Time taken:  0.033339500427246094