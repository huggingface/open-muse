from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    PreTrainedTokenizer,
    T5EncoderModel,
)

from .modeling_maskgit_vqgan import MaskGitVQGAN
from .modeling_transformer import MaskGitTransformer


class PipelineMuse:
    def __init__(
        self,
        text_encoder: Union[T5EncoderModel, CLIPTextModel],
        tokenizer: PreTrainedTokenizer,
        vae: MaskGitVQGAN,
        transformer: MaskGitTransformer,
    ) -> None:
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.vae = vae
        self.transformer = transformer
        self.device = "cpu"

    def to(self, device="cpu", dtype=torch.float32) -> None:
        self.text_encoder.to(device, dtype=dtype)
        self.vae.to(device, dtype=dtype)
        self.transformer.to(device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        return self

    @torch.no_grad()
    def __call__(
        self,
        text: Union[str, List[str]],
        timesteps: int = 8,
        guidance_scale: float = 8.0,
        temperature: float = 1.0,
        topk_filter_thres: float = 0.9,
        num_images_per_prompt: int = 1,
        use_maskgit_generate: bool = False,
    ):
        if isinstance(text, str):
            text = [text]

        input_ids = self.tokenizer(
            text, return_tensors="pt", padding="max_length", truncation=True, max_length=16
        ).input_ids  # TODO: remove hardcode
        input_ids = input_ids.to(self.device)
        encoder_hidden_states = self.text_encoder(input_ids).last_hidden_state

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = encoder_hidden_states.shape
        encoder_hidden_states = encoder_hidden_states.repeat(1, num_images_per_prompt, 1)
        encoder_hidden_states = encoder_hidden_states.view(bs_embed * num_images_per_prompt, seq_len, -1)

        generate = self.transformer.generate
        if use_maskgit_generate:
            generate = self.transformer.generate2

        generated_tokens = generate(
            encoder_hidden_states=encoder_hidden_states,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            temperature=temperature,
            topk_filter_thres=topk_filter_thres,
        )

        images = self.vae.decode_code(generated_tokens)

        # Convert to PIL images
        images = [self.to_pil_image(image) for image in images]
        return images

    def to_pil_image(self, image: torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        image = 2.0 * image - 1.0
        image = np.clip(image, -1.0, 1.0)
        image = (image + 1.0) / 2.0
        image = (255 * image).astype(np.uint8)
        image = Image.fromarray(image).convert("RGB")
        return image

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = None,
        text_encoder_path: Optional[str] = None,
        vae_path: Optional[str] = None,
        transformer_path: Optional[str] = None,
    ) -> None:
        """
        Instantiate a PipelineMuse from a pretrained model. Either model_name_or_path or all of text_encoder_path, vae_path, and
        transformer_path must be provided.
        """
        if model_name_or_path is None:
            if text_encoder_path is None or vae_path is None or transformer_path is None:
                raise ValueError(
                    "If model_name_or_path is None, then text_encoder_path, vae_path, and transformer_path must be"
                    " provided."
                )

            text_encoder = T5EncoderModel.from_pretrained(text_encoder_path)
            tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
            vae = MaskGitVQGAN.from_pretrained(vae_path)
            transformer = MaskGitTransformer.from_pretrained(transformer_path)
        else:
            text_encoder = T5EncoderModel.from_pretrained(model_name_or_path, subfolder="text_encoder")
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, subfolder="text_encoder")
            vae = MaskGitVQGAN.from_pretrained(model_name_or_path, subfolder="vae")
            transformer = MaskGitTransformer.from_pretrained(model_name_or_path, subfolder="transformer")

        return cls(text_encoder, tokenizer, vae, transformer)
