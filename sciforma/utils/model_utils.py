from typing import List, Union
import re
import itertools
import os
import os.path as osp
import math
import argparse
import torch
from torch.hub import download_url_to_file
import json
from accelerate.logging import get_logger
from diffusers import (
    AutoencoderKLQwenImage,
    BitsAndBytesConfig,
    FlowMatchEulerDiscreteScheduler,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
from peft import LoraConfig, prepare_model_for_kbit_training, set_peft_model_state_dict


from mmengine import Config, DictAction
import random
import numpy as np
import copy

logger = get_logger(__name__)
from sciforma.registry import MODELS

def initialize_QwenImage_all_models(config):
    logger.info(f"[INFO] start load tokenizer")
    tokenizer = Qwen2Tokenizer.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=config.revision,
        cache_dir=config.cache_dir,
        token=config.huggingface_token,
    )

    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=config.revision,
        variant=config.variant,
        cache_dir=config.cache_dir,
        token=config.huggingface_token,
        torch_dtype=config.weight_dtype
    )
    text_encoder.requires_grad_(False)

    logger.info(f"[INFO] start load vae")
    vae = AutoencoderKLQwenImage.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="vae",
        revision=config.revision,
        variant=config.variant,
        cache_dir=config.cache_dir,
        token=config.huggingface_token,
    )
    vae.requires_grad_(False)

    logger.info(f"[INFO] start load mmdit")
    quantization_config = None
    if config.bnb_quantization_config_path is not None:
        with open(config.bnb_quantization_config_path, "r") as f:
            config_kwargs = json.load(f)
            if "load_in_4bit" in config_kwargs and config_kwargs["load_in_4bit"]:
                config_kwargs["bnb_4bit_compute_dtype"] = config.weight_dtype
        quantization_config = BitsAndBytesConfig(**config_kwargs)

    model_type = config.transformer_cfg.pop('type')
    #transformer = MODELS.get(model_type).from_pretrained(
    transformer = QwenImageTransformer2DModel.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=config.revision,
        variant=config.variant,
        cache_dir=config.cache_dir,
        token=config.huggingface_token,
        quantization_config=quantization_config,
        torch_dtype=config.weight_dtype,
    )
    
    logger.info(f"[INFO] finish load mmdit model: {model_type} {transformer}")
    if config.bnb_quantization_config_path is not None:
        transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)
        
    
    if config.use_lora:
        # 使用lora微调
        logger.info(f"[INFO] Using LoRA fine-tuning ...")
        transformer.requires_grad_(False)
    else:
        logger.info(f"[INFO] Fine-tuning the full model ...")
        transformer.requires_grad_(True)

    if config.gradient_checkpointing:
        logger.info(f"[INFO] Enabling gradient checkpointing to model {model_type} ...")
        transformer.enable_gradient_checkpointing()

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="scheduler",
        revision=config.revision,
        cache_dir=config.cache_dir,
        token=config.huggingface_token,
        shift=3.0
    )

    logger.info(f"[INFO] calulate vae scale factors, mean std ...")

    vae_scale_factor = 2 ** len(vae.temperal_downsample) # 几层下采样，8下采样

    text_encoding_pipeline = QwenImagePipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
    )

    return (
        vae,
        transformer,
        tokenizer,
        text_encoder,
        noise_scheduler,
        text_encoding_pipeline,
        vae_scale_factor,
    )


def initialize_VAE_Qwen25_only(config):
    logger.info(f"[INFO] start load tokenizer")
    tokenizer = Qwen2Tokenizer.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=config.revision,
        cache_dir=config.cache_dir,
        token=config.huggingface_token,
    )

    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=config.revision,
        variant=config.variant,
        cache_dir=config.cache_dir,
        token=config.huggingface_token,
        torch_dtype=config.weight_dtype
    )
    text_encoder.requires_grad_(False)
    
    logger.info(f"[INFO] start load vae")
    vae = AutoencoderKLQwenImage.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="vae",
        revision=config.revision,
        variant=config.variant,
        cache_dir=config.cache_dir,
        token=config.huggingface_token,
    )
    vae.requires_grad_(False)

    logger.info(f"[INFO] calulate vae scale factors, mean std ...")

    vae_scale_factor = 2 ** len(vae.temperal_downsample) # 几层下采样，8下采样
    
    text_encoding_pipeline = QwenImagePipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
    )
    
    return (
        vae,
        tokenizer,
        text_encoder,
        text_encoding_pipeline,
        vae_scale_factor,
    )
    
    
def prepare_latent_image_ids(height, width, device, dtype):
    # single image version
    # debug: 到底要不要vae scale factor?
    latent_image_ids = torch.zeros(height // 2, width // 2, 3) # [h/2, w/2, 3]
    latent_image_ids[..., 0] = layer_idx # use the first dimension for layer representation
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]
    latent_image_ids = latent_image_ids.flatten(0, 1) # size, 3
    
    return latent_image_ids.to(device=device, dtype=dtype)


def compute_text_embeddings(prompt, text_encoding_pipeline, max_sequence_length):
        with torch.no_grad():
            prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                prompt=prompt, max_sequence_length=max_sequence_length
            )
        return prompt_embeds, prompt_embeds_mask
    
    
## simple naive pack & unpack (maybe in the future we may use more complex methods)
def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

    return latents


# ─── SciForma model loading utilities ─────────────────────────────────────────

def load_ema_into_transformer(ema_weights_path: str, transformer) -> None:
    """
    Load shadow_params from a SciForma ema_weights.pt into the transformer.

    The EMA file has the structure:
        {"shadow_params": [tensor, tensor, ...]}
    where the list is ordered identically to transformer.parameters().
    """
    ema_state = torch.load(ema_weights_path, map_location="cpu", weights_only=False)
    shadow_params = ema_state.get("shadow_params")
    if shadow_params is None:
        raise ValueError(f"'shadow_params' not found in {ema_weights_path}")

    param_names = [n for n, _ in transformer.named_parameters()]
    if len(shadow_params) != len(param_names):
        raise ValueError(
            f"Param count mismatch: shadow_params={len(shadow_params)}, "
            f"transformer params={len(param_names)}"
        )
    with torch.no_grad():
        for name, shadow in zip(param_names, shadow_params):
            param = dict(transformer.named_parameters())[name]
            param.data.copy_(shadow.to(param.device, dtype=param.dtype))


def load_sciforma_transformer(
    model_id: str = "microsoft/SciForma-9B",
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    ema_weights_path: str = None,
    hf_token: str = None,
):
    """
    Load a SciForma transformer from HuggingFace or a local EMA checkpoint.

    Args:
        model_id: HF model ID (e.g., 'microsoft/SciForma-9B') or local path.
        dtype: Model dtype (default: bfloat16).
        device: Target device.
        ema_weights_path: Optional path to local ema_weights.pt.
        hf_token: HuggingFace token for private repos.

    Examples:
        # From HuggingFace (recommended)
        transformer = load_sciforma_transformer("microsoft/SciForma-9B")

        # From local EMA checkpoint
        transformer = load_sciforma_transformer(
            "black-forest-labs/FLUX.2-klein-base-9B",
            ema_weights_path="/path/to/ema_weights.pt"
        )
    """
    import os
    from diffusers import Flux2Transformer2DModel

    token = hf_token or os.environ.get("HF_TOKEN")

    transformer = Flux2Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=dtype,
        token=token,
    )

    if ema_weights_path:
        load_ema_into_transformer(ema_weights_path, transformer)

    return transformer.to(device=device, dtype=dtype)
