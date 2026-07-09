"""
Model Factory Module for SciForma

This module provides a unified interface for loading different diffusion models
(e.g., QwenImage, Flux2Klein) through a registry-based approach.

Usage in config:
    model_cfg = dict(
        type='QwenImage',  # or 'Flux2Klein'
        pretrained_model_name_or_path='...',
        ...
    )

The factory will automatically resolve:
- VAE class
- Transformer class
- Text Encoder class
- Tokenizer class
- Pipeline class
- Scheduler class
"""

import importlib
import os
from typing import Dict, Any, Tuple, Optional, Type
from dataclasses import dataclass, field
import json
import copy

import torch
from accelerate.logging import get_logger
from peft import prepare_model_for_kbit_training

logger = get_logger(__name__)


# Try to import from _local_secrets.py for HF token
try:
    from _local_secrets import HF_TOKEN as _SECRETS_HF_TOKEN
except ImportError:
    _SECRETS_HF_TOKEN = None


def get_hf_token(config) -> Optional[str]:
    """Get HuggingFace token from config, environment variable, or _local_secrets.py.
    
    Priority:
    1. HF_TOKEN environment variable (highest - for explicit override)
    2. config.huggingface_token (if set and not None)
    3. _local_secrets.py HF_TOKEN
    4. None
    """
    # 1. Environment variable (highest priority for cloud/explicit override)
    env_token = os.environ.get('HF_TOKEN')
    if env_token:
        return env_token
    
    # 2. Config file token
    token = getattr(config, 'huggingface_token', None)
    if token:
        return token
    
    # 3. _local_secrets.py
    if _SECRETS_HF_TOKEN:
        return _SECRETS_HF_TOKEN
    
    # 4. None
    return None


@dataclass
class ModelSpec:
    """Specification for a model family (e.g., QwenImage, Flux2Klein)"""
    # Module paths (e.g., 'diffusers' or 'sciforma.models')
    vae_module: str = "diffusers"
    transformer_module: str = "diffusers"
    text_encoder_module: str = "transformers"
    tokenizer_module: str = "transformers"
    pipeline_module: str = "diffusers"
    scheduler_module: str = "diffusers"
    
    # Class names
    vae_class: str = ""
    transformer_class: str = ""
    text_encoder_class: str = ""
    tokenizer_class: str = ""
    pipeline_class: str = ""
    scheduler_class: str = "FlowMatchEulerDiscreteScheduler"
    
    # Subfolder names for loading from pretrained
    vae_subfolder: str = "vae"
    transformer_subfolder: str = "transformer"
    text_encoder_subfolder: str = "text_encoder"
    tokenizer_subfolder: str = "tokenizer"
    scheduler_subfolder: str = "scheduler"
    
    # Additional config
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    vae_scale_factor_attr: str = "temperal_downsample"  # attribute to get scale factor
    latents_mean_attr: str = "latents_mean"
    latents_std_attr: str = "latents_std"
    z_dim_attr: str = "z_dim"
    
    # Whether the model uses specific features
    has_dual_text_encoder: bool = False
    text_encoder_two_class: str = ""
    text_encoder_two_module: str = "transformers"
    text_encoder_two_subfolder: str = "text_encoder_2"
    tokenizer_two_class: str = ""
    tokenizer_two_module: str = "transformers"
    tokenizer_two_subfolder: str = "tokenizer_2"


# ============================================================
# MODEL REGISTRY: Define all supported model families here
# ============================================================

MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "QwenImage": ModelSpec(
        vae_module="diffusers",
        vae_class="AutoencoderKLQwenImage",
        transformer_module="diffusers",
        transformer_class="QwenImageTransformer2DModel",
        text_encoder_module="transformers",
        text_encoder_class="Qwen2_5_VLForConditionalGeneration",
        tokenizer_module="transformers",
        tokenizer_class="Qwen2Tokenizer",
        pipeline_module="diffusers",
        pipeline_class="QwenImagePipeline",
        scheduler_module="diffusers",
        scheduler_class="FlowMatchEulerDiscreteScheduler",
        scheduler_kwargs={"shift": 3.0},
        vae_scale_factor_attr="temperal_downsample",
        latents_mean_attr="latents_mean",
        latents_std_attr="latents_std",
        z_dim_attr="z_dim",
        has_dual_text_encoder=False,
    ),
    
    "Flux2Klein": ModelSpec(
        vae_module="diffusers",
        vae_class="AutoencoderKLFlux2",
        transformer_module="diffusers",
        transformer_class="Flux2Transformer2DModel",
        text_encoder_module="transformers",
        text_encoder_class="Qwen3ForCausalLM",
        tokenizer_module="transformers",
        tokenizer_class="Qwen2TokenizerFast",
        pipeline_module="diffusers",
        pipeline_class="Flux2KleinPipeline",
        scheduler_module="diffusers",
        scheduler_class="FlowMatchEulerDiscreteScheduler",
        scheduler_kwargs={},
        vae_scale_factor_attr="block_out_channels",  # Different for Flux
        latents_mean_attr="latents_mean",
        latents_std_attr="latents_std",
        z_dim_attr="latent_channels",
        has_dual_text_encoder=False,
    ),
    
    # Add more model types here as needed:
    # "SD3": ModelSpec(...),
    # "SDXL": ModelSpec(...),
}


def get_class_from_module(module_path: str, class_name: str) -> Type:
    """Dynamically import a class from a module path."""
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Cannot import {class_name} from {module_path}: {e}")


def get_model_spec(model_type: str) -> ModelSpec:
    """Get the model specification for a given model type."""
    if model_type not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
    return MODEL_REGISTRY[model_type]


def register_model(name: str, spec: ModelSpec):
    """Register a new model type to the registry."""
    MODEL_REGISTRY[name] = spec
    logger.info(f"Registered new model type: {name}")


class ModelFactory:
    """
    Factory class for loading diffusion model components.
    
    Usage:
        factory = ModelFactory(config)
        vae, transformer, tokenizer, text_encoder, scheduler, pipeline, vae_scale_factor = factory.load_all()
    """
    
    def __init__(self, config):
        """
        Args:
            config: Configuration object with model_type and other settings.
                   Required attributes:
                   - model_type (str): e.g., 'QwenImage', 'Flux2Klein'
                   - pretrained_model_name_or_path (str)
                   - revision (str, optional)
                   - variant (str, optional)
                   - cache_dir (str, optional)
                   - huggingface_token (str, optional)
                   - weight_dtype (torch.dtype)
                   - bnb_quantization_config_path (str, optional)
        """
        self.config = config
        self.model_type = getattr(config, 'model_type', 'QwenImage')
        self.spec = get_model_spec(self.model_type)
        
        # Log model factory initialization details
        logger.info("="*60)
        logger.info("🏭 Model Factory Initialized")
        logger.info(f"   Model Type: {self.model_type}")
        logger.info(f"   Pretrained Path: {getattr(config, 'pretrained_model_name_or_path', 'NOT SET')}")
        logger.info(f"   Cache Dir: {getattr(config, 'cache_dir', 'None')}")
        logger.info(f"   VAE Class: {self.spec.vae_class}")
        logger.info(f"   Transformer Class: {self.spec.transformer_class}")
        logger.info(f"   Text Encoder Class: {self.spec.text_encoder_class}")
        logger.info(f"   Pipeline Class: {self.spec.pipeline_class}")
        logger.info("="*60)
        
        # Cache loaded classes
        self._class_cache: Dict[str, Type] = {}
        
    def _get_class(self, module_path: str, class_name: str) -> Type:
        """Get class with caching."""
        key = f"{module_path}.{class_name}"
        if key not in self._class_cache:
            self._class_cache[key] = get_class_from_module(module_path, class_name)
        return self._class_cache[key]
    
    @property
    def VAEClass(self) -> Type:
        return self._get_class(self.spec.vae_module, self.spec.vae_class)
    
    @property
    def TransformerClass(self) -> Type:
        return self._get_class(self.spec.transformer_module, self.spec.transformer_class)
    
    @property
    def TextEncoderClass(self) -> Type:
        return self._get_class(self.spec.text_encoder_module, self.spec.text_encoder_class)
    
    @property
    def TokenizerClass(self) -> Type:
        return self._get_class(self.spec.tokenizer_module, self.spec.tokenizer_class)
    
    @property
    def PipelineClass(self) -> Type:
        return self._get_class(self.spec.pipeline_module, self.spec.pipeline_class)
    
    @property
    def SchedulerClass(self) -> Type:
        return self._get_class(self.spec.scheduler_module, self.spec.scheduler_class)
    
    def load_tokenizer(self):
        """Load the tokenizer."""
        logger.info(f"[INFO] Loading tokenizer: {self.spec.tokenizer_class}")
        import os as _os
        base = self.config.pretrained_model_name_or_path
        subfolder = self.spec.tokenizer_subfolder
        # For local paths, join the subfolder directly because transformers>=4.55.0
        # incorrectly validates local abs paths as HF repo IDs when subfolder= is used.
        # For HuggingFace repo IDs, use subfolder= argument (joining would create an
        # invalid repo ID like 'namespace/repo/tokenizer').
        is_local = _os.path.isabs(base) or _os.path.isdir(base)
        if subfolder and is_local:
            tok_path = _os.path.join(base, subfolder)
            tok_subfolder = None
        else:
            tok_path = base
            tok_subfolder = subfolder if subfolder else None
        logger.info(f"[INFO] Tokenizer path: {tok_path}" + (f", subfolder: {tok_subfolder}" if tok_subfolder else ""))
        return self.TokenizerClass.from_pretrained(
            tok_path,
            subfolder=tok_subfolder,
            revision=getattr(self.config, 'revision', None),
            cache_dir=getattr(self.config, 'cache_dir', None),
            token=get_hf_token(self.config),
        )
    
    def load_text_encoder(self, skip_load=False):
        """Load the text encoder.
        
        Args:
            skip_load: If True, return a dummy/minimal text encoder.
                      Useful for parquet dataset where text embeddings are pre-computed.
        """
        logger.info(f"[INFO] Loading text encoder: {self.spec.text_encoder_class}")
        
        if skip_load:
            logger.info(f"[INFO] Skipping text encoder load (parquet dataset mode)")
            return None
        
        text_encoder = self.TextEncoderClass.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder=self.spec.text_encoder_subfolder,
            revision=getattr(self.config, 'revision', None),
            variant=getattr(self.config, 'variant', None),
            cache_dir=getattr(self.config, 'cache_dir', None),
            token=get_hf_token(self.config),
            torch_dtype=getattr(self.config, 'weight_dtype', torch.bfloat16),
        )
        text_encoder.requires_grad_(False)
        return text_encoder
    
    def load_vae(self):
        """Load the VAE."""
        logger.info(f"[INFO] Loading VAE: {self.spec.vae_class}")
        vae = self.VAEClass.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder=self.spec.vae_subfolder,
            revision=getattr(self.config, 'revision', None),
            variant=getattr(self.config, 'variant', None),
            cache_dir=getattr(self.config, 'cache_dir', None),
            token=get_hf_token(self.config),
        )
        vae.requires_grad_(False)
        return vae
    
    def load_transformer(self):
        """Load the transformer/DiT model."""
        logger.info(f"[INFO] Loading transformer: {self.spec.transformer_class}")
        
        # Handle quantization config
        quantization_config = None
        bnb_config_path = getattr(self.config, 'bnb_quantization_config_path', None)
        if bnb_config_path is not None:
            from diffusers import BitsAndBytesConfig
            with open(bnb_config_path, "r") as f:
                config_kwargs = json.load(f)
                if "load_in_4bit" in config_kwargs and config_kwargs["load_in_4bit"]:
                    config_kwargs["bnb_4bit_compute_dtype"] = self.config.weight_dtype
            quantization_config = BitsAndBytesConfig(**config_kwargs)
        
        transformer = self.TransformerClass.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder=self.spec.transformer_subfolder,
            revision=getattr(self.config, 'revision', None),
            variant=getattr(self.config, 'variant', None),
            cache_dir=getattr(self.config, 'cache_dir', None),
            token=get_hf_token(self.config),
            quantization_config=quantization_config,
            torch_dtype=getattr(self.config, 'weight_dtype', torch.bfloat16),
        )
        
        if bnb_config_path is not None:
            transformer = prepare_model_for_kbit_training(
                transformer, use_gradient_checkpointing=False
            )
        
        # Set requires_grad based on training mode
        if getattr(self.config, 'use_lora', True):
            logger.info(f"[INFO] Using LoRA fine-tuning ...")
            transformer.requires_grad_(False)
        else:
            logger.info(f"[INFO] Fine-tuning the full model ...")
            transformer.requires_grad_(True)
        
        # Enable gradient checkpointing if requested
        if getattr(self.config, 'gradient_checkpointing', False):
            logger.info(f"[INFO] Enabling gradient checkpointing for transformer")
            transformer.enable_gradient_checkpointing()
        
        return transformer
    
    def load_scheduler(self):
        """Load the noise scheduler."""
        logger.info(f"[INFO] Loading scheduler: {self.spec.scheduler_class}")
        scheduler_kwargs = copy.deepcopy(self.spec.scheduler_kwargs)
        return self.SchedulerClass.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder=self.spec.scheduler_subfolder,
            revision=getattr(self.config, 'revision', None),
            cache_dir=getattr(self.config, 'cache_dir', None),
            token=get_hf_token(self.config),
            **scheduler_kwargs,
        )
    
    def load_text_encoding_pipeline(self, tokenizer, text_encoder):
        """Load the pipeline for text encoding only."""
        logger.info(f"[INFO] Loading text encoding pipeline: {self.spec.pipeline_class}")
        return self.PipelineClass.from_pretrained(
            self.config.pretrained_model_name_or_path,
            vae=None,
            transformer=None,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=None,
            token=get_hf_token(self.config),
        )
    
    def get_vae_scale_factor(self, vae) -> int:
        """Calculate VAE scale factor based on model type."""
        attr_name = self.spec.vae_scale_factor_attr
        if hasattr(vae, attr_name):
            attr_value = getattr(vae, attr_name)
            if isinstance(attr_value, (list, tuple)):
                # e.g., temperal_downsample for QwenImage
                return 2 ** len(attr_value)
            elif isinstance(attr_value, int):
                return attr_value
        # Default fallback
        return 8
    
    def get_latents_stats(self, vae, device):
        """Get latents mean and std tensors."""
        mean_attr = self.spec.latents_mean_attr
        std_attr = self.spec.latents_std_attr
        z_dim_attr = self.spec.z_dim_attr
        
        z_dim = getattr(vae.config, z_dim_attr, 16)
        
        if hasattr(vae.config, mean_attr):
            latents_mean = torch.tensor(getattr(vae.config, mean_attr)).view(1, z_dim, 1, 1, 1).to(device)
        else:
            latents_mean = torch.zeros(1, z_dim, 1, 1, 1).to(device)
        
        if hasattr(vae.config, std_attr):
            latents_std = 1.0 / torch.tensor(getattr(vae.config, std_attr)).view(1, z_dim, 1, 1, 1).to(device)
        else:
            latents_std = torch.ones(1, z_dim, 1, 1, 1).to(device)
        
        return latents_mean, latents_std
    
    def load_all(self, device=None, skip_text_encoder=False, skip_vae=False) -> Tuple:
        """
        Load all model components.
        
        Args:
            device: Device to load models on (optional)
            skip_text_encoder: If True, skip loading text encoder (for parquet dataset)
            skip_vae: If True, skip loading VAE (for parquet dataset, though VAE is needed for BN stats)
        
        Returns:
            Tuple of (vae, transformer, tokenizer, text_encoder, scheduler, 
                     text_encoding_pipeline, vae_scale_factor)
        """
        tokenizer = self.load_tokenizer()
        text_encoder = self.load_text_encoder(skip_load=skip_text_encoder)
        vae = self.load_vae() if not skip_vae else None
        transformer = self.load_transformer()
        scheduler = self.load_scheduler()
        
        text_encoding_pipeline = self.load_text_encoding_pipeline(tokenizer, text_encoder) if text_encoder else None
        vae_scale_factor = self.get_vae_scale_factor(vae) if vae else 8  # Default to 8 if no VAE
        
        logger.info(f"[INFO] VAE scale factor: {vae_scale_factor}")
        
        return (
            vae,
            transformer,
            tokenizer,
            text_encoder,
            scheduler,
            text_encoding_pipeline,
            vae_scale_factor,
        )


def initialize_models(config):
    """
    Unified model initialization function.
    
    This is the main entry point for loading models in training scripts.
    It automatically detects the model type from config and loads all components.
    
    Args:
        config: Configuration object with model_type and other settings.
        
    Returns:
        Tuple of (vae, transformer, tokenizer, text_encoder, scheduler,
                 text_encoding_pipeline, vae_scale_factor)
    """
    factory = ModelFactory(config)
    return factory.load_all()


def get_pipeline_class(model_type: str) -> Type:
    """Get the pipeline class for a model type (useful for save/load hooks)."""
    spec = get_model_spec(model_type)
    return get_class_from_module(spec.pipeline_module, spec.pipeline_class)


def get_transformer_class(model_type: str) -> Type:
    """Get the transformer class for a model type."""
    spec = get_model_spec(model_type)
    return get_class_from_module(spec.transformer_module, spec.transformer_class)


def get_text_encoder_class(model_type: str) -> Type:
    """Get the text encoder class for a model type."""
    spec = get_model_spec(model_type)
    if spec.text_encoder_class is None:
        raise ValueError(f"Model type {model_type} does not have a text encoder")
    return get_class_from_module(spec.text_encoder_module, spec.text_encoder_class)


def get_tokenizer_class(model_type: str) -> Type:
    """Get the tokenizer class for a model type."""
    spec = get_model_spec(model_type)
    if spec.tokenizer_class is None:
        raise ValueError(f"Model type {model_type} does not have a tokenizer")
    return get_class_from_module(spec.tokenizer_module, spec.tokenizer_class)
