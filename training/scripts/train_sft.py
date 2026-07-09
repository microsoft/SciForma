"""
SciForma SFT Training Script (Stage 1 & Stage 2)

This script performs full parameter fine-tuning of diffusion models
without LoRA adapters. VAE and text encoder are offloaded and frozen.

Supported Models:
- Flux2Klein (9B params): Use with DeepSpeed ZeRO-2

Key features:
1. No LoRA configuration - direct transformer parameter training
2. Uses parquet dataset with pre-computed latents/embeddings
3. VAE and text encoder remain offloaded
4. Optimized for memory efficiency with gradient checkpointing
5. Supports BF16/FP16 mixed precision training

Usage:
    # Stage 1:
    accelerate launch --config_file training/accelerate_cfg/b200_zero2_bf16_8gpu.yaml \
        training/scripts/train_sft.py training/configs/stage1_sft.py

    # Stage 2:
    accelerate launch --config_file training/accelerate_cfg/b200_zero2_bf16_8gpu.yaml \
        training/scripts/train_sft.py training/configs/stage2_sft.py
"""

# =========================================================
# Path & Environment
# =========================================================
import os
import sys
import os.path as osp
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.insert(
    0,
    osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), "..", "..")),
)

# =========================================================
# Load secrets from _local_secrets.py (for HF_TOKEN and WANDB)
# =========================================================
try:
    from _local_secrets import (
        HF_TOKEN,
        WANDB_API_KEY,
        WANDB_ENTITY,
        WANDB_PROJECT,
        WANDB_BASE_URL,
    )
    # Set as environment variables (if not already set)
    if not os.environ.get('HF_TOKEN'):
        os.environ['HF_TOKEN'] = HF_TOKEN
    if not os.environ.get('WANDB_API_KEY'):
        os.environ['WANDB_API_KEY'] = WANDB_API_KEY
    if not os.environ.get('WANDB_ENTITY'):
        os.environ['WANDB_ENTITY'] = WANDB_ENTITY
    if not os.environ.get('WANDB_PROJECT'):
        os.environ['WANDB_PROJECT'] = WANDB_PROJECT
    if not os.environ.get('WANDB_BASE_URL'):
        os.environ['WANDB_BASE_URL'] = WANDB_BASE_URL
except ImportError:
    pass  # _local_secrets.py not available, use env vars directly

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# =========================================================
# Standard Library
# =========================================================
import copy
import json
import math
import shutil
import logging
import warnings
import itertools
import tempfile
import time
import gc
from datetime import timedelta
from functools import partial
from contextlib import nullcontext

# =========================================================
# Third-party Libraries
# =========================================================
from tqdm.auto import tqdm

import torch
import torch.utils.checkpoint
from torch.utils.data import RandomSampler
from torch.utils.data import DistributedSampler

# Accelerate
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from accelerate import Accelerator, DataLoaderConfiguration

# Transformers / Diffusers
import transformers
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
    EMAModel,
)
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.torch_utils import is_compiled_module

# =========================================================
# Project: SciForma
# =========================================================
from sciforma.registry import (
    DATASETS,
    TRAIN_ITERATION_FUNCS,
    VALIDATION_FUNCS,
)
from sciforma.utils import (
    parse_config,
    unwrap_model,
    # Model Factory: Dynamic model loading
    ModelFactory,
)

# Require a minimum version of diffusers.
check_min_version("0.32.0")

logger = get_logger(__name__)


class LossTracker:
    """
    Track loss statistics for monitoring training progress.
    
    Expected behavior for Flux2Klein flow matching:
    - Initial loss: ~0.5-1.5 (depending on data)
    - After warmup: should decrease to ~0.3-0.8
    - Stable training: fluctuates between 0.2-0.6
    """
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.losses = []
        self.min_loss = float('inf')
        self.max_loss = float('-inf')
        self.initial_loss = None
        
    def update(self, loss):
        if self.initial_loss is None:
            self.initial_loss = loss
        
        self.losses.append(loss)
        if len(self.losses) > self.window_size:
            self.losses.pop(0)
        
        self.min_loss = min(self.min_loss, loss)
        self.max_loss = max(self.max_loss, loss)
    
    def get_stats(self):
        if not self.losses:
            return {}
        
        avg = sum(self.losses) / len(self.losses)
        recent_min = min(self.losses)
        recent_max = max(self.losses)
        
        return {
            "loss_avg": avg,
            "loss_recent_min": recent_min,
            "loss_recent_max": recent_max,
            "loss_all_time_min": self.min_loss,
            "loss_initial": self.initial_loss,
        }
    
    def check_health(self, current_loss, step):
        """
        Check if loss is in expected range.
        
        Returns:
            tuple: (is_healthy, message)
        """
        # Loss sanity checks
        if current_loss > 10.0:
            return False, f"❌ Loss too high ({current_loss:.4f})! Check data/model."
        
        if current_loss < 0:
            return False, f"❌ Negative loss ({current_loss:.4f})! Something is wrong."
        
        if step > 100 and current_loss > 5.0:
            return False, f"⚠️ Loss not decreasing after 100 steps ({current_loss:.4f})"
        
        # Normal range for flow matching
        if 0.1 <= current_loss <= 2.0:
            return True, f"✅ Loss in normal range ({current_loss:.4f})"
        
        return True, f"Loss: {current_loss:.4f}"


def save_model_checkpoint(
    transformer,
    accelerator,
    config,
    global_step,
    logger,
    is_final=False,
    ema_transformer=None,
):
    """
    Save full model checkpoint (no LoRA, direct transformer weights).
    
    For FSDP: Uses accelerator.save_state() which handles sharded states.
    For DDP: Saves unwrapped model directly.
    
    Args:
        transformer: The transformer model to save
        accelerator: Accelerator instance
        config: Training configuration
        global_step: Current training step
        logger: Logger instance
        is_final: Whether this is the final checkpoint
        ema_transformer: Optional EMA model to save
    """
    if is_final:
        save_path = os.path.join(config.model_output_dir, "final_model")
    else:
        save_path = os.path.join(config.model_output_dir, f"checkpoint-{global_step}")
    
    # Detect DeepSpeed ZeRO-3 (params are partitioned, need gathering for save)
    is_deepspeed_zero3 = (
        accelerator.distributed_type == DistributedType.DEEPSPEED and
        hasattr(accelerator.state, 'deepspeed_plugin') and
        getattr(accelerator.state.deepspeed_plugin, 'zero_stage', 0) == 3
    )
    
    # For FSDP, use accelerator.save_state
    if accelerator.distributed_type == DistributedType.FSDP:
        logger.info(f"Saving FSDP checkpoint to {save_path}")
        accelerator.save_state(save_path)
    elif is_deepspeed_zero3:
        # ZeRO-3: params are partitioned across GPUs.
        # Use GatheredParameters to gather full params one-by-one, then save.
        import deepspeed
        os.makedirs(save_path, exist_ok=True)
        
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        unwrapped_transformer = unwrapped_transformer._orig_mod if is_compiled_module(unwrapped_transformer) else unwrapped_transformer
        
        # Gather full state dict param by param to avoid OOM
        full_state_dict = {}
        for name, param in unwrapped_transformer.named_parameters():
            with deepspeed.zero.GatheredParameters(param):
                if accelerator.is_main_process:
                    full_state_dict[name] = param.data.detach().cpu().clone()
        
        if accelerator.is_main_process:
            # Save transformer in safetensors format (compatible with our loading code)
            # Cannot use save_pretrained() since it calls model.state_dict() internally
            # which returns partitioned params under ZeRO-3
            from safetensors.torch import save_file as safetensors_save_file
            transformer_save_path = os.path.join(save_path, "transformer")
            os.makedirs(transformer_save_path, exist_ok=True)
            safetensors_save_file(full_state_dict, os.path.join(transformer_save_path, "diffusion_pytorch_model.safetensors"))
            # Save config.json for from_pretrained compatibility (optional)
            if hasattr(unwrapped_transformer, 'config'):
                import json
                config_dict = dict(unwrapped_transformer.config)
                with open(os.path.join(transformer_save_path, "config.json"), "w") as f:
                    json.dump(config_dict, f, indent=2, default=str)
            del full_state_dict
            logger.info(f"Saved ZeRO-3 transformer to {transformer_save_path} (safetensors)")
            
            # Save EMA weights if available
            # Cast shadow_params to bf16 for save (compatible with inference code)
            # but keep internal float32 for ongoing EMA updates
            if ema_transformer is not None:
                ema_save_path = os.path.join(save_path, "ema_weights.pt")
                ema_state = ema_transformer.state_dict()
                ema_state['shadow_params'] = [p.bfloat16() for p in ema_state['shadow_params']]
                torch.save(ema_state, ema_save_path)
                logger.info(f"Saved EMA weights to {ema_save_path} (cast to bf16)")
            
            # Save training state
            torch.save({"global_step": global_step}, os.path.join(save_path, "training_state.pt"))
            logger.info(f"Saved ZeRO-3 checkpoint to {save_path}")
        
        # Synchronize all ranks after save
        accelerator.wait_for_everyone()
    else:
        # For DDP, single GPU, or DeepSpeed ZeRO-2 (params are full on each GPU)
        if accelerator.is_main_process:
            os.makedirs(save_path, exist_ok=True)
            
            # Unwrap model
            unwrapped_transformer = accelerator.unwrap_model(transformer)
            unwrapped_transformer = unwrapped_transformer._orig_mod if is_compiled_module(unwrapped_transformer) else unwrapped_transformer
            
            # Save transformer weights
            transformer_save_path = os.path.join(save_path, "transformer")
            unwrapped_transformer.save_pretrained(transformer_save_path)
            
            # Save EMA weights if available
            # Cast shadow_params to bf16 for save (compatible with inference code)
            if ema_transformer is not None:
                ema_save_path = os.path.join(save_path, "ema_weights.pt")
                ema_state = ema_transformer.state_dict()
                ema_state['shadow_params'] = [p.bfloat16() for p in ema_state['shadow_params']]
                torch.save(ema_state, ema_save_path)
                logger.info(f"Saved EMA weights to {ema_save_path} (cast to bf16)")
            
            # Save training state
            state_dict = {
                "global_step": global_step,
            }
            torch.save(state_dict, os.path.join(save_path, "training_state.pt"))
            
            logger.info(f"Saved checkpoint to {save_path}")
    
    # Save accelerator state (optimizer, scheduler, etc.)
    accelerator.save_state(os.path.join(save_path, "accelerator"))
    

def main():
    config = parse_config(train=True)

    # =========================================================
    # Logging & Accelerator Setup
    # =========================================================
    hub_token = config.get('hub_token', None)
    if config.report_to == "wandb" and hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and config.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(config.model_output_dir, config.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=config.model_output_dir,
        logging_dir=logging_dir
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)  # No unused params in full fine-tuning
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=180))  # 3 hours timeout for saving large checkpoints
    
    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs, init_kwargs],
        dataloader_config=dataloader_config,
    )
    
    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if config.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if config.seed is not None:
        set_seed(config.seed)
    
    # Optional: Enable deterministic mode for full reproducibility (slower)
    if config.get('deterministic', False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Note: This may significantly slow down training
        logger.info("[INFO] Deterministic mode enabled - training may be slower")

    # Handle the repository creation
    if accelerator.is_main_process:
        if config.model_output_dir is not None:
            os.makedirs(config.model_output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    config.weight_dtype = weight_dtype

    # Enable TF32 for faster training on Ampere GPUs
    if config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.scale_lr:
        config.learning_rate = (
            config.learning_rate * config.gradient_accumulation_steps * config.train_batch_size * accelerator.num_processes
        )

    # =========================================================
    # Model Initialization via Model Factory
    # =========================================================
    model_type = getattr(config, 'model_type', 'Flux2Klein')
    logger.info(f"[INFO] Using model type: {model_type}")
    
    # Check distributed type for proper dtype handling
    # Need to check this BEFORE model loading for DeepSpeed ZeRO-3 compatibility
    distributed_type_str = str(getattr(accelerator.state, 'distributed_type', 'NO'))
    is_fsdp = 'FSDP' in distributed_type_str
    is_deepspeed = 'DEEPSPEED' in distributed_type_str
    
    # For multi-GPU: Only main process downloads models first, others wait
    # This prevents race conditions when downloading from HuggingFace
    # Skip text encoder in parquet mode — embeddings are pre-computed.
    # Text encoder will be lazy-loaded only if validation needs it.
    skip_text_encoder = config.use_parquet_dataset
    with accelerator.main_process_first():
        model_factory = ModelFactory(config)
        (
            vae,
            transformer,
            tokenizer,
            text_encoder,
            noise_scheduler,
            text_encoding_pipeline,
            vae_scale_factor,
        ) = model_factory.load_all(skip_text_encoder=skip_text_encoder)
    
    # For distributed training with bf16, keep model in bf16
    # This avoids gradient dtype mismatch issues
    if is_fsdp:
        logger.info(f"[INFO] FSDP detected - keeping transformer in bf16 for pure bf16 training")
        # With mixed_precision='no' in accelerate config, we use pure bf16
        # Model stays in bf16, gradients are bf16, no dtype conversion needed
        transformer = transformer.to(dtype=torch.bfloat16)
    elif is_deepspeed:
        logger.info(f"[INFO] DeepSpeed detected - keeping transformer in bf16 for ZeRO-3")
        # DeepSpeed ZeRO-3 with bf16 works with bf16 parameters
        transformer = transformer.to(dtype=torch.bfloat16)
    
    # Get latents mean/std (model-specific)
    latents_mean, latents_std = model_factory.get_latents_stats(vae, accelerator.device)

    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # Link stats to config for train_iteration_func
    config.vae_scale_factor = vae_scale_factor
    config.latents_mean = latents_mean
    config.latents_std = latents_std
    
    # =========================================================
    # Model Device Assignment & Offloading
    # =========================================================
    logger.info(f"[INFO] Configuring model devices and offloading")
    
    # is_fsdp and is_deepspeed already set above
    
    # For parquet dataset: VAE and text encoder stay on CPU (offloaded)
    # They are not used during training (only for validation if needed)
    if config.use_parquet_dataset:
        logger.info(f"[INFO] Using parquet dataset - VAE remains on CPU, text encoder skipped")
        # Keep VAE on CPU with weight_dtype for BatchNorm stats access
        vae.to(dtype=weight_dtype, device="cpu")
        if text_encoder is not None:
            text_encoder.to(dtype=weight_dtype, device="cpu")
    else:
        raise ValueError("Full fine-tuning script currently only supports parquet dataset")
    
    # Handle device placement based on distributed strategy
    if is_fsdp:
        # FSDP: Keep model in float32 on CPU, FSDP will handle sharding
        logger.info("[INFO] FSDP mode: transformer stays on CPU, accelerator.prepare will handle placement")
    elif is_deepspeed:
        # DeepSpeed: Keep model on CPU, ZeRO-3 init will handle placement
        logger.info("[INFO] DeepSpeed mode: transformer stays on CPU, ZeRO-3 will handle placement")
    else:
        # Non-distributed or DDP: Move transformer to GPU directly
        transformer.to(device=accelerator.device, dtype=weight_dtype)
        logger.info(f"[INFO] Moving transformer to {accelerator.device}")
    
    # =========================================================
    # Enable Gradient Checkpointing for Memory Efficiency
    # =========================================================
    if config.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        logger.info("[INFO] Gradient checkpointing enabled")
    
    # =========================================================
    # Create EMA Model (if enabled)
    # ema_on_gpu=True  → keep shadow_params on GPU (B200 with 192GB VRAM)
    # ema_on_gpu=False → keep shadow_params on CPU (A100 with 80GB VRAM)
    # =========================================================
    ema_transformer = None
    ema_on_gpu = config.get('ema_on_gpu', False)
    if config.get('use_ema', False):
        ema_decay = config.get('ema_decay', 0.9999)
        ema_update_after_step = config.get('ema_update_after_step', 0)
        
        ema_transformer = EMAModel(
            transformer.parameters(),
            decay=ema_decay,
            update_after_step=ema_update_after_step,
        )
        # IMPORTANT: Always use float32 to avoid bf16 precision loss during EMA updates
        # (update magnitude ~1e-4 * delta is below bf16 ULP, causing updates to be rounded to zero)
        if ema_on_gpu:
            ema_device = accelerator.device
            ema_transformer.shadow_params = [p.to(ema_device).float() for p in ema_transformer.shadow_params]
            logger.info(f"[INFO] EMA model created on GPU ({ema_device}, float32) with decay={ema_decay}, update_after_step={ema_update_after_step}")
        else:
            ema_transformer.shadow_params = [p.cpu().float() for p in ema_transformer.shadow_params]
            logger.info(f"[INFO] EMA model created on CPU (float32) with decay={ema_decay}, update_after_step={ema_update_after_step}")
    
    # =========================================================
    # Prepare Transformer for Full Fine-tuning
    # =========================================================
    # Unfreeze all transformer parameters
    for param in transformer.parameters():
        param.requires_grad = True
    
    transformer_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    
    num_trainable_params = sum(p.numel() for p in transformer_parameters) / 1e6
    logger.info(f"[INFO] Number of trainable parameters: {num_trainable_params:.2f}M")
    
    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_parameters, "lr": config.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]
    
    # =========================================================
    # Optimizer Creation
    # =========================================================
    # Note: 8bit Adam is not compatible with FSDP or DeepSpeed, disable it for those modes
    use_8bit = config.use_8bit_adam and not is_fsdp and not is_deepspeed
    if (is_fsdp or is_deepspeed) and config.use_8bit_adam:
        logger.warning(f"[WARNING] 8-bit Adam is not compatible with {'FSDP' if is_fsdp else 'DeepSpeed'}, using standard AdamW")
    
    if config.optimizer.lower() == "adamw":
        if use_8bit:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install bitsandbytes: `pip install bitsandbytes`"
                )
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.adam_weight_decay,
            eps=config.adam_epsilon,
        )
    elif config.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install prodigyopt: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        optimizer = optimizer_class(
            params_to_optimize,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            beta3=config.prodigy_beta3,
            weight_decay=config.adam_weight_decay,
            eps=config.adam_epsilon,
            decouple=config.prodigy_decouple,
            use_bias_correction=config.prodigy_use_bias_correction,
            safeguard_warmup=config.prodigy_safeguard_warmup,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    # =========================================================
    # Dataset & DataLoader
    # =========================================================
    logger.info("[INFO] Loading dataset")
    
    dataset_cfg = config.dataset_cfg
    dataset = DATASETS.build(dataset_cfg)
    
    # Distributed sampler for parquet dataset
    if hasattr(dataset, 'collate_fn'):
        collate_fn = dataset.collate_fn
    else:
        collate_fn = None
    
    # Use distributed bucket sampler if available
    sampler_cfg = config.get('sampler_cfg', None)
    if sampler_cfg is not None:
        # Fill in the dataset reference
        sampler_cfg['dataset'] = dataset
        # IMPORTANT: Force num_replicas=1, rank=0 to prevent double sharding!
        # The sampler should NOT do distributed sharding because accelerator.prepare()
        # will handle the distribution across GPUs. If we set num_replicas=num_gpus here,
        # each GPU would only see 1/4 of 1/4 = 1/16 of the data!
        sampler_cfg['num_replicas'] = 1  # Force no sharding in sampler
        sampler_cfg['rank'] = 0  # Force rank 0
        # Override batch_size from config
        sampler_cfg['batch_size'] = config.train_batch_size
        # Set seed for reproducibility
        if sampler_cfg.get('seed') is None:
            sampler_cfg['seed'] = config.seed if config.seed is not None else 42
        batch_sampler = DATASETS.build(sampler_cfg)
        # DistributedBucketSamplerV2 is a batch_sampler, not a sampler
        # It returns batches of indices, so we use batch_sampler parameter
        logger.info(f"[INFO] Creating DataLoader with num_workers={config.dataloader_num_workers}, prefetch_factor=4, persistent_workers=True")
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,  # Use batch_sampler, not sampler
            collate_fn=collate_fn,
            num_workers=config.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=True if config.dataloader_num_workers > 0 else False,
            prefetch_factor=4 if config.dataloader_num_workers > 0 else None,
        )
        # For DeepSpeed: set train_micro_batch_size_per_gpu since batch_sampler makes batch_size None
        if is_deepspeed and hasattr(accelerator.state, 'deepspeed_plugin'):
            accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = config.train_batch_size
            logger.info(f"[INFO] Set DeepSpeed train_micro_batch_size_per_gpu to {config.train_batch_size}")
    else:
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=True if config.dataloader_num_workers > 0 else False,
            prefetch_factor=4 if config.dataloader_num_workers > 0 else None,
        )

    # =========================================================
    # Learning Rate Scheduler
    # =========================================================
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
        num_cycles=config.lr_num_cycles,
        power=config.lr_power,
    )

    # =========================================================
    # Load Transformer Weights from Checkpoint (if resuming)
    # This must happen BEFORE accelerator.prepare() to ensure
    # the correct weights are wrapped/sharded by DeepSpeed/FSDP
    # =========================================================
    resume_checkpoint_path = None
    
    # AUTO-DETECT: Always check for existing checkpoints even if config.resume_from_checkpoint is None
    # This ensures AMLT retry jobs automatically resume from the latest checkpoint
    auto_resume = False
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint != "latest":
            # Check if it's a full path or just checkpoint name
            if os.path.isabs(config.resume_from_checkpoint):
                resume_checkpoint_path = config.resume_from_checkpoint
            else:
                resume_checkpoint_path = os.path.join(config.model_output_dir, config.resume_from_checkpoint)
        else:
            auto_resume = True
    else:
        # Even if resume_from_checkpoint is None, auto-detect existing checkpoints
        # This is critical for AMLT retry scenarios
        auto_resume = True
        logger.info(f"[INFO] resume_from_checkpoint is None, but checking for existing checkpoints in {config.model_output_dir}...")
    
    if auto_resume:
        # Get the most recent checkpoint
        if os.path.exists(config.model_output_dir):
            dirs = os.listdir(config.model_output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            if dirs:
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                resume_checkpoint_path = os.path.join(config.model_output_dir, dirs[-1])
                logger.info(f"[INFO] Auto-detected checkpoint: {resume_checkpoint_path}")
            else:
                logger.info(f"[INFO] No checkpoints found in {config.model_output_dir}, starting fresh")
    
    # Load transformer weights from checkpoint (works for both explicit path and auto-detected)
    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        transformer_ckpt_path = os.path.join(resume_checkpoint_path, "transformer")
        if os.path.exists(transformer_ckpt_path):
            logger.info(f"[INFO] Loading transformer weights from checkpoint: {transformer_ckpt_path}")
            try:
                # Try loading with safetensors first (faster)
                safetensors_file = os.path.join(transformer_ckpt_path, "diffusion_pytorch_model.safetensors")
                if os.path.exists(safetensors_file):
                    from safetensors.torch import load_file
                    state_dict = load_file(safetensors_file)
                    # Load state dict into transformer
                    missing_keys, unexpected_keys = transformer.load_state_dict(state_dict, strict=False)
                    if missing_keys:
                        logger.warning(f"[WARNING] Missing keys when loading checkpoint: {missing_keys[:5]}...")
                    if unexpected_keys:
                        logger.warning(f"[WARNING] Unexpected keys when loading checkpoint: {unexpected_keys[:5]}...")
                    logger.info(f"[INFO] Successfully loaded transformer weights from checkpoint (safetensors)")
                else:
                    # Fallback to PyTorch format
                    pytorch_file = os.path.join(transformer_ckpt_path, "diffusion_pytorch_model.bin")
                    if os.path.exists(pytorch_file):
                        state_dict = torch.load(pytorch_file, map_location="cpu")
                        transformer.load_state_dict(state_dict, strict=False)
                        logger.info(f"[INFO] Successfully loaded transformer weights from checkpoint (pytorch)")
                    else:
                        logger.warning(f"[WARNING] No model weights found in checkpoint: {transformer_ckpt_path}")
                        resume_checkpoint_path = None
            except Exception as e:
                logger.error(f"[ERROR] Failed to load transformer weights from checkpoint: {e}")
                logger.error(f"[ERROR] Starting from pretrained weights instead")
                resume_checkpoint_path = None
        else:
            logger.warning(f"[WARNING] Transformer checkpoint not found at {transformer_ckpt_path}")
            resume_checkpoint_path = None
    elif resume_checkpoint_path:
        logger.warning(f"[WARNING] Checkpoint path does not exist: {resume_checkpoint_path}")
        resume_checkpoint_path = None

    # =========================================================
    # Init from external weights (e.g., Stage 1 EMA) - fallback when no checkpoint
    # =========================================================
    if resume_checkpoint_path is None:
        init_weights_path = config.get('init_weights_path', None)
        if init_weights_path and os.path.exists(init_weights_path):
            logger.info(f"[INFO] No checkpoint found. Initializing transformer from: {init_weights_path}")
            try:
                init_state = torch.load(init_weights_path, map_location="cpu", weights_only=False)
                if 'shadow_params' in init_state:
                    # EMA state dict format — copy shadow_params to transformer
                    shadow_params = init_state['shadow_params']
                    model_params = list(transformer.parameters())
                    if len(shadow_params) != len(model_params):
                        logger.error(f"[ERROR] Param count mismatch: shadow={len(shadow_params)}, model={len(model_params)}")
                    else:
                        for p, s in zip(model_params, shadow_params):
                            p.data.copy_(s.data)
                        logger.info(f"[INFO] Loaded {len(shadow_params)} params from EMA shadow_params")
                else:
                    # Raw state_dict format
                    missing, unexpected = transformer.load_state_dict(init_state, strict=False)
                    if missing:
                        logger.warning(f"[WARNING] Missing keys: {missing[:5]}...")
                    if unexpected:
                        logger.warning(f"[WARNING] Unexpected keys: {unexpected[:5]}...")
                    logger.info(f"[INFO] Loaded transformer state_dict from init_weights_path")
            except Exception as e:
                logger.error(f"[ERROR] Failed to load init_weights_path: {e}")
                logger.error(f"[ERROR] Continuing with base pretrained weights")
        elif init_weights_path:
            logger.warning(f"[WARNING] init_weights_path does not exist: {init_weights_path}")

    # =========================================================
    # Prepare with Accelerator
    # =========================================================
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # =========================================================
    # EMA device placement:
    #   ema_on_gpu=True  → shadow_params on GPU (fast update, needs ~18GB extra VRAM)
    #   ema_on_gpu=False → shadow_params on CPU (saves GPU memory)
    # =========================================================
    ema_target_device = accelerator.device if ema_on_gpu else "cpu"
    if ema_transformer is not None:
        logger.info(f"[INFO] EMA model on {'GPU (' + str(accelerator.device) + ')' if ema_on_gpu else 'CPU'} to {'maximize speed' if ema_on_gpu else 'save GPU memory'}")
        
        # Load EMA weights from checkpoint if available
        if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
            ema_ckpt_path = os.path.join(resume_checkpoint_path, "ema_weights.pt")
            if os.path.exists(ema_ckpt_path):
                try:
                    ema_state_dict = torch.load(ema_ckpt_path, map_location="cpu")
                    ema_transformer.load_state_dict(ema_state_dict)
                    # Move to target device in float32 (loaded .pt may be bf16)
                    ema_transformer.shadow_params = [p.to(ema_target_device).float() for p in ema_transformer.shadow_params]
                    logger.info(f"[INFO] Loaded EMA weights from checkpoint: {ema_ckpt_path} (promoted to float32 on {ema_target_device})")
                except Exception as e:
                    logger.warning(f"[WARNING] Failed to load EMA weights: {e}")
                    logger.warning(f"[WARNING] EMA will start from current transformer weights")
            else:
                logger.info(f"[INFO] No EMA checkpoint found, using fresh EMA from transformer weights")
        else:
            # No checkpoint — try init_weights_path for EMA too
            init_weights_path = config.get('init_weights_path', None)
            if init_weights_path and os.path.exists(init_weights_path):
                try:
                    ema_init_state = torch.load(init_weights_path, map_location="cpu", weights_only=False)
                    if 'shadow_params' in ema_init_state:
                        ema_transformer.load_state_dict(ema_init_state)
                        ema_transformer.shadow_params = [p.to(ema_target_device).float() for p in ema_transformer.shadow_params]
                        logger.info(f"[INFO] Initialized EMA from init_weights_path: {init_weights_path} (promoted to float32 on {ema_target_device})")
                    else:
                        logger.info(f"[INFO] init_weights_path is not EMA format, EMA starts from transformer weights")
                except Exception as e:
                    logger.warning(f"[WARNING] Failed to load EMA from init_weights_path: {e}")
            else:
                logger.info(f"[INFO] No init_weights_path, EMA starts from base transformer weights")

    # =========================================================
    # Training Loop Setup
    # =========================================================
    # Compute total training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if config.max_train_steps is None or config.max_train_steps <= 0:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    
    config.num_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)
    
    total_batch_size = config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")

    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    # Resume accelerator state (optimizer, scheduler) from checkpoint
    # Note: Transformer weights were already loaded BEFORE accelerator.prepare()
    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        accelerator_state_path = os.path.join(resume_checkpoint_path, "accelerator")
        if os.path.exists(accelerator_state_path):
            logger.info(f"[INFO] Loading accelerator state from: {accelerator_state_path}")
            try:
                # PyTorch 2.6+ compatibility: DeepSpeed checkpoints contain custom classes
                # that are not in the default safe globals list. We have two options:
                # 1. Add all DeepSpeed classes to safe globals (may miss some)
                # 2. Temporarily set weights_only=False globally (more reliable)
                # Using option 2 for reliability with DeepSpeed ZeRO checkpoints
                # Note: torch is already imported at module level, don't import again here
                original_load = torch.load
                def patched_load(*args, **kwargs):
                    # Force weights_only=False for DeepSpeed compatibility
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                torch.load = patched_load
                logger.info("[INFO] Patched torch.load to use weights_only=False for DeepSpeed compatibility")
                
                try:
                    accelerator.load_state(accelerator_state_path)
                    # Extract step number from checkpoint path
                    # Note: rstrip('/') handles paths with trailing slash (e.g., from tab completion)
                    checkpoint_name = os.path.basename(resume_checkpoint_path.rstrip('/'))
                    if checkpoint_name.startswith("checkpoint-"):
                        global_step = int(checkpoint_name.split("-")[1])
                        initial_global_step = global_step
                        first_epoch = global_step // num_update_steps_per_epoch
                        logger.info(f"[INFO] Resuming from step {global_step}, epoch {first_epoch}")
                    else:
                        logger.warning(f"[WARNING] Could not parse step from checkpoint name: {checkpoint_name}")
                finally:
                    # Restore original torch.load
                    torch.load = original_load
                    
            except Exception as e:
                logger.error(f"[ERROR] Failed to load accelerator state: {e}")
                logger.error(f"[ERROR] Starting optimizer/scheduler from scratch")
        else:
            logger.warning(f"[WARNING] Accelerator state not found at {accelerator_state_path}")
            logger.warning(f"[WARNING] Transformer weights loaded, but optimizer/scheduler will start fresh")

    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # =========================================================
    # Get Training Iteration Function
    # =========================================================
    train_iteration_func_name = config.get('train_iteration_func', 'Flux2Klein_fulltune_train_iteration')
    train_iteration_func = TRAIN_ITERATION_FUNCS.get(train_iteration_func_name)
    
    if train_iteration_func is None:
        raise ValueError(f"Training iteration function '{train_iteration_func_name}' not found in registry")
    
    logger.info(f"[INFO] Using training iteration function: {train_iteration_func_name}")
    
    # =========================================================
    # Get Validation Function (if configured)
    # =========================================================
    # Auto-select validation function based on model type if not explicitly set
    if 'validation_func' in config:
        validation_func_name = config['validation_func']
    else:
        # Choose default validation function based on model type
        if se:
            validation_func_name = 'Flux2Klein_fulltune_validation_func_parquet'
    
    validation_func = VALIDATION_FUNCS.get(validation_func_name) if validation_func_name else None
    validation_steps = config.get('validation_steps', 500)
    validation_prompts = config.get('validation_prompts', None)
    
    if validation_func is not None:
        logger.info(f"[INFO] Using validation function: {validation_func_name}")
        logger.info(f"[INFO] Validation every {validation_steps} steps")
    else:
        logger.info(f"[INFO] No validation function configured")

    # =========================================================
    # Initialize Loss Tracker
    # =========================================================
    loss_tracker = LossTracker(window_size=100)

    # =========================================================
    # Initialize WandB Tracking
    # =========================================================
    if accelerator.is_main_process and config.report_to == "wandb":
        import wandb
        
        # Use tracker_run_name from config, fallback to run_name, then default
        run_name = config.get('tracker_run_name', None) or config.get('run_name', f"fulltune_{model_type}")
        
        accelerator.init_trackers(
            project_name=config.get('wandb_project', 'SciForma-SFT'),
            config=vars(config),
            init_kwargs={"wandb": {"name": run_name, "resume": "allow" if resume_checkpoint_path else None}}
        )

    # =========================================================
    # Main Training Loop
    # =========================================================
    logger.info("\n" + "="*70)
    logger.info("Starting Training Loop")
    logger.info("="*70)
    
    for epoch in range(first_epoch, config.num_train_epochs):
        transformer.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0
        
        # Set epoch for distributed sampler
        if hasattr(train_dataloader, 'batch_sampler'):
            if hasattr(train_dataloader.batch_sampler, 'set_epoch'):
                train_dataloader.batch_sampler.set_epoch(epoch)
            elif hasattr(train_dataloader.batch_sampler, 'sampler') and hasattr(train_dataloader.batch_sampler.sampler, 'set_epoch'):
                train_dataloader.batch_sampler.sampler.set_epoch(epoch)
        
        # Calculate how many steps to skip in the first epoch after resuming
        # Note: With gradient_accumulation_steps > 1, each global_step corresponds to 
        # gradient_accumulation_steps batches, so we need to skip accordingly
        if epoch == first_epoch and initial_global_step > 0:
            # Number of batches to skip = global_step * gradient_accumulation_steps
            # But since we count by optimization steps, we use global_step directly for step comparison
            resume_step = initial_global_step % num_update_steps_per_epoch
            if resume_step > 0:
                # Use accelerate's skip_first_batches for efficient skipping (doesn't load data)
                num_batches_to_skip = resume_step * config.gradient_accumulation_steps
                logger.info(f"[INFO] Skipping {num_batches_to_skip} batches ({resume_step} steps * {config.gradient_accumulation_steps} GA) in epoch {epoch} to resume from step {initial_global_step}")
                active_dataloader = accelerator.skip_first_batches(train_dataloader, num_batches_to_skip)
            else:
                active_dataloader = train_dataloader
        else:
            resume_step = 0
            active_dataloader = train_dataloader
        
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(transformer):
                # Run training iteration
                loss = train_iteration_func(
                    batch=batch,
                    vae=vae,
                    noise_scheduler_copy=noise_scheduler_copy,
                    transformer=transformer,
                    config=config,
                    accelerator=accelerator,
                    global_step=global_step,
                    weight_dtype=weight_dtype,
                )

                # Backward pass
                accelerator.backward(loss)
                
                # Gradient clipping
                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Update EMA model
                # ema_on_gpu=True  → update directly on GPU (no CPU↔GPU copy overhead)
                # ema_on_gpu=False → copy params to CPU then update (saves GPU memory)
                ema_steps = config.get('ema_steps', 100)
                if ema_transformer is not None and global_step % ema_steps == 0:
                    # For ZeRO-2: transformer params are full on each GPU → direct copy.
                    # For ZeRO-3: transformer params are PARTITIONED → must gather first.
                    
                    ema_decay = ema_transformer.cur_decay_value if hasattr(ema_transformer, 'cur_decay_value') else ema_transformer.decay
                    ema_transformer.optimization_step += 1
                    decay = ema_transformer.get_decay(ema_transformer.optimization_step)
                    ema_transformer.cur_decay_value = decay
                    one_minus_decay = 1 - decay
                    
                    # Detect ZeRO-3 (params partitioned across GPUs)
                    _is_zero3 = (
                        accelerator.distributed_type == DistributedType.DEEPSPEED and
                        hasattr(accelerator.state, 'deepspeed_plugin') and
                        getattr(accelerator.state.deepspeed_plugin, 'zero_stage', 0) == 3
                    )
                    
                    with torch.no_grad():
                        if _is_zero3:
                            import deepspeed
                            for s_param, param in zip(ema_transformer.shadow_params, transformer.parameters()):
                                # GatheredParameters: all ranks gather full param (collective op)
                                with deepspeed.zero.GatheredParameters(param):
                                    # CRITICAL: promote to float32 for EMA arithmetic.
                                    # param.data is bf16, and (1-decay) * delta ≈ 1e-7..1e-4
                                    # is below bf16 ULP → bf16 arithmetic rounds to zero.
                                    param_f32 = param.data.detach().to(ema_target_device).float()
                                    if param.requires_grad:
                                        s_param.sub_(one_minus_decay * (s_param - param_f32))
                                    else:
                                        s_param.copy_(param_f32)
                                    del param_f32
                        elif ema_on_gpu:
                            # ZeRO-2 / DDP + GPU EMA: shadow_params on same GPU, no copy needed
                            for s_param, param in zip(ema_transformer.shadow_params, transformer.parameters()):
                                param_f32 = param.data.detach().float()  # bf16 → float32 on GPU
                                if param.requires_grad:
                                    s_param.sub_(one_minus_decay * (s_param - param_f32))
                                else:
                                    s_param.copy_(param_f32)
                                del param_f32
                        else:
                            # ZeRO-2 / DDP + CPU EMA: copy params to CPU then update
                            for s_param, param in zip(ema_transformer.shadow_params, transformer.parameters()):
                                param_cpu = param.data.detach().cpu().float()
                                if param.requires_grad:
                                    s_param.sub_(one_minus_decay * (s_param - param_cpu))
                                else:
                                    s_param.copy_(param_cpu)
                                del param_cpu
                    
                    if global_step % (ema_steps * 10) == 0:  # Log every 10 EMA updates
                        logger.info(f"[INFO] EMA updated on {'GPU' if ema_on_gpu else 'CPU'} at step {global_step}, decay={decay:.6f}")
                
                # Logging - only call loss.item() when we actually need to log
                # This avoids GPU-CPU sync every step which was causing ~0.5s overhead
                log_steps = config.get('logging_steps', config.get('log_steps', 10))
                if global_step % log_steps == 0:
                    # Only sync loss to CPU when logging
                    current_loss = loss.detach().item()
                    loss_tracker.update(current_loss)
                    epoch_loss_sum += current_loss
                    epoch_steps += 1
                    loss_stats = loss_tracker.get_stats()
                    is_healthy, health_msg = loss_tracker.check_health(current_loss, global_step)
                    
                    logs = {
                        "loss": current_loss,
                        "loss_avg": loss_stats.get("loss_avg", current_loss),
                        "loss_min": loss_stats.get("loss_all_time_min", current_loss),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step,
                        "epoch": epoch,
                    }
                    progress_bar.set_postfix(loss=f"{current_loss:.4f}", lr=f"{logs['lr']:.2e}")
                    accelerator.log(logs, step=global_step)
                    
                    # Log health status periodically (controlled by verbose_logging)
                    verbose_logging = config.get('verbose_logging', True)
                    if verbose_logging and global_step % 100 == 0 and accelerator.is_main_process:
                        logger.info(f"\n[Step {global_step}] {health_msg}")
                        logger.info(f"  Loss avg (last 100): {loss_stats.get('loss_avg', 0):.4f}")
                        logger.info(f"  Loss range: [{loss_stats.get('loss_recent_min', 0):.4f}, {loss_stats.get('loss_recent_max', 0):.4f}]")

                # Checkpointing
                if global_step % config.checkpointing_steps == 0:
                    save_model_checkpoint(
                        transformer=transformer,
                        accelerator=accelerator,
                        config=config,
                        global_step=global_step,
                        logger=logger,
                        ema_transformer=ema_transformer,
                    )
                    
                    # Clean old checkpoints (only on main process)
                    if accelerator.is_main_process and config.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(config.model_output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        if len(checkpoints) > config.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - config.checkpoints_total_limit
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint_path = os.path.join(config.model_output_dir, removing_checkpoint)
                                logger.info(f"Removing old checkpoint: {removing_checkpoint_path}")
                                
                                for i in range(3): 
                                    try:
                                        shutil.rmtree(removing_checkpoint_path, ignore_errors=True)
                                        
                                        if os.path.exists(removing_checkpoint_path):
                                            continue
                                        break
                                    
                                    except Exception as e:
                                        logger.warning(f"Failed to remove {removing_checkpoint_path}, retry {i+1}/3: {e}")
                                        time.sleep(5)
                
                
                # Validation
                if validation_func is not None and validation_prompts is not None:
                    should_validate = (
                        global_step % validation_steps == 0 or 
                        (global_step == 1 and validation_steps <= 1000) or
                        global_step == config.max_train_steps
                    )
                    
                    if should_validate:
                        logger.info(f"\n🔍 Running validation at step {global_step}...")
                        
                        # === Only main process generates validation images ===
                        # Non-main ranks skip heavy work and wait at the barrier below.
                        # Without this, ranks 1-3 finish instantly and proceed to the
                        # next training step's NCCL ALLREDUCE while rank 0 is still
                        # generating images, causing NCCL timeout after 10 minutes.
                        if accelerator.is_main_process:
                            # Free memory before validation
                            gc.collect()
                            torch.cuda.empty_cache()
                            
                            # Lazy-load text encoder for validation if it was skipped
                            _te_was_lazy_loaded = False
                            if text_encoder is None:
                                logger.info("[INFO] Lazy-loading text encoder for validation...")
                                text_encoder = model_factory.load_text_encoder(skip_load=False)
                                _te_was_lazy_loaded = True
                            
                            # Move VAE and text encoder to GPU for validation
                            if config.use_parquet_dataset:
                                vae.to(accelerator.device)
                                if text_encoder is not None:
                                    text_encoder.to(accelerator.device)
                            
                            try:
                                validation_func(
                                    vae=vae,
                                    transformer=transformer,
                                    text_encoder=text_encoder,
                                    tokenizer=tokenizer,
                                    scheduler=noise_scheduler,
                                    accelerator=accelerator,
                                    args=config,
                                    global_step=global_step,
                                )
                            except Exception as e:
                                logger.error(f"Validation failed: {e}")
                            
                            # Move back to CPU and aggressively free memory
                            if config.use_parquet_dataset:
                                vae.to("cpu")
                                if text_encoder is not None:
                                    text_encoder.to("cpu")
                                # If lazy loaded, free it entirely to save CPU memory
                                if _te_was_lazy_loaded:
                                    del text_encoder
                                    text_encoder = None
                            
                            # Aggressive memory cleanup to prevent OOM on next training step
                            gc.collect()
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            gc.collect()
                        
                        # === Critical: sync all ranks before resuming training ===
                        # Ranks 1-3 wait here while rank 0 generates validation images.
                        # Uses default process group (timeout=180min), safe for long validations.
                        accelerator.wait_for_everyone()
                        
                        # Set transformer back to train mode (all ranks)
                        transformer.train()

            if global_step >= config.max_train_steps:
                break
        
        # End of epoch summary
        if accelerator.is_main_process and epoch_steps > 0:
            epoch_avg_loss = epoch_loss_sum / epoch_steps
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch} completed: avg_loss = {epoch_avg_loss:.4f}")
            logger.info(f"{'='*50}\n")

    # =========================================================
    # Final Checkpoint & Cleanup
    # =========================================================
    accelerator.wait_for_everyone()
    
    # Save final checkpoint
    save_model_checkpoint(
        transformer=transformer,
        accelerator=accelerator,
        config=config,
        global_step=global_step,
        logger=logger,
        is_final=True,
        ema_transformer=ema_transformer,
    )
    
    # Final validation
    if validation_func is not None and validation_prompts is not None and accelerator.is_main_process:
        logger.info("\n🎯 Running final validation...")
        # Lazy-load text encoder if needed
        if text_encoder is None:
            logger.info("[INFO] Lazy-loading text encoder for final validation...")
            text_encoder = model_factory.load_text_encoder(skip_load=False)
        if config.use_parquet_dataset:
            vae.to(accelerator.device)
            if text_encoder is not None:
                text_encoder.to(accelerator.device)
        
        try:
            validation_func(
                vae=vae,
                transformer=transformer,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                scheduler=noise_scheduler,
                accelerator=accelerator,
                args=config,
                global_step=global_step,
            )
        except Exception as e:
            logger.error(f"Final validation failed: {e}")
        
        if config.use_parquet_dataset:
            vae.to("cpu")
            if text_encoder is not None:
                text_encoder.to("cpu")
            torch.cuda.empty_cache()

    # Ensure all ranks sync after final validation before cleanup
    accelerator.wait_for_everyone()

    accelerator.end_training()
    logger.info("🎉 Training completed successfully!")


if __name__ == "__main__":
    main()
