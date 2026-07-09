"""
SciForma M-DPO Training Script

Trains a Flux2Klein policy model using DPO on winner/loser image pairs.
The reference model is a frozen copy of the stage-2 checkpoint (EMA weights).

Key differences from train_sft.py (SFT):
  1. Two transformer instances: policy (trainable) + reference (frozen).
  2. DPO loss instead of flow-matching MSE.
  3. Dataset supplies (winner_latent, loser_latent, text_embeds) per sample.
  4. Reference model is placed on GPU but NOT wrapped by accelerator.prepare.

Supported setups:
  - Flux2Klein 9B + ZeRO-2 (4× B200): reference fits without issues.
  - Flux2Klein 9B + ZeRO-2 + ref_on_cpu=True: reference kept on CPU to save VRAM.

Usage:
    # 4-GPU ZeRO-2 (B200):
    accelerate launch --config_file training/accelerate_cfg/b200_zero2_bf16_4gpu_dpo.yaml \\
        training/scripts/train_dpo.py training/configs/mdpo/mdpo.py
"""

# =========================================================
# Path & Environment
# =========================================================
import os
import sys
import os.path as osp
from pathlib import Path

sys.path.insert(
    0,
    osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), "..", "..")),
)

# ── Secrets ──────────────────────────────────────────────
try:
    from _local_secrets import (
        HF_TOKEN, WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT, WANDB_BASE_URL,
    )
    for key, val in [
        ('HF_TOKEN', HF_TOKEN), ('WANDB_API_KEY', WANDB_API_KEY),
        ('WANDB_ENTITY', WANDB_ENTITY), ('WANDB_PROJECT', WANDB_PROJECT),
        ('WANDB_BASE_URL', WANDB_BASE_URL),
    ]:
        if not os.environ.get(key):
            os.environ[key] = val
except ImportError:
    pass

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# =========================================================
# Standard Library
# =========================================================
import copy
import json
import math
import shutil
import logging
import time
import gc
from datetime import timedelta
from pathlib import Path
from collections import OrderedDict

# =========================================================
# Third-party
# =========================================================
from tqdm.auto import tqdm
import torch
import torch.utils.checkpoint
from torch.utils.data import DataLoader

# Accelerate
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from accelerate import DataLoaderConfiguration

# Diffusers / Transformers
import transformers
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_loss_weighting_for_sd3
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

# =========================================================
# Project
# =========================================================
from sciforma.registry import DATASETS, TRAIN_ITERATION_FUNCS, VALIDATION_FUNCS
from sciforma.utils import parse_config, unwrap_model, ModelFactory

# Ensure iteration func and dataset modules are imported (triggers registry)
# Only include modules that exist in SciForma/sciforma/
import sciforma.train_iteration_funcs.Flux2Klein_dpo_iteration_func  # noqa: F401  (helper funcs used by md3po)
import sciforma.datasets.arxiv_parquet_dataset_md3po                 # noqa: F401  (M-DPO dataset)

check_min_version("0.32.0")
logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Loss Tracker
# ─────────────────────────────────────────────────────────────────────────────

class DPOLossTracker:
    """Track DPO-specific loss metrics over a rolling window."""

    def __init__(self, window=100):
        self.window = window
        self._buf = []

    def update(self, metrics: dict):
        self._buf.append({k: float(v) for k, v in metrics.items()})
        if len(self._buf) > self.window:
            self._buf.pop(0)

    def get_stats(self):
        if not self._buf:
            return {}
        keys = self._buf[0].keys()
        return {f"{k}_avg": sum(d[k] for d in self._buf) / len(self._buf)
                for k in keys}


# ─────────────────────────────────────────────────────────────────────────────
# Reference Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_ref_transformer_from_ema(ema_weights_path, transformer_template):
    """
    Build a frozen reference transformer by loading EMA shadow_params
    into a fresh copy of the transformer architecture.

    Args:
        ema_weights_path: path to  ema_weights.pt  (has 'shadow_params' key)
        transformer_template: already-loaded policy transformer (for architecture)

    Returns:
        ref_transformer: frozen nn.Module (same dtype as template)
    """
    logger.info(f"[DPO] Loading reference from EMA weights: {ema_weights_path}")
    ema_state = torch.load(ema_weights_path, map_location="cpu", weights_only=False)
    shadow_params = ema_state.get("shadow_params")
    if shadow_params is None:
        raise ValueError(f"'shadow_params' not found in {ema_weights_path}")

    # Build state dict from shadow_params
    param_names = list(transformer_template.state_dict().keys())
    if len(shadow_params) != len(param_names):
        raise ValueError(
            f"EMA shadow_params count ({len(shadow_params)}) != "
            f"transformer params count ({len(param_names)})"
        )

    new_state = OrderedDict()
    template_dtype = next(transformer_template.parameters()).dtype
    for name, sp in zip(param_names, shadow_params):
        new_state[name] = sp.to(dtype=template_dtype)

    # Create reference transformer (deep copy of architecture)
    ref_transformer = copy.deepcopy(transformer_template)
    missing, unexpected = ref_transformer.load_state_dict(new_state, strict=True)
    if missing or unexpected:
        logger.warning(f"[DPO] Ref state_dict mismatch — missing: {missing}, unexpected: {unexpected}")

    ref_transformer.requires_grad_(False)
    ref_transformer.eval()
    logger.info(f"[DPO] Reference transformer loaded and frozen (dtype={template_dtype})")
    return ref_transformer


def load_ref_transformer_from_safetensors(safetensors_dir_or_file, transformer_template):
    """
    Load reference transformer from a safetensors checkpoint (transformer/
    folder saved by save_model_checkpoint). Supports both single-file and
    sharded (index.json) safetensors.
    """
    import json
    from safetensors.torch import load_file as st_load

    if os.path.isdir(safetensors_dir_or_file):
        safetensors_dir = safetensors_dir_or_file
        single_file = os.path.join(safetensors_dir, "diffusion_pytorch_model.safetensors")
        index_file = os.path.join(safetensors_dir, "diffusion_pytorch_model.safetensors.index.json")
        if os.path.exists(single_file):
            logger.info(f"[DPO] Loading reference from safetensors: {single_file}")
            state = st_load(single_file, device="cpu")
        elif os.path.exists(index_file):
            with open(index_file) as f:
                idx = json.load(f)
            shard_files = sorted(set(idx["weight_map"].values()))
            state = {}
            for sf in shard_files:
                state.update(st_load(os.path.join(safetensors_dir, sf), device="cpu"))
            logger.info(f"[DPO] Loaded reference from {len(shard_files)} safetensor shards.")
        else:
            raise FileNotFoundError(f"No safetensors found at {safetensors_dir}")
    else:
        # Direct file path (backward compat)
        logger.info(f"[DPO] Loading reference from safetensors: {safetensors_dir_or_file}")
        state = st_load(safetensors_dir_or_file, device="cpu")

    template_dtype = next(transformer_template.parameters()).dtype
    state = {k: v.to(dtype=template_dtype) for k, v in state.items()}

    ref_transformer = copy.deepcopy(transformer_template)
    missing, unexpected = ref_transformer.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"[DPO] Ref missing keys (will use init): {missing}")
    if unexpected:
        logger.warning(f"[DPO] Ref unexpected keys (ignored): {unexpected}")

    ref_transformer.requires_grad_(False)
    ref_transformer.eval()
    logger.info(f"[DPO] Reference transformer loaded and frozen (dtype={template_dtype})")
    return ref_transformer


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint Save
# ─────────────────────────────────────────────────────────────────────────────

def save_model_checkpoint(transformer, accelerator, config, global_step, logger, is_final=False, ema_transformer=None):
    """Save DPO policy checkpoint + optional EMA weights."""
    save_path = (
        os.path.join(config.model_output_dir, "final_model") if is_final
        else os.path.join(config.model_output_dir, f"checkpoint-{global_step}")
    )

    is_deepspeed_zero3 = (
        accelerator.distributed_type == DistributedType.DEEPSPEED
        and hasattr(accelerator.state, "deepspeed_plugin")
        and getattr(accelerator.state.deepspeed_plugin, "zero_stage", 0) == 3
    )

    if accelerator.distributed_type == DistributedType.FSDP:
        logger.info(f"[DPO] Saving FSDP checkpoint to {save_path}")
        accelerator.save_state(save_path)

    elif is_deepspeed_zero3:
        import deepspeed
        os.makedirs(save_path, exist_ok=True)
        unwrapped = accelerator.unwrap_model(transformer)
        unwrapped = unwrapped._orig_mod if is_compiled_module(unwrapped) else unwrapped

        full_state = {}
        for name, param in unwrapped.named_parameters():
            with deepspeed.zero.GatheredParameters(param):
                if accelerator.is_main_process:
                    full_state[name] = param.data.detach().cpu().clone()

        if accelerator.is_main_process:
            from safetensors.torch import save_file as st_save
            tr_path = os.path.join(save_path, "transformer")
            os.makedirs(tr_path, exist_ok=True)
            st_save(full_state, os.path.join(tr_path, "diffusion_pytorch_model.safetensors"))
            if hasattr(unwrapped, "config"):
                with open(os.path.join(tr_path, "config.json"), "w") as f:
                    json.dump(dict(unwrapped.config), f, indent=2, default=str)
            del full_state
            # Save EMA weights if available
            if ema_transformer is not None:
                ema_save_path = os.path.join(save_path, "ema_weights.pt")
                ema_state = ema_transformer.state_dict()
                ema_state['shadow_params'] = [p.bfloat16() for p in ema_state['shadow_params']]
                torch.save(ema_state, ema_save_path)
                logger.info(f"[DPO] Saved EMA weights to {ema_save_path} (cast to bf16)")
            torch.save({"global_step": global_step},
                       os.path.join(save_path, "training_state.pt"))
            logger.info(f"[DPO] Saved ZeRO-3 checkpoint to {save_path}")
        accelerator.wait_for_everyone()

    else:
        if accelerator.is_main_process:
            os.makedirs(save_path, exist_ok=True)
            unwrapped = accelerator.unwrap_model(transformer)
            unwrapped = unwrapped._orig_mod if is_compiled_module(unwrapped) else unwrapped
            if getattr(config, 'use_lora', False):
                # LoRA: save only adapter weights (adapter_config.json + adapter_model.safetensors)
                adapter_path = os.path.join(save_path, "lora_adapter")
                unwrapped.save_pretrained(adapter_path)
                logger.info(f"[DPO-LoRA] Saved LoRA adapter to {adapter_path}")
            else:
                tr_path = os.path.join(save_path, "transformer")
                unwrapped.save_pretrained(tr_path)
            # Save EMA weights if available
            if ema_transformer is not None:
                ema_save_path = os.path.join(save_path, "ema_weights.pt")
                ema_state = ema_transformer.state_dict()
                ema_state['shadow_params'] = [p.bfloat16() for p in ema_state['shadow_params']]
                torch.save(ema_state, ema_save_path)
                logger.info(f"[DPO] Saved EMA weights to {ema_save_path} (cast to bf16)")
            torch.save({"global_step": global_step},
                       os.path.join(save_path, "training_state.pt"))
            logger.info(f"[DPO] Saved checkpoint to {save_path}")

    accelerator.save_state(os.path.join(save_path, "accelerator"))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    config = parse_config(train=True)

    # ──────────────────────────────────────────────────────
    # Accelerator setup
    # ──────────────────────────────────────────────────────
    if config.report_to == "wandb" and config.get("hub_token", None):
        raise ValueError("Do not pass hub_token when using wandb.")

    if torch.backends.mps.is_available() and config.mixed_precision == "bf16":
        raise ValueError("BF16 not supported on MPS.")

    logging_dir = Path(config.model_output_dir, config.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.model_output_dir, logging_dir=logging_dir
    )
    # LoRA DPO: frozen base-model params receive no grads → need find_unused_parameters=True
    kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=getattr(config, 'use_lora', False)
    )
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=180))
    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs, init_kwargs],
        dataloader_config=dataloader_config,
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if config.report_to == "wandb" and not is_wandb_available():
        raise ImportError("Install wandb for logging.")

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

    if config.seed is not None:
        set_seed(config.seed)

    if accelerator.is_main_process:
        os.makedirs(config.model_output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    config.weight_dtype = weight_dtype

    if config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.scale_lr:
        config.learning_rate = (
            config.learning_rate
            * config.gradient_accumulation_steps
            * config.train_batch_size
            * accelerator.num_processes
        )

    # ──────────────────────────────────────────────────────
    # Distributed type detection (needed before model load)
    # ──────────────────────────────────────────────────────
    distributed_type_str = str(getattr(accelerator.state, "distributed_type", "NO"))
    is_fsdp      = "FSDP"      in distributed_type_str
    is_deepspeed = "DEEPSPEED" in distributed_type_str

    # ──────────────────────────────────────────────────────
    # Load base models (policy transformer + VAE)
    # ──────────────────────────────────────────────────────
    model_type = getattr(config, "model_type", "Flux2Klein")
    logger.info(f"[DPO] Model type: {model_type}")

    with accelerator.main_process_first():
        model_factory = ModelFactory(config)
        (
            vae,
            transformer,       # policy (will be made trainable)
            tokenizer,
            text_encoder,
            noise_scheduler,
            text_encoding_pipeline,
            vae_scale_factor,
        ) = model_factory.load_all(skip_text_encoder=True)   # parquet mode → skip TE

    # Cast to bf16 for distributed training
    if is_fsdp or is_deepspeed:
        transformer = transformer.to(dtype=torch.bfloat16)

    # VAE / text encoder stay on CPU
    vae.to(dtype=weight_dtype, device="cpu")
    if text_encoder is not None:
        text_encoder.to(dtype=weight_dtype, device="cpu")

    latents_mean, latents_std = model_factory.get_latents_stats(vae, accelerator.device)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    config.vae_scale_factor = vae_scale_factor
    config.latents_mean = latents_mean
    config.latents_std  = latents_std

    # ──────────────────────────────────────────────────────
    # Load policy weights from stage-2 checkpoint
    # ──────────────────────────────────────────────────────
    policy_init_path = config.get("policy_init_path", None)
    if policy_init_path:
        logger.info(f"[DPO] Initialising policy from: {policy_init_path}")
        if policy_init_path.endswith(".pt"):
            # EMA weights
            ema_state = torch.load(policy_init_path, map_location="cpu", weights_only=False)
            if "shadow_params" in ema_state:
                param_names = list(transformer.state_dict().keys())
                new_state = OrderedDict(
                    (n, sp.to(dtype=torch.bfloat16))
                    for n, sp in zip(param_names, ema_state["shadow_params"])
                )
                transformer.load_state_dict(new_state, strict=True)
                logger.info(f"[DPO] Policy loaded from EMA shadow_params.")
            else:
                transformer.load_state_dict(ema_state, strict=False)
        else:
            # safetensors directory (single or sharded)
            from safetensors.torch import load_file as st_load
            safetensors_file = os.path.join(policy_init_path, "diffusion_pytorch_model.safetensors")
            index_file = os.path.join(policy_init_path, "diffusion_pytorch_model.safetensors.index.json")
            if os.path.exists(safetensors_file):
                state = st_load(safetensors_file, device="cpu")
            elif os.path.exists(index_file):
                import json
                with open(index_file) as f:
                    idx = json.load(f)
                shard_files = sorted(set(idx["weight_map"].values()))
                state = {}
                for sf in shard_files:
                    state.update(st_load(os.path.join(policy_init_path, sf), device="cpu"))
                logger.info(f"[DPO] Loaded {len(shard_files)} safetensor shards.")
            else:
                raise FileNotFoundError(f"No safetensors found at {policy_init_path}")
            state = {k: v.to(dtype=torch.bfloat16) for k, v in state.items()}
            missing, unexpected = transformer.load_state_dict(state, strict=False)
            if missing:
                logger.warning(f"[DPO] Missing keys (will use init): {missing}")
            if unexpected:
                logger.warning(f"[DPO] Unexpected keys (ignored): {unexpected}")
            logger.info(f"[DPO] Policy loaded from safetensors.")
    else:
        logger.warning("[DPO] policy_init_path not set — policy uses base pretrained weights.")

    # ──────────────────────────────────────────────────────
    # LoRA setup OR load frozen reference transformer
    # ──────────────────────────────────────────────────────
    # KEY INSIGHT: with LoRA we do NOT need a second copy of the 9B model.
    # The reference forward is the base model (LoRA adapters disabled).
    # This saves ~18 GB VRAM per GPU vs the full-copy approach.
    use_lora = getattr(config, 'use_lora', False)
    ref_on_cpu = config.get("ref_on_cpu", False)
    ref_transformer = None

    if use_lora:
        # Attach LoRA adapters to the policy transformer.
        # Policy = base weights + trainable LoRA Δ.
        # Reference = base weights only (LoRA disabled at forward time).
        from peft import LoraConfig, get_peft_model
        _lora_targets = [l.strip() for l in config.lora_layers.split(",")]
        _lora_cfg = LoraConfig(
            r=config.rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=getattr(config, 'lora_dropout', 0.0),
            target_modules=_lora_targets,
            bias="none",
        )
        transformer = get_peft_model(transformer, _lora_cfg)
        transformer.print_trainable_parameters()
        logger.info("[DPO-LoRA] LoRA adapters attached. Reference = base model (adapters disabled).")
        ref_on_cpu = False   # no separate ref model; CPU offload not applicable

    else:
        # Full fine-tune mode: load a frozen copy of the model as the reference.
        ref_init_path = config.get("ref_init_path", None)
        if ref_init_path is None:
            ref_init_path = policy_init_path
            logger.info("[DPO] ref_init_path not set, defaulting to policy_init_path.")

        if ref_init_path and ref_init_path.endswith(".pt"):
            ref_transformer = load_ref_transformer_from_ema(ref_init_path, transformer)
        elif ref_init_path:
            ref_transformer = load_ref_transformer_from_safetensors(ref_init_path, transformer)
        else:
            logger.warning("[DPO] ref_init_path not set — reference uses base pretrained weights!")
            ref_transformer = copy.deepcopy(transformer)
            ref_transformer.requires_grad_(False)
            ref_transformer.eval()

        # Materialize any meta-device parameters (e.g. newly-init keys absent from checkpoint)
        # before moving to target device to avoid "Cannot copy out of meta tensor" error.
        _meta_keys = [n for n, p in ref_transformer.named_parameters() if p.is_meta]
        if _meta_keys:
            from accelerate.utils import set_module_tensor_to_device
            for _n in _meta_keys:
                _p = dict(ref_transformer.named_parameters())[_n]
                set_module_tensor_to_device(
                    ref_transformer, _n, device="cpu",
                    value=torch.zeros(_p.shape, dtype=weight_dtype),
                )
            logger.warning(f"[DPO] Materialized {len(_meta_keys)} meta-tensor param(s) in ref: {_meta_keys}")

        if ref_on_cpu:
            ref_transformer = ref_transformer.cpu()
            logger.info("[DPO] Reference transformer kept on CPU (ref_on_cpu=True).")
        else:
            ref_transformer = ref_transformer.to(device=accelerator.device, dtype=weight_dtype)
            logger.info(f"[DPO] Reference transformer on GPU ({accelerator.device}).")

    # ──────────────────────────────────────────────────────
    # Gradient checkpointing on policy only
    # ──────────────────────────────────────────────────────
    if config.gradient_checkpointing:
        if use_lora:
            # PEFT + gradient checkpointing for diffusers models:
            # 1) Register forward hook so first-layer outputs require grad
            #    (needed because LoRA freezes base params, but checkpointing
            #    requires grads to flow through recomputed segments).
            def _make_inputs_require_grad(module, input, output):
                if isinstance(output, tuple):
                    for o in output:
                        if hasattr(o, 'requires_grad_'):
                            o.requires_grad_(True)
                elif hasattr(output, 'requires_grad_'):
                    output.requires_grad_(True)

            _bm = getattr(transformer, 'base_model', transformer)
            _inner = getattr(_bm, 'model', _bm)
            _inner.register_forward_hook(_make_inputs_require_grad)
            # 2) Enable gradient checkpointing on the inner diffusers model
            _inner.enable_gradient_checkpointing()
        else:
            transformer.enable_gradient_checkpointing()
        logger.info("[DPO] Gradient checkpointing enabled on policy.")

    # ──────────────────────────────────────────────────────
    # Policy trainability (pre-accelerator.prepare)
    # ──────────────────────────────────────────────────────
    if not use_lora:
        # Full fine-tune: mark all params trainable.
        # (LoRA already configures this via get_peft_model.)
        for param in transformer.parameters():
            param.requires_grad = True

    if not (is_fsdp or is_deepspeed):
        # Materialize any meta-tensor parameters before moving to device
        def _materialize_meta(module):
            for name, param in module.named_parameters():
                if param.is_meta:
                    module_path = name.rsplit('.', 1)
                    parent = module
                    for part in name.split('.')[:-1]:
                        parent = getattr(parent, part)
                    attr_name = name.split('.')[-1]
                    new_param = torch.nn.Parameter(
                        torch.zeros(param.shape, dtype=weight_dtype, device='cpu'),
                        requires_grad=param.requires_grad,
                    )
                    setattr(parent, attr_name, new_param)
        _materialize_meta(transformer)
        transformer.to(device=accelerator.device, dtype=weight_dtype)

    num_trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e6
    logger.info(f"[DPO] Policy trainable parameters: {num_trainable:.2f}M")

    # ──────────────────────────────────────────────────────
    # EMA Model (if enabled)
    # ──────────────────────────────────────────────────────
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
        if ema_on_gpu:
            ema_device = accelerator.device
            ema_transformer.shadow_params = [p.to(ema_device).float() for p in ema_transformer.shadow_params]
            logger.info(f"[DPO] EMA model created on GPU ({ema_device}, float32) with decay={ema_decay}")
        else:
            ema_transformer.shadow_params = [p.cpu().float() for p in ema_transformer.shadow_params]
            logger.info(f"[DPO] EMA model created on CPU (float32) with decay={ema_decay}")

    # ──────────────────────────────────────────────────────
    # Optimizer
    # ──────────────────────────────────────────────────────
    use_8bit = config.use_8bit_adam and not is_fsdp and not is_deepspeed
    if (is_fsdp or is_deepspeed) and config.use_8bit_adam:
        logger.warning("[DPO] 8-bit Adam disabled for FSDP/DeepSpeed.")

    if config.optimizer.lower() == "adamw":
        if use_8bit:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW
        optimizer = optimizer_cls(
            [p for p in transformer.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.adam_weight_decay,
            eps=config.adam_epsilon,
        )
    elif config.optimizer.lower() == "prodigy":
        import prodigyopt
        optimizer = prodigyopt.Prodigy(
            [p for p in transformer.parameters() if p.requires_grad],
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

    # ──────────────────────────────────────────────────────
    # Dataset & DataLoader
    # ──────────────────────────────────────────────────────
    logger.info("[DPO] Loading DPO dataset...")
    dataset_cfg = config.dataset_cfg
    dataset = DATASETS.build(dataset_cfg)

    collate_fn = getattr(dataset, "collate_fn", None)

    sampler_cfg = config.get("sampler_cfg", None)
    if sampler_cfg is not None:
        sampler_cfg["dataset"]       = dataset
        sampler_cfg["num_replicas"]  = 1
        sampler_cfg["rank"]          = 0
        sampler_cfg["batch_size"]    = config.train_batch_size
        if sampler_cfg.get("seed") is None:
            sampler_cfg["seed"] = config.seed or 42
        batch_sampler = DATASETS.build(sampler_cfg)
        train_dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=config.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=config.dataloader_num_workers > 0,
            prefetch_factor=4 if config.dataloader_num_workers > 0 else None,
        )
        if is_deepspeed and hasattr(accelerator.state, "deepspeed_plugin"):
            accelerator.state.deepspeed_plugin.deepspeed_config[
                "train_micro_batch_size_per_gpu"] = config.train_batch_size
    else:
        train_dataloader = DataLoader(
            dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=config.dataloader_num_workers > 0,
            prefetch_factor=4 if config.dataloader_num_workers > 0 else None,
        )

    # ──────────────────────────────────────────────────────
    # LR Scheduler
    # ──────────────────────────────────────────────────────
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
        num_cycles=config.lr_num_cycles,
        power=config.lr_power,
    )

    # ──────────────────────────────────────────────────────
    # Resume checkpoint (policy weights only)
    # ──────────────────────────────────────────────────────
    resume_checkpoint_path = None
    auto_resume = True
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint == "latest":
            auto_resume = True
        else:
            pth = config.resume_from_checkpoint
            resume_checkpoint_path = (
                pth if os.path.isabs(pth)
                else os.path.join(config.model_output_dir, pth)
            )
            auto_resume = False

    if auto_resume and os.path.exists(config.model_output_dir):
        dirs = sorted(
            [d for d in os.listdir(config.model_output_dir) if d.startswith("checkpoint")],
            key=lambda x: int(x.split("-")[1])
        )
        if dirs:
            resume_checkpoint_path = os.path.join(config.model_output_dir, dirs[-1])
            logger.info(f"[DPO] Auto-detected checkpoint: {resume_checkpoint_path}")

    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        if use_lora:
            # LoRA checkpoint: load adapter weights from lora_adapter/ folder.
            lora_ckpt = os.path.join(resume_checkpoint_path, "lora_adapter")
            if os.path.exists(lora_ckpt):
                logger.info(f"[DPO-LoRA] Loading LoRA adapter from: {lora_ckpt}")
                try:
                    from peft import set_peft_model_state_dict
                    from safetensors.torch import load_file as st_load
                    adapter_st = os.path.join(lora_ckpt, "adapter_model.safetensors")
                    if os.path.exists(adapter_st):
                        lora_state = st_load(adapter_st, device="cpu")
                        set_peft_model_state_dict(transformer, lora_state)
                        logger.info("[DPO-LoRA] LoRA adapter loaded from checkpoint.")
                    else:
                        logger.warning(f"[DPO-LoRA] adapter_model.safetensors not found in {lora_ckpt}")
                except Exception as e:
                    logger.error(f"[DPO-LoRA] Failed to load LoRA adapter: {e}")
            else:
                logger.warning(f"[DPO-LoRA] No lora_adapter/ in checkpoint; LoRA starts from zero delta.")
        else:
            # Full fine-tune checkpoint: load transformer safetensors.
            tr_ckpt = os.path.join(resume_checkpoint_path, "transformer")
            if os.path.exists(tr_ckpt):
                logger.info(f"[DPO] Loading policy weights from: {tr_ckpt}")
                try:
                    st_file = os.path.join(tr_ckpt, "diffusion_pytorch_model.safetensors")
                    if os.path.exists(st_file):
                        from safetensors.torch import load_file as st_load
                        state = {k: v.to(dtype=torch.bfloat16)
                                 for k, v in st_load(st_file, device="cpu").items()}
                        transformer.load_state_dict(state, strict=True)
                        logger.info("[DPO] Policy weights loaded from safetensors.")
                    else:
                        transformer.load_state_dict(
                            torch.load(os.path.join(tr_ckpt, "pytorch_model.bin"),
                                       map_location="cpu", weights_only=False),
                            strict=True,
                        )
                except Exception as e:
                    logger.error(f"[DPO] Failed to load policy from checkpoint: {e}")
                    resume_checkpoint_path = None
            else:
                resume_checkpoint_path = None

    # ──────────────────────────────────────────────────────
    # accelerator.prepare  (policy only — NOT reference)
    # ──────────────────────────────────────────────────────
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # Move reference to device AFTER prepare (device may change for some backends)
    if ref_transformer is not None and not ref_on_cpu:
        ref_transformer = ref_transformer.to(device=accelerator.device, dtype=weight_dtype)

    # ──────────────────────────────────────────────────────
    # Training bookkeeping
    # ──────────────────────────────────────────────────────
    # Training bookkeeping
    # ──────────────────────────────────────────────────────
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps
    )
    logger.info(f"  len(dataset)={len(dataset)}, len(dataloader)={len(train_dataloader)}, "
                f"num_processes={accelerator.num_processes}, "
                f"batch_size={config.train_batch_size}, GA={config.gradient_accumulation_steps} → "
                f"num_update_steps_per_epoch={num_update_steps_per_epoch}")
    if not config.max_train_steps or config.max_train_steps <= 0:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    config.num_train_epochs = math.ceil(
        config.max_train_steps / num_update_steps_per_epoch
    )
    total_batch_size = (
        config.train_batch_size
        * accelerator.num_processes
        * config.gradient_accumulation_steps
    )

    logger.info("***** Running DPO Training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs   = {config.num_train_epochs}")
    logger.info(f"  Batch/GPU    = {config.train_batch_size}")
    logger.info(f"  Total batch  = {total_batch_size}")
    logger.info(f"  Grad accum   = {config.gradient_accumulation_steps}")
    logger.info(f"  Total steps  = {config.max_train_steps}")
    logger.info(f"  DPO beta     = {config.get('dpo_beta', 2000.0)}")
    logger.info(f"  SFT weight   = {config.get('dpo_sft_weight', 0.0)}")

    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    # Restore optimizer / scheduler from checkpoint
    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        acc_state = os.path.join(resume_checkpoint_path, "accelerator")
        if os.path.exists(acc_state):
            try:
                orig_load = torch.load
                def _patched_load(*args, **kwargs):
                    kwargs["weights_only"] = False
                    return orig_load(*args, **kwargs)
                torch.load = _patched_load
                try:
                    accelerator.load_state(acc_state)
                    ckpt_name = os.path.basename(resume_checkpoint_path.rstrip("/"))
                    if ckpt_name.startswith("checkpoint-"):
                        global_step = int(ckpt_name.split("-")[1])
                        initial_global_step = global_step
                        first_epoch = global_step // num_update_steps_per_epoch
                        logger.info(f"[DPO] Resuming from step {global_step}")
                finally:
                    torch.load = orig_load
            except Exception as e:
                logger.error(f"[DPO] Failed to load accelerator state: {e}")

    # Load EMA weights from checkpoint if available
    ema_target_device = accelerator.device if ema_on_gpu else "cpu"
    if ema_transformer is not None and resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        ema_ckpt_path = os.path.join(resume_checkpoint_path, "ema_weights.pt")
        if os.path.exists(ema_ckpt_path):
            try:
                ema_state_dict = torch.load(ema_ckpt_path, map_location="cpu", weights_only=False)
                ema_transformer.load_state_dict(ema_state_dict)
                ema_transformer.shadow_params = [p.to(ema_target_device).float() for p in ema_transformer.shadow_params]
                logger.info(f"[DPO] Loaded EMA weights from checkpoint: {ema_ckpt_path}")
            except Exception as e:
                logger.warning(f"[DPO] Failed to load EMA weights: {e}, using fresh EMA")
        else:
            logger.info("[DPO] No EMA checkpoint found, using fresh EMA from transformer weights")

    progress_bar = tqdm(
        range(config.max_train_steps),
        initial=initial_global_step,
        desc="DPO Steps",
        disable=not accelerator.is_local_main_process,
    )

    # ──────────────────────────────────────────────────────
    # Get training iteration function
    # ──────────────────────────────────────────────────────
    iter_func_name = config.get("train_iteration_func", "Flux2Klein_dpo_train_iteration")
    iter_func = TRAIN_ITERATION_FUNCS.get(iter_func_name)
    if iter_func is None:
        raise ValueError(f"Iteration func '{iter_func_name}' not in registry.")
    logger.info(f"[DPO] Using iteration func: {iter_func_name}")

    # ──────────────────────────────────────────────────────
    # WandB
    # ──────────────────────────────────────────────────────
    if accelerator.is_main_process and config.report_to == "wandb":
        run_name = (
            config.get("tracker_run_name", None)
            or config.get("run_name", f"dpo_{model_type}")
        )
        accelerator.init_trackers(
            project_name=config.get("wandb_project", "SciForma-MDPO"),
            config=vars(config),
            init_kwargs={"wandb": {
                "name": run_name,
                "resume": "allow" if resume_checkpoint_path else None,
            }},
        )

    # ──────────────────────────────────────────────────────
    # Validation setup
    # ──────────────────────────────────────────────────────
    validation_steps = config.get('validation_steps', 0)  # 0 = disabled
    validation_func = None
    if validation_steps > 0:
        val_func_name = config.get(
            'validation_func',
            'Flux2Klein_fulltune_validation_func_parquet',
        )
        validation_func = VALIDATION_FUNCS.get(val_func_name)
        if validation_func is None:
            logger.warning(f"[DPO] Validation func '{val_func_name}' not found, disabling validation.")
            validation_steps = 0
        else:
            logger.info(f"[DPO] Validation every {validation_steps} steps using {val_func_name}")

    # ──────────────────────────────────────────────────────
    # Loss tracker
    # ──────────────────────────────────────────────────────
    loss_tracker = DPOLossTracker(window=100)

    # ──────────────────────────────────────────────────────
    # Main Training Loop
    # ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("Starting DPO Training Loop")
    logger.info("=" * 70)

    # Run validation at step 0 (before any training) so WandB shows the base model output.
    if validation_func is not None and validation_steps > 0 and initial_global_step == 0:
        if accelerator.is_main_process:
            if text_encoder is None:
                logger.info("[DPO] Lazy-loading text_encoder for step-0 validation...")
                text_encoder = model_factory.load_text_encoder(skip_load=False)
            text_encoder.to(device=accelerator.device, dtype=weight_dtype)
            vae.to(device=accelerator.device, dtype=weight_dtype)
            validation_func(
                vae=vae,
                transformer=transformer,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                scheduler=noise_scheduler,
                accelerator=accelerator,
                args=config,
                global_step=0,
            )
            vae.to(device="cpu")
            text_encoder.to(device="cpu")
            gc.collect()
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    for epoch in range(first_epoch, config.num_train_epochs):
        transformer.train()
        if ref_transformer is not None:
            ref_transformer.eval()   # reference always in eval mode (full fine-tune mode)

        # set_epoch for bucket samplers
        if hasattr(train_dataloader, "batch_sampler"):
            bs = train_dataloader.batch_sampler
            if hasattr(bs, "set_epoch"):
                bs.set_epoch(epoch)
            elif hasattr(bs, "sampler") and hasattr(bs.sampler, "set_epoch"):
                bs.sampler.set_epoch(epoch)

        # Skip already-processed batches in resumed epoch
        if epoch == first_epoch and initial_global_step > 0:
            resume_step = initial_global_step % num_update_steps_per_epoch
            if resume_step > 0:
                n_skip = resume_step * config.gradient_accumulation_steps
                logger.info(f"[DPO] Skipping {n_skip} batches to resume at step {initial_global_step}")
                active_dl = accelerator.skip_first_batches(train_dataloader, n_skip)
            else:
                active_dl = train_dataloader
        else:
            active_dl = train_dataloader

        # Accumulators for metrics across GA micro-steps
        _ga_metrics_accum = {}   # key → running sum (tensor)
        _ga_loss_accum = 0.0     # running sum of loss scalars
        _ga_count = 0            # number of micro-steps accumulated

        for step, batch in enumerate(active_dl):
            with accelerator.accumulate(transformer):
                # Move reference to GPU if needed.
                # LoRA mode: _ref=None → iteration func uses disable_adapter() instead.
                if use_lora:
                    _ref = None
                elif ref_on_cpu:
                    _ref = ref_transformer.to(device=accelerator.device, dtype=weight_dtype)
                else:
                    _ref = ref_transformer

                loss, metrics = iter_func(
                    batch=batch,
                    vae=vae,
                    noise_scheduler_copy=noise_scheduler_copy,
                    transformer=transformer,
                    ref_transformer=_ref,
                    config=config,
                    accelerator=accelerator,
                    global_step=global_step,
                    weight_dtype=weight_dtype,
                )

                # Move reference back to CPU if applicable (full fine-tune only).
                if ref_on_cpu and _ref is not None:
                    _ref.cpu()
                    del _ref

                # Accumulate metrics across GA micro-steps for accurate logging
                with torch.no_grad():
                    _ga_loss_accum += loss.detach().item()
                    _ga_count += 1
                    for k, v in metrics.items():
                        if isinstance(v, torch.Tensor):
                            if k not in _ga_metrics_accum:
                                _ga_metrics_accum[k] = v.detach().clone()
                            else:
                                _ga_metrics_accum[k] += v.detach()
                        else:
                            _ga_metrics_accum[k] = _ga_metrics_accum.get(k, 0.0) + float(v)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    # Capture grad norm BEFORE clipping (ZeRO-aware, all-reduced across GPUs)
                    _grad_norm = accelerator.clip_grad_norm_(
                        transformer.parameters(), config.max_grad_norm
                    )
                    _grad_norm_val = float(_grad_norm) if hasattr(_grad_norm, '__float__') else 0.0
                else:
                    _grad_norm_val = 0.0

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # ── Post-step ──────────────────────────────────────────────────
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Update EMA model
                ema_steps = config.get('ema_steps', 100)
                if ema_transformer is not None and global_step % ema_steps == 0:
                    ema_transformer.optimization_step += 1
                    decay = ema_transformer.get_decay(ema_transformer.optimization_step)
                    ema_transformer.cur_decay_value = decay
                    one_minus_decay = 1 - decay

                    _is_zero3 = (
                        accelerator.distributed_type == DistributedType.DEEPSPEED and
                        hasattr(accelerator.state, 'deepspeed_plugin') and
                        getattr(accelerator.state.deepspeed_plugin, 'zero_stage', 0) == 3
                    )

                    with torch.no_grad():
                        if _is_zero3:
                            import deepspeed
                            for s_param, param in zip(ema_transformer.shadow_params, transformer.parameters()):
                                with deepspeed.zero.GatheredParameters(param):
                                    param_f32 = param.data.detach().to(ema_target_device).float()
                                    if param.requires_grad:
                                        s_param.sub_(one_minus_decay * (s_param - param_f32))
                                    else:
                                        s_param.copy_(param_f32)
                                    del param_f32
                        elif ema_on_gpu:
                            for s_param, param in zip(ema_transformer.shadow_params, transformer.parameters()):
                                param_f32 = param.data.detach().float()
                                if param.requires_grad:
                                    s_param.sub_(one_minus_decay * (s_param - param_f32))
                                else:
                                    s_param.copy_(param_f32)
                                del param_f32
                        else:
                            for s_param, param in zip(ema_transformer.shadow_params, transformer.parameters()):
                                param_cpu = param.data.detach().cpu().float()
                                if param.requires_grad:
                                    s_param.sub_(one_minus_decay * (s_param - param_cpu))
                                else:
                                    s_param.copy_(param_cpu)
                                del param_cpu

                    if global_step % (ema_steps * 10) == 0:
                        logger.info(f"[DPO] EMA updated on {'GPU' if ema_on_gpu else 'CPU'} at step {global_step}, decay={decay:.6f}")

                # Logging — use GA-accumulated metrics (averaged over all micro-steps)
                log_steps = config.get("logging_steps", config.get("log_steps", 10))
                if global_step % log_steps == 0:
                    # Average accumulated metrics over GA micro-steps
                    ga_n = max(_ga_count, 1)
                    ga_avg_metrics = {}
                    for k, v in _ga_metrics_accum.items():
                        if isinstance(v, torch.Tensor):
                            ga_avg_metrics[k] = v / ga_n
                        else:
                            ga_avg_metrics[k] = v / ga_n
                    ga_avg_loss = _ga_loss_accum / ga_n

                    # ── Gather metrics across all ranks (ZeRO / DDP) ─────────────────
                    # Each rank has GA-averaged metrics; gather across GPUs.
                    def _gather_scalar(t) -> float:
                        """All-gather a 0-dim or scalar tensor, return mean as float."""
                        if isinstance(t, (int, float)):
                            t = torch.tensor(t, device=accelerator.device, dtype=torch.float32)
                        v = t.detach().to(device=accelerator.device).float().reshape(1)
                        gathered = accelerator.gather(v)   # -> (num_procs,)
                        return gathered.float().mean().item()

                    loss_val = _gather_scalar(torch.tensor(ga_avg_loss))
                    metrics_cpu = {
                        k: _gather_scalar(v) if isinstance(v, torch.Tensor) else _gather_scalar(v)
                        for k, v in ga_avg_metrics.items()
                    }

                    # ── Cross-GPU logits_std ──────────────────────────────────────────
                    # Each GPU has GA-averaged logits_mean (= mean logit over GA micro-steps).
                    # Gather all per-GPU logits → compute std across GPUs+GA.
                    # This is the meaningful "how spread out are per-sample logits".
                    _logit_v = ga_avg_metrics.get("logits_mean", torch.zeros(1))
                    if isinstance(_logit_v, torch.Tensor):
                        _logit_v = _logit_v.detach().to(accelerator.device).float().reshape(1)
                    else:
                        _logit_v = torch.tensor(_logit_v, device=accelerator.device, dtype=torch.float32).reshape(1)
                    _logits_all = accelerator.gather(_logit_v).float()  # (num_gpus,)
                    _cross_gpu_logits_std = _logits_all.std().item() if _logits_all.numel() > 1 else 0.0

                    # Grad norm (already all-reduced by DeepSpeed, valid on all ranks)
                    metrics_cpu["grad_norm"] = _grad_norm_val
                    loss_tracker.update(metrics_cpu)
                    stats = loss_tracker.get_stats()

                    logs = {
                        "loss":            loss_val,
                        "lr":              lr_scheduler.get_last_lr()[0],
                        "step":            global_step,
                        "epoch":           global_step / max(1, num_update_steps_per_epoch),
                        # VisionFlow-aligned metrics
                        "raw_model_loss":  metrics_cpu.get("raw_model_loss",
                                               0.5 * (metrics_cpu.get("loss_policy_w", 0) +
                                                       metrics_cpu.get("loss_policy_l", 0))),
                        "ref_loss":        metrics_cpu.get("ref_loss",
                                               0.5 * (metrics_cpu.get("loss_ref_w", 0) +
                                                       metrics_cpu.get("loss_ref_l", 0))),
                        "implicit_acc":    metrics_cpu.get("implicit_acc",
                                               metrics_cpu.get("reward_acc", 0)),
                        "logits_mean":     metrics_cpu.get("logits_mean",
                                               metrics_cpu.get("implicit_margin", 0)),
                        "logits_std":      _cross_gpu_logits_std,
                        "logits_abs_mean": metrics_cpu.get("logits_abs_mean", 0),
                        "latent_diff_mean": metrics_cpu.get("latent_diff_mean", 0),
                        "grad_norm":        metrics_cpu.get("grad_norm", 0),
                    }
                    # Add remaining metrics from metrics_cpu without overwriting
                    # the explicitly-set keys above
                    _explicit_keys = set(logs.keys())
                    for k, v in metrics_cpu.items():
                        if k not in _explicit_keys:
                            logs[k] = v if isinstance(v, (int, float)) else (v.item() if hasattr(v, "item") else float(v))
                    progress_bar.set_postfix(
                        loss=f"{loss_val:.4f}",
                        implicit_acc=f"{metrics_cpu.get('implicit_acc', metrics_cpu.get('reward_acc', 0)):.3f}",
                        logits_abs=f"{metrics_cpu.get('logits_abs_mean', 0):.5f}",
                        lr=f"{logs['lr']:.2e}",
                    )
                    accelerator.log(logs, step=global_step)

                    if (config.get("verbose_logging", True)
                            and global_step % 100 == 0
                            and accelerator.is_main_process):
                        logger.info(
                            f"\n[Step {global_step}] "
                            f"dpo_loss={metrics_cpu.get('dpo_loss', 0):.4f}  "
                            f"implicit_acc={metrics_cpu.get('implicit_acc', metrics_cpu.get('reward_acc', 0)):.3f}  "
                            f"logits_mean={metrics_cpu.get('logits_mean', metrics_cpu.get('implicit_margin', 0)):.5f}  "
                            f"logits_abs={metrics_cpu.get('logits_abs_mean', 0):.5f}  "
                            f"latent_diff={metrics_cpu.get('latent_diff_mean', 0):.4f}"
                        )

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

                    if (accelerator.is_main_process
                            and config.checkpoints_total_limit is not None):
                        ckpts = sorted(
                            [d for d in os.listdir(config.model_output_dir)
                             if d.startswith("checkpoint")],
                            key=lambda x: int(x.split("-")[1]),
                        )
                        while len(ckpts) > config.checkpoints_total_limit:
                            remove = ckpts.pop(0)
                            remove_path = os.path.join(config.model_output_dir, remove)
                            logger.info(f"[DPO] Removing old checkpoint: {remove_path}")
                            for _ in range(3):
                                try:
                                    shutil.rmtree(remove_path, ignore_errors=True)
                                    if not os.path.exists(remove_path):
                                        break
                                    time.sleep(2)
                                except Exception as e:
                                    logger.warning(f"Failed to remove {remove_path}: {e}")
                                    time.sleep(5)

                # Reset GA metric accumulators for next optimizer step
                _ga_metrics_accum = {}
                _ga_loss_accum = 0.0
                _ga_count = 0

                # Validation (generate sample images)
                if (validation_func is not None
                        and validation_steps > 0
                        and global_step % validation_steps == 0):
                    if accelerator.is_main_process:
                        # Lazy-load text_encoder for inference if not available
                        if text_encoder is None:
                            logger.info("[DPO] Lazy-loading text_encoder for validation...")
                            text_encoder = model_factory.load_text_encoder(skip_load=False)
                        text_encoder.to(device=accelerator.device, dtype=weight_dtype)
                        vae.to(device=accelerator.device, dtype=weight_dtype)

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

                        vae.to(device="cpu")
                        text_encoder.to(device="cpu")
                        gc.collect()
                        torch.cuda.empty_cache()
                    accelerator.wait_for_everyone()

            if global_step >= config.max_train_steps:
                break

    # ──────────────────────────────────────────────────────
    # Final checkpoint
    # ──────────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    save_model_checkpoint(
        transformer=transformer,
        accelerator=accelerator,
        config=config,
        global_step=global_step,
        logger=logger,
        is_final=True,
        ema_transformer=ema_transformer,
    )
    accelerator.wait_for_everyone()
    accelerator.end_training()
    logger.info("DPO Training completed!")


if __name__ == "__main__":
    main()
