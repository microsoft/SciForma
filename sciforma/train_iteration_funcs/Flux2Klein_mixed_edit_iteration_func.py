"""
Flux2Klein Mixed Edit + Generation Training Iteration Function (V2 - Fixed)

Supports two batch modes (batches are guaranteed same-type by sampler):
1. "gen"  - Pure generation: standard flow matching on single image
2. "edit" - Pure editing: source concatenated as conditioning tokens

== Editing Approach (matches official train_dreambooth_lora_flux2_klein_img2img.py) ==
- Source image is patchified, packed, and concatenated along SEQUENCE dim
  with the noised target latents
- Source gets _prepare_image_ids() (different T-coordinate) 
  while target gets _prepare_latent_ids() (T=0)
- After transformer forward, only the target portion of output is kept
- Loss target is  noise - target_packed  (SAME as generation!)
- The model learns standard flow matching while attending to source tokens

== Generation Approach ==
- Same as Flux2Klein_fulltune_train_iteration (V1)
"""

from contextlib import nullcontext
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from diffusers.training_utils import compute_loss_weighting_for_sd3

from sciforma.utils import (
    get_sigmas,
    compute_density_for_timestep_sampling,
)
from sciforma.registry import TRAIN_ITERATION_FUNCS


# ============================================================================
# Shared Helpers
# ============================================================================

def _prepare_latent_image_ids(batch_size, height, width, device, dtype, already_patchified=False):
    """
    Prepare 4D position IDs for TARGET latent tokens [T, H, W, L].
    T=0 for target latents (standard).
    """
    if already_patchified:
        latent_h, latent_w = height, width
    else:
        latent_h, latent_w = height // 2, width // 2
    
    t = torch.arange(1, device=device)   # [0]
    h = torch.arange(latent_h, device=device)
    w = torch.arange(latent_w, device=device)
    l = torch.arange(1, device=device)   # [0]
    
    latent_ids = torch.cartesian_prod(t, h, w, l).to(dtype=dtype)
    return latent_ids.unsqueeze(0).expand(batch_size, -1, -1)


def _prepare_cond_image_ids(cond_latents_4d, device, dtype, scale=10):
    """
    Prepare 4D position IDs for SOURCE / conditioning image tokens [T, H, W, L].
    
    Matches official Flux2KleinPipeline._prepare_image_ids():
    - Each conditioning image gets T = scale * (i + 1), e.g. T=10 for the first image.
    - This distinguishes conditioning tokens from target tokens (T=0) in RoPE.
    
    Args:
        cond_latents_4d: (B, C, H, W) - patchified conditioning latents
        device, dtype: tensor device/dtype
        scale: T-coordinate separation (default 10, matching official code)
    
    Returns:
        cond_ids: (B, H*W, 4)
    """
    bsz, C, H, W = cond_latents_4d.shape
    
    # For each sample in the batch, create IDs with T = scale (first conditioning image)
    t_coord = torch.tensor([scale], device=device)  # T=10
    h = torch.arange(H, device=device)
    w = torch.arange(W, device=device)
    l = torch.arange(1, device=device)  # [0]
    
    cond_ids = torch.cartesian_prod(t_coord, h, w, l).to(dtype=dtype)
    return cond_ids.unsqueeze(0).expand(bsz, -1, -1)


def _prepare_text_ids(seq_len, batch_size, device, dtype):
    """Prepare 4D position IDs for text tokens [T, H, W, L]."""
    t = torch.arange(1, device=device)   # [0]
    h = torch.arange(1, device=device)   # [0]
    w = torch.arange(1, device=device)   # [0]
    l = torch.arange(seq_len, device=device)
    
    txt_ids = torch.cartesian_prod(t, h, w, l).to(dtype=dtype)
    return txt_ids.unsqueeze(0).expand(batch_size, -1, -1)


def _pack_latents_flux2(latents, patch_size=2):
    """Pack latents: (B, C, H, W) -> (B, H*W/patch^2, C*patch^2)."""
    B, C, H, W = latents.shape
    latents = latents.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(B, (H // patch_size) * (W // patch_size), C * patch_size * patch_size)
    return latents


def _patchify_and_normalize(latents, vae, device, weight_dtype):
    """
    Patchify + BatchNorm normalization.
    
    Input:  latents (B, C, H, W)  - raw VAE latents
    Output: packed (B, SeqLen, C*4), patchified_4d (B, C*4, H//2, W//2)
    
    Returns both packed sequence and 4D form (needed for position ID prep).
    """
    bsz, C, H, W = latents.shape
    
    # Patchify: (B, C, H, W) -> (B, C*4, H//2, W//2)
    model_input = latents.view(bsz, C, H // 2, 2, W // 2, 2)
    model_input = model_input.permute(0, 1, 3, 5, 2, 4)
    model_input = model_input.reshape(bsz, C * 4, H // 2, W // 2)
    
    # Apply VAE BatchNorm
    if hasattr(vae, 'bn'):
        bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device=device, dtype=weight_dtype)
        bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.bn.eps).to(device=device, dtype=weight_dtype)
        model_input = (model_input - bn_mean) / bn_std
    
    # Pack: (B, C*4, H//2, W//2) -> (B, (H//2)*(W//2), C*4)
    packed = _pack_latents_flux2(model_input, patch_size=1)
    return packed, model_input  # return both packed and 4D form


def _compute_flow_matching_loss(
    packed_latents, prompt_embeds, text_mask,
    transformer, vae, noise_scheduler_copy, config, accelerator, 
    global_step, weight_dtype, latent_h, latent_w, bsz,
):
    """
    Compute standard generation flow matching loss.
    Same logic as Flux2Klein_fulltune_train_iteration.
    
    Returns: scalar loss tensor
    """
    device = accelerator.device
    
    # Sample noise
    noise = torch.randn_like(packed_latents)
    
    # Sample timesteps
    u = compute_density_for_timestep_sampling(
        weighting_scheme=config.weighting_scheme,
        batch_size=bsz,
        logit_mean=config.logit_mean,
        logit_std=config.logit_std,
        mode_scale=config.mode_scale,
    )
    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
    timesteps = noise_scheduler_copy.timesteps[indices].to(device=device)
    
    sigmas = get_sigmas(timesteps, device=device, noise_scheduler_copy=noise_scheduler_copy,
                        n_dim=packed_latents.ndim, dtype=weight_dtype)
    noisy_model_input = (1.0 - sigmas) * packed_latents + sigmas * noise
    
    # Position IDs
    transformer_dtype = transformer.module.dtype if hasattr(transformer, 'module') else transformer.dtype
    
    img_ids = _prepare_latent_image_ids(
        batch_size=bsz, height=latent_h, width=latent_w,
        device=device, dtype=transformer_dtype, already_patchified=True,
    )
    
    seq_len = prompt_embeds.shape[1]
    txt_ids = _prepare_text_ids(
        seq_len=seq_len, batch_size=bsz,
        device=device, dtype=transformer_dtype,
    )
    
    # Guidance
    _gs = getattr(config, 'guidance_scale', None) or config.validation_guidance_scale
    guidance_scale = torch.tensor(
        [_gs], device=device, dtype=weight_dtype
    ).expand(bsz)
    
    # Forward pass
    model_pred = transformer(
        hidden_states=noisy_model_input,
        timestep=timesteps / 1000.0,
        guidance=guidance_scale,
        encoder_hidden_states=prompt_embeds,
        img_ids=img_ids,
        txt_ids=txt_ids,
        return_dict=False,
    )[0]
    
    # Loss: target = noise - clean_latents
    weighting = compute_loss_weighting_for_sd3(
        weighting_scheme=config.weighting_scheme, sigmas=sigmas
    )
    target = noise - packed_latents
    loss = torch.mean(
        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(bsz, -1),
        dim=1,
    )
    return loss.mean()


def _compute_edit_flow_matching_loss(
    target_packed, target_4d, source_packed, source_4d,
    prompt_embeds, text_mask,
    transformer, vae, noise_scheduler_copy, config, accelerator,
    global_step, weight_dtype, latent_h, latent_w, bsz,
    edit_mask=None,
):
    """
    Compute editing flow matching loss following official Flux2Klein img2img approach.
    
    Official method (from train_dreambooth_lora_flux2_klein_img2img.py):
    1. Add random noise to TARGET (not source): noisy = (1-sigma)*target + sigma*noise
    2. Pack both noisy target and clean source
    3. Concatenate source tokens AFTER target tokens along sequence dim
    4. Concatenate their position IDs (target: T=0, source: T=scale)
    5. Forward pass → get output for ALL tokens
    6. PRUNE: keep only the first N tokens (target portion)
    7. Loss target = noise - target (same as generation!)
    
    Returns: scalar loss tensor
    """
    device = accelerator.device
    
    # Step 1: Sample noise for TARGET (random, NOT source!)
    noise = torch.randn_like(target_packed)
    
    # Sample timesteps
    u = compute_density_for_timestep_sampling(
        weighting_scheme=config.weighting_scheme,
        batch_size=bsz,
        logit_mean=config.logit_mean,
        logit_std=config.logit_std,
        mode_scale=config.mode_scale,
    )
    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
    timesteps = noise_scheduler_copy.timesteps[indices].to(device=device)
    
    sigmas = get_sigmas(timesteps, device=device, noise_scheduler_copy=noise_scheduler_copy,
                        n_dim=target_packed.ndim, dtype=weight_dtype)
    
    # Step 2: Standard flow matching noise on TARGET
    noisy_target = (1.0 - sigmas) * target_packed + sigmas * noise
    
    # Step 3: Concatenate [noisy_target, clean_source] along sequence dim
    # noisy_target: (B, target_seq, C)
    # source_packed: (B, source_seq, C)
    packed_combined = torch.cat([noisy_target, source_packed], dim=1)
    orig_target_seq_len = noisy_target.shape[1]
    
    # Step 4: Position IDs
    transformer_dtype = transformer.module.dtype if hasattr(transformer, 'module') else transformer.dtype
    
    # Target position IDs: T=0 (standard latent IDs)
    target_ids = _prepare_latent_image_ids(
        batch_size=bsz, height=latent_h, width=latent_w,
        device=device, dtype=transformer_dtype, already_patchified=True,
    )
    
    # Source/conditioning position IDs: T=scale (distinct from target)
    # source_4d is (B, C*4, H//2, W//2) after patchify
    cond_ids = _prepare_cond_image_ids(
        source_4d, device=device, dtype=transformer_dtype, scale=10,
    )
    
    # Concatenate position IDs
    combined_img_ids = torch.cat([target_ids, cond_ids], dim=1)
    
    seq_len = prompt_embeds.shape[1]
    txt_ids = _prepare_text_ids(
        seq_len=seq_len, batch_size=bsz,
        device=device, dtype=transformer_dtype,
    )
    
    # Guidance
    _gs = getattr(config, 'guidance_scale', None) or config.validation_guidance_scale
    guidance_scale = torch.tensor(
        [_gs], device=device, dtype=weight_dtype
    ).expand(bsz)
    
    # Step 5: Forward pass with concatenated hidden states
    model_pred = transformer(
        hidden_states=packed_combined,
        timestep=timesteps / 1000.0,
        guidance=guidance_scale,
        encoder_hidden_states=prompt_embeds,
        img_ids=combined_img_ids,
        txt_ids=txt_ids,
        return_dict=False,
    )[0]
    
    # Step 6: Prune - keep only target portion of output
    model_pred = model_pred[:, :orig_target_seq_len, :]
    
    # Step 7: Loss  (SAME target as generation: noise - clean)
    weighting = compute_loss_weighting_for_sd3(
        weighting_scheme=config.weighting_scheme, sigmas=sigmas
    )
    target_velocity = noise - target_packed
    
    # Step 8: Edit region loss weighting
    # Controlled by config.edit_loss_mode (explicit) or inferred from legacy params:
    #
    #   "uniform"  — Standard MSE, no region weighting (original flow-matching loss)
    #
    #   "weighted" — Ground truth bbox mask × multiplicative weight (edit_region_weight).
    #               edit tokens get weight W, non-edit get weight 1, then normalize.
    #               Falls back to adaptive L2 threshold if edit_mask unavailable.
    #               Config: edit_region_weight (float, e.g. 10.0)
    #
    #   "balanced" — Area-normalized beta-weighted loss.
    #               L = (1-β) * mean(L_preserve) + β * mean(L_edit)
    #               Area-invariant: 1% or 50% edit area gives the same gradient ratio.
    #               Config: edit_loss_ratio (float, e.g. 0.5)
    
    edit_loss_mode = getattr(config, 'edit_loss_mode', None)
    
    # Legacy fallback: infer mode from old config params for backward compatibility
    if edit_loss_mode is None:
        if getattr(config, 'edit_loss_ratio', None) is not None and edit_mask is not None:
            edit_loss_mode = "balanced"
        elif getattr(config, 'edit_region_weight', 1.0) > 1.0:
            edit_loss_mode = "weighted"
        else:
            edit_loss_mode = "uniform"
    
    if edit_loss_mode == "balanced":
        # ===== Balanced Region Loss (ground truth bbox mask) =====
        beta = float(getattr(config, 'edit_loss_ratio', 0.5))
        
        per_token_loss = (model_pred.float() - target_velocity.float()) ** 2  # (B, SeqLen, C)
        per_token_loss = per_token_loss.mean(dim=-1)  # (B, SeqLen)
        weighted_loss = weighting.float().view(bsz, 1) * per_token_loss  # (B, SeqLen)
        
        # Ground truth mask from bbox
        edit_mask_f = edit_mask.float().to(weighted_loss.device)  # (B, SeqLen)
        non_edit_mask = 1.0 - edit_mask_f
        
        n_edit = edit_mask_f.sum(dim=1).clamp(min=1)       # (B,)
        n_non_edit = non_edit_mask.sum(dim=1).clamp(min=1)  # (B,)
        
        edit_loss = (weighted_loss * edit_mask_f).sum(dim=1) / n_edit          # (B,)
        preserve_loss = (weighted_loss * non_edit_mask).sum(dim=1) / n_non_edit  # (B,)
        
        loss = (1.0 - beta) * preserve_loss + beta * edit_loss  # (B,)
        
        if global_step % 50 == 0 and accelerator.is_main_process:
            avg_coverage = edit_mask_f.mean().item() * 100
            print(f"  [BalancedLoss] β={beta:.2f}, mask_coverage={avg_coverage:.1f}%, "
                  f"edit_loss={edit_loss.mean().item():.4f}, "
                  f"preserve_loss={preserve_loss.mean().item():.4f}, "
                  f"combined={loss.mean().item():.4f}")
    
    elif edit_loss_mode == "weighted":
        # ===== Multiplicative weighting (ground truth bbox mask required) =====
        assert edit_mask is not None, \
            "edit_loss_mode='weighted' requires edit_mask from bbox. " \
            "Ensure parquet has edit_bboxes column."
        edit_region_weight = getattr(config, 'edit_region_weight', 10.0)
        
        mask_f = edit_mask.float().to(device)  # (B, SeqLen)
        
        with torch.no_grad():
            spatial_weight = 1.0 + (edit_region_weight - 1.0) * mask_f  # (B, SeqLen)
            spatial_weight = spatial_weight / spatial_weight.mean(dim=1, keepdim=True)
            spatial_weight = spatial_weight.unsqueeze(-1)  # (B, SeqLen, 1)
        
        per_token_loss = (model_pred.float() - target_velocity.float()) ** 2
        loss = torch.mean(
            (weighting.float() * spatial_weight * per_token_loss).reshape(bsz, -1),
            dim=1,
        )
        
        if global_step % 50 == 0 and accelerator.is_main_process:
            avg_coverage = mask_f.mean().item() * 100
            print(f"  [WeightedLoss] w={edit_region_weight:.0f}, "
                  f"coverage={avg_coverage:.1f}%, loss={loss.mean().item():.4f}")
    
    else:  # "uniform"
        # ===== Standard uniform MSE loss (original flow-matching) =====
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target_velocity.float()) ** 2).reshape(bsz, -1),
            dim=1,
        )
        
        if global_step % 50 == 0 and accelerator.is_main_process:
            print(f"  [UniformLoss] loss={loss.mean().item():.4f}")
    
    return loss.mean()


# ============================================================================
# Main Registered Iteration Function
# ============================================================================

@TRAIN_ITERATION_FUNCS.register_module()
def Flux2Klein_mixed_edit_train_iteration(
    batch,
    vae,
    noise_scheduler_copy,
    transformer,
    config,
    accelerator,
    global_step,
    weight_dtype,
):
    """
    Mixed generation + editing training iteration for Flux2Klein.
    
    Dispatches based on batch['batch_mode']:
    - "gen":  Standard flow matching (noise → image)
    - "edit": Editing with source concatenation (official Flux2Klein img2img approach)
    
    NOTE: "mixed" mode is supported but discouraged. Use the bucket sampler
    with (bucket_h, bucket_w, data_type) grouping to ensure homogeneous batches.
    
    Args:
        batch: Dict from ArXiVParquetDatasetV4.collate_fn
        vae: VAE model (used only for BatchNorm stats)
        noise_scheduler_copy: Noise scheduler for sigmas
        transformer: Flux2Klein transformer (trainable)
        config: Training config
        accelerator: Accelerator
        global_step: Current step
        weight_dtype: BF16/FP16/FP32
    
    Returns:
        loss: Combined loss value
    """
    batch_mode = batch.get("batch_mode", "gen")
    # Handle case where default_collate converts string to list (HF dataset format)
    if isinstance(batch_mode, (list, tuple)):
        batch_mode = batch_mode[0] if batch_mode else "gen"
    device = accelerator.device
    
    edit_loss_weight = getattr(config, 'edit_loss_weight', 1.0)
    gen_loss_weight = getattr(config, 'gen_loss_weight', 1.0)
    
    if batch_mode == "gen":
        return _process_gen_batch(batch, vae, noise_scheduler_copy, transformer,
                                  config, accelerator, global_step, weight_dtype)
    
    elif batch_mode == "edit":
        return _process_edit_batch(batch, vae, noise_scheduler_copy, transformer,
                                   config, accelerator, global_step, weight_dtype)
    
    elif batch_mode == "mixed":
        # Mixed batch fallback - process gen and edit sub-batches separately
        gen_indices = batch.get("gen_indices", [])
        edit_indices = batch.get("edit_indices", [])
        
        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        total_weight = 0.0
        
        if gen_indices and "latents" in batch:
            gen_sub_batch = {
                "latents": batch["latents"],
                "text_embeds": batch["text_embeds"][gen_indices],
                "text_mask": batch["text_mask"][gen_indices] if batch.get("text_mask") is not None else None,
                "bucket_size": batch["bucket_size"],
            }
            gen_loss = _process_gen_batch(gen_sub_batch, vae, noise_scheduler_copy,
                                          transformer, config, accelerator, global_step, weight_dtype)
            total_loss = total_loss + gen_loss_weight * gen_loss * len(gen_indices)
            total_weight += gen_loss_weight * len(gen_indices)
        
        if edit_indices and "source_latents" in batch:
            edit_sub_batch = {
                "source_latents": batch["source_latents"],
                "target_latents": batch["target_latents"],
                "text_embeds": batch["text_embeds"][edit_indices],
                "text_mask": batch["text_mask"][edit_indices] if batch.get("text_mask") is not None else None,
                "bucket_size": batch["bucket_size"],
            }
            edit_loss = _process_edit_batch(edit_sub_batch, vae, noise_scheduler_copy,
                                            transformer, config, accelerator, global_step, weight_dtype)
            total_loss = total_loss + edit_loss_weight * edit_loss * len(edit_indices)
            total_weight += edit_loss_weight * len(edit_indices)
        
        if total_weight > 0:
            total_loss = total_loss / total_weight
        
        return total_loss
    
    else:
        raise ValueError(f"Unknown batch_mode: {batch_mode}")


def _process_gen_batch(batch, vae, noise_scheduler_copy, transformer,
                       config, accelerator, global_step, weight_dtype):
    """Process a pure generation batch."""
    latents = batch["latents"].to(accelerator.device, dtype=weight_dtype)
    prompt_embeds = batch["text_embeds"].to(accelerator.device, dtype=weight_dtype)
    text_mask = batch.get("text_mask")
    if text_mask is not None:
        text_mask = text_mask.to(accelerator.device)
    
    bsz, C, H, W = latents.shape
    
    packed_latents, _ = _patchify_and_normalize(latents, vae, accelerator.device, weight_dtype)
    
    return _compute_flow_matching_loss(
        packed_latents, prompt_embeds, text_mask,
        transformer, vae, noise_scheduler_copy, config, accelerator,
        global_step, weight_dtype,
        latent_h=H // 2, latent_w=W // 2, bsz=bsz,
    )


def _process_edit_batch(batch, vae, noise_scheduler_copy, transformer,
                        config, accelerator, global_step, weight_dtype):
    """
    Process a pure editing batch using official concatenation approach.
    
    Source image tokens are concatenated with noised target tokens along
    the sequence dimension. The model attends to source via self-attention
    and learns standard flow matching on the target.
    """
    source_latents = batch["source_latents"].to(accelerator.device, dtype=weight_dtype)
    target_latents = batch["target_latents"].to(accelerator.device, dtype=weight_dtype)
    prompt_embeds = batch["text_embeds"].to(accelerator.device, dtype=weight_dtype)
    text_mask = batch.get("text_mask")
    if text_mask is not None:
        text_mask = text_mask.to(accelerator.device)
    
    # Ground truth edit mask from bbox (may be None if parquet lacks bbox data)
    edit_mask = batch.get("edit_mask")
    
    bsz, C, H, W = target_latents.shape
    
    # Patchify and normalize both source and target
    # Returns (packed_sequence, patchified_4d) for each
    target_packed, target_4d = _patchify_and_normalize(target_latents, vae, accelerator.device, weight_dtype)
    source_packed, source_4d = _patchify_and_normalize(source_latents, vae, accelerator.device, weight_dtype)
    
    return _compute_edit_flow_matching_loss(
        target_packed, target_4d, source_packed, source_4d,
        prompt_embeds, text_mask,
        transformer, vae, noise_scheduler_copy, config, accelerator,
        global_step, weight_dtype,
        latent_h=H // 2, latent_w=W // 2, bsz=bsz,
        edit_mask=edit_mask,
    )


# ============================================================================
# Validation Iteration (placeholder, same as fulltune)
# ============================================================================

@TRAIN_ITERATION_FUNCS.register_module()
def Flux2Klein_mixed_edit_validation_iteration(
    batch, vae, noise_scheduler, transformer, config,
    accelerator, global_step, weight_dtype,
):
    """Placeholder validation iteration for mixed training."""
    raise NotImplementedError("Use validation_func instead")
