"""
Flux2Klein Full Fine-tuning Training Iteration Functions for SciForma

Full fine-tuning version - directly train transformer parameters without LoRA.
VAE and text encoder are offloaded and frozen.

Key features:
1. Uses img_ids and txt_ids for positional encoding (RoPE)
2. Supports gradient checkpointing for memory efficiency
3. Compatible with parquet dataset (pre-computed latents/embeddings)
4. Mixed precision training (BF16)
"""

from contextlib import nullcontext
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.training_utils import compute_loss_weighting_for_sd3

from sciforma.utils import (
    get_sigmas,
    compute_density_for_timestep_sampling,
)
from sciforma.registry import TRAIN_ITERATION_FUNCS


def _prepare_latent_image_ids(batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype, already_patchified: bool = False):
    """
    Prepare 4D position IDs for image latent tokens.
    Matches official Flux2Klein pipeline format: [T, H, W, L]
    
    Args:
        batch_size: Batch size
        height: Height of latent
        width: Width of latent
        device: Device to create tensor on
        dtype: Data type for the tensor
        already_patchified: If True, height/width are already after patchify (H//2, W//2).
                           If False, will divide by 2 internally.
    
    Returns:
        img_ids: Tensor of shape (batch_size, H*W, 4) for RoPE
                 Columns are [T=0, H_idx, W_idx, L=0]
    """
    # For Flux2, latent is packed with patch_size=2
    if already_patchified:
        latent_h = height
        latent_w = width
    else:
        latent_h = height // 2
        latent_w = width // 2
    
    # Use cartesian_prod like official implementation: [T, H, W, L]
    t = torch.arange(1, device=device)  # [0] - time dimension
    h = torch.arange(latent_h, device=device)
    w = torch.arange(latent_w, device=device)
    l = torch.arange(1, device=device)  # [0] - layer dimension
    
    # Create position IDs: (H*W, 4) with format [T, H, W, L]
    latent_ids = torch.cartesian_prod(t, h, w, l).to(dtype=dtype)
    
    # Expand to batch: (B, H*W, 4)
    latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)
    
    return latent_ids


def _prepare_text_ids(seq_len: int, batch_size: int, device: torch.device, dtype: torch.dtype):
    """
    Prepare 4D position IDs for text tokens.
    Matches official Flux2Klein pipeline format: [T, H, W, L]
    
    Args:
        seq_len: Sequence length of text tokens
        batch_size: Batch size
        device: Device to create tensor on
        dtype: Data type
    
    Returns:
        txt_ids: Tensor of shape (batch_size, seq_len, 4) for RoPE
                 Columns are [T=0, H=0, W=0, L=idx]
    """
    # Use cartesian_prod like official implementation: [T, H, W, L]
    t = torch.arange(1, device=device)  # [0]
    h = torch.arange(1, device=device)  # [0]
    w = torch.arange(1, device=device)  # [0]
    l = torch.arange(seq_len, device=device)  # [0, 1, 2, ..., seq_len-1]
    
    # Create position IDs: (seq_len, 4) with format [T, H, W, L]
    txt_ids = torch.cartesian_prod(t, h, w, l).to(dtype=dtype)
    
    # Expand to batch: (B, seq_len, 4)
    txt_ids = txt_ids.unsqueeze(0).expand(batch_size, -1, -1)
    
    return txt_ids


def _pack_latents_flux2(latents: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """
    Pack latents for Flux2 Transformer.
    
    Flux2 expects latents in shape (B, seq_len, C) where:
    - B is batch size
    - seq_len = (H // patch_size) * (W // patch_size)
    - C = original_channels * patch_size * patch_size
    
    Args:
        latents: Tensor of shape (B, C, H, W)
        patch_size: Patch size (default 2 for Flux2)
    
    Returns:
        packed: Tensor of shape (B, seq_len, C * patch_size^2)
    """
    B, C, H, W = latents.shape
    assert H % patch_size == 0 and W % patch_size == 0, \
        f"Height {H} and Width {W} must be divisible by patch_size {patch_size}"
    
    # Reshape to patches
    latents = latents.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    latents = latents.permute(0, 2, 4, 1, 3, 5)  # (B, H', W', C, p, p)
    latents = latents.reshape(B, (H // patch_size) * (W // patch_size), C * patch_size * patch_size)
    
    return latents


def _unpack_latents_flux2(
    latents: torch.Tensor, 
    latent_height: int, 
    latent_width: int, 
    patch_size: int = 2,
) -> torch.Tensor:
    """
    Unpack Flux2 Transformer output back to image latent format.
    
    Args:
        latents: Tensor of shape (B, seq_len, C * patch_size^2)
        latent_height: Height of latent (after VAE encoding, before packing)
        latent_width: Width of latent (after VAE encoding, before packing)
        patch_size: Patch size (default 2 for Flux2)
    
    Returns:
        unpacked: Tensor of shape (B, C, latent_height, latent_width)
    """
    B, seq_len, packed_channels = latents.shape
    
    # Calculate packed dimensions
    H_packed = latent_height // patch_size
    W_packed = latent_width // patch_size
    C = packed_channels // (patch_size * patch_size)
    
    assert seq_len == H_packed * W_packed, \
        f"Sequence length {seq_len} doesn't match packed dimensions {H_packed}x{W_packed}"
    
    # Reshape back
    latents = latents.reshape(B, H_packed, W_packed, C, patch_size, patch_size)
    latents = latents.permute(0, 3, 1, 4, 2, 5)  # (B, C, H', p, W', p)
    latents = latents.reshape(B, C, latent_height, latent_width)
    
    return latents


@TRAIN_ITERATION_FUNCS.register_module()
def Flux2Klein_fulltune_train_iteration(
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
    Full fine-tuning training iteration for Flux2Klein.
    
    This function directly trains the transformer without LoRA adapters.
    VAE and text encoder are assumed to be offloaded and frozen.
    
    Args:
        batch: Batch from parquet dataset containing:
            - latents: Pre-computed VAE latents (B, C, H, W)
            - text_embeds: Pre-computed text embeddings (B, seq_len, dim)
            - text_mask: Attention mask for text (B, seq_len)
            - text_ids: Position IDs for text (B, seq_len, 4) - optional
        vae: VAE model (used only for BatchNorm stats)
        noise_scheduler_copy: Noise scheduler for computing sigmas
        transformer: Flux2Klein transformer model (trainable)
        config: Training configuration
        accelerator: Accelerator for distributed training
        global_step: Current training step
        weight_dtype: Data type for weights (BF16/FP16/FP32)
    
    Returns:
        loss: Computed loss value
    """
    # Get pre-computed latents and embeddings
    latents = batch["latents"].to(accelerator.device, dtype=weight_dtype)
    prompt_embeds = batch["text_embeds"].to(accelerator.device, dtype=weight_dtype)
    pooled_prompt_embeds = None  # Flux2Klein doesn't use pooled embeddings
    text_mask = batch.get("text_mask", None)
    if text_mask is not None:
        text_mask = text_mask.to(accelerator.device)
    
    bsz, C, H, W = latents.shape
    
    # Step 1: Patchify latents (C, H, W) -> (C*4, H//2, W//2)
    model_input = latents.view(bsz, C, H//2, 2, W//2, 2)
    model_input = model_input.permute(0, 1, 3, 5, 2, 4)
    model_input = model_input.reshape(bsz, C*4, H//2, W//2)
    
    # Step 2: Apply VAE BatchNorm normalization
    # Get BatchNorm stats from VAE
    if hasattr(vae, 'bn'):
        bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device=accelerator.device, dtype=weight_dtype)
        bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.bn.eps).to(device=accelerator.device, dtype=weight_dtype)
        model_input = (model_input - bn_mean) / bn_std
    
    # Step 3: Pack latents for transformer: (B, C, H, W) -> (B, H*W, C)
    # Note: After patchify, dimensions are (B, C*4, H//2, W//2)
    packed_latents = _pack_latents_flux2(model_input, patch_size=1)  # Already patchified, so patch_size=1
    # Alternative: Simple flatten and transpose
    # packed_latents = model_input.flatten(2).transpose(1, 2)  # (B, H*W, C)
    
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
    timesteps = noise_scheduler_copy.timesteps[indices].to(device=accelerator.device)
    
    # Add noise to latents (flow matching formula)
    sigmas = get_sigmas(timesteps, device=accelerator.device, noise_scheduler_copy=noise_scheduler_copy, n_dim=packed_latents.ndim, dtype=weight_dtype)
    noisy_model_input = (1.0 - sigmas) * packed_latents + sigmas * noise
    
    # Prepare position IDs for image and text
    # Get transformer dtype (handle DDP)
    transformer_dtype = transformer.module.dtype if hasattr(transformer, 'module') else transformer.dtype
    
    img_ids = _prepare_latent_image_ids(
        batch_size=bsz,
        height=H // 2,  # Already patchified height
        width=W // 2,   # Already patchified width
        device=accelerator.device,
        dtype=transformer_dtype,
        already_patchified=True
    )
    
    # Use pre-computed text_ids if available, otherwise generate
    if "text_ids" in batch and batch["text_ids"] is not None:
        txt_ids = batch["text_ids"].to(accelerator.device, dtype=transformer_dtype)
    else:
        seq_len = prompt_embeds.shape[1]
        txt_ids = _prepare_text_ids(
            seq_len=seq_len,
            batch_size=bsz,
            device=accelerator.device,
            dtype=transformer_dtype
        )
    
    # guidance_scale for Flux2Klein (typically 3.5 for guidance-distilled model)
    _gs = getattr(config, 'guidance_scale', None) or config.validation_guidance_scale
    guidance_scale = torch.tensor([_gs], device=accelerator.device, dtype=weight_dtype).expand(bsz)
    
    # Forward pass through transformer
    # Note: Flux2Klein does NOT use pooled_projections
    model_pred = transformer(
        hidden_states=noisy_model_input,
        timestep=timesteps / 1000.0,  # Normalize timesteps to [0, 1]
        guidance=guidance_scale,
        encoder_hidden_states=prompt_embeds,
        img_ids=img_ids,
        txt_ids=txt_ids,
        return_dict=False,
    )[0]
    
    # Compute loss weights
    weighting = compute_loss_weighting_for_sd3(weighting_scheme=config.weighting_scheme, sigmas=sigmas)
    
    # Compute loss (flow matching: predict velocity v = noise - latent)
    # IMPORTANT: target is noise - latent, NOT latent - noise!
    target = noise - packed_latents
    loss = torch.mean(
        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(bsz, -1),
        dim=1,
    )
    loss = loss.mean()
    
    # # Debug logging every 10 steps
    # if global_step % 10 == 0 and accelerator.is_main_process:
    #     print(f"\n[Step {global_step}] Training Debug Info:")
    #     print(f"  Loss: {loss.item():.6f}")
    #     print(f"  Latent shape: {latents.shape}, Packed shape: {packed_latents.shape}")
    #     print(f"  Latent mean: {packed_latents.mean().item():.4f}, std: {packed_latents.std().item():.4f}")
    #     print(f"  Noise mean: {noise.mean().item():.4f}, std: {noise.std().item():.4f}")
    #     print(f"  Target mean: {target.mean().item():.4f}, std: {target.std().item():.4f}")
    #     print(f"  Model pred mean: {model_pred.mean().item():.4f}, std: {model_pred.std().item():.4f}")
    #     # sigmas might be (bsz, 1, 1) so need to flatten first
    #     sigmas_flat = sigmas.flatten()
    #     ts_flat = timesteps.flatten() if timesteps.dim() > 0 else timesteps.unsqueeze(0)
    #     print(f"  Sigmas: {sigmas_flat[:min(4, len(sigmas_flat))].tolist()}... (timesteps: {ts_flat[:min(4, len(ts_flat))].tolist()})")
    
    return loss


@TRAIN_ITERATION_FUNCS.register_module()
def Flux2Klein_fulltune_validation_iteration(
    batch,
    vae,
    noise_scheduler,
    transformer,
    config,
    accelerator,
    global_step,
    weight_dtype,
):
    """
    Validation iteration for Flux2Klein full fine-tuning.
    
    Generates images from validation prompts to monitor training progress.
    """
    # Similar to training iteration but runs full denoising process
    # This is a placeholder - full implementation would require pipeline integration
    raise NotImplementedError("Validation iteration for full fine-tuning not yet implemented")
