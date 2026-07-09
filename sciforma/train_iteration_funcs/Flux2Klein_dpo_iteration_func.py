"""
Flux2Klein DPO (Diffusion DPO / D3PO) Training Iteration Function

Implements Diffusion DPO loss for flow-matching models (Flux2Klein).

Reference:
    Wallace et al., "Diffusion Model Alignment Using Direct Preference Optimization"
    (D3PO), NeurIPS 2024.

Algorithm (per batch):
    1. Sample the SAME timestep t and INDEPENDENT noises ε_w, ε_l for the
       winner and loser latents.
    2. Construct noisy latents:   x̂_w = (1-σ)*x_w + σ*ε_w
                                  x̂_l = (1-σ)*x_l + σ*ε_l
    3. Targets (flow-matching): v_w* = ε_w - x_w,  v_l* = ε_l - x_l
    4. Concatenate winner+loser along batch dim → run policy transformer once.
    5. Same for reference transformer (no_grad).
    6. Per-sample MSE losses:
         L_π_w = MSE(v_π(x̂_w), v_w*)   L_π_l = MSE(v_π(x̂_l), v_l*)
         L_ref_w = MSE(v_ref(x̂_w), v_w*) L_ref_l = MSE(v_ref(x̂_l), v_l*)
    7. Implicit reward margin:
         Δ = β * [(L_ref_w - L_π_w) - (L_ref_l - L_π_l)]
    8. DPO loss: -logsigmoid(Δ).mean()

Memory note:
    - Policy transformer is wrapped by accelerator (DDP / DeepSpeed).
    - Reference transformer is a plain nn.Module on accelerator.device (no DDP).
    - Reference forward is always inside torch.no_grad().
"""

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from diffusers.training_utils import compute_loss_weighting_for_sd3

from sciforma.utils import (
    get_sigmas,
    compute_density_for_timestep_sampling,
)
from sciforma.registry import TRAIN_ITERATION_FUNCS


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers (copied / adapted from fulltune iteration func)
# ──────────────────────────────────────────────────────────────────────────────

def _prepare_latent_image_ids(batch_size, height, width, device, dtype,
                               already_patchified=False):
    """Prepare [T, H, W, L] 4D position IDs for image latent tokens."""
    latent_h = height if already_patchified else height // 2
    latent_w = width  if already_patchified else width  // 2

    t = torch.arange(1, device=device)
    h = torch.arange(latent_h, device=device)
    w = torch.arange(latent_w, device=device)
    l = torch.arange(1, device=device)

    latent_ids = torch.cartesian_prod(t, h, w, l).to(dtype=dtype)          # (H*W, 4)
    return latent_ids.unsqueeze(0).expand(batch_size, -1, -1)              # (B, H*W, 4)


def _prepare_text_ids(seq_len, batch_size, device, dtype):
    """Prepare [T, H, W, L] 4D position IDs for text tokens."""
    t = torch.arange(1, device=device)
    h = torch.arange(1, device=device)
    w = torch.arange(1, device=device)
    l = torch.arange(seq_len, device=device)

    txt_ids = torch.cartesian_prod(t, h, w, l).to(dtype=dtype)             # (seq_len, 4)
    return txt_ids.unsqueeze(0).expand(batch_size, -1, -1)                 # (B, seq_len, 4)


def _pack_latents_flux2(latents: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """Pack (B, C, H, W) → (B, H/p * W/p, C*p²)."""
    B, C, H, W = latents.shape
    latents = latents.reshape(B, C, H // patch_size, patch_size,
                              W // patch_size, patch_size)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(B, (H // patch_size) * (W // patch_size),
                           C * patch_size * patch_size)


def _patchify_and_normalize(latents, vae, weight_dtype, device):
    """
    Convert raw VAE latents (B, C, H, W) to packed tokens (B, L, C') after
    Flux2Klein patchify + BatchNorm normalisation.

    Uses the canonical _pack_latents_flux2(patch_size=2) path, which matches
    the exact channel layout that vae.bn.running_mean/var were calibrated on
    during Flux2Klein pre-training.

    Returns:
        packed: (B, L, C')   where L = (H//2)*(W//2),  C' = C*4
    """
    bsz, C, H, W = latents.shape
    lat_h, lat_w = H // 2, W // 2

    # Step 1: canonical 2×2 spatial patchify → packed sequence  (B, L, C*4)
    #   _pack_latents_flux2(ps=2): (B,C,H,W) → reshape → permute(0,2,4,1,3,5)
    #   → (B, H/2, W/2, C, 2, 2) → reshape → (B, L, C*4).  Channel at each
    #   spatial position (h,w) is ordered (c, ph, pw) → c*4+ph*2+pw, which
    #   is exactly the order vae.bn stats were computed for.
    packed = _pack_latents_flux2(latents.to(device=device, dtype=weight_dtype),
                                 patch_size=2)   # (B, L, C*4)

    # Step 2: BatchNorm normalisation — operates on the (C*4) channel dim.
    #   Reshape packed → feature map (B, C*4, H//2, W//2), apply BN,
    #   then reshape back to sequence format.  This is identical to calling
    #   vae.bn in feature-map mode, but without relying on BN.training state.
    if hasattr(vae, 'bn'):
        # Reshape to spatial feature map for BN broadcast
        x = packed.view(bsz, lat_h, lat_w, C * 4).permute(0, 3, 1, 2)  # (B, C*4, h, w)

        bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device=device, dtype=weight_dtype)
        bn_eps  = getattr(vae.config, 'batch_norm_eps', vae.bn.eps)  # use config eps (matches inference)
        bn_std  = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + bn_eps).to(device=device, dtype=weight_dtype)
        x = (x - bn_mean) / bn_std

        # Reshape back to sequence (B, L, C*4)
        packed = x.permute(0, 2, 3, 1).reshape(bsz, lat_h * lat_w, C * 4)

    return packed


def _forward_transformer(
    transformer,
    noisy_hidden,
    timesteps,
    guidance,
    prompt_embeds,
    img_ids,
    txt_ids,
    no_grad: bool = False,
    disable_lora: bool = False,
):
    """
    Run transformer forward, optionally inside no_grad context.

    Args:
        no_grad:      If True, wrap forward in torch.no_grad().
        disable_lora: If True, temporarily disable PEFT LoRA adapters so the
                      forward uses the frozen base model (= reference policy).
    """
    no_grad_ctx = torch.no_grad() if no_grad else nullcontext()

    # Resolve the underlying PEFT model (handles DDP .module wrapper)
    _m = transformer.module if hasattr(transformer, 'module') else transformer

    if disable_lora and hasattr(_m, 'disable_adapter'):
        lora_ctx = _m.disable_adapter()
    else:
        lora_ctx = nullcontext()

    with no_grad_ctx:
        with lora_ctx:
            pred = transformer(
                hidden_states=noisy_hidden,
                timestep=timesteps / 1000.0,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                img_ids=img_ids,
                txt_ids=txt_ids,
                return_dict=False,
            )[0]
    return pred


def _per_sample_mse(pred, target, weighting):
    """
    Weighted MSE loss per sample in the batch.

    Args:
        pred:      (B, L, C)
        target:    (B, L, C)
        weighting: (B, 1, 1) loss weights from loss-weighting scheme

    Returns:
        loss_per_sample: (B,)
    """
    diff_sq = (pred.float() - target.float()) ** 2          # (B, L, C)
    weighted = weighting.float() * diff_sq                  # (B, L, C)
    return weighted.reshape(weighted.shape[0], -1).mean(dim=1)  # (B,)


# ──────────────────────────────────────────────────────────────────────────────
# DPO Iteration Function
# ──────────────────────────────────────────────────────────────────────────────

@TRAIN_ITERATION_FUNCS.register_module()
def Flux2Klein_dpo_train_iteration(
    batch,
    vae,
    noise_scheduler_copy,
    transformer,          # policy model  (trainable, wrapped by accelerator)
    ref_transformer,      # reference model (frozen, plain nn.Module on device)
    config,
    accelerator,
    global_step,
    weight_dtype,
):
    """
    DPO training iteration for Flux2Klein.

    Expected batch keys (from ArXiVParquetDatasetDPO):
        winner_latents  : (B, C, H, W) — winner image VAE latents (sampled from vae_h)
        loser_latents   : (B, C, H, W) — loser image VAE latents
        text_embeds     : (B, L, D)    — shared text embeddings
        text_mask       : (B, L)       — shared attention mask
        text_ids        : (B, L, 4)    — optional; generated if absent

    Config keys consumed (all optional):
        dpo_beta          (float, default 2000.0) — DPO temperature
        weighting_scheme  (str)
        logit_mean/std/mode_scale
        guidance_scale / validation_guidance_scale
    """
    device = accelerator.device

    # ── Move inputs to device ────────────────────────────────────────────────
    winner_latents = batch["winner_latents"].to(device, dtype=weight_dtype)
    loser_latents  = batch["loser_latents"].to(device, dtype=weight_dtype)
    prompt_embeds  = batch["text_embeds"].to(device, dtype=weight_dtype)

    bsz, C, H, W = winner_latents.shape
    lat_h, lat_w = H // 2, W // 2   # spatial dims after 2×2 patchify

    # ── Shape assertion: winner and loser must be from the SAME bucket ────────
    # If they differ, img_ids would be built for the wrong token count and
    # the transformer RoPE coordinates would silently mis-align.
    if winner_latents.shape != loser_latents.shape:
        raise ValueError(
            f"[DPO] winner/loser latent shape mismatch: "
            f"{winner_latents.shape} vs {loser_latents.shape}. "
            "Both must come from the same resolution bucket."
        )

    # ── Patchify + normalize both latents ────────────────────────────────────
    packed_winner = _patchify_and_normalize(winner_latents, vae, weight_dtype, device)
    packed_loser  = _patchify_and_normalize(loser_latents,  vae, weight_dtype, device)
    # packed_*: (B, L, C')  where L = lat_h * lat_w = (H//2)*(W//2)

    # ── Sample ONE set of timesteps per pair (same t for winner & loser) ─────
    u = compute_density_for_timestep_sampling(
        weighting_scheme=config.weighting_scheme,
        batch_size=bsz,
        logit_mean=config.logit_mean,
        logit_std=config.logit_std,
        mode_scale=config.mode_scale,
        u_value_min=getattr(config, 'u_value_min', None),
        u_value_max=getattr(config, 'u_value_max', None),
    )
    # Clamp indices to [0, T-1] — guards against u_value_max == 1.0 edge case
    T = noise_scheduler_copy.config.num_train_timesteps
    indices   = (u * T).long().clamp(0, T - 1)
    timesteps = noise_scheduler_copy.timesteps[indices].to(device=device)   # (B,)

    # ── D3PO: use ONE shared noise for winner AND loser ───────────────────────
    # Sharing ε across the pair eliminates noise-driven variance from the
    # MSE difference (L_pi_w - L_pi_l), leaving only the model's preference
    # signal.  See Wallace et al. 2023, §3.1.
    noise  = torch.randn_like(packed_winner)                                # (B, L, C')
    sigmas = get_sigmas(timesteps, device=device,
                        noise_scheduler_copy=noise_scheduler_copy,
                        n_dim=packed_winner.ndim, dtype=weight_dtype)       # (B, 1, 1)

    noisy_winner = (1.0 - sigmas) * packed_winner + sigmas * noise
    noisy_loser  = (1.0 - sigmas) * packed_loser  + sigmas * noise

    # Flow-matching targets: v* = ε - latent  (same ε for both)
    target_w = noise - packed_winner  # (B, L, C')
    target_l = noise - packed_loser   # (B, L, C')

    # Loss-weighting scheme (σ-based)
    weighting = compute_loss_weighting_for_sd3(
        weighting_scheme=config.weighting_scheme, sigmas=sigmas
    )  # (B, 1, 1)

    # ── Position IDs ─────────────────────────────────────────────────────────
    # Get transformer dtype (DDP wraps model with .module)
    _unwrapped = transformer.module if hasattr(transformer, 'module') else transformer
    transformer_dtype = _unwrapped.dtype

    # Image position IDs — same spatial layout for winner and loser.
    # Derived from the original latent dims (H, W), divided by the patchify
    # factor of 2.  Both winner and loser share these IDs because they are
    # guaranteed to have the same bucket shape (asserted above).
    img_ids_single = _prepare_latent_image_ids(
        batch_size=bsz,
        height=lat_h, width=lat_w,
        device=device, dtype=transformer_dtype,
        already_patchified=True,
    )  # (B, lat_h*lat_w, 4)

    # Text position IDs
    if "text_ids" in batch and batch["text_ids"] is not None:
        txt_ids_single = batch["text_ids"].to(device, dtype=transformer_dtype)
    else:
        seq_len = prompt_embeds.shape[1]
        txt_ids_single = _prepare_text_ids(seq_len, bsz, device, transformer_dtype)

    # Guidance scale (CFG) — broadcast to (B,)
    _gs = getattr(config, 'guidance_scale', None) or config.validation_guidance_scale
    guidance_single = torch.tensor([_gs], device=device, dtype=weight_dtype).expand(bsz)

    # ── Concatenate winner+loser along batch dim for a single forward pass ───
    # Shape: (2B, L, C') — winner rows first, loser rows second
    noisy_both   = torch.cat([noisy_winner, noisy_loser],  dim=0)  # (2B, L, C')
    target_both  = torch.cat([target_w,     target_l],     dim=0)  # (2B, L, C')
    weight_both  = weighting.repeat(2, 1, 1)                        # (2B, 1, 1)

    img_ids_both  = img_ids_single.repeat(2, 1, 1)
    txt_ids_both  = txt_ids_single.repeat(2, 1, 1)
    prompt_both   = prompt_embeds.repeat(2, 1, 1)
    guidance_both = guidance_single.repeat(2)
    timesteps_both = timesteps.repeat(2)

    # ── Policy forward (with gradient) ───────────────────────────────────────
    policy_pred = _forward_transformer(
        transformer, noisy_both, timesteps_both, guidance_both,
        prompt_both, img_ids_both, txt_ids_both, no_grad=False,
    )  # (2B, L, C')

    loss_policy_w = _per_sample_mse(policy_pred[:bsz], target_both[:bsz], weight_both[:bsz])
    loss_policy_l = _per_sample_mse(policy_pred[bsz:], target_both[bsz:], weight_both[bsz:])

    # ── Reference forward (no gradient) ──────────────────────────────────────
    # LoRA mode (ref_transformer=None): disable LoRA adapters → base model = reference.
    # Full fine-tune mode: use frozen ref_transformer copy.
    use_lora = getattr(config, 'use_lora', False)
    if use_lora:
        # Reference = same transformer with LoRA disabled.
        ref_pred = _forward_transformer(
            transformer, noisy_both, timesteps_both, guidance_both,
            prompt_both, img_ids_both, txt_ids_both,
            no_grad=True, disable_lora=True,
        )  # (2B, L, C')
    else:
        ref_pred = _forward_transformer(
            ref_transformer, noisy_both, timesteps_both, guidance_both,
            prompt_both, img_ids_both, txt_ids_both, no_grad=True,
        )  # (2B, L, C')

    loss_ref_w = _per_sample_mse(ref_pred[:bsz].detach(), target_both[:bsz], weight_both[:bsz])
    loss_ref_l = _per_sample_mse(ref_pred[bsz:].detach(), target_both[bsz:], weight_both[bsz:])

    # ── DPO loss ──────────────────────────────────────────────────────────────
    # logits[b] = (loss_ref_w[b] - loss_policy_w[b]) - (loss_ref_l[b] - loss_policy_l[b])
    #           = ref_diff[b] - model_diff[b]  (matches VisionFlow naming)
    # Positive logit → policy correctly prefers winner over loser.
    beta = float(config.get('dpo_beta', 2000.0))
    logits = (loss_ref_w - loss_policy_w) - (loss_ref_l - loss_policy_l)   # (B,) per-sample
    dpo_loss = -F.logsigmoid(beta * logits).mean()

    # implicit_acc: fraction of samples where policy correctly prefers winner
    # (with 0.5 weight for ties, matching VisionFlow)
    implicit_acc = (logits > 0).float().mean() + 0.5 * (logits == 0).float().mean()  # scalar

    # raw_model_loss and ref_loss (matching VisionFlow logging)
    raw_model_loss = 0.5 * (loss_policy_w.mean() + loss_policy_l.mean())
    ref_loss       = 0.5 * (loss_ref_w.mean()    + loss_ref_l.mean())

    # ── Optional auxiliary SFT loss on winner (regularisation) ───────────────
    sft_weight = float(config.get('dpo_sft_weight', 0.0))
    if sft_weight > 0:
        sft_loss = loss_policy_w.mean()
        loss = dpo_loss + sft_weight * sft_loss
    else:
        loss = dpo_loss

    # ── DEBUG diagnostics (rank-0, every step when debug_mode=True) ──────────
    # Enable via config: debug_mode = True
    debug_mode = bool(config.get('debug_mode', False))
    if accelerator.is_main_process and debug_mode:
        with torch.no_grad():
            # 1. Are winner and loser latents actually different?
            diff = (packed_winner - packed_loser).float()
            diff_abs_mean = diff.abs().mean().item()
            diff_abs_max  = diff.abs().max().item()
            diff_zero_frac = (diff.abs() < 1e-6).float().mean().item()

            # 2. Fallback-zero detection (zero latents = NPZ read failed → _fallback())
            w_norm = packed_winner.float().abs().flatten(1).max(dim=1).values  # (B,)
            l_norm = packed_loser.float().abs().flatten(1).max(dim=1).values   # (B,)
            fallback_frac = ((w_norm < 1e-6) | (l_norm < 1e-6)).float().mean().item()

            # 3. Per-sample logits stats (VisionFlow: logits_mean / logits_std / logits_abs_mean)
            logits_f = logits.float().detach()
            logits_mean_val    = logits_f.mean().item()
            logits_std_val     = logits_f.std().item() if bsz > 1 else 0.0
            logits_abs_mean_val = logits_f.abs().mean().item()

            # 4. Ref model param norm — should be CONSTANT across steps
            if ref_transformer is not None:
                # Use first and last param only to avoid O(9B) scan every step
                _params = list(ref_transformer.parameters())
                ref_param_norm_approx = (_params[0].float().norm().item() +
                                         _params[-1].float().norm().item())
            else:
                ref_param_norm_approx = float('nan')

            # 5. Sigma / timestep stats
            sigma_mean = sigmas.float().mean().item()
            sigma_std  = sigmas.float().std().item() if bsz > 1 else 0.0

            print(
                f"\n[DPO debug step={global_step}]\n"
                f"  DATA: latent_diff_mean={diff_abs_mean:.5f}  latent_diff_max={diff_abs_max:.4f}  "
                f"zero_frac={diff_zero_frac:.3f}  fallback_frac={fallback_frac:.3f}\n"
                f"  LOSS: policy_w={loss_policy_w.mean().item():.4f}  "
                f"policy_l={loss_policy_l.mean().item():.4f}  "
                f"delta_policy={loss_policy_w.mean().item() - loss_policy_l.mean().item():.5f}  "
                f"ref_w={loss_ref_w.mean().item():.4f}  "
                f"ref_l={loss_ref_l.mean().item():.4f}  "
                f"delta_ref={loss_ref_w.mean().item() - loss_ref_l.mean().item():.5f}\n"
                f"  DPO:  logits_mean={logits_mean_val:.6f}  logits_std={logits_std_val:.6f}  "
                f"logits_abs_mean={logits_abs_mean_val:.6f}  "
                f"implicit_acc={implicit_acc.item():.3f}  "
                f"dpo_loss={dpo_loss.item():.5f}\n"
                f"  REF:  ref_param_norm_approx={ref_param_norm_approx:.4f}  "
                f"sigma_mean={sigma_mean:.4f}  sigma_std={sigma_std:.4f}  "
                f"t_sample={timesteps[:min(4, bsz)].tolist()}"
            )

    # ── KL divergence estimates ───────────────────────────────────────────────
    # Per-sample implicit log-ratio: log(π/π_ref) ∝ L_ref - L_policy (MSE proxy).
    # kl_winner: policy improvement on winner samples (positive = policy closer to winner than ref)
    # kl_loser:  policy improvement on loser  samples (positive = policy also drifted toward loser)
    # kl_loss:   average KL proxy across both sides; indicates overall deviation from reference.
    #            Together: logits = kl_winner - kl_loser (the DPO preference margin).
    kl_winner = (loss_ref_w - loss_policy_w).mean().detach()
    kl_loser  = (loss_ref_l - loss_policy_l).mean().detach()
    kl_loss   = 0.5 * (kl_winner + kl_loser)

    # delta_ref: ref model's INTRINSIC preference for winner over loser.
    # Formula: delta_ref = (loss_ref_w - loss_ref_l).mean()
    # Interpretation:
    #   ≈ 0  → ref treats winner and loser equally (typical for rollout vs rollout)
    #   >> 0 → ref is inherently WORSE at winner (e.g. winner=GT, ref trained on rollout)
    #   << 0 → ref is inherently BETTER at winner (unusual; would mean ref agrees with labels)
    #
    # DIAGNOSTIC:
    #   - scored-pair runs: delta_ref ≈ 0 always (both rollout, same distribution to ref)
    #   - gt_all run:       delta_ref >> 0   ← ref pays higher MSE on GT (out-of-distribution)
    #
    # If delta_ref >> 0 from step 0, it means reward_acc rises because of ref model bias,
    # NOT because the policy is learning a quality preference.  Compare delta_ref vs
    # delta_policy = (loss_policy_w - loss_policy_l).mean() as training proceeds:
    #   - delta_policy should become NEGATIVE (policy improves more on winner than loser)
    #   - if delta_policy stays ≈ 0 while reward_acc rises, the signal comes entirely from delta_ref
    delta_ref    = (loss_ref_w    - loss_ref_l).mean().detach()
    delta_policy = (loss_policy_w - loss_policy_l).mean().detach()

    return loss, {
        # ── Core DPO metrics (matches VisionFlow naming) ──────────────────
        "loss":            loss.detach(),
        "dpo_loss":        dpo_loss.detach(),
        "raw_model_loss":  raw_model_loss.detach(),
        "ref_loss":        ref_loss.detach(),
        "implicit_acc":    implicit_acc.detach(),
        # logits = per-sample (B,) → log their statistics
        "logits_mean":     logits.float().mean().detach(),
        "logits_std":      logits.float().std().detach() if bsz > 1 else torch.zeros(1).squeeze(),
        "logits_abs_mean": logits.float().abs().mean().detach(),
        # ── KL divergence proxies ─────────────────────────────────────────
        "kl_winner":       kl_winner,   # log π/π_ref on winning samples
        "kl_loser":        kl_loser,    # log π/π_ref on losing  samples
        "kl_loss":         kl_loss,     # 0.5*(kl_winner+kl_loser): overall KL offset
        # ── Ref-asymmetry diagnostic ──────────────────────────────────────
        # delta_ref > 0 means the ref model inherently struggles more on the winner
        # (e.g. winner=GT images are OOD for a ref trained on rollout).
        # If delta_ref >> 0 from step-0, reward_acc rises from ref-model bias, NOT
        # from the policy learning a quality preference.
        # delta_policy becomes negative as the policy learns to prefer the winner.
        "delta_ref":       delta_ref,       # loss_ref_w  - loss_ref_l  (ref intrinsic asymmetry)
        "delta_policy":    delta_policy,    # loss_pol_w  - loss_pol_l  (policy learning signal)
        # ── Legacy names kept for backward compat ────────────────────────
        "loss_policy_w":   loss_policy_w.mean().detach(),
        "loss_policy_l":   loss_policy_l.mean().detach(),
        "loss_ref_w":      loss_ref_w.mean().detach(),
        "loss_ref_l":      loss_ref_l.mean().detach(),
        "implicit_margin": logits.mean().detach(),   # same as logits_mean, kept for compat
        "reward_acc":      implicit_acc.detach(),    # same as implicit_acc
        # ── Data-sanity diagnostics ───────────────────────────────────────
        "latent_diff_mean": (packed_winner - packed_loser).float().abs().mean().detach(),
    }
