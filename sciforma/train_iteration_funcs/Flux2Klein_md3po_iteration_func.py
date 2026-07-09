"""
Flux2Klein MD3PO iteration (1 winner vs N dimension-anchored losers + optional global worst).

Loss (without global):
  L = mean_b [ sum_d alpha_d(b) * ( -logsigmoid(beta * logits_d(b)) ) ]

Loss (with global):
  L = mean_b [ sum_d alpha_d(b) * ( -logsigmoid(beta * logits_d(b)) ) ]
      + lambda_global * mean_b [ -logsigmoid(beta * logits_global(b)) ]

  Global worst is an additive term with fixed weight (md3po_global_loss_weight),
  NOT mixed into the alpha softmax.  Alpha only distributes weight among the
  dimension-specific losers (component, text, …).

Contrastive mode (md3po_agg_mode='contrastive'):
  L = mean_b [ log(1 + sum_d exp(-beta * logits_d(b))) ]
  Losers coupled via logsumexp in forward pass.

Adaptive mode (md3po_agg_mode='adaptive'):
  w_d = softmax(-beta * logits, dim=D)   [detached]
  L = mean_b [ sum_d w_d * (-logsigmoid(beta * logits_d)) ]
  Contrastive-derived weights but independent -logsigmoid per dim.
  Avoids gradient conflict from forward-pass coupling in shared-noise
  diffusion DPO while focusing on hardest loser.

where
  logits_d = (L_ref_w - L_pi_w) - (L_ref_l_d - L_pi_l_d)

alpha modes (applied to dimension-specific losers only, mean mode):
  - uniform:             equal weight per dim
  - hardness:            softmax over policy loss gaps
  - hardness_timestep:   hardness × timestep gate
"""

import torch
import torch.nn.functional as F
from diffusers.training_utils import compute_loss_weighting_for_sd3

from sciforma.utils import get_sigmas, compute_density_for_timestep_sampling
from sciforma.registry import TRAIN_ITERATION_FUNCS
from sciforma.train_iteration_funcs.Flux2Klein_dpo_iteration_func import (
    _prepare_latent_image_ids,
    _prepare_text_ids,
    _patchify_and_normalize,
    _forward_transformer,
    _per_sample_mse,
)


def _safe_norm_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.sum(dim=1, keepdim=True) + eps)


def _compute_alpha(sigmas, policy_w, policy_ls, config, num_specific):
    """Return (B, num_specific) alpha weights for dimension-specific losers only.

    Global worst (if present) is handled separately as an additive loss term
    in the main function — it never enters the alpha softmax.

    Args:
        num_specific: number of dimension-specific losers (excluding global).
    """
    mode = str(config.get("md3po_alpha_mode", "uniform")).lower()
    tau_h = float(config.get("md3po_tau_h", 0.20))
    detach_alpha = bool(config.get("md3po_detach_alpha", True))
    gate_eps = float(config.get("md3po_gate_eps", 1e-3))

    bsz = policy_w.shape[0]
    specific_ls = policy_ls[:num_specific]

    if mode == "uniform" or num_specific <= 1:
        raw = torch.ones((bsz, num_specific), device=policy_w.device, dtype=policy_w.dtype)
        return _safe_norm_rows(raw)

    hardness = torch.stack([(pl - policy_w) for pl in specific_ls], dim=1)
    if detach_alpha:
        hardness = hardness.detach()
    raw = torch.exp(hardness / max(1e-6, tau_h))

    if mode == "hardness_timestep":
        sigma = sigmas.view(bsz).float()
        split = float(config.get("md3po_timestep_split", 0.55))
        split = min(0.99, max(0.01, split))
        gate_early = torch.clamp((sigma - split) / max(1e-6, 1.0 - split), 0.0, 1.0)
        gate_late = torch.clamp((split - sigma) / max(1e-6, split), 0.0, 1.0)
        gate_list = [gate_early] + [gate_late] * (num_specific - 1)
        gates = torch.stack(gate_list, dim=1).to(raw.dtype)
        raw = raw * (gates + gate_eps)

    return _safe_norm_rows(raw)


@TRAIN_ITERATION_FUNCS.register_module()
def Flux2Klein_md3po_train_iteration(
    batch,
    vae,
    noise_scheduler_copy,
    transformer,
    ref_transformer,
    config,
    accelerator,
    global_step,
    weight_dtype,
):
    device = accelerator.device

    # Discover active dimensions from batch (set by dataset)
    dim_aliases = batch.get("_dim_aliases", ["component", "text", "arrow"])
    num_dims = len(dim_aliases)

    winner_latents = batch["winner_latents"].to(device, dtype=weight_dtype)
    losers = []
    for alias in dim_aliases:
        losers.append(batch[f"loser_{alias}_latents"].to(device, dtype=weight_dtype))
    prompt_embeds = batch["text_embeds"].to(device, dtype=weight_dtype)

    bsz, _, h, w = winner_latents.shape
    lat_h, lat_w = h // 2, w // 2
    num_all = 1 + num_dims  # winner + losers

    for i, l in enumerate(losers):
        if l.shape != winner_latents.shape:
            raise ValueError(
                f"[MD3PO] winner/loser[{i}] shape mismatch: {winner_latents.shape} vs {l.shape}"
            )

    packed_winner = _patchify_and_normalize(winner_latents, vae, weight_dtype, device)
    packed_losers = [_patchify_and_normalize(l, vae, weight_dtype, device) for l in losers]

    u = compute_density_for_timestep_sampling(
        weighting_scheme=config.weighting_scheme,
        batch_size=bsz,
        logit_mean=config.logit_mean,
        logit_std=config.logit_std,
        mode_scale=config.mode_scale,
        u_value_min=getattr(config, "u_value_min", None),
        u_value_max=getattr(config, "u_value_max", None),
    )
    tmax = noise_scheduler_copy.config.num_train_timesteps
    indices = (u * tmax).long().clamp(0, tmax - 1)
    timesteps = noise_scheduler_copy.timesteps[indices].to(device=device)

    noise = torch.randn_like(packed_winner)
    sigmas = get_sigmas(
        timesteps,
        device=device,
        noise_scheduler_copy=noise_scheduler_copy,
        n_dim=packed_winner.ndim,
        dtype=weight_dtype,
    )

    noisy_w = (1.0 - sigmas) * packed_winner + sigmas * noise
    target_w = noise - packed_winner

    noisy_ls = [(1.0 - sigmas) * pl + sigmas * noise for pl in packed_losers]
    target_ls = [noise - pl for pl in packed_losers]

    weighting = compute_loss_weighting_for_sd3(
        weighting_scheme=config.weighting_scheme,
        sigmas=sigmas,
    )

    num_all = 1 + num_dims

    # Concat order: winner, loser_0, loser_1, ... loser_{D-1}
    noisy_all = torch.cat([noisy_w] + noisy_ls, dim=0)
    target_all = torch.cat([target_w] + target_ls, dim=0)
    weight_all = weighting.repeat(num_all, 1, 1)
    timesteps_all = timesteps.repeat(num_all)

    _unwrapped = transformer.module if hasattr(transformer, "module") else transformer
    transformer_dtype = _unwrapped.dtype

    img_ids_single = _prepare_latent_image_ids(
        batch_size=bsz,
        height=lat_h,
        width=lat_w,
        device=device,
        dtype=transformer_dtype,
        already_patchified=True,
    )

    if "text_ids" in batch and batch["text_ids"] is not None:
        txt_ids_single = batch["text_ids"].to(device, dtype=transformer_dtype)
    else:
        txt_ids_single = _prepare_text_ids(prompt_embeds.shape[1], bsz, device, transformer_dtype)

    _gs = getattr(config, "guidance_scale", None) or config.validation_guidance_scale
    guidance_single = torch.tensor([_gs], device=device, dtype=weight_dtype).expand(bsz)

    img_ids_all = img_ids_single.repeat(num_all, 1, 1)
    txt_ids_all = txt_ids_single.repeat(num_all, 1, 1)
    prompt_all = prompt_embeds.repeat(num_all, 1, 1)
    guidance_all = guidance_single.repeat(num_all)

    policy_pred = _forward_transformer(
        transformer,
        noisy_all,
        timesteps_all,
        guidance_all,
        prompt_all,
        img_ids_all,
        txt_ids_all,
        no_grad=False,
    )

    use_lora = bool(config.get("use_lora", False))
    if use_lora:
        ref_pred = _forward_transformer(
            transformer,
            noisy_all,
            timesteps_all,
            guidance_all,
            prompt_all,
            img_ids_all,
            txt_ids_all,
            no_grad=True,
            disable_lora=True,
        )
    else:
        ref_pred = _forward_transformer(
            ref_transformer,
            noisy_all,
            timesteps_all,
            guidance_all,
            prompt_all,
            img_ids_all,
            txt_ids_all,
            no_grad=True,
        )

    pol_losses = [_per_sample_mse(policy_pred[i * bsz:(i + 1) * bsz], target_all[i * bsz:(i + 1) * bsz], weight_all[i * bsz:(i + 1) * bsz]) for i in range(num_all)]
    ref_losses = [_per_sample_mse(ref_pred[i * bsz:(i + 1) * bsz].detach(), target_all[i * bsz:(i + 1) * bsz], weight_all[i * bsz:(i + 1) * bsz]) for i in range(num_all)]

    loss_policy_w = pol_losses[0]
    loss_policy_ls = pol_losses[1:]
    loss_ref_w = ref_losses[0]
    loss_ref_ls = ref_losses[1:]

    beta = float(config.get("dpo_beta", 2000.0))
    logits_list = [
        (loss_ref_w - loss_policy_w) - (loss_ref_ls[i] - loss_policy_ls[i])
        for i in range(num_dims)
    ]
    logits = torch.stack(logits_list, dim=1)  # (B, D)
    per_dim_dpo = -F.logsigmoid(beta * logits)  # (B, D)

    # --- Split dimension-specific vs global ---
    _has_global = "global" in dim_aliases
    num_specific = num_dims - 1 if _has_global else num_dims

    # --- Loss aggregation mode ---
    agg_mode = str(config.get("md3po_agg_mode", "mean")).lower()

    if agg_mode == "contrastive":
        # ── Multi-choice Bradley-Terry (contrastive DPO) ──────────────
        # L = (1/D) * log(1 + Σ_d exp(-β·Δ_d))
        # Scaled by 1/D to match mean-mode loss magnitude and prevent
        # disproportionate gradient clipping.
        neg_beta_logits = -beta * logits  # (B, D)
        zeros = torch.zeros(bsz, 1, device=device, dtype=neg_beta_logits.dtype)
        dpo_loss = torch.logsumexp(
            torch.cat([zeros, neg_beta_logits], dim=1), dim=1
        ).mean() / float(num_dims)

        with torch.no_grad():
            _exp_neg = torch.exp(neg_beta_logits)
            _denom = 1.0 + _exp_neg.sum(dim=1, keepdim=True)
            alpha_metric = (_exp_neg / _denom)  # (B, D) — auto-adaptive weights

        global_dpo_loss = torch.zeros((), device=device, dtype=weight_dtype)

    elif agg_mode == "adaptive":
        # ── Detached Contrastive Weights + Independent -logσ ──────────
        # Weights from multi-choice Bradley-Terry (detached, no gradient):
        #   w_d = softmax(-β·Δ) across losers
        # Loss uses independent per-dim -logσ (no forward-pass coupling):
        #   L = Σ_d w_d · (-logσ(β·Δ_d))
        #
        # Avoids shared-noise gradient conflict (each loser flows through
        # its own sigmoid independently), while focusing on hardest loser.
        with torch.no_grad():
            alpha_metric = F.softmax(-beta * logits, dim=1)  # (B, D)

        dpo_loss = (alpha_metric.detach() * per_dim_dpo).sum(dim=1).mean()
        global_dpo_loss = torch.zeros((), device=device, dtype=weight_dtype)

    else:
        # ── Original mean/alpha aggregation (backward-compatible) ─────
        alpha = _compute_alpha(
            sigmas, loss_policy_w, loss_policy_ls, config, num_specific=num_specific,
        ).to(per_dim_dpo.dtype)

        dim_dpo_loss = (alpha * per_dim_dpo[:, :num_specific]).sum(dim=1).mean()

        if _has_global:
            global_loss_weight = float(config.get("md3po_global_loss_weight", 0.0))
            global_dpo_loss = per_dim_dpo[:, -1].mean()
        else:
            global_loss_weight = 0.0
            global_dpo_loss = torch.zeros((), device=device, dtype=weight_dtype)

        dpo_loss = dim_dpo_loss + global_loss_weight * global_dpo_loss

        if _has_global:
            alpha_metric = torch.cat([
                alpha, torch.full((alpha.shape[0], 1), global_loss_weight, device=device, dtype=alpha.dtype)
            ], dim=1)
            alpha_metric = _safe_norm_rows(alpha_metric)
        else:
            alpha_metric = alpha

    sft_weight = float(config.get("dpo_sft_weight", 0.0))
    if sft_weight > 0:
        sft_loss = loss_policy_w.mean()
        loss = dpo_loss + sft_weight * sft_loss
    else:
        sft_loss = torch.zeros((), device=device, dtype=weight_dtype)
        loss = dpo_loss

    # --- Metrics (aligned across mean / contrastive modes) ---
    tie_weighted_acc = (logits > 0).float() + 0.5 * (logits == 0).float()

    if agg_mode in ("contrastive", "adaptive"):
        # implicit_acc: simple 1/D mean per-dim acc — same formula as mean-mode uniform
        implicit_acc = tie_weighted_acc.mean(dim=1).mean()
        # reward_acc: strict all-correct
        reward_acc = (logits > 0).all(dim=1).float().mean()
    else:
        implicit_acc = (alpha_metric * tie_weighted_acc).sum(dim=1).mean()
        reward_acc = implicit_acc

    raw_model_loss = (loss_policy_w.mean() + sum(x.mean() for x in loss_policy_ls)) / float(num_all)
    ref_loss = (loss_ref_w.mean() + sum(x.mean() for x in loss_ref_ls)) / float(num_all)

    kl_winner = (loss_ref_w - loss_policy_w).mean().detach()

    # For aggregated metrics use alpha in mean mode, simple 1/D mean in contrastive/adaptive
    if agg_mode in ("contrastive", "adaptive"):
        _alpha_for_agg = torch.ones(bsz, num_dims, device=device, dtype=weight_dtype) / float(num_dims)
    else:
        _alpha_for_agg = _safe_norm_rows(alpha_metric) if alpha_metric.sum() > 0 else alpha_metric
    kl_loser = (_alpha_for_agg * torch.stack([(loss_ref_ls[i] - loss_policy_ls[i]) for i in range(num_dims)], dim=1)).sum(dim=1).mean().detach()
    kl_loss = 0.5 * (kl_winner + kl_loser)

    delta_ref = (_alpha_for_agg * torch.stack([(loss_ref_w - loss_ref_ls[i]) for i in range(num_dims)], dim=1)).sum(dim=1).mean().detach()
    delta_policy = (_alpha_for_agg * torch.stack([(loss_policy_w - loss_policy_ls[i]) for i in range(num_dims)], dim=1)).sum(dim=1).mean().detach()

    metrics = {
        "loss": loss.detach(),
        "dpo_loss": dpo_loss.detach(),
        "dim_dpo_loss": dpo_loss.detach(),
        "global_dpo_loss": global_dpo_loss.detach(),
        "sft_loss": sft_loss.detach(),
        "raw_model_loss": raw_model_loss.detach(),
        "ref_loss": ref_loss.detach(),
        "implicit_acc": implicit_acc.detach(),
        "reward_acc": reward_acc.detach(),
        "logits_mean": logits.float().mean().detach(),
        "logits_std": logits.float().std().detach() if logits.numel() > 1 else torch.zeros(1, device=device).squeeze(),
        "logits_abs_mean": logits.float().abs().mean().detach(),
        "kl_winner": kl_winner,
        "kl_loser": kl_loser,
        "kl_loss": kl_loss,
        "delta_ref": delta_ref,
        "delta_policy": delta_policy,
        "loss_policy_w": loss_policy_w.mean().detach(),
        "loss_ref_w": loss_ref_w.mean().detach(),
        "latent_diff_mean": (packed_winner - packed_losers[0]).float().abs().mean().detach(),
    }

    # Per-dimension metrics (dynamic)
    for i, alias in enumerate(dim_aliases):
        metrics[f"loss_policy_l_{alias}"] = loss_policy_ls[i].mean().detach()
        metrics[f"loss_ref_l_{alias}"] = loss_ref_ls[i].mean().detach()
        metrics[f"logits_{alias}"] = logits[:, i].float().mean().detach()
        metrics[f"implicit_acc_{alias}"] = ((logits[:, i] > 0).float() + 0.5 * (logits[:, i] == 0).float()).mean().detach()
        metrics[f"alpha_{alias}"] = alpha_metric[:, i].mean().detach()

    return loss, metrics
