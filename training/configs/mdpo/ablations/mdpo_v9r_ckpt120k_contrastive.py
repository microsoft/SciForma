import os
"""
MD3PO Contrastive (logsumexp) — v9r 16K — ckpt-120000 — B200 4-GPU.

Same dataset/loss as 260413 contrastive, only change: base model ckpt-120000.
Comparison: 260413 uses ckpt-90000.
"""

_base_ = ['../mdpo.py']

# ── Base model: checkpoint-120000 ────────────────────────────────────────
policy_init_path = (
    os.environ.get("SCIFORMA_DATA_ROOT", "") + "/experiments/260216_stage2_mixed_gen_edit_b200_uniform_12wstep"
    "/checkpoint-120000/ema_weights.pt"
)
ref_init_path = policy_init_path

# ── Run isolation ────────────────────────────────────────────────────────
model_output_dir = os.environ.get('SCIFORMA_DATA_ROOT', '') + '/experiments/260414_md3po_contrastive_v9r16k_ckpt120k'
tracker_run_name = '260414_md3po_contrastive_v9r16k_ckpt120k'
