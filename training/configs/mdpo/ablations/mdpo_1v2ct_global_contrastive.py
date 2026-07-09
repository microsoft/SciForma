import os
"""
MD3PO Contrastive — 1v2+global (comp + text + global worst) — B200 x4 GPU.

Key innovation: Multi-choice Bradley-Terry contrastive loss
  L = (1/D) · log(1 + Σ_d exp(-β·Δ_d))
instead of mean/alpha:
  L = Σ_d α_d · (-log σ(β·Δ_d))

1/D scaling matches loss magnitude to standard DPO, preventing
disproportionate gradient clipping under max_grad_norm=1.0.

Properties:
  - Strict generalization of DPO (reduces to DPO when D=1)
  - Hardest-to-distinguish loser automatically gets largest gradient
    weight via softmax, no heuristic alpha needed
  - Multiple losers' evidence ADDS in logsumexp → stronger signal
  - Same data curation as previous MD3PO (dimensional hard negatives)

Tuple construction per group:
  winner + {loser_component, loser_text, loser_global}

Training: 4 GPU × batch=1 × GA=3 → eff_batch=12
"""

_base_ = ['./mdpo_1v3_uniform.py']

# ── Dataset: 1v2 + global worst ──────────────────────────────────────────
dataset_cfg = dict(
    _delete_=True,
    type='ArXiVParquetDatasetMD3PO',
    base_dir=os.environ.get('SCIFORMA_DATA_ROOT', '/data/yuxuanluo'),
    parquet_base_path='ArXiV_parquet/0407_longshort_gdro_vae_v9r',
    parquet_glob='gdro_rank_*.parquet',

    num_workers=8,
    debug_mode=False,
    is_main_process=True,
    stat_data=True,

    path_remapping={'/mnt/data/': os.environ.get('SCIFORMA_DATA_ROOT', '') + '/'},
    deterministic_latents=True,

    target_dims=('component_score', 'text_score'),
    winner_key='reward',
    min_winner_score=0.70,
    target_min_gap=0.25,
    other_min_gap=0.00,
    other_max_gap=0.60,
    loser_balance_lambda=0.50,
    min_total_gap=0.15,
    strict_all_dims=True,
    require_distinct_losers=True,
    min_group_images=4,
    min_reward=None,

    # Global worst injection
    inject_global_worst=True,
    global_worst_min_gap=0.20,
)

# ── Contrastive aggregation (replaces alpha/mean) ────────────────────────
md3po_agg_mode = 'contrastive'

# These are ignored in contrastive mode but kept for backward-compat:
md3po_alpha_mode = 'uniform'
md3po_global_loss_weight = 0.0   # not used in contrastive mode

train_iteration_func = 'Flux2Klein_md3po_train_iteration'
ref_on_cpu = False
checkpoints_total_limit = None

# ── Match ablation settings (4 GPU) ─────────────────────────────────────
gradient_accumulation_steps = 3       # 4 GPU × 1 × 3 = eff_batch 12
lr_scheduler = 'constant_with_warmup'
lr_warmup_steps = 50
gradient_checkpointing = True
ema_on_gpu = True
max_grad_norm = 1.5                   # 1.5× base (compensate multi-loser signal)

# ── Run isolation ────────────────────────────────────────────────────────
model_output_dir = os.environ.get('SCIFORMA_DATA_ROOT', '') + '/experiments/260412_md3po_1v2ct_global_contrastive_v2_b2008'
tracker_run_name = 'md3po_1v2ct_global_contrastive_v2_b2008'
resume_from_checkpoint = None

max_train_steps = 10000
checkpointing_steps = 500
validation_steps = 500
