import os
"""
MD3PO 1v2+global (comp + text + global worst) - hardness+timestep alpha — B200 x8 GPU.

Tuple construction per group:
  winner + {loser_component, loser_text, loser_global}
  - loser_component: dim-anchored hard negative on component_score
  - loser_text: dim-anchored hard negative on text_score
  - loser_global: overall worst sample (argmin reward)

Loss:
  mean_b[ sum_d alpha_d(hardness, timestep) * -logsigmoid(beta * logits_d) ]

Alpha gate: component dominates high-noise, text+global dominate low-noise.

Fixes over previous MD3PO:
  - NaN/Inf rows filtered before group processing
  - Deterministic tie-breaking for argmax/argmin
  - Global worst provides strong universal negative signal
"""

_base_ = ['./mdpo_1v3_uniform.py']

# Override to 2 dimensions + global worst injection
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
    min_winner_score=0.80,
    target_min_gap=0.40,
    other_min_gap=0.00,
    other_max_gap=0.50,
    loser_balance_lambda=0.50,
    min_total_gap=0.20,
    strict_all_dims=True,
    require_distinct_losers=True,
    min_group_images=6,
    min_reward=None,

    # Global worst injection
    inject_global_worst=True,
    global_worst_min_gap=0.20,
)

# B200 has 8x192GB VRAM — keep ref on GPU
ref_on_cpu = False
checkpoints_total_limit = None

# Hardness + timestep gated alpha (comp, text only; global gets fixed weight)
md3po_alpha_mode = 'hardness_timestep'
md3po_tau_h = 0.20
md3po_timestep_split = 0.55
md3po_gate_eps = 1e-2
md3po_detach_alpha = True
md3po_global_loss_weight = 0.50

# Run isolation
model_output_dir = os.environ.get('SCIFORMA_DATA_ROOT', '') + '/experiments/260411_md3po_1v2ct_global_htimestep_b2008'
tracker_run_name = 'md3po_1v2ct_global_htimestep_b2008'
resume_from_checkpoint = None

max_train_steps = 10000
checkpointing_steps = 500
validation_steps = 500
