import os
"""
MD3PO 1v2 (component + text only, no arrow) - hardness+timestep alpha — B200 x8 GPU.

Tuple construction per group:
  winner + {loser_component, loser_text}

Loss:
  mean_b[ sum_d alpha_d(hardness, timestep) * -logsigmoid(beta * logits_d) ]

Alpha gate: component dominates high-noise, text dominates low-noise.
"""

_base_ = ['./mdpo_1v3_uniform.py']

# Override to 2 dimensions only
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
    target_min_gap=0.30,
    other_min_gap=0.00,
    other_max_gap=0.40,
    loser_balance_lambda=0.50,
    min_total_gap=0.20,
    strict_all_dims=True,
    require_distinct_losers=True,
    min_group_images=6,
    min_reward=None,
)

# B200 has 8x192GB VRAM — keep ref on GPU
ref_on_cpu = False
checkpoints_total_limit = None

# Hardness + timestep gated alpha
md3po_alpha_mode = 'hardness_timestep'
md3po_tau_h = 0.20
md3po_timestep_split = 0.55
md3po_gate_eps = 1e-2
md3po_detach_alpha = True

# Run isolation
model_output_dir = os.environ.get('SCIFORMA_DATA_ROOT', '') + '/experiments/260409_md3po_1v2_ct_htimestep_b2008'
tracker_run_name = 'md3po_1v2_ct_htimestep_b2008'
resume_from_checkpoint = None

max_train_steps = 10000
checkpointing_steps = 500
validation_steps = 500
