import os
"""
MD3PO 1v3 uniform alpha — B200 x8 GPU.

Tuple construction per group:
  winner + {loser_component, loser_text, loser_arrow}

Loss:
  mean_b[ (1/3) * sum_d -logsigmoid(beta * logits_d) ]
"""

_base_ = ['../_dpo_base.py']

# Dataset: dimension-anchored 1v3 tuples from v9r grouped parquet
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

    target_dims=('component_score', 'text_score', 'arrow_score'),
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

train_iteration_func = 'Flux2Klein_md3po_train_iteration'

# B200 has 8x192GB VRAM — keep ref on GPU
ref_on_cpu = False
checkpoints_total_limit = None

# Alpha weighting
md3po_alpha_mode = 'uniform'
md3po_tau_h = 0.20
md3po_detach_alpha = True

# Run isolation
model_output_dir = os.environ.get('SCIFORMA_DATA_ROOT', '') + '/experiments/260409_md3po_1v3_uniform_b2008'
tracker_run_name = 'md3po_1v3_uniform_b2008'
resume_from_checkpoint = None

max_train_steps = 10000
checkpointing_steps = 500
validation_steps = 500
