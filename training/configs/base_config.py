# ====== 通用运行设置（通常不变/默认） ======
seed = 42
device = 'cuda'
dtype = 'float32'

# ====== 模型/资源通用 ======
revision = None
variant = None

bnb_quantization_config_path = None

# ====== Model Type (for model factory) ======
# Supported: 'QwenImage', 'Flux2Klein'
# This determines which model classes to load (VAE, Transformer, Pipeline, etc.)
model_type = 'QwenImage'

# Transformer 架构
transformer_cfg = dict(
    type='QwenImageTransformer2DModel',
)

# ====== 模型权重与结构 ======
pretrained_model_name_or_path = (
    "Qwen/Qwen-Image-2512"
)
# HuggingFace token will be read from HF_TOKEN environment variable at runtime
# Set via: export HF_TOKEN=your_token_here
huggingface_token = None

# LoRA 通用默认
use_lora = True
lora_layers = "to_k,to_q,to_v"
rank = 64
lora_alpha = 4
lora_dropout = 0.0
layer_weighting = 5.0

# VAE 架构默认
pos_embedding = 'rope'
decoder_arch = 'vit'


# 分辨率与增强
use_parquet_dataset = False

# 训练规模与优化器默认
train_batch_size = 1
num_train_epochs = 1
max_train_steps = 80000
gradient_accumulation_steps = 1
dataloader_num_workers = 4  # Number of DataLoader workers
gradient_checkpointing = True
cache_latents = False      # --cache_latents

optimizer = "AdamW"
use_8bit_adam = False
learning_rate = 2e-4
lr_scheduler = "constant"
lr_warmup_steps = 500
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_weight_decay = 1e-4
adam_epsilon = 1e-08
max_grad_norm = 1.0

prodigy_beta3 = None
prodigy_decouple = True
prodigy_use_bias_correction = True
prodigy_safeguard_warmup = True


checkpointing_steps = 500
resume_from_checkpoint = "latest"
checkpoints_total_limit = None

# EMA (Exponential Moving Average) settings
use_ema = False           # Whether to use EMA model
ema_decay = 0.9999        # EMA decay rate
ema_update_after_step = 0 # Start EMA after this step
ema_steps = 100           # Update EMA every N steps

# 混合精度
mixed_precision = "bf16"
allow_tf32 = False
upcast_before_saving = False
offload = False

#可视化
report_to = "wandb"
push_to_hub = False
hub_token = None
hub_model_id = None
cache_dir = None

# 验证/推理默认
scale_lr = False           # 等价 CLI: --scale_lr
lr_num_cycles = 1          # --lr_num_cycles
lr_power = 1.0             # --lr_power

weighting_scheme = "none"  #choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"], help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
logit_mean = 0.0                      # --logit_mean
logit_std = 1.0                       # --logit_std
mode_scale = 1.29                     # --mode_scale

validation_guidance_scale = 1.0

