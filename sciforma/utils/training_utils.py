import math

import torch

from diffusers.utils.torch_utils import is_compiled_module

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def get_sigmas(timesteps, device, noise_scheduler_copy, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(device)
    timesteps = timesteps.to(device)
    
    # OPTIMIZED: Vectorized indexing instead of Python list comprehension
    # This avoids GPU-CPU sync that was causing ~2s overhead per step
    # Use searchsorted for O(log n) lookup instead of O(n) comparison per timestep
    # schedule_timesteps is sorted in descending order, so we need to handle that
    sorted_schedule = schedule_timesteps.flip(0)  # ascending order
    sorted_indices = torch.searchsorted(sorted_schedule, timesteps)
    # Convert back to descending order indices  
    num_steps = schedule_timesteps.shape[0]
    step_indices = num_steps - 1 - sorted_indices

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma    


def get_trainable_params(config, accelerator, transformer, text_encoder_one, text_encoder_two=None):

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {
        "params": transformer_lora_parameters, 
        "lr": config.learning_rate,
    }
    params_to_optimize = [transformer_parameters_with_lr]

    if config.train_text_encoder:
        text_encoder_one_params = list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
        text_parameters_one_with_lr = {
            "params": text_encoder_one_params,
            "weight_decay": config.adam_weight_decay_text_encoder,
            "lr": config.text_encoder_lr if config.text_encoder_lr else config.learning_rate,
        }
        params_to_optimize.append(text_parameters_one_with_lr)
        
        if text_encoder_two is not None:
            text_encoder_two_params = list(filter(lambda p: p.requires_grad, text_encoder_two.parameters()))
            text_parameters_two_with_lr = {
                "params": text_encoder_two_params,
                "weight_decay": config.adam_weight_decay_text_encoder,
                "lr": config.text_encoder_lr if config.text_encoder_lr else config.learning_rate,
            }
            params_to_optimize.append(text_parameters_two_with_lr)
        else:
            text_encoder_two_params = None
    
    if accelerator.is_main_process:
        with torch.no_grad():
            for i, param_set in enumerate(params_to_optimize):
                num_params = sum([p.numel() for p in param_set["params"]]) / 1e+6
                print(f"Trainable Params Set {i}: {num_params:02f}M")
    
    if not config.train_text_encoder:
        return params_to_optimize, transformer_lora_parameters
    else:
        return params_to_optimize, transformer_lora_parameters, text_encoder_one_params, text_encoder_two_params

def compute_density_for_timestep_sampling(
    weighting_scheme: str, 
    batch_size: int, 
    logit_mean: float = None, 
    logit_std: float = None, 
    mode_scale: float = None,
    u_value_min: float = None,
    u_value_max: float = None,
):
    """
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    elif weighting_scheme == "uniform_in_range":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = u_value_min + (u_value_max - u_value_min) * u
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u
