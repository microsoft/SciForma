"""
Flux2Klein Full Fine-tuning Validation Function

For ZeRO-2: No weight gathering needed - each GPU has complete model weights.
Simply move VAE and text_encoder to GPU, run inference, move back.
"""

import gc
import os
import torch
import wandb
from PIL import Image

from accelerate.logging import get_logger
from diffusers.utils.torch_utils import is_compiled_module

from sciforma.registry import VALIDATION_FUNCS
from sciforma.utils.model_factory import get_pipeline_class

logger = get_logger(__name__)


def save_images_to_disk(images, save_dir, global_step, prompt_idx, prompt_text):
    """Save validation images to local disk."""
    os.makedirs(save_dir, exist_ok=True)
    
    safe_prompt = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in prompt_text[:50])
    safe_prompt = safe_prompt.strip().replace(' ', '_')
    
    saved_paths = []
    
    if isinstance(images, list):
        for i, img in enumerate(images):
            filename = f"step{global_step:06d}_prompt{prompt_idx:02d}_{i}_{safe_prompt}.png"
            filepath = os.path.join(save_dir, filename)
            img.save(filepath)
            saved_paths.append(filepath)
    else:
        filename = f"step{global_step:06d}_prompt{prompt_idx:02d}_{safe_prompt}.png"
        filepath = os.path.join(save_dir, filename)
        images.save(filepath)
        saved_paths.append(filepath)
    
    return saved_paths


@VALIDATION_FUNCS.register_module()
def Flux2Klein_fulltune_validation_func(
    vae,
    transformer,
    text_encoder,
    tokenizer,
    scheduler,
    accelerator,
    args,
    global_step=0,
):
    """Validation function for Flux2Klein full fine-tuning.
    
    For ZeRO-2: Each GPU has complete weights, so we can directly use them.
    Just move VAE and text_encoder to GPU for inference, then back to CPU.
    """
    if not accelerator.is_main_process:
        return []
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Running validation at step {global_step}...")
    logger.info(f"{'='*60}")
    
    PipelineClass = get_pipeline_class('Flux2Klein')
    
    # Unwrap transformer
    unwrapped_transformer = accelerator.unwrap_model(transformer)
    if is_compiled_module(unwrapped_transformer):
        unwrapped_transformer = unwrapped_transformer._orig_mod
    
    unwrapped_transformer.eval()
    
    # Prepare output directory
    validation_output_dir = os.path.join(
        args.model_output_dir, 
        "validation_images",
        f"step_{global_step:06d}"
    )
    os.makedirs(validation_output_dir, exist_ok=True)
    
    image_logs = []
    all_saved_paths = []
    
    validation_prompts = getattr(args, 'validation_prompts', None)
    if validation_prompts is None:
        validation_prompts = [
            "A beautiful sunset over the ocean with golden clouds",
            "A scientific diagram showing the structure of a cell",
        ]
        args.resolution_list = [(1024, 1024), (1024, 1024)]
    
    # Determine dtype
    if args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32
    
    # Save original devices
    original_vae_device = next(vae.parameters()).device
    original_te_device = next(text_encoder.parameters()).device
    
    try:
        # Move VAE and text_encoder to GPU for inference
        logger.info("  Moving VAE and text_encoder to GPU...")
        vae.to(accelerator.device)
        text_encoder.to(accelerator.device)
        
        # Create pipeline with all components
        pipeline = PipelineClass(
            vae=vae,
            transformer=unwrapped_transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )
        
        pipeline = pipeline.to(accelerator.device)
        pipeline = pipeline.to(dtype=weight_dtype)
        
        # Generate images
        for prompt_idx, prompt in enumerate(validation_prompts):
            logger.info(f"  Generating image {prompt_idx + 1}/{len(validation_prompts)}: {prompt[:80]}...")
            
            if hasattr(args, 'resolution_list') and prompt_idx < len(args.resolution_list):
                width, height = args.resolution_list[prompt_idx]
            else:
                width, height = 1024, 1024
            
            generator = None
            if hasattr(args, 'seed') and args.seed is not None:
                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed + prompt_idx)
            
            pipeline_kwargs = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": getattr(args, 'num_inference_steps', 28),
                "guidance_scale": getattr(args, 'validation_guidance_scale', 3.5),
                "generator": generator,
                "output_type": "pil",
            }
            
            if hasattr(args, 'max_sequence_length') and args.max_sequence_length:
                pipeline_kwargs["max_sequence_length"] = args.max_sequence_length
            
            try:
                with torch.no_grad():
                    with torch.amp.autocast(accelerator.device.type, dtype=weight_dtype):
                        output = pipeline(**pipeline_kwargs)
                        images = output.images
            except Exception as e:
                logger.error(f"  Error generating image: {e}")
                import traceback
                logger.error(traceback.format_exc())
                images = [Image.new('RGB', (width, height), color='gray')]
            
            saved_paths = save_images_to_disk(
                images=images,
                save_dir=validation_output_dir,
                global_step=global_step,
                prompt_idx=prompt_idx,
                prompt_text=prompt,
            )
            all_saved_paths.extend(saved_paths)
            
            image_logs.append({
                "images": images,
                "prompt": prompt,
                "paths": saved_paths,
            })
            
            logger.info(f"    Saved to: {saved_paths[0]}")
        
        # Cleanup pipeline
        del pipeline
        
    finally:
        # Move VAE and text_encoder back to original devices
        logger.info("  Moving VAE and text_encoder back to CPU...")
        vae.to(original_vae_device)
        text_encoder.to(original_te_device)
    
    # Log to WandB - use fixed keys so images show progress over steps
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            dict_to_log = {}
            for sample_idx, log in enumerate(image_logs):
                images = log["images"]
                prompt = log["prompt"]
                
                # Use fixed key names (validation/sample_X) so WandB shows timeline progression
                if isinstance(images, list):
                    for i, img in enumerate(images):
                        dict_to_log[f"validation/sample_{sample_idx}_{i}"] = wandb.Image(
                            img, caption=f"[Step {global_step}] {prompt[:120]}..."
                        )
                else:
                    dict_to_log[f"validation/sample_{sample_idx}"] = wandb.Image(
                        images, caption=f"[Step {global_step}] {prompt[:120]}..."
                    )
            
            tracker.log(dict_to_log, step=global_step)
    
    logger.info(f"\n  ✅ Validation complete! Saved {len(all_saved_paths)} images to:")
    logger.info(f"     {validation_output_dir}")
    logger.info(f"{'='*60}\n")
    
    unwrapped_transformer.train()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return image_logs


@VALIDATION_FUNCS.register_module()
def Flux2Klein_fulltune_validation_func_parquet(
    vae,
    transformer,
    text_encoder,
    tokenizer,
    scheduler,
    accelerator,
    args,
    global_step=0,
):
    """Validation function for Flux2Klein full fine-tuning with parquet dataset.
    
    Same as main function - for ZeRO-2, we just move models to GPU and run.
    """
    if not accelerator.is_main_process:
        return []
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Running validation at step {global_step} (parquet mode)...")
    logger.info(f"{'='*60}")
    
    image_logs = Flux2Klein_fulltune_validation_func(
        vae=vae,
        transformer=transformer,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        accelerator=accelerator,
        args=args,
        global_step=global_step,
    )
    
    return image_logs
