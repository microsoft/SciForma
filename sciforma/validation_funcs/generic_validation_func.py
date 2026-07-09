"""
Generic Validation Function for SciForma

This validation function works with any model type by dynamically
resolving the Pipeline class from the model factory.

Supported models:
- QwenImage: Uses QwenImagePipeline
- Flux2Klein: Uses Flux2KleinPipeline (Qwen3 text encoder, single encoder)
"""

import gc
import torch
import wandb
from PIL import Image

from accelerate.logging import get_logger

from sciforma.registry import VALIDATION_FUNCS
from sciforma.utils.model_factory import get_pipeline_class

logger = get_logger(__name__)


@VALIDATION_FUNCS.register_module()
def generic_validation_func(
    vae,
    transformer,
    text_encoder,
    accelerator,
    scheduler,
    tokenizer,
    args,
    pipeline_class=None,  # Optional: pass pipeline class explicitly
):
    """
    Generic validation function that works with any model type.
    
    Args:
        vae: VAE model
        transformer: Transformer/DiT model
        text_encoder: Text encoder model
        accelerator: Accelerate accelerator
        scheduler: Noise scheduler
        tokenizer: Tokenizer
        args: Config/args object with:
            - model_type: str (e.g., 'QwenImage', 'Flux2Klein')
            - mixed_precision: str
            - validation_prompts: list
            - resolution_list: list of tuples
            - num_inference_steps: int
            - true_cfg_scale: float (optional)
            - negative_prompt: str (optional)
            - seed: int (optional)
            - max_sequence_length: int (optional)
        pipeline_class: Optional pipeline class override
    """
    # 1. Resolve Pipeline class dynamically
    if pipeline_class is None:
        model_type = getattr(args, 'model_type', 'QwenImage')
        PipelineClass = get_pipeline_class(model_type)
        logger.info(f"Using pipeline class: {PipelineClass.__name__} for model type: {model_type}")
    else:
        PipelineClass = pipeline_class
    
    # 2. Initialize Pipeline
    pipeline = PipelineClass(
        vae=vae,
        transformer=accelerator.unwrap_model(transformer),
        text_encoder=accelerator.unwrap_model(text_encoder),
        scheduler=scheduler,
        tokenizer=tokenizer,
    )

    logger.info(f"Running validation...")
    pipeline = pipeline.to(accelerator.device)
    
    # Set precision
    if args.mixed_precision == "fp16":
        pipeline.to(dtype=torch.float16)
    elif args.mixed_precision == "bf16":
        pipeline.to(dtype=torch.bfloat16)

    image_logs = []

    # 3. Iterate through validation prompts
    for validation_prompt_idx in range(len(args.validation_prompts)):
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if getattr(args, 'seed', None) is not None else None

        validation_prompt = args.validation_prompts[validation_prompt_idx]
        
        # Build pipeline kwargs (model-specific parameters)
        pipeline_kwargs = {
            "prompt": validation_prompt,
            "height": args.resolution_list[validation_prompt_idx][1],
            "width": args.resolution_list[validation_prompt_idx][0],
            "num_inference_steps": getattr(args, 'num_inference_steps', 28),
            "generator": generator,
            "output_type": "pil",
        }
        
        # Add optional parameters if they exist
        if hasattr(args, 'true_cfg_scale') and args.true_cfg_scale is not None:
            pipeline_kwargs["true_cfg_scale"] = args.true_cfg_scale
        if hasattr(args, 'guidance_scale') and args.guidance_scale is not None:
            pipeline_kwargs["guidance_scale"] = args.guidance_scale
        if hasattr(args, 'negative_prompt') and args.negative_prompt:
            pipeline_kwargs["negative_prompt"] = args.negative_prompt
        if hasattr(args, 'max_sequence_length') and args.max_sequence_length:
            pipeline_kwargs["max_sequence_length"] = args.max_sequence_length

        with torch.no_grad():
            with torch.amp.autocast(
                accelerator.device.type,
                dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
            ):
                try:
                    images = pipeline(**pipeline_kwargs).images
                except TypeError as e:
                    # Some models may not support all kwargs, remove unsupported ones
                    logger.warning(f"Pipeline call failed with: {e}. Retrying with minimal kwargs...")
                    minimal_kwargs = {
                        "prompt": validation_prompt,
                        "height": args.resolution_list[validation_prompt_idx][1],
                        "width": args.resolution_list[validation_prompt_idx][0],
                        "num_inference_steps": getattr(args, 'num_inference_steps', 28),
                        "generator": generator,
                    }
                    images = pipeline(**minimal_kwargs).images

        image_logs.append({
            "images": images,
            "caption": validation_prompt,
        })

    # 4. WandB logging
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            dict_to_log = {}
            for sample_idx, log in enumerate(image_logs):
                images = log["images"]
                caption = log["caption"]

                if isinstance(images, list):
                    for i, img in enumerate(images):
                        dict_to_log[f"sample_{sample_idx}_{i}"] = wandb.Image(
                            img, caption=f"{caption[:100]}..."
                        )
                else:
                    dict_to_log[f"sample_{sample_idx}"] = wandb.Image(
                        images, caption=f"{caption[:100]}..."
                    )

            tracker.log(dict_to_log, commit=False)

    # 5. Clean up
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return image_logs
