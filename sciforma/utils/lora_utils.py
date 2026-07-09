import os.path as osp
import torch

from accelerate.logging import get_logger
from diffusers import QwenImagePipeline, FluxPipeline
from diffusers.utils import convert_unet_state_dict_to_peft
from diffusers.training_utils import cast_training_params, _set_state_dict_into_text_encoder
from peft import LoraConfig
from peft import set_peft_model_state_dict

from .general_util_funcs import get_module_recursively

logger = get_logger(__name__)

def add_lora_and_load_ckpt_to_models(
    config,
    transformer,
    text_encoder_one,
    text_encoder_two=None,
    pipeline_cls=QwenImagePipeline,
    filesystem=None,
):
    def strip_transformer_prefix(lora_state_dict):
        new_state_dict = {}
        for k, v in lora_state_dict.items():
            if k.startswith("transformer."):
                new_key = k[len("transformer."):]  # 去掉前缀
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    if config.pretrained_full_ckpt_path is not None:
        raise NotImplementedError('loading full fine-tuned checkpoint is not supported for lora training now.')

    if config.pretrained_lora_ckpt_path is not None and config.fuse_pretrained_lora:
        assert not config.train_text_encoder, 'fusing lora to text encoder is not supported currently'
        if isinstance(config.pretrained_lora_ckpt_path, str):
            pretrained_lora_ckpt_path_list = [config.pretrained_lora_ckpt_path]
        else:
            pretrained_lora_ckpt_path_list = config.pretrained_lora_ckpt_path  ## here, fuse multiple lora
        for pretrained_lora_ckpt in pretrained_lora_ckpt_path_list:

            if filesystem:
                from ray.train import Checkpoint
                lora_checkpoint = Checkpoint(
                    path=pretrained_lora_ckpt.rstrip("/"),  # e.g. s3://bucket/prefix/
                    filesystem=filesystem,
                )
                with lora_checkpoint.as_directory() as lora_local_checkpoint_dir:

                    lora_state_dict, metadata = pipeline_cls.lora_state_dict(lora_local_checkpoint_dir, return_lora_metadata=True)
            else:
                lora_state_dict, metadata = pipeline_cls.lora_state_dict(pretrained_lora_ckpt, return_lora_metadata=True)

            lora_state_dict = strip_transformer_prefix(lora_state_dict)
            print("metadata: ", metadata)
            if pipeline_cls is QwenImagePipeline:
                print("qwen load lora")
                pipeline_cls.load_lora_into_transformer(lora_state_dict, transformer, metadata = metadata)

            else:
                pipeline_cls.load_lora_into_transformer(
                    state_dict=lora_state_dict,
                    network_alphas=None,
                    transformer=transformer,
                )

            print("LoRA params just after load:")
            for name, _ in transformer.named_parameters():
                if "lora" in name.lower():
                    print(" ", name)

            module_path = "transformer_blocks.15.attn.add_v_proj" # for qwen
            mod = get_module_recursively(transformer, module_path)
            w = mod.weight.detach().float()
            before = w.norm().item()
            print(f"[CHECK] Before fuse: {module_path} weight norm = {before:.6f}")

            transformer.fuse_lora(safe_fusing=True)

            w_after = mod.weight.detach().float()
            after = w_after.norm().item()
            print(f"[CHECK] After fuse : {module_path} weight norm = {after:.6f}")


            transformer.unload_lora()

            if abs(after - before) < 1e-6:
                print("[CHECK] ⚠️ 权重没变化，看起来 LoRA 没有真正生效")
            else:
                print("[CHECK] ✅ 权重发生变化，LoRA 成功融合进 transformer")

            logger.info(f"[INFO] loaded pretrained lora weights from {pretrained_lora_ckpt} and fused the lora to the base model")
            print(f"[INFO] loaded pretrained lora weights from {pretrained_lora_ckpt} and fused the lora to the base model")


import torch

class DummyImagePipeline(QwenImagePipeline):
    def __init__(self, transformer, scheduler=None, vae=None, text_encoder=None, tokenizer=None):
        super().__init__(transformer=transformer, scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer)
        # 注册 transformer 作为 pipeline 的模块
        # 如果有 scheduler，也注册
        modules = {"transformer": transformer}
        self.register_modules(**modules)
