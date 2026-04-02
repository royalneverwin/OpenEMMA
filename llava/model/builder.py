#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import shutil
import warnings

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
from llava.model import *
from llava.model.multimodal_projector.builder import build_vision_projector
from llava.model.quantization import enable_custom_bitsandbytes


enable_custom_bitsandbytes()


def _load_torch_weights(model_path, filename):
    local_path = os.path.join(model_path, filename)
    if os.path.exists(local_path):
        return torch.load(local_path, map_location="cpu")

    from huggingface_hub import hf_hub_download

    cache_file = hf_hub_download(repo_id=model_path, filename=filename)
    return torch.load(cache_file, map_location="cpu")


def _load_mm_projector_weights(model_path):
    mm_projector_weights = _load_torch_weights(model_path, "mm_projector.bin")
    return {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}


def _infer_module_device(module):
    for parameter in module.parameters(recurse=True):
        return parameter.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _restore_fp16_mm_projector(model, model_path):
    try:
        mm_projector_weights = _load_mm_projector_weights(model_path)
    except Exception as exc:
        warnings.warn(
            f"Unable to restore fp16 mm_projector from `{model_path}`. "
            f"Keeping the quantized projector instead. Details: {exc}"
        )
        return

    mm_projector_device = _infer_module_device(model.get_model().mm_projector)
    rebuilt_projector = build_vision_projector(model.config).to(device=mm_projector_device, dtype=torch.float16)
    model.get_model().mm_projector = rebuilt_projector
    model.load_state_dict(mm_projector_weights, strict=False)


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    use_flash_attn=False,
    **kwargs,
):
    visual_token_num = kwargs.pop("visual_token_num", None)

    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}
    resolved_device_map = kwargs["device_map"]

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    llava_kwargs = dict(kwargs)
    if visual_token_num is not None:
        llava_kwargs["visual_token_num"] = visual_token_num

    if "llava" in model_name.lower():
        if "lora" in model_name.lower() and model_base is None:
            warnings.warn(
                "There is `lora` in model name but no `model_base` is provided. "
                "If you are loading a LoRA model, please provide the `model_base` argument. "
                "Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged."
            )
        if "lora" in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig

            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print("Loading LLaVA from base model...")
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_base,
                low_cpu_mem_usage=True,
                config=lora_cfg_pretrained,
                **llava_kwargs,
            )
            token_num, token_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(
                    torch.empty(token_num, token_dim, device=model.device, dtype=model.dtype)
                )
                model.model.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(token_num, token_dim, device=model.device, dtype=model.dtype)
                )

            print("Loading additional LLaVA weights...")
            non_lora_trainables = _load_torch_weights(model_path, "non_lora_trainables.bin")
            non_lora_trainables = {
                (k[11:] if k.startswith("base_model.") else k): v
                for k, v in non_lora_trainables.items()
            }
            if any(k.startswith("model.model.") for k in non_lora_trainables):
                non_lora_trainables = {
                    (k[6:] if k.startswith("model.") else k): v
                    for k, v in non_lora_trainables.items()
                }
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel

            print("Loading LoRA weights...")
            model = PeftModel.from_pretrained(model, model_path)
            print("Merging LoRA weights...")
            model = model.merge_and_unload()
            print("Model is loaded...")
        elif model_base is not None:
            print("Loading LLaVA from base model...")
            if "mpt" in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, "configuration_mpt.py")):
                    shutil.copyfile(
                        os.path.join(model_base, "configuration_mpt.py"),
                        os.path.join(model_path, "configuration_mpt.py"),
                    )
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(
                    model_base,
                    low_cpu_mem_usage=True,
                    config=cfg_pretrained,
                    **llava_kwargs,
                )
            elif "mistral" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_base)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_base,
                    low_cpu_mem_usage=True,
                    config=cfg_pretrained,
                    **llava_kwargs,
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_base,
                    low_cpu_mem_usage=True,
                    config=cfg_pretrained,
                    **llava_kwargs,
                )

            model.load_state_dict(_load_mm_projector_weights(model_path), strict=False)
        else:
            if "mpt" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **llava_kwargs,
                )
            elif "mistral" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    sliding_window=4096,
                    **llava_kwargs,
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **llava_kwargs,
                )

            if load_8bit or load_4bit:
                _restore_fp16_mm_projector(model, model_path)
    else:
        if model_base is not None:
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print("Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.float16)
        else:
            if "mpt" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    **kwargs,
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs,
                )

    image_processor = None

    if "llava" in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=resolved_device_map)
        if visual_token_num is not None and hasattr(vision_tower, "load_text_tower"):
            vision_tower.load_text_tower(device_map=resolved_device_map)
        if isinstance(resolved_device_map, str) and resolved_device_map != "auto":
            vision_tower.to(device=resolved_device_map, dtype=torch.float16)
        elif isinstance(resolved_device_map, dict) and device != "cuda":
            vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
