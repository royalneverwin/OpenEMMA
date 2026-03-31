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


from typing import List, Optional, Tuple, Union

import torch

from transformers import AutoConfig, AutoModelForCausalLM, MptConfig, MptForCausalLM, MptModel
from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaMptConfig(MptConfig):
    model_type = "llava_mpt"


class LlavaMptModel(LlavaMetaModel, MptModel):
    config_class = LlavaMptConfig

    def __init__(self, config: MptConfig):
        config.hidden_size = config.d_model
        super(LlavaMptModel, self).__init__(config)

    def embed_tokens(self, x):
        return self.wte(x)


class LlavaMptForCausalLM(MptForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaMptConfig
    supports_gradient_checkpointing = True

    def __init__(self, config, visual_token_num: Optional[int] = None):
        super(MptForCausalLM, self).__init__(config)

        self.transformer = LlavaMptModel(config)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.visual_token_num = visual_token_num
        self.last_visual_token_num = None

        self.post_init()

    def get_model(self):
        return self.transformer

    def get_visual_token_num(self):
        return self.visual_token_num

    def get_last_visual_token_num(self):
        return self.last_visual_token_num

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlavaMptModel):
            module.gradient_checkpointing = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        texts: Optional[Union[str, List[str]]] = None,
        add_quant: bool = False,
        alpha: float = 0.7,
        dynamic_alpha: bool = False,
        quant_method: str = "l2_norm",
        pruning_method: str = "cdpruner",
    ):

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes=image_sizes,
                texts=texts,
                add_quant=add_quant,
                alpha=alpha,
                dynamic_alpha=dynamic_alpha,
                quant_method=quant_method,
                pruning_method=pruning_method,
            )
        del position_ids

        return super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        texts: Optional[Union[str, List[str]]] = None,
        add_quant: bool = False,
        alpha: float = 0.7,
        dynamic_alpha: bool = False,
        quant_method: str = "l2_norm",
        pruning_method: str = "cdpruner",
        return_visual_token_num: bool = False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                texts=texts,
                add_quant=add_quant,
                alpha=alpha,
                dynamic_alpha=dynamic_alpha,
                quant_method=quant_method,
                pruning_method=pruning_method,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
            self.last_visual_token_num = 0

        outputs = super().generate(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        del position_ids
        if return_visual_token_num:
            return outputs, self.last_visual_token_num
        return outputs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_mpt", LlavaMptConfig)
AutoModelForCausalLM.register(LlavaMptConfig, LlavaMptForCausalLM)
