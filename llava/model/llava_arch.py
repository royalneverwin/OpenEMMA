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


from abc import ABC, abstractmethod
from typing import List, Union

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


def compress_by_vit_attn_v3(
    x: torch.Tensor,
    attn_weights: torch.Tensor,
    quant_sensitivity: torch.Tensor = None,
    alpha: float = 0.7,
    target_dim: Union[int, List[int]] = 128,
    dominant_token_num: int = 128,
    keep_order: bool = True,
):
    del keep_order

    batch_size, seq_len, _ = x.shape

    global_target_dim = target_dim if isinstance(target_dim, int) else max(target_dim)
    local_target_dim = target_dim if isinstance(target_dim, int) else min(target_dim)
    assert global_target_dim == local_target_dim or global_target_dim == seq_len, (
        f"global_target_dim:{global_target_dim} must be equal to "
        f"local_target_dim:{local_target_dim} or original token num: {seq_len}"
    )
    assert local_target_dim >= dominant_token_num, (
        f"dominant_token_num:{dominant_token_num} cannot be greater than "
        f"the minimum target_dim:{target_dim}"
    )

    attn_sum = attn_weights.mean(1)

    if quant_sensitivity is not None:
        attn_min, attn_max = attn_sum.min(), attn_sum.max()
        attn_sum = (attn_sum - attn_min) / (attn_max - attn_min + 1e-8)

        quant_min, quant_max = quant_sensitivity.min(), quant_sensitivity.max()
        quant_sensitivity = (quant_sensitivity - quant_min + 1e-8) / (quant_max - quant_min + 1e-8)

        attn_sum = alpha * attn_sum + (1 - alpha) * quant_sensitivity

    topk_indices = attn_sum.topk(dominant_token_num, dim=1).indices
    batch_indices = torch.arange(batch_size, device=x.device)[:, None]

    prune_mask = torch.zeros(batch_size, seq_len, device=x.device, dtype=torch.bool)
    prune_mask[batch_indices, topk_indices] = True

    contextual_num = local_target_dim - dominant_token_num
    if contextual_num > 0:
        non_dominant_token_mask = ~prune_mask
        non_dominant_token_num = seq_len - dominant_token_num

        original_indices = torch.arange(seq_len, device=x.device).expand(batch_size, -1)
        non_dominant_indices = original_indices[non_dominant_token_mask].reshape(batch_size, non_dominant_token_num)

        step = max(1, non_dominant_token_num // contextual_num)
        contextual_indices_in_nondominant = torch.arange(
            0,
            non_dominant_token_num,
            step,
            device=x.device,
        )[:contextual_num]

        contextual_original_indices = torch.gather(
            non_dominant_indices,
            1,
            contextual_indices_in_nondominant.expand(batch_size, -1),
        )
        prune_mask[batch_indices, contextual_original_indices] = True

    if isinstance(target_dim, list):
        for i in range(batch_size):
            if target_dim[i] == global_target_dim:
                prune_mask[i] = True
    return prune_mask


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(
        self,
        images,
        texts=None,
        add_quant=False,
        alpha=0.7,
        dynamic_alpha=False,
        quant_method="l2_norm",
        pruning_method="cdpruner",
    ):
        if getattr(self, "visual_token_num", None) is not None and texts is not None:
            if pruning_method == "visionzip":
                image_features, image_embeds, text_embeds, attentions = self.get_model().get_vision_tower()(
                    images,
                    texts=texts,
                    output_attentions=True,
                )
                batch_size, seq_len, hidden_dim = image_features.shape
                del hidden_dim

                image_features = self.get_model().mm_projector(image_features)
                attn_weights = attentions[-1][:, :, 1:, 1:].mean(1)

                if add_quant:
                    if quant_method == "quant_error_group":
                        group_size = 128
                        x_grouped = image_features.view(batch_size, seq_len, image_features.shape[-1] // group_size, group_size)
                        scale_grouped = x_grouped.abs().max(dim=-1, keepdim=True)[0] / 7.0
                        x_q_grouped = torch.round(x_grouped / (scale_grouped + 1e-8)) * scale_grouped
                        x_q = x_q_grouped.view(batch_size, seq_len, image_features.shape[-1])
                        quant_sensitivity = torch.norm(image_features - x_q, dim=-1)
                    else:
                        raise ValueError(f"Unknown quant_method: {quant_method}")
                else:
                    quant_sensitivity = None

                index_masks = compress_by_vit_attn_v3(
                    image_features,
                    attn_weights,
                    quant_sensitivity=quant_sensitivity,
                    alpha=alpha,
                    target_dim=self.visual_token_num,
                    dominant_token_num=self.visual_token_num,
                )
                return image_features, index_masks

            if pruning_method == "cdpruner":
                image_features, image_embeds, text_embeds = self.get_model().get_vision_tower()(images, texts=texts)

                batch_size, seq_len, _ = image_features.shape
                image_features = self.get_model().mm_projector(image_features)
                device = image_features.device

                image_normalized = image_features / image_features.norm(dim=-1, keepdim=True)
                image_normalized = image_normalized.float()
                similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2))

                image_embeds = image_embeds.to(device=device, dtype=torch.float32)
                text_embeds = text_embeds.to(device=device, dtype=torch.float32)
                if text_embeds.ndim == 1:
                    text_embeds = text_embeds.unsqueeze(0)
                if text_embeds.shape[0] == 1 and batch_size > 1:
                    text_embeds = text_embeds.expand(batch_size, -1)
                elif text_embeds.shape[0] != batch_size:
                    print(
                        "QAPruner text/image batch mismatch: "
                        f"text_embeds.shape={tuple(text_embeds.shape)}, "
                        f"image_embeds.shape={tuple(image_embeds.shape)}, "
                        f"image_features.shape={tuple(image_features.shape)}, "
                        f"batch_size={batch_size}, texts={texts}"
                    )
                    raise ValueError(
                        "Text embedding batch size does not match image batch size: "
                        f"{text_embeds.shape[0]} vs {batch_size}"
                    )
                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
                relevance = -torch.einsum("bsd,bd->bs", image_embeds, text_embeds)
                relevance_min = relevance.amin(dim=-1, keepdim=True)
                relevance_max = relevance.amax(dim=-1, keepdim=True)
                relevance = (relevance - relevance_min + 1e-6) / (relevance_max - relevance_min + 1e-8)

                if add_quant:
                    if quant_method == "l2_norm":
                        quant_sensitivity = image_features.norm(dim=-1)
                    elif quant_method == "l1_norm":
                        quant_sensitivity = image_features.abs().mean(dim=-1)
                    elif quant_method == "l_inf":
                        quant_sensitivity = image_features.abs().max(dim=-1)[0]
                    elif quant_method == "variance":
                        quant_sensitivity = image_features.var(dim=-1)
                    elif quant_method == "l2":
                        quant_sensitivity = (image_features ** 2).sum(dim=-1)
                    elif quant_method == "quant_error":
                        scale = image_features.abs().max(dim=-1, keepdim=True)[0] / 7.0
                        x_q = torch.round(image_features / (scale + 1e-8)) * scale
                        quant_sensitivity = torch.norm(image_features - x_q, dim=-1)
                    elif quant_method == "quant_error_clip":
                        topk_keep = int(image_features.shape[-1] * 0.99)
                        clip_val = torch.topk(image_features.abs(), topk_keep, dim=-1).values[..., -1:]
                        scale = clip_val / 7.0
                        x_q = torch.round(
                            image_features.clamp(min=-clip_val, max=clip_val) / (scale + 1e-8)
                        ) * scale
                        quant_sensitivity = torch.norm(image_features - x_q, dim=-1)
                    elif quant_method == "quant_error_group":
                        group_size = 128
                        x_grouped = image_features.view(batch_size, seq_len, image_features.shape[-1] // group_size, group_size)
                        scale_grouped = x_grouped.abs().max(dim=-1, keepdim=True)[0] / 7.0
                        x_q_grouped = torch.round(x_grouped / (scale_grouped + 1e-8)) * scale_grouped
                        x_q = x_q_grouped.view(batch_size, seq_len, image_features.shape[-1])
                        quant_sensitivity = torch.norm(image_features - x_q, dim=-1)
                    elif quant_method == "dynamic_range":
                        quant_sensitivity = image_features.max(dim=-1)[0] - image_features.min(dim=-1)[0]
                    elif quant_method == "complex":
                        group_size = 128
                        x_grouped = image_features.view(batch_size, seq_len, image_features.shape[-1] // group_size, group_size)
                        scale_grouped = x_grouped.abs().max(dim=-1, keepdim=True)[0] / 7.0
                        x_q_grouped = torch.round(x_grouped / (scale_grouped + 1e-8)) * scale_grouped
                        x_q = x_q_grouped.view(batch_size, seq_len, image_features.shape[-1])
                        err_group = torch.norm(image_features - x_q, dim=-1)

                        dyn_range = image_features.max(dim=-1)[0] - image_features.min(dim=-1)[0]

                        err_min, err_max = err_group.min(dim=-1, keepdim=True)[0], err_group.max(dim=-1, keepdim=True)[0]
                        err_norm = (err_group - err_min + 1e-8) / (err_max - err_min + 1e-8)

                        dyn_min, dyn_max = dyn_range.min(dim=-1, keepdim=True)[0], dyn_range.max(dim=-1, keepdim=True)[0]
                        dyn_norm = (dyn_range - dyn_min + 1e-8) / (dyn_max - dyn_min + 1e-8)

                        quant_sensitivity = 0.5 * err_norm + 0.5 * dyn_norm
                    elif quant_method == "complex_l1":
                        group_size = 128
                        x_grouped = image_features.view(batch_size, seq_len, image_features.shape[-1] // group_size, group_size)
                        scale_grouped = x_grouped.abs().max(dim=-1, keepdim=True)[0] / 7.0
                        x_q_grouped = torch.round(x_grouped / (scale_grouped + 1e-8)) * scale_grouped
                        x_q = x_q_grouped.view(batch_size, seq_len, image_features.shape[-1])
                        err_group = torch.norm(image_features - x_q, dim=-1)
                        dyn_range = image_features.max(dim=-1)[0] - image_features.min(dim=-1)[0]

                        err_norm = err_group / (err_group.sum(dim=-1, keepdim=True) + 1e-8)
                        dyn_norm = dyn_range / (dyn_range.sum(dim=-1, keepdim=True) + 1e-8)
                        quant_sensitivity = 0.5 * err_norm + 0.5 * dyn_norm
                    elif quant_method == "complex_mul":
                        group_size = 128
                        x_grouped = image_features.view(batch_size, seq_len, image_features.shape[-1] // group_size, group_size)
                        scale_grouped = x_grouped.abs().max(dim=-1, keepdim=True)[0] / 7.0
                        x_q_grouped = torch.round(x_grouped / (scale_grouped + 1e-8)) * scale_grouped
                        x_q = x_q_grouped.view(batch_size, seq_len, image_features.shape[-1])
                        err_group = torch.norm(image_features - x_q, dim=-1)
                        dyn_range = image_features.max(dim=-1)[0] - image_features.min(dim=-1)[0]
                        quant_sensitivity = err_group * dyn_range
                    else:
                        raise ValueError(f"Unknown quant_method: {quant_method}")

                    if dynamic_alpha:
                        qs_mean = quant_sensitivity.mean(dim=-1, keepdim=True)
                        qs_std = quant_sensitivity.std(dim=-1, keepdim=True)
                        cv = qs_std / (qs_mean + 1e-8)
                        cv_norm = torch.clamp((cv - 0.2) / 1.0, min=0.0, max=1.0)
                        alpha = 0.55 - (0.55 - 0.45) * cv_norm

                    if quant_method not in ["complex", "complex_l1"]:
                        quant_min = quant_sensitivity.amin(dim=-1, keepdim=True)
                        quant_max = quant_sensitivity.amax(dim=-1, keepdim=True)
                        quant_sensitivity = (quant_sensitivity - quant_min + 1e-8) / (quant_max - quant_min + 1e-8)

                    quant_sensitivity = quant_sensitivity.to(device=device, dtype=relevance.dtype)
                    relevance = alpha * relevance + (1 - alpha) * quant_sensitivity

                kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1)

                cis = torch.zeros((self.visual_token_num, batch_size, seq_len), device=device)
                di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone()
                select_idx = torch.empty((self.visual_token_num, batch_size), dtype=torch.long, device=device)
                for i in range(self.visual_token_num):
                    j = torch.argmax(di2s, dim=-1)
                    select_idx[i] = j

                    eis = (
                        kernel[torch.arange(batch_size), j]
                        - torch.einsum("tb,tbn->bn", cis[:i, torch.arange(batch_size), j], cis[:i])
                    ) / torch.sqrt(di2s[torch.arange(batch_size), j]).unsqueeze(-1)
                    cis[i, :, :] = eis
                    di2s -= torch.square(eis)
                    di2s[torch.arange(batch_size), j] = -float("inf")

                select_idx = torch.sort(select_idx.t()).values
                index_masks = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
                index_masks.scatter_(1, select_idx, True)
                return image_features, index_masks

            raise ValueError(f"Unknown pruning_method: {pruning_method}")

        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        image_sizes=None,
        texts=None,
        add_quant=False,
        alpha=0.7,
        dynamic_alpha=False,
        quant_method="l2_norm",
        pruning_method="cdpruner",
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        if input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if getattr(self, "visual_token_num", None) is not None and texts is not None:
            if type(images) is list or images.ndim == 5:
                if type(images) is list:
                    images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
                concat_images = torch.cat([image for image in images], dim=0)
                image_features, index_masks = self.encode_images(
                    concat_images,
                    texts=texts,
                    add_quant=add_quant,
                    alpha=alpha,
                    dynamic_alpha=dynamic_alpha,
                    quant_method=quant_method,
                    pruning_method=pruning_method,
                )
                split_sizes = [image.shape[0] for image in images]
                image_features = torch.split(image_features, split_sizes, dim=0)
                index_masks = torch.split(index_masks, split_sizes, dim=0)
                mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
                mm_patch_merge_type = mm_patch_merge_type.replace("_unpad", "")
                image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
                if mm_patch_merge_type == "flat":
                    image_features = [x.flatten(0, 1) for x in image_features]
                    index_masks = [x.flatten(0, 1) for x in index_masks]
                    image_features = [x[m] for x, m in zip(image_features, index_masks)]
                elif mm_patch_merge_type.startswith("spatial"):
                    new_image_features = []
                    for image_idx, (image_feature, index_mask) in enumerate(zip(image_features, index_masks)):
                        if image_feature.shape[0] > 1:
                            base_image_feature = image_feature[0]
                            image_feature = image_feature[1:]
                            base_index_mask = index_mask[0]
                            index_mask = index_mask[1:]
                            height = width = self.get_vision_tower().num_patches_per_side
                            assert height * width == base_image_feature.shape[0]
                            if image_aspect_ratio == "anyres":
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                                    image_sizes[image_idx],
                                    self.config.image_grid_pinpoints,
                                    self.get_vision_tower().config.image_size,
                                )
                                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                                index_mask = index_mask.view(num_patch_height, num_patch_width, height, width)
                            else:
                                raise NotImplementedError
                            if "unpad" in mm_patch_merge_type:
                                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                                image_feature = torch.cat(
                                    (
                                        image_feature,
                                        self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device),
                                    ),
                                    dim=-1,
                                )
                                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                                index_mask = index_mask.permute(0, 2, 1, 3).contiguous().unsqueeze(0)
                                index_mask = index_mask.flatten(1, 2).flatten(2, 3)
                                index_mask = unpad_image(index_mask, image_sizes[image_idx])
                                index_mask = torch.cat(
                                    (
                                        index_mask,
                                        torch.ones(*index_mask.shape[:-1], 1, dtype=torch.bool).to(index_mask.device),
                                    ),
                                    dim=-1,
                                )
                                index_mask = index_mask.flatten(1, 2).squeeze(0)
                                image_feature = image_feature[index_mask]
                            else:
                                image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                                image_feature = image_feature.flatten(0, 3)
                                index_mask = index_mask.permute(0, 2, 1, 3).contiguous()
                                index_mask = index_mask.flatten(0, 3)
                                image_feature = image_feature[index_mask]
                            base_image_feature = base_image_feature[base_index_mask]
                            image_feature = torch.cat((base_image_feature, image_feature))
                        else:
                            image_feature = image_feature[0]
                            index_mask = index_mask[0]
                            if "unpad" in mm_patch_merge_type:
                                image_feature = torch.cat(
                                    (
                                        image_feature,
                                        self.model.image_newline[None].to(image_feature.device),
                                    ),
                                    dim=0,
                                )
                                index_mask = torch.cat(
                                    (
                                        index_mask,
                                        torch.ones(1, dtype=torch.bool).to(index_mask.device),
                                    ),
                                    dim=0,
                                )
                            image_feature = image_feature[index_mask]
                        new_image_features.append(image_feature)
                    image_features = new_image_features
                else:
                    raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
            else:
                image_features, index_masks = self.encode_images(
                    images,
                    texts=texts,
                    add_quant=add_quant,
                    alpha=alpha,
                    dynamic_alpha=dynamic_alpha,
                    quant_method=quant_method,
                    pruning_method=pruning_method,
                )
                image_features = image_features[index_masks].unsqueeze(0)
        else:
            if type(images) is list or images.ndim == 5:
                if type(images) is list:
                    images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
                concat_images = torch.cat([image for image in images], dim=0)
                image_features = self.encode_images(concat_images)
                split_sizes = [image.shape[0] for image in images]
                image_features = torch.split(image_features, split_sizes, dim=0)
                mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
                image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
                if mm_patch_merge_type == "flat":
                    image_features = [x.flatten(0, 1) for x in image_features]
                elif mm_patch_merge_type.startswith("spatial"):
                    new_image_features = []
                    for image_idx, image_feature in enumerate(image_features):
                        if image_feature.shape[0] > 1:
                            base_image_feature = image_feature[0]
                            image_feature = image_feature[1:]
                            height = width = self.get_vision_tower().num_patches_per_side
                            assert height * width == base_image_feature.shape[0]
                            if image_aspect_ratio == "anyres":
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                                    image_sizes[image_idx],
                                    self.config.image_grid_pinpoints,
                                    self.get_vision_tower().config.image_size,
                                )
                                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            else:
                                raise NotImplementedError
                            if "unpad" in mm_patch_merge_type:
                                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                                image_feature = torch.cat(
                                    (
                                        image_feature,
                                        self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device),
                                    ),
                                    dim=-1,
                                )
                                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            else:
                                image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                                image_feature = image_feature.flatten(0, 3)
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        else:
                            image_feature = image_feature[0]
                            if "unpad" in mm_patch_merge_type:
                                image_feature = torch.cat(
                                    (
                                        image_feature,
                                        self.model.image_newline[None].to(image_feature.device),
                                    ),
                                    dim=0,
                                )
                        new_image_features.append(image_feature)
                    image_features = new_image_features
                else:
                    raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
            else:
                image_features = self.encode_images(images)

        if isinstance(image_features, list):
            visual_token_num = image_features[0].shape[0] if image_features else 0
        else:
            visual_token_num = image_features.shape[1] if image_features.ndim > 2 else image_features.shape[0]
        setattr(self, "last_visual_token_num", int(visual_token_num))
        print(f"last_visual_token_num = {self.last_visual_token_num}")

        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        _input_ids = input_ids
        del _input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0,
                        cur_len,
                        dtype=position_ids.dtype,
                        device=position_ids.device,
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0,
                        cur_len,
                        dtype=position_ids.dtype,
                        device=position_ids.device,
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. "
                        f"Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
