import torch
import torch.nn as nn

from transformers import (
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizerFast,
    CLIPVisionConfig,
    CLIPVisionModel,
    CLIPVisionModelWithProjection,
)


def _clip_from_pretrained(model_cls, model_name, device_map=None):
    try:
        return model_cls.from_pretrained(
            model_name,
            device_map=device_map,
            use_safetensors=True,
        )
    except Exception as exc:
        print(
            f"Falling back to default checkpoint loading for `{model_name}` because "
            f"safetensors loading was unavailable: {exc}"
        )
        return model_cls.from_pretrained(model_name, device_map=device_map)


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f"{self.vision_tower_name} is already loaded, `load_model` called again, skipping.")
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = _clip_from_pretrained(
            CLIPVisionModel,
            self.vision_tower_name,
            device_map=device_map,
        )
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def load_text_tower(self, device_map=None):
        if not self.is_loaded:
            self.load_model(device_map=device_map)
        if hasattr(self, "text_tower"):
            return

        CLIPVisionModelWithProjection._no_split_modules = ["CLIPEncoderLayer"]
        vision_tower_with_projection = _clip_from_pretrained(
            CLIPVisionModelWithProjection,
            self.vision_tower_name,
            device_map=device_map,
        )
        self.vision_tower.visual_projection = vision_tower_with_projection.visual_projection

        self.text_tokenizer = CLIPTokenizerFast.from_pretrained(self.vision_tower_name)
        self.text_tower = _clip_from_pretrained(
            CLIPTextModelWithProjection,
            self.vision_tower_name,
            device_map=device_map,
        )
        self.text_tower.requires_grad_(False)
        self.max_position_embeddings = self.text_tower.config.max_position_embeddings

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def _encode_texts(self, texts):
        if not hasattr(self, "text_tower"):
            raise RuntimeError(
                "Text tower is not loaded. Call `load_text_tower` before using QAPruner/CDPruner."
            )

        text_inputs = self.text_tokenizer(text=texts, return_tensors="pt", padding=True)
        text_segment_count = (text_inputs.input_ids.shape[1] - 1) // self.max_position_embeddings + 1
        text_padding = self.max_position_embeddings * text_segment_count - text_inputs.input_ids.shape[1]
        text_inputs = {
            key: torch.cat(
                [value, value.new_zeros((value.shape[0], text_padding))],
                dim=1,
            ).reshape(-1, self.max_position_embeddings).to(device=self.text_tower.device)
            for key, value in text_inputs.items()
        }
        return self.text_tower(**text_inputs).text_embeds

    @torch.no_grad()
    def forward(self, images, texts=None, output_attentions=False):
        if type(images) is list and texts is None:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                    output_attentions=output_attentions,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
            return image_features

        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,
            output_attentions=output_attentions,
        )
        image_outputs = self.feature_select(image_forward_outs)
        image_features = image_outputs.to(images.dtype)

        if texts is None:
            return image_features

        text_embeds = self._encode_texts(texts)
        image_embeds = self.vision_tower.vision_model.post_layernorm(image_outputs)
        projection_weight = self.vision_tower.visual_projection.weight
        image_embeds = image_embeds.to(
            device=projection_weight.device,
            dtype=projection_weight.dtype,
        )
        image_embeds = self.vision_tower.visual_projection(image_embeds)
        text_embeds = text_embeds.to(
            device=image_embeds.device,
            dtype=image_embeds.dtype,
        )

        if output_attentions:
            return image_features, image_embeds, text_embeds, image_forward_outs.attentions
        return image_features, image_embeds, text_embeds

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, "s2_scales", "336,672,1008")
        self.s2_scales = list(map(int, self.s2_scales.split(",")))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError(
                "Package s2wrapper not found! Please install by running: \n"
                "pip install git+https://github.com/bfshi/scaling_on_scales.git"
            )
        self.multiscale_forward = multiscale_forward

        if not delay_load or getattr(args, "unfreeze_mm_vision_tower", False):
            self.image_processor.size["shortest_edge"] = self.s2_image_size
            self.image_processor.crop_size["height"] = self.image_processor.crop_size["width"] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f"{self.vision_tower_name} is already loaded, `load_model` called again, skipping.")
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = _clip_from_pretrained(
            CLIPVisionModel,
            self.vision_tower_name,
            device_map=device_map,
        )
        self.vision_tower.requires_grad_(False)

        self.image_processor.size["shortest_edge"] = self.s2_image_size
        self.image_processor.crop_size["height"] = self.image_processor.crop_size["width"] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,
        )
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images, texts=None, output_attentions=False):
        if texts is not None or output_attentions:
            raise NotImplementedError("QAPruner/CDPruner is not supported with S2 vision towers yet.")

        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(
                    self.forward_feature,
                    image.unsqueeze(0),
                    img_sizes=self.s2_scales,
                    max_split_size=self.s2_split_size,
                )
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(
                self.forward_feature,
                images,
                img_sizes=self.s2_scales,
                max_split_size=self.s2_split_size,
            )

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
