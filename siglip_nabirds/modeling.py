from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, SiglipModel


def load_siglip(model_name: str):
    processor = AutoProcessor.from_pretrained(model_name)
    model = SiglipModel.from_pretrained(model_name)
    return model, processor


def set_processor_image_size(processor, image_size: Optional[int]) -> None:
    if image_size is None:
        return
    image_size = int(image_size)
    if hasattr(processor, "image_processor"):
        ip = processor.image_processor
        if hasattr(ip, "size"):
            ip.size = {"height": image_size, "width": image_size}
        if hasattr(ip, "crop_size"):
            ip.crop_size = {"height": image_size, "width": image_size}


def freeze_module(module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def configure_trainable_parts(model, freeze_text: bool = True, freeze_vision: bool = False) -> None:
    if freeze_text and hasattr(model, "text_model"):
        freeze_module(model.text_model)
    if freeze_vision and hasattr(model, "vision_model"):
        freeze_module(model.vision_model)


def siglip_pairwise_loss(logits_per_image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """SigLIP-style pairwise logistic loss.

    Standard image-text contrastive CE assumes only one positive pair per batch.
    NABirds batches can contain duplicate species, so this loss treats every sample
    with the same class label as a positive pair.
    """
    same_class = labels[:, None].eq(labels[None, :])
    targets = torch.where(same_class, torch.ones_like(logits_per_image), -torch.ones_like(logits_per_image))
    return F.softplus(-targets * logits_per_image).mean()


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


def get_logit_scale_and_bias(model, device: torch.device):
    if hasattr(model, "logit_scale"):
        scale = model.logit_scale.exp().to(device)
    else:
        scale = torch.tensor(1.0, device=device)
    if hasattr(model, "logit_bias"):
        bias = model.logit_bias.to(device)
    else:
        bias = torch.tensor(0.0, device=device)
    return scale, bias


@torch.no_grad()
def encode_class_texts(
    model,
    processor,
    class_texts: Dict[int, str],
    device: torch.device,
    batch_size: int = 128,
    max_length: int = 64,
) -> torch.Tensor:
    model.eval()
    texts = [class_texts[i] for i in range(len(class_texts))]
    all_features: List[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = processor(
            text=batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        features = model.get_text_features(**encoded)
        all_features.append(normalize_features(features).cpu())
    return torch.cat(all_features, dim=0)


@torch.no_grad()
def image_text_logits(model, pixel_values: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    device = pixel_values.device
    image_features = model.get_image_features(pixel_values=pixel_values)
    image_features = normalize_features(image_features)
    text_features = text_features.to(device)
    scale, bias = get_logit_scale_and_bias(model, device)
    return image_features @ text_features.T * scale + bias
