from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import NABirdsSiglipDataset, make_eval_collate_fn
from .modeling import encode_class_texts, image_text_logits


def compute_metrics(y_true: np.ndarray, logits: np.ndarray, num_classes: int) -> Dict[str, float]:
    preds = logits.argmax(axis=1)
    top1 = float((preds == y_true).mean())
    k = min(5, num_classes)
    topk = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    top5 = float(np.any(topk == y_true[:, None], axis=1).mean())
    return {
        "top1": top1,
        "top5": top5,
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, preds, average="macro", zero_division=0)),
    }


@torch.no_grad()
def evaluate(
    model,
    processor,
    records,
    class_texts: Dict[int, str],
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    max_length: int = 64,
    text_batch_size: int = 128,
    use_bbox: bool = False,
    bbox_margin: float = 0.05,
    desc: str = "eval",
) -> Dict[str, float]:
    dataset = NABirdsSiglipDataset(records, class_texts, use_bbox=use_bbox, bbox_margin=bbox_margin)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=make_eval_collate_fn(processor),
    )
    model.eval()
    text_features = encode_class_texts(
        model=model,
        processor=processor,
        class_texts=class_texts,
        device=device,
        batch_size=text_batch_size,
        max_length=max_length,
    )

    all_logits = []
    all_labels = []
    for batch in tqdm(loader, desc=desc, leave=False):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].numpy()
        logits = image_text_logits(model, pixel_values, text_features).float().cpu().numpy()
        all_logits.append(logits)
        all_labels.append(labels)

    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    return compute_metrics(labels_np, logits_np, num_classes=len(class_texts))
