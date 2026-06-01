from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class NABirdsRecord:
    image_id: str
    image_path: Path
    class_id: int
    label_idx: int
    class_name: str
    is_train: bool
    bbox: Optional[Tuple[float, float, float, float]] = None


def _read_id_value_file(path: Path, cast_value: Callable[[str], object] = str) -> Dict[str, object]:
    values: Dict[str, object] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            values[parts[0]] = cast_value(parts[1])
    return values


def read_classes(classes_path: Path) -> Dict[int, str]:
    classes: Dict[int, str] = {}
    with classes_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            class_id_str, class_name = line.split(maxsplit=1)
            classes[int(class_id_str)] = class_name
    return classes


def read_bboxes(bbox_path: Path) -> Dict[str, Tuple[float, float, float, float]]:
    bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    if not bbox_path.exists():
        return bboxes
    with bbox_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            image_id = parts[0]
            x, y, w, h = map(float, parts[1:5])
            bboxes[image_id] = (x, y, w, h)
    return bboxes


def resolve_image_path(root: Path, rel_path: str) -> Path:
    rel = Path(rel_path)
    candidates = [root / rel, root / "images" / rel]
    for p in candidates:
        if p.exists():
            return p
    # Return first candidate for clearer downstream error message.
    return candidates[0]


def load_nabirds_records(root: str | Path) -> Tuple[List[NABirdsRecord], Dict[int, int], Dict[int, str]]:
    """Load NABirds metadata and remap non-contiguous class IDs to 0..C-1.

    Expected official NABirds files under ``root``:
      - images.txt
      - image_class_labels.txt
      - train_test_split.txt
      - classes.txt
      - bounding_boxes.txt (optional but normally available)
    """
    root = Path(root)
    required = ["images.txt", "image_class_labels.txt", "train_test_split.txt", "classes.txt"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing NABirds metadata files under {root}: {missing}. "
            "Make sure nabirds_root points to the extracted NABirds directory."
        )

    image_rel_paths = _read_id_value_file(root / "images.txt", str)
    image_class_ids = _read_id_value_file(root / "image_class_labels.txt", lambda x: int(x))
    train_flags = _read_id_value_file(root / "train_test_split.txt", lambda x: bool(int(x)))
    class_names = read_classes(root / "classes.txt")
    bboxes = read_bboxes(root / "bounding_boxes.txt")

    observed_class_ids = sorted(set(int(v) for v in image_class_ids.values()))
    class_id_to_idx = {class_id: idx for idx, class_id in enumerate(observed_class_ids)}

    records: List[NABirdsRecord] = []
    for image_id, rel_path in image_rel_paths.items():
        class_id = int(image_class_ids[image_id])
        if class_id not in class_id_to_idx:
            continue
        records.append(
            NABirdsRecord(
                image_id=image_id,
                image_path=resolve_image_path(root, str(rel_path)),
                class_id=class_id,
                label_idx=class_id_to_idx[class_id],
                class_name=class_names.get(class_id, f"class_{class_id}"),
                is_train=bool(train_flags[image_id]),
                bbox=bboxes.get(image_id),
            )
        )

    idx_to_class_name = {class_id_to_idx[cid]: class_names.get(cid, f"class_{cid}") for cid in observed_class_ids}
    return records, class_id_to_idx, idx_to_class_name


def stratified_train_val_split(
    records: List[NABirdsRecord], val_fraction: float = 0.1, seed: int = 42
) -> Tuple[List[NABirdsRecord], List[NABirdsRecord], List[NABirdsRecord]]:
    """Use official NABirds test split, and carve a fixed val split from official train."""
    import random

    train_records = [r for r in records if r.is_train]
    test_records = [r for r in records if not r.is_train]
    by_class: Dict[int, List[NABirdsRecord]] = {}
    for r in train_records:
        by_class.setdefault(r.label_idx, []).append(r)

    rng = random.Random(seed)
    final_train: List[NABirdsRecord] = []
    val: List[NABirdsRecord] = []
    for _, items in sorted(by_class.items()):
        items = list(items)
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_fraction))) if len(items) > 1 else 0
        val.extend(items[:n_val])
        final_train.extend(items[n_val:])

    rng.shuffle(final_train)
    rng.shuffle(val)
    return final_train, val, test_records


def crop_with_bbox(image: Image.Image, bbox: Tuple[float, float, float, float], margin: float = 0.05) -> Image.Image:
    width, height = image.size
    x, y, w, h = bbox
    mx, my = w * margin, h * margin
    left = max(0, int(round(x - mx)))
    top = max(0, int(round(y - my)))
    right = min(width, int(round(x + w + mx)))
    bottom = min(height, int(round(y + h + my)))
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


class NABirdsSiglipDataset(Dataset):
    def __init__(
        self,
        records: List[NABirdsRecord],
        class_texts: Dict[int, str],
        use_bbox: bool = False,
        bbox_margin: float = 0.05,
    ) -> None:
        self.records = records
        self.class_texts = class_texts
        self.use_bbox = use_bbox
        self.bbox_margin = bbox_margin

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        record = self.records[idx]
        try:
            image = Image.open(record.image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to read image {record.image_path}") from exc

        if self.use_bbox and record.bbox is not None:
            image = crop_with_bbox(image, record.bbox, margin=self.bbox_margin)

        return {
            "image": image,
            "label_idx": record.label_idx,
            "class_id": record.class_id,
            "class_name": record.class_name,
            "text": self.class_texts[record.label_idx],
            "image_id": record.image_id,
        }


def make_train_collate_fn(processor, max_length: int = 64):
    def collate(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        images = [item["image"] for item in batch]
        texts = [str(item["text"]) for item in batch]
        encoded = processor(
            images=images,
            text=texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([int(item["label_idx"]) for item in batch], dtype=torch.long)
        return encoded

    return collate


def make_eval_collate_fn(processor):
    def collate(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        images = [item["image"] for item in batch]
        encoded = processor(images=images, return_tensors="pt")
        encoded["labels"] = torch.tensor([int(item["label_idx"]) for item in batch], dtype=torch.long)
        return encoded

    return collate
