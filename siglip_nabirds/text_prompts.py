from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict, List, Mapping

import pandas as pd


TEXT_MODES = ("label", "attributes", "prompt", "hybrid")


def normalize_label(text: str) -> str:
    return " ".join(str(text).strip().split()).lower()


def parse_attribute_list(value: object) -> List[str]:
    """Parse the expert_visual_description column into a Python list of short attributes."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text or text.lower() == "nan" or text == "[]":
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return [chunk.strip().strip('"\'') for chunk in text.split(",") if chunk.strip()]
    if isinstance(parsed, list):
        return [str(x).strip() for x in parsed if str(x).strip()]
    return []


def attributes_to_sentence(attributes: List[str]) -> str:
    if not attributes:
        return ""
    return ", ".join(attributes)


def fallback_prompt_from_attributes(attributes: List[str], class_name: str) -> str:
    attr_text = attributes_to_sentence(attributes)
    if attr_text:
        return f"A bird with {attr_text}."
    return class_name


def clean_sentence(text: str) -> str:
    text = " ".join(str(text).strip().split())
    if not text:
        return ""
    if not text.endswith("."):
        text += "."
    return text


def hybrid_prompt(class_name: str, attrs: List[str], prompt_value: object) -> str:
    """Combine the discriminative class label with expert visual description."""
    label_part = f"A photo of a {class_name}."

    if prompt_value is not None and not pd.isna(prompt_value) and str(prompt_value).strip():
        desc = clean_sentence(str(prompt_value))
    else:
        desc = clean_sentence(fallback_prompt_from_attributes(attrs, class_name))

    if desc.lower() == class_name.lower() or desc.lower() == clean_sentence(class_name).lower():
        return label_part

    return f"{label_part} {desc}"


def load_expert_csv(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"class_id", "class_label", "expert_visual_description", "expert_visual_description_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in expert CSV: {sorted(missing)}")
    df = df.copy()
    df["class_id"] = df["class_id"].astype(int)
    df["_label_norm"] = df["class_label"].map(normalize_label)
    df["_attributes"] = df["expert_visual_description"].map(parse_attribute_list)
    return df


def build_class_texts(
    idx_to_class_name: Mapping[int, str],
    class_id_to_idx: Mapping[int, int],
    csv_path: str | Path,
    mode: str,
    label_template: str = "{label}",
) -> Dict[int, str]:
    """Build class-level text input for each NABirds class.

    Modes:
      - label: only the NABirds class label.
      - attributes: comma-separated short visual attributes.
      - prompt: expert_visual_description_text only.
      - hybrid: class label + expert_visual_description_text.
    """
    if mode not in TEXT_MODES:
        raise ValueError(f"Unknown text mode {mode!r}; expected one of {TEXT_MODES}")

    df = load_expert_csv(csv_path)
    records = df.to_dict(orient="records")
    by_id = {int(row["class_id"]): row for row in records}
    by_label = {normalize_label(row["class_label"]): row for row in records}

    idx_to_class_id = {idx: cid for cid, idx in class_id_to_idx.items()}
    class_texts: Dict[int, str] = {}

    for idx, class_name in idx_to_class_name.items():
        class_id = idx_to_class_id[idx]
        row = by_id.get(class_id) or by_label.get(normalize_label(class_name))

        attrs: List[str] = []
        prompt_value = None

        if row is not None:
            attrs = list(row.get("_attributes", []))
            prompt_value = row.get("expert_visual_description_text")

        if mode == "label":
            text = label_template.format(label=class_name)
        elif mode == "attributes":
            text = attributes_to_sentence(attrs) or class_name
        elif mode == "prompt":
            if prompt_value is not None and not pd.isna(prompt_value) and str(prompt_value).strip():
                text = str(prompt_value).strip()
            else:
                text = fallback_prompt_from_attributes(attrs, class_name)
        elif mode == "hybrid":
            text = hybrid_prompt(class_name, attrs, prompt_value)
        else:
            raise ValueError(f"Unknown text mode: {mode}")

        class_texts[int(idx)] = " ".join(text.split())

    return class_texts


def export_text_table(
    idx_to_class_name: Mapping[int, str],
    class_id_to_idx: Mapping[int, int],
    csv_path: str | Path,
    output_path: str | Path,
    label_template: str = "{label}",
) -> pd.DataFrame:
    rows = []
    idx_to_class_id = {idx: cid for cid, idx in class_id_to_idx.items()}

    for mode in TEXT_MODES:
        class_texts = build_class_texts(
            idx_to_class_name=idx_to_class_name,
            class_id_to_idx=class_id_to_idx,
            csv_path=csv_path,
            mode=mode,
            label_template=label_template,
        )
        for idx, text in class_texts.items():
            rows.append(
                {
                    "mode": mode,
                    "class_idx": idx,
                    "class_id": idx_to_class_id[idx],
                    "class_label": idx_to_class_name[idx],
                    "text": text,
                }
            )

    out_df = pd.DataFrame(rows).sort_values(["mode", "class_idx"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    return out_df
