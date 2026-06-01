from __future__ import annotations

import argparse

from siglip_nabirds.data import load_nabirds_records
from siglip_nabirds.text_prompts import export_text_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Export label/attribute/prompt text tables for NABirds SigLIP experiments.")
    parser.add_argument("--nabirds_root", required=True)
    parser.add_argument("--text_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--label_template", default="{label}")
    args = parser.parse_args()

    _, class_id_to_idx, idx_to_class_name = load_nabirds_records(args.nabirds_root)
    df = export_text_table(
        idx_to_class_name=idx_to_class_name,
        class_id_to_idx=class_id_to_idx,
        csv_path=args.text_csv,
        output_path=args.output_csv,
        label_template=args.label_template,
    )
    print(f"Wrote {len(df)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
