#!/usr/bin/env bash
set -euo pipefail

NABIRDS_ROOT=${NABIRDS_ROOT:-/data/nabirds}
TEXT_CSV=${TEXT_CSV:-/data/bird_class_expert_visual_attributes.csv}
OUT_ROOT=${OUT_ROOT:-/data/runs/siglip_nabirds}
WANDB_PROJECT=${WANDB_PROJECT:-cs231n-siglip-nabirds}
MODEL_NAME=${MODEL_NAME:-google/siglip-base-patch16-224}

for MODE in label attributes prompt; do
  python -m siglip_nabirds.train \
    --nabirds_root "${NABIRDS_ROOT}" \
    --text_csv "${TEXT_CSV}" \
    --output_dir "${OUT_ROOT}/${MODE}" \
    --text_mode "${MODE}" \
    --model_name "${MODEL_NAME}" \
    --image_size 224 \
    --epochs 8 \
    --batch_size 32 \
    --eval_batch_size 64 \
    --grad_accum_steps 1 \
    --lr 2e-5 \
    --weight_decay 0.05 \
    --warmup_ratio 0.05 \
    --mixed_precision bf16 \
    --use_bbox \
    --num_workers 8 \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "siglip_nabirds_${MODE}"
done
