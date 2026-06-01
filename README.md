# SigLIP + NABirds Training

This package trains **SigLIP** on NABirds using three text variants from `bird_class_expert_visual_attributes.csv`:

1. `label`: only the NABirds class label.
2. `attributes`: a compact comma-separated list from `expert_visual_description`.
3. `prompt`: the full sentence from `expert_visual_description_text`, e.g. `A bird with ...`.

The training loss is a SigLIP-style pairwise logistic image-text loss. During evaluation, the model embeds all class texts and ranks each image against all NABirds classes. Metrics saved: top-1, top-5, macro-F1, macro-precision, and macro-recall.

## Expected NABirds layout

After extracting NABirds, `nabirds_root` should contain files like:

```text
nabirds/
  images.txt
  image_class_labels.txt
  train_test_split.txt
  classes.txt
  bounding_boxes.txt
  images/
    ... jpg files ...
```

## Local install

```bash
cd siglip_nabirds_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Create the prompt table preview

```bash
python scripts/make_text_prompts.py \
  --nabirds_root /path/to/nabirds \
  --text_csv resources/bird_class_expert_visual_attributes.csv \
  --output_csv outputs/class_texts_all_modes.csv
```

## Run one local experiment

```bash
python -m siglip_nabirds.train \
  --nabirds_root /path/to/nabirds \
  --text_csv resources/bird_class_expert_visual_attributes.csv \
  --output_dir outputs/siglip_label \
  --text_mode label \
  --model_name google/siglip-base-patch16-224 \
  --image_size 224 \
  --epochs 8 \
  --batch_size 32 \
  --eval_batch_size 64 \
  --lr 2e-5 \
  --mixed_precision bf16 \
  --use_bbox \
  --wandb_project cs231n-siglip-nabirds \
  --wandb_run_name siglip_nabirds_label
```

For the other two models, change `--text_mode` and output/run names:

```bash
--text_mode attributes --output_dir outputs/siglip_attributes --wandb_run_name siglip_nabirds_attributes
--text_mode prompt     --output_dir outputs/siglip_prompt     --wandb_run_name siglip_nabirds_prompt
```

You can also run all three locally:

```bash
NABIRDS_ROOT=/path/to/nabirds \
TEXT_CSV=resources/bird_class_expert_visual_attributes.csv \
OUT_ROOT=outputs/siglip_nabirds \
WANDB_PROJECT=cs231n-siglip-nabirds \
bash scripts/run_three_experiments.sh
```

## Modal setup

Create a Modal volume and upload data:

```bash
modal setup
modal volume create nabirds-siglip-data
modal volume put nabirds-siglip-data /path/to/nabirds /nabirds
modal volume put nabirds-siglip-data resources/bird_class_expert_visual_attributes.csv /bird_class_expert_visual_attributes.csv
modal volume ls nabirds-siglip-data /
```

Create a W&B secret:

```bash
modal secret create wandb-secret WANDB_API_KEY=your_wandb_key_here
```

Run all three SigLIP experiments on Modal:

```bash
modal run modal_train_siglip_nabirds.py --text-mode all --epochs 8 --batch-size 32
```

Run only one variant:

```bash
modal run modal_train_siglip_nabirds.py --text-mode prompt --epochs 8 --batch-size 32
```

Download results:

```bash
modal volume get nabirds-siglip-data /runs ./runs
```

## Suggested experiments

Start with:

```text
model_name = google/siglip-base-patch16-224
image_size = 224
batch_size = 32
lr = 2e-5
epochs = 8
use_bbox = true
freeze_text_encoder = true (default)
```

Then try stronger but more expensive runs:

```bash
modal run modal_train_siglip_nabirds.py \
  --text-mode all \
  --model-name google/siglip-so400m-patch14-384 \
  --image-size 384 \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --epochs 8
```

If the prompt/attribute variants underperform, try the same commands without bbox cropping, or add `--train-text-encoder` from the Modal entrypoint to let the text tower adapt.

## Main outputs

Each variant writes:

```text
/data/runs/siglip_nabirds/<mode>/
  config.json
  class_texts.json
  class_texts_all_modes.csv
  idx_to_class_name.json
  class_id_to_idx.json
  best_model/
  last_model/
  best_val_metrics.json
  final_metrics.json
```
