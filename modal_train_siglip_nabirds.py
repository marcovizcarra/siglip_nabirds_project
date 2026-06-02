from __future__ import annotations

import modal

APP_NAME = "siglip-nabirds"
VOLUME_NAME = "nabirds-siglip-data"
DATA_ROOT = "/data"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch>=2.3.0",
        "torchvision>=0.18.0",
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "scikit-learn>=1.3.0",
        "wandb>=0.17.0",
        "tqdm>=4.66.0",
        "sentencepiece>=0.2.0",
        "protobuf>=4.25.0",
    )
    .add_local_dir("siglip_nabirds", remote_path="/root/siglip_nabirds")
)


@app.function(
    image=image,
    gpu="RTX-PRO-6000",
    timeout=60 * 60 * 24,
    volumes={DATA_ROOT: volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_one(
    text_mode: str,
    nabirds_root: str = f"{DATA_ROOT}/nabirds",
    text_csv: str = f"{DATA_ROOT}/bird_class_expert_visual_attributes.csv",
    out_root: str = f"{DATA_ROOT}/runs/siglip_nabirds",
    model_name: str = "google/siglip-base-patch16-224",
    epochs: int = 8,
    batch_size: int = 32,
    eval_batch_size: int = 64,
    grad_accum_steps: int = 1,
    lr: float = 2e-5,
    image_size: int = 224,
    mixed_precision: str = "bf16",
    use_bbox: bool = True,
    train_text_encoder: bool = False,
    wandb_project: str = "cs231n-siglip-nabirds",
):
    from siglip_nabirds.train import main as train_main

    argv = [
        "--nabirds_root",
        nabirds_root,
        "--text_csv",
        text_csv,
        "--output_dir",
        f"{out_root}/{text_mode}",
        "--text_mode",
        text_mode,
        "--model_name",
        model_name,
        "--image_size",
        str(image_size),
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--eval_batch_size",
        str(eval_batch_size),
        "--grad_accum_steps",
        str(grad_accum_steps),
        "--lr",
        str(lr),
        "--mixed_precision",
        mixed_precision,
        "--num_workers",
        "8",
        "--wandb_project",
        wandb_project,
        "--wandb_run_name",
        f"siglip_nabirds_{text_mode}",
    ]
    if use_bbox:
        argv.append("--use_bbox")
    if train_text_encoder:
        argv.append("--train_text_encoder")

    metrics = train_main(argv)
    volume.commit()
    return metrics


@app.local_entrypoint()
def main(
    text_mode: str = "all",
    epochs: int = 8,
    batch_size: int = 32,
    grad_accum_steps: int = 1,
    lr: float = 2e-5,
    image_size: int = 224,
    model_name: str = "google/siglip-base-patch16-224",
    train_text_encoder: bool = False,
):
    modes = ["label", "attributes", "prompt", "hybrid"] if text_mode == "all" else [text_mode]
    for mode in modes:
        if mode not in {"label", "attributes", "prompt", "hybrid"}:
            raise ValueError("text_mode must be one of: all, label, attributes, prompt, hybrid")
        train_one.spawn(
            text_mode=mode,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            lr=lr,
            image_size=image_size,
            model_name=model_name,
            train_text_encoder=train_text_encoder,
        )
