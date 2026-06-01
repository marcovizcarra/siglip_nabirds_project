from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from .data import NABirdsSiglipDataset, load_nabirds_records, make_train_collate_fn, stratified_train_val_split
from .eval import evaluate
from .modeling import configure_trainable_parts, load_siglip, set_processor_image_size, siglip_pairwise_loss
from .text_prompts import TEXT_MODES, build_class_texts, export_text_table
from .utils import save_json, seed_everything, unwrap_for_json


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SigLIP on NABirds with label/attribute/prompt text variants.")
    parser.add_argument("--nabirds_root", type=str, required=True, help="Path to extracted NABirds directory.")
    parser.add_argument("--text_csv", type=str, required=True, help="Path to bird_class_expert_visual_attributes.csv.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where checkpoints/metrics are written.")
    parser.add_argument("--text_mode", type=str, choices=TEXT_MODES, required=True)
    parser.add_argument("--model_name", type=str, default="google/siglip-base-patch16-224")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--label_template", type=str, default="{label}", help="Used only for --text_mode label.")

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--use_bbox", action="store_true", help="Crop images using NABirds bounding boxes before SigLIP preprocessing.")
    parser.add_argument("--bbox_margin", type=float, default=0.05)

    parser.add_argument("--train_text_encoder", action="store_true", help="By default the text tower is frozen. Set this to train it too.")
    parser.add_argument("--freeze_vision_encoder", action="store_true")
    parser.add_argument("--eval_every_epoch", action="store_true", default=True)
    parser.add_argument("--skip_test", action="store_true")

    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None, help="online, offline, or disabled.")
    return parser.parse_args(argv)


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "labels"}


def train(argv: Optional[List[str]] = None) -> Dict[str, float]:
    args = parse_args(argv)
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), output_dir / "config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print(f"[WARN] CUDA is not available; training on {device}. This is only good for smoke tests.")

    records, class_id_to_idx, idx_to_class_name = load_nabirds_records(args.nabirds_root)
    train_records, val_records, test_records = stratified_train_val_split(records, args.val_fraction, args.seed)
    class_texts = build_class_texts(
        idx_to_class_name=idx_to_class_name,
        class_id_to_idx=class_id_to_idx,
        csv_path=args.text_csv,
        mode=args.text_mode,
        label_template=args.label_template,
    )
    export_text_table(
        idx_to_class_name=idx_to_class_name,
        class_id_to_idx=class_id_to_idx,
        csv_path=args.text_csv,
        output_path=output_dir / "class_texts_all_modes.csv",
        label_template=args.label_template,
    )
    save_json({str(k): v for k, v in class_texts.items()}, output_dir / "class_texts.json")
    save_json({str(k): v for k, v in idx_to_class_name.items()}, output_dir / "idx_to_class_name.json")
    save_json({str(k): v for k, v in class_id_to_idx.items()}, output_dir / "class_id_to_idx.json")

    print(f"Loaded NABirds: {len(records)} images, {len(class_texts)} classes")
    print(f"Split: train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")
    print(f"Text mode: {args.text_mode}; sample text: {next(iter(class_texts.values()))}")

    model, processor = load_siglip(args.model_name)
    set_processor_image_size(processor, args.image_size)
    configure_trainable_parts(model, freeze_text=not args.train_text_encoder, freeze_vision=args.freeze_vision_encoder)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.to(device)

    train_dataset = NABirdsSiglipDataset(train_records, class_texts, use_bbox=args.use_bbox, bbox_margin=args.bbox_margin)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=make_train_collate_fn(processor, max_length=args.max_length),
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    update_steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum_steps))
    total_steps = update_steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    use_amp = device.type == "cuda" and args.mixed_precision != "no"
    amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and args.mixed_precision == "fp16"))

    run = None
    if args.wandb_project and args.wandb_mode != "disabled":
        import wandb

        if args.wandb_mode:
            os.environ["WANDB_MODE"] = args.wandb_mode
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f"siglip_nabirds_{args.text_mode}",
            config=vars(args),
        )

    best_val_top1 = -1.0
    best_metrics: Dict[str, float] = {}
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(progress, start=1):
            labels = batch["labels"].to(device, non_blocking=True)
            model_inputs = to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(**model_inputs)
                loss = siglip_pairwise_loss(outputs.logits_per_image, labels)
                loss = loss / max(1, args.grad_accum_steps)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % args.grad_accum_steps == 0:
                if args.max_grad_norm > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running_loss += float(loss.detach().cpu()) * max(1, args.grad_accum_steps)
            avg_loss = running_loss / step
            progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if run is not None and global_step > 0 and step % args.grad_accum_steps == 0:
                run.log({"train/loss": avg_loss, "train/lr": scheduler.get_last_lr()[0], "epoch": epoch}, step=global_step)

        val_metrics = evaluate(
            model=model,
            processor=processor,
            records=val_records,
            class_texts=class_texts,
            device=device,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            max_length=args.max_length,
            use_bbox=args.use_bbox,
            bbox_margin=args.bbox_margin,
            desc="val",
        )
        val_metrics = {f"val/{k}": v for k, v in val_metrics.items()}
        print(f"Epoch {epoch} validation: {val_metrics}")
        if run is not None:
            run.log(val_metrics, step=global_step)

        if val_metrics["val/top1"] > best_val_top1:
            best_val_top1 = val_metrics["val/top1"]
            best_metrics = val_metrics.copy()
            best_dir = output_dir / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            save_json(best_metrics, output_dir / "best_val_metrics.json")
            print(f"Saved new best model to {best_dir}")

        last_dir = output_dir / "last_model"
        last_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(last_dir)
        processor.save_pretrained(last_dir)

    final_metrics = best_metrics.copy()
    if not args.skip_test:
        # Load best weights before final test eval.
        from transformers import SiglipModel

        best_dir = output_dir / "best_model"
        if best_dir.exists():
            model = SiglipModel.from_pretrained(best_dir).to(device)
        test_metrics = evaluate(
            model=model,
            processor=processor,
            records=test_records,
            class_texts=class_texts,
            device=device,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            max_length=args.max_length,
            use_bbox=args.use_bbox,
            bbox_margin=args.bbox_margin,
            desc="test",
        )
        test_metrics = {f"test/{k}": v for k, v in test_metrics.items()}
        final_metrics.update(test_metrics)
        save_json(final_metrics, output_dir / "final_metrics.json")
        print(f"Final metrics: {final_metrics}")
        if run is not None:
            run.log(test_metrics, step=global_step)
            run.summary.update(unwrap_for_json(final_metrics))
    else:
        save_json(final_metrics, output_dir / "final_metrics.json")

    if run is not None:
        run.finish()
    return final_metrics


def main(argv: Optional[List[str]] = None) -> Dict[str, float]:
    return train(argv)


if __name__ == "__main__":
    main()
