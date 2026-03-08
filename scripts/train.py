#!/usr/bin/env python3
"""End-to-end DoRA fine-tuning of π₀ for kitchen dishwashing grasps.

This is the main training script. It:
  1. Generates or loads training data (synthetic + optional real)
  2. Prepares (image, instruction, action) tuples
  3. Loads the π₀ base model
  4. Applies DoRA adapter via PEFT
  5. Fine-tunes with HuggingFace Trainer
  6. Evaluates on holdout set
  7. Saves adapter weights + metrics

Usage:
    # Quick validation (50 samples, 2 epochs) — ~5 min on A10G
    python scripts/train.py --quick

    # Full training (5000 samples, 5 epochs) — ~2 hrs on A10G
    python scripts/train.py --samples 5000 --epochs 5

    # From existing annotations
    python scripts/train.py --annotations data/synthetic_annotations.json

    # Local CPU dry run (no model, validates data pipeline)
    python scripts/train.py --dry-run --samples 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


def _preflight(args: argparse.Namespace) -> None:
    """Verify critical dependencies are importable before doing any work."""
    errors: list[str] = []

    for mod_name, pip_pkg, purpose in [
        ("torch", "torch", "PyTorch"),
        ("transformers", "transformers>=4.47.0", "model loading"),
        ("peft", "peft>=0.14.0", "DoRA/LoRA adapters"),
        ("accelerate", "accelerate>=0.35.0", "distributed / device_map"),
    ]:
        try:
            __import__(mod_name)
        except ImportError:
            errors.append(f"  pip install \"{pip_pkg}\"   # {purpose}")

    if errors:
        print("\n❌ Missing required packages:")
        print("\n".join(errors))
        print("\nInstall them with:")
        print("  pip install -e '.[dev,gpu]'")
        print("  # or: python scripts/check_deps.py --install")
        sys.exit(1)

    # CUDA check (warn, don't block)
    if not args.dry_run:
        import torch
        if not torch.cuda.is_available():
            print("\n⚠️  WARNING: No CUDA GPU detected. Training will be extremely slow on CPU.")
            print("   Consider using --dry-run to validate data pipeline only.\n")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune π₀ with DoRA for kitchen dishwashing grasps"
    )

    # Data
    p.add_argument("--annotations", type=str, default=None,
                   help="Path to existing annotations JSON. If not provided, generates synthetic data.")
    p.add_argument("--samples", type=int, default=5000,
                   help="Number of synthetic samples to generate (if no annotations provided)")
    p.add_argument("--data-dir", type=str, default="data/training",
                   help="Directory for rendered training images")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--generation-strategy", type=str, default="balanced", choices=["balanced", "random"],
                   help="Synthetic data generation strategy")
    p.add_argument("--include-failures", action="store_true",
                   help="Keep failed attempts in the dataset manifest instead of filtering to successes only")

    # Model
    p.add_argument("--base-model", type=str, default="physical-intelligence/pi0-base",
                   help="Base VLA model to fine-tune")
    p.add_argument("--adapter-type", type=str, default="dora", choices=["dora", "lora"],
                   help="Adapter type: DoRA (recommended) or LoRA")
    p.add_argument("--rank", type=int, default=16, help="LoRA/DoRA rank")
    p.add_argument("--alpha", type=int, default=32, help="LoRA/DoRA alpha")
    p.add_argument("--target-modules", type=str, nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                   help="Target modules for adapter")

    # Training
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--gradient-accumulation", type=int, default=4,
                   help="Gradient accumulation steps (effective batch = batch_size * this)")
    p.add_argument("--eval-pct", type=float, default=0.15,
                   help="Fraction of data held out for evaluation")
    p.add_argument("--fp16", action="store_true", default=True,
                   help="Use FP16 mixed precision")

    # Output
    p.add_argument("--output-dir", type=str, default="models/dora/kitchen_v1",
                   help="Directory to save adapter weights")
    p.add_argument("--profile-name", type=str, default="kitchen_default_v1",
                   help="Name for the kitchen profile")

    # Modes
    p.add_argument("--quick", action="store_true",
                   help="Quick validation: 50 samples, 2 epochs")
    p.add_argument("--dry-run", action="store_true",
                   help="Data pipeline only, no model loading/training")
    p.add_argument("--render", action="store_true", default=True,
                   help="Render synthetic images (disable for metadata-only)")
    p.add_argument("--no-render", dest="render", action="store_false")

    return p.parse_args()


def generate_or_load_annotations(args: argparse.Namespace) -> Path:
    """Generate synthetic annotations or load existing ones."""
    if args.annotations and Path(args.annotations).exists():
        print(f"📂 Loading existing annotations from {args.annotations}")
        return Path(args.annotations)

    from src.data.synthetic_generator import generate_balanced_batch, generate_batch

    count = 50 if args.quick else args.samples
    print(f"🔧 Generating {count} synthetic annotations (seed={args.seed})")

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = data_dir / "annotations.json"

    if args.generation_strategy == "balanced":
        samples = generate_balanced_batch(count=count, seed=args.seed)
    else:
        samples = generate_batch(count=count, seed=args.seed)
    annotations_path.write_text(
        json.dumps([s.model_dump(mode="json") for s in samples], indent=2)
    )

    success_rate = sum(1 for s in samples if s.success) / len(samples)
    print(f"   ✅ {len(samples)} annotations generated ({success_rate:.0%} success rate)")
    print(f"   → {annotations_path}")

    return annotations_path


def prepare_dataset(args: argparse.Namespace, annotations_path: Path) -> tuple:
    """Prepare training dataset with rendered images."""
    from src.data.dataset import prepare_training_dataset, GraspDataset

    data_dir = Path(args.data_dir) / "rendered"
    max_samples = 50 if args.quick else None

    print(f"\n🖼️  Preparing dataset (render={args.render})")
    samples = prepare_training_dataset(
        annotations_path=annotations_path,
        output_dir=data_dir,
        max_samples=max_samples,
        render=args.render,
        seed=args.seed,
    )

    manifest_path = data_dir / "manifest.json"
    success_only = None if args.include_failures else True
    dataset = GraspDataset(manifest_path, success_only=success_only)
    train_ds, eval_ds = dataset.train_test_split(test_pct=args.eval_pct, seed=args.seed)

    if args.include_failures:
        print("   Dataset mode: all attempts included (successes + failures)")
        print("   Note: failed attempts are retained for broader scene coverage and diagnostics.")
    else:
        print("   Dataset mode: successful grasps only")

    print(f"   Train: {len(train_ds)} samples")
    print(f"   Eval:  {len(eval_ds)} samples")

    return train_ds, eval_ds


def _auto_model_class():
    """Return the best available Auto class for vision-language models."""
    from transformers import AutoModel
    try:
        from transformers import AutoModelForVision2Seq
        return AutoModelForVision2Seq
    except ImportError:
        pass
    try:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM
    except ImportError:
        pass
    return AutoModel


def load_model_and_adapter(args: argparse.Namespace):
    """Load base model and apply DoRA/LoRA adapter."""
    import torch
    from transformers import AutoProcessor
    from peft import LoraConfig, get_peft_model, TaskType

    ModelClass = _auto_model_class()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (args.fp16 and device == "cuda") else torch.float32

    print(f"\n🧠 Loading base model: {args.base_model}")
    print(f"   Device: {device}, Dtype: {dtype}")
    print(f"   Model class: {ModelClass.__name__}")

    processor = AutoProcessor.from_pretrained(
        args.base_model, trust_remote_code=True,
    )
    model = ModelClass.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    # Apply DoRA/LoRA adapter
    use_dora = args.adapter_type == "dora"
    peft_config = LoraConfig(
        use_dora=use_dora,
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=args.target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    adapter_name = "DoRA" if use_dora else "LoRA"
    print(f"   {adapter_name} applied: rank={args.rank}, alpha={args.alpha}")
    print(f"   Target modules: {args.target_modules}")
    print(f"   Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    return model, processor, device


def collate_fn(batch: list[dict]) -> dict:
    """Custom collation for grasp training samples."""
    import torch

    result = {}

    # Stack images if present
    if "pixel_values" in batch[0]:
        result["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])
    if "input_ids" in batch[0]:
        result["input_ids"] = torch.stack([b["input_ids"] for b in batch])
        result["attention_mask"] = torch.stack([b["attention_mask"] for b in batch])

    # Actions as tensor
    result["actions"] = torch.tensor([b["action"] for b in batch], dtype=torch.float32)

    # Labels for loss computation (use input_ids as labels for causal LM)
    if "input_ids" in result:
        result["labels"] = result["input_ids"].clone()

    return result


def train(args: argparse.Namespace, model, processor, train_ds, eval_ds, device: str):
    """Run the training loop."""
    from transformers import TrainingArguments, Trainer

    epochs = 2 if args.quick else args.epochs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🏋️ Starting training")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {args.batch_size} × {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation} effective")
    print(f"   Learning rate: {args.lr}")
    print(f"   Output: {output_dir}")

    # Assign processor to datasets for on-the-fly tokenisation
    train_ds.processor = processor
    eval_ds.processor = processor

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16 and device == "cuda",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        logging_dir=str(output_dir / "logs"),
        report_to="none",  # Disable wandb/tensorboard for now
        dataloader_pin_memory=device == "cuda",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
    )

    start = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start

    print(f"\n✅ Training complete")
    print(f"   Time: {elapsed / 60:.1f} min")
    print(f"   Final train loss: {train_result.training_loss:.4f}")

    # Evaluate
    eval_result = trainer.evaluate()
    print(f"   Eval loss: {eval_result['eval_loss']:.4f}")

    return trainer, train_result, eval_result, elapsed


def save_adapter(args: argparse.Namespace, model, train_result, eval_result, elapsed: float):
    """Save adapter weights and training metadata."""
    output_dir = Path(args.output_dir)
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving adapter to {adapter_dir}")
    model.save_pretrained(str(adapter_dir))

    # Save training metadata
    metadata = {
        "profile_name": args.profile_name,
        "base_model": args.base_model,
        "adapter_type": args.adapter_type,
        "rank": args.rank,
        "alpha": args.alpha,
        "target_modules": args.target_modules,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "training_loss": train_result.training_loss,
        "eval_loss": eval_result["eval_loss"],
        "training_time_s": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    meta_path = output_dir / "training_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"   Metadata → {meta_path}")
    print(f"\n🎉 Fine-tuning complete! Adapter ready at: {adapter_dir}")

    return metadata


def dry_run(args: argparse.Namespace, train_ds, eval_ds):
    """Validate the data pipeline without loading/training a model."""
    print(f"\n🧪 Dry run — validating data pipeline (no model)")

    # Check a few samples
    for i in range(min(3, len(train_ds))):
        sample = train_ds[i]
        print(f"\n   Sample {i}:")
        print(f"     Instruction: {sample['instruction']}")
        print(f"     Action: {[f'{a:.3f}' for a in sample['action']]}")
        print(f"     Object: {sample['object_type']}")
        if "image" in sample:
            print(f"     Image shape: {sample['image'].shape}")
        if "depth" in sample:
            print(f"     Depth shape: {sample['depth'].shape}")

    print(f"\n✅ Dry run complete. Data pipeline works.")
    print(f"   Train samples: {len(train_ds)}")
    print(f"   Eval samples:  {len(eval_ds)}")
    print(f"\n   Next: Run without --dry-run to train the model:")
    print(f"   python scripts/train.py --samples {args.samples} --epochs {args.epochs}")


def main():
    args = parse_args()

    print("=" * 60)
    print("  DishSpace — π₀ Fine-Tuning with DoRA")
    print("  Kitchen Dishwashing Grasp Planning")
    print("=" * 60)

    # ── Pre-flight checks ──
    _preflight(args)

    # Step 1: Generate or load annotations
    annotations_path = generate_or_load_annotations(args)

    # Step 2: Prepare dataset
    train_ds, eval_ds = prepare_dataset(args, annotations_path)

    # Step 3: Dry run or full training
    if args.dry_run:
        dry_run(args, train_ds, eval_ds)
        return

    # Step 4: Load model + adapter
    model, processor, device = load_model_and_adapter(args)

    # Step 5: Train
    trainer, train_result, eval_result, elapsed = train(
        args, model, processor, train_ds, eval_ds, device
    )

    # Step 6: Save adapter
    metadata = save_adapter(args, model, train_result, eval_result, elapsed)

    # Step 7: Print next steps
    print(f"\n{'=' * 60}")
    print("  Next steps:")
    print(f"  1. Evaluate: python scripts/evaluate.py --adapter {args.output_dir}/adapter")
    print(f"  2. Deploy:   python scripts/deploy_robot.py --adapter {args.output_dir}/adapter")
    print(f"  3. Compare:  python scripts/evaluate.py --compare-baseline")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
