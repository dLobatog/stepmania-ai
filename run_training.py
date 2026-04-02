#!/usr/bin/env python3
"""Convenience script for fast local training runs."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from stepmania_ai.data.dataset import StepChartDataset, discover_sm_files
from stepmania_ai.train import (
    split_sm_files,
    train_onset_detector,
    train_pattern_generator,
)


DEFAULT_SMALL_PACK = "~/Downloads/StepmaniaPipelineSmaller"
DEFAULT_FULL_PACK = "~/Downloads/StepmaniaPipeline"


def main():
    parser = argparse.ArgumentParser(description="Run a quick StepMania AI training pass")
    parser.add_argument(
        "--pack-dir",
        default=os.environ.get("STEP_PACK_DIR", DEFAULT_SMALL_PACK),
        help="Pack directory to train on. Defaults to the smaller local pack.",
    )
    parser.add_argument(
        "--full-pack-dir",
        default=os.environ.get("STEP_FULL_PACK_DIR", DEFAULT_FULL_PACK),
        help="Reference path for the larger pack, documented for convenience.",
    )
    parser.add_argument("--difficulty", default="Challenge")
    parser.add_argument("--cache-dir", default=".cache/features")
    parser.add_argument("--run-name")
    parser.add_argument("--output-dir")
    parser.add_argument("--log-dir")
    parser.add_argument("--max-songs", type=int, default=96)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--epochs-onset", type=int, default=6)
    parser.add_argument("--epochs-pattern", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pattern-batch-size", type=int, default=32)
    parser.add_argument("--train-samples-per-epoch", type=int, default=100_000)
    parser.add_argument("--val-samples", type=int, default=25_000)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-workers", type=int, default=4)
    args = parser.parse_args()

    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("checkpoints") / run_name
    log_dir = Path(args.log_dir) if args.log_dir else Path("runs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    pack_dir = Path(os.path.expanduser(args.pack_dir))
    full_pack_dir = Path(os.path.expanduser(args.full_pack_dir))
    print(f"Quick pack: {pack_dir}")
    print(f"Full pack:  {full_pack_dir}")

    sm_files = discover_sm_files([pack_dir])
    if not sm_files:
        raise SystemExit(f"No .sm files found under {pack_dir}")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(sm_files)
    sm_files = sm_files[:args.max_songs]
    train_files, val_files = split_sm_files(sm_files, args.validation_split, args.seed)
    print(f"Using {len(train_files)} training songs and {len(val_files)} validation songs")

    train_dataset = StepChartDataset(
        pack_dirs=[],
        sm_files=train_files,
        difficulty=args.difficulty,
        cache_dir=args.cache_dir,
        n_workers=args.n_workers,
    )
    val_dataset = StepChartDataset(
        pack_dirs=[],
        sm_files=val_files,
        difficulty=args.difficulty,
        cache_dir=args.cache_dir,
        n_workers=args.n_workers,
    ) if val_files else None

    print("\n=== Starting onset detector training ===")
    train_onset_detector(
        train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs_onset,
        batch_size=args.batch_size,
        lr=1e-3,
        samples_per_epoch=args.train_samples_per_epoch,
        val_samples=args.val_samples,
        patience=args.patience,
        save_path=str(output_dir / "onset_detector.pt"),
        log_dir=str(log_dir),
    )

    print("\n=== Starting pattern generator training ===")
    train_pattern_generator(
        train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs_pattern,
        batch_size=args.pattern_batch_size,
        lr=5e-4,
        seq_len=64,
        patience=args.patience,
        save_path=str(output_dir / "pattern_generator.pt"),
        log_dir=str(log_dir),
    )

    print("\nAll done!")
    print(f"Run name: {run_name}")
    print(f"Models:   {output_dir}")
    print(f"Logs:     {log_dir}")


if __name__ == "__main__":
    main()
