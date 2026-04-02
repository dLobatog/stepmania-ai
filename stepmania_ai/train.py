"""Training pipeline for StepMania AI models.

Two-phase training:
  1. Onset detector: binary classification on every audio frame
  2. Pattern generator: arrow prediction on onset frames only

Usage:
  smai-train <pack_dir> [--epochs 50] [--batch-size 64] [--lr 1e-3]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from stepmania_ai.data.audio_features import AudioFeatures
from stepmania_ai.data.dataset import (
    BalancedStepChartDataset,
    StepChartDataset,
    build_song_data,
)
from stepmania_ai.models.onset_detector import OnsetDetector
from stepmania_ai.models.pattern_generator import PatternGenerator


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Phase 1: Onset Detection Training ────────────────────────────────────────


def train_onset_detector(
    dataset: StepChartDataset,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: torch.device | None = None,
    save_path: str = "onset_detector.pt",
) -> OnsetDetector:
    """Train the onset detection model."""
    device = device or get_device()
    print(f"Training onset detector on {device}")

    balanced = BalancedStepChartDataset(dataset, oversample_ratio=0.5)
    loader = DataLoader(balanced, batch_size=batch_size, shuffle=True, num_workers=0)

    model = OnsetDetector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Weighted BCE to handle remaining imbalance
    pos_weight = torch.tensor([1.0 / (dataset.onset_ratio + 1e-8)]).to(device)
    pos_weight = torch.clamp(pos_weight, max=20.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        tp, fp, fn = 0, 0, 0

        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            audio = batch["audio"].to(device)  # (B, n_feat, ctx)
            labels = batch["onset_label"].to(device)  # (B,)

            # Single-frame mode (no GRU sequence)
            logits = model.forward_framewise(audio).squeeze(-1)  # (B,)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Track metrics
            preds = (torch.sigmoid(logits) > 0.5).float()
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

        scheduler.step()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        avg_loss = total_loss / n_batches

        print(f"  Loss: {avg_loss:.4f} | P: {precision:.3f} R: {recall:.3f} F1: {f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model (F1={f1:.3f})")

    print(f"Best F1: {best_f1:.3f}")
    model.load_state_dict(torch.load(save_path, weights_only=True))
    return model


# ── Phase 2: Pattern Generation Training ─────────────────────────────────────


class PatternDataset(Dataset):
    """Dataset of onset-only frames for training the pattern generator.

    Extracts sequences of notes (only frames with onsets) from each song,
    with their audio features and time deltas.
    """

    def __init__(self, base_dataset: StepChartDataset, seq_len: int = 64):
        self.seq_len = seq_len
        self.sequences: list[dict] = []

        for song in base_dataset.songs:
            features: AudioFeatures = song["features"]
            onset_labels = song["onset_labels"]
            arrow_labels = song["arrow_labels"]

            # Get onset frame indices
            onset_frames = np.where(onset_labels > 0.5)[0]
            if len(onset_frames) < 4:
                continue

            # Get audio windows for onset frames
            windows = []
            for f in onset_frames:
                windows.append(features.get_context_window(f, base_dataset.context_frames))
            windows = np.stack(windows)  # (n_onsets, n_feat, ctx)

            # Time deltas between consecutive notes
            times = np.array([features.frame_to_time(f) for f in onset_frames])
            deltas = np.diff(times, prepend=times[0])

            # Arrow labels for onset frames
            arrows = arrow_labels[onset_frames]  # (n_onsets, 4)

            # Split into subsequences
            n_onsets = len(onset_frames)
            for start in range(0, n_onsets - seq_len + 1, seq_len // 2):
                end = start + seq_len
                if end > n_onsets:
                    break
                self.sequences.append({
                    "audio_windows": torch.tensor(windows[start:end], dtype=torch.float32),
                    "arrows": torch.tensor(arrows[start:end], dtype=torch.float32),
                    "time_deltas": torch.tensor(deltas[start:end], dtype=torch.float32),
                })

        print(f"Pattern dataset: {len(self.sequences)} sequences of length {seq_len}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        # Shift arrows right for teacher forcing (previous note's arrows)
        prev_arrows = torch.zeros_like(seq["arrows"])
        prev_arrows[1:] = seq["arrows"][:-1]
        return {
            "audio_windows": seq["audio_windows"],
            "target_arrows": seq["arrows"],
            "prev_arrows": prev_arrows,
            "time_deltas": seq["time_deltas"],
        }


def train_pattern_generator(
    dataset: StepChartDataset,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 5e-4,
    seq_len: int = 64,
    device: torch.device | None = None,
    save_path: str = "pattern_generator.pt",
) -> PatternGenerator:
    """Train the pattern generation model."""
    device = device or get_device()
    print(f"\nTraining pattern generator on {device}")

    pattern_ds = PatternDataset(dataset, seq_len=seq_len)
    loader = DataLoader(pattern_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = PatternGenerator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        correct = 0
        total = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            audio = batch["audio_windows"].to(device)
            target = batch["target_arrows"].to(device)
            prev = batch["prev_arrows"].to(device)
            td = batch["time_deltas"].to(device)

            logits = model(audio, prev, td)  # (B, S, 4)
            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == target).sum().item()
            total += target.numel()

        scheduler.step()

        avg_loss = total_loss / n_batches
        accuracy = correct / total if total > 0 else 0

        print(f"  Loss: {avg_loss:.4f} | Arrow accuracy: {accuracy:.3f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model (loss={avg_loss:.4f})")

    model.load_state_dict(torch.load(save_path, weights_only=True))
    return model


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train StepMania AI models")
    parser.add_argument("pack_dirs", nargs="+", help="Paths to simfile pack directories")
    parser.add_argument("--epochs-onset", type=int, default=50)
    parser.add_argument("--epochs-pattern", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cache-dir", type=str, default=".cache/features")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--difficulty", type=str, default="Challenge")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset
    dataset = StepChartDataset(
        pack_dirs=args.pack_dirs,
        difficulty=args.difficulty,
        cache_dir=args.cache_dir,
    )

    if len(dataset.songs) == 0:
        print("No songs found! Check your pack directory paths.")
        return

    # Phase 1: Onset detector
    onset_model = train_onset_detector(
        dataset,
        epochs=args.epochs_onset,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=str(output_dir / "onset_detector.pt"),
    )

    # Phase 2: Pattern generator
    pattern_model = train_pattern_generator(
        dataset,
        epochs=args.epochs_pattern,
        batch_size=32,
        lr=5e-4,
        save_path=str(output_dir / "pattern_generator.pt"),
    )

    print("\nTraining complete!")
    print(f"Models saved to {output_dir}/")


if __name__ == "__main__":
    main()
