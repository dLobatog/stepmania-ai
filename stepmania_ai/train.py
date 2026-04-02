"""Training pipeline for StepMania AI models.

Two-phase training:
  1. Onset detector: binary classification on every audio frame
  2. Pattern generator: arrow prediction on onset frames only

Usage:
  smai-train <pack_dir> [--epochs-onset 50] [--epochs-pattern 80]
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from stepmania_ai.data.audio_features import AudioFeatures
from stepmania_ai.data.dataset import (
    BalancedStepChartDataset,
    StepChartDataset,
    discover_sm_files,
)
from stepmania_ai.models.onset_detector import OnsetDetector
from stepmania_ai.models.pattern_generator import PatternGenerator
from stepmania_ai.models.pattern_token_generator import PatternTokenGenerator
from stepmania_ai.models.pattern_vocab import START_TOKEN, VOCAB_SIZE, patterns_to_tokens


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_sm_files(
    sm_files: list[Path],
    validation_split: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """Shuffle songs deterministically, then split by song for validation."""
    files = [Path(p).expanduser() for p in sm_files]
    if len(files) < 2 or validation_split <= 0:
        return sorted(files), []

    rng = np.random.default_rng(seed)
    rng.shuffle(files)

    n_val = max(1, int(round(len(files) * validation_split)))
    n_val = min(n_val, len(files) - 1)
    val_files = sorted(files[:n_val])
    train_files = sorted(files[n_val:])
    return train_files, val_files


def compute_binary_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_onset_detector(
    model: OnsetDetector,
    dataset: StepChartDataset | None,
    criterion: nn.Module,
    batch_size: int,
    device: torch.device,
    val_samples: int | None,
) -> dict[str, float] | None:
    """Evaluate onset detection on a deterministic balanced validation sample."""
    if dataset is None or len(dataset.song_meta) == 0:
        return None

    eval_ds = BalancedStepChartDataset(
        dataset,
        samples_per_epoch=val_samples,
        negative_offset=1,
    )
    loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    total_loss = 0.0
    n_batches = 0
    tp, fp, fn = 0, 0, 0

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            labels = batch["onset_label"].to(device)

            logits = model.forward_framewise(audio).squeeze(-1)
            loss = criterion(logits, labels)

            preds = (torch.sigmoid(logits) > 0.5).float()

            total_loss += loss.item()
            n_batches += 1
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    if n_batches == 0:
        return None

    metrics = compute_binary_metrics(tp, fp, fn)
    metrics["loss"] = total_loss / n_batches
    return metrics


def _log_onset_metrics(
    writer: SummaryWriter,
    prefix: str,
    metrics: dict[str, float],
    epoch: int,
):
    writer.add_scalar(f"{prefix}/epoch_loss", metrics["loss"], epoch)
    writer.add_scalar(f"{prefix}/precision", metrics["precision"], epoch)
    writer.add_scalar(f"{prefix}/recall", metrics["recall"], epoch)
    writer.add_scalar(f"{prefix}/f1", metrics["f1"], epoch)


# ── Phase 1: Onset Detection Training ────────────────────────────────────────


def train_onset_detector(
    dataset: StepChartDataset,
    val_dataset: StepChartDataset | None = None,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: torch.device | None = None,
    save_path: str = "onset_detector.pt",
    log_dir: str = "runs",
    samples_per_epoch: int | None = None,
    val_samples: int | None = 100_000,
    patience: int = 5,
) -> OnsetDetector:
    """Train the onset detection model."""
    device = device or get_device()
    print(f"Training onset detector on {device}")

    writer = SummaryWriter(log_dir=f"{log_dir}/onset_detector")

    train_ds = BalancedStepChartDataset(dataset, samples_per_epoch=samples_per_epoch)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = OnsetDetector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    pos_weight = torch.tensor([1.0 / (dataset.onset_ratio + 1e-8)]).to(device)
    pos_weight = torch.clamp(pos_weight, max=20.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    writer.add_text(
        "hparams",
        f"lr={lr}, batch_size={batch_size}, epochs={epochs}, "
        f"pos_weight={pos_weight.item():.2f}, onset_ratio={dataset.onset_ratio:.4f}, "
        f"train_songs={len(dataset.song_meta)}, val_songs={len(val_dataset.song_meta) if val_dataset else 0}, "
        f"train_samples_per_epoch={len(train_ds)}, val_samples={val_samples}",
    )

    best_score = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        tp, fp, fn = 0, 0, 0

        for batch in tqdm(train_loader, desc=f"Onset epoch {epoch + 1}/{epochs}", leave=False):
            audio = batch["audio"].to(device)
            labels = batch["onset_label"].to(device)

            logits = model.forward_framewise(audio).squeeze(-1)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            global_step += 1

            preds = (torch.sigmoid(logits) > 0.5).float()
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

            if global_step % 50 == 0:
                writer.add_scalar("train_onset/batch_loss", loss.item(), global_step)

        scheduler.step()

        train_metrics = compute_binary_metrics(tp, fp, fn)
        train_metrics["loss"] = total_loss / max(n_batches, 1)
        val_metrics = evaluate_onset_detector(
            model,
            val_dataset,
            criterion,
            batch_size=batch_size,
            device=device,
            val_samples=val_samples,
        )

        _log_onset_metrics(writer, "train_onset", train_metrics, epoch)
        writer.add_scalar("train_onset/learning_rate", scheduler.get_last_lr()[0], epoch)

        if val_metrics is not None:
            _log_onset_metrics(writer, "val_onset", val_metrics, epoch)

        monitored_score = val_metrics["f1"] if val_metrics is not None else train_metrics["f1"]
        status = (
            f"  Train loss: {train_metrics['loss']:.4f} | "
            f"Train F1: {train_metrics['f1']:.3f}"
        )
        if val_metrics is not None:
            status += f" | Val loss: {val_metrics['loss']:.4f} | Val F1: {val_metrics['f1']:.3f}"
        print(status)

        if monitored_score > best_score + 1e-4:
            best_score = monitored_score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best onset model ({'val' if val_metrics else 'train'} F1={monitored_score:.3f})")
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"  Early stopping onset training at epoch {epoch + 1}")
                break

    writer.add_hparams(
        {"lr": lr, "batch_size": batch_size, "epochs": epochs, "patience": patience},
        {"hparam/best_f1": best_score, "hparam/best_epoch": float(best_epoch)},
    )
    writer.close()

    print(f"Best onset F1: {best_score:.3f} (epoch {best_epoch + 1})")
    model.load_state_dict(torch.load(save_path, weights_only=True, map_location=device))
    return model


# ── Phase 2: Pattern Generation Training ─────────────────────────────────────


class PatternDataset(Dataset):
    """Dataset of onset-only frames for training the pattern generator."""

    def __init__(self, base_dataset: StepChartDataset, seq_len: int = 64):
        self.seq_len = seq_len
        self.sequences: list[dict[str, torch.Tensor]] = []

        n_songs = len(base_dataset.song_meta)
        stride = max(1, seq_len // 2)

        for i in tqdm(range(n_songs), desc="Building pattern sequences"):
            song = base_dataset._load_song(i)
            features: AudioFeatures = song["features"]
            onset_labels = song["onset_labels"]
            arrow_labels = song["arrow_labels"]

            onset_frames = np.where(onset_labels > 0.5)[0]
            if len(onset_frames) < seq_len:
                continue

            windows = features.get_context_windows(
                onset_frames,
                window_size=base_dataset.context_frames,
            )

            times = np.array([features.frame_to_time(int(f)) for f in onset_frames], dtype=np.float32)
            deltas = np.diff(times, prepend=times[0])
            arrows = arrow_labels[onset_frames]

            for start in range(0, len(onset_frames) - seq_len + 1, stride):
                end = start + seq_len
                self.sequences.append({
                    "audio_windows": torch.tensor(windows[start:end], dtype=torch.float32),
                    "arrows": torch.tensor(arrows[start:end], dtype=torch.float32),
                    "time_deltas": torch.tensor(deltas[start:end], dtype=torch.float32),
                })

            if i % 50 == 0:
                base_dataset._load_song.cache_clear()

        base_dataset._load_song.cache_clear()
        print(f"Pattern dataset: {len(self.sequences)} sequences of length {seq_len}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        prev_arrows = torch.zeros_like(seq["arrows"])
        prev_arrows[1:] = seq["arrows"][:-1]
        return {
            "audio_windows": seq["audio_windows"],
            "target_arrows": seq["arrows"],
            "prev_arrows": prev_arrows,
            "time_deltas": seq["time_deltas"],
        }


class TokenPatternDataset(Dataset):
    """Dataset of onset-only frames with ergonomic token targets."""

    def __init__(self, base_dataset: StepChartDataset, seq_len: int = 64):
        self.seq_len = seq_len
        self.sequences: list[dict[str, torch.Tensor]] = []
        self.token_counts = np.zeros(VOCAB_SIZE, dtype=np.int64)
        self.exact_matches = 0
        self.total_tokens = 0

        n_songs = len(base_dataset.song_meta)
        stride = max(1, seq_len // 2)

        for i in tqdm(range(n_songs), desc="Building token pattern sequences"):
            song = base_dataset._load_song(i)
            features: AudioFeatures = song["features"]
            onset_labels = song["onset_labels"]
            arrow_labels = song["arrow_labels"]

            onset_frames = np.where(onset_labels > 0.5)[0]
            if len(onset_frames) < seq_len:
                continue

            windows = features.get_context_windows(
                onset_frames,
                window_size=base_dataset.context_frames,
            )

            times = np.array([features.frame_to_time(int(f)) for f in onset_frames], dtype=np.float32)
            deltas = np.diff(times, prepend=times[0])
            tokens, exact = patterns_to_tokens(arrow_labels[onset_frames])
            self.token_counts += np.bincount(tokens, minlength=VOCAB_SIZE)
            self.exact_matches += int(exact.sum())
            self.total_tokens += int(len(tokens))

            for start in range(0, len(onset_frames) - seq_len + 1, stride):
                end = start + seq_len
                self.sequences.append({
                    "audio_windows": torch.tensor(windows[start:end], dtype=torch.float32),
                    "tokens": torch.tensor(tokens[start:end], dtype=torch.long),
                    "time_deltas": torch.tensor(deltas[start:end], dtype=torch.float32),
                })

            if i % 50 == 0:
                base_dataset._load_song.cache_clear()

        base_dataset._load_song.cache_clear()
        coverage = self.exact_coverage * 100.0
        print(
            f"Token pattern dataset: {len(self.sequences)} sequences of length {seq_len} "
            f"| exact vocab coverage {coverage:.1f}%"
        )

    @property
    def exact_coverage(self) -> float:
        return self.exact_matches / max(self.total_tokens, 1)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        prev_tokens = torch.full_like(seq["tokens"], START_TOKEN)
        prev_tokens[1:] = seq["tokens"][:-1]
        return {
            "audio_windows": seq["audio_windows"],
            "target_tokens": seq["tokens"],
            "prev_tokens": prev_tokens,
            "time_deltas": seq["time_deltas"],
        }


def evaluate_pattern_generator(
    model: PatternGenerator,
    dataset: PatternDataset | None,
    criterion: nn.Module,
    batch_size: int,
    device: torch.device,
) -> dict[str, float] | None:
    """Evaluate the pattern model on held-out onset sequences."""
    if dataset is None or len(dataset) == 0:
        return None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    total_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0
    per_arrow_correct = [0, 0, 0, 0]
    per_arrow_total = [0, 0, 0, 0]

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio_windows"].to(device)
            target = batch["target_arrows"].to(device)
            prev = batch["prev_arrows"].to(device)
            td = batch["time_deltas"].to(device)

            logits = model(audio, prev, td)
            loss = criterion(logits, target)
            preds = (torch.sigmoid(logits) > 0.5).float()

            total_loss += loss.item()
            n_batches += 1
            correct += (preds == target).sum().item()
            total += target.numel()

            for arrow_idx in range(4):
                per_arrow_correct[arrow_idx] += (preds[:, :, arrow_idx] == target[:, :, arrow_idx]).sum().item()
                per_arrow_total[arrow_idx] += target[:, :, arrow_idx].numel()

    if n_batches == 0:
        return None

    metrics = {
        "loss": total_loss / n_batches,
        "accuracy": correct / max(total, 1),
    }
    arrow_names = ["left", "down", "up", "right"]
    for arrow_idx, name in enumerate(arrow_names):
        metrics[f"accuracy_{name}"] = per_arrow_correct[arrow_idx] / (per_arrow_total[arrow_idx] + 1e-8)
    return metrics


def train_pattern_generator(
    dataset: StepChartDataset,
    val_dataset: StepChartDataset | None = None,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 5e-4,
    seq_len: int = 64,
    device: torch.device | None = None,
    save_path: str = "pattern_generator.pt",
    log_dir: str = "runs",
    patience: int = 5,
) -> PatternGenerator:
    """Train the pattern generation model."""
    device = device or get_device()
    print(f"\nTraining pattern generator on {device}")

    writer = SummaryWriter(log_dir=f"{log_dir}/pattern_generator")

    train_pattern_ds = PatternDataset(dataset, seq_len=seq_len)
    if len(train_pattern_ds) == 0:
        raise RuntimeError("No onset sequences found for pattern training.")

    val_pattern_ds = PatternDataset(val_dataset, seq_len=seq_len) if val_dataset else None
    if val_pattern_ds is not None and len(val_pattern_ds) == 0:
        print("Validation split produced no pattern sequences; disabling pattern validation.")
        val_pattern_ds = None

    train_loader = DataLoader(train_pattern_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = PatternGenerator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    writer.add_text(
        "hparams",
        f"lr={lr}, batch_size={batch_size}, epochs={epochs}, seq_len={seq_len}, "
        f"train_sequences={len(train_pattern_ds)}, val_sequences={len(val_pattern_ds) if val_pattern_ds else 0}",
    )

    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        correct = 0
        total = 0
        per_arrow_correct = [0, 0, 0, 0]
        per_arrow_total = [0, 0, 0, 0]

        for batch in tqdm(train_loader, desc=f"Pattern epoch {epoch + 1}/{epochs}", leave=False):
            audio = batch["audio_windows"].to(device)
            target = batch["target_arrows"].to(device)
            prev = batch["prev_arrows"].to(device)
            td = batch["time_deltas"].to(device)

            logits = model(audio, prev, td)
            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            global_step += 1

            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == target).sum().item()
            total += target.numel()

            for arrow_idx in range(4):
                per_arrow_correct[arrow_idx] += (preds[:, :, arrow_idx] == target[:, :, arrow_idx]).sum().item()
                per_arrow_total[arrow_idx] += target[:, :, arrow_idx].numel()

            if global_step % 50 == 0:
                writer.add_scalar("train_pattern/batch_loss", loss.item(), global_step)

        scheduler.step()

        train_metrics = {
            "loss": total_loss / max(n_batches, 1),
            "accuracy": correct / max(total, 1),
        }
        arrow_names = ["left", "down", "up", "right"]
        for arrow_idx, name in enumerate(arrow_names):
            train_metrics[f"accuracy_{name}"] = per_arrow_correct[arrow_idx] / (per_arrow_total[arrow_idx] + 1e-8)

        val_metrics = evaluate_pattern_generator(
            model,
            val_pattern_ds,
            criterion,
            batch_size=batch_size,
            device=device,
        )

        writer.add_scalar("train_pattern/epoch_loss", train_metrics["loss"], epoch)
        writer.add_scalar("train_pattern/accuracy", train_metrics["accuracy"], epoch)
        writer.add_scalar("train_pattern/learning_rate", scheduler.get_last_lr()[0], epoch)
        for name in arrow_names:
            writer.add_scalar(f"train_pattern/accuracy_{name}", train_metrics[f"accuracy_{name}"], epoch)

        if val_metrics is not None:
            writer.add_scalar("val_pattern/epoch_loss", val_metrics["loss"], epoch)
            writer.add_scalar("val_pattern/accuracy", val_metrics["accuracy"], epoch)
            for name in arrow_names:
                writer.add_scalar(f"val_pattern/accuracy_{name}", val_metrics[f"accuracy_{name}"], epoch)

        monitored_loss = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
        status = (
            f"  Train loss: {train_metrics['loss']:.4f} | "
            f"Train acc: {train_metrics['accuracy']:.3f}"
        )
        if val_metrics is not None:
            status += f" | Val loss: {val_metrics['loss']:.4f} | Val acc: {val_metrics['accuracy']:.3f}"
        print(status)

        if monitored_loss < best_loss - 1e-4:
            best_loss = monitored_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best pattern model ({'val' if val_metrics else 'train'} loss={monitored_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"  Early stopping pattern training at epoch {epoch + 1}")
                break

    writer.add_hparams(
        {"lr": lr, "batch_size": batch_size, "epochs": epochs, "seq_len": seq_len, "patience": patience},
        {"hparam/best_loss": best_loss, "hparam/best_epoch": float(best_epoch)},
    )
    writer.close()

    print(f"Best pattern loss: {best_loss:.4f} (epoch {best_epoch + 1})")
    model.load_state_dict(torch.load(save_path, weights_only=True, map_location=device))
    return model


def evaluate_pattern_token_generator(
    model: PatternTokenGenerator,
    dataset: TokenPatternDataset | None,
    criterion: nn.Module,
    batch_size: int,
    device: torch.device,
) -> dict[str, float] | None:
    if dataset is None or len(dataset) == 0:
        return None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    total_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio_windows"].to(device)
            target = batch["target_tokens"].to(device)
            prev = batch["prev_tokens"].to(device)
            td = batch["time_deltas"].to(device)

            logits = model(audio, prev, td)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            preds = torch.argmax(logits, dim=-1)

            total_loss += loss.item()
            n_batches += 1
            correct += (preds == target).sum().item()
            total += target.numel()

    if n_batches == 0:
        return None

    return {
        "loss": total_loss / n_batches,
        "accuracy": correct / max(total, 1),
    }


def _save_pattern_checkpoint(
    model: PatternGenerator | PatternTokenGenerator,
    save_path: str,
    pattern_mode: str,
):
    payload = {
        "model_type": "pattern_token_generator" if pattern_mode == "token" else "pattern_generator",
        "pattern_mode": pattern_mode,
        "state_dict": model.state_dict(),
    }
    if pattern_mode == "token":
        payload["vocab_size"] = VOCAB_SIZE
    torch.save(payload, save_path)


def train_pattern_token_generator(
    dataset: StepChartDataset,
    val_dataset: StepChartDataset | None = None,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 5e-4,
    seq_len: int = 64,
    device: torch.device | None = None,
    save_path: str = "pattern_generator.pt",
    log_dir: str = "runs",
    patience: int = 5,
) -> PatternTokenGenerator:
    device = device or get_device()
    print(f"\nTraining token pattern generator on {device}")

    writer = SummaryWriter(log_dir=f"{log_dir}/pattern_generator")
    train_pattern_ds = TokenPatternDataset(dataset, seq_len=seq_len)
    if len(train_pattern_ds) == 0:
        raise RuntimeError("No onset sequences found for token pattern training.")

    val_pattern_ds = TokenPatternDataset(val_dataset, seq_len=seq_len) if val_dataset else None
    if val_pattern_ds is not None and len(val_pattern_ds) == 0:
        print("Validation split produced no token pattern sequences; disabling pattern validation.")
        val_pattern_ds = None

    train_loader = DataLoader(train_pattern_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model = PatternTokenGenerator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    counts = np.maximum(train_pattern_ds.token_counts.astype(np.float32), 1.0)
    class_weights = np.sqrt(counts.sum() / counts)
    class_weights = np.clip(class_weights / class_weights.mean(), 0.5, 3.0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    writer.add_text(
        "hparams",
        f"lr={lr}, batch_size={batch_size}, epochs={epochs}, seq_len={seq_len}, "
        f"train_sequences={len(train_pattern_ds)}, val_sequences={len(val_pattern_ds) if val_pattern_ds else 0}, "
        f"train_vocab_coverage={train_pattern_ds.exact_coverage:.4f}",
    )

    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Token pattern epoch {epoch + 1}/{epochs}", leave=False):
            audio = batch["audio_windows"].to(device)
            target = batch["target_tokens"].to(device)
            prev = batch["prev_tokens"].to(device)
            td = batch["time_deltas"].to(device)

            logits = model(audio, prev, td)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            preds = torch.argmax(logits, dim=-1)
            total_loss += loss.item()
            n_batches += 1
            global_step += 1
            correct += (preds == target).sum().item()
            total += target.numel()

            if global_step % 50 == 0:
                writer.add_scalar("train_pattern/batch_loss", loss.item(), global_step)

        scheduler.step()
        train_metrics = {
            "loss": total_loss / max(n_batches, 1),
            "accuracy": correct / max(total, 1),
        }
        val_metrics = evaluate_pattern_token_generator(
            model,
            val_pattern_ds,
            criterion,
            batch_size=batch_size,
            device=device,
        )

        writer.add_scalar("train_pattern/epoch_loss", train_metrics["loss"], epoch)
        writer.add_scalar("train_pattern/accuracy", train_metrics["accuracy"], epoch)
        writer.add_scalar("train_pattern/learning_rate", scheduler.get_last_lr()[0], epoch)
        writer.add_scalar("train_pattern/exact_vocab_coverage", train_pattern_ds.exact_coverage, epoch)

        if val_metrics is not None:
            writer.add_scalar("val_pattern/epoch_loss", val_metrics["loss"], epoch)
            writer.add_scalar("val_pattern/accuracy", val_metrics["accuracy"], epoch)

        monitored_loss = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
        status = (
            f"  Train loss: {train_metrics['loss']:.4f} | "
            f"Train acc: {train_metrics['accuracy']:.3f}"
        )
        if val_metrics is not None:
            status += f" | Val loss: {val_metrics['loss']:.4f} | Val acc: {val_metrics['accuracy']:.3f}"
        print(status)

        if monitored_loss < best_loss - 1e-4:
            best_loss = monitored_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_pattern_checkpoint(model, save_path, pattern_mode="token")
            print(f"  Saved best token pattern model ({'val' if val_metrics else 'train'} loss={monitored_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"  Early stopping token pattern training at epoch {epoch + 1}")
                break

    writer.add_hparams(
        {"lr": lr, "batch_size": batch_size, "epochs": epochs, "seq_len": seq_len, "patience": patience},
        {"hparam/best_loss": best_loss, "hparam/best_epoch": float(best_epoch)},
    )
    writer.close()

    print(f"Best token pattern loss: {best_loss:.4f} (epoch {best_epoch + 1})")
    payload = torch.load(save_path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    return model


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train StepMania AI models")
    parser.add_argument("pack_dirs", nargs="+", help="Paths to simfile pack directories")
    parser.add_argument("--run-name", type=str, help="Optional name for this training run")
    parser.add_argument("--epochs-onset", type=int, default=50)
    parser.add_argument("--epochs-pattern", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256, help="Onset detector batch size")
    parser.add_argument("--pattern-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="Onset detector learning rate")
    parser.add_argument("--pattern-lr", type=float, default=5e-4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--cache-dir", type=str, default=".cache/features")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--log-dir", type=str)
    parser.add_argument("--difficulty", type=str, default="Challenge")
    parser.add_argument("--pattern-mode", choices=["token", "binary"], default="token")
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--max-songs", type=int)
    parser.add_argument("--train-samples-per-epoch", type=int)
    parser.add_argument("--val-samples", type=int, default=100_000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-workers", type=int)
    parser.add_argument("--skip-onset-training", action="store_true")
    parser.add_argument("--onset-checkpoint", type=str)
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Fast verification preset: fewer songs, fewer epochs, and early stopping.",
    )
    args = parser.parse_args()

    if args.dev:
        dev_defaults = {
            "epochs_onset": 6,
            "epochs_pattern": 6,
            "max_songs": 96,
            "train_samples_per_epoch": 100_000,
            "val_samples": 25_000,
            "patience": 2,
        }
        for name, value in dev_defaults.items():
            if getattr(args, name) == parser.get_default(name):
                setattr(args, name, value)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("checkpoints") / run_name
    log_dir = Path(args.log_dir) if args.log_dir else Path("runs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    sm_files = discover_sm_files(args.pack_dirs)
    if not sm_files:
        print("No songs found! Check your pack directory paths.")
        return

    rng = np.random.default_rng(args.seed)
    rng.shuffle(sm_files)
    if args.max_songs is not None:
        sm_files = sm_files[:args.max_songs]

    train_files, val_files = split_sm_files(sm_files, args.validation_split, args.seed)
    print(
        f"Using {len(train_files)} training songs and "
        f"{len(val_files)} validation songs"
    )

    dataset = StepChartDataset(
        pack_dirs=[],
        sm_files=train_files,
        difficulty=args.difficulty,
        cache_dir=args.cache_dir,
        n_workers=args.n_workers,
    )

    val_dataset = None
    if val_files:
        val_dataset = StepChartDataset(
            pack_dirs=[],
            sm_files=val_files,
            difficulty=args.difficulty,
            cache_dir=args.cache_dir,
            n_workers=args.n_workers,
        )

    if len(dataset.song_meta) == 0:
        print("Training split produced no usable songs.")
        return

    onset_save_path = output_dir / "onset_detector.pt"
    if args.skip_onset_training:
        if not args.onset_checkpoint:
            raise SystemExit("--skip-onset-training requires --onset-checkpoint")
        source = Path(args.onset_checkpoint).expanduser()
        if not source.exists():
            raise SystemExit(f"Onset checkpoint not found: {source}")
        if source.resolve() != onset_save_path.resolve():
            shutil.copy2(source, onset_save_path)
        print(f"Reusing onset checkpoint from {source}")
    else:
        train_onset_detector(
            dataset,
            val_dataset=val_dataset,
            epochs=args.epochs_onset,
            batch_size=args.batch_size,
            lr=args.lr,
            save_path=str(onset_save_path),
            log_dir=str(log_dir),
            samples_per_epoch=args.train_samples_per_epoch,
            val_samples=args.val_samples,
            patience=args.patience,
        )

    if args.pattern_mode == "token":
        train_pattern_token_generator(
            dataset,
            val_dataset=val_dataset,
            epochs=args.epochs_pattern,
            batch_size=args.pattern_batch_size,
            lr=args.pattern_lr,
            seq_len=args.seq_len,
            save_path=str(output_dir / "pattern_generator.pt"),
            log_dir=str(log_dir),
            patience=args.patience,
        )
    else:
        train_pattern_generator(
            dataset,
            val_dataset=val_dataset,
            epochs=args.epochs_pattern,
            batch_size=args.pattern_batch_size,
            lr=args.pattern_lr,
            seq_len=args.seq_len,
            save_path=str(output_dir / "pattern_generator.pt"),
            log_dir=str(log_dir),
            patience=args.patience,
        )

    print("\nTraining complete!")
    print(f"Run name: {run_name}")
    print(f"Models saved to {output_dir}/")
    print(f"Logs saved to {log_dir}/")


if __name__ == "__main__":
    main()
