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
from functools import lru_cache
from pathlib import Path
import pickle
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover - optional monitoring dependency
    wandb = None

from stepmania_ai.data.audio_features import AudioFeatures
from stepmania_ai.data.dataset import (
    BalancedStepChartDataset,
    StepChartDataset,
    discover_sm_files,
)
from stepmania_ai.models.onset_detector import OnsetDetector
from stepmania_ai.models.hold_note_predictor import HoldNotePredictor
from stepmania_ai.models.hold_utils import bucket_to_duration_beats, quantize_hold_duration
from stepmania_ai.models.pattern_generator import PatternGenerator
from stepmania_ai.models.pattern_token_generator import PatternTokenGenerator
from stepmania_ai.models.pattern_vocab import START_TOKEN, VOCAB_SIZE, patterns_to_tokens

SEQUENCE_CACHE_VERSION = "v1-onset-window-cache"


class MetricLogger:
    """Mirror TensorBoard scalars to W&B when an active run exists."""

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)
        self._wandb_metric_defs_initialized = False
        self._initialize_wandb_metrics()

    def _initialize_wandb_metrics(self) -> None:
        if self._wandb_metric_defs_initialized:
            return
        if wandb is None or wandb.run is None:
            return

        for prefix in [
            "train_onset",
            "val_onset",
            "train_pattern",
            "val_pattern",
            "train_hold",
            "val_hold",
        ]:
            batch_step = f"{prefix}/batch_step"
            epoch_step = f"{prefix}/epoch"
            wandb.define_metric(batch_step)
            wandb.define_metric(epoch_step)
            wandb.define_metric(f"{prefix}/batch_*", step_metric=batch_step)
            wandb.define_metric(f"{prefix}/epoch_*", step_metric=epoch_step)
            wandb.define_metric(f"{prefix}/accuracy*", step_metric=epoch_step)
            wandb.define_metric(f"{prefix}/precision", step_metric=epoch_step)
            wandb.define_metric(f"{prefix}/recall", step_metric=epoch_step)
            wandb.define_metric(f"{prefix}/f1", step_metric=epoch_step)
            wandb.define_metric(f"{prefix}/learning_rate", step_metric=epoch_step)
            wandb.define_metric(f"{prefix}/duration_accuracy", step_metric=epoch_step)
            wandb.define_metric(f"{prefix}/exact_vocab_coverage", step_metric=epoch_step)

        self._wandb_metric_defs_initialized = True

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)
        if wandb is not None and wandb.run is not None:
            self._initialize_wandb_metrics()
            prefix = tag.split("/", 1)[0]
            step_key = f"{prefix}/batch_step" if "/batch_" in tag else f"{prefix}/epoch"
            wandb.log({step_key: step, tag: value})

    def add_text(self, tag: str, text_string: str) -> None:
        self.writer.add_text(tag, text_string)

    def add_hparams(self, hparam_dict: dict[str, float | int], metric_dict: dict[str, float]) -> None:
        self.writer.add_hparams(hparam_dict, metric_dict)
        if wandb is not None and wandb.run is not None:
            wandb.config.update(hparam_dict, allow_val_change=True)
            for key, value in metric_dict.items():
                wandb.run.summary[key] = value

    def close(self) -> None:
        self.writer.close()


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
    writer: MetricLogger,
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

    writer = MetricLogger(log_dir=f"{log_dir}/onset_detector")

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


def _sequence_cache_paths(base_dataset: StepChartDataset, song_idx: int) -> tuple[Path, Path]:
    cache_stem = Path(base_dataset.song_meta[song_idx]["cache_path"]).stem
    root = base_dataset.cache_dir / "sequence_cache"
    root.mkdir(parents=True, exist_ok=True)
    prefix = root / f"{cache_stem}_ctx{base_dataset.context_frames}_{SEQUENCE_CACHE_VERSION}"
    return prefix.with_suffix(".npz"), prefix.with_suffix(".meta.pkl")


def _build_onset_sequence_cache(base_dataset: StepChartDataset, song_idx: int) -> dict:
    data_path, meta_path = _sequence_cache_paths(base_dataset, song_idx)
    song = base_dataset._load_song(song_idx)
    features: AudioFeatures = song["features"]
    onset_frames = np.where(song["onset_labels"] > 0.5)[0].astype(np.int64)

    if len(onset_frames) > 0:
        windows = features.get_context_windows(
            onset_frames,
            window_size=base_dataset.context_frames,
        ).astype(np.float16, copy=False)
        arrows = song["arrow_labels"][onset_frames].astype(np.uint8, copy=False)
        times = onset_frames.astype(np.float32) * (features.hop_length / features.sr)
        deltas = np.diff(times, prepend=times[0]).astype(np.float32, copy=False)
    else:
        windows = np.zeros(
            (0, features.combined_features.shape[0], base_dataset.context_frames),
            dtype=np.float16,
        )
        arrows = np.zeros((0, 4), dtype=np.uint8)
        deltas = np.zeros((0,), dtype=np.float32)

    tokens, exact = patterns_to_tokens(arrows.astype(np.float32, copy=False))
    token_counts = np.bincount(tokens, minlength=VOCAB_SIZE).astype(np.int64, copy=False)

    hold_labels = song.get("hold_start_labels")
    hold_durations = song.get("hold_duration_beats")
    hold_targets = np.zeros(len(onset_frames), dtype=np.uint8)
    duration_targets = np.full(len(onset_frames), -1, dtype=np.int16)
    if hold_labels is not None and hold_durations is not None:
        hold_targets = hold_labels[onset_frames].astype(np.uint8, copy=False)
        for idx, frame in enumerate(onset_frames):
            if hold_targets[idx] > 0 and int(arrows[idx].sum()) == 1:
                duration = float(hold_durations[frame])
                if duration > 0:
                    duration_targets[idx] = quantize_hold_duration(duration)
                else:
                    hold_targets[idx] = 0
            else:
                hold_targets[idx] = 0

    np.savez(
        data_path,
        audio_windows=windows,
        arrows=arrows,
        time_deltas=deltas,
        tokens=tokens.astype(np.int16, copy=False),
        hold_targets=hold_targets,
        duration_targets=duration_targets,
    )

    meta = {
        "cache_path": str(data_path),
        "n_onsets": int(len(onset_frames)),
        "token_counts": token_counts,
        "exact_matches": int(exact.sum()),
        "total_tokens": int(len(tokens)),
        "n_hold_targets": int(hold_targets.sum()),
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    return meta


def _load_or_build_sequence_cache(base_dataset: StepChartDataset, song_idx: int) -> dict:
    data_path, meta_path = _sequence_cache_paths(base_dataset, song_idx)
    if data_path.exists() and meta_path.exists():
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            if Path(meta.get("cache_path", "")) == data_path:
                return meta
        except Exception:
            pass
    return _build_onset_sequence_cache(base_dataset, song_idx)


class _CachedOnsetSequenceDataset(Dataset):
    prepare_desc = "Preparing onset sequence caches"

    def __init__(self, base_dataset: StepChartDataset, seq_len: int = 64):
        self.base_dataset = base_dataset
        self.seq_len = seq_len
        self.stride = max(1, seq_len // 2)
        self.song_cache_meta: list[dict] = []
        self._song_sequence_counts: list[int] = []
        self._total_sequences = 0

        self._init_stats()

        for i in tqdm(range(len(base_dataset.song_meta)), desc=self.prepare_desc):
            meta = _load_or_build_sequence_cache(base_dataset, i)
            self.song_cache_meta.append(meta)
            n_onsets = int(meta["n_onsets"])
            n_sequences = 0 if n_onsets < seq_len else 1 + (n_onsets - seq_len) // self.stride
            self._song_sequence_counts.append(n_sequences)
            self._update_stats(meta)
            if i % 50 == 0:
                base_dataset._load_song.cache_clear()

        base_dataset._load_song.cache_clear()
        self._song_cum_sequences = np.concatenate([[0], np.cumsum(self._song_sequence_counts, dtype=np.int64)])
        self._total_sequences = int(self._song_cum_sequences[-1])
        self._load_song_cache = lru_cache(maxsize=8)(self._load_song_cache_uncached)
        self._print_summary()

    def _init_stats(self) -> None:
        pass

    def _update_stats(self, meta: dict) -> None:
        pass

    def _print_summary(self) -> None:
        print(f"Cached onset sequence dataset: {self._total_sequences} sequences of length {self.seq_len}")

    def _load_song_cache_uncached(self, song_idx: int) -> dict[str, np.ndarray]:
        cache_path = Path(self.song_cache_meta[song_idx]["cache_path"])
        with np.load(cache_path, allow_pickle=False) as payload:
            return {name: payload[name] for name in payload.files}

    def _sequence_to_song(self, idx: int) -> tuple[int, int]:
        song_idx = int(np.searchsorted(self._song_cum_sequences[1:], idx, side="right"))
        local_idx = idx - int(self._song_cum_sequences[song_idx])
        return song_idx, local_idx

    def __len__(self) -> int:
        return self._total_sequences

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        song_idx, local_idx = self._sequence_to_song(idx)
        start = local_idx * self.stride
        end = start + self.seq_len
        song_cache = self._load_song_cache(song_idx)
        return self._build_item(song_cache, start, end)

    def _build_item(
        self,
        song_cache: dict[str, np.ndarray],
        start: int,
        end: int,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class PatternDataset(_CachedOnsetSequenceDataset):
    """Dataset of onset-only frames for training the pattern generator."""

    prepare_desc = "Preparing pattern caches"

    def _build_item(
        self,
        song_cache: dict[str, np.ndarray],
        start: int,
        end: int,
    ) -> dict[str, torch.Tensor]:
        arrows = torch.tensor(song_cache["arrows"][start:end], dtype=torch.float32)
        prev_arrows = torch.zeros_like(arrows)
        prev_arrows[1:] = arrows[:-1]
        return {
            "audio_windows": torch.tensor(song_cache["audio_windows"][start:end], dtype=torch.float32),
            "target_arrows": arrows,
            "prev_arrows": prev_arrows,
            "time_deltas": torch.tensor(song_cache["time_deltas"][start:end], dtype=torch.float32),
        }


class TokenPatternDataset(_CachedOnsetSequenceDataset):
    """Dataset of onset-only frames with ergonomic token targets."""

    prepare_desc = "Preparing token pattern caches"

    def _init_stats(self) -> None:
        self.token_counts = np.zeros(VOCAB_SIZE, dtype=np.int64)
        self.exact_matches = 0
        self.total_tokens = 0

    def _update_stats(self, meta: dict) -> None:
        self.token_counts += np.asarray(meta["token_counts"], dtype=np.int64)
        self.exact_matches += int(meta["exact_matches"])
        self.total_tokens += int(meta["total_tokens"])

    def _print_summary(self) -> None:
        coverage = self.exact_coverage * 100.0
        print(
            f"Token pattern dataset: {self._total_sequences} sequences of length {self.seq_len} "
            f"| exact vocab coverage {coverage:.1f}%"
        )

    @property
    def exact_coverage(self) -> float:
        return self.exact_matches / max(self.total_tokens, 1)

    def _build_item(
        self,
        song_cache: dict[str, np.ndarray],
        start: int,
        end: int,
    ) -> dict[str, torch.Tensor]:
        tokens = torch.tensor(song_cache["tokens"][start:end], dtype=torch.long)
        prev_tokens = torch.full_like(tokens, START_TOKEN)
        prev_tokens[1:] = tokens[:-1]
        return {
            "audio_windows": torch.tensor(song_cache["audio_windows"][start:end], dtype=torch.float32),
            "target_tokens": tokens,
            "prev_tokens": prev_tokens,
            "time_deltas": torch.tensor(song_cache["time_deltas"][start:end], dtype=torch.float32),
        }


class HoldDataset(_CachedOnsetSequenceDataset):
    """Dataset of onset sequences labeled with simple hold starts and durations."""

    prepare_desc = "Preparing hold caches"

    def _init_stats(self) -> None:
        self.hold_ratio = 0.0
        self.n_hold_targets = 0
        self.total_steps = 0

    def _update_stats(self, meta: dict) -> None:
        self.total_steps += int(meta["n_onsets"])
        self.n_hold_targets += int(meta["n_hold_targets"])

    def _print_summary(self) -> None:
        self.hold_ratio = self.n_hold_targets / max(self.total_steps, 1)
        print(
            f"Hold dataset: {self._total_sequences} sequences of length {self.seq_len} "
            f"| hold ratio {self.hold_ratio:.4f}"
        )

    def _build_item(
        self,
        song_cache: dict[str, np.ndarray],
        start: int,
        end: int,
    ) -> dict[str, torch.Tensor]:
        arrows = torch.tensor(song_cache["arrows"][start:end], dtype=torch.float32)
        prev_arrows = torch.zeros_like(arrows)
        prev_arrows[1:] = arrows[:-1]
        return {
            "audio_windows": torch.tensor(song_cache["audio_windows"][start:end], dtype=torch.float32),
            "current_arrows": arrows,
            "prev_arrows": prev_arrows,
            "hold_targets": torch.tensor(song_cache["hold_targets"][start:end], dtype=torch.float32),
            "duration_targets": torch.tensor(song_cache["duration_targets"][start:end], dtype=torch.long),
            "time_deltas": torch.tensor(song_cache["time_deltas"][start:end], dtype=torch.float32),
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

    writer = MetricLogger(log_dir=f"{log_dir}/pattern_generator")

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
    transition_loss_weight: float = 0.0,
) -> dict[str, float] | None:
    if dataset is None or len(dataset) == 0:
        return None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    total_loss = 0.0
    total_token_loss = 0.0
    total_transition_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio_windows"].to(device)
            target = batch["target_tokens"].to(device)
            prev = batch["prev_tokens"].to(device)
            td = batch["time_deltas"].to(device)

            logits, next_logits = model.forward_with_aux(audio, prev, td)
            token_loss = criterion(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            transition_loss = torch.zeros((), device=device)
            if transition_loss_weight > 0 and target.shape[1] > 1:
                transition_loss = criterion(
                    next_logits[:, :-1].reshape(-1, next_logits.shape[-1]),
                    target[:, 1:].reshape(-1),
                )
            loss = token_loss + transition_loss_weight * transition_loss
            preds = torch.argmax(logits, dim=-1)

            total_loss += loss.item()
            total_token_loss += token_loss.item()
            total_transition_loss += transition_loss.item()
            n_batches += 1
            correct += (preds == target).sum().item()
            total += target.numel()

    if n_batches == 0:
        return None

    return {
        "loss": total_loss / n_batches,
        "token_loss": total_token_loss / n_batches,
        "transition_loss": total_transition_loss / n_batches,
        "accuracy": correct / max(total, 1),
    }


def evaluate_hold_note_predictor(
    model: HoldNotePredictor,
    dataset: HoldDataset | None,
    hold_criterion: nn.Module,
    duration_criterion: nn.Module,
    batch_size: int,
    device: torch.device,
) -> dict[str, float] | None:
    if dataset is None or len(dataset) == 0:
        return None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    total_loss = 0.0
    n_batches = 0
    tp = fp = fn = 0
    duration_correct = 0
    duration_total = 0

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio_windows"].to(device)
            current = batch["current_arrows"].to(device)
            prev = batch["prev_arrows"].to(device)
            hold_targets = batch["hold_targets"].to(device)
            duration_targets = batch["duration_targets"].to(device)
            td = batch["time_deltas"].to(device)

            hold_logits, duration_logits = model(audio, current, prev, td)
            hold_loss = hold_criterion(hold_logits, hold_targets)
            duration_loss = duration_criterion(
                duration_logits.reshape(-1, duration_logits.shape[-1]),
                duration_targets.reshape(-1),
            )
            loss = hold_loss + 0.5 * duration_loss

            preds = (torch.sigmoid(hold_logits) > 0.5).float()
            total_loss += loss.item()
            n_batches += 1
            tp += ((preds == 1) & (hold_targets == 1)).sum().item()
            fp += ((preds == 1) & (hold_targets == 0)).sum().item()
            fn += ((preds == 0) & (hold_targets == 1)).sum().item()

            positive_mask = duration_targets >= 0
            if positive_mask.any():
                duration_pred = torch.argmax(duration_logits, dim=-1)
                duration_correct += (duration_pred[positive_mask] == duration_targets[positive_mask]).sum().item()
                duration_total += positive_mask.sum().item()

    if n_batches == 0:
        return None

    metrics = compute_binary_metrics(tp, fp, fn)
    metrics["loss"] = total_loss / n_batches
    metrics["duration_accuracy"] = duration_correct / max(duration_total, 1)
    return metrics


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


def _save_hold_checkpoint(model: HoldNotePredictor, save_path: str):
    torch.save(
        {
            "model_type": "hold_note_predictor",
            "state_dict": model.state_dict(),
        },
        save_path,
    )


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
    transition_loss_weight: float = 0.25,
) -> PatternTokenGenerator:
    device = device or get_device()
    print(f"\nTraining token pattern generator on {device}")

    writer = MetricLogger(log_dir=f"{log_dir}/pattern_generator")
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
        f"train_vocab_coverage={train_pattern_ds.exact_coverage:.4f}, "
        f"transition_loss_weight={transition_loss_weight:.3f}",
    )

    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_token_loss = 0.0
        total_transition_loss = 0.0
        n_batches = 0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Token pattern epoch {epoch + 1}/{epochs}", leave=False):
            audio = batch["audio_windows"].to(device)
            target = batch["target_tokens"].to(device)
            prev = batch["prev_tokens"].to(device)
            td = batch["time_deltas"].to(device)

            logits, next_logits = model.forward_with_aux(audio, prev, td)
            token_loss = criterion(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            transition_loss = torch.zeros((), device=device)
            if transition_loss_weight > 0 and target.shape[1] > 1:
                transition_loss = criterion(
                    next_logits[:, :-1].reshape(-1, next_logits.shape[-1]),
                    target[:, 1:].reshape(-1),
                )
            loss = token_loss + transition_loss_weight * transition_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            preds = torch.argmax(logits, dim=-1)
            total_loss += loss.item()
            total_token_loss += token_loss.item()
            total_transition_loss += transition_loss.item()
            n_batches += 1
            global_step += 1
            correct += (preds == target).sum().item()
            total += target.numel()

            if global_step % 50 == 0:
                writer.add_scalar("train_pattern/batch_loss", loss.item(), global_step)

        scheduler.step()
        train_metrics = {
            "loss": total_loss / max(n_batches, 1),
            "token_loss": total_token_loss / max(n_batches, 1),
            "transition_loss": total_transition_loss / max(n_batches, 1),
            "accuracy": correct / max(total, 1),
        }
        val_metrics = evaluate_pattern_token_generator(
            model,
            val_pattern_ds,
            criterion,
            batch_size=batch_size,
            device=device,
            transition_loss_weight=transition_loss_weight,
        )

        writer.add_scalar("train_pattern/epoch_loss", train_metrics["loss"], epoch)
        writer.add_scalar("train_pattern/token_loss", train_metrics["token_loss"], epoch)
        writer.add_scalar("train_pattern/transition_loss", train_metrics["transition_loss"], epoch)
        writer.add_scalar("train_pattern/accuracy", train_metrics["accuracy"], epoch)
        writer.add_scalar("train_pattern/learning_rate", scheduler.get_last_lr()[0], epoch)
        writer.add_scalar("train_pattern/exact_vocab_coverage", train_pattern_ds.exact_coverage, epoch)

        if val_metrics is not None:
            writer.add_scalar("val_pattern/epoch_loss", val_metrics["loss"], epoch)
            writer.add_scalar("val_pattern/token_loss", val_metrics["token_loss"], epoch)
            writer.add_scalar("val_pattern/transition_loss", val_metrics["transition_loss"], epoch)
            writer.add_scalar("val_pattern/accuracy", val_metrics["accuracy"], epoch)

        monitored_loss = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
        status = (
            f"  Train loss: {train_metrics['loss']:.4f} | "
            f"Train acc: {train_metrics['accuracy']:.3f} | "
            f"Train next: {train_metrics['transition_loss']:.4f}"
        )
        if val_metrics is not None:
            status += (
                f" | Val loss: {val_metrics['loss']:.4f} | "
                f"Val acc: {val_metrics['accuracy']:.3f} | "
                f"Val next: {val_metrics['transition_loss']:.4f}"
            )
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
        {
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "seq_len": seq_len,
            "patience": patience,
            "transition_loss_weight": transition_loss_weight,
        },
        {"hparam/best_loss": best_loss, "hparam/best_epoch": float(best_epoch)},
    )
    writer.close()

    print(f"Best token pattern loss: {best_loss:.4f} (epoch {best_epoch + 1})")
    payload = torch.load(save_path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    return model


def train_hold_note_predictor(
    dataset: StepChartDataset,
    val_dataset: StepChartDataset | None = None,
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 5e-4,
    seq_len: int = 64,
    device: torch.device | None = None,
    save_path: str = "hold_note_predictor.pt",
    log_dir: str = "runs",
    patience: int = 5,
) -> HoldNotePredictor:
    device = device or get_device()
    print(f"\nTraining hold note predictor on {device}")

    writer = MetricLogger(log_dir=f"{log_dir}/hold_note_predictor")
    train_hold_ds = HoldDataset(dataset, seq_len=seq_len)
    if len(train_hold_ds) == 0:
        raise RuntimeError("No onset sequences found for hold training.")

    val_hold_ds = HoldDataset(val_dataset, seq_len=seq_len) if val_dataset else None
    if val_hold_ds is not None and len(val_hold_ds) == 0:
        print("Validation split produced no hold sequences; disabling hold validation.")
        val_hold_ds = None

    train_loader = DataLoader(train_hold_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model = HoldNotePredictor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    pos_weight = torch.tensor([1.0 / max(train_hold_ds.hold_ratio, 1e-4)], dtype=torch.float32, device=device)
    pos_weight = torch.clamp(pos_weight, max=20.0)
    hold_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    duration_criterion = nn.CrossEntropyLoss(ignore_index=-1)

    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        tp = fp = fn = 0
        duration_correct = 0
        duration_total = 0

        for batch in tqdm(train_loader, desc=f"Hold epoch {epoch + 1}/{epochs}", leave=False):
            audio = batch["audio_windows"].to(device)
            current = batch["current_arrows"].to(device)
            prev = batch["prev_arrows"].to(device)
            hold_targets = batch["hold_targets"].to(device)
            duration_targets = batch["duration_targets"].to(device)
            td = batch["time_deltas"].to(device)

            hold_logits, duration_logits = model(audio, current, prev, td)
            hold_loss = hold_criterion(hold_logits, hold_targets)
            duration_loss = duration_criterion(
                duration_logits.reshape(-1, duration_logits.shape[-1]),
                duration_targets.reshape(-1),
            )
            loss = hold_loss + 0.5 * duration_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            preds = (torch.sigmoid(hold_logits) > 0.5).float()
            total_loss += loss.item()
            n_batches += 1
            global_step += 1
            tp += ((preds == 1) & (hold_targets == 1)).sum().item()
            fp += ((preds == 1) & (hold_targets == 0)).sum().item()
            fn += ((preds == 0) & (hold_targets == 1)).sum().item()

            positive_mask = duration_targets >= 0
            if positive_mask.any():
                duration_pred = torch.argmax(duration_logits, dim=-1)
                duration_correct += (duration_pred[positive_mask] == duration_targets[positive_mask]).sum().item()
                duration_total += positive_mask.sum().item()

            if global_step % 50 == 0:
                writer.add_scalar("train_hold/batch_loss", loss.item(), global_step)

        scheduler.step()
        train_metrics = compute_binary_metrics(tp, fp, fn)
        train_metrics["loss"] = total_loss / max(n_batches, 1)
        train_metrics["duration_accuracy"] = duration_correct / max(duration_total, 1)

        val_metrics = evaluate_hold_note_predictor(
            model,
            val_hold_ds,
            hold_criterion,
            duration_criterion,
            batch_size=batch_size,
            device=device,
        )

        for prefix, metrics in [("train_hold", train_metrics)] + ([("val_hold", val_metrics)] if val_metrics else []):
            if metrics is None:
                continue
            writer.add_scalar(f"{prefix}/epoch_loss", metrics["loss"], epoch)
            writer.add_scalar(f"{prefix}/precision", metrics["precision"], epoch)
            writer.add_scalar(f"{prefix}/recall", metrics["recall"], epoch)
            writer.add_scalar(f"{prefix}/f1", metrics["f1"], epoch)
            writer.add_scalar(f"{prefix}/duration_accuracy", metrics["duration_accuracy"], epoch)
        writer.add_scalar("train_hold/learning_rate", scheduler.get_last_lr()[0], epoch)

        monitored_loss = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
        status = (
            f"  Train loss: {train_metrics['loss']:.4f} | "
            f"Train F1: {train_metrics['f1']:.3f} | "
            f"Train duration acc: {train_metrics['duration_accuracy']:.3f}"
        )
        if val_metrics is not None:
            status += (
                f" | Val loss: {val_metrics['loss']:.4f} | "
                f"Val F1: {val_metrics['f1']:.3f} | "
                f"Val duration acc: {val_metrics['duration_accuracy']:.3f}"
            )
        print(status)

        if monitored_loss < best_loss - 1e-4:
            best_loss = monitored_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_hold_checkpoint(model, save_path)
            print(f"  Saved best hold model ({'val' if val_metrics else 'train'} loss={monitored_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"  Early stopping hold training at epoch {epoch + 1}")
                break

    writer.close()
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
    parser.add_argument("--epochs-hold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256, help="Onset detector batch size")
    parser.add_argument("--pattern-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="Onset detector learning rate")
    parser.add_argument("--pattern-lr", type=float, default=5e-4)
    parser.add_argument("--transition-loss-weight", type=float, default=0.25)
    parser.add_argument("--hold-lr", type=float, default=5e-4)
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
    parser.add_argument(
        "--song-timeout-seconds",
        type=float,
        default=0,
        help="If > 0, isolate per-song feature extraction in subprocesses and skip songs that exceed this timeout.",
    )
    parser.add_argument("--skip-onset-training", action="store_true")
    parser.add_argument("--onset-checkpoint", type=str)
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, help="W&B entity or team name")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
        help="W&B logging mode when --wandb is enabled",
    )
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
            "epochs_hold": 4,
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
        song_timeout_seconds=args.song_timeout_seconds,
    )

    val_dataset = None
    if val_files:
        val_dataset = StepChartDataset(
            pack_dirs=[],
            sm_files=val_files,
            difficulty=args.difficulty,
            cache_dir=args.cache_dir,
            n_workers=args.n_workers,
            song_timeout_seconds=args.song_timeout_seconds,
        )

    if len(dataset.song_meta) == 0:
        print("Training split produced no usable songs.")
        return

    if args.wandb:
        if wandb is None:
            raise SystemExit("--wandb requested but wandb is not installed. Run `pip install wandb`.")
        wandb.init(
            project=args.wandb_project or "stepmania-ai",
            entity=args.wandb_entity,
            name=run_name,
            mode=args.wandb_mode,
            dir=str(log_dir.resolve()),
            config={
                "run_name": run_name,
                "epochs_onset": args.epochs_onset,
                "epochs_pattern": args.epochs_pattern,
                "epochs_hold": args.epochs_hold,
                "batch_size": args.batch_size,
                "pattern_batch_size": args.pattern_batch_size,
                "lr": args.lr,
                "pattern_lr": args.pattern_lr,
                "transition_loss_weight": args.transition_loss_weight,
                "hold_lr": args.hold_lr,
                "seq_len": args.seq_len,
                "difficulty": args.difficulty,
                "pattern_mode": args.pattern_mode,
                "validation_split": args.validation_split,
                "max_songs": args.max_songs,
                "train_samples_per_epoch": args.train_samples_per_epoch,
                "val_samples": args.val_samples,
                "patience": args.patience,
                "seed": args.seed,
                "pack_dirs": [str(Path(p).expanduser()) for p in args.pack_dirs],
                "train_song_count": len(train_files),
                "val_song_count": len(val_files),
                "loaded_train_songs": len(dataset.song_meta),
                "loaded_val_songs": len(val_dataset.song_meta) if val_dataset else 0,
            },
        )

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
            transition_loss_weight=args.transition_loss_weight,
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

    if args.epochs_hold > 0:
        train_hold_note_predictor(
            dataset,
            val_dataset=val_dataset,
            epochs=args.epochs_hold,
            batch_size=args.pattern_batch_size,
            lr=args.hold_lr,
            seq_len=args.seq_len,
            save_path=str(output_dir / "hold_note_predictor.pt"),
            log_dir=str(log_dir),
            patience=args.patience,
        )

    print("\nTraining complete!")
    print(f"Run name: {run_name}")
    print(f"Models saved to {output_dir}/")
    print(f"Logs saved to {log_dir}/")
    if args.wandb and wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
