"""Dataset that pairs audio features with step chart labels.

For each audio frame (~10ms), we produce:
  - onset_label: 1 if there's a note at this frame, 0 otherwise
  - arrow_label: 5-class per column (empty/tap/hold_head/hold_tail/mine) or
                 simplified to 4-bit binary (tap or not per arrow)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from stepmania_ai.data.audio_features import (
    FRAME_RATE,
    AudioFeatures,
    extract_features,
)
from stepmania_ai.utils.sm_parser import Chart, NoteRow, Simfile, parse_sm


def _snap_notes_to_frames(chart: Chart, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Align chart notes to audio frames.

    Returns:
        onset_labels: (n_frames,) binary array — 1 where a note exists
        arrow_labels: (n_frames, 4) binary array — which arrows are active per frame
    """
    onset_labels = np.zeros(n_frames, dtype=np.float32)
    arrow_labels = np.zeros((n_frames, 4), dtype=np.float32)

    for row in chart.note_rows:
        if not row.has_tap:
            continue
        frame = int(round(row.time * FRAME_RATE))
        if 0 <= frame < n_frames:
            onset_labels[frame] = 1.0
            for col in row.tap_columns:
                arrow_labels[frame, col] = 1.0

    return onset_labels, arrow_labels


def build_song_data(
    sm_path: str | Path,
    difficulty: str = "Challenge",
    cache_dir: str | Path | None = None,
) -> dict | None:
    """Build features + labels for a single song.

    Returns dict with keys: features, onset_labels, arrow_labels, title, path
    or None if no matching chart found.
    """
    sm_path = Path(sm_path)

    # Check cache
    if cache_dir:
        cache_path = Path(cache_dir) / f"{sm_path.stem}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    sm = parse_sm(sm_path)
    chart = sm.get_chart(difficulty=difficulty)

    # Fall back to Expert if no Challenge chart
    if chart is None:
        chart = sm.get_chart(difficulty="Expert")
    if chart is None:
        chart = sm.get_chart(difficulty="Hard")
    if chart is None:
        return None

    audio_path = sm.audio_path
    if not audio_path.exists():
        # Try case-insensitive match
        parent = audio_path.parent
        name_lower = audio_path.name.lower()
        for f in parent.iterdir():
            if f.name.lower() == name_lower:
                audio_path = f
                break
        else:
            return None

    features = extract_features(audio_path)
    onset_labels, arrow_labels = _snap_notes_to_frames(chart, features.n_frames)

    data = {
        "features": features,
        "onset_labels": onset_labels,
        "arrow_labels": arrow_labels,
        "title": sm.title,
        "difficulty": chart.difficulty,
        "rating": chart.rating,
        "path": str(sm_path),
    }

    if cache_dir:
        cache_path = Path(cache_dir) / f"{sm_path.stem}.pkl"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    return data


class StepChartDataset(Dataset):
    """PyTorch dataset for training onset detection and arrow placement models.

    Each sample is a context window of audio features centered on a frame,
    paired with onset and arrow labels for that frame.
    """

    def __init__(
        self,
        pack_dirs: list[str | Path],
        difficulty: str = "Challenge",
        context_frames: int = 7,
        cache_dir: str | Path | None = None,
        quality_label: float = 1.0,
    ):
        self.context_frames = context_frames
        self.quality_label = quality_label
        self.songs: list[dict] = []

        # Flatten frame indices: (song_idx, frame_idx)
        self._index_map: list[tuple[int, int]] = []

        # Find all .sm files
        sm_files = []
        for d in pack_dirs:
            d = Path(d)
            if d.is_file() and d.suffix == ".sm":
                sm_files.append(d)
            else:
                sm_files.extend(sorted(d.rglob("*.sm")))

        print(f"Found {len(sm_files)} .sm files")

        for sm_path in tqdm(sm_files, desc="Loading songs"):
            data = build_song_data(sm_path, difficulty=difficulty, cache_dir=cache_dir)
            if data is None:
                continue
            song_idx = len(self.songs)
            self.songs.append(data)
            n_frames = data["features"].n_frames
            for frame_idx in range(n_frames):
                self._index_map.append((song_idx, frame_idx))

        print(f"Loaded {len(self.songs)} songs, {len(self._index_map)} frames total")

        # Compute class balance info
        total_onset = sum(s["onset_labels"].sum() for s in self.songs)
        total_frames = sum(s["features"].n_frames for s in self.songs)
        self.onset_ratio = total_onset / total_frames if total_frames > 0 else 0
        print(f"Onset ratio: {self.onset_ratio:.4f} ({int(total_onset)} notes / {total_frames} frames)")

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> dict:
        song_idx, frame_idx = self._index_map[idx]
        song = self.songs[song_idx]
        features: AudioFeatures = song["features"]

        # Audio context window: (n_features, context_frames)
        window = features.get_context_window(frame_idx, self.context_frames)

        return {
            "audio": torch.tensor(window, dtype=torch.float32),
            "onset_label": torch.tensor(song["onset_labels"][frame_idx], dtype=torch.float32),
            "arrow_label": torch.tensor(song["arrow_labels"][frame_idx], dtype=torch.float32),
            "quality": torch.tensor(self.quality_label, dtype=torch.float32),
        }


class BalancedStepChartDataset(Dataset):
    """Wraps StepChartDataset with balanced sampling for onset detection.

    Since notes are sparse (~1-5% of frames), this oversamples frames with notes
    to get roughly 50/50 balance during training.
    """

    def __init__(self, base_dataset: StepChartDataset, oversample_ratio: float = 0.5):
        self.base = base_dataset

        # Separate positive and negative indices
        self.positive_indices = []
        self.negative_indices = []

        for idx, (song_idx, frame_idx) in enumerate(base_dataset._index_map):
            if base_dataset.songs[song_idx]["onset_labels"][frame_idx] > 0.5:
                self.positive_indices.append(idx)
            else:
                self.negative_indices.append(idx)

        # Target: oversample_ratio fraction of each epoch is positive
        n_pos = len(self.positive_indices)
        n_neg = int(n_pos * (1 - oversample_ratio) / oversample_ratio)
        self._epoch_size = n_pos + n_neg

        print(f"Balanced dataset: {n_pos} positive, {n_neg} negative per epoch "
              f"(from {len(self.negative_indices)} total negative)")

    def __len__(self) -> int:
        return self._epoch_size

    def __getitem__(self, idx: int) -> dict:
        n_pos = len(self.positive_indices)
        if idx < n_pos:
            real_idx = self.positive_indices[idx]
        else:
            # Random sample from negatives
            neg_idx = np.random.randint(len(self.negative_indices))
            real_idx = self.negative_indices[neg_idx]
        return self.base[real_idx]
