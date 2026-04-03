"""Dataset that pairs audio features with step chart labels.

Memory-efficient design: only metadata is held in RAM during init.
Song data is loaded from disk cache on-the-fly during training,
with an LRU cache to keep recently-used songs hot.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import pickle
import subprocess
import sys
from functools import lru_cache, partial
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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


CACHE_VERSION = "v3-holds"


def _snap_notes_to_frames(
    chart: Chart,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align chart notes to audio frames.

    Returns:
        onset_labels: (n_frames,) binary array — 1 where a note exists
        arrow_labels: (n_frames, 4) binary array — which arrows are active per frame
        hold_start_labels: (n_frames,) binary array — 1 where a simple hold starts
        hold_duration_beats: (n_frames,) float array — duration in beats for simple hold starts
        roll_start_labels: (n_frames,) binary array — 1 where a simple roll starts
    """
    onset_labels = np.zeros(n_frames, dtype=np.float32)
    arrow_labels = np.zeros((n_frames, 4), dtype=np.float32)
    hold_start_labels = np.zeros(n_frames, dtype=np.float32)
    hold_duration_beats = np.zeros(n_frames, dtype=np.float32)
    roll_start_labels = np.zeros(n_frames, dtype=np.float32)
    active_heads: dict[int, tuple[int, float, bool, bool]] = {}

    for row in chart.note_rows:
        frame = int(round(row.time * FRAME_RATE))
        if not row.has_tap:
            if row.tail_columns:
                for col in row.tail_columns:
                    start = active_heads.pop(col, None)
                    if start is None:
                        continue
                    start_frame, start_beat, is_roll, is_trackable = start
                    if is_trackable and not is_roll and 0 <= start_frame < n_frames:
                        hold_duration_beats[start_frame] = max(0.0, row.beat - start_beat)
            continue

        if 0 <= frame < n_frames:
            onset_labels[frame] = 1.0
            for col in row.tap_columns:
                arrow_labels[frame, col] = 1.0

        is_simple_single = len(row.tap_columns) == 1

        for col in row.hold_head_columns:
            is_trackable = is_simple_single and len(row.hold_head_columns) == 1 and len(row.roll_head_columns) == 0
            active_heads[col] = (frame, row.beat, False, is_trackable)
            if is_trackable and 0 <= frame < n_frames:
                hold_start_labels[frame] = 1.0

        for col in row.roll_head_columns:
            is_trackable = is_simple_single and len(row.roll_head_columns) == 1 and len(row.hold_head_columns) == 0
            active_heads[col] = (frame, row.beat, True, is_trackable)
            if is_trackable and 0 <= frame < n_frames:
                roll_start_labels[frame] = 1.0

        for col in row.tail_columns:
            start = active_heads.pop(col, None)
            if start is None:
                continue
            start_frame, start_beat, is_roll, is_trackable = start
            if is_trackable and 0 <= start_frame < n_frames:
                if is_roll:
                    roll_start_labels[start_frame] = 1.0
                else:
                    hold_duration_beats[start_frame] = max(0.0, row.beat - start_beat)

    return onset_labels, arrow_labels, hold_start_labels, hold_duration_beats, roll_start_labels


def _cache_key(sm_path: Path) -> str:
    payload = f"{CACHE_VERSION}:{sm_path.resolve()}"
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def discover_sm_files(pack_dirs: list[str | Path]) -> list[Path]:
    """Find and sort all simfiles under one or more pack directories."""
    sm_files: list[Path] = []
    for d in pack_dirs:
        path = Path(d).expanduser()
        if path.is_file() and path.suffix.lower() == ".sm":
            sm_files.append(path)
        elif path.exists():
            sm_files.extend(sorted(path.rglob("*.sm")))
    return sorted(sm_files)


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

    key = _cache_key(sm_path)
    if cache_dir:
        cache_path = Path(cache_dir) / f"{key}_{sm_path.stem}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass

    sm = parse_sm(sm_path)
    chart = sm.get_chart(difficulty=difficulty)

    if chart is None:
        chart = sm.get_chart(difficulty="Expert")
    if chart is None:
        chart = sm.get_chart(difficulty="Hard")
    if chart is None:
        return None

    audio_path = sm.audio_path
    if not audio_path.exists():
        parent = audio_path.parent
        name_lower = audio_path.name.lower()
        for f in parent.iterdir():
            if f.name.lower() == name_lower:
                audio_path = f
                break
        else:
            return None

    features = extract_features(audio_path, skip_beats=True)
    onset_labels, arrow_labels, hold_start_labels, hold_duration_beats, roll_start_labels = _snap_notes_to_frames(
        chart,
        features.n_frames,
    )

    data = {
        "features": features,
        "onset_labels": onset_labels,
        "arrow_labels": arrow_labels,
        "hold_start_labels": hold_start_labels,
        "hold_duration_beats": hold_duration_beats,
        "roll_start_labels": roll_start_labels,
        "title": sm.title,
        "difficulty": chart.difficulty,
        "rating": chart.rating,
        "path": str(sm_path),
    }

    if cache_dir:
        cache_path = Path(cache_dir) / f"{key}_{sm_path.stem}.pkl"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    return data


def _extract_and_cache(sm_path: Path, difficulty: str, cache_dir: str) -> dict | None:
    """Worker function: extract and cache, return only metadata (not full data)."""
    data = build_song_data(sm_path, difficulty=difficulty, cache_dir=cache_dir)
    if data is None:
        return None
    # Return only metadata — don't keep full features in the result
    return {
        "n_frames": data["features"].n_frames,
        "n_onsets": int(data["onset_labels"].sum()),
        "title": data["title"],
        "path": str(sm_path),
        "cache_path": str(Path(cache_dir) / f"{_cache_key(sm_path)}_{sm_path.stem}.pkl"),
    }


def _extract_and_cache_via_subprocess(
    sm_path: Path,
    difficulty: str,
    cache_dir: str,
    timeout_seconds: float,
) -> dict | None:
    """Run extraction in an isolated Python subprocess.

    This is slower than the in-process worker path, but protects the main
    training process from native-library hangs or segfaults in feature extraction.
    """
    cmd = [
        sys.executable,
        "-m",
        "stepmania_ai.data.extract_song_cache",
        "--sm-path",
        str(sm_path),
        "--difficulty",
        difficulty,
        "--cache-dir",
        cache_dir,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"Skipping {sm_path}: extraction timed out after {timeout_seconds:.0f}s")
        return None

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        suffix = f" ({detail[-1]})" if detail else ""
        print(f"Skipping {sm_path}: extractor exited with code {proc.returncode}{suffix}")
        return None

    payload = proc.stdout.strip()
    if not payload:
        return None

    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        print(f"Skipping {sm_path}: invalid extractor output ({exc})")
        return None


class StepChartDataset(Dataset):
    """Memory-efficient dataset for training.

    Only holds song metadata and cache paths in RAM. Song data is loaded
    from disk cache on-demand with an LRU cache holding the N most recent songs.

    Each epoch randomly samples frames across all songs with balanced
    onset/non-onset sampling.
    """

    def __init__(
        self,
        pack_dirs: list[str | Path],
        sm_files: list[str | Path] | None = None,
        difficulty: str = "Challenge",
        context_frames: int = 7,
        cache_dir: str | Path = ".cache/features",
        quality_label: float = 1.0,
        n_workers: int | None = None,
        lru_size: int = 32,
        max_songs: int | None = None,
        song_timeout_seconds: float | None = None,
    ):
        self.context_frames = context_frames
        self.quality_label = quality_label
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if sm_files is None:
            sm_files = discover_sm_files(pack_dirs)
        else:
            sm_files = [Path(p).expanduser() for p in sm_files]

        if max_songs is not None:
            sm_files = sm_files[:max_songs]

        print(f"Found {len(sm_files)} .sm files")

        if n_workers is None:
            n_workers = min(mp.cpu_count(), 4)

        # Extract features to cache and collect metadata
        if song_timeout_seconds is not None and song_timeout_seconds > 0:
            worker_fn = partial(
                _extract_and_cache_via_subprocess,
                difficulty=difficulty,
                cache_dir=str(self.cache_dir),
                timeout_seconds=float(song_timeout_seconds),
            )
            print(
                f"Extracting features with {n_workers} isolated workers "
                f"(timeout {song_timeout_seconds:.0f}s per song)..."
            )
            results: list[dict | None] = []
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(worker_fn, path) for path in sm_files]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Loading songs"):
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        print(f"Skipping song after worker error: {exc}")
                        results.append(None)
        else:
            worker_fn = partial(_extract_and_cache, difficulty=difficulty, cache_dir=str(self.cache_dir))
            if n_workers > 1:
                print(f"Extracting features with {n_workers} workers...")
                with mp.Pool(n_workers, maxtasksperchild=1) as pool:
                    results = list(
                        tqdm(
                            pool.imap_unordered(worker_fn, sm_files),
                            total=len(sm_files),
                            desc="Loading songs",
                        )
                    )
            else:
                results = [worker_fn(p) for p in tqdm(sm_files, desc="Loading songs")]

        # Store only lightweight metadata
        self.song_meta = sorted(
            (r for r in results if r is not None),
            key=lambda item: item["path"],
        )
        self._song_frame_counts = np.array([s["n_frames"] for s in self.song_meta], dtype=np.int64)
        self._song_cumframes = np.concatenate([[0], np.cumsum(self._song_frame_counts)])
        self._total_frames = int(self._song_cumframes[-1])

        total_onsets = sum(s["n_onsets"] for s in self.song_meta)
        self.onset_ratio = total_onsets / self._total_frames if self._total_frames > 0 else 0

        print(f"Loaded {len(self.song_meta)} songs, {self._total_frames} frames total")
        print(f"Onset ratio: {self.onset_ratio:.4f} ({total_onsets} notes / {self._total_frames} frames)")

        # LRU cache for loaded songs
        self._lru_size = lru_size
        self._load_song = lru_cache(maxsize=lru_size)(self._load_song_uncached)

    def _load_song_uncached(self, song_idx: int) -> dict:
        """Load full song data from cache."""
        cache_path = self.song_meta[song_idx]["cache_path"]
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    def _frame_to_song(self, global_frame: int) -> tuple[int, int]:
        """Convert global frame index to (song_idx, local_frame_idx)."""
        song_idx = int(np.searchsorted(self._song_cumframes[1:], global_frame, side="right"))
        local_frame = global_frame - int(self._song_cumframes[song_idx])
        return song_idx, local_frame

    def __len__(self) -> int:
        return self._total_frames

    def __getitem__(self, idx: int) -> dict:
        song_idx, frame_idx = self._frame_to_song(idx)
        song = self._load_song(song_idx)
        features: AudioFeatures = song["features"]

        window = features.get_context_window(frame_idx, self.context_frames)

        return {
            "audio": torch.tensor(window, dtype=torch.float32),
            "onset_label": torch.tensor(song["onset_labels"][frame_idx], dtype=torch.float32),
            "arrow_label": torch.tensor(song["arrow_labels"][frame_idx], dtype=torch.float32),
            "quality": torch.tensor(self.quality_label, dtype=torch.float32),
        }

    # Expose songs property for pattern dataset compatibility
    @property
    def songs(self) -> list[dict]:
        """Load all songs — only use for pattern dataset (small subset of frames)."""
        return [self._load_song(i) for i in range(len(self.song_meta))]


class BalancedStepChartDataset(Dataset):
    """Balanced sampling for onset detection training.

    Instead of iterating over every frame (22M+), samples a fixed number of
    frames per epoch with ~50/50 onset balance. Much faster epochs.
    """

    def __init__(
        self,
        base: StepChartDataset,
        samples_per_epoch: int | None = None,
        negative_offset: int = 0,
    ):
        self.base = base

        # Build onset/non-onset indices per song (lightweight — just frame numbers)
        self._positive_by_song: list[np.ndarray] = []
        self._negative_by_song: list[np.ndarray] = []
        total_pos = 0
        total_neg = 0

        for i, meta in enumerate(base.song_meta):
            song = base._load_song(i)
            onsets = song["onset_labels"]
            pos = np.where(onsets > 0.5)[0]
            neg = np.where(onsets <= 0.5)[0]
            self._positive_by_song.append(pos)
            self._negative_by_song.append(neg)
            total_pos += len(pos)
            total_neg += len(neg)
            # Clear from LRU if we're just scanning
            base._load_song.cache_clear()

        # Flatten positive indices as (song_idx, frame_idx) — only onsets, so manageable size
        self._all_positive = []
        for song_idx, pos in enumerate(self._positive_by_song):
            for f in pos:
                self._all_positive.append((song_idx, int(f)))
        self._all_positive = np.array(self._all_positive, dtype=np.int64)  # (n_pos, 2)

        # For negatives, store counts — sample randomly during __getitem__
        self._neg_counts = np.array([len(n) for n in self._negative_by_song])
        self._neg_cum_counts = np.concatenate([[0], np.cumsum(self._neg_counts)])
        self._total_neg = int(self._neg_cum_counts[-1])
        self._negative_offset = negative_offset % max(self._total_neg, 1)

        n_pos = len(self._all_positive)
        if samples_per_epoch is None:
            # 2x positives (once as positive, once matched with a negative)
            samples_per_epoch = n_pos * 2
        self._epoch_size = samples_per_epoch

        print(f"Balanced dataset: {n_pos} positive, {self._total_neg} negative available, "
              f"{self._epoch_size} samples per epoch")

    def __len__(self) -> int:
        return self._epoch_size

    def _sample_negative(self, idx: int) -> tuple[int, int]:
        if self._total_neg == 0:
            return int(self._all_positive[0][0]), int(self._all_positive[0][1])

        neg_global_idx = ((idx // 2) * 15485863 + self._negative_offset) % self._total_neg
        song_idx = int(np.searchsorted(self._neg_cum_counts[1:], neg_global_idx, side="right"))
        local_neg_idx = neg_global_idx - int(self._neg_cum_counts[song_idx])
        frame_idx = int(self._negative_by_song[song_idx][local_neg_idx])
        return song_idx, frame_idx

    def __getitem__(self, idx: int) -> dict:
        n_pos = len(self._all_positive)

        if idx % 2 == 0 and n_pos > 0:
            # Positive sample
            pos_idx = idx // 2 % n_pos
            song_idx, frame_idx = self._all_positive[pos_idx]
        else:
            song_idx, frame_idx = self._sample_negative(idx)

        song = self.base._load_song(int(song_idx))
        features: AudioFeatures = song["features"]
        window = features.get_context_window(int(frame_idx), self.base.context_frames)

        return {
            "audio": torch.tensor(window, dtype=torch.float32),
            "onset_label": torch.tensor(song["onset_labels"][int(frame_idx)], dtype=torch.float32),
            "arrow_label": torch.tensor(song["arrow_labels"][int(frame_idx)], dtype=torch.float32),
            "quality": torch.tensor(self.base.quality_label, dtype=torch.float32),
        }
