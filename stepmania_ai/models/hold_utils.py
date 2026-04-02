"""Utilities for first-pass hold-note modeling."""

from __future__ import annotations

import numpy as np


HOLD_DURATION_BUCKETS_BEATS: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)


def quantize_hold_duration(duration_beats: float) -> int:
    values = np.asarray(HOLD_DURATION_BUCKETS_BEATS, dtype=np.float32)
    return int(np.argmin(np.abs(values - float(duration_beats))))


def bucket_to_duration_beats(bucket: int) -> float:
    return float(HOLD_DURATION_BUCKETS_BEATS[int(bucket)])
