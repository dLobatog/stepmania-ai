"""Shared ergonomic pattern vocabulary utilities."""

from __future__ import annotations

import numpy as np


ERGONOMIC_PATTERN_VOCAB: tuple[tuple[int, int, int, int], ...] = (
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1),
    (1, 1, 0, 0),
    (1, 0, 1, 0),
    (1, 0, 0, 1),
    (0, 1, 1, 0),
    (0, 1, 0, 1),
    (0, 0, 1, 1),
    (1, 1, 1, 0),
    (1, 1, 0, 1),
    (1, 0, 1, 1),
    (0, 1, 1, 1),
    (1, 1, 1, 1),
)

VOCAB_SIZE = len(ERGONOMIC_PATTERN_VOCAB)
_PATTERN_TO_INDEX = {pattern: idx for idx, pattern in enumerate(ERGONOMIC_PATTERN_VOCAB)}


def start_token(vocab_size: int = VOCAB_SIZE) -> int:
    return int(vocab_size)


START_TOKEN = start_token()


def get_vocab_patterns(vocab_size: int = VOCAB_SIZE) -> np.ndarray:
    vocab_size = min(int(vocab_size), VOCAB_SIZE)
    return np.asarray(ERGONOMIC_PATTERN_VOCAB[:vocab_size], dtype=np.float32)


def pattern_activity(pattern: np.ndarray | list[float] | tuple[int, ...]) -> int:
    return int(sum(normalize_pattern(pattern)))


def normalize_pattern(pattern: np.ndarray | list[float] | tuple[int, ...]) -> tuple[int, int, int, int]:
    arr = np.asarray(pattern, dtype=np.float32).reshape(-1)
    bits = tuple(int(x > 0.5) for x in arr[:4])
    if sum(bits) == 0:
        return ERGONOMIC_PATTERN_VOCAB[0]
    return bits  # type: ignore[return-value]


def pattern_to_token(pattern: np.ndarray | list[float] | tuple[int, ...]) -> tuple[int, bool]:
    bits = normalize_pattern(pattern)
    idx = _PATTERN_TO_INDEX.get(bits)
    if idx is not None:
        return idx, True

    best_idx = 0
    best_key: tuple[int, int, int] | None = None
    for idx, candidate in enumerate(ERGONOMIC_PATTERN_VOCAB):
        distance = sum(abs(a - b) for a, b in zip(bits, candidate))
        overlap = sum(1 for a, b in zip(bits, candidate) if a == 1 and b == 1)
        activity = sum(candidate)
        key = (distance, -overlap, activity)
        if best_key is None or key < best_key:
            best_key = key
            best_idx = idx

    return best_idx, False


def patterns_to_tokens(patterns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(patterns, dtype=np.float32)
    tokens = np.zeros(arr.shape[0], dtype=np.int64)
    exact = np.zeros(arr.shape[0], dtype=bool)
    for i, pattern in enumerate(arr):
        token, is_exact = pattern_to_token(pattern)
        tokens[i] = token
        exact[i] = is_exact
    return tokens, exact


def token_to_pattern(token: int) -> np.ndarray:
    return np.asarray(ERGONOMIC_PATTERN_VOCAB[int(token)], dtype=np.float32)
