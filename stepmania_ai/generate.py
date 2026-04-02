"""Inference pipeline: generate a .sm file from audio.

Takes an audio file, runs onset detection + pattern generation, and writes
a valid StepMania .sm file.

Usage:
  smai-generate <audio_file> [--onset-model checkpoints/onset_detector.pt]
                              [--pattern-model checkpoints/pattern_generator.pt]
                              [--output song.sm]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from stepmania_ai.data.audio_features import (
    FRAME_RATE,
    HOP_LENGTH,
    SAMPLE_RATE,
    AudioFeatures,
    extract_features,
)
from stepmania_ai.models.onset_detector import OnsetDetector
from stepmania_ai.models.pattern_generator import PatternGenerator
from stepmania_ai.models.pattern_token_generator import PatternTokenGenerator
from stepmania_ai.models.pattern_vocab import ERGONOMIC_PATTERN_VOCAB, VOCAB_SIZE, token_to_pattern


ARROW_CHARS = "0123"  # left, down, up, right


def _pattern_is_jump(pattern: np.ndarray) -> bool:
    return int(pattern.sum()) >= 2


def _pattern_single_column(pattern: np.ndarray) -> int | None:
    active = np.flatnonzero(pattern)
    if len(active) == 1:
        return int(active[0])
    return None


def _stairs_like(last_four_columns: list[int]) -> bool:
    if len(last_four_columns) != 4:
        return False
    return last_four_columns in ([0, 1, 2, 3], [3, 2, 1, 0])


def _transition_penalty(
    candidate: np.ndarray,
    history: list[np.ndarray],
    delta_t: float,
) -> float:
    """Rule-based penalty that discourages awkward local transitions.

    This is intentionally simple and explicit so we can ablate it later.
    """
    penalty = 0.0
    candidate_is_jump = _pattern_is_jump(candidate)
    candidate_col = _pattern_single_column(candidate)

    if candidate_is_jump:
        penalty -= 0.15
        if delta_t < 0.25:
            penalty -= 0.75
        if delta_t < 0.18:
            penalty -= 0.75

    if history:
        prev = history[-1]
        prev_is_jump = _pattern_is_jump(prev)
        prev_col = _pattern_single_column(prev)

        if candidate_is_jump and prev_is_jump:
            penalty -= 0.65
            if np.array_equal(candidate, prev):
                penalty -= 0.5

        if (
            candidate_col is not None
            and prev_col is not None
            and candidate_col == prev_col
        ):
            if delta_t < 0.20:
                penalty -= 1.75
            elif delta_t < 0.35:
                penalty -= 0.85
            else:
                penalty -= 0.25

        if candidate_is_jump and not prev_is_jump and delta_t < 0.20:
            penalty -= 0.4

    if len(history) >= 3 and candidate_col is not None:
        last_three_cols = [_pattern_single_column(p) for p in history[-3:]]
        if all(col is not None for col in last_three_cols):
            four_cols = [int(col) for col in last_three_cols] + [candidate_col]
            if _stairs_like(four_cols):
                penalty -= 1.0

    recent_jumps = sum(1 for pattern in history[-4:] if _pattern_is_jump(pattern))
    if candidate_is_jump and recent_jumps >= 2:
        penalty -= 0.65

    return penalty


def _candidate_scores(
    step_logits: torch.Tensor,
    candidate_patterns: torch.Tensor,
) -> torch.Tensor:
    """Score constrained pattern candidates under the factorized arrow model."""
    logits = step_logits.unsqueeze(0).expand(candidate_patterns.shape[0], -1)
    return (
        F.logsigmoid(logits) * candidate_patterns
        + F.logsigmoid(-logits) * (1.0 - candidate_patterns)
    ).sum(dim=1)


def load_pattern_model(
    pattern_model_path: str | Path,
    device: torch.device,
) -> tuple[PatternGenerator | PatternTokenGenerator, str]:
    payload = torch.load(pattern_model_path, map_location=device)

    if isinstance(payload, dict) and "model_type" in payload and "state_dict" in payload:
        model_type = payload["model_type"]
        if model_type == "pattern_token_generator":
            model = PatternTokenGenerator(vocab_size=int(payload.get("vocab_size", VOCAB_SIZE)))
            model.load_state_dict(payload["state_dict"])
            model.to(device)
            return model, "token"
        if model_type == "pattern_generator":
            model = PatternGenerator()
            model.load_state_dict(payload["state_dict"])
            model.to(device)
            return model, "binary"

    model = PatternGenerator()
    model.load_state_dict(payload)
    model.to(device)
    return model, "binary"


def detect_onsets(
    features: AudioFeatures,
    model: OnsetDetector,
    device: torch.device,
    threshold: float = 0.5,
    min_gap_ms: float = 50.0,
    context_window: int = 7,
) -> np.ndarray:
    """Run onset detection on full song, return frame indices of detected onsets."""
    model.eval()
    n_frames = features.n_frames
    min_gap_frames = int(min_gap_ms / 1000.0 * FRAME_RATE)

    # Process in chunks to avoid OOM
    chunk_size = 2048
    all_probs = np.zeros(n_frames)

    with torch.no_grad():
        starts = range(0, n_frames, chunk_size)
        starts = tqdm(starts, desc="Scoring onset chunks", total=(n_frames + chunk_size - 1) // chunk_size, leave=False)
        for start in starts:
            end = min(start + chunk_size, n_frames)
            windows = features.get_context_windows(
                np.arange(start, end),
                window_size=context_window,
            )
            batch = torch.tensor(windows, dtype=torch.float32).to(device)
            logits = model.forward_framewise(batch).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs[start:end] = probs

    # Peak picking with minimum gap
    onset_frames = []
    candidates = np.where(all_probs > threshold)[0]

    for frame in candidates:
        if not onset_frames or (frame - onset_frames[-1]) >= min_gap_frames:
            onset_frames.append(frame)

    return np.array(onset_frames)


def generate_patterns(
    features: AudioFeatures,
    onset_frames: np.ndarray,
    model: PatternGenerator | PatternTokenGenerator,
    device: torch.device,
    temperature: float = 0.8,
    decode_strategy: str = "ergonomic",
    pattern_mode: str = "binary",
    context_window: int = 7,
    max_history_steps: int = 64,
) -> np.ndarray:
    """Generate arrow patterns for detected onsets."""
    model.eval()
    n_onsets = len(onset_frames)

    if n_onsets == 0:
        return np.zeros((0, 4))

    # Build audio windows for onset frames
    windows = features.get_context_windows(onset_frames, window_size=context_window)
    audio_windows = torch.tensor(
        windows, dtype=torch.float32
    ).unsqueeze(0).to(device)  # (1, n_onsets, n_feat, ctx)

    # Time deltas
    times = np.array([features.frame_to_time(f) for f in onset_frames])
    deltas = np.diff(times, prepend=times[0])
    time_deltas = torch.tensor(deltas, dtype=torch.float32).unsqueeze(0).to(device)

    if pattern_mode == "binary" and decode_strategy == "raw":
        patterns = model.generate(
            audio_windows,
            time_deltas,
            temperature=temperature,
            show_progress=True,
            max_history_steps=max_history_steps,
        )
        return patterns.cpu().numpy()

    candidate_patterns_np = np.asarray(ERGONOMIC_PATTERN_VOCAB, dtype=np.float32)
    candidate_patterns = torch.tensor(candidate_patterns_np, dtype=torch.float32, device=device)

    generated = torch.zeros(n_onsets, 4, device=device)
    history: list[np.ndarray] = []

    if pattern_mode == "binary":
        prev_arrows = torch.zeros(1, n_onsets, 4, device=device)
    else:
        prev_tokens = torch.full(
            (1, n_onsets),
            VOCAB_SIZE,
            dtype=torch.long,
            device=device,
        )

    steps = tqdm(range(n_onsets), total=n_onsets, desc="Generating patterns", leave=False)
    for t in steps:
        start = max(0, t + 1 - max_history_steps)
        if pattern_mode == "binary":
            logits = model(
                audio_windows[:, start:t + 1],
                prev_arrows[:, start:t + 1],
                time_deltas[:, start:t + 1],
            )
            step_logits = logits[0, -1] / temperature
            raw_scores = _candidate_scores(step_logits, candidate_patterns).detach().cpu().numpy()
        else:
            logits = model(
                audio_windows[:, start:t + 1],
                prev_tokens[:, start:t + 1],
                time_deltas[:, start:t + 1],
            )
            step_logits = logits[0, -1] / temperature
            raw_scores = F.log_softmax(step_logits, dim=-1).detach().cpu().numpy()
        if decode_strategy == "ergonomic":
            penalties = np.array(
                [
                    _transition_penalty(candidate, history, float(deltas[t]))
                    for candidate in candidate_patterns_np
                ],
                dtype=np.float32,
            )
        else:
            penalties = np.zeros(len(candidate_patterns_np), dtype=np.float32)
        best_idx = int(np.argmax(raw_scores + penalties))
        best_pattern = candidate_patterns[best_idx]

        generated[t] = best_pattern
        history.append(candidate_patterns_np[best_idx])

        if t + 1 < n_onsets:
            if pattern_mode == "binary":
                prev_arrows[0, t + 1] = best_pattern
            else:
                prev_tokens[0, t + 1] = best_idx

    return generated.cpu().numpy()


def quantize_to_beats(
    onset_times: np.ndarray,
    bpm: float,
    offset: float = 0.0,
    snap: int = 16,
) -> list[tuple[int, int]]:
    """Quantize onset times to beat positions (measure, subdivision).

    Returns list of (measure_number, row_within_measure) tuples.
    snap: quantization grid (4=quarter, 8=eighth, 16=sixteenth, etc.)
    """
    beats_per_measure = 4.0
    secs_per_beat = 60.0 / bpm
    secs_per_subdivision = secs_per_beat * beats_per_measure / snap

    positions = []
    for t in onset_times:
        adjusted = t - offset
        if adjusted < 0:
            continue
        subdivision = round(adjusted / secs_per_subdivision)
        measure = subdivision // snap
        row = subdivision % snap
        positions.append((measure, row))

    return positions


def write_sm_file(
    output_path: str | Path,
    audio_filename: str,
    title: str,
    artist: str,
    bpm: float,
    offset: float,
    onset_times: np.ndarray,
    arrow_patterns: np.ndarray,
    difficulty: str = "Challenge",
    rating: int = 10,
    snap: int = 16,
):
    """Write a valid .sm file from generated chart data."""
    # Quantize onsets to beat grid
    positions = quantize_to_beats(onset_times, bpm, offset, snap)

    if not positions:
        print("Warning: no valid note positions after quantization")
        return

    # Build measure data
    max_measure = max(m for m, _ in positions) + 1

    # Create measure arrays (snap rows per measure)
    measures: list[list[str]] = []
    for _ in range(max_measure):
        measures.append(["0000"] * snap)

    # Place arrows
    for (measure, row), arrows in zip(positions, arrow_patterns):
        if measure < len(measures) and row < snap:
            arrow_str = ""
            for a in arrows:
                arrow_str += "1" if a > 0.5 else "0"
            # Ensure at least one arrow
            if arrow_str == "0000":
                arrow_str = "1000"
            measures[measure][row] = arrow_str

    # Format .sm content
    lines = [
        f"#TITLE:{title};",
        f"#SUBTITLE:;",
        f"#ARTIST:{artist};",
        f"#TITLETRANSLIT:;",
        f"#SUBTITLETRANSLIT:;",
        f"#ARTISTTRANSLIT:;",
        f"#CREDIT:StepMania AI;",
        f"#BANNER:;",
        f"#BACKGROUND:;",
        f"#LYRICSPATH:;",
        f"#CDTITLE:;",
        f"#MUSIC:{audio_filename};",
        f"#OFFSET:{-offset:.3f};",
        f"#SAMPLESTART:30.000;",
        f"#SAMPLELENGTH:15.000;",
        f"#SELECTABLE:YES;",
        f"#DISPLAYBPM:{bpm:.3f};",
        f"#BPMS:0.000={bpm:.3f};",
        f"#STOPS:;",
        f"#BGCHANGES:;",
        f"",
        f"//---------------dance-single - ----------------",
        f"#NOTES:",
        f"     dance-single:",
        f"     StepMania AI:",
        f"     {difficulty}:",
        f"     {rating}:",
        f"     0.000,0.000,0.000,0.000,0.000:",
    ]

    for i, measure in enumerate(measures):
        for row in measure:
            lines.append(row)
        if i < len(measures) - 1:
            lines.append(",")

    lines.append(";")
    lines.append("")

    output_path = Path(output_path)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written {output_path} ({len(positions)} notes across {max_measure} measures)")


def generate_chart(
    audio_path: str | Path,
    onset_model_path: str | Path,
    pattern_model_path: str | Path,
    output_path: str | Path | None = None,
    title: str | None = None,
    artist: str = "Unknown",
    threshold: float = 0.5,
    temperature: float = 0.8,
    decode_strategy: str = "ergonomic",
    difficulty: str = "Challenge",
    rating: int = 10,
) -> Path:
    """Full pipeline: audio file → .sm file."""
    audio_path = Path(audio_path)
    if output_path is None:
        output_path = audio_path.with_suffix(".sm")
    if title is None:
        title = audio_path.stem

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    print(f"Extracting audio features from {audio_path}...")
    features = extract_features(audio_path)
    print(f"  {features.n_frames} frames, {features.duration:.1f}s")

    # Load models
    print("Loading models...")
    onset_model = OnsetDetector()
    onset_model.load_state_dict(torch.load(onset_model_path, weights_only=True, map_location=device))
    onset_model.to(device)

    pattern_model, pattern_mode = load_pattern_model(pattern_model_path, device)
    print(f"  Pattern mode: {pattern_mode}")

    # Detect BPM
    tempo = librosa.beat.tempo(
        onset_envelope=features.onset_envelope,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )
    bpm = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
    print(f"  Detected BPM: {bpm:.1f}")

    # Phase 1: Onset detection
    print("Detecting onsets...")
    onset_frames = detect_onsets(features, onset_model, device, threshold=threshold)
    onset_times = np.array([features.frame_to_time(f) for f in onset_frames])
    print(f"  Found {len(onset_frames)} onsets")

    # Phase 2: Pattern generation
    print("Generating patterns...")
    patterns = generate_patterns(
        features,
        onset_frames,
        pattern_model,
        device,
        temperature=temperature,
        decode_strategy=decode_strategy,
        pattern_mode=pattern_mode,
    )

    # Write .sm file
    offset = onset_times[0] if len(onset_times) > 0 else 0.0
    write_sm_file(
        output_path=output_path,
        audio_filename=audio_path.name,
        title=title,
        artist=artist,
        bpm=bpm,
        offset=offset,
        onset_times=onset_times,
        arrow_patterns=patterns,
        difficulty=difficulty,
        rating=rating,
    )

    return Path(output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate StepMania chart from audio")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--onset-model", default="checkpoints/onset_detector.pt")
    parser.add_argument("--pattern-model", default="checkpoints/pattern_generator.pt")
    parser.add_argument("--output", "-o", help="Output .sm path")
    parser.add_argument("--title", help="Song title")
    parser.add_argument("--artist", default="Unknown")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--decode-strategy",
        choices=["ergonomic", "raw"],
        default="ergonomic",
        help="Use the ergonomics-aware constrained decoder or the raw binary sampler.",
    )
    parser.add_argument("--difficulty", default="Challenge")
    parser.add_argument("--rating", type=int, default=10)
    args = parser.parse_args()

    generate_chart(
        audio_path=args.audio,
        onset_model_path=args.onset_model,
        pattern_model_path=args.pattern_model,
        output_path=args.output,
        title=args.title,
        artist=args.artist,
        threshold=args.threshold,
        temperature=args.temperature,
        decode_strategy=args.decode_strategy,
        difficulty=args.difficulty,
        rating=args.rating,
    )


if __name__ == "__main__":
    main()
