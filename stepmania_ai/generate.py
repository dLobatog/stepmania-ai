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


ARROW_CHARS = "0123"  # left, down, up, right


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
    model: PatternGenerator,
    device: torch.device,
    temperature: float = 0.8,
    context_window: int = 7,
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

    # Generate autoregressively
    patterns = model.generate(
        audio_windows,
        time_deltas,
        temperature=temperature,
        show_progress=True,
    )
    return patterns.cpu().numpy()


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
        adjusted = t + offset
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

    pattern_model = PatternGenerator()
    pattern_model.load_state_dict(torch.load(pattern_model_path, weights_only=True, map_location=device))
    pattern_model.to(device)

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
    patterns = generate_patterns(features, onset_frames, pattern_model, device, temperature=temperature)

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
        difficulty=args.difficulty,
        rating=args.rating,
    )


if __name__ == "__main__":
    main()
