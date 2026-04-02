# StepMania AI

ML-powered StepMania chart generator. Takes an audio file and generates a `.sm` simfile with Expert/Challenge-level step charts.

## Architecture

Two-stage pipeline trained on human-authored simfiles:

1. **Onset Detector** (CNN + BiGRU, ~554K params) — classifies each ~10ms audio frame as "place a note here" or not, learning the relationship between audio features and note placement
2. **Pattern Generator** (Transformer decoder, ~1.16M params) — autoregressively generates arrow patterns (left/down/up/right) conditioned on audio features, previous arrows, and rhythm timing

### Audio Features

Extracted at ~100fps (~10ms per frame) using librosa. All features are aligned to the same time grid:

| Feature | Shape per song | Size | What it captures |
|---------|---------------|------|-----------------|
| **Mel spectrogram** | (80, n_frames) | ~6.8 MB | Frequency content across 80 mel-scaled bands. Captures the timbral texture of the audio — bass hits, hi-hats, vocals, synths |
| **Onset strength envelope** | (n_frames,) | ~87 KB | How likely a musical "event" (hit, note, transient) is at each frame. Derived from spectral flux. Primary signal for note placement |
| **Chroma** | (12, n_frames) | ~1 MB | Pitch class energy (C, C#, D, ... B). Captures harmonic/melodic content — useful for placing notes on melodic phrases vs percussion |

These three features are stacked into a 93-channel feature vector (80 + 1 + 12) per frame. The model sees a context window of 7 frames (~70ms) centered on each position.

### Training

- **Balanced sampling**: only ~5% of frames have notes, so onset frames are oversampled to 50/50 balance
- **Two-phase**: onset detector trains first on all frames, then pattern generator trains only on onset frames
- **Parallel data loading**: audio feature extraction runs across all CPU cores (8 workers by default)
- **TensorBoard**: training logs loss, precision/recall/F1, per-arrow accuracy, and learning rate to `runs/`

## Memory & Performance

### Data loading

Audio feature extraction happens once per song and is cached to `.cache/features/`. Each cached song is ~8 MB in memory (mel + onset + chroma + labels). Extraction uses multiprocessing (8 workers by default) and takes ~3-5 seconds per uncached song.

| Songs | In-memory | Cache on disk | First load | Subsequent loads |
|-------|-----------|---------------|------------|-----------------|
| 24 | ~200 MB | ~200 MB | ~2 min | seconds |
| 200 | ~1.6 GB | ~1.6 GB | ~12 min | seconds |
| 1000+ | ~8 GB | ~8 GB | ~60 min | seconds |

### Training

Training uses MPS (Apple Silicon), CUDA, or CPU. GPU memory usage is modest (~1-2 GB) since batches are small feature windows, not full spectrograms.

**Recommended minimum**: 16 GB RAM for up to ~1500 songs.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.11+. Uses PyTorch with MPS (Apple Silicon), CUDA, or CPU.

## Usage

### Train on simfile packs

```bash
smai-train path/to/pack1 path/to/pack2 \
    --epochs-onset 50 \
    --epochs-pattern 80 \
    --batch-size 256
```

### Monitor training

```bash
tensorboard --logdir runs
# Open http://localhost:6006
```

Tracks: onset loss/precision/recall/F1, pattern loss/accuracy (overall + per-arrow), learning rates.

### Generate a chart from audio

```bash
smai-generate song.ogg \
    --onset-model checkpoints/onset_detector.pt \
    --pattern-model checkpoints/pattern_generator.pt \
    --difficulty Challenge \
    --rating 10 \
    --threshold 0.5 \
    --temperature 0.8
```

### Utilities

```bash
# Parse and inspect a .sm file
smai-parse path/to/song.sm

# Extract and summarize audio features
smai-extract path/to/song.ogg
```

## Tuning

- **`--threshold`** (onset detector): higher = fewer notes, lower = more notes. 0.3-0.7 is the useful range.
- **`--temperature`** (pattern generator): lower = more predictable/repetitive patterns, higher = more variety. 0.5-1.0 is typical.
- **More training data** improves quality significantly. Add multiple packs with `smai-train pack1 pack2 pack3`.

## .sm Format

The generator outputs standard StepMania `.sm` files compatible with StepMania 5, OpenITG, and other SM-compatible engines. Place the generated `.sm` file alongside the audio file in a song folder within your StepMania Songs directory.
