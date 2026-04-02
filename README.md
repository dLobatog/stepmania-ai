# StepMania AI

ML-powered StepMania chart generator. Takes an audio file and generates a `.sm` simfile with Expert/Challenge-level step charts.

## Architecture

Two-stage pipeline trained on human-authored simfiles:

1. **Onset Detector** (CNN + BiGRU) — classifies each ~10ms audio frame as "place a note here" or not, learning the relationship between audio features and note placement
2. **Pattern Generator** (Transformer decoder) — autoregressively generates arrow patterns (left/down/up/right) conditioned on audio features, previous arrows, and rhythm timing

### Audio Features

Extracted at ~100fps using librosa:
- Mel spectrogram (80 bands)
- Onset strength envelope
- Chroma (12-tone pitch class)
- Tempogram

### Training

- Balanced sampling to handle sparse note placement (~5% of frames have notes)
- Two-phase: onset detector trains first, then pattern generator trains on onset-only frames
- Supports quality labels for training on curated "good" vs "bad" chart collections

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
