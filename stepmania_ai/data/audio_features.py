"""Audio feature extraction for StepMania chart generation.

Extracts mel spectrograms, onset strength, tempogram, and chroma features
at a fixed time resolution aligned with potential note positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch


# We operate at ~100 frames/sec (10ms hop) to capture 16th notes at up to 240 BPM
SAMPLE_RATE = 22050
HOP_LENGTH = 220  # ~10ms at 22050 Hz
N_FFT = 2048
N_MELS = 80
FRAME_RATE = SAMPLE_RATE / HOP_LENGTH  # ~100.2 fps


@dataclass
class AudioFeatures:
    """Extracted audio features for a song, all aligned to the same time grid."""

    mel_spectrogram: np.ndarray  # (n_mels, n_frames)
    onset_envelope: np.ndarray  # (n_frames,)
    chroma: np.ndarray  # (12, n_frames)
    beat_frames: np.ndarray  # frame indices of detected beats
    duration: float  # seconds
    sr: int = SAMPLE_RATE
    hop_length: int = HOP_LENGTH

    @property
    def n_frames(self) -> int:
        return self.mel_spectrogram.shape[1]

    def frame_to_time(self, frame: int) -> float:
        return frame * self.hop_length / self.sr

    def time_to_frame(self, time: float) -> int:
        return int(round(time * self.sr / self.hop_length))

    def get_context_window(self, center_frame: int, window_size: int = 7) -> np.ndarray:
        """Get a feature window around a frame, combining all features.

        Returns shape (n_features, window_size) with zero-padding at boundaries.
        """
        half = window_size // 2
        # Stack all features: mel(80) + onset(1) + chroma(12) = 93 channels
        combined = np.vstack([
            self.mel_spectrogram,
            self.onset_envelope[np.newaxis, :],
            self.chroma,
        ])
        n_feat, n_frames = combined.shape

        # Extract window with padding
        start = center_frame - half
        end = center_frame + half + 1
        pad_left = max(0, -start)
        pad_right = max(0, end - n_frames)
        start = max(0, start)
        end = min(n_frames, end)

        window = np.zeros((n_feat, window_size), dtype=np.float32)
        window[:, pad_left:window_size - pad_right] = combined[:, start:end]
        return window

    def to_tensor_sequence(self, window_size: int = 7) -> torch.Tensor:
        """Convert entire song to a sequence of context windows.

        Returns shape (n_frames, n_features, window_size).
        """
        windows = []
        for i in range(self.n_frames):
            windows.append(self.get_context_window(i, window_size))
        return torch.tensor(np.stack(windows), dtype=torch.float32)


def extract_features(audio_path: str | Path) -> AudioFeatures:
    """Extract all audio features from an audio file."""
    audio_path = Path(audio_path)

    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    duration = len(y) / sr

    # Mel spectrogram (log-scaled)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalize to [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    # Onset strength envelope
    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=HOP_LENGTH
    )
    # Normalize
    onset_env = onset_env / (onset_env.max() + 1e-8)

    # Chroma features
    chroma = librosa.feature.chroma_cqt(
        y=y, sr=sr, hop_length=HOP_LENGTH
    )

    # Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH
    )

    # Ensure all features have same number of frames
    n_frames = min(mel_db.shape[1], len(onset_env), chroma.shape[1])
    mel_db = mel_db[:, :n_frames]
    onset_env = onset_env[:n_frames]
    chroma = chroma[:, :n_frames]

    return AudioFeatures(
        mel_spectrogram=mel_db.astype(np.float32),
        onset_envelope=onset_env.astype(np.float32),
        chroma=chroma.astype(np.float32),
        beat_frames=beat_frames,
        duration=duration,
    )


def main():
    """CLI: extract features and print summary."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: smai-extract <audio_file>")
        sys.exit(1)

    features = extract_features(sys.argv[1])
    print(f"Duration: {features.duration:.1f}s")
    print(f"Frames: {features.n_frames} ({FRAME_RATE:.1f} fps)")
    print(f"Mel shape: {features.mel_spectrogram.shape}")
    print(f"Onset envelope shape: {features.onset_envelope.shape}")
    print(f"Chroma shape: {features.chroma.shape}")
    print(f"Detected beats: {len(features.beat_frames)}")


if __name__ == "__main__":
    main()
