"""Audio feature extraction for StepMania chart generation.

Extracts mel spectrograms, onset strength, and chroma features
at a fixed time resolution aligned with potential note positions.

Uses torchaudio for fast audio decoding and librosa for feature computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
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
    _combined_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _padded_cache: dict[int, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    @property
    def n_frames(self) -> int:
        return self.mel_spectrogram.shape[1]

    def frame_to_time(self, frame: int) -> float:
        return frame * self.hop_length / self.sr

    def time_to_frame(self, time: float) -> int:
        return int(round(time * self.sr / self.hop_length))

    @property
    def combined_features(self) -> np.ndarray:
        """Stack all feature channels once and reuse them across window lookups."""
        if not hasattr(self, "_combined_cache"):
            self._combined_cache = None
        if self._combined_cache is None:
            self._combined_cache = np.vstack([
                self.mel_spectrogram,
                self.onset_envelope[np.newaxis, :],
                self.chroma,
            ]).astype(np.float32, copy=False)
        return self._combined_cache

    def _get_padded_features(self, window_size: int) -> np.ndarray:
        if not hasattr(self, "_padded_cache"):
            self._padded_cache = {}
        padded = self._padded_cache.get(window_size)
        if padded is None:
            half = window_size // 2
            padded = np.pad(
                self.combined_features,
                ((0, 0), (half, half)),
                mode="constant",
            ).astype(np.float32, copy=False)
            self._padded_cache[window_size] = padded
        return padded

    def get_context_window(self, center_frame: int, window_size: int = 7) -> np.ndarray:
        """Get a feature window around a frame, combining all features.

        Returns shape (n_features, window_size) with zero-padding at boundaries.
        """
        padded = self._get_padded_features(window_size)
        start = int(center_frame)
        return padded[:, start:start + window_size]

    def get_context_windows(
        self,
        center_frames: np.ndarray | list[int],
        window_size: int = 7,
    ) -> np.ndarray:
        """Vectorized context-window lookup for many frames at once."""
        centers = np.asarray(center_frames, dtype=np.int64)
        if centers.size == 0:
            return np.zeros((0, self.combined_features.shape[0], window_size), dtype=np.float32)

        padded = self._get_padded_features(window_size)
        offsets = np.arange(window_size, dtype=np.int64)
        windows = padded[:, centers[:, None] + offsets]
        return np.transpose(windows, (1, 0, 2)).astype(np.float32, copy=False)

    def to_tensor_sequence(self, window_size: int = 7) -> torch.Tensor:
        """Convert entire song to a sequence of context windows.

        Returns shape (n_frames, n_features, window_size).
        """
        windows = self.get_context_windows(np.arange(self.n_frames), window_size)
        return torch.tensor(windows, dtype=torch.float32)


def _load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load audio using the fastest available method.

    Tries torchaudio first (fast C++ decoder), falls back to soundfile, then librosa.
    """
    try:
        import torchaudio
        waveform, sr = torchaudio.load(str(audio_path))
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        y = waveform.squeeze(0).numpy()
        # Resample if needed
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        return y, SAMPLE_RATE
    except Exception:
        pass

    try:
        y, sr = sf.read(str(audio_path), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        return y, SAMPLE_RATE
    except Exception:
        pass

    # Final fallback
    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    return y, sr


def extract_features(audio_path: str | Path, skip_beats: bool = False) -> AudioFeatures:
    """Extract all audio features from an audio file.

    Args:
        audio_path: Path to audio file (ogg, mp3, wav, etc.)
        skip_beats: If True, skip beat tracking (faster, for training only)
    """
    audio_path = Path(audio_path)

    y, sr = _load_audio(audio_path)
    duration = len(y) / sr

    # Mel spectrogram (log-scaled)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalize to [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    # Onset strength envelope — compute from mel spectrogram (avoids recomputing STFT)
    onset_env = librosa.onset.onset_strength(
        S=librosa.power_to_db(mel, ref=np.max), sr=sr, hop_length=HOP_LENGTH
    )
    # Normalize
    onset_env = onset_env / (onset_env.max() + 1e-8)

    # Chroma — compute from an explicit STFT + chroma filterbank instead of
    # librosa.feature.chroma_stft(). On Linux we observed native segfaults for
    # some songs inside that helper, while the lower-level steps remain stable.
    stft_power = np.abs(
        librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    ) ** 2
    chroma_filter = librosa.filters.chroma(sr=sr, n_fft=N_FFT)
    chroma = chroma_filter @ stft_power
    chroma = chroma / np.maximum(chroma.sum(axis=0, keepdims=True), 1e-8)

    # Beat tracking (skip during training for speed — only needed at generation time)
    if skip_beats:
        beat_frames = np.array([], dtype=np.int64)
    else:
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH
        )

    # Ensure all features have same number of frames
    n_frames = min(mel_db.shape[1], len(onset_env), chroma.shape[1])
    mel_db = mel_db[:, :n_frames]
    onset_env = onset_env[:n_frames]
    chroma = chroma[:, :n_frames]

    # Free raw audio immediately
    del y

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
