"""Onset detection model: predicts whether each audio frame should have a note.

Architecture: CNN over the mel/chroma/onset feature window, followed by a
bidirectional GRU to capture temporal context, then a sigmoid classifier.

This is the "when to place notes" model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OnsetDetector(nn.Module):
    """Predicts note onsets from audio features.

    Input: (batch, n_features, context_window) — a context window per frame
    Output: (batch, 1) — onset probability

    For sequence mode (full song):
    Input: (batch, seq_len, n_features, context_window)
    Output: (batch, seq_len, 1)
    """

    def __init__(
        self,
        n_features: int = 93,  # 80 mel + 1 onset + 12 chroma
        context_window: int = 7,
        hidden_dim: int = 128,
        n_gru_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=n_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_gru_layers > 1 else 0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """Process a single frame's context window.

        x: (batch, n_features, context_window)
        returns: (batch, 128) — CNN embedding
        """
        return self.cnn(x).squeeze(-1)  # (batch, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process a sequence of frames.

        x: (batch, seq_len, n_features, context_window)
        returns: (batch, seq_len, 1) — onset probabilities
        """
        batch, seq_len, n_feat, ctx = x.shape

        # CNN over each frame
        x_flat = x.reshape(batch * seq_len, n_feat, ctx)
        cnn_out = self.cnn(x_flat).squeeze(-1)  # (batch*seq_len, 128)
        cnn_out = cnn_out.reshape(batch, seq_len, -1)  # (batch, seq_len, 128)

        # GRU over sequence
        gru_out, _ = self.gru(cnn_out)  # (batch, seq_len, hidden*2)

        # Classify each frame
        logits = self.classifier(gru_out)  # (batch, seq_len, 1)
        return logits

    def forward_framewise(self, x: torch.Tensor) -> torch.Tensor:
        """Process individual frames without GRU context (for single-frame inference).

        x: (batch, n_features, context_window)
        returns: (batch, 1)
        """
        cnn_out = self.forward_single(x)  # (batch, 128)
        # Fake sequence dim for GRU
        cnn_out = cnn_out.unsqueeze(1)  # (batch, 1, 128)
        gru_out, _ = self.gru(cnn_out)
        logits = self.classifier(gru_out.squeeze(1))  # (batch, 1)
        return logits
