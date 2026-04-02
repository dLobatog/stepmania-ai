"""Predict simple hold starts and durations on generated note sequences."""

from __future__ import annotations

import torch
import torch.nn as nn

from stepmania_ai.models.hold_utils import HOLD_DURATION_BUCKETS_BEATS


class HoldNotePredictor(nn.Module):
    """Predicts whether an onset should become a hold and for how long."""

    def __init__(
        self,
        n_audio_features: int = 93,
        d_model: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        num_duration_buckets: int = len(HOLD_DURATION_BUCKETS_BEATS),
    ):
        super().__init__()
        self.num_duration_buckets = num_duration_buckets

        self.audio_cnn = nn.Sequential(
            nn.Conv1d(n_audio_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.current_arrow_embed = nn.Linear(4, d_model // 4)
        self.prev_arrow_embed = nn.Linear(4, d_model // 4)
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(),
        )
        self.combine = nn.Linear(d_model + d_model // 4 + d_model // 4 + d_model // 4, hidden_dim)
        self.context = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.hold_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_duration_buckets),
        )

    def forward(
        self,
        audio_windows: torch.Tensor,
        current_arrows: torch.Tensor,
        prev_arrows: torch.Tensor,
        time_deltas: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, n_feat, ctx = audio_windows.shape
        audio_flat = audio_windows.reshape(batch * seq_len, n_feat, ctx)
        audio_enc = self.audio_cnn(audio_flat).squeeze(-1).reshape(batch, seq_len, -1)
        current_enc = self.current_arrow_embed(current_arrows)
        prev_enc = self.prev_arrow_embed(prev_arrows)
        time_enc = self.time_embed(torch.log1p(time_deltas).unsqueeze(-1))
        combined = torch.cat([audio_enc, current_enc, prev_enc, time_enc], dim=-1)
        combined = self.combine(combined)
        context_out, _ = self.context(combined)
        hold_logits = self.hold_head(context_out).squeeze(-1)
        duration_logits = self.duration_head(context_out)
        return hold_logits, duration_logits
