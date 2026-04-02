"""Pattern generation model: predicts which arrows to place at each onset.

Architecture: Transformer decoder that generates arrow patterns conditioned on
audio features and previous arrow history. This captures the physical flow
constraints of pad play.

This is the "which arrows to place" model.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from tqdm import tqdm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PatternGenerator(nn.Module):
    """Generates arrow patterns conditioned on audio features.

    Input:
        audio_features: (batch, seq_len, n_audio_features) — from onset detector CNN
        prev_arrows: (batch, seq_len, 4) — previous arrow patterns (teacher forcing)
        time_deltas: (batch, seq_len) — time since previous note (encodes rhythm)

    Output:
        arrow_logits: (batch, seq_len, 4) — independent logit per arrow column
    """

    def __init__(
        self,
        n_audio_features: int = 93,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.2,
        context_window: int = 7,
    ):
        super().__init__()

        # Audio feature encoder (same CNN arch as onset detector)
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

        # Arrow history embedding
        self.arrow_embed = nn.Linear(4, d_model // 4)

        # Time delta embedding (log-scaled)
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(),
        )

        # Combine audio + arrows + time
        self.combine = nn.Linear(d_model + d_model // 4 + d_model // 4, d_model)

        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Transformer decoder (causal — can only look at previous notes)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Per-arrow output heads
        self.arrow_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )
            for _ in range(4)
        ])

    def forward(
        self,
        audio_windows: torch.Tensor,
        prev_arrows: torch.Tensor,
        time_deltas: torch.Tensor,
    ) -> torch.Tensor:
        """
        audio_windows: (batch, seq_len, n_features, context_window)
        prev_arrows: (batch, seq_len, 4) — shifted right (previous step's arrows)
        time_deltas: (batch, seq_len) — seconds since previous note

        returns: (batch, seq_len, 4) — logits per arrow
        """
        batch, seq_len, n_feat, ctx = audio_windows.shape

        # Encode audio
        audio_flat = audio_windows.reshape(batch * seq_len, n_feat, ctx)
        audio_enc = self.audio_cnn(audio_flat).squeeze(-1)  # (B*S, d_model)
        audio_enc = audio_enc.reshape(batch, seq_len, -1)  # (B, S, d_model)

        # Encode previous arrows
        arrow_enc = self.arrow_embed(prev_arrows)  # (B, S, d_model//4)

        # Encode time deltas (log scale for better distribution)
        td = torch.log1p(time_deltas).unsqueeze(-1)  # (B, S, 1)
        time_enc = self.time_embed(td)  # (B, S, d_model//4)

        # Combine
        combined = torch.cat([audio_enc, arrow_enc, time_enc], dim=-1)
        combined = self.combine(combined)  # (B, S, d_model)
        combined = self.pos_encoding(combined)

        # Causal mask — each position can only attend to itself and earlier positions
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=combined.device)

        # Use audio as memory, combined as target
        decoded = self.transformer(
            tgt=combined,
            memory=audio_enc,
            tgt_mask=causal_mask,
        )  # (B, S, d_model)

        # Per-arrow predictions
        arrow_logits = torch.cat(
            [head(decoded) for head in self.arrow_heads], dim=-1
        )  # (B, S, 4)

        return arrow_logits

    @torch.no_grad()
    def generate(
        self,
        audio_windows: torch.Tensor,
        time_deltas: torch.Tensor,
        temperature: float = 1.0,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """Autoregressive generation — no teacher forcing.

        audio_windows: (1, seq_len, n_features, context_window)
        time_deltas: (1, seq_len)

        returns: (seq_len, 4) — generated arrow pattern
        """
        self.eval()
        batch, seq_len, n_feat, ctx = audio_windows.shape

        prev_arrows = torch.zeros(1, seq_len, 4, device=audio_windows.device)
        generated = torch.zeros(seq_len, 4, device=audio_windows.device)

        steps = range(seq_len)
        if show_progress:
            steps = tqdm(steps, total=seq_len, desc="Generating patterns", leave=False)

        for t in steps:
            # Only process up to current timestep
            logits = self.forward(
                audio_windows[:, :t + 1],
                prev_arrows[:, :t + 1],
                time_deltas[:, :t + 1],
            )
            # Get prediction for current step
            step_logits = logits[0, t] / temperature  # (4,)
            probs = torch.sigmoid(step_logits)
            arrows = (torch.rand(4, device=probs.device) < probs).float()

            # Ensure at least one arrow is active
            if arrows.sum() == 0:
                arrows[torch.argmax(probs)] = 1.0

            generated[t] = arrows

            # Feed back for next step
            if t + 1 < seq_len:
                prev_arrows[0, t + 1] = arrows

        return generated
