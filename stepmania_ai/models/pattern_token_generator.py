"""Tokenized ergonomic pattern generation model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from tqdm import tqdm

from stepmania_ai.models.pattern_vocab import START_TOKEN, VOCAB_SIZE


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PatternTokenGenerator(nn.Module):
    """Generates ergonomic step tokens instead of independent arrow bits."""

    def __init__(
        self,
        n_audio_features: int = 93,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.2,
        vocab_size: int = VOCAB_SIZE,
    ):
        super().__init__()
        self.vocab_size = vocab_size

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

        self.token_embed = nn.Embedding(vocab_size + 1, d_model // 4)
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(),
        )
        self.combine = nn.Linear(d_model + d_model // 4 + d_model // 4, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_size),
        )

    def forward(
        self,
        audio_windows: torch.Tensor,
        prev_tokens: torch.Tensor,
        time_deltas: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, n_feat, ctx = audio_windows.shape

        audio_flat = audio_windows.reshape(batch * seq_len, n_feat, ctx)
        audio_enc = self.audio_cnn(audio_flat).squeeze(-1).reshape(batch, seq_len, -1)

        token_enc = self.token_embed(prev_tokens)
        time_enc = self.time_embed(torch.log1p(time_deltas).unsqueeze(-1))

        combined = torch.cat([audio_enc, token_enc, time_enc], dim=-1)
        combined = self.combine(combined)
        combined = self.pos_encoding(combined)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=combined.device)
        decoded = self.transformer(
            tgt=combined,
            memory=audio_enc,
            tgt_mask=causal_mask,
        )
        return self.output(decoded)

    @torch.no_grad()
    def generate(
        self,
        audio_windows: torch.Tensor,
        time_deltas: torch.Tensor,
        temperature: float = 1.0,
        show_progress: bool = False,
        max_history_steps: int = 64,
    ) -> torch.Tensor:
        self.eval()
        _, seq_len, _, _ = audio_windows.shape
        prev_tokens = torch.full(
            (1, seq_len),
            START_TOKEN,
            dtype=torch.long,
            device=audio_windows.device,
        )
        generated = torch.zeros(seq_len, dtype=torch.long, device=audio_windows.device)

        steps = range(seq_len)
        if show_progress:
            steps = tqdm(steps, total=seq_len, desc="Generating patterns", leave=False)

        for t in steps:
            start = max(0, t + 1 - max_history_steps)
            logits = self.forward(
                audio_windows[:, start:t + 1],
                prev_tokens[:, start:t + 1],
                time_deltas[:, start:t + 1],
            )
            step_logits = logits[0, -1] / temperature
            token = int(torch.argmax(step_logits).item())
            generated[t] = token
            if t + 1 < seq_len:
                prev_tokens[0, t + 1] = token

        return generated
