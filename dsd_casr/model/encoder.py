"""GRU / Transformer Encoder の共通ラッパー。"""
from __future__ import annotations
import math
from typing import Literal

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """固定サイン/コサイン位置エンコーディング（Transformer 用）。"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class SequenceEncoder(nn.Module):
    """
    GRU または TransformerEncoder をラップした汎用系列エンコーダ。

    入力  : (B, T, d_in)
    出力  : (B, T, d_hidden)  ── 全ステップの隠れ状態系列
            後段モジュール（TemporalAttention, CrossStreamAttention）が
            系列全体を参照できるよう、最終ステップではなく全系列を返す。
    """

    def __init__(
        self,
        d_in:         int,
        d_hidden:     int,
        n_layers:     int,
        dropout:      float,
        encoder_type: Literal["gru", "transformer"],
        n_heads:      int = 4,
        d_ff_mult:    int = 4,
        seq_len:      int = 10,
    ):
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == "gru":
            self.encoder = nn.GRU(
                input_size  = d_in,
                hidden_size = d_hidden,
                num_layers  = n_layers,
                batch_first = True,
                dropout     = dropout if n_layers > 1 else 0.0,
            )
        else:
            self.input_proj = nn.Linear(d_in, d_hidden)
            self.pos_enc    = PositionalEncoding(d_hidden, max_len=seq_len + 1, dropout=dropout)
            self.encoder    = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_hidden, nhead=n_heads,
                    dim_feedforward=d_hidden * d_ff_mult,
                    dropout=dropout, batch_first=True,
                ),
                num_layers=n_layers,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns full sequence output (B, T, d_hidden)."""
        if self.encoder_type == "gru":
            seq_out, _ = self.encoder(x)   # (B, T, d_hidden)
            return seq_out
        else:
            x = self.pos_enc(self.input_proj(x))
            return self.encoder(x)         # (B, T, d_hidden)
