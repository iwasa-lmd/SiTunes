"""
Module A: ValenceStream  ── 環境 × ユーザー → val_seq (B, T, D)
Module B: ArousalStream  ── 楽曲 × 活動   → aro_seq (B, T, D)

2 ストリームは重みを一切共有しない独立したネットワーク。
各ストリームは GRU/Transformer の全系列出力 (B, T, D) を返す。
後段の CrossStreamAttention, TemporalAttention がこの系列を使用する。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dsd_casr.config import DSDConfig
from dsd_casr.model.encoder import SequenceEncoder


class ValenceStream(nn.Module):
    """
    Module A: 気分エンコーダ

    入力の焦点: *環境* と *ユーザー個人差*
      user_id    → 個人の感情ベースライン
      weather    → 外部環境（天候は気分と相関が強い）
      mobility   → 移動文脈（散歩中は気分が変わりやすい等）

    出力: val_seq (B, T, d_val)  ── 全ステップの隠れ状態系列
    """

    def __init__(self, cfg: DSDConfig):
        super().__init__()
        self.user_embed     = nn.Embedding(cfg.n_users,    cfg.d_user,     padding_idx=0)
        self.weather_embed  = nn.Embedding(cfg.n_weather,  cfg.d_weather,  padding_idx=0)
        self.mobility_embed = nn.Embedding(cfg.n_mobility, cfg.d_mobility, padding_idx=0)

        d_in = cfg.d_user + cfg.d_weather + cfg.d_mobility
        self.encoder = SequenceEncoder(
            d_in=d_in, d_hidden=cfg.d_val, n_layers=cfg.n_layers,
            dropout=cfg.dropout, encoder_type=cfg.encoder_type,
            n_heads=cfg.n_heads, d_ff_mult=cfg.d_ff_mult, seq_len=cfg.seq_len,
        )

    def forward(
        self,
        user_id:      torch.Tensor,  # (B,)
        weather_seq:  torch.Tensor,  # (B, T)
        mobility_seq: torch.Tensor,  # (B, T)
    ) -> torch.Tensor:               # → (B, T, d_val)

        T = weather_seq.shape[1]
        u_emb = self.user_embed(user_id).unsqueeze(1).expand(-1, T, -1)
        w_emb = self.weather_embed(weather_seq)
        m_emb = self.mobility_embed(mobility_seq)

        return self.encoder(torch.cat([u_emb, w_emb, m_emb], dim=-1))


class ArousalStream(nn.Module):
    """
    Module B: テンションエンコーダ

    入力の焦点: *音楽* と *身体活動*
      item_seq       → 聴いてきた曲の ID 列
      audio_cat_seq  → キー・ジャンル（カテゴリ音楽特徴）
      audio_cont_seq → 連続音響特徴量（energy / BPM 等）
      intensity_seq  → 活動強度（身体的覚醒と連動）

    出力: aro_seq (B, T, d_aro)  ── 全ステップの隠れ状態系列
    """

    def __init__(self, cfg: DSDConfig):
        super().__init__()
        self.item_embed  = nn.Embedding(cfg.n_items,  cfg.d_item,  padding_idx=0)
        self.key_embed   = nn.Embedding(cfg.n_keys,   cfg.d_key,   padding_idx=0)
        self.genre_embed = nn.Embedding(cfg.n_genres, cfg.d_genre, padding_idx=0)

        self.intensity_proj  = nn.Linear(1,                cfg.d_intensity)
        self.audio_cont_proj = nn.Linear(cfg.n_audio_cont, cfg.d_audio_cont)

        d_in = cfg.d_item + cfg.d_key + cfg.d_genre + cfg.d_intensity + cfg.d_audio_cont
        self.encoder = SequenceEncoder(
            d_in=d_in, d_hidden=cfg.d_aro, n_layers=cfg.n_layers,
            dropout=cfg.dropout, encoder_type=cfg.encoder_type,
            n_heads=cfg.n_heads, d_ff_mult=cfg.d_ff_mult, seq_len=cfg.seq_len,
        )

    def forward(
        self,
        item_seq:       torch.Tensor,  # (B, T)
        audio_cat_seq:  torch.Tensor,  # (B, T, 2)  [key_id, genre_id]
        audio_cont_seq: torch.Tensor,  # (B, T, n_audio_cont)
        intensity_seq:  torch.Tensor,  # (B, T, 1)
    ) -> torch.Tensor:                 # → (B, T, d_aro)

        x = torch.cat([
            self.item_embed(item_seq),
            self.key_embed(audio_cat_seq[..., 0]),
            self.genre_embed(audio_cat_seq[..., 1]),
            F.relu(self.audio_cont_proj(audio_cont_seq)),
            F.relu(self.intensity_proj(intensity_seq)),
        ], dim=-1)

        return self.encoder(x)
