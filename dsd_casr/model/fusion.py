"""
改善モジュール集

[改善1] TemporalAttention
  STAMP (Liu et al., KDD 2018) に基づく短期/長期分離 attention。
  短期: 直近アイテムの感情状態  →  即時的な気分・テンション
  長期: セッション全体の加重平均 →  ユーザーの持続的な傾向
  最終: 両者を MLP で統合し、GRU の「最後だけ使う」欠陥を補完。

[改善2] CrossStreamAttention
  Valence と Arousal ストリームを相互参照させる cross-attention 層。
  根拠: Russell's Circumplex Model —— 感情空間の V-A 次元は相関しており
  独立に処理すると次元間の依存性を学習できない。
  「高 Valence 状態では低 Arousal の曲を選ぶ」などのパターンを捕捉。

[改善3] GatedFusion
  Highway network 式のゲートで 2 ストリームを融合。
  単純な concat は両ストリームを常に等重みで扱うが、
  ゲートは「運動中は Arousal 優先」「リラックス時は Valence 優先」のような
  文脈依存の重み付けをモデルが自律的に学習できる。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    STAMP-style 短期/長期分離 Temporal Attention。

    入力 : seq (B, T, D) ── GRU/Transformer の全ステップ出力
    出力 : (B, D)         ── 短期・長期を統合したユーザー表現

    注意重み α_i = softmax(proj(tanh(W_x·x_i + W_ms·m_s + W_mt·m_t + b)))
      m_s : セッション平均（長期的嗜好）
      m_t : 最終ステップ（短期的注目）
    """

    def __init__(self, d: int):
        super().__init__()
        # attention scoring の各射影
        self.w_x  = nn.Linear(d, d, bias=False)   # 各ステップへの射影
        self.w_ms = nn.Linear(d, d, bias=False)   # 長期記憶へのバイアス
        self.w_mt = nn.Linear(d, d, bias=False)   # 短期注目へのバイアス
        self.b_a  = nn.Parameter(torch.zeros(d))
        self.proj = nn.Linear(d, 1, bias=False)   # スカラー attention スコア

        # 加重sum（長期）と最終ステップ（短期）を統合する MLP
        self.mlp = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (B, T, D)
        m_s = seq.mean(dim=1, keepdim=True)  # (B, 1, D) 長期セッション平均
        m_t = seq[:, -1:, :]                 # (B, 1, D) 直近ステップ（短期）

        # α_i を計算: 長期・短期の両方からバイアスを受ける attention
        score = torch.tanh(
            self.w_x(seq) + self.w_ms(m_s) + self.w_mt(m_t) + self.b_a
        )  # (B, T, D)
        alpha = F.softmax(self.proj(score), dim=1)     # (B, T, 1)
        weighted = (alpha * seq).sum(dim=1)             # (B, D) 加重sum

        # 長期（加重sum）と短期（最終ステップ）を MLP で融合
        return self.mlp(torch.cat([weighted, m_t.squeeze(1)], dim=-1))


class CrossStreamAttention(nn.Module):
    """
    ValenceStream と ArousalStream の相互参照 Cross-Attention。

    方向: query_seq ← context_seq として attention を計算し、
          context の情報で query を強化する（残差接続付き）。
    用法: モデル内で V→A, A→V の両方向を適用する。

    入力 : query_seq   (B, T, D)  ── 強化したいストリームの系列
           context_seq (B, T, D)  ── 参照するストリームの系列
    出力 : (B, T, D)               ── context で強化された query 系列
    """

    def __init__(self, d: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=n_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self, query_seq: torch.Tensor, context_seq: torch.Tensor
    ) -> torch.Tensor:
        # cross-attention: query がこちら、K/V が相手ストリーム
        attended, _ = self.attn(query_seq, context_seq, context_seq)
        # 残差接続 + LayerNorm（Transformer 標準）
        return self.norm(query_seq + self.dropout(attended))


class GatedFusion(nn.Module):
    """
    Highway-Network 式 Gated Fusion。

    単純な concat との違い:
      concat では両ストリームが常に等価に扱われるが、
      ゲートはストリームの相対的重要度を文脈から動的に学習する。

      例: 活動強度が高い → Arousal ストリームを優先
          週末の晴れた日  → Valence ストリームを優先

    出力次元は入力と同じ d_val + d_aro を維持するため、
    下流 (user_proj) の変更が不要。

    入力 : h_val (B, d_val), h_aro (B, d_aro)
    出力 : (B, d_val + d_aro)
    """

    def __init__(self, d_val: int, d_aro: int):
        super().__init__()
        d = d_val + d_aro
        # ゲート: どの次元をどれだけ変換するかを決める
        self.gate      = nn.Linear(d, d)
        # 変換: 2 ストリームの非線形結合
        self.transform = nn.Linear(d, d)

    def forward(self, h_val: torch.Tensor, h_aro: torch.Tensor) -> torch.Tensor:
        h = torch.cat([h_val, h_aro], dim=-1)       # (B, d_val+d_aro)
        g = torch.sigmoid(self.gate(h))             # ゲート値 ∈ (0, 1)
        t = torch.tanh(self.transform(h))           # 非線形変換
        # highway: g * 変換値 + (1-g) * 元の値
        return g * t + (1.0 - g) * h
