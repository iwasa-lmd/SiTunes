"""
KGMMoEConfig: ハイパーパラメータ設定

設計方針:
  Valence Tower : user_id + weather_id + mobility_id (ユーザー・環境側)
  Arousal Tower : item_id + audio (アイテム側)
  MMoE Expert   : 両タワーを統合し、cluster_id でゲートを条件付け
  Arousal Head  : 残差学習 (baseline_shift(pre_arousal) + deep_pred)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class KGMMoEConfig:
    # ── Vocabulary sizes ──────────────────────────────────────────────────────
    n_users:    int = 32    # user_id の最大値 + 1 (実データで上書き)
    n_items:    int = 1001  # item_id の最大値 + 1 (実データで上書き)
    n_weather:  int = 3     # 0=Sunny, 1=Cloudy, 2=Rainy
    n_mobility: int = 6     # 0=Still,1=Act2Still,2=Walking,3=Missing,4=Lying,5=Running
    n_keys:     int = 13    # 0-11 = C-B, +1 for safety (実データで上書き)
    n_genres:   int = 14    # 0-indexed (実データで上書き)
    n_clusters: int = 4     # 感情ボラティリティに基づくユーザークラスタ数

    # ── Embedding dims ────────────────────────────────────────────────────────
    d_user:     int = 64    # user-heavy: Valence はユーザー依存が強い
    d_item:     int = 128   # item-heavy: Arousal は楽曲依存が強い
    d_weather:  int = 8
    d_mobility: int = 16
    d_key:      int = 8
    d_genre:    int = 16
    d_cluster:  int = 32    # ゲートを条件付けるクラスタ埋め込み

    # ── Audio continuous features ─────────────────────────────────────────────
    n_audio_cont: int = 16  # 16次元音響特徴量
    d_audio_cont: int = 64  # projection 後の次元

    # ── Tower hidden dims ─────────────────────────────────────────────────────
    d_val_tower: int = 128  # Valence Tower 出力次元
    d_aro_tower: int = 128  # Arousal Tower 出力次元

    # ── MMoE ──────────────────────────────────────────────────────────────────
    n_experts:  int = 4     # Expert 数
    d_expert:   int = 256   # Expert 出力次元 (= d_val_tower + d_aro_tower 推奨)
    n_tasks:    int = 3     # rec / val / aro

    # ── Head dims ─────────────────────────────────────────────────────────────
    d_candidate: int = 128  # 推薦: user repr → candidate inner product 次元
    d_val_head:  int = 64   # Valence Head MLP hidden dim
    d_aro_head:  int = 64   # Arousal Head MLP hidden dim

    # ── Multi-task loss weights ───────────────────────────────────────────────
    alpha: float = 0.4  # Valence MSE 重み
    beta:  float = 0.4  # Arousal MSE 重み

    # ── BPR sampling ──────────────────────────────────────────────────────────
    n_neg_samples: int = 10  # BPR 用ネガティブサンプル数

    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir:    str       = "SiTunes_dataset/SiTunes"
    train_stage: int       = 2          # 学習に使う Stage
    test_stages: list[int] = field(default_factory=lambda: [2, 3])
    train_ratio: float     = 0.70
    val_ratio:   float     = 0.15

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size:   int   = 64
    n_epochs:     int   = 50
    lr:           float = 1e-3
    weight_decay: float = 1e-4
    lr_step:      int   = 15
    lr_gamma:     float = 0.5
    seed:         int   = 42
    top_k:        int   = 10
