from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass
class DSDConfig:
    # ── 語彙サイズ ─────────────────────────────────────────────────────────────
    n_users:    int = 31    # user_id の最大値 + 1（0 は padding 用）
    n_items:    int = 1000  # item_id の最大値 + 1
    n_weather:  int = 4     # 0=pad, 1=sunny, 2=cloudy, 3=rainy
    n_mobility: int = 7     # 0=pad, 1=still, 2=act2still, 3=walking,
                            #        4=none,  5=sleeping, 6=running
    n_keys:     int = 13    # 0=pad, 1-12 = 音楽キー (C〜B)
    n_genres:   int = 20    # 0=pad, 1〜 = general_genre_id

    # ── 埋め込み次元 ──────────────────────────────────────────────────────────
    d_user:       int = 32
    d_weather:    int = 16
    d_mobility:   int = 16
    d_item:       int = 64
    d_key:        int = 8
    d_genre:      int = 16
    d_intensity:  int = 16  # 活動強度（連続値）の射影次元
    d_audio_cont: int = 32  # 連続音響特徴量の射影次元

    # ── エンコーダ隠れ次元 ────────────────────────────────────────────────────
    d_val:   int = 128  # Valence Stream の隠れ次元
    d_aro:   int = 128  # Arousal Stream の隠れ次元
    n_layers: int = 2
    dropout:  float = 0.2

    # ── シーケンス長 ──────────────────────────────────────────────────────────
    seq_len: int = 10

    # ── 連続音響特徴量の次元数（SiTunes の 16 特徴量に対応） ─────────────────
    # loudness, danceability, energy, speechiness, acousticness,
    # instrumentalness, audio_valence, tempo,
    # F0final_sma_amean, F0final_sma_stddev,
    # audspec_lengthL1norm_sma_stddev, pcm_RMSenergy_sma_stddev,
    # pcm_fftMag_psySharpness_sma_{amean,stddev}, pcm_zcr_sma_{amean,stddev}
    n_audio_cont: int = 16

    # ── 候補アイテム射影次元（d_val + d_aro を推奨） ─────────────────────────
    d_candidate: int = 128

    # ── マルチタスク損失の重み ────────────────────────────────────────────────
    alpha: float = 0.1  # Valence MSE 損失の重み
    beta:  float = 0.1  # Arousal MSE 損失の重み

    # ── エンコーダ種別 ────────────────────────────────────────────────────────
    encoder_type: Literal["gru", "transformer"] = "gru"

    # ── Transformer 専用パラメータ ────────────────────────────────────────────
    n_heads:   int = 4
    d_ff_mult: int = 4  # FFN 隠れ次元 = d_hidden * d_ff_mult

    # ── 改善モジュールのフラグ ─────────────────────────────────────────────────
    # [改善1] STAMP-style: 短期（直近アイテム）と長期（セッション平均）を分離した attention
    use_temporal_attn: bool = True
    # [改善2] Cross-Stream Attention: ValenceとArousalが相互に参照する（感情空間の相関を学習）
    #         ※ d_val == d_aro が必要
    use_cross_stream_attn: bool = True
    # [改善3] Gated Fusion: concat の代わりに highway-network 式のゲートで融合
    use_gated_fusion: bool = True
    # [改善4] 損失関数: "ce"=全アイテム CE, "bpr"=BPR（ランキング最適化, AUC最大化に等価）
    loss_type: Literal["ce", "bpr"] = "bpr"
