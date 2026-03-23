"""
Module C: Fusion & Multi-Task Heads + DSD_CASR メインモデル

改善後のデータフロー:

  ValenceStream → val_seq (B,T,D) ─┐
                                   ├─[改善2] CrossStreamAttention ─┐
  ArousalStream → aro_seq (B,T,D) ─┘                               │
                                                                    ↓
                                   val_seq' (B,T,D), aro_seq' (B,T,D)
                                        ↓ [改善1] TemporalAttention
                                   h_val (B,D),  h_aro (B,D)
                                        ↓ [改善3] GatedFusion
                                   h_user (B, d_val+d_aro)
                                        ↓
         Head 1: h_user → q → q·cand^T → rec_logits (B, n_items)
         Head 2: h_val  → Linear(1) → pred_ΔV
         Head 3: h_aro  → Linear(1) → pred_ΔA
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dsd_casr.config import DSDConfig
from dsd_casr.model.streams import ValenceStream, ArousalStream
from dsd_casr.model.fusion import TemporalAttention, CrossStreamAttention, GatedFusion


class CandidateProjector(nn.Module):
    """
    候補アイテムの統合ベクトルを生成する MLP。
    (item_embed + key_embed + genre_embed + audio_cont) → d_candidate
    """

    def __init__(self, cfg: DSDConfig):
        super().__init__()
        self.item_embed      = nn.Embedding(cfg.n_items,  cfg.d_item,  padding_idx=0)
        self.key_embed       = nn.Embedding(cfg.n_keys,   cfg.d_key,   padding_idx=0)
        self.genre_embed     = nn.Embedding(cfg.n_genres, cfg.d_genre, padding_idx=0)
        self.audio_cont_proj = nn.Linear(cfg.n_audio_cont, cfg.d_audio_cont)

        d_in = cfg.d_item + cfg.d_key + cfg.d_genre + cfg.d_audio_cont
        self.mlp = nn.Sequential(
            nn.Linear(d_in, cfg.d_candidate),
            nn.ReLU(),
            nn.Linear(cfg.d_candidate, cfg.d_candidate),
        )

    def forward(
        self,
        item_ids:   torch.Tensor,  # (N,)
        key_ids:    torch.Tensor,  # (N,)
        genre_ids:  torch.Tensor,  # (N,)
        audio_cont: torch.Tensor,  # (N, n_audio_cont)
    ) -> torch.Tensor:             # → (N, d_candidate)
        x = torch.cat([
            self.item_embed(item_ids),
            self.key_embed(key_ids),
            self.genre_embed(genre_ids),
            F.relu(self.audio_cont_proj(audio_cont)),
        ], dim=-1)
        return self.mlp(x)


class DSD_CASR(nn.Module):
    """
    Dual-Stream Decoupled Context-Aware Sequential Recommender（改善版）。

    設定フラグ (DSDConfig):
      use_temporal_attn    : STAMP-style 短期/長期 attention
      use_cross_stream_attn: V-A 相互参照 cross-attention
      use_gated_fusion     : highway-network 式ゲート融合

    forward() の返り値:
      rec_logits     : (B, n_items)  全アイテムに対するスコア
      pred_delta_val : (B,)          予測 ΔValence
      pred_delta_aro : (B,)          予測 ΔArousal
    """

    def __init__(self, cfg: DSDConfig):
        super().__init__()
        self.cfg = cfg

        # Module A / B  ── 各ストリームは (B, T, D) を返す
        self.valence_stream = ValenceStream(cfg)
        self.arousal_stream = ArousalStream(cfg)

        # [改善2] Cross-Stream Attention（d_val == d_aro が必要）
        if cfg.use_cross_stream_attn:
            assert cfg.d_val == cfg.d_aro, (
                "use_cross_stream_attn=True には d_val == d_aro が必要です。"
                f"現在: d_val={cfg.d_val}, d_aro={cfg.d_aro}"
            )
            self.val_cross_attn = CrossStreamAttention(cfg.d_val, cfg.n_heads)
            self.aro_cross_attn = CrossStreamAttention(cfg.d_aro, cfg.n_heads)

        # [改善1] Temporal Attention（各ストリーム独立）
        if cfg.use_temporal_attn:
            self.val_temporal = TemporalAttention(cfg.d_val)
            self.aro_temporal = TemporalAttention(cfg.d_aro)

        # Module C: Fusion & Heads
        # [改善3] Gated Fusion
        if cfg.use_gated_fusion:
            self.gated_fusion = GatedFusion(cfg.d_val, cfg.d_aro)

        self.user_proj      = nn.Linear(cfg.d_val + cfg.d_aro, cfg.d_candidate)
        self.candidate_proj = CandidateProjector(cfg)
        self.val_head       = nn.Linear(cfg.d_val, 1)
        self.aro_head       = nn.Linear(cfg.d_aro, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding) and m.padding_idx is not None:
                nn.init.normal_(m.weight, std=0.01)
                with torch.no_grad():
                    m.weight[m.padding_idx].zero_()

    def encode(
        self,
        user_id:        torch.Tensor,
        weather_seq:    torch.Tensor,
        mobility_seq:   torch.Tensor,
        item_seq:       torch.Tensor,
        audio_cat_seq:  torch.Tensor,
        audio_cont_seq: torch.Tensor,
        intensity_seq:  torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        各改善モジュールを適用して (h_val, h_aro) を返す。

        内部フロー:
          1. 各ストリームで GRU/Transformer → (B, T, D)
          2. [改善2] Cross-Stream Attention で相互強化
          3. [改善1] Temporal Attention で (B, D) に集約
        """
        # Step 1: 各ストリームの系列エンコード
        val_seq = self.valence_stream(user_id, weather_seq, mobility_seq)   # (B, T, d_val)
        aro_seq = self.arousal_stream(item_seq, audio_cat_seq, audio_cont_seq, intensity_seq)  # (B, T, d_aro)

        # Step 2: [改善2] Cross-Stream Attention
        #   val_seq が Arousal 系列を参照 → 音楽文脈で気分を補正
        #   aro_seq が Valence 系列を参照 → 気分文脈でテンションを補正
        if self.cfg.use_cross_stream_attn:
            val_seq = self.val_cross_attn(query_seq=val_seq, context_seq=aro_seq)
            aro_seq = self.aro_cross_attn(query_seq=aro_seq, context_seq=val_seq)

        # Step 3: [改善1] Temporal Attention で系列を (B, D) に集約
        if self.cfg.use_temporal_attn:
            h_val = self.val_temporal(val_seq)  # (B, d_val)
            h_aro = self.aro_temporal(aro_seq)  # (B, d_aro)
        else:
            # フォールバック: 最終ステップのみ使用（改善なしの挙動）
            h_val = val_seq[:, -1, :]
            h_aro = aro_seq[:, -1, :]

        return h_val, h_aro

    def forward(
        self,
        user_id:        torch.Tensor,
        weather_seq:    torch.Tensor,
        mobility_seq:   torch.Tensor,
        item_seq:       torch.Tensor,
        audio_cat_seq:  torch.Tensor,
        audio_cont_seq: torch.Tensor,
        intensity_seq:  torch.Tensor,
        all_item_ids:   torch.Tensor,  # (n_items,)
        all_key_ids:    torch.Tensor,  # (n_items,)
        all_genre_ids:  torch.Tensor,  # (n_items,)
        all_audio_cont: torch.Tensor,  # (n_items, n_audio_cont)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        h_val, h_aro = self.encode(
            user_id, weather_seq, mobility_seq,
            item_seq, audio_cat_seq, audio_cont_seq, intensity_seq,
        )

        # [改善3] Gated Fusion で h_val / h_aro を統合
        if self.cfg.use_gated_fusion:
            h_fused = self.gated_fusion(h_val, h_aro)  # (B, d_val+d_aro)
        else:
            h_fused = torch.cat([h_val, h_aro], dim=-1)

        # Head 1: 推薦スコア = q · cand^T
        q          = self.user_proj(h_fused)
        cand_vec   = self.candidate_proj(all_item_ids, all_key_ids, all_genre_ids, all_audio_cont)
        rec_logits = q @ cand_vec.T  # (B, n_items)

        # Head 2 / 3: 感情変化量の予測（各ストリームの純粋な表現を使用）
        pred_delta_val = self.val_head(h_val).squeeze(-1)  # (B,)
        pred_delta_aro = self.aro_head(h_aro).squeeze(-1)  # (B,)

        return rec_logits, pred_delta_val, pred_delta_aro
