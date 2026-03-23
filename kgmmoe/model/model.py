"""
model.py: KGMMoE 統合モデル

処理フロー:
  1. ValenceTower(user, weather, mobility) → h_val
  2. ArousalTower(item, audio, intensity)  → h_aro
  3. MMoE(h_val, h_aro, cluster_id)        → [rec_repr, val_repr, aro_repr]
  4. RecHead   : rec_repr → user_repr → dot(user_repr, candidate) → logits (B, N)
  5. ValHead   : val_repr → pred_delta_val (B,)
  6. AroHead   : baseline_shift(pre_arousal) + deep_pred(aro_repr) → pred_delta_aro (B,)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import KGMMoEConfig
from .mmoe import MMoE
from .towers import ArousalTower, CandidateProjector, ValenceTower


class KGMMoE(nn.Module):
    """
    Knowledge-Guided MMoE for SiTunes music recommendation.

    Forward returns:
        rec_logits     : (B, n_items)  推薦スコア (全候補アイテム)
        pred_delta_val : (B,)          予測 ΔValence
        pred_delta_aro : (B,)          予測 ΔArousal
    """

    def __init__(self, cfg: KGMMoEConfig, dropout: float = 0.1):
        super().__init__()
        self.cfg = cfg

        # ── タワー ─────────────────────────────────────────────────────────────
        self.val_tower = ValenceTower(cfg)
        self.aro_tower = ArousalTower(cfg)

        # ── MMoE ───────────────────────────────────────────────────────────────
        self.mmoe = MMoE(cfg, dropout=dropout)

        # ── Recommendation Head ────────────────────────────────────────────────
        # rec_repr (d_expert) → user_repr (d_candidate)
        self.rec_proj = nn.Sequential(
            nn.Linear(cfg.d_expert, cfg.d_candidate * 2),
            nn.GELU(),
            nn.Linear(cfg.d_candidate * 2, cfg.d_candidate),
        )
        self.candidate_proj = CandidateProjector(cfg)

        # ── Valence Head ───────────────────────────────────────────────────────
        self.val_head = nn.Sequential(
            nn.Linear(cfg.d_expert, cfg.d_val_head),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cfg.d_val_head, 1),
        )

        # ── Arousal Head (残差学習) ─────────────────────────────────────────────
        # baseline_shift: pre_arousal → 1 層 Linear (重みとバイアスのみ)
        self.aro_baseline = nn.Linear(1, 1)

        # deep_prediction: aro_repr → scalar
        self.aro_head = nn.Sequential(
            nn.Linear(cfg.d_expert, cfg.d_aro_head),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cfg.d_aro_head, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        # ユーザー・環境コンテキスト
        user_id:       torch.Tensor,  # (B,)
        cluster_id:    torch.Tensor,  # (B,)
        weather_id:    torch.Tensor,  # (B,)
        mobility_id:   torch.Tensor,  # (B,)
        # アイテム・活動コンテキスト
        item_id:       torch.Tensor,  # (B,)  現在インタラクションのアイテム
        key_id:        torch.Tensor,  # (B,)
        genre_id:      torch.Tensor,  # (B,)
        audio_cont:    torch.Tensor,  # (B, 16)
        act_intensity: torch.Tensor,  # (B,)
        # 残差学習用
        pre_arousal:   torch.Tensor,  # (B,)
        # 全候補アイテム (推薦スコア計算用)
        all_item_ids:   torch.Tensor,  # (n_items,)
        all_key_ids:    torch.Tensor,  # (n_items,)
        all_genre_ids:  torch.Tensor,  # (n_items,)
        all_audio_cont: torch.Tensor,  # (n_items, 16)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # ── Step 1: タワー ─────────────────────────────────────────────────────
        h_val = self.val_tower(user_id, weather_id, mobility_id)  # (B, d_val_tower)
        h_aro = self.aro_tower(item_id, key_id, genre_id, audio_cont, act_intensity)  # (B, d_aro_tower)

        # ── Step 2: MMoE ───────────────────────────────────────────────────────
        task_reprs = self.mmoe(h_val, h_aro, cluster_id)
        rec_repr = task_reprs[0]  # (B, d_expert)
        val_repr = task_reprs[1]  # (B, d_expert)
        aro_repr = task_reprs[2]  # (B, d_expert)

        # ── Step 3: Recommendation Head ────────────────────────────────────────
        user_repr = self.rec_proj(rec_repr)  # (B, d_candidate)

        # 全候補アイテムのベクトル: (n_items, d_candidate)
        candidate_vecs = self.candidate_proj(
            all_item_ids, all_key_ids, all_genre_ids, all_audio_cont
        )

        # 内積スコア: (B, n_items)
        rec_logits = user_repr @ candidate_vecs.t()

        # ── Step 4: Valence Head ───────────────────────────────────────────────
        pred_delta_val = self.val_head(val_repr).squeeze(-1)  # (B,)

        # ── Step 5: Arousal Head (残差学習) ────────────────────────────────────
        baseline  = self.aro_baseline(pre_arousal.unsqueeze(-1)).squeeze(-1)  # (B,)
        deep_pred = self.aro_head(aro_repr).squeeze(-1)                        # (B,)
        pred_delta_aro = baseline + deep_pred                                   # (B,)

        return rec_logits, pred_delta_val, pred_delta_aro
