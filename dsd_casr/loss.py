"""
マルチタスク損失の計算。

[改善4] BPR Loss
  全アイテム Cross-Entropy は負例なしで計算するが非効率。
  BPR (Bayesian Personalized Ranking, Rendle et al. 2009) は
  正例と負例のスコア差を sigmoid-log で最大化し、AUC を直接最適化する。
  SASRec / BERT4Rec / GRU4Rec いずれも BPR をオプションとして採用している。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _bpr_loss(rec_logits: torch.Tensor, target_item: torch.Tensor) -> torch.Tensor:
    """
    Sampled BPR Loss。

    1 バッチにつき、正例スコアとランダムサンプリングした負例スコアを比較する。
    計算量: O(B) ─ 全アイテムスコアは既に計算済みなので gather のみ。
    """
    n_items = rec_logits.size(1)

    # 正例スコア
    pos_score = rec_logits.gather(1, target_item.unsqueeze(1)).squeeze(1)

    # 負例: ランダムサンプリング（正例と被らないよう試みる）
    neg_item = torch.randint(0, n_items, target_item.shape, device=rec_logits.device)
    # 万一正例と一致した場合は (n_items-1) で逃がす
    neg_item = torch.where(neg_item == target_item, (neg_item + 1) % n_items, neg_item)
    neg_score = rec_logits.gather(1, neg_item.unsqueeze(1)).squeeze(1)

    return -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8))


def compute_loss(
    rec_logits:       torch.Tensor,  # (B, n_items)
    pred_delta_val:   torch.Tensor,  # (B,)
    pred_delta_aro:   torch.Tensor,  # (B,)
    target_item:      torch.Tensor,  # (B,)
    target_delta_val: torch.Tensor,  # (B,)
    target_delta_aro: torch.Tensor,  # (B,)
    alpha:     float = 0.1,
    beta:      float = 0.1,
    loss_type: str   = "bpr",        # "ce" or "bpr"
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Loss_total = Loss_rec + alpha * Loss_val + beta * Loss_aro

    loss_type:
      "ce"  : 全アイテム Cross-Entropy（学習安定、小規模データ向き）
      "bpr" : Sampled BPR（ランキング精度・AUC を直接最適化）

    Returns
    -------
    loss_total : backward 可能なスカラー Tensor
    breakdown  : {"rec": ..., "val": ..., "aro": ...}
    """
    if loss_type == "bpr":
        loss_rec = _bpr_loss(rec_logits, target_item)
    else:
        loss_rec = F.cross_entropy(rec_logits, target_item)

    loss_val = F.mse_loss(pred_delta_val, target_delta_val)
    loss_aro = F.mse_loss(pred_delta_aro, target_delta_aro)

    loss_total = loss_rec + alpha * loss_val + beta * loss_aro
    return loss_total, {
        "rec": loss_rec.item(),
        "val": loss_val.item(),
        "aro": loss_aro.item(),
    }
