"""
loss.py: BPR Loss + MSE の統合損失関数

Loss_total = Loss_rec (Sampled BPR) + alpha * Loss_val (MSE) + beta * Loss_aro (MSE)

BPR (Bayesian Personalized Ranking):
  L_BPR = -mean(log(sigmoid(pos_score - neg_score)))

  ネガティブサンプリング:
    各バッチサンプルに対して n_neg_samples 個の負例をランダムに選ぶ。
    正例 item_id と同じアイテムが選ばれた場合は別のアイテムに差し替える
    (完全排除は高コストのため、衝突確率は低い前提で簡易実装)。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def bpr_loss(
    rec_logits:   torch.Tensor,  # (B, n_items)
    target_items: torch.Tensor,  # (B,)  正例アイテム ID
    n_neg:        int = 10,
) -> torch.Tensor:
    """
    Sampled BPR Loss。

    Parameters
    ----------
    rec_logits   : (B, n_items) 全アイテムへのスコア
    target_items : (B,)         正例アイテム ID
    n_neg        : ネガティブサンプル数

    Returns
    -------
    scalar tensor
    """
    B, n_items = rec_logits.shape
    device = rec_logits.device

    # 正例スコア: (B,)
    pos_scores = rec_logits[torch.arange(B, device=device), target_items]  # (B,)

    # ネガティブサンプリング: (B, n_neg)
    neg_ids = torch.randint(0, n_items, (B, n_neg), device=device)

    # 正例と被ったらランダムにずらす (簡易実装: +1 mod n_items)
    collision = (neg_ids == target_items.unsqueeze(1))
    neg_ids = torch.where(collision, (neg_ids + 1) % n_items, neg_ids)

    # 負例スコア: (B, n_neg)
    neg_scores = rec_logits.gather(1, neg_ids)  # (B, n_neg)

    # BPR: -log(sigmoid(pos - neg)), 全負例について平均
    # pos_scores を (B, 1) にブロードキャスト
    diff = pos_scores.unsqueeze(1) - neg_scores  # (B, n_neg)
    loss = -F.logsigmoid(diff).mean()

    return loss


def compute_loss(
    rec_logits:    torch.Tensor,  # (B, n_items)
    pred_delta_val:torch.Tensor,  # (B,)
    pred_delta_aro:torch.Tensor,  # (B,)
    target_items:  torch.Tensor,  # (B,)
    target_dv:     torch.Tensor,  # (B,)
    target_da:     torch.Tensor,  # (B,)
    alpha:         float,
    beta:          float,
    n_neg:         int = 10,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    統合損失関数。

    L_total = L_rec + alpha * L_val + beta * L_aro

    Returns
    -------
    (total_loss, breakdown_dict)
    """
    loss_rec = bpr_loss(rec_logits, target_items, n_neg=n_neg)
    loss_val = F.mse_loss(pred_delta_val, target_dv)
    loss_aro = F.mse_loss(pred_delta_aro, target_da)

    total = loss_rec + alpha * loss_val + beta * loss_aro

    return total, {
        "rec":   loss_rec.item(),
        "val":   loss_val.item(),
        "aro":   loss_aro.item(),
        "total": total.item(),
    }
