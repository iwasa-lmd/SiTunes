"""
metrics.py: AUC, MAE, RMSE, Hit@K, NDCG@K の計算関数

evaluate() は DataLoader を受け取り、全バッチで集計した評価指標辞書を返す。
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import KGMMoEConfig
from .model import KGMMoE


def _rec_auc(rec_logits: torch.Tensor, target_items: torch.Tensor) -> list[float]:
    """
    Rec AUC = 正例スコアが全負例スコアを上回る割合 (per sample)。

    Parameters
    ----------
    rec_logits   : (B, n_items)
    target_items : (B,)
    """
    B, n_items = rec_logits.shape
    aucs = []
    for i in range(B):
        pos_item  = target_items[i].item()
        pos_score = rec_logits[i, pos_item].item()
        all_scores = rec_logits[i].cpu().numpy()

        neg_mask = np.ones(n_items, dtype=bool)
        if pos_item < n_items:
            neg_mask[pos_item] = False
        neg_scores = all_scores[neg_mask]

        aucs.append(float(np.mean(pos_score > neg_scores)))
    return aucs


def _hit_ndcg(
    rec_logits:   torch.Tensor,
    target_items: torch.Tensor,
    top_k:        int,
) -> tuple[list[float], list[float]]:
    """Hit@K と NDCG@K を計算 (per sample)。"""
    hits, ndcgs = [], []
    for i in range(len(target_items)):
        pos_item = target_items[i].item()
        topk_idx = rec_logits[i].topk(top_k).indices.cpu().numpy()
        if pos_item in topk_idx:
            hits.append(1.0)
            rank = int(np.where(topk_idx == pos_item)[0][0]) + 1
            ndcgs.append(1.0 / math.log2(rank + 1))
        else:
            hits.append(0.0)
            ndcgs.append(0.0)
    return hits, ndcgs


@torch.no_grad()
def evaluate(
    model:          KGMMoE,
    loader:         DataLoader,
    cfg:            KGMMoEConfig,
    device:         torch.device,
    all_item_ids:   torch.Tensor,
    all_key_ids:    torch.Tensor,
    all_genre_ids:  torch.Tensor,
    all_audio_cont: torch.Tensor,
    top_k:          int = 10,
) -> dict[str, float]:
    """
    評価指標を計算して返す。

    Returns
    -------
    dict keys:
      Rec_AUC, Hit@K, NDCG@K,
      Val_MAE, Val_RMSE,
      Aro_MAE, Aro_RMSE
    """
    model.eval()

    auc_list: list[float] = []
    hit_list: list[float] = []
    ndcg_list:list[float] = []
    dv_true, dv_pred = [], []
    da_true, da_pred = [], []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        rec_logits, pred_dv, pred_da = model(
            batch["user_id"],
            batch["cluster_id"],
            batch["weather_id"],
            batch["mobility_id"],
            batch["item_id"],
            batch["key_id"],
            batch["genre_id"],
            batch["audio_cont"],
            batch["act_intensity"],
            batch["pre_arousal"],
            all_item_ids.to(device),
            all_key_ids.to(device),
            all_genre_ids.to(device),
            all_audio_cont.to(device),
        )

        target_items = batch["target_item"]

        auc_list.extend(_rec_auc(rec_logits, target_items))
        hits, ndcgs = _hit_ndcg(rec_logits, target_items, top_k)
        hit_list.extend(hits)
        ndcg_list.extend(ndcgs)

        dv_true.extend(batch["target_delta_val"].cpu().tolist())
        dv_pred.extend(pred_dv.cpu().tolist())
        da_true.extend(batch["target_delta_aro"].cpu().tolist())
        da_pred.extend(pred_da.cpu().tolist())

    dv_true_arr = np.array(dv_true)
    dv_pred_arr = np.array(dv_pred)
    da_true_arr = np.array(da_true)
    da_pred_arr = np.array(da_pred)

    return {
        "Rec_AUC":       float(np.mean(auc_list)),
        f"Hit@{top_k}":  float(np.mean(hit_list)),
        f"NDCG@{top_k}": float(np.mean(ndcg_list)),
        "Val_MAE":        float(np.mean(np.abs(dv_true_arr - dv_pred_arr))),
        "Val_RMSE":       float(np.sqrt(np.mean((dv_true_arr - dv_pred_arr) ** 2))),
        "Aro_MAE":        float(np.mean(np.abs(da_true_arr - da_pred_arr))),
        "Aro_RMSE":       float(np.sqrt(np.mean((da_true_arr - da_pred_arr) ** 2))),
    }
