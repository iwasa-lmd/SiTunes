"""
評価指標の計算と結果の CSV 書き出し。

推薦精度  : Hit@K, NDCG@K, AUC（推薦スコアの正例 vs 全負例）
感情予測  : MAE, RMSE（Valence / Arousal それぞれ）
"""
from __future__ import annotations

import csv
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dsd_casr.model.model import DSD_CASR
from dsd_casr.config import DSDConfig


# ---------------------------------------------------------------------------
# 評価本体
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:          DSD_CASR,
    loader:         DataLoader,
    cfg:            DSDConfig,
    device:         torch.device,
    all_item_ids:   torch.Tensor,
    all_key_ids:    torch.Tensor,
    all_genre_ids:  torch.Tensor,
    all_audio_cont: torch.Tensor,
    top_k: int = 10,
) -> dict[str, float]:
    """
    Returns
    -------
    {
      "Hit@K":     ...,
      "NDCG@K":    ...,
      "AUC":       ...,   # P(score_pos > score_neg) の平均
      "Val_MAE":   ...,
      "Val_RMSE":  ...,
      "Aro_MAE":   ...,
      "Aro_RMSE":  ...,
    }
    """
    model.eval()

    all_items  = all_item_ids.to(device)
    all_keys   = all_key_ids.to(device)
    all_genres = all_genre_ids.to(device)
    all_audio  = all_audio_cont.to(device)

    hits, ndcgs, aucs   = [], [], []
    val_abs_errs, val_sq_errs = [], []
    aro_abs_errs, aro_sq_errs = [], []

    for batch in loader:
        uid    = batch["user_id"].to(device)
        w_seq  = batch["weather_seq"].to(device)
        m_seq  = batch["mobility_seq"].to(device)
        i_seq  = batch["item_seq"].to(device)
        a_cat  = batch["audio_cat_seq"].to(device)
        a_cont = batch["audio_cont_seq"].to(device)
        intens = batch["intensity_seq"].to(device)

        t_item = batch["target_item"].to(device)
        t_dv   = batch["target_delta_val"].to(device)
        t_da   = batch["target_delta_aro"].to(device)

        rec_logits, pred_dv, pred_da = model(
            uid, w_seq, m_seq, i_seq, a_cat, a_cont, intens,
            all_items, all_keys, all_genres, all_audio,
        )

        # ── 推薦指標 ──────────────────────────────────────────────────────────
        scores    = rec_logits.cpu()
        topk_ids  = scores.topk(top_k, dim=-1).indices  # (B, K)
        pos_items = t_item.cpu()

        for i, pos in enumerate(pos_items):
            pos_score = scores[i, pos].item()
            neg_scores = torch.cat([scores[i, :pos], scores[i, pos + 1:]])

            # Hit@K / NDCG@K
            rank_in_topk = (topk_ids[i] == pos).nonzero(as_tuple=True)
            if len(rank_in_topk[0]) > 0:
                rank = rank_in_topk[0][0].item() + 1
                hits.append(1.0)
                ndcgs.append(1.0 / math.log2(rank + 1))
            else:
                hits.append(0.0)
                ndcgs.append(0.0)

            # AUC: 正例スコアが負例スコアを上回る割合
            auc = float((neg_scores < pos_score).float().mean().item())
            aucs.append(auc)

        # ── 感情予測指標 ──────────────────────────────────────────────────────
        dv_err = (pred_dv - t_dv).cpu()
        da_err = (pred_da - t_da).cpu()

        val_abs_errs.extend(dv_err.abs().tolist())
        val_sq_errs.extend(dv_err.pow(2).tolist())
        aro_abs_errs.extend(da_err.abs().tolist())
        aro_sq_errs.extend(da_err.pow(2).tolist())

    return {
        f"Hit@{top_k}":  float(np.mean(hits)),
        f"NDCG@{top_k}": float(np.mean(ndcgs)),
        "AUC":           float(np.mean(aucs)),
        "Val_MAE":       float(np.mean(val_abs_errs)),
        "Val_RMSE":      float(np.sqrt(np.mean(val_sq_errs))),
        "Aro_MAE":       float(np.mean(aro_abs_errs)),
        "Aro_RMSE":      float(np.sqrt(np.mean(aro_sq_errs))),
    }


# ---------------------------------------------------------------------------
# CSV 書き出し
# ---------------------------------------------------------------------------

def save_metrics(
    metrics:    dict[str, float],
    epoch:      int,
    split:      str,
    output_dir: str | Path = "results",
    filename:   str = "metrics.csv",
) -> None:
    """
    評価結果を CSV に追記する。

    ファイルが存在しない場合はヘッダ付きで新規作成する。

    Parameters
    ----------
    metrics    : evaluate() の返り値
    epoch      : 現在のエポック番号
    split      : "train" / "val" / "test" 等のラベル
    output_dir : 出力ディレクトリ
    filename   : CSV ファイル名
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epoch":     epoch,
        "split":     split,
        **{k: f"{v:.6f}" for k, v in metrics.items()},
    }

    write_header = not filepath.exists()
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
