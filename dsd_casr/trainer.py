"""1エポック分の学習ループ。"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dsd_casr.config import DSDConfig
from dsd_casr.loss import compute_loss
from dsd_casr.model.model import DSD_CASR


def train_one_epoch(
    model:          DSD_CASR,
    loader:         DataLoader,
    optimizer:      torch.optim.Optimizer,
    cfg:            DSDConfig,
    device:         torch.device,
    all_item_ids:   torch.Tensor,
    all_key_ids:    torch.Tensor,
    all_genre_ids:  torch.Tensor,
    all_audio_cont: torch.Tensor,
) -> dict[str, float]:
    """
    Returns
    -------
    {"total": ..., "rec": ..., "val": ..., "aro": ...}  ── バッチ平均
    """
    model.train()
    totals: dict[str, float] = {"total": 0.0, "rec": 0.0, "val": 0.0, "aro": 0.0}

    all_items  = all_item_ids.to(device)
    all_keys   = all_key_ids.to(device)
    all_genres = all_genre_ids.to(device)
    all_audio  = all_audio_cont.to(device)

    for batch in loader:
        optimizer.zero_grad()

        rec_logits, pred_dv, pred_da = model(
            batch["user_id"].to(device),
            batch["weather_seq"].to(device),
            batch["mobility_seq"].to(device),
            batch["item_seq"].to(device),
            batch["audio_cat_seq"].to(device),
            batch["audio_cont_seq"].to(device),
            batch["intensity_seq"].to(device),
            all_items, all_keys, all_genres, all_audio,
        )

        loss, breakdown = compute_loss(
            rec_logits, pred_dv, pred_da,
            batch["target_item"].to(device),
            batch["target_delta_val"].to(device),
            batch["target_delta_aro"].to(device),
            alpha=cfg.alpha, beta=cfg.beta,
            loss_type=cfg.loss_type,
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        totals["total"] += loss.item()
        for k, v in breakdown.items():
            totals[k] += v

    n = len(loader)
    return {k: v / n for k, v in totals.items()}
