"""
DSD-CASR 学習・評価エントリーポイント。

使い方:
    python -m dsd_casr.main
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split

from dsd_casr.config import DSDConfig
from dsd_casr.dataset import SiTunesDataset, make_mock_records, make_all_item_tensors
from dsd_casr.model.model import DSD_CASR
from dsd_casr.trainer import train_one_epoch
from dsd_casr.metrics import evaluate, save_metrics

RESULTS_DIR = "results"
TOP_K = 10


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── 設定 ──────────────────────────────────────────────────────────────────
    cfg = DSDConfig(
        n_users      = 6,
        n_items      = 51,
        seq_len      = 8,
        n_layers     = 2,
        encoder_type = "gru",   # "transformer" に切り替え可
    )

    # ── データ ────────────────────────────────────────────────────────────────
    records = make_mock_records(
        n_users=cfg.n_users - 1,
        n_items=cfg.n_items - 1,
        n_interactions_per_user=30,
        n_audio_cont=cfg.n_audio_cont,
    )
    dataset  = SiTunesDataset(records, seq_len=cfg.seq_len)
    n_train  = int(len(dataset) * 0.8)
    train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False)

    print(f"Dataset: total={len(dataset)}, train={len(train_ds)}, val={len(val_ds)}\n")

    # ── モデル・最適化器 ──────────────────────────────────────────────────────
    model     = DSD_CASR(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    all_tensors = make_all_item_tensors(cfg.n_items, cfg.n_keys, cfg.n_genres, cfg.n_audio_cont)

    # ── 学習ループ ────────────────────────────────────────────────────────────
    for epoch in range(1, 11):
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg, device, *all_tensors)
        val_metrics = evaluate(model, val_loader, cfg, device, *all_tensors, top_k=TOP_K)
        scheduler.step()

        # 標準出力
        print(
            f"Epoch {epoch:2d} | "
            f"Loss {train_loss['total']:.4f} "
            f"(rec={train_loss['rec']:.4f} val={train_loss['val']:.4f} aro={train_loss['aro']:.4f}) | "
            f"Hit@{TOP_K}={val_metrics[f'Hit@{TOP_K}']:.3f}  "
            f"NDCG@{TOP_K}={val_metrics[f'NDCG@{TOP_K}']:.3f}  "
            f"AUC={val_metrics['AUC']:.3f}  "
            f"Val_MAE={val_metrics['Val_MAE']:.4f}  "
            f"Val_RMSE={val_metrics['Val_RMSE']:.4f}  "
            f"Aro_MAE={val_metrics['Aro_MAE']:.4f}  "
            f"Aro_RMSE={val_metrics['Aro_RMSE']:.4f}"
        )

        # 学習損失・評価指標を別ファイルに保存
        save_metrics(
            {f"loss_{k}": v for k, v in train_loss.items()},
            epoch=epoch, split="train",
            output_dir=RESULTS_DIR, filename="train_loss.csv",
        )
        save_metrics(
            val_metrics,
            epoch=epoch, split="val",
            output_dir=RESULTS_DIR, filename="val_metrics.csv",
        )

    print(f"\n結果を {RESULTS_DIR}/train_loss.csv, {RESULTS_DIR}/val_metrics.csv に保存しました。")


if __name__ == "__main__":
    main()
