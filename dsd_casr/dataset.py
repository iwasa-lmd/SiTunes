from __future__ import annotations
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


class SiTunesDataset(Dataset):
    """
    SiTunes Stage 2 のスライディングウィンドウ Dataset。

    各サンプルは時刻 t を終端とする長さ seq_len のウィンドウと
    t+1 ステップの正解ラベルから構成される。

    Parameters
    ----------
    records : list[dict]
        ユーザーごとに時系列ソート済みのインタラクション辞書リスト。
        必須キー:
          user_id      : int
          item_id      : int
          weather_id   : int   (1=sunny, 2=cloudy, 3=rainy)
          mobility_id  : int   (1=still … 6=running)
          act_intensity: float (activity_intensity_mean)
          audio_cat    : tuple[int, int]   (key_id, genre_id)
          audio_cont   : np.ndarray shape=(n_audio_cont,)  正規化済み
          delta_valence: float (emo_post_valence - emo_pre_valence)
          delta_arousal: float (emo_post_arousal - emo_pre_arousal)
    seq_len : int
        ウィンドウ幅。
    """

    def __init__(self, records: list[dict], seq_len: int = 10):
        self.seq_len = seq_len
        self.samples: list[dict] = []

        user_records: dict[int, list[dict]] = defaultdict(list)
        for r in records:
            user_records[r["user_id"]].append(r)

        for uid, recs in user_records.items():
            if len(recs) < seq_len + 1:
                continue
            for end in range(seq_len, len(recs)):
                window   = recs[end - seq_len : end]
                target_r = recs[end]
                self.samples.append(_build_sample(uid, window, target_r))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def _build_sample(uid: int, window: list[dict], target: dict) -> dict:
    return {
        "user_id": uid,
        # Stream A: 環境コンテキスト
        "weather_seq":    torch.tensor([r["weather_id"]    for r in window], dtype=torch.long),
        "mobility_seq":   torch.tensor([r["mobility_id"]   for r in window], dtype=torch.long),
        # Stream B: 楽曲・活動コンテキスト
        "item_seq":       torch.tensor([r["item_id"]       for r in window], dtype=torch.long),
        "intensity_seq":  torch.tensor([r["act_intensity"] for r in window], dtype=torch.float).unsqueeze(-1),
        "audio_cat_seq":  torch.tensor([list(r["audio_cat"]) for r in window], dtype=torch.long),
        "audio_cont_seq": torch.tensor(
            np.stack([r["audio_cont"] for r in window]), dtype=torch.float
        ),
        # 正解ラベル
        "target_item":      torch.tensor(target["item_id"],       dtype=torch.long),
        "target_delta_val": torch.tensor(target["delta_valence"], dtype=torch.float),
        "target_delta_aro": torch.tensor(target["delta_arousal"], dtype=torch.float),
    }


# ---------------------------------------------------------------------------
# モックデータ生成（実データがない環境での動作確認用）
# ---------------------------------------------------------------------------

def make_mock_records(
    n_users: int = 5,
    n_items: int = 50,
    n_interactions_per_user: int = 30,
    n_audio_cont: int = 16,
    seed: int = 42,
) -> list[dict]:
    """SiTunes 実データの代わりに使えるランダムレコードを生成する。"""
    rng = np.random.default_rng(seed)
    records = []
    for uid in range(1, n_users + 1):
        for _ in range(n_interactions_per_user):
            records.append({
                "user_id":       uid,
                "item_id":       int(rng.integers(1, n_items)),
                "weather_id":    int(rng.integers(1, 4)),
                "mobility_id":   int(rng.integers(1, 7)),
                "act_intensity": float(rng.uniform(0, 1)),
                "audio_cat":     (int(rng.integers(1, 13)), int(rng.integers(1, 10))),
                "audio_cont":    rng.standard_normal(n_audio_cont).astype(np.float32),
                "delta_valence": float(rng.uniform(-1, 1)),
                "delta_arousal": float(rng.uniform(-1, 1)),
            })
    return records


def make_all_item_tensors(n_items: int, n_keys: int, n_genres: int, n_audio_cont: int, seed: int = 0):
    """全アイテムのルックアップテーブルを返す（学習中は定数として扱う）。"""
    rng = np.random.default_rng(seed)
    return (
        torch.arange(n_items, dtype=torch.long),
        torch.from_numpy(rng.integers(0, n_keys,   n_items).astype(np.int64)),
        torch.from_numpy(rng.integers(0, n_genres, n_items).astype(np.int64)),
        torch.from_numpy(rng.standard_normal((n_items, n_audio_cont)).astype(np.float32)),
    )
