"""
KGMMoEDataset: SiTunes ポイントワイズ Dataset

設計方針:
  - 逐次推薦ではなく、各インタラクションを独立したサンプルとして扱う
  - cluster_id: 感情変化ボラティリティに基づく K-means クラスタ (学習データから fit)
  - mobility_id: wrist センサの activity type 最頻値 (0-5 の 6 クラス)
  - 除外特徴量: 時間帯・心拍数 (col 0)・歩数 (col 2)
  - pre_arousal: Arousal Head の残差学習 (baseline_shift) に使用

モックデータ生成:
  generate_mock_records() で実データなしに動作確認できる
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

# ── 使用する音響特徴量カラム ───────────────────────────────────────────────────
AUDIO_CONT_COLS: list[str] = [
    "loudness", "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "valence", "tempo",
    "F0final_sma_amean", "F0final_sma_stddev",
    "audspec_lengthL1norm_sma_stddev", "pcm_RMSenergy_sma_stddev",
    "pcm_fftMag_psySharpness_sma_amean", "pcm_fftMag_psySharpness_sma_stddev",
    "pcm_zcr_sma_amean", "pcm_zcr_sma_stddev",
]

# wrist.npy の列インデックス
_WRIST_INTENSITY_COL = 1   # 活動強度 (使用)
_WRIST_ACT_TYPE_COL  = 3   # 活動タイプ (mobility_id に使用)
# col 0 = 心拍数, col 2 = 歩数 → 除外


# =============================================================================
# データ読み込み
# =============================================================================

def load_records(data_dir: str, stage: int) -> list[dict]:
    """
    指定 Stage のインタラクションを読み込みレコードリストを返す。

    Returns
    -------
    list[dict]
        必須キー:
          user_id, item_id, timestamp, weather_id, mobility_id, act_intensity,
          key_id, genre_id, audio_cont (np.ndarray 16次元),
          pre_valence, pre_arousal,
          delta_valence, delta_arousal, rating
    """
    data_path = Path(data_dir)
    stage_dir = data_path / f"Stage{stage}"

    # 音楽メタデータ
    music_df = pd.read_csv(data_path / "music_metadata" / "music_info.csv")
    music_df = music_df.set_index("item_id")
    for col in AUDIO_CONT_COLS:
        if col in music_df.columns:
            music_df[col] = music_df[col].fillna(music_df[col].median())

    inter_df = pd.read_csv(stage_dir / "interactions.csv")

    with open(stage_dir / "env.json") as f:
        env_data: dict[str, dict] = json.load(f)

    wrist_data: np.ndarray = np.load(stage_dir / "wrist.npy")

    records: list[dict] = []

    for _, row in inter_df.iterrows():
        inter_id = int(row["inter_id"])

        # 感情データ欠損をスキップ
        if any(
            pd.isna(row.get(c))
            for c in ["emo_pre_valence", "emo_pre_arousal",
                      "emo_post_valence", "emo_post_arousal"]
        ):
            continue

        item_id = int(row["item_id"])
        if item_id not in music_df.index:
            continue

        # Wrist センサ
        wrist_idx = inter_id - 1
        if wrist_idx >= len(wrist_data):
            continue
        wrist_row = wrist_data[wrist_idx]  # (30, 4)

        act_intensity = float(np.mean(wrist_row[:, _WRIST_INTENSITY_COL]))

        act_types = wrist_row[:, _WRIST_ACT_TYPE_COL].astype(int).tolist()
        mobility_id = int(Counter(act_types).most_common(1)[0][0])
        mobility_id = max(0, min(mobility_id, 5))  # clip to 0-5

        # 環境データ
        env_key = str(inter_id)
        if env_key not in env_data:
            continue
        weather_id = int(env_data[env_key]["weather"][0])
        weather_id = max(0, min(weather_id, 2))  # clip to 0-2

        # 音楽特徴量
        mrow = music_df.loc[item_id]
        key_id   = int(mrow["key"])             if not pd.isna(mrow["key"])             else 0
        genre_id = int(mrow["general_genre_id"]) if not pd.isna(mrow["general_genre_id"]) else 0
        audio_cont = np.array(
            [float(mrow.get(c, 0.0)) for c in AUDIO_CONT_COLS], dtype=np.float32
        )

        # 感情・評価
        pre_val  = float(row["emo_pre_valence"])
        pre_aro  = float(row["emo_pre_arousal"])
        post_val = float(row["emo_post_valence"])
        post_aro = float(row["emo_post_arousal"])
        rating   = float(row["rating"]) if not pd.isna(row.get("rating")) else 3.0

        records.append({
            "user_id":       int(row["user_id"]),
            "item_id":       item_id,
            "timestamp":     int(row["timestamp"]),
            "weather_id":    weather_id,
            "mobility_id":   mobility_id,
            "act_intensity": act_intensity,
            "key_id":        key_id,
            "genre_id":      genre_id,
            "audio_cont":    audio_cont,
            "pre_valence":   pre_val,
            "pre_arousal":   pre_aro,
            "delta_valence": post_val - pre_val,
            "delta_arousal": post_aro - pre_aro,
            "rating":        rating,
        })

    return records


# =============================================================================
# ユーザークラスタリング
# =============================================================================

def fit_user_clusters(
    train_records: list[dict],
    n_clusters: int = 4,
) -> tuple[dict[int, int], KMeans, StandardScaler]:
    """
    学習データから感情ボラティリティに基づくユーザークラスタを計算する。

    特徴量: [std(Δval), std(Δaro), mean(|Δval|), mean(|Δaro|)] per user

    Returns
    -------
    cluster_map   : user_id → cluster_id (0-indexed)
    kmeans        : fit 済み KMeans (未知ユーザーの割り当てに使用)
    feat_scaler   : fit 済み StandardScaler
    """
    user_deltas: dict[int, list] = defaultdict(list)
    for r in train_records:
        user_deltas[r["user_id"]].append(
            (r["delta_valence"], r["delta_arousal"])
        )

    uids, feats = [], []
    for uid, deltas in user_deltas.items():
        dvs = [d[0] for d in deltas]
        das = [d[1] for d in deltas]
        feats.append([
            float(np.std(dvs)),
            float(np.std(das)),
            float(np.mean(np.abs(dvs))),
            float(np.mean(np.abs(das))),
        ])
        uids.append(uid)

    X = np.array(feats, dtype=np.float32)
    feat_scaler = StandardScaler()
    X_scaled = feat_scaler.fit_transform(X)

    # n_clusters をデータ数に合わせてクリップ
    k = min(n_clusters, len(uids))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    cluster_map = {uid: int(label) for uid, label in zip(uids, labels)}
    return cluster_map, kmeans, feat_scaler


def assign_cluster(
    records: list[dict],
    cluster_map: dict[int, int],
    kmeans: KMeans,
    feat_scaler: StandardScaler,
    default_cluster: int = 0,
) -> None:
    """
    records に cluster_id フィールドをインプレースで追加する。
    未知ユーザーは K-means 最近傍クラスタを割り当て。
    """
    # 未知ユーザーを検出してクラスタ割り当て
    unknown_uid_deltas: dict[int, list] = defaultdict(list)
    for r in records:
        uid = r["user_id"]
        if uid not in cluster_map:
            unknown_uid_deltas[uid].append(
                (r["delta_valence"], r["delta_arousal"])
            )

    unknown_cluster_map: dict[int, int] = {}
    if unknown_uid_deltas:
        for uid, deltas in unknown_uid_deltas.items():
            dvs = [d[0] for d in deltas]
            das = [d[1] for d in deltas]
            feat = np.array([[
                float(np.std(dvs)),
                float(np.std(das)),
                float(np.mean(np.abs(dvs))),
                float(np.mean(np.abs(das))),
            ]], dtype=np.float32)
            feat_scaled = feat_scaler.transform(feat)
            label = int(kmeans.predict(feat_scaled)[0])
            unknown_cluster_map[uid] = label

    for r in records:
        uid = r["user_id"]
        r["cluster_id"] = cluster_map.get(uid, unknown_cluster_map.get(uid, default_cluster))


# =============================================================================
# スケーラー
# =============================================================================

def fit_scalers(train_records: list[dict]) -> tuple[StandardScaler, StandardScaler]:
    """音響特徴量・活動強度の StandardScaler を学習データで fit する。"""
    audio_mat     = np.stack([r["audio_cont"]    for r in train_records])
    intensity_mat = np.array([[r["act_intensity"]] for r in train_records])

    audio_scaler     = StandardScaler().fit(audio_mat)
    intensity_scaler = StandardScaler().fit(intensity_mat)
    return audio_scaler, intensity_scaler


# =============================================================================
# 時系列分割
# =============================================================================

def split_by_time(
    records:     list[dict],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
) -> tuple[list[dict], list[dict], list[dict]]:
    """ユーザーごとにタイムスタンプ順でリーク無し分割。"""
    user_records: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        user_records[r["user_id"]].append(r)

    train_recs, val_recs, test_recs = [], [], []
    for uid, recs in user_records.items():
        recs_sorted = sorted(recs, key=lambda x: x["timestamp"])
        n = len(recs_sorted)
        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, int(n * val_ratio))

        train_recs.extend(recs_sorted[:n_train])
        val_recs.extend(recs_sorted[n_train: n_train + n_val])
        test_recs.extend(recs_sorted[n_train + n_val:])

    return train_recs, val_recs, test_recs


# =============================================================================
# 全候補アイテムテンソル
# =============================================================================

def build_candidate_tensors(
    data_dir:     str,
    n_items:      int,
    n_keys:       int,
    n_genres:     int,
    audio_scaler: StandardScaler | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """全アイテムの候補テンソルを構築する (推薦スコア計算用)。"""
    music_df = pd.read_csv(Path(data_dir) / "music_metadata" / "music_info.csv")
    music_df = music_df.set_index("item_id")
    for col in AUDIO_CONT_COLS:
        if col in music_df.columns:
            music_df[col] = music_df[col].fillna(music_df[col].median())

    item_ids_arr   = np.arange(n_items, dtype=np.int64)
    key_ids_arr    = np.zeros(n_items, dtype=np.int64)
    genre_ids_arr  = np.zeros(n_items, dtype=np.int64)
    audio_cont_arr = np.zeros((n_items, len(AUDIO_CONT_COLS)), dtype=np.float32)

    for item_id in range(n_items):
        if item_id in music_df.index:
            row = music_df.loc[item_id]
            key_ids_arr[item_id]   = int(row["key"])              if not pd.isna(row["key"])              else 0
            genre_ids_arr[item_id] = int(row["general_genre_id"]) if not pd.isna(row["general_genre_id"]) else 0
            audio_cont_arr[item_id] = np.array(
                [float(row.get(c, 0.0)) for c in AUDIO_CONT_COLS], dtype=np.float32
            )

    if audio_scaler is not None:
        audio_cont_arr = audio_scaler.transform(audio_cont_arr).astype(np.float32)

    key_ids_arr   = np.clip(key_ids_arr,   0, n_keys   - 1)
    genre_ids_arr = np.clip(genre_ids_arr, 0, n_genres - 1)

    return (
        torch.from_numpy(item_ids_arr),
        torch.from_numpy(key_ids_arr),
        torch.from_numpy(genre_ids_arr),
        torch.from_numpy(audio_cont_arr),
    )


# =============================================================================
# PyTorch Dataset
# =============================================================================

class KGMMoEDataset(Dataset):
    """
    ポイントワイズ Dataset。各インタラクションが 1 サンプル。

    cluster_id は事前に assign_cluster() で records に付与しておくこと。
    """

    def __init__(
        self,
        records:          list[dict],
        audio_scaler:     StandardScaler | None = None,
        intensity_scaler: StandardScaler | None = None,
    ):
        self.samples: list[dict] = []

        for r in records:
            # 音響特徴量の正規化
            audio_cont = r["audio_cont"].copy()
            if audio_scaler is not None:
                audio_cont = audio_scaler.transform(audio_cont.reshape(1, -1))[0].astype(np.float32)

            # 活動強度の正規化
            act_intensity = r["act_intensity"]
            if intensity_scaler is not None:
                act_intensity = float(intensity_scaler.transform([[act_intensity]])[0, 0])

            self.samples.append({
                "user_id":      torch.tensor(r["user_id"],    dtype=torch.long),
                "cluster_id":   torch.tensor(r.get("cluster_id", 0), dtype=torch.long),
                "weather_id":   torch.tensor(r["weather_id"], dtype=torch.long),
                "mobility_id":  torch.tensor(r["mobility_id"],dtype=torch.long),
                "item_id":      torch.tensor(r["item_id"],    dtype=torch.long),
                "key_id":       torch.tensor(r["key_id"],     dtype=torch.long),
                "genre_id":     torch.tensor(r["genre_id"],   dtype=torch.long),
                "audio_cont":   torch.tensor(audio_cont,      dtype=torch.float),
                "act_intensity":torch.tensor(act_intensity,   dtype=torch.float),
                "pre_arousal":  torch.tensor(r["pre_arousal"],dtype=torch.float),
                # ラベル
                "target_item":      torch.tensor(r["item_id"],        dtype=torch.long),
                "target_delta_val": torch.tensor(r["delta_valence"],  dtype=torch.float),
                "target_delta_aro": torch.tensor(r["delta_arousal"],  dtype=torch.float),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


# =============================================================================
# モックデータ生成 (動作確認用)
# =============================================================================

def generate_mock_records(
    n_records:  int = 200,
    n_users:    int = 10,
    n_items:    int = 50,
    n_clusters: int = 4,
    seed:       int = 42,
) -> list[dict]:
    """
    実データなしにパイプライン全体を動作確認するためのモックレコードを生成する。
    """
    rng = np.random.default_rng(seed)

    records = []
    for i in range(n_records):
        records.append({
            "user_id":       int(rng.integers(1, n_users + 1)),
            "item_id":       int(rng.integers(1, n_items + 1)),
            "timestamp":     int(1_625_000_000 + i * 3600),
            "weather_id":    int(rng.integers(0, 3)),
            "mobility_id":   int(rng.integers(0, 6)),
            "act_intensity": float(rng.uniform(0.0, 2.0)),
            "key_id":        int(rng.integers(0, 12)),
            "genre_id":      int(rng.integers(0, 10)),
            "audio_cont":    rng.standard_normal(16).astype(np.float32),
            "pre_valence":   float(rng.uniform(-1.0, 1.0)),
            "pre_arousal":   float(rng.uniform(-1.0, 1.0)),
            "delta_valence": float(rng.uniform(-0.5, 0.5)),
            "delta_arousal": float(rng.uniform(-0.5, 0.5)),
            "rating":        float(rng.integers(1, 6)),
            "cluster_id":    int(rng.integers(0, n_clusters)),
        })
    return records
