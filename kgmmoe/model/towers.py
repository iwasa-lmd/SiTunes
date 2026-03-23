"""
towers.py: Valence Tower, Arousal Tower, Candidate Projector

非対称特徴量ルーティング:
  ValenceTower  ← user_id, weather_id, mobility_id (ユーザー・環境)
  ArousalTower  ← item_id, key_id, genre_id, audio_cont, act_intensity (アイテム)
  CandidateProjector ← 全アイテム候補の埋め込み (推薦スコア内積用)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import KGMMoEConfig


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
        nn.GELU(),
    )


class ValenceTower(nn.Module):
    """
    Valence に強く影響するユーザー・環境側の特徴抽出タワー。

    Input : user_id (B,), weather_id (B,), mobility_id (B,)
    Output: (B, d_val_tower)
    """

    def __init__(self, cfg: KGMMoEConfig):
        super().__init__()
        self.user_emb    = nn.Embedding(cfg.n_users,    cfg.d_user,    padding_idx=0)
        self.weather_emb = nn.Embedding(cfg.n_weather,  cfg.d_weather)
        self.mobility_emb= nn.Embedding(cfg.n_mobility, cfg.d_mobility)

        in_dim = cfg.d_user + cfg.d_weather + cfg.d_mobility
        self.mlp = _mlp(in_dim, in_dim * 2, cfg.d_val_tower)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    nn.init.zeros_(m.weight[m.padding_idx])
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        user_id:     torch.Tensor,  # (B,)
        weather_id:  torch.Tensor,  # (B,)
        mobility_id: torch.Tensor,  # (B,)
    ) -> torch.Tensor:              # (B, d_val_tower)
        x = torch.cat([
            self.user_emb(user_id),
            self.weather_emb(weather_id),
            self.mobility_emb(mobility_id),
        ], dim=-1)
        return self.mlp(x)


class ArousalTower(nn.Module):
    """
    Arousal に強く影響するアイテム・活動側の特徴抽出タワー。

    Input : item_id (B,), key_id (B,), genre_id (B,),
            audio_cont (B, n_audio_cont), act_intensity (B,)
    Output: (B, d_aro_tower)
    """

    def __init__(self, cfg: KGMMoEConfig):
        super().__init__()
        self.item_emb  = nn.Embedding(cfg.n_items,  cfg.d_item,  padding_idx=0)
        self.key_emb   = nn.Embedding(cfg.n_keys,   cfg.d_key)
        self.genre_emb = nn.Embedding(cfg.n_genres, cfg.d_genre)

        self.audio_proj = nn.Sequential(
            nn.Linear(cfg.n_audio_cont, cfg.d_audio_cont),
            nn.GELU(),
        )

        # act_intensity は 1次元 → そのまま結合
        in_dim = cfg.d_item + cfg.d_key + cfg.d_genre + cfg.d_audio_cont + 1
        self.mlp = _mlp(in_dim, in_dim, cfg.d_aro_tower)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    nn.init.zeros_(m.weight[m.padding_idx])
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        item_id:      torch.Tensor,  # (B,)
        key_id:       torch.Tensor,  # (B,)
        genre_id:     torch.Tensor,  # (B,)
        audio_cont:   torch.Tensor,  # (B, n_audio_cont)
        act_intensity:torch.Tensor,  # (B,)
    ) -> torch.Tensor:               # (B, d_aro_tower)
        x = torch.cat([
            self.item_emb(item_id),
            self.key_emb(key_id),
            self.genre_emb(genre_id),
            self.audio_proj(audio_cont),
            act_intensity.unsqueeze(-1),
        ], dim=-1)
        return self.mlp(x)


class CandidateProjector(nn.Module):
    """
    全アイテム候補を固定次元ベクトルに射影する。
    推薦スコア = user_repr @ candidate_matrix.T

    Input : item_ids (N,), key_ids (N,), genre_ids (N,), audio_cont (N, 16)
    Output: (N, d_candidate)
    """

    def __init__(self, cfg: KGMMoEConfig):
        super().__init__()
        self.item_emb  = nn.Embedding(cfg.n_items,  cfg.d_item,  padding_idx=0)
        self.key_emb   = nn.Embedding(cfg.n_keys,   cfg.d_key)
        self.genre_emb = nn.Embedding(cfg.n_genres, cfg.d_genre)

        self.audio_proj = nn.Sequential(
            nn.Linear(cfg.n_audio_cont, cfg.d_audio_cont),
            nn.GELU(),
        )

        in_dim = cfg.d_item + cfg.d_key + cfg.d_genre + cfg.d_audio_cont
        self.proj = nn.Sequential(
            nn.Linear(in_dim, cfg.d_candidate * 2),
            nn.GELU(),
            nn.Linear(cfg.d_candidate * 2, cfg.d_candidate),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    nn.init.zeros_(m.weight[m.padding_idx])
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        item_ids:   torch.Tensor,  # (N,)
        key_ids:    torch.Tensor,  # (N,)
        genre_ids:  torch.Tensor,  # (N,)
        audio_cont: torch.Tensor,  # (N, 16)
    ) -> torch.Tensor:             # (N, d_candidate)
        x = torch.cat([
            self.item_emb(item_ids),
            self.key_emb(key_ids),
            self.genre_emb(genre_ids),
            self.audio_proj(audio_cont),
        ], dim=-1)
        return self.proj(x)
