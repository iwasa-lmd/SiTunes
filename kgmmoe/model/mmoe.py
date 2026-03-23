"""
mmoe.py: Expert, Gate, MMoE 本体

設計:
  - Expert: Val Tower + Aro Tower の結合を入力とする共有 MLP
  - Gate  : タワー出力 + cluster_id Embedding を入力とし、Expert の重みを出力
            → 感情の揺れやすさでどの Expert を重視するかを動的に変える
  - MMoE  : n_tasks 個のタスク固有 Gate × n_experts 個の Expert
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import KGMMoEConfig


class Expert(nn.Module):
    """
    共有 Expert ネットワーク。

    Input : (B, d_val_tower + d_aro_tower)
    Output: (B, d_expert)
    """

    def __init__(self, in_dim: int, d_expert: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_expert * 2),
            nn.LayerNorm(d_expert * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_expert * 2, d_expert),
            nn.GELU(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Gate(nn.Module):
    """
    タスク固有ゲートネットワーク。

    cluster_id の Embedding を入力に含めることで、
    感情ボラティリティに応じて Expert の重み付けを動的に変える。

    Input : combined (B, d_val_tower + d_aro_tower), cluster_emb (B, d_cluster)
    Output: (B, n_experts)  ← softmax 済み重みベクトル
    """

    def __init__(self, in_dim: int, d_cluster: int, n_experts: int, dropout: float = 0.1):
        super().__init__()
        gate_in = in_dim + d_cluster
        self.net = nn.Sequential(
            nn.Linear(gate_in, n_experts * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_experts * 4, n_experts),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        combined:    torch.Tensor,  # (B, d_val_tower + d_aro_tower)
        cluster_emb: torch.Tensor,  # (B, d_cluster)
    ) -> torch.Tensor:              # (B, n_experts)
        gate_input = torch.cat([combined, cluster_emb], dim=-1)
        return F.softmax(self.net(gate_input), dim=-1)


class MMoE(nn.Module):
    """
    Multi-gate Mixture-of-Experts 本体。

    n_experts 個の Expert を n_tasks 個の Gate で重み付けして混合する。
    Gate には cluster_id の Embedding を条件として与える。

    Input:
        h_val      : (B, d_val_tower)   ← Valence Tower 出力
        h_aro      : (B, d_aro_tower)   ← Arousal Tower 出力
        cluster_id : (B,)               ← ユーザークラスタ ID

    Output:
        list of (B, d_expert), length = n_tasks
        順序: [rec_repr, val_repr, aro_repr]
    """

    TASK_REC = 0
    TASK_VAL = 1
    TASK_ARO = 2

    def __init__(self, cfg: KGMMoEConfig, dropout: float = 0.1):
        super().__init__()
        in_dim = cfg.d_val_tower + cfg.d_aro_tower

        self.experts = nn.ModuleList([
            Expert(in_dim, cfg.d_expert, dropout)
            for _ in range(cfg.n_experts)
        ])

        self.cluster_emb = nn.Embedding(cfg.n_clusters, cfg.d_cluster)
        nn.init.normal_(self.cluster_emb.weight, std=0.02)

        self.gates = nn.ModuleList([
            Gate(in_dim, cfg.d_cluster, cfg.n_experts, dropout)
            for _ in range(cfg.n_tasks)
        ])

        self.n_experts = cfg.n_experts

    def forward(
        self,
        h_val:      torch.Tensor,  # (B, d_val_tower)
        h_aro:      torch.Tensor,  # (B, d_aro_tower)
        cluster_id: torch.Tensor,  # (B,)
    ) -> list[torch.Tensor]:       # list of (B, d_expert), length = n_tasks
        combined    = torch.cat([h_val, h_aro], dim=-1)  # (B, d_val + d_aro)
        cluster_emb = self.cluster_emb(cluster_id)        # (B, d_cluster)

        # Expert 出力をスタック: (B, n_experts, d_expert)
        expert_outs = torch.stack(
            [expert(combined) for expert in self.experts], dim=1
        )

        task_outputs = []
        for gate in self.gates:
            gate_weights = gate(combined, cluster_emb)   # (B, n_experts)
            # 重み付き和: (B, d_expert)
            weighted = (expert_outs * gate_weights.unsqueeze(-1)).sum(dim=1)
            task_outputs.append(weighted)

        return task_outputs  # [rec_repr, val_repr, aro_repr]
