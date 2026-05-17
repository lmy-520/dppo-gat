"""
models/edge_gat.py

轻量版边感知 GAT 路由策略。

输入：
    node_features: [N, node_dim]
    edge_index: [2, E]
    edge_attr: [E, edge_dim]

输出：
    logits: [N, N]

含义：
    logits[i, j] 表示节点 i 选择节点 j 作为下一跳的得分。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeGATLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()

        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, edge_index, edge_attr):
        """
        h:
            [N, hidden_dim]

        edge_index:
            [2, E]
            edge_index[0] = src
            edge_index[1] = dst

        edge_attr:
            [E, edge_dim]
        """
        n_nodes = h.size(0)
        src = edge_index[0]
        dst = edge_index[1]

        h_src = h[src]
        h_dst = h[dst]

        attn_input = torch.cat([h_src, h_dst, edge_attr], dim=-1)
        attn_score = self.attn_mlp(attn_input).squeeze(-1)

        # 对每个 src 的出边做 softmax
        attn_weight = torch.zeros_like(attn_score)

        for i in range(n_nodes):
            mask = src == i

            if mask.any():
                attn_weight[mask] = F.softmax(attn_score[mask], dim=0)

        msg_input = torch.cat([h_dst, edge_attr], dim=-1)
        msg = self.msg_mlp(msg_input)
        msg = msg * attn_weight.unsqueeze(-1)

        agg = torch.zeros_like(h)
        agg.index_add_(0, src, msg)

        out = self.update(torch.cat([h, agg], dim=-1))

        return out


class EdgeGATPolicy(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.layers = nn.ModuleList(
            [
                EdgeGATLayer(hidden_dim=hidden_dim, edge_dim=edge_dim)
                for _ in range(num_layers)
            ]
        )

        self.edge_score = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features, edge_index, edge_attr):
        """
        返回：
            logits: [N, N]
        """
        h = self.node_encoder(node_features)

        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)

        n_nodes = node_features.size(0)
        device = node_features.device

        logits = torch.full(
            (n_nodes, n_nodes),
            -1e9,
            dtype=torch.float32,
            device=device,
        )

        if edge_index.numel() == 0:
            return logits

        src = edge_index[0]
        dst = edge_index[1]

        h_src = h[src]
        h_dst = h[dst]

        score_input = torch.cat([h_src, h_dst, edge_attr], dim=-1)
        scores = self.edge_score(score_input).squeeze(-1)

        logits[src, dst] = scores

        return logits
