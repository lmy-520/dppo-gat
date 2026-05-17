"""
models/actor_critic.py

基于 EdgeGAT 的 Actor-Critic 网络。

Actor:
    输入动态图状态
    输出每个节点选择下一跳的 logits: [N, N]

Critic:
    输入动态图状态
    输出当前图状态的价值 value: scalar
"""

import torch
import torch.nn as nn

from models.edge_gat import EdgeGATLayer, EdgeGATPolicy


class GraphValueCritic(nn.Module):
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
                EdgeGATLayer(
                    hidden_dim=hidden_dim,
                    edge_dim=edge_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features, edge_index, edge_attr):
        """
        输入：
            node_features: [N, node_dim]
            edge_index: [2, E]
            edge_attr: [E, edge_dim]

        输出：
            value: [1]
        """
        h = self.node_encoder(node_features)

        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)

        # 图级表示：所有节点 embedding 平均池化
        graph_emb = h.mean(dim=0)

        value = self.value_head(graph_emb)

        return value.squeeze(-1)


class EdgeGATActorCritic(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()

        self.actor = EdgeGATPolicy(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        self.critic = GraphValueCritic(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    def forward(self, node_features, edge_index, edge_attr):
        """
        返回：
            logits: [N, N]
            value: [1]
        """
        logits = self.actor(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        value = self.critic(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        return logits, value
