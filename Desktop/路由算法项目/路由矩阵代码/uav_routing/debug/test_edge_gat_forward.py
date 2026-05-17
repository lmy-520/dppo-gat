"""
debug/test_edge_gat_forward.py

测试 EdgeGATPolicy 是否能完成：
state -> logits -> actions -> env.step(actions)
"""

import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from env.uav_env import UAVConfig, UAVNetworkEnv
from models.edge_gat import EdgeGATPolicy
from routing.projection import logits_to_actions


def main():
    cfg = UAVConfig(
        n_nodes=20,
        episode_steps=20,
        seed=42,
    )

    env = UAVNetworkEnv(cfg)
    state = env.reset()

    model = EdgeGATPolicy(
        node_dim=9,
        edge_dim=5,
        hidden_dim=64,
        num_layers=2,
    )

    model.eval()

    print("Testing EdgeGAT forward...")

    for step in range(10):
        node_features = torch.tensor(
            state["node_features"],
            dtype=torch.float32,
        )

        edge_index = torch.tensor(
            state["edge_index"],
            dtype=torch.long,
        )

        edge_attr = torch.tensor(
            state["edge_attr"],
            dtype=torch.float32,
        )

        with torch.no_grad():
            logits = model(node_features, edge_index, edge_attr)

        actions = logits_to_actions(
            logits.detach().cpu().numpy(),
            state["adj"],
        )

        state, metrics = env.step(actions=actions)

        print(
            f"step={step + 1}, "
            f"logits_shape={tuple(logits.shape)}, "
            f"delivered={metrics['delivered']:.0f}, "
            f"dropped={metrics['dropped']:.0f}, "
            f"violations={metrics['constraint_violations']:.0f}"
        )

    print("EdgeGAT forward interface OK.")


if __name__ == "__main__":
    main()
