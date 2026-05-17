"""
debug/test_action_interface.py

测试环境是否支持外部动作 actions。
"""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from env.uav_env import UAVConfig, UAVNetworkEnv
from routing.projection import random_valid_actions, logits_to_actions


def test_random_valid_actions():
    cfg = UAVConfig(
        n_nodes=20,
        episode_steps=20,
        seed=42,
    )

    env = UAVNetworkEnv(cfg)
    state = env.reset()

    print("Test 1: random valid actions")

    for step in range(10):
        actions = random_valid_actions(state["adj"], env.rng)

        state, metrics = env.step(actions=actions)

        print(
            f"step={step + 1}, "
            f"delivered={metrics['delivered']:.0f}, "
            f"dropped={metrics['dropped']:.0f}, "
            f"violations={metrics['constraint_violations']:.0f}"
        )

    print("Random valid action interface OK.")


def test_logits_to_actions():
    cfg = UAVConfig(
        n_nodes=20,
        episode_steps=20,
        seed=42,
    )

    env = UAVNetworkEnv(cfg)
    state = env.reset()

    print("\nTest 2: logits to actions")

    for step in range(10):
        n = cfg.n_nodes

        logits = np.random.randn(n, n)
        actions = logits_to_actions(logits, state["adj"])

        state, metrics = env.step(actions=actions)

        print(
            f"step={step + 1}, "
            f"delivered={metrics['delivered']:.0f}, "
            f"dropped={metrics['dropped']:.0f}, "
            f"violations={metrics['constraint_violations']:.0f}"
        )

    print("Logits action interface OK.")


if __name__ == "__main__":
    test_random_valid_actions()
    test_logits_to_actions()
