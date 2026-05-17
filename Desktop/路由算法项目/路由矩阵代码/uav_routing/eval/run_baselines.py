"""
eval/run_baselines.py

运行 V0 基础路由策略，并把结果保存到 data/results/baseline_results.csv

运行方式：
    在项目根目录 uav_routing 下执行：
    python -m eval.run_baselines
"""

import os
import sys
import pandas as pd

# 保证可以从项目根目录导入 env/uav_env.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from env.uav_env import UAVConfig, UAVNetworkEnv


def run_one_seed(seed: int):
    cfg = UAVConfig(
        n_nodes=20,
        episode_steps=200,
        seed=seed,
    )

    policies = ["random", "shortest", "min_delay", "energy"]
    rows = []

    for policy in policies:
        env = UAVNetworkEnv(cfg)
        metrics = env.run_episode(policy=policy)

        row = {
            "seed": seed,
            "policy": policy,
        }
        row.update(metrics)
        rows.append(row)

        print(f"[Seed {seed}] Policy={policy}, delivery_ratio={metrics['delivery_ratio']:.4f}, drop_ratio={metrics['drop_ratio']:.4f}")

    return rows


def main():
    all_rows = []

    # 先跑 5 个随机种子，后面可以改成 10、20
    seeds = [0, 1, 2, 3, 4]

    for seed in seeds:
        rows = run_one_seed(seed)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    save_dir = os.path.join(PROJECT_ROOT, "data", "results")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "baseline_results.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("\nSaved results to:")
    print(save_path)

    print("\nAverage results:")
    print(
        df.groupby("policy")[
            [
                "delivery_ratio",
                "drop_ratio",
                "avg_delay",
                "avg_remaining_energy",
                "constraint_violations",
                "total_hops",
            ]
        ].mean()
    )


if __name__ == "__main__":
    main()
