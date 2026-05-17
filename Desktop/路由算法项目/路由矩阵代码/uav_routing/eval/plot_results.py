"""
eval/plot_results.py

读取 data/results/baseline_results.csv，
统计不同 baseline 的平均结果，并画图。

运行方式：
    在项目根目录 uav_routing 下执行：
    python -m eval.plot_results
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


def plot_bar(df_summary, metric, ylabel, save_path):
    plt.figure(figsize=(8, 5))

    policies = df_summary["policy"]
    values = df_summary[metric]

    plt.bar(policies, values)
    plt.xlabel("Policy")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Comparison")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    result_dir = os.path.join(PROJECT_ROOT, "data", "results")
    csv_path = os.path.join(result_dir, "baseline_results.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find result file: {csv_path}")

    df = pd.read_csv(csv_path)

    summary = (
        df.groupby("policy")
        .agg(
            delivery_ratio=("delivery_ratio", "mean"),
            drop_ratio=("drop_ratio", "mean"),
            avg_delay=("avg_delay", "mean"),
            avg_remaining_energy=("avg_remaining_energy", "mean"),
            constraint_violations=("constraint_violations", "mean"),
            total_hops=("total_hops", "mean"),
        )
        .reset_index()
    )

    summary_path = os.path.join(result_dir, "baseline_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("Summary:")
    print(summary)

    fig_dir = os.path.join(result_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plot_bar(
        summary,
        "delivery_ratio",
        "Delivery Ratio",
        os.path.join(fig_dir, "delivery_ratio.png"),
    )

    plot_bar(
        summary,
        "drop_ratio",
        "Drop Ratio",
        os.path.join(fig_dir, "drop_ratio.png"),
    )

    plot_bar(
        summary,
        "avg_delay",
        "Average Delay",
        os.path.join(fig_dir, "avg_delay.png"),
    )

    plot_bar(
        summary,
        "avg_remaining_energy",
        "Average Remaining Energy",
        os.path.join(fig_dir, "avg_remaining_energy.png"),
    )

    plot_bar(
        summary,
        "constraint_violations",
        "Constraint Violations",
        os.path.join(fig_dir, "constraint_violations.png"),
    )

    plot_bar(
        summary,
        "total_hops",
        "Total Hops",
        os.path.join(fig_dir, "total_hops.png"),
    )

    print("\nSaved summary to:")
    print(summary_path)

    print("\nSaved figures to:")
    print(fig_dir)


if __name__ == "__main__":
    main()
