"""
train/train_gat_rl.py

最小版 PPO 训练脚本。

目标：
1. 使用 EdgeGATActorCritic 作为 Actor-Critic
2. 从环境中采样 rollout
3. 使用 PPO 更新模型
4. 保存训练日志和模型权重

运行方式：
    在项目根目录 uav_routing 下执行：
    python -m train.train_gat_rl
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from env.uav_env import UAVConfig, UAVNetworkEnv
from models.actor_critic import EdgeGATActorCritic
from routing.action_sampler import (
    sample_actions_from_logits,
    evaluate_actions_from_logits,
)


@dataclass
class PPOConfig:
    seed: int = 42

    total_updates: int = 30
    rollout_steps: int = 64
    ppo_epochs: int = 4

    gamma: float = 0.99
    gae_lambda: float = 0.95

    clip_eps: float = 0.2
    lr: float = 3e-4

    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    hidden_dim: int = 64
    num_layers: int = 2


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def state_to_tensors(state: Dict[str, np.ndarray]):
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

    return node_features, edge_index, edge_attr


def copy_state(state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "node_features": state["node_features"].copy(),
        "edge_index": state["edge_index"].copy(),
        "edge_attr": state["edge_attr"].copy(),
        "adj": state["adj"].copy(),
    }


def compute_reward(prev_metrics: Dict[str, float], cur_metrics: Dict[str, float]) -> float:
    """
    用累计指标差分构造单步 reward。

    当前只是最小可训练版本，后面可以继续调权重。
    """
    delta_delivered = cur_metrics["delivered"] - prev_metrics["delivered"]
    delta_dropped = cur_metrics["dropped"] - prev_metrics["dropped"]
    delta_violations = (
        cur_metrics["constraint_violations"]
        - prev_metrics["constraint_violations"]
    )
    delta_hops = cur_metrics["total_hops"] - prev_metrics["total_hops"]

    energy_cost = (
        prev_metrics["avg_remaining_energy"]
        - cur_metrics["avg_remaining_energy"]
    )

    reward = (
        2.0 * delta_delivered
        - 0.4 * delta_dropped
        - 0.8 * delta_violations
        - 0.005 * delta_hops
        - 0.01 * cur_metrics["avg_queue_len"]
        - 0.02 * energy_cost
    )

    return float(reward)


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    last_value: float,
    gamma: float,
    gae_lambda: float,
):
    advantages = []
    gae = 0.0
    next_value = last_value

    for t in reversed(range(len(rewards))):
        if dones[t]:
            next_non_terminal = 0.0
            next_value = 0.0
        else:
            next_non_terminal = 1.0

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae

        advantages.insert(0, gae)
        next_value = values[t]

    returns = [adv + val for adv, val in zip(advantages, values)]

    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages


def collect_rollout(model, env, state, cfg: PPOConfig):
    rollout = []

    rewards = []
    values = []
    dones = []

    for _ in range(cfg.rollout_steps):
        saved_state = copy_state(state)

        node_features, edge_index, edge_attr = state_to_tensors(state)

        with torch.no_grad():
            logits, value = model(
                node_features=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )

            actions, log_prob, entropy = sample_actions_from_logits(
                logits=logits,
                adj=state["adj"],
                deterministic=False,
            )

        prev_metrics = env.get_metrics()
        next_state, cur_metrics = env.step(actions=actions)

        reward = compute_reward(prev_metrics, cur_metrics)

        done = env.t >= env.cfg.episode_steps

        rollout.append(
            {
                "state": saved_state,
                "actions": actions.copy(),
                "old_log_prob": log_prob.detach(),
                "value": float(value.item()),
                "reward": reward,
                "done": done,
            }
        )

        rewards.append(reward)
        values.append(float(value.item()))
        dones.append(done)

        state = next_state

        if done:
            state = env.reset()

    with torch.no_grad():
        if dones[-1]:
            last_value = 0.0
        else:
            node_features, edge_index, edge_attr = state_to_tensors(state)
            _, value = model(
                node_features=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
            last_value = float(value.item())

    returns, advantages = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        last_value=last_value,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
    )

    for i in range(len(rollout)):
        rollout[i]["return"] = returns[i]
        rollout[i]["advantage"] = advantages[i]

    return rollout, state


def ppo_update(model, optimizer, rollout, cfg: PPOConfig):
    total_actor_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_loss_value = 0.0

    n_items = len(rollout)

    for _ in range(cfg.ppo_epochs):
        np.random.shuffle(rollout)

        for item in rollout:
            state = item["state"]
            actions = item["actions"]
            old_log_prob = item["old_log_prob"]
            target_return = item["return"]
            advantage = item["advantage"]

            node_features, edge_index, edge_attr = state_to_tensors(state)

            logits, value = model(
                node_features=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )

            new_log_prob, entropy = evaluate_actions_from_logits(
                logits=logits,
                adj=state["adj"],
                actions=actions,
            )

            ratio = torch.exp(new_log_prob - old_log_prob)

            unclipped = ratio * advantage
            clipped = torch.clamp(
                ratio,
                1.0 - cfg.clip_eps,
                1.0 + cfg.clip_eps,
            ) * advantage

            actor_loss = -torch.min(unclipped, clipped)

            value_loss = F.mse_loss(
                value.squeeze(),
                target_return,
            )

            loss = (
                actor_loss
                + cfg.value_coef * value_loss
                - cfg.entropy_coef * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.max_grad_norm,
            )
            optimizer.step()

            total_actor_loss += float(actor_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy += float(entropy.item())
            total_loss_value += float(loss.item())

    denom = max(n_items * cfg.ppo_epochs, 1)

    return {
        "actor_loss": total_actor_loss / denom,
        "value_loss": total_value_loss / denom,
        "entropy": total_entropy / denom,
        "total_loss": total_loss_value / denom,
    }


def evaluate_policy(model, seed: int = 123, episodes: int = 3):
    """
    用 deterministic=True 简单评估当前策略。
    """
    results = []

    for ep in range(episodes):
        cfg = UAVConfig(
            n_nodes=20,
            episode_steps=200,
            seed=seed + ep,
        )

        env = UAVNetworkEnv(cfg)
        state = env.reset()

        for _ in range(cfg.episode_steps):
            node_features, edge_index, edge_attr = state_to_tensors(state)

            with torch.no_grad():
                logits, _ = model(
                    node_features=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                )

                actions, _, _ = sample_actions_from_logits(
                    logits=logits,
                    adj=state["adj"],
                    deterministic=True,
                )

            state, metrics = env.step(actions=actions)

        results.append(metrics)

    delivery_ratio = np.mean([m["delivery_ratio"] for m in results])
    drop_ratio = np.mean([m["drop_ratio"] for m in results])
    avg_delay = np.mean([m["avg_delay"] for m in results])
    violations = np.mean([m["constraint_violations"] for m in results])

    return {
        "eval_delivery_ratio": delivery_ratio,
        "eval_drop_ratio": drop_ratio,
        "eval_avg_delay": avg_delay,
        "eval_constraint_violations": violations,
    }


def main():
    cfg = PPOConfig()
    set_seed(cfg.seed)

    env_cfg = UAVConfig(
        n_nodes=20,
        episode_steps=200,
        seed=cfg.seed,
    )

    env = UAVNetworkEnv(env_cfg)
    state = env.reset()

    model = EdgeGATActorCritic(
        node_dim=9,
        edge_dim=5,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
    )

    log_rows = []

    print("Start PPO training...")

    for update in range(1, cfg.total_updates + 1):
        model.train()

        rollout, state = collect_rollout(
            model=model,
            env=env,
            state=state,
            cfg=cfg,
        )

        update_info = ppo_update(
            model=model,
            optimizer=optimizer,
            rollout=rollout,
            cfg=cfg,
        )

        avg_reward = np.mean([item["reward"] for item in rollout])
        sum_reward = np.sum([item["reward"] for item in rollout])

        model.eval()
        eval_info = evaluate_policy(
            model=model,
            seed=1000 + update * 10,
            episodes=3,
        )

        row = {
            "update": update,
            "avg_reward": avg_reward,
            "sum_reward": sum_reward,
        }
        row.update(update_info)
        row.update(eval_info)
        log_rows.append(row)

        print(
            f"update={update:03d}, "
            f"avg_reward={avg_reward:.4f}, "
            f"actor_loss={update_info['actor_loss']:.4f}, "
            f"value_loss={update_info['value_loss']:.4f}, "
            f"entropy={update_info['entropy']:.4f}, "
            f"eval_delivery={eval_info['eval_delivery_ratio']:.4f}, "
            f"eval_drop={eval_info['eval_drop_ratio']:.4f}"
        )

    result_dir = os.path.join(PROJECT_ROOT, "data", "results")
    os.makedirs(result_dir, exist_ok=True)

    log_path = os.path.join(result_dir, "gat_rl_training_log.csv")
    pd.DataFrame(log_rows).to_csv(
        log_path,
        index=False,
        encoding="utf-8-sig",
    )

    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "gat_rl")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, "edge_gat_actor_critic.pt")
    torch.save(model.state_dict(), ckpt_path)

    print("\nTraining finished.")
    print("Saved log to:")
    print(log_path)
    print("Saved checkpoint to:")
    print(ckpt_path)


if __name__ == "__main__":
    main()
