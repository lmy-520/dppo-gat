"""
routing/baselines.py

基础路由策略。

目前包含：
1. random
2. shortest
3. min_delay
4. energy

这些策略会被 env/uav_env.py 调用。
"""

from typing import Optional

import numpy as np


def get_progress_candidates(env, src: int, dst: int) -> np.ndarray:
    """
    优先选择比当前节点更接近目的节点的邻居，避免绕圈。
    如果没有这样的邻居，则退化为所有邻居。
    """
    nbrs = env.neighbors(src)

    if len(nbrs) == 0:
        return nbrs

    current_dist = np.linalg.norm(env.pos[src] - env.pos[dst])

    progress_nbrs = []

    for j in nbrs:
        j = int(j)
        j_dist = np.linalg.norm(env.pos[j] - env.pos[dst])

        if j_dist < current_dist:
            progress_nbrs.append(j)

    if len(progress_nbrs) > 0:
        return np.array(progress_nbrs, dtype=np.int64)

    return nbrs


def random_policy(env, src: int, dst: int) -> Optional[int]:
    candidates = get_progress_candidates(env, src, dst)

    if len(candidates) == 0:
        return None

    return int(env.rng.choice(candidates))


def shortest_policy(env, src: int, dst: int) -> Optional[int]:
    candidates = get_progress_candidates(env, src, dst)

    if len(candidates) == 0:
        return None

    dst_pos = env.pos[dst]
    scores = []

    for j in candidates:
        j = int(j)
        score = np.linalg.norm(env.pos[j] - dst_pos)
        scores.append(score)

    return int(candidates[int(np.argmin(scores))])


def min_delay_policy(env, src: int, dst: int) -> Optional[int]:
    candidates = get_progress_candidates(env, src, dst)

    if len(candidates) == 0:
        return None

    scores = []

    for j in candidates:
        j = int(j)
        _, delay, _, _, _ = env._link_features(src, j)
        scores.append(delay)

    return int(candidates[int(np.argmin(scores))])


def energy_policy(env, src: int, dst: int) -> Optional[int]:
    candidates = get_progress_candidates(env, src, dst)

    if len(candidates) == 0:
        return None

    scores = []

    for j in candidates:
        j = int(j)

        cap, _, loss, dist, _ = env._link_features(src, j)

        distance_to_dst = np.linalg.norm(env.pos[j] - env.pos[dst])
        progress_score = 1.0 - min(distance_to_dst / env.cfg.area_size, 1.0)

        score = (
            0.35 * (env.energy[j] / env.cfg.init_energy)
            + 0.25 * (cap / 13.0)
            + 0.25 * progress_score
            - 0.10 * loss
            - 0.05 * (dist / env.cfg.comm_range)
        )

        scores.append(score)

    return int(candidates[int(np.argmax(scores))])


def choose_next_hop(env, src: int, dst: int, policy: str) -> Optional[int]:
    """
    统一路由策略接口。

    env:
        UAVNetworkEnv 对象

    src:
        当前节点

    dst:
        目的节点

    policy:
        random / shortest / min_delay / energy
    """
    if policy == "random":
        return random_policy(env, src, dst)

    if policy == "shortest":
        return shortest_policy(env, src, dst)

    if policy == "min_delay":
        return min_delay_policy(env, src, dst)

    if policy == "energy":
        return energy_policy(env, src, dst)

    raise ValueError(f"Unknown policy: {policy}")
