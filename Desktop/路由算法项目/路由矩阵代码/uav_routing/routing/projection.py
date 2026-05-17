"""
routing/projection.py

动作投影与合法性约束。

作用：
1. 把模型输出的 logits 矩阵转成合法下一跳动作
2. 防止选择不存在的链路
3. 防止选择自己
4. 给后续 GAT-RL / Diffusion Policy 共用
"""

from typing import Optional

import numpy as np


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    对 logits 做 masked softmax。

    logits:
        shape = [N, N]

    mask:
        shape = [N, N]
        mask[i, j] = 1 表示 i 可以选择 j
        mask[i, j] = 0 表示不可选
    """
    masked_logits = logits.copy()
    masked_logits[mask == 0] = -1e9

    max_logits = np.max(masked_logits, axis=1, keepdims=True)
    exp_logits = np.exp(masked_logits - max_logits)
    exp_logits = exp_logits * mask

    denom = np.sum(exp_logits, axis=1, keepdims=True) + 1e-8
    probs = exp_logits / denom

    return probs


def logits_to_actions(logits: np.ndarray, adj: np.ndarray) -> np.ndarray:
    """
    把模型输出的 [N, N] logits 转成每个节点的下一跳动作。

    返回：
        actions[i] = j 表示节点 i 选择 j 作为下一跳
        actions[i] = -1 表示节点 i 没有合法下一跳
    """
    n_nodes = adj.shape[0]

    mask = adj.copy().astype(np.int64)

    for i in range(n_nodes):
        mask[i, i] = 0

    probs = masked_softmax(logits, mask)

    actions = np.full(n_nodes, -1, dtype=np.int64)

    for i in range(n_nodes):
        valid_neighbors = np.where(mask[i] == 1)[0]

        if len(valid_neighbors) == 0:
            actions[i] = -1
        else:
            actions[i] = int(np.argmax(probs[i]))

    return actions


def random_valid_actions(adj: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    生成一个随机合法动作，用于调试模型动作接口。
    """
    if rng is None:
        rng = np.random.default_rng()

    n_nodes = adj.shape[0]
    actions = np.full(n_nodes, -1, dtype=np.int64)

    for i in range(n_nodes):
        valid_neighbors = np.where(adj[i] == 1)[0]

        if len(valid_neighbors) == 0:
            actions[i] = -1
        else:
            actions[i] = int(rng.choice(valid_neighbors))

    return actions


def is_valid_action(adj: np.ndarray, src: int, next_hop: int) -> bool:
    """
    判断动作是否合法。
    """
    if next_hop < 0:
        return False

    if src == next_hop:
        return False

    if adj[src, next_hop] == 0:
        return False

    return True
