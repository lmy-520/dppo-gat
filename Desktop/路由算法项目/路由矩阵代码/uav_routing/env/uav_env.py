"""
uav_env.py

轻量版无人机动态网络仿真环境。

功能：
1. 无人机移动
2. 动态拓扑生成
3. 链路容量 / 时延 / 丢包率
4. 业务流生成
5. 队列更新
6. 简单能耗模型
7. 四个基础路由策略：
   - random
   - shortest
   - min_delay
   - energy

运行：
    python uav_env.py
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class UAVConfig:
    n_nodes: int = 20
    area_size: float = 1000.0

    altitude_min: float = 80.0
    altitude_max: float = 120.0

    comm_range: float = 280.0
    max_speed: float = 15.0
    dt: float = 1.0

    episode_steps: int = 200

    init_energy: float = 1000.0
    min_energy: float = 1.0

    queue_capacity: int = 80
    traffic_prob: float = 0.25
    max_new_packets: int = 3

    base_delay: float = 1.0

    tx_energy_per_dist: float = 0.001
    rx_energy_per_packet: float = 0.02
    move_energy_factor: float = 0.002

    seed: Optional[int] = 42


class UAVNetworkEnv:
    def __init__(self, cfg: UAVConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.reset()

    def reset(self) -> Dict[str, np.ndarray]:
        c = self.cfg
        self.t = 0

        xy = self.rng.uniform(0, c.area_size, size=(c.n_nodes, 2))
        z = self.rng.uniform(c.altitude_min, c.altitude_max, size=(c.n_nodes, 1))
        self.pos = np.concatenate([xy, z], axis=1)

        direction = self.rng.normal(size=(c.n_nodes, 3))
        direction = direction / (
            np.linalg.norm(direction, axis=1, keepdims=True) + 1e-8
        )

        speed = self.rng.uniform(
            0.2 * c.max_speed,
            c.max_speed,
            size=(c.n_nodes, 1),
        )

        self.vel = direction * speed

        self.energy = np.full(c.n_nodes, c.init_energy, dtype=np.float64)
        self.queues: List[List[Dict]] = [[] for _ in range(c.n_nodes)]

        self.total_generated = 0
        self.total_delivered = 0
        self.total_dropped = 0
        self.total_delay = 0.0
        self.constraint_violations = 0
        self.total_hops = 0

        self._update_graph()
        return self.get_state()

    def _move_nodes(self):
        c = self.cfg
        self.pos += self.vel * c.dt

        for i in range(c.n_nodes):
            for d in [0, 1]:
                if self.pos[i, d] < 0:
                    self.pos[i, d] = 0
                    self.vel[i, d] *= -1
                elif self.pos[i, d] > c.area_size:
                    self.pos[i, d] = c.area_size
                    self.vel[i, d] *= -1

            if self.pos[i, 2] < c.altitude_min:
                self.pos[i, 2] = c.altitude_min
                self.vel[i, 2] *= -1
            elif self.pos[i, 2] > c.altitude_max:
                self.pos[i, 2] = c.altitude_max
                self.vel[i, 2] *= -1

        speed = np.linalg.norm(self.vel, axis=1)
        self.energy -= c.move_energy_factor * speed * c.dt
        self.energy = np.maximum(self.energy, 0.0)

    def _link_features(self, i: int, j: int) -> Tuple[float, float, float, float, float]:
        c = self.cfg

        dist = float(np.linalg.norm(self.pos[i] - self.pos[j]))
        dist_ratio = min(dist / c.comm_range, 1.0)

        capacity = max(1.0, 12.0 * (1.0 - dist_ratio) + 1.0)
        delay = c.base_delay + 8.0 * dist_ratio
        loss_rate = min(0.5, 0.03 + 0.45 * (dist_ratio ** 2))
        snr = max(1.0, 30.0 * (1.0 - dist_ratio) + 1.0)

        return capacity, delay, loss_rate, dist, snr

    def _update_graph(self):
        c = self.cfg

        edges = []
        attrs = []
        adj = np.zeros((c.n_nodes, c.n_nodes), dtype=np.int64)

        for i in range(c.n_nodes):
            for j in range(c.n_nodes):
                if i == j:
                    continue

                dist = float(np.linalg.norm(self.pos[i] - self.pos[j]))

                if (
                    dist <= c.comm_range
                    and self.energy[i] > c.min_energy
                    and self.energy[j] > c.min_energy
                ):
                    cap, delay, loss, distance, snr = self._link_features(i, j)

                    edges.append((i, j))
                    attrs.append(
                        [
                            cap / 13.0,
                            delay / 9.0,
                            loss,
                            distance / c.comm_range,
                            snr / 31.0,
                        ]
                    )

                    adj[i, j] = 1

        if len(edges) > 0:
            self.edge_index = np.array(edges, dtype=np.int64).T
            self.edge_attr = np.array(attrs, dtype=np.float64)
        else:
            self.edge_index = np.zeros((2, 0), dtype=np.int64)
            self.edge_attr = np.zeros((0, 5), dtype=np.float64)

        self.adj = adj

    def _generate_traffic(self):
        c = self.cfg

        for src in range(c.n_nodes):
            if self.rng.random() < c.traffic_prob:
                n_packets = int(self.rng.integers(1, c.max_new_packets + 1))

                for _ in range(n_packets):
                    dst = int(self.rng.integers(0, c.n_nodes - 1))

                    if dst >= src:
                        dst += 1

                    packet = {
                        "src": src,
                        "dst": dst,
                        "birth_time": self.t,
                        "ttl": c.n_nodes + 5,
                    }

                    if len(self.queues[src]) < c.queue_capacity:
                        self.queues[src].append(packet)
                        self.total_generated += 1
                    else:
                        self.total_dropped += 1

    def get_state(self) -> Dict[str, np.ndarray]:
        c = self.cfg

        queue_len = np.array([len(q) for q in self.queues], dtype=np.float64)
        traffic_load = queue_len / max(c.queue_capacity, 1)

        node_features = np.column_stack(
            [
                queue_len / max(c.queue_capacity, 1),
                self.energy / max(c.init_energy, 1e-8),
                self.pos[:, 0] / c.area_size,
                self.pos[:, 1] / c.area_size,
                (self.pos[:, 2] - c.altitude_min)
                / max(c.altitude_max - c.altitude_min, 1e-8),
                self.vel[:, 0] / c.max_speed,
                self.vel[:, 1] / c.max_speed,
                self.vel[:, 2] / c.max_speed,
                traffic_load,
            ]
        )

        return {
            "node_features": node_features.astype(np.float32),
            "edge_index": self.edge_index.astype(np.int64),
            "edge_attr": self.edge_attr.astype(np.float32),
            "adj": self.adj.astype(np.int64),
        }

    def neighbors(self, node: int) -> np.ndarray:
        return np.where(self.adj[node] == 1)[0]

    def _choose_next_hop(self, src: int, dst: int, policy: str) -> Optional[int]:
        """
        调用 routing/baselines.py 中的路由策略。
        """
        try:
            from routing.baselines import choose_next_hop
        except ModuleNotFoundError:
            import os
            import sys

            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..")
            )
            sys.path.insert(0, project_root)

            from routing.baselines import choose_next_hop

        return choose_next_hop(self, src, dst, policy)



    def step(self, policy: str = "random", actions=None):
        """
        执行一个仿真步。

        时间顺序：
        1. 使用当前拓扑和当前队列执行路由转发
        2. 更新队列、能量、丢包、时延等指标
        3. 无人机移动
        4. 更新下一时刻拓扑
        5. 生成新业务，作为下一时刻状态的一部分

        这样可以保证：
        - state_t 里的 adj
        - 模型基于 adj_t 产生的 actions
        - env.step(actions=actions)

        三者使用的是同一个拓扑。
        """
        c = self.cfg

        new_queues: List[List[Dict]] = [[] for _ in range(c.n_nodes)]
        link_load = {}

        for i in range(c.n_nodes):
            for packet in self.queues[i]:
                dst = packet["dst"]

                if i == dst:
                    delay = self.t - packet["birth_time"]
                    self.total_delivered += 1
                    self.total_delay += delay
                    continue

                if actions is not None:
                    next_hop = int(actions[i])

                    if next_hop < 0:
                        next_hop = None
                else:
                    next_hop = self._choose_next_hop(i, dst, policy)

                if next_hop is None or self.adj[i, next_hop] == 0:
                    self.total_dropped += 1
                    self.constraint_violations += 1
                    continue

                cap, link_delay, loss_rate, dist, _ = self._link_features(i, next_hop)

                key = (i, next_hop)
                link_load[key] = link_load.get(key, 0) + 1

                if link_load[key] > cap:
                    self.total_dropped += 1
                    self.constraint_violations += 1
                    continue

                if self.rng.random() < loss_rate:
                    self.total_dropped += 1
                    continue

                tx_cost = c.tx_energy_per_dist * dist + 0.01
                rx_cost = c.rx_energy_per_packet

                self.energy[i] = max(0.0, self.energy[i] - tx_cost)
                self.energy[next_hop] = max(0.0, self.energy[next_hop] - rx_cost)

                packet["ttl"] -= 1
                self.total_hops += 1

                if packet["ttl"] <= 0:
                    self.total_dropped += 1
                    self.constraint_violations += 1
                    continue

                if next_hop == dst:
                    delay = (self.t - packet["birth_time"]) + link_delay
                    self.total_delivered += 1
                    self.total_delay += delay
                else:
                    if len(new_queues[next_hop]) < c.queue_capacity:
                        new_queues[next_hop].append(packet)
                    else:
                        self.total_dropped += 1
                        self.constraint_violations += 1

        self.queues = new_queues

        self.t += 1

        self._move_nodes()
        self._update_graph()
        self._generate_traffic()

        return self.get_state(), self.get_metrics()


    def get_metrics(self) -> Dict[str, float]:
        generated = max(self.total_generated, 1)
        attempted = max(self.total_delivered + self.total_dropped, 1)

        return {
            "step": float(self.t),
            "generated": float(self.total_generated),
            "delivered": float(self.total_delivered),
            "dropped": float(self.total_dropped),
            "delivery_ratio": self.total_delivered / generated,
            "drop_ratio": self.total_dropped / attempted,
            "avg_delay": self.total_delay / max(self.total_delivered, 1),
            "avg_remaining_energy": float(np.mean(self.energy)),
            "min_remaining_energy": float(np.min(self.energy)),
            "constraint_violations": float(self.constraint_violations),
            "num_edges": float(self.edge_index.shape[1]),
            "avg_queue_len": float(np.mean([len(q) for q in self.queues])),
            "total_hops": float(self.total_hops),
        }

    def run_episode(self, policy: str = "random") -> Dict[str, float]:
        self.reset()

        for _ in range(self.cfg.episode_steps):
            self.step(policy=policy)

        return self.get_metrics()


def smoke_test():
    cfg = UAVConfig(
        n_nodes=20,
        episode_steps=200,
        seed=42,
    )

    env = UAVNetworkEnv(cfg)
    state = env.reset()

    print("Initial state:")
    print("  node_features:", state["node_features"].shape)
    print("  edge_index:", state["edge_index"].shape)
    print("  edge_attr:", state["edge_attr"].shape)
    print("  adj:", state["adj"].shape)

    policies = ["random", "shortest", "min_delay", "energy"]

    print("\nEpisode results:")

    for policy in policies:
        env = UAVNetworkEnv(cfg)
        metrics = env.run_episode(policy=policy)

        print(f"\nPolicy: {policy}")

        for k, v in metrics.items():
            if k in {
                "generated",
                "delivered",
                "dropped",
                "constraint_violations",
                "num_edges",
                "total_hops",
            }:
                print(f"  {k:24s}: {int(v)}")
            else:
                print(f"  {k:24s}: {v:.4f}")


if __name__ == "__main__":
    smoke_test()
