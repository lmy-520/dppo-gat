# 无人机动态路由算法项目开发日志

## 0. 项目概述

本项目目标是从零搭建一个面向无人机自组网的动态路由算法实验平台，最终支持：

```text
动态图仿真环境
+ 传统 baseline 路由策略
+ 边感知 GAT 路由模型
+ 强化学习训练
+ 拓扑约束投影
+ 后续扩散模型 / Graph-MoE 扩展
```

当前开发原则是：

```text
先跑通最小闭环
再逐步加入 GAT、RL、Diffusion、MoE 等复杂模块
```

---

## 1. 总体路线规划

项目最初被划分为以下几个阶段：

```text
V0：动态无人机网络仿真环境
V1：传统路由 baseline
V2：GAT-RL 路由模型
V3：扩散路由生成模型
V4：Graph-MoE 专家机制
V5：自适应少步数扩散
```

截至目前，实际进度为：

```text
V0 已完成
V1 已完成基础拆分
V2 前置动作接口已完成
下一步进入 EdgeGAT 模型前向测试
```

---

## 2. 项目目录结构设计

最初设计的完整目录结构如下：

```text
uav_routing/
│
├── README.md
├── requirements.txt
├── main.py
│
├── configs/
│   ├── default.yaml
│   ├── baseline.yaml
│   ├── gat_rl.yaml
│   └── diffusion_moe.yaml
│
├── env/
│   ├── __init__.py
│   └── uav_env.py
│
├── routing/
│   ├── __init__.py
│   ├── baselines.py
│   ├── projection.py
│   └── constraints.py
│
├── models/
│   ├── __init__.py
│   ├── edge_gat.py
│   ├── actor_critic.py
│   ├── denoiser.py
│   ├── graph_moe.py
│   └── diffusion_policy.py
│
├── train/
│   ├── __init__.py
│   ├── train_baseline.py
│   ├── train_gat_rl.py
│   ├── pretrain_diffusion.py
│   └── finetune_diffusion_rl.py
│
├── eval/
│   ├── __init__.py
│   ├── run_baselines.py
│   ├── evaluate.py
│   ├── ablation.py
│   └── plot_results.py
│
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   ├── replay_buffer.py
│   ├── graph_utils.py
│   └── seed.py
│
├── data/
│   ├── expert/
│   ├── logs/
│   └── results/
│
├── checkpoints/
│   ├── gat_rl/
│   ├── diffusion/
│   └── moe/
│
└── scripts/
    ├── run_v0_baselines.sh
    ├── train_gat_rl.sh
    └── run_ablation.sh
```

为了避免一开始工程过重，实际先采用了最小可运行结构：

```text
uav_routing/
│
├── requirements.txt
│
├── env/
│   ├── __init__.py
│   └── uav_env.py
│
├── routing/
│   ├── __init__.py
│   ├── baselines.py
│   └── projection.py
│
├── eval/
│   ├── __init__.py
│   ├── run_baselines.py
│   └── plot_results.py
│
├── debug/
│   └── test_action_interface.py
│
└── data/
    └── results/
```

---

## 3. V0：轻量版无人机动态网络仿真环境

首先实现了：

```text
env/uav_env.py
```

该环境支持以下功能：

```text
1. 无人机节点初始化
2. 无人机位置和速度更新
3. 动态拓扑生成
4. 链路容量、时延、丢包率、距离、SNR 特征计算
5. 业务流生成
6. 队列更新
7. 简单能耗模型
8. 路由转发过程
9. 指标统计
```

环境输出状态如下：

```python
state = {
    "node_features": node_features,
    "edge_index": edge_index,
    "edge_attr": edge_attr,
    "adj": adj,
}
```

各字段含义：

```text
node_features: [N, 9]
edge_index: [2, E]
edge_attr: [E, 5]
adj: [N, N]
```

节点特征包括：

```text
queue_len
energy
pos_x
pos_y
pos_z
vel_x
vel_y
vel_z
traffic_load
```

边特征包括：

```text
capacity
delay
loss_rate
distance
snr
```

---

## 4. 第一次运行与代码修复

第一次运行时出现过代码粘贴错误：

```text
SyntaxError: invalid syntax
```

问题代码中，`get_metrics()` 部分被错误粘贴，出现了类似：

```python
"dropped": float(self.total        self.queues = new_queues
```

后续修复为：

```python
"dropped": float(self.total_dropped),
```

同时还修复了一个缩进问题：

```python
def get_metrics(self)
```

该函数一开始被错误缩进到了 `step()` 函数内部，后来调整为与 `step()`、`run_episode()` 同级。

修复后，`env/uav_env.py` 成功运行，初始输出为：

```text
Initial state:
  node_features: (20, 9)
  edge_index: (2, 72)
  edge_attr: (72, 5)
  adj: (20, 20)
```

这说明动态图状态已经正常生成。

---

## 5. 四种基础 baseline 策略

最初在 `env/uav_env.py` 内部实现了四种传统路由策略：

```text
random
shortest
min_delay
energy
```

第一次实验中，`min_delay` 和 `energy` 的投递率很低、跳数很高。

典型现象为：

```text
min_delay delivery_ratio 约 0.0606
energy delivery_ratio 约 0.0551
total_hops 超过 13000
```

分析后认为原因是：

```text
min_delay 只看当前链路时延
energy 只看能量和链路质量
二者都没有判断下一跳是否更接近目的节点
```

因此加入了 progress 约束：

```text
优先选择比当前节点更接近目的节点的邻居
如果没有这样的邻居，再退化为所有邻居
```

修改后，结果明显改善。

代表性结果如下：

```text
Policy: random
  delivery_ratio          : 0.2361
  drop_ratio              : 0.7615
  avg_delay               : 6.7410
  constraint_violations   : 294
  total_hops              : 4046

Policy: shortest
  delivery_ratio          : 0.2635
  drop_ratio              : 0.7332
  avg_delay               : 7.2527
  constraint_violations   : 412
  total_hops              : 4354

Policy: min_delay
  delivery_ratio          : 0.2588
  drop_ratio              : 0.7377
  avg_delay               : 7.1438
  constraint_violations   : 404
  total_hops              : 6860

Policy: energy
  delivery_ratio          : 0.2689
  drop_ratio              : 0.7287
  avg_delay               : 7.1353
  constraint_violations   : 312
  total_hops              : 6496
```

阶段结论：

```text
加入 progress 约束后，MinDelay 和 Energy-aware 的绕行问题明显缓解。
当前 Energy-aware baseline 表现最好。
```

---

## 6. baseline 批量实验脚本

随后新增：

```text
eval/run_baselines.py
```

该脚本用于批量运行四种 baseline：

```text
random
shortest
min_delay
energy
```

运行方式：

```powershell
python -m eval.run_baselines
```

结果保存到：

```text
data/results/baseline_results.csv
```

CSV 中每一行表示：

```text
某个 seed 下，某个 policy 跑完一个 episode 后的指标
```

例如：

```text
seed,policy,step,generated,delivered,dropped,delivery_ratio,...
0,random,...
0,shortest,...
0,min_delay,...
0,energy,...
```

多 seed 初步结果显示：

```text
random    平均投递率约 0.277
shortest  平均投递率约 0.287
min_delay 平均投递率约 0.300
energy    平均投递率约 0.316
```

当前结论：

```text
Energy-aware baseline 在当前 V0 环境下整体表现最好。
```

---

## 7. 实验结果画图脚本

随后新增：

```text
eval/plot_results.py
```

该脚本读取：

```text
data/results/baseline_results.csv
```

并生成：

```text
data/results/baseline_summary.csv
data/results/figures/delivery_ratio.png
data/results/figures/drop_ratio.png
data/results/figures/avg_delay.png
data/results/figures/avg_remaining_energy.png
data/results/figures/constraint_violations.png
data/results/figures/total_hops.png
```

运行方式：

```powershell
python -m eval.plot_results
```

完成后，项目已经具备：

```text
1. 仿真环境
2. baseline 策略
3. 多 seed 实验
4. CSV 结果保存
5. 图表生成
```

---

## 8. V1：拆分 baseline 策略

之后将原先写在 `env/uav_env.py` 中的 baseline 策略拆分到：

```text
routing/baselines.py
```

拆分后的职责如下：

### `env/uav_env.py`

负责：

```text
仿真环境
状态更新
队列
能耗
链路
指标统计
```

### `routing/baselines.py`

负责：

```text
random_policy
shortest_policy
min_delay_policy
energy_policy
choose_next_hop
```

`env/uav_env.py` 中只保留统一调用接口：

```python
def _choose_next_hop(self, src: int, dst: int, policy: str) -> Optional[int]:
    from routing.baselines import choose_next_hop
    return choose_next_hop(self, src, dst, policy)
```

这一步的意义是：

```text
环境和路由策略解耦
后续加入 GAT、RL、Diffusion 时不用继续污染环境代码
```

---

## 9. 外部动作接口

进入 V2 前置工作后，开始让环境支持外部模型动作。

原先环境只能这样调用：

```python
env.step(policy="energy")
```

但后续 GAT-RL 模型会输出动作：

```python
actions[i] = j
```

表示：

```text
第 i 个无人机选择第 j 个无人机作为下一跳
```

因此修改了 `env.step()` 接口：

```python
def step(self, policy: str = "random", actions=None):
```

现在环境同时支持：

```python
env.step(policy="energy")
```

和：

```python
env.step(actions=actions)
```

这为后续模型接入打通了接口。

---

## 10. 动作投影模块

新增：

```text
routing/projection.py
```

主要函数包括：

```python
masked_softmax(logits, mask)
logits_to_actions(logits, adj)
random_valid_actions(adj, rng)
is_valid_action(adj, src, next_hop)
```

其中最重要的是：

```python
logits_to_actions(logits, adj)
```

作用是将模型输出的：

```text
logits: [N, N]
```

转成合法动作：

```text
actions: [N]
```

含义为：

```text
actions[i] = j 表示节点 i 选择节点 j 作为下一跳
actions[i] = -1 表示节点 i 没有合法下一跳
```

该模块后续会被 GAT-RL、Diffusion Policy 共用。

---

## 11. 动作接口测试

新增测试文件：

```text
debug/test_action_interface.py
```

测试内容包括：

```text
1. random_valid_actions 是否可以生成动作
2. logits_to_actions 是否可以把 logits 转成动作
3. env.step(actions=actions) 是否可以正常执行
```

运行输出为：

```text
Test 1: random valid actions
step=1, delivered=0, dropped=0, violations=0
...
Random valid action interface OK.

Test 2: logits to actions
step=1, delivered=1, dropped=2, violations=1
...
Logits action interface OK.
```

说明：

```text
外部动作接口已经打通
模型输出动作后，环境可以执行
```

---

## 12. 修正 `step()` 时间顺序

在测试动作接口时发现一个细节：

```text
actions 是基于当前 state["adj"] 生成的
但 env.step() 开头会先移动节点、更新拓扑
导致动作基于旧拓扑，执行时却使用新拓扑
```

这会引入额外的 constraint violation。

因此调整了 `step()` 的时间顺序。

原逻辑类似：

```text
移动节点
更新拓扑
生成业务
执行动作
返回状态
```

修改后为更适合强化学习的逻辑：

```text
当前 state
↓
策略 / 模型选择 action
↓
在当前拓扑上执行转发
↓
更新队列、能耗、丢包、时延
↓
无人机移动
↓
更新下一时刻拓扑
↓
生成新业务
↓
返回 next_state
```

修改意义：

```text
state_t、actions_t、env.step(actions_t) 使用的是同一时刻的拓扑
```

这为后续 GAT-RL 训练打下基础。

---

## 13. 当前项目状态总结

截至目前，已经完成：

```text
1. 项目目录结构搭建
2. V0 无人机动态网络仿真环境
3. 四种传统 baseline 策略
4. baseline 绕圈问题修正
5. baseline 多 seed 批量实验
6. baseline 结果保存为 CSV
7. baseline 结果画图脚本
8. baseline 策略从环境中拆分
9. 动作投影模块 projection.py
10. env.step(actions=actions) 外部动作接口
11. 动作接口测试
12. step() 时间顺序修正
```

当前核心文件包括：

```text
env/uav_env.py
routing/baselines.py
routing/projection.py
eval/run_baselines.py
eval/plot_results.py
debug/test_action_interface.py
data/results/baseline_results.csv
data/results/baseline_summary.csv
```

---

## 14. 当前技术链路

当前已经打通的链路是：

```text
UAVNetworkEnv
    ↓
生成 state:
    node_features
    edge_index
    edge_attr
    adj
    ↓
baseline policy 或外部 actions
    ↓
env.step()
    ↓
next_state + metrics
```

环境既能跑传统策略：

```python
state, metrics = env.step(policy="energy")
```

也能跑模型动作：

```python
actions = logits_to_actions(logits, state["adj"])
state, metrics = env.step(actions=actions)
```

---

## 15. 下一步计划

接下来进入：

```text
V2：EdgeGAT 路由模型
```

下一步要写：

```text
models/edge_gat.py
debug/test_edge_gat_forward.py
```

目标不是训练，而是先打通：

```text
state
↓
EdgeGATPolicy
↓
logits: [N, N]
↓
logits_to_actions()
↓
actions: [N]
↓
env.step(actions=actions)
```

也就是验证：

```text
动态图状态 → GAT 模型 → 动作投影 → 环境执行
```

完成后，项目就可以进入：

```text
Actor-Critic / PPO 训练
```

阶段。
