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


# 无人机动态路由算法项目开发日志补充

本文件用于追加到已有的 `uav_routing_development_log.md` 后面，记录从上一次日志之后到当前阶段的新增工作。

---

## 十六、V2：EdgeGAT 路由模型前向测试

在完成 V0/V1 环境和 baseline 拆分后，开始进入 V2 阶段：构建基于图神经网络的路由模型。

本阶段首先实现了轻量版边感知 GAT 模型：

```text
models/edge_gat.py
```

该模型输入为：

```text
node_features: [N, 9]
edge_index: [2, E]
edge_attr: [E, 5]
```

输出为：

```text
logits: [N, N]
```

其中：

```text
logits[i, j] 表示节点 i 选择节点 j 作为下一跳的得分
```

随后新增测试文件：

```text
debug/test_edge_gat_forward.py
```

测试链路为：

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

测试通过，说明：

```text
动态图状态 → GAT 模型 → 动作投影 → 环境执行
```

这条前向链路已经打通。

---

## 十七、Actor-Critic 网络封装

在 EdgeGAT 前向测试通过后，进一步封装了 Actor-Critic 网络：

```text
models/actor_critic.py
```

其中：

```text
Actor:
    使用 EdgeGATPolicy
    输出 logits: [N, N]

Critic:
    使用图级平均池化
    输出 value 标量
```

测试文件为：

```text
debug/test_actor_critic_forward.py
```

运行结果如下：

```text
Testing EdgeGAT Actor-Critic forward...
step=1, logits_shape=(20, 20), value=0.0827, delivered=0, dropped=0, violations=0
...
step=10, logits_shape=(20, 20), value=0.0827, delivered=4, dropped=26, violations=0
Actor-Critic forward interface OK.
```

说明：

```text
1. Actor 能输出合法 logits
2. Critic 能输出状态价值 value
3. env.step(actions=actions) 可以正常执行
4. 模型前向链路稳定
```

---

## 十八、PPO 动作采样器实现

为了支持 PPO 训练，新增了动作采样模块：

```text
routing/action_sampler.py
```

该模块包含两个核心函数：

```python
sample_actions_from_logits()
evaluate_actions_from_logits()
```

功能包括：

```text
1. 从 logits 中按概率采样 actions
2. 计算 actions 对应的 log_prob
3. 计算策略 entropy
4. 支持 deterministic=True 用于评估
5. PPO 更新时可重新计算旧 actions 的 log_prob
```

新增测试文件：

```text
debug/test_action_sampler.py
```

测试输出如下：

```text
Testing PPO action sampler...
step=1, value=0.1119, log_prob=-24.4103, check_log_prob=-24.4103, entropy=1.2209, delivered=0, dropped=0, violations=0
...
step=10, value=0.1120, log_prob=-25.9847, check_log_prob=-25.9847, entropy=1.2998, delivered=3, dropped=52, violations=20
PPO action sampler interface OK.
```

关键结论：

```text
log_prob 与 check_log_prob 完全一致，说明 PPO 更新阶段重新计算动作概率的逻辑正确。
entropy 处于正常范围，说明策略仍具有探索性。
```

---

## 十九、最小版 PPO 训练脚本

随后实现了最小版 PPO 训练脚本：

```text
train/train_gat_rl.py
```

该脚本完成：

```text
1. 使用 EdgeGATActorCritic 作为策略网络
2. 从环境采样 rollout
3. 计算 reward
4. 使用 GAE 计算 advantage
5. 使用 PPO clip loss 更新 Actor-Critic
6. 保存训练日志
7. 保存模型权重
```

运行后生成：

```text
data/results/gat_rl_training_log.csv
checkpoints/gat_rl/edge_gat_actor_critic.pt
```

训练结果显示：

```text
PPO 训练流程可以正常运行，但模型效果较差。
```

主要表现为：

```text
1. eval_delivery_ratio 大多只有 0.03 ~ 0.07
2. 明显低于 Energy-aware baseline 的约 0.316
3. value_loss 波动很大
4. actor_loss 接近 0，说明 Actor 更新信号较弱
```

阶段结论：

```text
最小版 PPO 训练流程已跑通，能够正常采样 rollout、计算 log_prob、value、entropy，并保存训练日志和模型权重。但从随机初始化直接使用 PPO 学习路由效果较差，策略明显弱于传统 Energy-aware baseline。因此后续不应继续盲目堆叠复杂网络，而应先采用专家策略监督预训练。
```

当前 PPO 权重保留为调试 checkpoint：

```text
checkpoints/gat_rl/edge_gat_actor_critic.pt
```

但不作为最终模型。

---

## 二十、PPO 训练曲线与 baseline 对比图

为了可视化 PPO 训练效果，新增绘图脚本：

```text
eval/plot_training_curves.py
```

运行命令：

```powershell
python -m eval.plot_training_curves
```

生成目录：

```text
data/results/training_figures/
```

生成的图包括：

```text
ppo_avg_reward.png
ppo_sum_reward.png
ppo_actor_loss.png
ppo_value_loss.png
ppo_entropy.png
ppo_eval_delivery_ratio.png
ppo_eval_drop_ratio.png
delivery_ratio_vs_baselines.png
drop_ratio_vs_baselines.png
avg_delay_vs_baselines.png
constraint_violations_vs_baselines.png
```

该步骤用于直观展示：

```text
GAT-PPO 在当前阶段明显弱于传统 baseline，需要引入专家预训练。
```

---

## 二十一、生成 Energy-aware 专家数据

根据 PPO 训练结果，确定采用：

```text
Energy-aware baseline
↓
生成专家数据
↓
行为克隆预训练 GAT
↓
再用 PPO 微调
```

新增专家数据生成脚本：

```text
train/generate_expert_data.py
```

专家样本格式设计为：

```text
state + src + dst -> expert_next_hop
```

也就是：

```text
当前图状态
当前转发节点 src
目标节点 dst
Energy-aware 策略给出的下一跳 expert_next_hop
```

运行命令：

```powershell
python -m train.generate_expert_data
```

输出结果：

```text
Generating expert dataset...
episode=001, samples=7341, total_samples=7341, delivery=0.2251, drop=0.7718
episode=002, samples=9294, total_samples=16635, delivery=0.1835, drop=0.8146
episode=003, samples=3387, total_samples=20022, delivery=0.3456, drop=0.6341

Expert dataset finished.
Total samples: 20000
```

生成文件：

```text
data/expert/energy_expert_dataset.pkl
data/expert/energy_expert_dataset_stats.csv
```

样本示例：

```text
src: 9
dst: 14
expert_next_hop: 4
node_features: (20, 9)
edge_index: (2, 60)
edge_attr: (60, 5)
adj: (20, 20)
```

---

## 二十二、Destination-aware GAT 模型

由于专家样本是：

```text
state + src + dst -> expert_next_hop
```

原来的 `actions[src] = next_hop` 形式不够精确，因此将模型升级为 destination-aware 形式。

首先扩展环境接口：

```python
def step(self, policy="random", actions=None, action_matrix=None):
```

新增支持：

```text
action_matrix[src, dst] = next_hop
```

这使模型可以针对不同目的节点选择不同下一跳。

随后新增模型文件：

```text
models/dest_edge_gat.py
```

该模型输入：

```text
node_features
edge_index
edge_attr
src_node
dst_node
```

输出：

```text
logits: [N]
```

其中：

```text
logits[j] 表示当前 src 针对目的节点 dst 选择 j 作为下一跳的得分
```

同时实现了：

```python
predict_action_matrix()
```

用于在完整环境中生成：

```text
action_matrix[src, dst] = next_hop
```

新增测试文件：

```text
debug/test_dest_edge_gat_forward.py
```

测试输出：

```text
Testing destination-aware EdgeGAT...
step=1, logits_shape=(20,), action_matrix_shape=(20, 20), delivered=0, dropped=0, violations=0
...
step=10, logits_shape=(20,), action_matrix_shape=(20, 20), delivered=4, dropped=26, violations=0
Destination-aware EdgeGAT interface OK.
```

说明：

```text
state + src + dst -> logits
action_matrix -> env.step(action_matrix)
```

这条链路已经打通。

---

## 二十三、行为克隆预训练 BC-GAT

新增行为克隆训练脚本：

```text
train/pretrain_gat_bc.py
```

使用数据：

```text
data/expert/energy_expert_dataset.pkl
```

训练目标：

```text
让 Destination-aware EdgeGAT 模仿 Energy-aware baseline 的下一跳选择
```

训练输出文件：

```text
data/results/bc_pretrain_log.csv
checkpoints/gat_bc/dest_edge_gat_bc.pt
```

训练结果如下：

```text
epoch,train_loss,train_acc,val_loss,val_acc
1,1.0882,0.6816,1.1413,0.6788
2,1.1517,0.6954,1.1682,0.6918
3,1.1421,0.7023,1.0898,0.6975
4,1.1039,0.7087,1.0593,0.7050
5,1.0883,0.7097,1.0965,0.7005
6,1.0824,0.7135,1.0350,0.7100
7,1.0459,0.7314,0.9148,0.7663
8,0.9039,0.7860,0.7758,0.8210
```

关键结果：

```text
val_acc 从 0.6788 提升到 0.8210
val_loss 从 1.4113 降到 0.7758
```

阶段结论：

```text
Destination-aware EdgeGAT 已经能够较好地模仿 Energy-aware expert 的单步下一跳选择，行为克隆预训练成功。
```

---

## 二十四、BC 预训练曲线

为了观察行为克隆训练过程，新增绘图脚本：

```text
eval/plot_bc_curves.py
```

运行命令：

```powershell
python -m eval.plot_bc_curves
```

生成目录：

```text
data/results/bc_figures/
```

生成图：

```text
bc_loss.png
bc_accuracy.png
```

---

## 二十五、BC-GAT 完整环境评估

为了验证 BC-GAT 在完整 episode 中的实际路由性能，新增评估脚本：

```text
eval/evaluate_bc_policy.py
```

运行命令：

```powershell
python -m eval.evaluate_bc_policy
```

评估结果：

```text
Using device: cpu
Evaluating BC-GAT policy...
seed=0, delivery_ratio=0.2147, drop_ratio=0.7790, avg_delay=7.6656, violations=412
seed=1, delivery_ratio=0.3169, drop_ratio=0.6783, avg_delay=8.0471, violations=303
seed=2, delivery_ratio=0.2571, drop_ratio=0.7360, avg_delay=7.9432, violations=350
seed=3, delivery_ratio=0.3168, drop_ratio=0.6747, avg_delay=7.9521, violations=261
seed=4, delivery_ratio=0.2964, drop_ratio=0.6975, avg_delay=7.7804, violations=291
```

平均结果：

```text
delivery_ratio              0.280360
drop_ratio                  0.713086
avg_delay                   7.877670
avg_remaining_energy      946.075607
constraint_violations     323.400000
total_hops               7729.800000
```

生成文件：

```text
data/results/bc_eval_results.csv
```

---

## 二十六、BC-GAT 与 baseline 对比

新增对比脚本：

```text
eval/compare_bc_with_baselines.py
```

运行后生成：

```text
data/results/baseline_bc_summary.csv
data/results/bc_vs_baselines_figures/
```

当前对比结果：

```text
policy,delivery_ratio,drop_ratio,avg_delay,avg_remaining_energy,constraint_violations,total_hops
random,0.2771,0.7206,6.8717,958.7856,305.6,4075.6
shortest,0.2871,0.7116,7.0660,962.4686,409.4,3672.4
min_delay,0.3005,0.6956,7.1675,951.4102,308.4,5889.4
energy,0.3160,0.6799,7.2167,951.6843,296.2,5811.6
bc_gat,0.2804,0.7131,7.8777,946.0756,323.4,7729.8
```

分析结论：

```text
BC-GAT 已经能够完整运行，并且投递率略高于 random baseline。
但目前仍低于 shortest、min_delay 和 energy baseline。
```

尤其需要注意：

```text
energy total_hops = 5811.6
bc_gat total_hops = 7729.8
```

说明 BC-GAT 虽然单步模仿准确率较高，但在完整 episode 中仍存在：

```text
1. 多跳误差累积
2. 绕行偏多
3. 跳数偏高
4. 时延偏大
5. 丢包率仍较高
```

---

## 二十七、当前阶段总结

从上一次日志到现在，新增完成内容如下：

```text
1. 实现 EdgeGATPolicy
2. 完成 EdgeGAT 前向测试
3. 实现 EdgeGATActorCritic
4. 完成 Actor-Critic 前向测试
5. 实现 PPO 动作采样器 action_sampler.py
6. 完成 log_prob / entropy 测试
7. 实现最小 PPO 训练脚本
8. 跑通 PPO 训练流程并保存 checkpoint
9. 绘制 PPO 训练曲线与 baseline 对比图
10. 分析发现随机初始化 PPO 难以学好路由策略
11. 生成 Energy-aware expert dataset
12. 实现 Destination-aware EdgeGAT
13. 扩展环境支持 action_matrix[src, dst]
14. 完成 Destination-aware GAT 前向测试
15. 使用 expert dataset 完成行为克隆预训练
16. BC-GAT 验证准确率达到 0.8210
17. 绘制 BC 训练曲线
18. 完成 BC-GAT 完整 episode 环境评估
19. 完成 BC-GAT 与传统 baseline 的结果对比
```

---

## 二十八、当前结论

目前项目已经从：

```text
只能跑传统 baseline 的仿真系统
```

推进到：

```text
可以训练和评估神经网络路由策略的完整实验系统
```

已经具备：

```text
动态图环境
baseline 策略
动作投影
EdgeGAT 模型
Actor-Critic 网络
PPO 训练流程
专家数据生成
Destination-aware GAT
行为克隆预训练
模型评估与对比画图
```

当前最重要的实验结论是：

```text
1. 随机初始化的 EdgeGAT-PPO 难以直接学到有效路由策略。
2. Energy-aware expert dataset 可以有效提升 GAT 的单步路由模仿能力。
3. Destination-aware GAT 在行为克隆验证集上的准确率达到 0.8210。
4. 但在完整环境评估中，BC-GAT 的 delivery_ratio 为 0.2804，仍低于 Energy-aware baseline 的 0.3160。
5. BC-GAT 的 total_hops 较高，说明多跳误差累积和绕行问题仍然存在。
```

---

## 二十九、下一步计划

下一步不建议立即进入 Diffusion 或 Graph-MoE，而是先继续增强 BC-GAT。

推荐路线：

```text
扩大 expert dataset
↓
重新训练 BC-GAT
↓
重新评估完整 episode 表现
↓
目标：delivery_ratio ≥ 0.30
↓
再使用 BC 权重初始化 PPO 微调
```

建议修改专家数据配置为：

```python
@dataclass
class ExpertDataConfig:
    n_nodes: int = 20
    episode_steps: int = 200

    num_episodes: int = 100
    seed_start: int = 1000

    max_samples: int = 100000
    expert_policy: str = "energy"
```

如果电脑较慢，可以先用：

```python
max_samples: int = 50000
```

下一步目标：

```text
让 BC-GAT 至少接近 min_delay baseline，最好接近 Energy-aware baseline。
```

达到该目标后，再进行：

```text
BC 初始化 + PPO 微调
```

然后再考虑继续加入：

```text
Diffusion Policy
Graph-MoE
Adaptive Denoising Steps
```
