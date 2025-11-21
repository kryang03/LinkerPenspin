# RL Teacher 训练重要参数总结

本文档列出了 RL Teacher 训练过程中的所有重要超参数及其当前值和定义位置。

---

## A. PPO 核心算法参数 (PPO Core Parameters)
**配置位置**: `configs/train/LinkerHandHora.yaml` → `ppo` 节点

| 参数名 | 当前值 | 定义位置 | 说明 |
|--------|--------|----------|------|
| **learning_rate** | 5e-3 | configs/train/LinkerHandHora.yaml:28 | 初始学习率 |
| **weight_decay** | 1e-4 | configs/train/LinkerHandHora.yaml:29 | AdamW 优化器权重衰减系数 |
| **gamma** | 0.99 | configs/train/LinkerHandHora.yaml:26 | 折扣因子，用于计算未来奖励的权重 |
| **tau** | 0.95 | configs/train/LinkerHandHora.yaml:27 | GAE (Generalized Advantage Estimation) λ参数 |
| **e_clip** | 0.2 | configs/train/LinkerHandHora.yaml:40 | PPO 裁剪范围 ε |
| **entropy_coef** | 0.0 | configs/train/LinkerHandHora.yaml:39 | 熵损失系数（鼓励探索） |
| **critic_coef** | 4 | configs/train/LinkerHandHora.yaml:38 | Critic (价值函数) 损失系数 |
| **bounds_loss_coef** | 0.0 | configs/train/LinkerHandHora.yaml:41 | 动作边界损失系数 |
| **kl_threshold** | 0.02 | configs/train/LinkerHandHora.yaml:30 | KL散度阈值（用于自适应学习率调整） |

---

## B. PPO 数据收集参数 (Data Collection)
**配置位置**: `configs/train/LinkerHandHora.yaml` → `ppo` 节点

| 参数名 | 当前值 | 定义位置 | 说明 |
|--------|--------|----------|------|
| **num_actors** | ${...task.env.numEnvs} | configs/train/LinkerHandHora.yaml:24 | 并行环境数量（通常为 8192） |
| **horizon_length** | 12 | configs/train/LinkerHandHora.yaml:34 | 每次 rollout 收集的步数 |
| **minibatch_size** | 16384 | configs/train/LinkerHandHora.yaml:35 | PPO 更新时的小批量大小 |
| **mini_epochs** | 5 | configs/train/LinkerHandHora.yaml:36 | 每个数据收集周期的训练轮数 |
| **max_agent_steps** | 3000000000 | configs/train/LinkerHandHora.yaml:50 | 最大训练步数 |

**说明**: batch_size = num_actors × horizon_length = 8192 × 12 = 98304

---

## C. 梯度与优化参数 (Gradient & Optimization)
**配置位置**: `configs/train/LinkerHandHora.yaml` → `ppo` 节点

| 参数名 | 当前值 | 定义位置 | 说明 |
|--------|--------|----------|------|
| **truncate_grads** | True | configs/train/LinkerHandHora.yaml:45 | 是否裁剪梯度 |
| **grad_norm** | 1.0 | configs/train/LinkerHandHora.yaml:46 | 梯度裁剪的最大范数 |
| **global_scheduler** | CosineAnnealingLR | ppo_rl_teacher.py:135-137 | 余弦退火学习率调度器 (eta_min=1e-6) |
| **adaptive_scheduler** | AdaptiveScheduler | ppo_rl_teacher.py:158 | 基于KL散度的自适应学习率调度器 |

---

## D. 标准化参数 (Normalization)
**配置位置**: `configs/train/LinkerHandHora.yaml` → `ppo` 节点

| 参数名 | 当前值 | 定义位置 | 说明 |
|--------|--------|----------|------|
| **normalize_input** | True | configs/train/LinkerHandHora.yaml:19 | 标准化观察输入 |
| **normalize_value** | True | configs/train/LinkerHandHora.yaml:20 | 标准化价值函数输出 |
| **normalize_priv** | True | configs/train/LinkerHandHora.yaml:21 | 标准化特权信息 |
| **normalize_point_cloud** | True | configs/train/LinkerHandHora.yaml:22 | 标准化点云数据 |
| **normalize_advantage** | True | configs/train/LinkerHandHora.yaml:25 | 标准化优势函数 |
| **value_bootstrap** | True | configs/train/LinkerHandHora.yaml:23 | 是否使用 Value Bootstrap |

---

## E. 环境与奖励参数 (Environment & Reward)

### E.1 奖励权重 (Reward Weights)
**配置位置**: `configs/task/LinkerHandHora.yaml` → `env.reward` 节点

| 参数名 | 当前值 | 定义位置 | 说明 |
|--------|--------|----------|------|
| **rotate_reward_scale** | 1.0 | configs/task/LinkerHandHora.yaml:55 | 旋转奖励权重 |
| **waypoint_sparse_reward_scale** | 1.0 | configs/task/LinkerHandHora.yaml:57 | 路径点稀疏奖励权重 |
| **torque_penalty_scale** | -0.1 | configs/task/LinkerHandHora.yaml:58 | 力矩惩罚权重 |
| **work_penalty_scale** | -1.0 | configs/task/LinkerHandHora.yaml:59 | 做功惩罚权重 |
| **pencil_z_dist_penalty_scale** | -1.0 | configs/task/LinkerHandHora.yaml:62 | Z轴高度差惩罚权重 |
| **position_penalty_scale** | -0.1 | configs/task/LinkerHandHora.yaml:64 | 位置惩罚权重 |
| **rotate_penalty_scale** | -0.3 | configs/task/LinkerHandHora.yaml:61 | 旋转惩罚权重 |
| **hand_pose_consistency_penalty_scale** | -1.0 | configs/task/LinkerHandHora.yaml:60 | 手部姿态一致性惩罚权重 |
| **obj_linvel_penalty_scale** | -0.3 | configs/task/LinkerHandHora.yaml:56 | 物体线速度惩罚权重 |

### E.2 奖励内部缩放因子 (Internal Reward Scaling)
**硬编码位置**: `penspin/tasks/linker_hand_hora.py` (第 58-72 行)

```python
REWARD_SCALE_DICT = {
    'obj_linvel_penalty': 1.0,           # [0]
    'rotate_reward': 0.7,                # [1]
    'waypoint_sparse_reward': 100,       # [2]
    'torque_penalty': 0.1,               # [3]
    'work_penalty': 0.05,                # [4]
    'pencil_z_dist_penalty': 3.0,        # [5]
    'position_penalty': 20000.0,         # [6]
    'rotate_penalty': 4.0,               # [7]
    'hand_pose_consistency_penalty': 80.0 # [8]
}
```

**最终奖励计算**: `reward = yaml_scale × internal_scale × raw_value`

### E.3 旋转奖励与惩罚参数
**配置位置**: `configs/task/LinkerHandHora.yaml` → `env.reward` 节点

| 参数名 | 当前值 | 定义位置 | 说明 |
|--------|--------|----------|------|
| **angvelClipMin** | -0.5 | configs/task/LinkerHandHora.yaml:52 | 角速度裁剪最小值 (rad/s) |
| **angvelClipMax** | 0.5 | configs/task/LinkerHandHora.yaml:53 | 角速度裁剪最大值 (rad/s) - **HPO重点** |
| **angvelPenaltyThres** | 1.0 | configs/task/LinkerHandHora.yaml:54 | 角速度惩罚阈值 (rad/s) - **HPO重点** |

**Shell脚本覆盖值** (`scripts/train_rl_teacher.sh`):
- `angvelClipMax=3.0` (第30行)
- `angvelPenaltyThres=3.5` (第31行)
- `angvelClipMin=-0.1` (第32行)

### E.4 环境基础参数
**配置位置**: `configs/task/LinkerHandHora.yaml` → `env` 节点

| 参数名 | 当前值 | 定义位置 | 说明 |
|--------|--------|----------|------|
| **numEnvs** | 8192 | configs/task/LinkerHandHora.yaml:8 | 并行环境数量 |
| **episodeLength** | 400 | configs/task/LinkerHandHora.yaml:14 | 最大 episode 长度（步） |
| **reset_height_threshold** | 0.12 | configs/task/LinkerHandHora.yaml:30 | 物体掉落重置高度阈值 (m) |
| **forceScale** | 2.0 | configs/task/LinkerHandHora.yaml:37 | 随机力缩放因子 |
| **randomForceProbScalar** | 0.25 | configs/task/LinkerHandHora.yaml:38 | 随机力应用概率 |

---

## F. 控制器参数 (Controller)
**配置位置**: `configs/task/LinkerHandHora.yaml` → `env.controller` 节点

| 参数名 | 当前值 | 定义位置 | 说明 |
|--------|--------|----------|------|
| **torque_control** | True | configs/task/LinkerHandHora.yaml:19 | 是否使用力矩控制 |
| **controlFrequencyInv** | 10 | configs/task/LinkerHandHora.yaml:20 | 控制频率倒数 (20Hz 控制频率) |
| **pgain** | 3 | configs/task/LinkerHandHora.yaml:21 | PD控制器比例增益 |
| **dgain** | 0.1 | configs/task/LinkerHandHora.yaml:22 | PD控制器微分增益 |
| **action_scale** | 0.04167 | configs/task/LinkerHandHora.yaml:23 | 动作缩放因子 |
| **torque_limit** | 0.7 | configs/task/LinkerHandHora.yaml:24 | 力矩限制 (Nm) |

---

## G. 仿真参数 (Simulation)
**配置位置**: `configs/task/LinkerHandHora.yaml` → `sim` 节点

| 参数名 | 当前值 | 定义位置 | 说明 |
|--------|--------|----------|------|
| **dt** | 0.005 | configs/task/LinkerHandHora.yaml:136 | 仿真时间步长 (200 Hz) |
| **substeps** | 1 | configs/task/LinkerHandHora.yaml:137 | 物理仿真子步数 |
| **gravity** | [0.0, 0.0, -9.81] | configs/task/LinkerHandHora.yaml:140 | 重力加速度 (m/s²) |

**PhysX 参数**:

| 参数名 | 当前值 | 定义位置 | 说明 |
|--------|--------|----------|------|
| **solver_type** | 1 | configs/config.yaml:23 | 求解器类型 (0: PGS, 1: TGS) |
| **num_position_iterations** | 8 | configs/task/LinkerHandHora.yaml:145 | 位置迭代次数 |
| **num_velocity_iterations** | 0 | configs/task/LinkerHandHora.yaml:146 | 速度迭代次数 |
| **contact_offset** | 0.002 | configs/task/LinkerHandHora.yaml:149 | 接触偏移 (m) |
| **rest_offset** | 0.0 | configs/task/LinkerHandHora.yaml:150 | 静止偏移 (m) |

---

## H. 网络结构参数 (Network Architecture)
**配置位置**: `configs/train/LinkerHandHora.yaml` → `network` 节点

| 参数名 | 当前值 | 定义位置 | 说明 |
|--------|--------|----------|------|
| **mlp.units** | [512, 256, 128] | configs/train/LinkerHandHora.yaml:5 | Actor/Critic 主干网络层单元数 |
| **priv_mlp.units** | [256, 128, 8] | configs/train/LinkerHandHora.yaml:7 | 特权信息 MLP 层单元数 |
| **point_mlp.units** | [32, 32, 32] | configs/train/LinkerHandHora.yaml:9 | 点云 MLP 层单元数 |
| **use_point_transformer** | False | configs/train/LinkerHandHora.yaml:12 | 是否使用 Point Transformer |

---

## I. 域随机化参数 (Domain Randomization)
**配置位置**: `configs/task/LinkerHandHora.yaml` → `env.randomization` 节点

### I.1 物理属性随机化

| 参数名 | 当前值 | 说明 |
|--------|--------|------|
| **randomizeMass** | True | 是否随机化物体质量 |
| **randomizeMassLower/Upper** | [0.01, 0.02] | 质量随机范围 (kg) |
| **randomizeCOM** | True | 是否随机化质心 |
| **randomizeCOMLower/Upper** | [-0.001, 0.001] | 质心偏移范围 (m) |
| **randomizeFriction** | True | 是否随机化摩擦系数 |
| **randomizeFrictionLower/Upper** | [0.3, 3.0] | 摩擦系数范围 |

### I.2 PD增益随机化

| 参数名 | 当前值 | 说明 |
|--------|--------|------|
| **randomizePDGains** | True | 是否随机化PD增益 |
| **randomizePGainLower/Upper** | [2.5, 3.5] | P增益随机范围 |
| **randomizeDGainLower/Upper** | [0.09, 0.11] | D增益随机范围 |

### I.3 噪声参数

| 参数名 | 当前值 | 说明 |
|--------|--------|------|
| **obs_noise_e_scale** | 0.01 | 观察噪声 episode 级别缩放 |
| **obs_noise_t_scale** | 0.005 | 观察噪声时间步级别缩放 |
| **action_noise_e_scale** | 0.01 | 动作噪声 episode 级别缩放 |
| **action_noise_t_scale** | 0.005 | 动作噪声时间步级别缩放 |

---

## J. Teacher 特有参数 (Teacher-Specific)
**配置位置**: `configs/train/LinkerHandHora.yaml` → `ppo` 节点

| 参数名 | 当前值 | 定义位置 | 说明 |
|--------|--------|----------|------|
| **priv_info** | True | configs/train/LinkerHandHora.yaml:52 | 使用特权信息（Teacher必须为True） |
| **use_point_cloud_info** | True | configs/train/LinkerHandHora.yaml:56 | 使用点云信息（Teacher必须为True） |
| **proprio_mode** | True | configs/train/LinkerHandHora.yaml:53 | 本体感觉模式 |
| **input_mode** | 'proprio-ends' | configs/train/LinkerHandHora.yaml:54 | 输入模式 |
| **proprio_len** | 30 | configs/train/LinkerHandHora.yaml:57 | 本体感觉历史长度 |
| **asymm_actor_critic** | False | configs/train/LinkerHandHora.yaml:61 | 是否使用非对称 Actor-Critic |
| **critic_info_dim** | 100 | configs/train/LinkerHandHora.yaml:62 | Critic 额外信息维度 |

---

## K. 模型保存参数 (Checkpoint)
**配置位置**: `configs/train/LinkerHandHora.yaml` → `ppo` 节点

| 参数名 | 当前值 | 定义位置 | 说明 |
|--------|--------|----------|------|
| **save_best_after** | 0 | configs/train/LinkerHandHora.yaml:48 | 在多少步后开始保存最优模型 |
| **save_frequency** | 500 | configs/train/LinkerHandHora.yaml:49 | 模型保存频率（epoch） |
| **output_name** | 'debug' | configs/train/LinkerHandHora.yaml:18 | 输出目录名称 |

---

## L. 特权信息维度配置 (Privileged Information)
**配置位置**: `configs/task/LinkerHandHora.yaml` → `env.privInfo` 节点

Teacher 默认开启的特权信息:

| 参数名 | 当前值 | 说明 |
|--------|--------|------|
| **enableObjPos** | True | 物体位置 (3维) |
| **enableObjScale** | True | 物体缩放 (1维) |
| **enableObjMass** | True | 物体质量 (1维) |
| **enableObjCOM** | True | 质心偏移 (3维) |
| **enableObjFriction** | True | 摩擦系数 (1维) |
| **enable_obj_restitution** | True | 恢复系数 (1维) |
| **enable_tactile** | True | 触觉信息 (CONTACT_DIM维) |
| **enable_obj_orientation** | True | 物体姿态 (4维，四元数) |
| **enable_obj_angvel** | True | 物体角速度 (3维) |
| **enable_ft_pos** | True | 指尖位置 (FINGERTIP_POS_DIM维) |

---

## M. 关键硬编码常量
**位置**: `penspin/tasks/linker_hand_hora.py`

```python
# 第 58-72 行: 奖励缩放字典
REWARD_SCALE_DICT = {...}  # 见 E.2 节

# 第 73 行: 触觉阈值
CONTACT_THRESH = 0.02

# 第 74 行: 触觉力上限
TACTILE_FORCE_MAX = 4.0

# 第 38-45 行: 物体平均位置
OBJ_CANON_POS = [-0.11722512543201447, 0.006986482068896294, 0.1717524379491806]

# 第 49 行: 手部相似度缩放因子
HAND_SIMILARITY_SCALE_FACTOR = 0.5

# 第 50 行: 姿态相似度阈值
ORIENTATION_SIMILARITY_THRESHOLD = 0.9
```

---

## N. HPO 优先级建议 (Hyperparameter Optimization Priority)

### 第一优先级（算法核心参数）:
1. `learning_rate` (5e-3)
2. `gamma` (0.99)
3. `tau` (0.95)
4. `entropy_coef` (0.0)
5. `critic_coef` (4)
6. `mini_epochs` (5)
7. `horizon_length` (12)

### 第二优先级（奖励塑造）:
1. `angvelClipMax` (3.0) - Shell脚本值
2. `angvelPenaltyThres` (3.5) - Shell脚本值
3. 奖励权重缩放参数 (configs/task/LinkerHandHora.yaml → reward节点)
4. 内部奖励缩放因子 (linker_hand_hora.py → REWARD_SCALE_DICT)

### 第三优先级（网络结构）:
1. `mlp.units` ([512, 256, 128])
2. `priv_mlp.units` ([256, 128, 8])
3. `point_mlp.units` ([32, 32, 32])

---

## O. 参数修改优先级

### 通过 Shell 脚本修改（推荐）:
```bash
scripts/train_rl_teacher.sh
# 可直接覆盖 YAML 中的参数
```

### 通过 YAML 配置文件修改:
```yaml
configs/train/LinkerHandHora.yaml     # PPO 算法参数
configs/task/LinkerHandHora.yaml       # 环境与奖励参数
```

### 需要修改代码:
```python
penspin/tasks/linker_hand_hora.py     # 硬编码常量（REWARD_SCALE_DICT等）
```

---

**文档版本**: 2025-11-17  
**适用代码版本**: LinkerPenspin master branch
