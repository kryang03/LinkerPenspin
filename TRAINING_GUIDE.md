# LinkerPenspin 训练流程说明

## 概述

本项目实现了三个阶段的训练流程，用于灵巧手转笔任务：

### 功能1: 纯PPO强化学习训练Teacher模型
- **训练器**: `PPOTeacher` (penspin/algo/ppo/ppo_rl_teacher.py)
- **输入**: 完整信息（特权信息 + 点云 + 本体感觉）
- **目标**: 在仿真中训练expert策略
- **特点**: 纯强化学习，无BC loss

### 功能2: RL+BC训练Teacher模型（可选）
- **训练器**: `PPO_RL_BC_Teacher` (penspin/algo/ppo/ppo_rl_bc_teacher.py)
- **输入**: 完整信息（特权信息 + 点云 + 本体感觉）
- **依赖**: 需要功能1训练好的Teacher模型
- **目标**: 使用已训练的Teacher模型回放，通过RL+BC微调策略
- **应用场景**: 例如保证完整转笔轨迹，但改变转笔速度

### 功能3: RL+BC训练Student模型
- **训练器**: `PPO_RL_BC_Student` (penspin/algo/ppo/ppo_rl_bc_student.py)
- **输入**: 仅本体感觉历史（proprio history）
- **依赖**: 需要功能1训练好的Teacher模型
- **目标**: 训练可以部署到真机的策略（不依赖特权信息）
- **特点**: Student只使用真机可获得的信息，通过Teacher蒸馏学习

## 训练流程

### 步骤1: 训练Teacher模型（功能1）

```bash
# 使用GPU 0, 随机种子42, 输出目录名称为teacher_baseline
bash scripts/train_rl_teacher.sh 0 42 teacher_baseline
```

**关键配置**:
- `train.algo=PPOTeacher`
- `train.ppo.priv_info=True` (使用特权信息)
- 输出目录: `outputs/LinkerHandHora/teacher_baseline/teacher_nn/`

**训练完成后**，会生成：
- `last.pth`: 最后一个checkpoint
- `best_reward_xxx.pth`: 最佳奖励的checkpoint

### 步骤2a: RL+BC微调Teacher模型（功能2，可选）

如果想在Teacher基础上通过RL+BC进行微调：

```bash
# 需要提供步骤1训练好的Teacher模型路径
bash scripts/train_rl_bc_teacher.sh 0 42 teacher_rl_bc \
  outputs/LinkerHandHora/teacher_baseline/teacher_nn/best_reward_xxx.pth
```

**关键配置**:
- `train.algo=PPO_RL_BC_Teacher`
- `train.demon_path`: Teacher模型路径
- `train.ppo.bc_loss_coef=1.0`: BC loss权重
- 输出目录: `outputs/LinkerHandHora/teacher_rl_bc/stage1_nn/`

### 步骤2b: 训练Student模型（功能3）

训练可部署到真机的Student策略：

```bash
# 需要提供步骤1训练好的Teacher模型路径
bash scripts/train_rl_bc_student.sh 0 42 student_deploy \
  outputs/LinkerHandHora/teacher_baseline/teacher_nn/best_reward_xxx.pth
```

**关键配置**:
- `train.algo=PPO_RL_BC_Student`
- `train.demon_path`: Teacher模型路径
- `train.ppo.priv_info=False` (不使用特权信息)
- `train.ppo.proprio_mode=True` (只使用本体感觉历史)
- `train.ppo.bc_loss_coef=1.0`: BC loss权重
- 输出目录: `outputs/LinkerHandHora/student_deploy/stage1_nn/`

## 模型架构差异

### Teacher模型 (TeacherActorCritic)
- **Actor输入**: proprio_hist (30×21) + priv_info (47) + point_cloud (100×3)
- **Critic输入**: 可选择非对称critic (critic_info)
- **输出**: 动作 (21维) + value

### Student模型 (StudentActorCritic)
- **Actor输入**: 仅 proprio_hist (30×21)
- **Critic输入**: 可选择非对称critic
- **输出**: 动作 (21维) + value
- **蒸馏**: 通过Teacher回放学习，对齐latent表示

## 损失函数

### PPOTeacher (功能1)
```
Total Loss = Actor Loss + critic_coef * Critic Loss 
           - entropy_coef * Entropy + bounds_loss_coef * Bounds Loss
```

### PPO_RL_BC_Teacher (功能2)
```
Total Loss = Actor Loss + critic_coef * Critic Loss 
           - entropy_coef * Entropy + bc_loss_coef * BC Loss
           + distill_loss_coef * Latent Loss (可选)
```

### PPO_RL_BC_Student (功能3)
```
Total Loss = Actor Loss + critic_coef * Critic Loss 
           - entropy_coef * Entropy + bc_loss_coef * BC Loss
           + distill_loss_coef * Latent Loss (可选)
```

其中：
- **BC Loss**: 行为克隆损失，使当前策略模仿Teacher策略
- **Latent Loss**: 潜在表示对齐损失（可选）

## 配置文件

### 主配置文件
- `configs/config.yaml`: 全局配置
- `configs/task/LinkerHandHora.yaml`: 任务配置
- `configs/train/LinkerHandHora.yaml`: 训练配置

### 重要参数

```yaml
# 功能1: PPO Teacher
train:
  algo: PPOTeacher
  ppo:
    priv_info: True
    use_point_cloud_info: True
    bc_loss_coef: 0.0  # 纯RL，不使用BC

# 功能2: PPO_RL_BC Teacher
train:
  algo: PPO_RL_BC_Teacher
  demon_path: <teacher_model_path>
  ppo:
    priv_info: True
    use_point_cloud_info: True
    bc_loss_coef: 1.0  # 启用BC loss

# 功能3: PPO_RL_BC Student
train:
  algo: PPO_RL_BC_Student
  demon_path: <teacher_model_path>
  ppo:
    priv_info: False  # 不使用特权信息
    proprio_mode: True
    input_mode: proprio
    use_point_cloud_info: False  # 不使用点云
    bc_loss_coef: 1.0  # 启用BC loss
```

## 环境要求

### 观测空间
- **proprio_hist**: 本体感觉历史 (30×21)
  - 21维: 关节位置(21) + 关节速度(21)  # 实际是根据PROPRIO_DIM定义
- **priv_info**: 特权信息 (47维)
  - 物体位置、方向、速度等仅在仿真中可获得的信息
- **point_cloud**: 点云 (100×3)
  - 物体表面采样点

### 动作空间
- 21维连续动作（关节位置目标）
- 范围: [-1, 1]

## 测试模型

训练完成后，可以使用测试模式评估模型：

```bash
# 测试Teacher模型
python train.py task=LinkerHandHora \
  train.algo=PPOTeacher \
  test=True \
  train.load_path=outputs/LinkerHandHora/teacher_baseline/teacher_nn/best_reward_xxx.pth \
  headless=False

# 测试Student模型
python train.py task=LinkerHandHora \
  train.algo=PPO_RL_BC_Student \
  test=True \
  train.load_path=outputs/LinkerHandHora/student_deploy/stage1_nn/best.pth \
  headless=False
```

## 训练监控

训练过程会在TensorBoard中记录：

```bash
# 启动TensorBoard
tensorboard --logdir outputs/LinkerHandHora/<experiment_name>/
```

监控指标：
- `episode_rewards/step`: 平均回合奖励
- `episode_lengths/step`: 平均回合长度
- `losses/actor_loss`: Actor损失
- `losses/critic_loss`: Critic损失
- `losses/bc_loss`: BC损失（功能2和3）
- `info/kl`: KL散度

## 故障排除

### 常见问题

1. **缺少demon_path参数**
   - 解决：在训练功能2或3时，必须提供Teacher模型路径

2. **CUDA内存不足**
   - 解决：减少`task.env.numEnvs`或`train.ppo.minibatch_size`

3. **模型无法加载**
   - 解决：确保checkpoint包含所需的所有权重（model, running_mean_std等）

4. **BC loss不生效**
   - 解决：检查`train.ppo.bc_loss_coef`是否设置为非零值

## 代码结构

```
penspin/
├── algo/ppo/
│   ├── ppo_rl_teacher.py      # 功能1: 纯PPO Teacher
│   ├── ppo_rl_bc_teacher.py   # 功能2: RL+BC Teacher
│   └── ppo_rl_bc_student.py   # 功能3: RL+BC Student
├── tasks/
│   └── linker_hand_hora.py    # 转笔任务环境
└── utils/
    └── robot_config.py        # 机器人配置常量

scripts/
├── train_rl_teacher.sh        # 功能1训练脚本
├── train_rl_bc_teacher.sh     # 功能2训练脚本
└── train_rl_bc_student.sh     # 功能3训练脚本

configs/
├── config.yaml                # 全局配置
├── task/LinkerHandHora.yaml   # 任务配置
└── train/LinkerHandHora.yaml  # 训练配置
```

## 总结

该训练框架支持三个关键训练阶段：

1. **功能1 (PPOTeacher)**: 使用完整信息训练expert策略
2. **功能2 (PPO_RL_BC_Teacher)**: 在Teacher基础上通过RL+BC微调（可选）
3. **功能3 (PPO_RL_BC_Student)**: 训练仅依赖本体感觉的可部署策略

这种设计允许在仿真中训练强大的Teacher模型，然后将知识蒸馏到只使用真机可获得信息的Student模型中，实现sim-to-real迁移。
