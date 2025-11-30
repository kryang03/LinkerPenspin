# RL Teacher 超参数优化指南

本文档说明如何使用 Optuna 进行 RL Teacher PPO 模型的自动超参数优化。

## 优化策略

**评分方法（综合评分）：**
- **主要指标**: 最佳奖励值 (best_reward)，权重 1.0
- **成功加成**: 成功率 (旋转角度>10弧度)，权重 1000
- **综合评分公式**: `composite_score = best_reward + 1000 * success_rate`

这个策略确保：
1. 以最终reward为主要优化目标
2. 当出现成功案例（旋转>4弧度）时，获得巨大加成
3. 因为成功案例非常罕见，所以一旦出现就会显著提升评分

## 优化的超参数

基于 `RL_TEACHER_PARAMETERS.md`，优化以下关键参数：

### A. PPO核心算法参数
- `learning_rate`: 学习率 [1e-4, 1e-2]
- `weight_decay`: 权重衰减 [1e-5, 1e-3]
- `gamma`: 折扣因子 {0.98, 0.99, 0.995}
- `tau`: GAE lambda {0.90, 0.95, 0.97}
- `e_clip`: PPO裁剪范围 {0.1, 0.2, 0.3}
- `entropy_coef`: 熵系数 [0.0, 0.01]
- `critic_coef`: Critic损失系数 {2, 4, 8}
- `kl_threshold`: KL散度阈值 [0.01, 0.05]

### B. PPO数据收集参数
- `mini_epochs`: 训练轮数 [3, 8]
- `minibatch_size`: 小批量大小 {8192, 16384, 32768}

### C. 梯度优化参数
- `grad_norm`: 梯度裁剪范数 {0.5, 1.0, 2.0}

### D. 环境与奖励参数（重点）
- `angvelClipMax`: 角速度裁剪最大值 [1.0, 5.0] ⭐
- `angvelPenaltyThres`: 角速度惩罚阈值 [2.0, 6.0] ⭐
- `rotate_reward_scale`: 旋转奖励权重 [0.5, 2.0]
- `torque_penalty_scale`: 力矩惩罚权重 [-0.5, -0.05]
- `work_penalty_scale`: 做功惩罚权重 [-2.0, -0.5]
- `position_penalty_scale`: 位置惩罚权重 [-0.5, -0.05]
- `rotate_penalty_scale`: 旋转惩罚权重 [-0.5, -0.1]

## 使用方法

### 1. 基础使用

```bash
# 在GPU 0上运行50次试验，每次训练100M步
python optuna/tune_teacher.py --gpu 0 --n_trials 50 --max_steps 100000000
```

### 2. 自定义配置

```bash
python optuna/tune_teacher.py \
    --gpu 0 \
    --n_trials 100 \
    --max_steps 200000000 \
    --storage "sqlite:///optuna/my_hpo.db" \
    --study_name "my_teacher_hpo" \
    --load_if_exists
```

### 3. 继续之前的优化

```bash
# 使用 --load_if_exists 标志可以从数据库中加载已有的study并继续优化
python optuna/tune_teacher.py \
    --gpu 0 \
    --n_trials 50 \
    --load_if_exists
```

### 4. 多GPU并行优化

在不同终端同时运行多个优化进程（共享同一个数据库）：

```bash
# 终端1 - GPU 0
python optuna/tune_teacher.py --gpu 0 --n_trials 25 --load_if_exists

# 终端2 - GPU 1
python optuna/tune_teacher.py --gpu 1 --n_trials 25 --load_if_exists

# 终端3 - GPU 2
python optuna/tune_teacher.py --gpu 2 --n_trials 25 --load_if_exists
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--gpu` | GPU ID | 0 |
| `--n_trials` | 优化试验次数 | 50 |
| `--max_steps` | 每次试验的最大训练步数 | 100000000 (100M) |
| `--storage` | Optuna数据库路径 | sqlite:///optuna/hpo_teacher.db |
| `--study_name` | Study名称 | teacher_ppo_hpo |
| `--load_if_exists` | 如果study已存在则加载并继续 | False |

## 输出文件

### 1. 训练输出
每个trial的训练结果保存在：
```
outputs/LinkerHandHora/optuna_trial_XXXX/
├── teacher_nn/          # 模型checkpoint
├── teacher_tb/          # TensorBoard日志
└── training.log         # 训练日志
```

### 2. Optuna数据库
```
optuna/hpo_teacher.db    # SQLite数据库，存储所有试验结果
```

### 3. 最佳参数
```
optuna/best_params_teacher_ppo_hpo.txt    # 最佳超参数配置
```

### 4. 可视化图表（如果安装了plotly）
```
optuna/param_importances_teacher_ppo_hpo.html    # 参数重要性
optuna/optimization_history_teacher_ppo_hpo.html  # 优化历史
```

## 结果分析

### 1. 查看最佳参数

```bash
cat optuna/best_params_teacher_ppo_hpo.txt
```

### 2. 使用最佳参数进行完整训练

将最佳参数应用到训练脚本：

```bash
scripts/train_rl_teacher.sh 0 42 best_hpo_run \
    train.ppo.learning_rate=0.003 \
    train.ppo.gamma=0.99 \
    task.env.reward.angvelClipMax=3.5 \
    # ... 其他最佳参数
```

### 3. 查看TensorBoard对比

```bash
# 查看所有试验的TensorBoard
tensorboard --logdir outputs/LinkerHandHora/optuna_trial_* --port 6006
```

### 4. 使用Optuna Dashboard（可选）

```bash
# 安装
pip install optuna-dashboard

# 启动
optuna-dashboard sqlite:///optuna/hpo_teacher.db
```

访问 http://localhost:8080 查看交互式界面。

## 监控指标

在训练过程中，系统会记录以下指标：

- `episode_rewards/step`: Episode平均奖励
- `episode_lengths/step`: Episode平均长度
- `total_rot_angle(rad)/step`: 平均旋转角度
- `success_rate/step`: 成功率（旋转>10弧度）⭐
- `success_count/step`: 成功案例数量⭐

## 优化建议

### 1. 快速迭代策略
- 初期使用较少的训练步数（50M-100M）快速探索
- 找到有希望的参数区域后，增加训练步数（200M-300M）精细优化

### 2. 计算资源分配
- 单个试验预计耗时：1-3小时（取决于max_steps）
- 建议使用多GPU并行加速优化过程
- 可以随时中断（Ctrl+C）并稍后继续

### 3. 参数空间调整
如果初步结果不理想，可以修改 `tune_teacher.py` 中的参数搜索范围：
```python
# 例如扩大学习率搜索范围
lr = trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True)
```

### 4. 重点关注参数
根据经验，以下参数对性能影响最大：
- `angvelClipMax` 和 `angvelPenaltyThres`：直接影响旋转行为
- `learning_rate` 和 `weight_decay`：影响训练稳定性
- `rotate_reward_scale` 等奖励权重：塑造学习目标

## 故障排除

### 问题1：训练过程中OOM（内存不足）
**解决方案**：
- 减小 `minibatch_size`
- 减少 `num_actors`（需要在配置文件中修改）

### 问题2：训练不稳定或发散
**解决方案**：
- 调整学习率范围，使用更保守的值
- 增加 `grad_norm` 裁剪
- 调整 `kl_threshold`

### 问题3：优化进度缓慢
**解决方案**：
- 使用多GPU并行
- 减少每次试验的 `max_steps`
- 使用更激进的pruner策略

## 依赖安装

```bash
# 核心依赖
pip install optuna

# 可视化（可选）
pip install plotly kaleido optuna-dashboard
```

## 注意事项

1. **数据库备份**：定期备份 `hpo_teacher.db` 文件
2. **磁盘空间**：确保有足够空间存储所有试验的输出
3. **GPU监控**：使用 `nvidia-smi` 监控GPU使用情况
4. **日志查看**：可以实时查看每个trial的训练日志

## 联系与支持

如有问题，请参考：
- 主项目文档：`README.md`
- 参数说明：`optuna/RL_TEACHER_PARAMETERS.md`
- 训练指南：`TRAINING_GUIDE.md`
