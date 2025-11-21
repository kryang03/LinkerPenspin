# Optuna 超参数优化 - 快速指南

## 📋 目录

- [快速开始](#快速开始)
- [文档说明](#文档说明)
- [核心概念](#核心概念)
- [使用示例](#使用示例)
- [预期效果](#预期效果)

---

## 🚀 快速开始

### 0. 首次使用必读 ⚠️

**重要更新（2025-11-18）**：系统已修复 Isaac Gym Foundation 重复创建问题。现在每个试验在独立子进程中运行，完全稳定！

```bash
# 首次使用前验证配置
bash optuna/verify_setup.sh

# 快速测试修复（推荐，约10分钟）
bash optuna/test_foundation_fix.sh
```

### 1. 第一次使用（测试功能）

```bash
# 运行快速测试（2次试验，每次1M步，约5-10分钟）
bash optuna/test_hpo.sh
```

### 2. 正式优化（推荐配置）

```bash
# 在GPU 0上运行50次试验，每次训练100M步
bash optuna/run_hpo.sh 0 50
```

### 3. 多GPU加速

```bash
# 终端1 - GPU 0
bash optuna/run_hpo.sh 0 25

# 终端2 - GPU 1
bash optuna/run_hpo.sh 1 25

# 终端3 - GPU 2
bash optuna/run_hpo.sh 2 25
```

### 4. 查看结果

```bash
# 查看最佳参数
cat optuna/best_params_teacher_ppo_hpo.txt

# 查看TensorBoard
tensorboard --logdir outputs/LinkerHandHora/optuna_trial_*
```

---

## 📚 文档说明

| 文档 | 说明 | 适合人群 |
|------|------|----------|
| **README.md** (本文件) | 快速入门和概览 | 所有人 |
| [EXAMPLES.md](EXAMPLES.md) | 详细使用示例和常见问题 | 新手 |
| [README_HPO.md](README_HPO.md) | 完整技术文档和参数说明 | 进阶用户 |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | 实现细节和技术总结 | 开发者 |
| [RL_TEACHER_PARAMETERS.md](RL_TEACHER_PARAMETERS.md) | 原始参数文档 | 参考 |

---

## 💡 核心概念

### 优化目标

**综合评分 = best_reward + 1000 × success_rate**

- **best_reward**：训练过程中的最佳平均奖励（主要指标）
- **success_rate**：旋转角度>10弧度的比例（关键成功指标）
- **权重1000**：确保任何成功案例都会显著提升评分

### 为什么这样设计？

1. ✅ 以reward为主要优化目标（更全面、更稳定）
2. ✅ 成功案例（旋转>4弧度）获得巨大加成
3. ✅ 平衡整体性能和突破性进展

### 优化的参数

基于 [RL_TEACHER_PARAMETERS.md](RL_TEACHER_PARAMETERS.md)，优化 **18个关键超参数**：

- **PPO算法参数**（8个）：learning_rate, gamma, tau, 等
- **训练参数**（3个）：mini_epochs, minibatch_size, grad_norm
- **奖励参数**（7个）：angvelClipMax, rotate_reward_scale, 等

详细参数列表见 [README_HPO.md](README_HPO.md#优化的超参数)

---

## 📖 使用示例

### 场景1：快速验证参数

```bash
# 使用较少步数快速试验
python optuna/tune_teacher.py --gpu 0 --n_trials 30 --max_steps 50000000
```

### 场景2：精细优化

```bash
# 增加训练步数获得更准确评估
python optuna/tune_teacher.py --gpu 0 --n_trials 30 --max_steps 200000000
```

### 场景3：继续之前的优化

```bash
# 自动加载已有数据库并继续
bash optuna/run_hpo.sh 0 20
```

### 场景4：使用最佳参数训练

```bash
# 查看最佳参数
cat optuna/best_params_teacher_ppo_hpo.txt

# 使用最佳参数进行完整训练
scripts/train_rl_teacher.sh 0 42 final_best \
    train.ppo.learning_rate=0.0025 \
    train.ppo.gamma=0.99 \
    task.env.reward.angvelClipMax=3.2 \
    # ... 其他最佳参数
```

更多示例见 [EXAMPLES.md](EXAMPLES.md)

---

## 📊 预期效果

### 优化前（默认参数）
- best_reward: ~50-70
- success_rate: ~0%
- mean_rot_angle: ~1-2 rad

### 优化后（理想情况）
- best_reward: ~100-150 ⬆️
- success_rate: ~1-5% ⬆️ (极大突破！)
- mean_rot_angle: ~2-3 rad ⬆️

**注意**：即使只有1%的成功率（0.01），综合评分也会提升10分！

---

## 🔧 关键文件

```
optuna/
├── README.md                    # 本文件 - 快速指南
├── EXAMPLES.md                  # 使用示例
├── README_HPO.md                # 完整文档
├── IMPLEMENTATION_SUMMARY.md    # 技术总结
├── RL_TEACHER_PARAMETERS.md     # 参数说明（原有）
│
├── tune_teacher.py              # 主优化脚本 ⭐
├── run_hpo.sh                   # 便捷启动脚本
└── test_hpo.sh                  # 测试脚本
```

---

## ⏱️ 时间估算

| 配置 | 单次试验 | 总时间（50次） | 推荐场景 |
|------|----------|----------------|----------|
| 50M步 | 30-60分钟 | 1-2天 | 快速探索 |
| 100M步 | 1-2小时 | 2-4天 | 标准优化 |
| 200M步 | 2-3小时 | 4-6天 | 精细优化 |

**加速技巧**：使用多GPU并行可以成倍减少总时间！

---

## ❓ 常见问题

### Q1: 优化需要多久？
A: 使用单GPU约2-6天，使用3个GPU并行约1-2天。

### Q2: 如何判断优化效果？
A: 主要看三个指标：
1. 综合评分（composite_score）- Optuna优化目标
2. 最佳奖励（best_reward）- 主要性能
3. 成功率（success_rate）- 关键突破

### Q3: 可以中途停止吗？
A: 可以！按Ctrl+C停止，数据自动保存到数据库，可随时继续。

### Q4: 没有成功案例怎么办？
A: 正常现象！继续优化，关注reward的提升。即使最终没有成功案例，优化后的参数也会让性能更好。

更多问题见 [EXAMPLES.md](EXAMPLES.md#常见问题)

---

## 📈 监控建议

### 实时监控

```bash
# GPU使用情况
watch -n 1 nvidia-smi

# 训练日志
tail -f outputs/LinkerHandHora/optuna_trial_*/training.log

# TensorBoard
tensorboard --logdir outputs/LinkerHandHora/optuna_trial_* --port 6006
```

### 关键指标

在TensorBoard中重点关注：
- `success_rate/step` - 成功率趋势 ⭐ 最重要
- `episode_rewards/step` - 奖励趋势
- `total_rot_angle(rad)/step` - 旋转角度趋势

---

## 🎯 推荐工作流程

### 第一阶段：快速探索（1-2天）
```bash
bash optuna/run_hpo.sh 0 50 50000000
```
目标：找到有希望的参数区域

### 第二阶段：精细优化（2-3天）
```bash
bash optuna/run_hpo.sh 0 30 200000000
```
目标：在有希望的区域深入搜索

### 第三阶段：最终训练（3-5天）
```bash
scripts/train_rl_teacher.sh 0 42 final_best \
    train.ppo.max_agent_steps=500000000 \
    # ... 使用最佳参数
```
目标：用最佳参数进行完整训练

---

## 🛠️ 依赖安装

```bash
# 核心依赖
pip install optuna

# 可视化（可选但推荐）
pip install plotly kaleido optuna-dashboard
```

---

## 📝 引用

如果使用了本优化系统，请引用：
- 原项目文档
- Optuna框架: https://optuna.org/

---

## 📧 支持

遇到问题？
1. 查看 [EXAMPLES.md](EXAMPLES.md) 中的常见问题
2. 查看 [README_HPO.md](README_HPO.md) 中的故障排除
3. 查看 [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) 了解技术细节

---

## 🎉 开始优化

准备好了吗？运行：

```bash
# 测试功能
bash optuna/test_hpo.sh

# 正式开始
bash optuna/run_hpo.sh 0 50
```

祝优化成功！🚀
