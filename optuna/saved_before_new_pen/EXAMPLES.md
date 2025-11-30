# 超参数优化使用示例

## 快速开始

### 1. 基础使用（推荐新手）

```bash
# 使用便捷脚本启动优化（GPU 0，50次试验，每次100M步）
bash optuna/run_hpo.sh 0
```

### 2. 自定义配置

```bash
# 自定义试验次数和训练步数
bash optuna/run_hpo.sh 0 30 200000000

# 或直接调用Python脚本
python optuna/tune_teacher.py \
    --gpu 0 \
    --n_trials 30 \
    --max_steps 200000000
```

### 3. 继续之前的优化

```bash
# 会自动加载已有的数据库并继续优化
bash optuna/run_hpo.sh 0 20
```

## 多GPU并行优化（加速）

在不同终端同时运行多个优化进程：

```bash
# 终端1 - GPU 0
bash optuna/run_hpo.sh 0 25

# 终端2 - GPU 1  
bash optuna/run_hpo.sh 1 25

# 终端3 - GPU 2
bash optuna/run_hpo.sh 2 25
```

这样可以同时进行75次试验，大大加快优化速度！

## 查看结果

### 1. 查看最佳参数

```bash
cat optuna/best_params_teacher_ppo_hpo.txt
```

输出示例：
```
Best Trial Number: 23
Best Composite Score: 145.67

Best Hyperparameters:
  learning_rate: 0.0025
  weight_decay: 0.0001
  gamma: 0.99
  tau: 0.95
  angvelClipMax: 3.2
  angvelPenaltyThres: 3.8
  ...
```

### 2. 可视化分析

```bash
# 在浏览器中打开生成的HTML文件
firefox optuna/param_importances_teacher_ppo_hpo.html
firefox optuna/optimization_history_teacher_ppo_hpo.html
```

### 3. TensorBoard对比

```bash
# 查看所有试验的训练曲线
tensorboard --logdir outputs/LinkerHandHora/optuna_trial_* --port 6006
```

然后访问 http://localhost:6006

### 4. 使用Optuna Dashboard（可选，需要额外安装）

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///optuna/hpo_teacher.db
```

访问 http://localhost:8080 获得交互式界面。

## 使用最佳参数进行完整训练

找到最佳参数后，进行完整训练（500M步）：

```bash
# 方法1：通过命令行覆盖参数
scripts/train_rl_teacher.sh 0 42 final_best_hpo \
    train.ppo.learning_rate=0.0025 \
    train.ppo.weight_decay=0.0001 \
    train.ppo.gamma=0.99 \
    train.ppo.tau=0.95 \
    task.env.reward.angvelClipMax=3.2 \
    task.env.reward.angvelPenaltyThres=3.8
    # ... 添加其他最佳参数
```

```bash
# 方法2：直接修改配置文件
# 编辑 configs/train/LinkerHandHora.yaml 和 configs/task/LinkerHandHora.yaml
# 将最佳参数写入配置文件，然后运行：
scripts/train_rl_teacher.sh 0 42 final_best_hpo
```

## 优化策略建议

### 阶段1：快速探索（1-2天）

```bash
# 使用较少的训练步数快速试验多个参数组合
bash optuna/run_hpo.sh 0 50 50000000   # 50次试验，每次50M步
```

### 阶段2：精细优化（2-3天）

```bash
# 增加训练步数，获得更准确的评估
bash optuna/run_hpo.sh 0 30 200000000  # 30次试验，每次200M步
```

### 阶段3：最终验证（3-5天）

```bash
# 使用最佳参数进行完整训练
scripts/train_rl_teacher.sh 0 42 final_best train.ppo.max_agent_steps=500000000
```

## 实时监控

### 监控GPU使用

```bash
watch -n 1 nvidia-smi
```

### 查看训练日志

```bash
# 查看最新trial的日志
tail -f outputs/LinkerHandHora/optuna_trial_*/training.log
```

### 监控成功率

```bash
# 查看TensorBoard中的 success_rate/step 指标
# 这是最关键的指标（旋转角度>10的比例）
```

## 常见问题

### Q1: 如何暂停和恢复优化？

按 `Ctrl+C` 停止，数据会自动保存到数据库。再次运行相同命令即可继续。

### Q2: 如何删除失败的试验？

如果有失败的试验导致无法查看最佳结果，可以清理它们：

```bash
# 查看失败的试验（不删除）
python optuna/clean_failed_trials.py --dry_run

# 删除失败的试验
python optuna/clean_failed_trials.py

# 或者重新开始一个新的study
python optuna/tune_teacher.py --study_name new_study_v2 ...
```

### Q3: 优化需要多长时间？

- 单次试验：1-3小时（取决于max_steps）
- 50次试验：2-6天（取决于GPU数量和并行度）
- 建议使用多GPU并行加速

### Q4: 如何判断优化效果？

主要看三个指标：
1. **综合评分 (composite_score)**：Optuna优化的目标
2. **最佳奖励 (best_reward)**：主要性能指标
3. **成功率 (success_rate)**：旋转>4弧度的比例（关键！）

### Q5: 参数搜索范围不合适怎么办？

修改 `optuna/tune_teacher.py` 中对应的参数范围：

```python
# 例如调整学习率范围
lr = trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True)

# 或调整离散参数的选项
gamma = trial.suggest_categorical("gamma", [0.97, 0.98, 0.99, 0.995])
```

## 预期结果

根据优化目标，期望看到：

1. **奖励提升**：best_reward 从 ~50 提升到 ~100+
2. **成功案例出现**：success_rate 从 0 提升到 >0（哪怕是0.01也是巨大进步！）
3. **训练稳定性**：训练曲线更平滑，不发散
4. **旋转角度增加**：mean_rot_angle 持续增长

## 进阶技巧

### 1. 自定义剪枝策略

在 `tune_teacher.py` 中修改pruner配置：

```python
pruner = MedianPruner(
    n_startup_trials=10,  # 前10次试验不剪枝
    n_warmup_steps=20,    # 每次试验前20个epoch不剪枝
)
```

### 2. 条件参数搜索

某些参数可能相互依赖，可以添加条件逻辑：

```python
if trial.suggest_categorical("use_larger_batch", [True, False]):
    minibatch_size = 32768
else:
    minibatch_size = 16384
```

### 3. 多目标优化

如果想同时优化多个目标（如reward和success_rate的平衡）：

```python
# 修改 create_study
study = optuna.create_study(
    directions=["maximize", "maximize"],  # 两个目标
    ...
)

# 修改 objective 返回值
return best_reward, success_rate
```

## 结果分享

优化完成后，可以分享：
- 最佳参数配置文件
- TensorBoard训练曲线截图
- 成功案例的视频/GIF（如果有）

祝优化顺利！🚀
