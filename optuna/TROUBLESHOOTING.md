# HPO 故障排查指南

## 问题1: "Foundation object exists already" / 段错误（核心已转储）

### 症状
```
Foundation object exists already. Only one instance per process can be created.
Segmentation fault (core dumped)
```

在运行第二个试验时出现，退出码139。

### 原因
Isaac Gym/PhysX Foundation 是全局单例对象，不能在同一进程中重复创建。多次试验在同一进程中会导致冲突。

### 解决方案
✅ **已自动修复**：现在每个试验都在独立的子进程中运行，完全避免了这个问题。

如果问题仍然存在：
1. 确保使用最新版本的 `optuna/tune_teacher.py`
2. 尝试减少并行GPU数量（每个GPU一个进程）
3. 手动运行单个试验测试：
```bash
python train.py task=LinkerHandHora headless=True train.algo=PPOTeacher
```

---

## 问题2: "Primary config module 'configs' not found"

### 症状
```
Primary config module 'configs' not found.
Check that it's correct and contains an __init__.py file
```

### 原因
Hydra需要configs目录包含`__init__.py`文件才能识别为Python模块。

### 解决方案
已自动创建 `configs/__init__.py` 文件。如果问题仍然存在：

```bash
# 1. 验证文件存在
ls -la configs/__init__.py

# 2. 如果不存在，手动创建
touch configs/__init__.py

# 3. 测试Hydra配置
python test_hydra_config.py
```

---

## 问题2: "ValueError: Record does not exist"

### 症状
```
ValueError: Record does not exist.
```
在尝试访问`study.best_value`时出现。

### 原因
数据库中有失败的试验，没有成功完成的试验记录。

### 解决方案

**选项1: 清理失败的试验**
```bash
# 查看失败的试验
python optuna/clean_failed_trials.py --dry_run

# 删除失败的试验
python optuna/clean_failed_trials.py
```

**选项2: 使用新的study名称**
```bash
python optuna/tune_teacher.py --study_name teacher_ppo_hpo_v2 --gpu 0 --n_trials 30
```

**选项3: 删除整个数据库重新开始**
```bash
rm optuna/hpo_teacher.db
bash optuna/run_hpo.sh 0 50
```

---

## 问题3: CUDA Out of Memory

### 症状
```
RuntimeError: CUDA out of memory
```

### 解决方案
减小batch相关参数：

```python
# 在tune_teacher.py中修改搜索范围
minibatch_size = trial.suggest_categorical("minibatch_size", [4096, 8192])  # 减小最大值
```

或者在configs中永久修改：
```yaml
# configs/task/LinkerHandHora.yaml
env:
  numEnvs: 4096  # 从8192减少到4096
```

---

## 问题4: 训练过程中发散

### 症状
- reward快速下降到负值
- loss变为NaN
- 训练不稳定

### 解决方案
调整学习率和梯度裁剪：

```python
# 在tune_teacher.py中使用更保守的范围
lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)  # 降低上限
grad_norm = trial.suggest_categorical("grad_norm", [0.5, 1.0])    # 更强的裁剪
```

---

## 问题5: 磁盘空间不足

### 症状
```
OSError: [Errno 28] No space left on device
```

### 解决方案

**检查磁盘使用：**
```bash
df -h
du -sh outputs/LinkerHandHora/optuna_trial_*
```

**清理旧的试验：**
```bash
# 保留最近10个试验，删除其他的
ls -t outputs/LinkerHandHora/optuna_trial_* | tail -n +11 | xargs rm -rf

# 或只保留模型文件，删除TensorBoard日志
find outputs/LinkerHandHora/optuna_trial_* -name "teacher_tb" -type d -exec rm -rf {} +
```

---

## 问题6: 多GPU并行冲突

### 症状
- 数据库锁定错误
- 试验重复

### 解决方案
确保使用支持并发的数据库：

```bash
# SQLite默认支持并发读写
# 如果遇到锁定问题，可以增加超时
python optuna/tune_teacher.py --storage "sqlite:///optuna/hpo_teacher.db?timeout=60"
```

或使用PostgreSQL/MySQL（更好的并发支持）：
```bash
# 需要先安装并配置数据库
python optuna/tune_teacher.py --storage "postgresql://user:pass@localhost/optuna_db"
```

---

## 调试技巧

### 1. 启用详细日志

```bash
# 设置环境变量查看完整错误
export HYDRA_FULL_ERROR=1
python optuna/tune_teacher.py --gpu 0 --n_trials 1
```

### 2. 使用极少步数快速测试

```bash
# 只训练1M步快速验证配置
python optuna/tune_teacher.py --gpu 0 --n_trials 1 --max_steps 1000000
```

### 3. 检查GPU可用性

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### 4. 验证Hydra配置

```bash
python test_hydra_config.py
```

### 5. 手动运行单次训练

```bash
# 不使用Optuna，直接运行训练验证
python train.py task=LinkerHandHora headless=True \
    train.algo=PPOTeacher \
    train.ppo.max_agent_steps=1000000 \
    task.env.grasp_cache_name=3pose
```

---

## 获取帮助

如果问题仍未解决：

1. 查看训练日志：`tail -100 outputs/LinkerHandHora/optuna_trial_*/training.log`
2. 检查TensorBoard：`tensorboard --logdir outputs/LinkerHandHora/optuna_trial_*`
3. 查看完整错误堆栈：设置 `HYDRA_FULL_ERROR=1`
4. 检查依赖版本：`pip list | grep -E "optuna|hydra|torch"`

---

## 预防措施

### 启动前检查清单

- [ ] GPU可用且有足够显存（至少8GB）
- [ ] 磁盘空间充足（建议至少100GB可用）
- [ ] configs/__init__.py 文件存在
- [ ] 已测试Hydra配置：`python test_hydra_config.py`
- [ ] 已清理失败的旧试验（如果有）

### 推荐配置

对于首次运行，使用保守的配置：

```bash
# 较少的训练步数快速验证
bash optuna/run_hpo.sh 0 5 10000000
```

如果成功，再增加到正常配置：

```bash
bash optuna/run_hpo.sh 0 50 100000000
```
