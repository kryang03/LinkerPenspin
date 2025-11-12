# 快速使用指南

## 三个训练功能一览

| 功能 | 训练器 | 输入信息 | 是否需要Teacher | BC Loss | 用途 |
|------|--------|----------|----------------|---------|------|
| **功能1** | `PPOTeacher` | 特权信息+点云+本体感觉 | ❌ | ❌ | 训练仿真expert |
| **功能2** | `PPO_RL_BC_Teacher` | 特权信息+点云+本体感觉 | ✅ | ✅ | 微调Teacher策略 |
| **功能3** | `PPO_RL_BC_Student` | 仅本体感觉历史 | ✅ | ✅ | 训练可部署策略 |

## 快速开始

### 1. 训练Teacher模型（必须先执行）

```bash
# 基础命令
bash scripts/train_rl_teacher.sh <GPU_ID> <SEED> <OUTPUT_NAME>

# 示例
bash scripts/train_rl_teacher.sh 0 42 teacher_baseline
```

**输出**：`outputs/LinkerHandHora/teacher_baseline/teacher_nn/best_reward_xxx.pth`

---

### 2a. 微调Teacher（可选，使用功能2）

```bash
# 基础命令
bash scripts/train_rl_bc_teacher.sh <GPU_ID> <SEED> <OUTPUT_NAME> <TEACHER_PATH>

# 示例
bash scripts/train_rl_bc_teacher.sh 0 42 teacher_finetuned \
  outputs/LinkerHandHora/teacher_baseline/teacher_nn/best_reward_10.50.pth
```

**输出**：`outputs/LinkerHandHora/teacher_finetuned/stage1_nn/best.pth`

---

### 2b. 训练Student（用于真机部署，使用功能3）

```bash
# 基础命令
bash scripts/train_rl_bc_student.sh <GPU_ID> <SEED> <OUTPUT_NAME> <TEACHER_PATH>

# 示例
bash scripts/train_rl_bc_student.sh 0 42 student_deploy \
  outputs/LinkerHandHora/teacher_baseline/teacher_nn/best_reward_10.50.pth
```

**输出**：`outputs/LinkerHandHora/student_deploy/stage1_nn/best.pth`

## 推荐训练流程

### 方案A: 基础流程（最简单）
```
功能1 (训练Teacher) → 功能3 (训练Student)
```

### 方案B: 完整流程（如需微调）
```
功能1 (训练Teacher) → 功能2 (微调Teacher) → 功能3 (训练Student)
```

## 训练参数调整

### 常用参数覆盖

在脚本后添加额外参数：

```bash
# 调整学习率
bash scripts/train_rl_teacher.sh 0 42 test \
  train.ppo.learning_rate=1e-4

# 调整环境数量
bash scripts/train_rl_teacher.sh 0 42 test \
  task.env.numEnvs=4096

# 调整BC loss权重
bash scripts/train_rl_bc_student.sh 0 42 test <TEACHER_PATH> \
  train.ppo.bc_loss_coef=2.0

# 启用latent loss
bash scripts/train_rl_bc_student.sh 0 42 test <TEACHER_PATH> \
  train.ppo.enable_latent_loss=True \
  train.ppo.distill_loss_coef=1.0
```

## 后台训练

使用nohup在后台运行训练：

```bash
# Teacher训练
nohup bash scripts/train_rl_teacher.sh 0 42 teacher_baseline > logs/teacher.log 2>&1 &

# Student训练
nohup bash scripts/train_rl_bc_student.sh 0 42 student <TEACHER_PATH> > logs/student.log 2>&1 &

# 查看日志
tail -f logs/teacher.log
```

## 关键配置对比

### 功能1: PPOTeacher
```yaml
train.algo: PPOTeacher
train.ppo.priv_info: True
train.ppo.use_point_cloud_info: True
train.ppo.bc_loss_coef: 0.0  # 不使用BC
```

### 功能2: PPO_RL_BC_Teacher
```yaml
train.algo: PPO_RL_BC_Teacher
train.demon_path: <path>
train.ppo.priv_info: True
train.ppo.use_point_cloud_info: True
train.ppo.bc_loss_coef: 1.0  # 使用BC
```

### 功能3: PPO_RL_BC_Student
```yaml
train.algo: PPO_RL_BC_Student
train.demon_path: <path>
train.ppo.priv_info: False  # 关键区别
train.ppo.proprio_mode: True
train.ppo.use_point_cloud_info: False  # 关键区别
train.ppo.bc_loss_coef: 1.0
```

## 测试模型

```bash
# 测试Teacher
python train.py task=LinkerHandHora test=True headless=False \
  train.algo=PPOTeacher \
  train.load_path=outputs/.../best.pth

# 测试Student
python train.py task=LinkerHandHora test=True headless=False \
  train.algo=PPO_RL_BC_Student \
  train.load_path=outputs/.../best.pth
```

## TensorBoard监控

```bash
# 单个实验
tensorboard --logdir outputs/LinkerHandHora/teacher_baseline/teacher_tb

# 所有实验对比
tensorboard --logdir outputs/LinkerHandHora
```

## 常见问题速查

| 问题 | 解决方案 |
|------|----------|
| 缺少demon_path | 训练功能2/3时必须提供Teacher路径 |
| CUDA OOM | 减少`numEnvs`或`minibatch_size` |
| 找不到模型类 | 确认`train.algo`设置正确 |
| BC loss为0 | 检查`bc_loss_coef`是否非零 |
| Teacher无法加载 | 确认checkpoint包含所有必需权重 |

## 检查训练进度

```bash
# 查看输出目录
ls -lh outputs/LinkerHandHora/<experiment_name>/*/

# 查看最新checkpoint
ls -lht outputs/LinkerHandHora/<experiment_name>/*/best*.pth

# 查看训练日志（如使用nohup）
tail -f logs/<experiment>.log | grep -E "Reward|Steps|FPS"
```

## 性能优化建议

1. **GPU利用率**: 增加`numEnvs`直到GPU接近满载
2. **训练速度**: 平衡`horizon_length`和`minibatch_size`
3. **收敛速度**: 调整学习率和`bc_loss_coef`
4. **样本效率**: 使用更多`mini_epochs`

## 下一步

完成训练后，参考以下文档：
- 详细训练说明：`TRAINING_GUIDE.md`
- 真机部署：`real/README.md`（如有）
- 架构说明：`FINAL_ARCHITECTURE_REVIEW_REPORT.md`
