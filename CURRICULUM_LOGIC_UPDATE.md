# Curriculum Learning 逻辑更新记录

## 更新日期
2025-01-XX

## 问题背景

### Time Horizon Bias 问题
在 Space E curriculum 学习中，当 α 较小时 (如 α=0.1)，物理时间被压缩：
- Episode 长度：400 步 × 0.005s = 2.0s（物理时间）
- 而在 α=1.0 时：400 步 × 0.005s / 0.1 = 20s（物理时间）

这导致原来的固定成功阈值 (10.0 rad) 在低 α 阶段几乎不可能达到：
- α=0.1：最大可能旋转 ≈ 2π × 2s × 1 rad/s = 12.56 rad（假设 1 Hz 旋转）
- 实际上在 α=0.1 的重力下 (g'=0.098 m/s²)，旋转更加困难

### 一致性问题
原 `time_warping.py` 在 `return False` 前更新了 `current_alpha`，导致：
- 物理参数未更新
- 但观测/奖励缩放已经使用了新的 alpha
- 造成不一致性

## 修改内容

### 1. time_warping.py

**改动：**
- 移除 `return False` 前的 `current_alpha` 更新
- 改用基于比率的更新触发：`alpha_new >= alpha_old * (1 + ratio_threshold)`
- 添加 `metrics` 参数支持自适应课程
- 添加 `ratio_threshold` 配置项（默认 5%）

**新增参数：**
```yaml
curriculum:
  ratio_threshold: 0.05  # 5% 相对变化触发物理更新
```

**update() 签名变更：**
```python
# 旧版
def update(self, global_agent_steps):
    ...

# 新版
def update(self, global_agent_steps, metrics=None):
    """
    Args:
        global_agent_steps: 当前全局 Agent 步数
        metrics: 可选的训练指标字典，包含:
            - success_rate: 成功率
            - survival_rate: 存活率
            - mean_rot_angle: 平均旋转角度
            - mean_reward: 平均奖励
            - mean_entropy: 平均熵
    """
```

### 2. ppo_rl_teacher.py

**改动：**
- 添加 survival_rate 计算（基于掉落/倾斜终止统计）
- 使用可配置的成功阈值 `success_rot_threshold`
- 支持自适应阈值：`effective_threshold = success_rot_threshold * alpha`
- 传递 metrics 到 `time_warper.update()`

**新增变量：**
```python
self.fall_count = 0  # 掉落 + 倾斜计数
self.drop_count = 0  # 仅掉落计数
self.tilt_count = 0  # 仅倾斜计数
```

**自适应阈值逻辑：**
```python
effective_threshold = self.success_rot_threshold
if self.use_adaptive_threshold and hasattr(self.env, 'time_warper'):
    effective_threshold = self.success_rot_threshold * self.env.time_warper.current_alpha
```

### 3. linker_hand_hora.py

**改动：**
- 添加 `termination_reason` 到 extras 输出
- 支持 Gaussian kernel 角速度奖励模式

**Gaussian Kernel 角速度奖励：**
```python
if self.use_gaussian_angvel_reward:
    # r = exp(-||ω - ω_target||² / σ²)
    angvel_error = vec_dot - self.target_angvel
    rotate_reward = torch.exp(-(angvel_error ** 2) / (self.angvel_sigma ** 2))
    # 惩罚反向旋转
    rotate_reward = torch.where(vec_dot < 0, torch.zeros_like(rotate_reward), rotate_reward)
    rotate_penalty = torch.zeros_like(rotate_reward)
else:
    # Clip-based 模式（原逻辑）
    rotate_reward = torch.clip(vec_dot, max=self.angvel_clip_max, min=self.angvel_clip_min)
    ...
```

### 4. LinkerHandHora.yaml

**新增配置：**
```yaml
reward:
  # Gaussian kernel 角速度奖励
  use_gaussian_angvel_reward: False
  target_angvel: 3.14  # rad/s
  angvel_sigma: 1.0    # 高斯核带宽

curriculum:
  ratio_threshold: 0.05        # 物理更新触发阈值
  success_rot_threshold: 10.0  # 成功旋转阈值 (rad)
  use_adaptive_threshold: True # 是否使用自适应阈值
```

### 5. 训练脚本

**SpaceE 新增参数：**
```bash
CURRICULUM_RATIO_THRESHOLD=0.05
SUCCESS_ROT_THRESHOLD=10.0
USE_ADAPTIVE_THRESHOLD=True
USE_GAUSSIAN_ANGVEL_REWARD=False
TARGET_ANGVEL=3.14
ANGVEL_REWARD_SIGMA=1.0
```

### 6. Optuna tune_teacher.py

**改动：**
- 添加 `use_gaussian_angvel_reward` 作为 categorical 参数
- 支持条件参数：Gaussian 模式时调优 `target_angvel` 和 `angvel_sigma`

```python
use_gaussian_angvel = trial.suggest_categorical("use_gaussian_angvel_reward", [True, False])

if use_gaussian_angvel:
    target_angvel = trial.suggest_float("target_angvel", 1.5, 6.0)
    angvel_reward_sigma = trial.suggest_float("angvel_sigma", 0.5, 2.0)
else:
    # Clip-based 参数...
```

## 使用说明

### 启动 SpaceE 训练
```bash
# 使用自适应阈值 + Gaussian 角速度奖励
scripts/TP_train_rl_teacher_spaceE.sh 0 42 spaceE_gaussian \
    task.env.reward.use_gaussian_angvel_reward=True \
    task.env.reward.target_angvel=3.14

# 使用自适应阈值 + Clip-based 角速度奖励
scripts/TP_train_rl_teacher_spaceE.sh 0 42 spaceE_clip
```

### TensorBoard 新增指标
- `curriculum/survival_rate`: 存活率

## 待办事项

1. [ ] 实现基于 metrics 的自适应课程门控
2. [ ] 添加 survival_rate 作为低 α 阶段的主要进度指标
3. [ ] 实验对比 Gaussian vs Clip-based 角速度奖励

## 相关文件

- `penspin/utils/time_warping.py`
- `penspin/algo/ppo/ppo_rl_teacher.py`
- `penspin/tasks/linker_hand_hora.py`
- `configs/task/LinkerHandHora.yaml`
- `scripts/TP_train_rl_teacher_spaceA.sh`
- `scripts/TP_train_rl_teacher_spaceE.sh`
- `optuna/tune_teacher.py`
