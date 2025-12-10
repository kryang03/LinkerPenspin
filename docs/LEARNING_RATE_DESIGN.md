# Learning Rate 设计文档

本文档详细说明 PPOTeacher 中学习率调度的设计，包括参数传入、调度器、上下限等。

**最后更新**: 2025-01-XX

> **重要更新 (v2.0)**: 移除了全局余弦退火调度器 (CosineAnnealingLR)，改为完全依赖 KL 自适应调度。
> 这一变更是为了解决 Space E 课程学习中的 LR 冻结问题。

---

## 1. 参数传入

### 1.1 配置文件位置

**主配置文件**: `configs/train/LinkerHandHora.yaml`

```yaml
ppo:
  learning_rate: 5e-3      # 初始学习率 (default: 0.005)
  kl_threshold: 0.02       # KL 散度阈值
  weight_decay: 1e-4       # 权重衰减
```

### 1.2 代码中的读取

**文件**: `penspin/algo/ppo/ppo_rl_teacher.py`

```python
# Line 132-136
self.last_lr = float(self.ppo_config['learning_rate'])  # 读取初始 LR
self.weight_decay = self.ppo_config.get('weight_decay', 0.0)
self.optimizer = torch.optim.AdamW(
    self.model.parameters(), 
    self.last_lr, 
    weight_decay=self.weight_decay
)
```

---

## 2. 学习率调度架构 (v2.0 - 纯 KL 自适应)

### 2.1 设计变更历史

| 版本 | 架构 | 问题 |
|------|------|------|
| v1.0 | 双调度器 (CosineAnnealing + KL) | 课程学习时 LR 冻结在低值 |
| **v2.0** | **纯 KL 自适应** | 当前版本，解决了冻结问题 |

### 2.2 为什么移除 CosineAnnealingLR？

在 Space E 课程学习中，训练可能持续数十亿步。CosineAnnealingLR 的问题：

1. **与课程进度不同步**: Cosine 衰减基于绝对步数，不关心课程阶段
2. **LR 冻结**: 当 alpha 从 0.3→1.0 增长时，策略需要重新学习，但 LR 已衰减到 1e-6
3. **无法恢复**: 即使 KL 散度很低（策略稳定），LR 也无法回升

**典型症状** (训练日志):
```
[Epoch 50000] alpha=0.3, reward=8.5, kl=0.002, lr=1e-6  ← LR 冻结
[Epoch 50001] alpha=0.35, reward=6.2, kl=0.025, lr=1e-6 ← alpha 变化但 LR 无法响应
```

### 2.3 当前架构: 纯 KL 自适应调度器

**初始化** (Line 187-191):
```python
self.kl_threshold = self.ppo_config['kl_threshold']  # 0.02
self.scheduler = AdaptiveScheduler(
    kl_threshold=self.kl_threshold,
    min_lr=1e-6,           # 下限
    max_lr=self.last_lr    # 固定上限 (5e-3)，不再被 Cosine 约束
)
```

**关键变化**: `max_lr` 现在是**固定的**，不再随训练进度下降。

---

## 3. AdaptiveScheduler 详解

### 3.1 调整规则

```python
class AdaptiveScheduler:
    def update(self, current_lr, kl_dist):
        lr = current_lr
        
        # KL 散度过大 → 降低学习率 (策略更新过激)
        if kl_dist > (2.0 * self.kl_threshold):  # > 0.04
            lr = max(current_lr / 1.5, self.min_lr)
        
        # KL 散度过小 → 提高学习率 (策略更新过保守)
        if kl_dist < (0.5 * self.kl_threshold):  # < 0.01
            lr = min(current_lr * 1.5, self.max_lr)
        
        return lr
```

### 3.2 KL 阈值行为

| KL 散度 | 行为 | 说明 |
|---------|------|------|
| `> 2×threshold` (> 0.04) | LR ÷ 1.5 | 策略变化过大，需要减速 |
| `0.5~2×threshold` (0.01~0.04) | 保持不变 | 策略更新正常 |
| `< 0.5×threshold` (< 0.01) | LR × 1.5 | 策略过于保守，可以加速 |

### 3.3 课程学习中的 LR 动态

```
LR
│
5e-3 ┤ ●───────────────────────────────────────────────────● max_lr (固定)
     │  \           ↗ alpha 增加，KL 变大，LR 自动下降
     │   \         /
     │    \       /     ↗ 策略适应后，KL 变小，LR 回升
     │     \     /     /
     │      \   /     /
     │       ↘ /     /
     │        ●─────●   [KL 自适应区间]
     │         \   /
     │          ↘ /
1e-6 ┤           ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ min_lr (硬下限)
     └────────────────────────────────────────────────────────→
                                epoch
```

**关键优势**: LR 可以在 `[1e-6, 5e-3]` 全范围内根据训练需求自由调整。

---

## 4. 学习率上下限总结

| 边界 | 值 | 来源 | 说明 |
|------|-----|------|------|
| **初始 LR** | `5e-3` | `learning_rate` 配置 | 训练开始时的学习率 |
| **硬下限** | `1e-6` | `AdaptiveScheduler.min_lr` | 不可突破的最低学习率 |
| **固定上限** | `5e-3` | `AdaptiveScheduler.max_lr` | 允许的最高学习率 |

---

## 5. TensorBoard 监控

**Scalar 标签**: `info/last_lr`

**代码位置** (Line 348):
```python
self.writer.add_scalar('info/last_lr', self.last_lr, self.agent_steps)
```

**预期曲线特征** (v2.0):
- LR 在 `[1e-6, 5e-3]` 范围内波动
- 课程阶段切换时可能看到 LR 先降后升
- 不再有平滑的余弦下降趋势

---

## 6. HPO 搜索范围

**文件**: `optuna/tune_teacher.py` (Line 111-112)

```python
lr = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
hpo_overrides.append(f"train.ppo.learning_rate={lr}")
```

| 参数 | 下界 | 上界 | 缩放 |
|------|------|------|------|
| `learning_rate` | `1e-4` | `1e-3` | 对数 |

**已找到的最佳值** (`optuna/best_params_TP_V1.txt`):
```
learning_rate: 0.0004988620204062084  # ≈ 5e-4
```

---

## 7. 相关代码位置索引

| 功能 | 文件 | 行号 |
|------|------|------|
| LR 配置读取 | `ppo_rl_teacher.py` | 132-136 |
| ~~全局调度器初始化~~ | ~~`ppo_rl_teacher.py`~~ | ~~138-143~~ (已移除) |
| KL 调度器初始化 | `ppo_rl_teacher.py` | 187-191 |
| 调度器更新逻辑 | `ppo_rl_teacher.py` | ~1029 |
| AdaptiveScheduler 类 | `ppo_rl_teacher.py` | 1235-1257 |
| TensorBoard 记录 | `ppo_rl_teacher.py` | 348 |
| 训练配置默认值 | `configs/train/LinkerHandHora.yaml` | 29-31 |

---

## 8. 历史变更记录

### v2.0 (2025-01)
- **移除** `CosineAnnealingLR` 全局调度器
- **原因**: 与 Space E 课程学习不兼容，导致 LR 在课程切换时无法响应
- **影响**: LR 现在完全由 KL 自适应调度器控制

### v1.0 (2025-12)
- 双调度器架构 (CosineAnnealing + KL Adaptive)

---

*文档版本: v2.0*
