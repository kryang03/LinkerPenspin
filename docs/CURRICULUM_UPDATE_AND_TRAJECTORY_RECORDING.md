# Space E Curriculum Update 与轨迹录制逻辑文档

## 概述

本文档详细说明 Space E 课程学习中的 alpha 更新机制和视频轨迹保存逻辑，包括数值估计和流程图。

**最后更新**: 2025-12-08

---

## 0. 视频录制快速参考

### 录制前提条件

要启用视频录制，**必须**设置 `enableCameraSensors=True`：

```bash
scripts/TP_train_rl_teacher_spaceE.sh 0 42 test task.env.enableCameraSensors=True
```

### 视频输出位置

| 类型 | 位置 | 说明 |
|------|------|------|
| **TensorBoard** | `outputs/<exp>/teacher_tb/` | 使用 `tensorboard --logdir=...` 查看 |
| **MP4 文件** | `outputs/<exp>/videos/` | 需要安装 `imageio` 和 `imageio-ffmpeg` |

### 两套独立的录制系统

| 系统 | 触发条件 | 输出格式 | 最大数量 |
|------|----------|----------|----------|
| **PPO GIF 录制** | 每 7500 帧录制 600 帧 | TensorBoard `rollout_gif` | 1 个视频/触发 |
| **Top-K 轨迹录制器** | 周期性(每100 epoch) + 课程更新时 | TensorBoard + MP4 | Top 3 最佳 |

---

## 1. Alpha 更新机制

### 1.1 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `alpha_start` | 0.3 | 课程起始 α 值（世界慢 3.3 倍）|
| `alpha_end` | 1.0 | 课程结束 α 值（真实世界速度）|
| `ratio_threshold` | 0.05 | 每次更新的相对增量（5%）|
| `curriculum_steps` | 1e8 | 课程参考步数（用于限速器）|

### 1.2 更新流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PPO Training Loop                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ 1. 动态调整录制阈值                                            │ │
│  │    adaptive_threshold = base_threshold × alpha × 0.6          │ │
│  │    例: alpha=0.3 时，阈值 = 10 × 0.3 × 0.6 = 1.8 rad           │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ 2. train_epoch()                                               │ │
│  │    - play_steps(): 收集数据 + 录制轨迹                         │ │
│  │    - update_model(): 更新策略                                   │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ 3. 计算训练指标                                                │ │
│  │    - success_rate: 成功率（rot_angle > threshold × alpha）     │ │
│  │    - survival_rate: 存活率（1 - 掉落率）                       │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ 4. [CRITICAL] 保存更新前的 alpha                               │ │
│  │    alpha_before_update = time_warper.current_alpha             │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ 5. time_warper.update(agent_steps, metrics)                    │ │
│  │    ┌─────────────────────────────────────────────────────────┐ │ │
│  │    │ 5a. 门控检查 (Gating)                                   │ │ │
│  │    │     - α < 0.5: survival_rate >= 0.8?                    │ │ │
│  │    │     - α >= 0.5: success_rate >= 0.7?                    │ │ │
│  │    ├─────────────────────────────────────────────────────────┤ │ │
│  │    │ 5b. 增量计算                                            │ │ │
│  │    │     increment = current_alpha × ratio_threshold         │ │ │
│  │    │     target = current_alpha + increment                  │ │ │
│  │    ├─────────────────────────────────────────────────────────┤ │ │
│  │    │ 5c. 限速器检查                                          │ │ │
│  │    │     time_limit = α_start + progress × (α_end - α_start) │ │ │
│  │    │     target = min(target, time_limit)                    │ │ │
│  │    ├─────────────────────────────────────────────────────────┤ │ │
│  │    │ 5d. 触发判定                                            │ │ │
│  │    │     relative_change >= ratio_threshold?                 │ │ │
│  │    │     → True: 返回 needs_update=True, 更新 current_alpha  │ │ │
│  │    │     → False: 返回 needs_update=False                    │ │ │
│  │    └─────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ 6. if needs_update:                                            │ │
│  │    6a. 导出轨迹视频（使用 alpha_before_update 命名）           │ │
│  │    6b. apply_curriculum_physics() - 更新物理参数               │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 Alpha 增长数值估计

假设 Agent 表现稳定（每次都通过门控），alpha 的增长轨迹如下：

| 更新次数 | Alpha 值 | 增量 | 重力缩放 (α²) | 备注 |
|----------|----------|------|---------------|------|
| 0 | 0.300 | - | 0.090 | 起始 |
| 1 | 0.315 | 0.015 | 0.099 | +5% |
| 2 | 0.331 | 0.016 | 0.110 | +5% |
| 5 | 0.383 | - | 0.147 | - |
| 10 | 0.489 | - | 0.239 | - |
| 15 | 0.624 | - | 0.389 | - |
| 20 | 0.796 | - | 0.634 | - |
| 24 | 0.966 | - | 0.933 | 接近目标 |
| 25 | 1.000 | - | 1.000 | 到达终点 |

**结论**: 从 α=0.3 到 α=1.0，理论上需要约 **25 次成功更新**。

### 1.4 时间估计

假设：
- `batch_size` = 8192 envs × 32 horizon = 262,144 steps/epoch
- 门控平均通过率 = 50%（即约每 2 个 epoch 触发一次更新）
- 限速器不生效（Agent 表现良好）

则：
- 每次更新约需 2 个 epoch = 524,288 agent steps
- 25 次更新约需 **13M agent steps**
- 实际中，高 α 阶段门控更严格，可能需要 **30-50M steps**

---

## 2. 视频录制系统详解

### 2.1 系统 A: PPO GIF 录制 (rollout_gif)

**文件**: `ppo_rl_teacher.py` 第 1079-1110 行

**设计目标**: 周期性录制训练过程的 rollout 片段，用于监控训练状态。

**参数**:
```python
self.gif_frame_counter = 0      # 帧计数器
self.gif_save_every_n = 7500    # 每 7500 帧触发一次录制
self.gif_save_length = 600      # 每次录制 600 帧
self.gif_frames = []            # 帧缓冲区 (FIFO)
```

**触发条件**:
```python
record_frame = (gif_frame_counter >= 7500) and (gif_frame_counter % 7500 < 600)
```

**输出**:
- TensorBoard Tag: `rollout_gif`
- 分辨率: 256×256（由 `_create_camera` 设置）
- FPS: 20
- 每次导出清空缓冲区，无累积

**性能影响**: 
- 录制期间启用相机传感器，会略微降低仿真速度（约 5-10%）
- 非录制期间相机自动禁用

---

### 2.2 系统 B: Top-K 轨迹录制器 (TrajectoryRecorder)

**文件**: `penspin/utils/trajectory_recorder.py`

**设计目标**: 基于成功度量（旋转角度）保留最佳轨迹，用于课程学习可视化。

#### 关键参数

```python
TrajectoryRecorder(
    env=env,
    num_record_envs=4,           # 追踪 4 个环境
    max_episode_length=1000,     # 防止内存溢出
    success_threshold=1.8,       # 动态调整的成功阈值
    min_trajectories_to_export=1,# 最少 1 条才导出
    keep_best_k=5,               # 缓冲区保留 Top-5 最佳
    device='cuda:0'
)
```

#### Top-K 优先级队列算法

```python
import heapq

# 当一个 episode 结束且成功时:
new_record = (rot_angle, env_id, frames.copy())

if len(best_trajectories) < keep_best_k:
    # 缓冲区未满，直接添加
    heapq.heappush(best_trajectories, new_record)
else:
    # 缓冲区已满，检查是否比最差的好
    if rot_angle > best_trajectories[0][0]:  # 堆顶是最小值
        heapq.heapreplace(best_trajectories, new_record)
```

**优势**: 
- 始终保留旋转角度最高的 K 条轨迹
- 避免 FIFO 模式下好轨迹被冲掉
- 导出时按 rot_angle 排序输出

#### 触发条件

| 触发类型 | 条件 | Tag 前缀 |
|----------|------|----------|
| **周期性导出** | `epoch_num % 100 == 0` | `periodic_trajectory` |
| **课程更新导出** | `time_warper.update()` 返回 `True` | `curriculum_trajectory` |

#### 输出格式

**TensorBoard**:
- Tag: `<prefix>_trajectory/step_<N>_alpha_<X.XX>_rank<1-3>`
- 示例: `periodic_trajectory/step_262144_alpha_0.30_rank1`
- 最多导出 Top-3 轨迹

**MP4 文件**:
- 路径: `outputs/<exp>/videos/`
- 命名: `<prefix>_step_<N>_alpha_<X.XX>_rank<R>_rot<Y.YY>.mp4`
- 示例: `periodic_step_000262144_alpha_0.30_rank1_rot3.14.mp4`

#### 动态成功阈值

| Alpha | 基础阈值 | 宽松系数 | 实际阈值 | 说明 |
|-------|----------|----------|----------|------|
| 0.1 | 10.0 rad | 0.6 | 0.6 rad | 极易录制（~35°旋转） |
| 0.3 | 10.0 rad | 0.6 | 1.8 rad | 较易录制（~103°旋转） |
| 0.5 | 10.0 rad | 0.6 | 3.0 rad | 中等难度（~172°旋转） |
| 1.0 | 10.0 rad | 0.6 | 6.0 rad | 完整难度（~344°旋转） |

#### 缓冲区管理

- **导出后清空**: 每次导出后 `best_trajectories = []`
- **独立周期**: 周期性导出和课程更新导出使用同一缓冲区
- **帧复用**: 复用 PPO GIF 系统捕获的帧，无额外相机开销

---

### 2.3 性能影响分析

| 操作 | CPU 影响 | GPU 影响 | 内存影响 |
|------|----------|----------|----------|
| 相机传感器渲染 | 低 | 中 (5-10%) | 低 |
| 帧存储 (4 env × 1000 帧 × 256×256×3) | - | - | ~750 MB 峰值 |
| TensorBoard 写入 | 中 (I/O) | - | 临时 |
| MP4 编码 (imageio) | 高 | - | 临时 |

**建议**:
- 长时间训练时可禁用相机以提速: `enableCameraSensors=False`
- 需要调试时再启用相机
- 周期性导出间隔（100 epoch）不会显著影响训练

---

### 2.4 录制流程图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                play_steps() 每一步                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  1. 判断是否录制帧 (PPO GIF 系统):                                               │
│     record_frame = (counter >= 7500) and (counter % 7500 < 600)                 │
│                              ↓                                                   │
│  2. if record_frame:                                                            │
│     enable_camera_sensors = True                                                │
│     captured_frame = env.capture_frame()                                        │
│     gif_frames.append(captured_frame)                                           │
│                              ↓                                                   │
│  3. 轨迹录制器追踪:                                                              │
│     trajectory_recorder.record_step(dones, infos, frame=captured_frame)         │
│     ┌───────────────────────────────────────────────────────────────────────┐   │
│     │ • 累加 rot_angle                                                       │   │
│     │ • 存储帧到 frame_buffers[env_id]                                       │   │
│     │ • if done && rot > threshold:                                          │   │
│     │     → Top-K 堆插入/替换                                                 │   │
│     │ • 重置该 env 的缓冲区                                                   │   │
│     └───────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                                   │
│  4. if len(gif_frames) == 600:                                                  │
│     writer.add_video('rollout_gif', ...)                                        │
│     gif_frames.clear()                                                          │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              epoch 结束后                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  A. 课程更新检查:                                                                │
│     if time_warper.update() == True:                                            │
│       trajectory_recorder.export_on_curriculum_update(...)                      │
│       env.apply_curriculum_physics()                                            │
│                              ↓                                                   │
│  B. 周期性导出检查:                                                              │
│     if epoch_num % 100 == 0:                                                    │
│       trajectory_recorder.export(..., tag_prefix="periodic")                    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### 2.5 故障排查

#### 问题: 没有 GIF/视频输出

**诊断步骤**:

1. 检查初始化日志:
   ```
   ================================================================================
   轨迹录制器初始化状态 (Trajectory Recorder Status)
   ================================================================================
     主进程 (LOCAL_RANK=0):  True
     相机已启用:             True/False  ← 检查这里
   ```

2. 如果相机未启用，添加参数:
   ```bash
   task.env.enableCameraSensors=True
   ```

3. 如果在无显示环境运行，使用虚拟显示:
   ```bash
   xvfb-run -s "-screen 0 1024x768x24" scripts/TP_train_rl_teacher_spaceE.sh ...
   ```

#### 问题: 有成功 episode 但没有录制到轨迹

可能原因:
- 成功阈值过高（检查 `success_threshold`）
- 帧数据缺失（`with_camera=False` 时只记录统计，不存储帧）

解决方案:
- 降低阈值或确保 `enableCameraSensors=True`

---

## 3. 门控逻辑详解

### 3.1 低 Alpha 阶段 (α < 0.5)

**门控指标**: `survival_rate >= 0.8`

**物理背景**:
- 重力缩放: g' = α² × g = 0.09g（极轻微）
- 物体几乎不会因重力掉落
- 主要考验: 策略是否能基本控制手指

**数值示例**:
- 8192 并行环境，一个 epoch 中 10000 个 episode 结束
- 其中 1500 个因掉落/倾倒终止
- survival_rate = 1 - 1500/10000 = 0.85 ✓ 通过门控

### 3.2 高 Alpha 阶段 (α >= 0.5)

**门控指标**: `success_rate >= 0.7`

**物理背景**:
- 重力逐渐接近正常，物体容易掉落
- 需要真正学会旋转技巧
- success_rate 使用自适应阈值: `rot_angle > base_threshold × alpha`

**数值示例** (α = 0.5):
- 自适应成功阈值 = 10.0 × 0.5 = 5.0 rad
- 10000 个完成的 episode 中，6800 个 rot_angle > 5.0 rad
- success_rate = 6800/10000 = 0.68 ✗ 未通过门控
- → alpha 保持不变，继续训练

---

## 4. 潜在问题与解决方案

### 4.1 已修复的 Bug

| Bug | 问题描述 | 解决方案 |
|-----|----------|----------|
| Alpha 标签错位 | 视频内容是旧 α，但标签是新 α | 在 update() 前保存 `alpha_before_update` |
| 录制阈值固定 | 低 α 时阈值过高导致无法录制 | 动态调整: `threshold × alpha × 0.6` |
| 起始值过低 | α=0.1 训练太慢 | 改为 α=0.3 开始 |

### 4.2 建议的优化（未实施）

1. **移除时间限速器**: 当前代码中 `time_based_limit` 可能阻碍高效训练
2. **最小 episode 数量检查**: 防止少量 episode 触发错误的门控判断
3. **平滑的门控阈值**: 在 α=0.5 附近使用线性插值而非硬切换

---

## 5. 配置参数一览

### 5.1 Time Warping 配置 (task.env.curriculum)

```yaml
curriculum:
  mode: 'SpaceE'           # SpaceA / SpaceD / SpaceE
  alpha_start: 0.3         # 起始 α
  alpha_end: 1.0           # 结束 α  
  curriculum_steps: 1e8    # 课程参考步数
  ratio_threshold: 0.05    # 每次更新的相对增量
  success_rot_threshold: 10.0  # 成功旋转阈值（弧度）
  use_adaptive_threshold: True  # 是否使用自适应阈值
```

### 5.2 轨迹录制配置 (在 PPO 初始化时)

```python
TrajectoryRecorder(
    env=env,
    num_record_envs=4,           # 追踪的环境数
    max_episode_length=1000,     # 最大帧数
    success_threshold=3.0,       # 初始阈值（会被动态覆盖）
    min_trajectories_to_export=2 # 最少导出轨迹数
)
```

---

## 6. 调试与验证

### 6.1 TensorBoard 监控

关键 Scalar:
- `curriculum/alpha`: 当前 α 值
- `curriculum/gravity_scale`: 重力缩放 (α²)
- `curriculum/survival_rate`: 存活率
- `curriculum/progress`: 时间进度

关键 Video:
- `curriculum_trajectory/alpha_X.XX_rankN`: 课程更新时的最佳轨迹

### 6.2 日志输出

每次 curriculum update 时会打印:
```
[TrajectoryRecorder] Curriculum Update #5 @ alpha=0.315
  导出 3 条成功轨迹
  轨迹 1: 420 帧, rot_angle=4.25 rad
  轨迹 2: 380 帧, rot_angle=3.87 rad
  轨迹 3: 350 帧, rot_angle=3.45 rad
  保存: outputs/.../curriculum_videos/step_00262144_alpha_0.30_rank1_rot4.25.mp4
```

---

*文档更新时间: 2025-12-08*
*版本: v2.0 - 新增 Top-K 轨迹录制器和双录制系统说明*
