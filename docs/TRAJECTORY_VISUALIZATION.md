# 课程学习轨迹可视化功能

## 概述

在 Space E 课程学习过程中，每当 curriculum alpha 发生更新时，自动录制并导出成功的完整轨迹视频。这有助于：

1. **监控训练进展**：观察不同 alpha 阶段智能体的行为变化
2. **调试问题**：发现策略在特定阶段的问题
3. **生成演示视频**：自动收集高质量的成功轨迹

## 设计原理

### 核心挑战

PPO 使用 `horizon_length=32` 的滚动窗口收集经验，这意味着 ExperienceBuffer 最多只存储 32 步的数据。然而，一个完整的 episode 通常需要数百步。因此，需要独立的机制来跨 rollout 追踪完整轨迹。

### 解决方案

`TrajectoryRecorder` 类实现了以下功能：

1. **跨 rollout 追踪**：维护独立的帧缓冲区，跨多个 `play_steps()` 调用追踪同一环境的完整 episode
2. **成功过滤**：只保存旋转角度超过阈值的成功轨迹
3. **相机复用**：利用 PPO 已有的 GIF 录制帧，避免额外的渲染开销
4. **Curriculum 触发导出**：只在 `time_warper.update()` 返回 True 时导出视频

## 使用方法

### 自动集成

轨迹录制器已集成到 `PPOTeacher` 中，无需额外配置即可使用。

**触发条件**：
- 主进程 (`LOCAL_RANK=0`)
- 环境启用相机 (`enableCameraSensors: True`)
- Curriculum 更新发生 (`time_warper.update()` 返回 True)

### 输出位置

1. **TensorBoard**：视频保存到 `curriculum_trajectory/` 标签下
   - 标签格式：`curriculum_trajectory/alpha_X.XX_rank1`
   
2. **文件**：MP4 文件保存到 `outputs/<exp_name>/curriculum_videos/`
   - 文件名格式：`step_XXXXXXXX_alpha_X.XX_rank1_rotX.XX.mp4`

### 配置选项

在 `PPOTeacher.__init__` 中可以调整以下参数：

```python
self.trajectory_recorder = TrajectoryRecorder(
    env=self.env,
    num_record_envs=4,           # 追踪的环境数量
    max_episode_length=1000,     # 最大 episode 长度
    success_threshold=3.0,       # 成功判定阈值（弧度）
    min_trajectories_to_export=2 # 最小导出轨迹数
)
```

### 注意事项

1. **相机开销**：帧捕获只在 PPO 的 GIF 录制时段内进行，不会产生额外开销
2. **显存使用**：每个追踪环境最多存储 1000 帧，约消耗几百 MB 显存
3. **依赖 imageio**：保存 MP4 文件需要安装 `imageio` 库

## 代码位置

- 轨迹录制器：`penspin/utils/trajectory_recorder.py`
- PPO 集成：`penspin/algo/ppo/ppo_rl_teacher.py`
  - 初始化：`__init__` 方法末尾
  - 帧记录：`play_steps()` 中 `env.step()` 后
  - 导出触发：`train()` 中 curriculum update 后

## 示例输出

训练日志中会显示：

```
[TrajectoryRecorder] 初始化完成
  录制环境数: 4
  最大 episode 长度: 1000
  成功阈值: 3.00 rad
  最小导出轨迹数: 2

...

[TrajectoryRecorder] Curriculum Update #1 @ alpha=0.105
  导出 3 条成功轨迹
  轨迹 1: 256 帧, rot_angle=5.23 rad
  轨迹 2: 198 帧, rot_angle=4.87 rad
  轨迹 3: 312 帧, rot_angle=4.12 rad
  保存: outputs/exp/curriculum_videos/step_00000512_alpha_0.11_rank1_rot5.23.mp4
```
