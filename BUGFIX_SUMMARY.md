# 问题修复总结

## 修复内容

### 1. 隐藏所有 Warnings ✅

**问题**：终端输出大量警告信息
- `Gym has been unmaintained` 警告
- `version_base parameter` 警告  
- `Future Hydra versions` 警告
- `torch.load weights_only` 警告

**解决方案**：
- 在 `train.py` 开头添加 warnings 过滤
- 为 `@hydra.main` 添加 `version_base='1.1'` 参数
- 所有 `torch.load()` 添加 `weights_only=False` 参数

**修改文件**：
- `train.py`
- `penspin/algo/ppo/ppo_rl_teacher.py`
- `penspin/algo/ppo/ppo_rl_bc_student.py`
- `penspin/algo/ppo/ppo_rl_bc_teacher.py`

### 2. 修复按键无响应问题 ✅

**问题根源**：
1. 按键事件查询在 `env.step()` **之前**，而 IsaacGym 需要在 step 之后才能正确捕获事件
2. 'V' 键已在 `vec_task.py` 中注册为 `toggle_viewer_sync`，重复注册会冲突
3. 窗口需要获得焦点才能接收按键事件

**解决方案**：
- 将按键事件查询移到 `env.step()` **之后**
- 'F' 键注册为 `toggle_fast`，'V' 键复用 `vec_task.py` 的 `toggle_viewer_sync`
- 添加提示信息告知用户需要点击窗口获取焦点
- 暂停时使用 `gym.sync_frame_time()` 保持实时渲染

### 3. Alpha 视频标签不匹配 ✅ (2024-01)

**问题**：轨迹视频标签显示的是更新后的 alpha，而非录制时的 alpha

**根因**：先调用 `time_warper.update()` 更新 alpha，再导出视频，导致标签使用了新 alpha

**解决方案**：
- 在调用 `update()` 前保存 `alpha_before_update = self.env.time_warper.current_alpha`
- 导出视频时使用 `alpha_before_update` 作为标签

**修改文件**：
- `penspin/algo/ppo/ppo_rl_teacher.py`

### 4. 固定录制阈值问题 ✅ (2024-01)

**问题**：低 alpha 阶段几乎无法录制到成功轨迹

**根因**：录制成功阈值固定为 3.0 rad，而 alpha=0.1 时真实阈值仅 1.0 rad

**解决方案**：
- 动态阈值 = 基础阈值 × alpha × 宽松系数(0.6)
- 例如：alpha=0.1 时阈值为 0.6 rad

**修改文件**：
- `penspin/algo/ppo/ppo_rl_teacher.py`

### 5. Alpha 起始值过低 ✅ (2024-01)

**问题**：alpha=0.1 时重力太弱 (0.01g)，物体几乎悬浮

**解决方案**：
- 默认 alpha_start 从 0.1 改为 0.3
- alpha=0.3 时重力约 0.09g，仍有显著缓冲但物理更真实

**修改文件**：
- `penspin/utils/time_warping.py`
- `scripts/TP_train_rl_teacher_spaceE.sh`
- `optuna/run_hpo.sh`

### 6. 低 Alpha 阶段"分母消失"问题 ✅ (2024-01)

**问题描述**：
在低 α 阶段 (α ≤ 0.3) 时，由于重力极低 (g' = α²g ≈ 0.09g)，episode 存活时间极长。
这导致一个 Epoch (4096 steps) 内完成的 episode 数量极少（可能只有 6 个左右）。

当用这么少的样本计算 survival_rate 时：
- `survival_rate = 1 - fall_count / total_episodes`
- 如果 6 个 episode 中 0 个 fall：survival_rate = 100%（但实际只是样本太少）
- 如果 6 个 episode 中 1 个 fall：survival_rate ≈ 83%（一次偶然事件导致巨大波动）

这种"分母消失"问题导致 curriculum 门控判断失真。

**解决方案 - 累积统计 (Accumulated Statistics)**：

1. **引入累积缓冲区**：
```python
self.accumulated_stats = {
    'total_episodes': 0,
    'success_count': 0,
    'fail_count': 0,
    'rot_angle_sum': 0.0,
    'reward_sum': 0.0,
}
self.min_episodes_for_curriculum = 500
```

2. **每个 Epoch 累加数据**：
```python
self.accumulated_stats['total_episodes'] += self.total_episodes
self.accumulated_stats['fail_count'] += self.fall_count
# ... 等
```

3. **样本量足够时才判断**：
```python
if accumulated_episodes >= self.min_episodes_for_curriculum:
    # 使用累积数据计算可靠的 survival_rate
    survival_rate = 1.0 - (fail_count / accumulated_episodes)
else:
    # 标记样本不足，跳过门控判断
    metrics['_insufficient_samples'] = True
```

4. **time_warping.py 检查样本量标记**：
```python
if metrics.get('_insufficient_samples', False):
    return False  # 跳过本次更新
```

5. **curriculum 更新后清空累积缓冲区**：
```python
if needs_update:
    self.accumulated_stats = {...}  # 清空，开始新阶段统计
```

**修改文件**：
- `penspin/algo/ppo/ppo_rl_teacher.py`
- `penspin/utils/time_warping.py`

**效果**：
- 低 α 阶段需要累积约 500 个 episode 才会判断是否进入下一阶段
- survival_rate 计算基于更大样本量，更加稳定可靠
- 避免因样本稀疏导致的误判（过早/过晚进入下一阶段）

## 使用说明

### 运行可视化

```bash
scripts/visualize.sh outputs/pose3_50k_cfg2/stage1_nn > contacts/pose3_50k_cfg2.txt
```

### 控制台输出（干净无警告）

```
[交互控制] 已激活:
  按 'V' 或 'F' 切换垂直同步 (关闭后可加速渲染)
  按 'P' 暂停/继续仿真
  注意：需要确保窗口获得焦点（点击窗口）
```

### 按键操作

**重要**：使用前需要**点击 IsaacGym 窗口**使其获得焦点！

- **V 键或 F 键**：切换垂直同步
  - 按下后显示：`[垂直同步] 关闭 (无限制加速)`
  
- **P 键**：暂停/继续
  - 按下后显示：`[仿真状态] 暂停`

## 技术细节

### 为什么按键事件要在 step() 之后查询？

IsaacGym 的事件系统在每次 `simulate()` 和 `render()` 调用后才更新事件队列。因为 `env.step()` 内部会调用这些方法，所以必须在 step 之后查询事件才能获取到最新的按键状态。

### 代码结构

```python
while True:
    # 暂停检查（在step前）
    if is_paused:
        # 在暂停时也检查事件，以便恢复
        events = gym.query_viewer_action_events(viewer)
        ...
        continue
    
    # 执行仿真步骤
    obs_dict = env.step(actions)
    
    # 在step后查询按键事件（关键！）
    events = gym.query_viewer_action_events(viewer)
    for evt in events:
        # 处理按键...
```

## 测试验证

1. 运行脚本后**不再有警告输出** ✅
2. **点击窗口**后按 'F' 或 'V' 键可以切换速度 ✅
3. 按 'P' 键可以暂停/继续 ✅
4. 控制台会实时显示状态变化 ✅
