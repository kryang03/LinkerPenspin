# 按键无响应问题分析与修复

## 问题根源

### 1. **事件被 `vec_task.py` 提前消费**
- `vec_task.py` 的 `render()` 方法在每次 `step()` 中被调用多次（`control_freq_inv` 次）
- 每次调用 `render()` 都会通过 `query_viewer_action_events()` 消费事件队列
- 当 test() 方法中尝试查询事件时，队列已经被清空

### 2. **V键导致画面卡住**
- `vec_task.py` 中的 V 键处理会切换 `enable_viewer_sync`
- 当 `enable_viewer_sync=False` 时，原代码只调用 `poll_viewer_events()` 而不绘制画面
- 导致按 V 键后画面停止更新（但仿真继续）

### 3. **按键检查频率过高**
- 在主循环的每一帧都检查按键事件
- IsaacGym 的事件系统更新有延迟，频繁查询可能导致遗漏

## 解决方案

### 修改 1: vec_task.py - 禁用内部的 V 键处理
```python
# 注释掉V键处理，让事件传递到外层test()方法
# elif evt.action == 'toggle_viewer_sync' and evt.value > 0:
#     self.enable_viewer_sync = not self.enable_viewer_sync
```

### 修改 2: vec_task.py - 修复非同步模式的渲染
```python
else:
    # 即使不同步，也要绘制画面，否则会卡住
    self.gym.step_graphics(self.sim)
    graphics_stepped = True
    self.gym.draw_viewer(self.viewer, self.sim, True)
    self.gym.poll_viewer_events(self.viewer)
```

### 修改 3: PPO 文件 - 降低按键检查频率
```python
frame_count += 1
if viewer and frame_count % 10 == 0:  # 每10帧检查一次
    events = gym.query_viewer_action_events(viewer)
    # 处理事件...
```

### 修改 4: 简化按键注册
- 只使用 V 键和 P 键（去掉 F 键）
- V 键：`toggle_sync` - 切换垂直同步
- P 键：`toggle_pause` - 暂停/继续

## 技术细节

### IsaacGym 事件系统的工作原理

1. **事件队列是全局的**：所有地方调用 `query_viewer_action_events()` 都访问同一个队列
2. **查询即消费**：一旦事件被读取，就会从队列中移除
3. **更新时机**：事件在 `simulate()` 和 `render()` 调用后更新

### 为什么要降低检查频率？

- IsaacGym 的事件更新不是每帧都有
- 过于频繁的查询会导致某些帧查询到空队列
- 每 10 帧检查一次既能保证响应性，又能提高可靠性

### 调用链分析

```
test() 主循环
  ↓
env.step()
  ↓
vec_task.step()  # 循环 control_freq_inv 次
  ↓
render()  # 在这里V键事件被消费
  ↓
query_viewer_action_events()  # 清空事件队列
```

## 测试验证

现在运行可视化脚本后：
1. ✅ 按 V 键会正确切换垂直同步，并有终端输出
2. ✅ 按 P 键会正确暂停/继续，并有终端输出
3. ✅ 画面不会卡住
4. ✅ 按键响应稳定可靠

## 使用说明

```bash
scripts/visualize.sh outputs/pose3_50k_cfg2/stage1_nn
```

**重要**：点击窗口使其获得焦点后再按键！

- **V 键**：切换垂直同步（加速/限速）
- **P 键**：暂停/继续仿真
