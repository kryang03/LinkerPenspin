# IsaacGym 可视化倍速功能说明

## 问题原因

之前在 `ppo_rl_teacher.py` 中的改动没有生效，根本原因是：

1. **IsaacGym 的渲染机制**：在 `vec_task.py` 的 `render()` 方法中，当 `enable_viewer_sync=True` 时，会调用 `gym.sync_frame_time(self.sim)`，这会**强制同步到实时时间**，限制了渲染速度。

2. **代码位置**：真正控制渲染速度的代码在 `/penspin/tasks/base/vec_task.py` 的第 354-358 行：
   ```python
   if self.enable_viewer_sync:
       self.gym.step_graphics(self.sim)
       graphics_stepped = True
       self.gym.draw_viewer(self.viewer, self.sim, True)
       self.gym.sync_frame_time(self.sim)  # 这里会同步到实时时间
   ```

3. **之前的尝试**：在 `ppo_rl_teacher.py` 中尝试修改一个局部变量 `enable_sync` 并没有修改到 `self.env.enable_viewer_sync`。

## 解决方案

直接修改 `self.env.enable_viewer_sync` 属性，通过关闭垂直同步来实现加速渲染。

### 修改的文件

1. `/penspin/algo/ppo/ppo_rl_teacher.py`
2. `/penspin/algo/ppo/ppo_rl_bc_student.py`
3. `/penspin/algo/ppo/ppo_rl_bc_teacher.py`

### 使用方法

运行可视化脚本后（例如 `scripts/visualize.sh`），在 IsaacGym 窗口中：

- **按 'V' 或 'F' 键**：切换垂直同步
  - 开启（默认）：渲染速度限制为实时
  - 关闭：无限制加速，渲染速度取决于GPU性能
  
- **按 'P' 键**：暂停/继续仿真
  - 暂停时只显示画面，不进行物理模拟

### 控制台提示

启动可视化时会看到：
```
[交互控制] 已激活:
  按 'V' 或 'F' 切换垂直同步 (关闭后可加速渲染)
  按 'P' 暂停/继续仿真
```

按键后会实时反馈：
```
[垂直同步] 关闭 (无限制加速)
[仿真状态] 暂停
```

## 技术细节

### 垂直同步的作用

- **开启时**：`sync_frame_time()` 会让程序等待，确保渲染帧率与仿真时间步长匹配实时时间
- **关闭时**：不调用 `sync_frame_time()`，GPU 可以全速渲染和模拟

### 性能提升

关闭垂直同步后：
- 单 GPU 环境：通常可达到 2-10 倍速
- 多 GPU 环境：取决于具体的 GPU 性能和场景复杂度
- 无显示器的 headless 模式：本来就没有同步，此功能无效

## 注意事项

1. 这个功能只在**有窗口可视化模式**下有效（非 headless 模式）
2. 倍速并非真正的"快进"，而是移除了渲染的时间限制
3. 实际加速倍数取决于 GPU 性能和场景复杂度
4. 加速模式下可能会出现画面不流畅的情况，这是正常的

## 测试命令

```bash
# 使用 visualize.sh 测试
scripts/visualize.sh outputs/pose3_50k_cfg2/stage1_nn

# 启动后按 'F' 或 'V' 键关闭垂直同步，观察速度提升
```
