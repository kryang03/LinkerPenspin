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
