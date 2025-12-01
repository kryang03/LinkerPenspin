# 旋转奖励算法重构总结

## 背景问题

在训练 Triangle Pass 转笔技巧时，发现策略学习到了一种"作弊"行为：笔在手指间滚动（自转/搓笔），而不是真正的公转（绕指定轴旋转）。

**问题根源**：原始的旋转奖励计算使用刚体角速度：
```python
vec_dot = (object_angvel * rot_axis_buf).sum(-1)
```
这种方法无法区分：
- **公转 (Revolution)**：笔绕外部轴旋转（期望的行为）
- **自转 (Self-rotation)**：笔绕自身长轴滚动（作弊行为）

## 解决方案

### 核心思路
不再使用刚体角速度，而是追踪笔长轴在旋转轴垂直平面上的投影角度变化。

### 算法步骤
1. 获取笔的当前朝向（local Z 轴的世界坐标方向）
2. 将笔轴投影到旋转轴的垂直平面上
3. 使用上一帧的投影方向和当前投影方向的叉积计算角度变化
4. 通过叉积与旋转轴的点积判断旋转方向（正向/反向）

### 数学公式
```
# 投影到旋转轴垂直平面
pen_axis_proj = pen_axis - (pen_axis · rot_axis) * rot_axis
pen_axis_proj = normalize(pen_axis_proj)

# 计算角度变化（使用叉积）
cross = prev_proj × curr_proj
signed_sin = cross · rot_axis
cos_angle = prev_proj · curr_proj
angle_delta = atan2(signed_sin, cos_angle)
```

## 修改的文件

### 1. `penspin/tasks/linker_hand_hora.py`

#### 修改位置 1：`rot_axis_buf` 初始化（约 257-280 行）
- **变更**：支持 3D 向量格式的 `rotation_axis` 配置
- **兼容**：同时保留对旧字符串格式（`'+z'`）的支持

```python
# 新格式：3D 向量
rotation_axis: [0, 0, 1]

# 旧格式（仍支持）
rotation_axis: '+z'
```

#### 修改位置 2：`compute_reward` 方法（约 1130-1205 行）
- **变更**：使用投影算法计算旋转角度
- **新增**：`self.prev_pen_axis_proj` 状态变量，追踪上一帧的投影方向
- **保留**：`object_angvel` 等观测值仍使用原始四元数计算

```python
# 关键代码片段
pen_axis_world = quat_apply(object_rot, pen_local_axis)
rot_axis_normalized = rot_axis_buf / (torch.norm(rot_axis_buf, dim=-1, keepdim=True) + 1e-8)
proj_on_rot_axis = (pen_axis_world * rot_axis_normalized).sum(-1, keepdim=True)
pen_axis_proj = pen_axis_world - proj_on_rot_axis * rot_axis_normalized
pen_axis_proj = pen_axis_proj / (torch.norm(pen_axis_proj, dim=-1, keepdim=True) + 1e-8)

cross = torch.cross(self.prev_pen_axis_proj, pen_axis_proj, dim=-1)
signed_sin = (cross * rot_axis_normalized).sum(-1)
cos_angle = (self.prev_pen_axis_proj * pen_axis_proj).sum(-1)
angle_delta = torch.atan2(signed_sin, cos_angle)
```

#### 不变的部分
- **观测值计算**：`object_angvel` 等传入 observation 的值仍使用原始的四元数差分方法
- **物理模拟**：所有物理相关的计算保持不变
- **其他奖励**：位置奖励、接触奖励等不受影响

### 2. `configs/task/LinkerHandHora.yaml`

#### 修改位置：`env.rotation_axis` 配置（约 96-118 行）
- **变更**：格式从字符串改为 3D 向量
- **新增**：详细的文档注释

```yaml
# 期望的转笔旋转轴 (3D向量格式)
# [0,0,1] = +Z轴 (逆时针)
# [0,0,-1] = -Z轴 (顺时针)
# 支持任意方向的旋转轴
rotation_axis: [0, 0, 1]
```

### 3. `cache/linker_pose/interactive_tune.py`

#### 新增功能
1. **配置加载**：自动从 `LinkerHandHora.yaml` 读取 `rotation_axis`
2. **可视化**：在 3D 场景中绘制旋转轴（黄色箭头线）
3. **交互控制**：
   - **V 键**：切换旋转轴显示/隐藏
   - **B 键**：循环切换预设旋转轴方向

#### 新增状态变量
```python
self.show_rotation_axis = True  # 是否显示旋转轴
self.current_rotation_axis = np.array(CONFIG["rotation_axis"])
self.rotation_axis_preset_idx = 0  # 预设索引
```

#### 预设旋转轴选项
```python
"rotation_axis_presets": [
    {"name": "+Z (CCW)", "axis": [0, 0, 1]},
    {"name": "-Z (CW)", "axis": [0, 0, -1]},
    {"name": "+X", "axis": [1, 0, 0]},
    {"name": "-X", "axis": [-1, 0, 0]},
    {"name": "+Y", "axis": [0, 1, 0]},
    {"name": "-Y", "axis": [0, -1, 0]},
]
```

## 影响范围

### 受影响的部分
| 功能 | 变更 |
|------|------|
| 旋转奖励计算 | ✅ 使用新投影算法 |
| `total_rot_angle` 累积 | ✅ 使用新算法的角度增量 |
| Debug 输出的旋转信息 | ✅ 使用新算法数据 |
| `rotation_axis` 配置格式 | ✅ 支持 3D 向量 |
| `interactive_tune.py` | ✅ 新增可视化功能 |

### 不受影响的部分
| 功能 | 说明 |
|------|------|
| `object_angvel` 观测值 | ❌ 仍使用四元数差分计算 |
| 其他观测值 | ❌ 保持原样 |
| 物理模拟参数 | ❌ 保持原样 |
| 其他奖励项 | ❌ 保持原样 |
| URDF/资产文件 | ❌ 无变更 |

## 使用方法

### 配置旋转轴
编辑 `configs/task/LinkerHandHora.yaml`：
```yaml
env:
  rotation_axis: [0, 0, 1]  # 修改为期望的旋转轴
```

### 使用交互式调试工具
```bash
cd /home/user/Code/LinkerPenspin
python cache/linker_pose/interactive_tune.py
```
- 按 **V** 键切换旋转轴可视化
- 按 **B** 键循环选择预设旋转轴

### 验证效果
训练时观察 debug 输出：
```
[env_000000] reward: 0.0000, rot_angle: 0.0000, total_rot: 0.00 deg, ...
```
- `rot_angle`：当前帧的旋转角度增量（弧度）
- `total_rot`：累积旋转角度（度）

现在 `total_rot` 只会在笔真正绕旋转轴公转时增加，自转不会被计入。

## 注意事项

1. **投影退化**：当笔轴与旋转轴平行时，投影会退化。代码中已添加数值保护（`+ 1e-8`）
2. **角度不连续**：使用 `atan2` 处理角度，避免 ±180° 附近的不连续问题
3. **配置兼容**：旧格式 `'+z'` 仍然支持，会自动转换为 `[0, 0, 1]`

---
*文档生成时间：2025年*
