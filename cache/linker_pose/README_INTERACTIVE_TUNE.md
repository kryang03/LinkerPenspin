# 交互式姿态调试工具使用指南

## 概述

`interactive_tune.py` 是一个整合型交互式调试脚本，专门为 25 DoF Flying Hand (`L25_dof_urdf_flying.urdf`) 设计。

**核心功能:**
- 整合 `find_pos`（找位置）、`find_rot`（找角度）和 `gravity_sim`（测稳定性）
- 键盘控制手部基座的 6 个自由度（3平移 + 3旋转）
- 多种预设手指姿态（打开、握拳、Triangle Pass、Pinch 等）
- 重力测试功能
- 一键导出可复制的姿态配置

## 快速开始

```bash
# 进入脚本目录
cd cache/linker_pose

# 使用默认设置运行 (spinning_pen + triangle 预设)
python interactive_tune.py

# 使用 pencil 物体
python interactive_tune.py --object pencil

# 使用特定手指预设
python interactive_tune.py --preset pen_hold

# 自定义物体缩放
python interactive_tune.py --object pencil --scale 0.3
```

## 键盘控制

### 位置控制 (WASD + QE)
| 按键 | 功能 | 说明 |
|------|------|------|
| W | 前进 | X轴正方向 |
| S | 后退 | X轴负方向 |
| A | 左移 | Y轴正方向 |
| D | 右移 | Y轴负方向 |
| Q | 上升 | Z轴正方向 |
| E | 下降 | Z轴负方向 |

### 旋转控制 (IJKL + UO)
| 按键 | 功能 | 说明 |
|------|------|------|
| I | Pitch 上 | 绕Y轴正向旋转 |
| K | Pitch 下 | 绕Y轴负向旋转 |
| J | Yaw 左 | 绕Z轴正向旋转 |
| L | Yaw 右 | 绕Z轴负向旋转 |
| U | Roll 左 | 绕X轴正向旋转 |
| O | Roll 右 | 绕X轴负向旋转 |

### 功能键
| 按键 | 功能 | 说明 |
|------|------|------|
| Space | 切换重力 | 开启/关闭重力测试抓取稳定性 |
| R | 重置物体 | 将笔重置到初始位置 |
| P | 打印姿态 | 导出可复制到代码的配置 |
| T | 切换预设 | 循环切换手指预设姿态 |
| F | 显示帮助 | 切换帮助信息显示 |
| +/- | 调整步长 | 增大/减小移动和旋转步长 |

### 手指微调
| 按键 | 功能 |
|------|------|
| 1 | 调整拇指弯曲度 |
| 2 | 调整食指弯曲度 |
| 3 | 调整中指弯曲度 |

## Triangle Pass 调试流程

1. **启动脚本**
   ```bash
   python interactive_tune.py --preset triangle
   ```

2. **对齐位置**
   - 使用 WASD/QE 移动手部基座
   - 使手掌接近笔的中心位置
   - 观察物体是否在手指之间

3. **调整角度**
   - 使用 IJKL/UO 旋转手部
   - 对于 Triangle Pass，手掌通常需要稍微侧倾
   - 确保拇指、食指、中指能形成三角形包围笔

4. **微调手指**
   - 按 T 切换不同预设，或使用 1/2/3 微调单个手指
   - 目标：拇指、食指、中指形成稳定的三点接触

5. **测试稳定性**
   - 按 Space 开启重力
   - 如果笔掉落，按 R 重置，继续调整
   - 如果笔保持稳定 3-5 秒，姿态可用

6. **导出姿态**
   - 找到稳定姿态后，按 P 打印
   - 复制输出内容到 `linker_hand_grasp.py`

## 导出格式说明

按 P 键后会输出三种格式：

### 格式1: 固定基座版本 (21 DoF)
适用于现有的 `linker_hand_grasp.py`：
```python
'hand': [...],  # 19 个手指关节角度
'object': [...]  # 位置(3) + 旋转(4)
```

### 格式2: Flying Hand 版本 (25 DoF)
适用于新的 Flying Hand 任务：
```python
'flying_hand': [
    {
        'hand': [...],    # 25 个关节角度 (6 基座 + 19 手指)
        'object': [...]   # 位置(3) + 旋转(4)
    }
]
```

### 格式3: 详细状态
用于调试和记录：
```python
# 基座状态 (6 DoF):
#   平移 (x,y,z): [0.0, 0.0, 0.35]
#   旋转 (r,p,y): [0.0, 0.0, 0.0]
# 手指状态 (19 DoF): [...]
# 物体世界坐标: ...
```

## URDF 关节映射

25 DoF Flying Hand 的关节顺序：

| 索引 | 关节名称 | 类型 | 限制 |
|------|---------|------|------|
| 0 | virtual_px | 平移X | [-0.05, 0.05] m |
| 1 | virtual_py | 平移Y | [-0.05, 0.05] m |
| 2 | virtual_pz | 平移Z | [0.30, 0.40] m |
| 3 | virtual_rx | Roll | [-1.57, 1.57] rad |
| 4 | virtual_ry | Pitch | [-1.57, 1.57] rad |
| 5 | virtual_rz | Yaw | [-3.14, 3.14] rad |
| 6-9 | little_joint0-3 | 小拇指 | 各不同 |
| 10-13 | ring_joint0-3 | 无名指 | 各不同 |
| 14-17 | middle_joint0-3 | 中指 | joint0 锁定 |
| 18-21 | index_joint0-3 | 食指 | 各不同 |
| 22-26 | thumb_joint0-4 | 拇指 | 各不同 |

## 将姿态添加到训练代码

### 步骤1: 编辑 `linker_hand_grasp.py`

找到 `canonical_pose_dict`，添加新条目：

```python
self.canonical_pose_dict = {
    'pencil': [...],  # 现有配置
    
    # 添加新的 Flying Hand 配置
    'flying_triangle': [
        {
            'hand': [/* 从脚本复制的 25 个值 */],
            'object': [/* 从脚本复制的 7 个值 */]
        }
    ]
}
```

### 步骤2: 更新配置文件

在 `configs/task/` 中更新相关配置：
```yaml
env:
  genGraspCategory: "flying_triangle"  # 使用新的姿态类别
  flyingHand:
    enabled: true  # 启用 Flying Hand
```

### 步骤3: 运行 grasp 生成

```bash
python scripts/linker_gen_grasp.sh
```

## 常见问题

### Q: 笔总是掉落
- 调整手指弯曲度，确保三点接触
- 尝试 `pen_hold` 预设
- 适当旋转手部，使掌心更好地托住笔

### Q: 手部无法移动到目标位置
- 检查关节限制 (virtual_px/py: ±5cm, virtual_pz: 30-40cm)
- 按 +/- 调整步长

### Q: 重力测试时物体晃动
- 增加接触点数量（调整手指姿态）
- 降低物体初始位置，减少下落冲击

### Q: 导出的姿态在训练中不工作
- 确认使用正确的格式（21 DoF vs 25 DoF）
- 检查物体位置是否需要根据基座偏移调整
