# Actor 级别物理参数修复

## 问题背景

在 Space E curriculum 训练中，当 `alpha` 较低（例如 0.1~0.3）时，出现两个严重的物理问题：

### 1. "吸附" 现象 (Adsorption)
- **表现**: 物体碰到手后"粘住"，不反弹
- **原因**: `bounce_threshold_velocity` 没有正确缩放
- **机理**: 当速度阈值没有随 α 缩放时，在慢动作仿真中几乎所有碰撞都被归类为"软着陆"，导致无反弹

### 2. "蹭飞" 现象 (Phantom Launching)
- **表现**: 物体轻轻接触手部后突然高速弹飞
- **原因**: `contact_offset` 只在全局 SimParams 中设置，但 **Actor 级别的 Shape Properties 会覆盖全局设置**
- **机理**: 在 `_create_envs` 中硬编码了 `contact_offset = 0.002`，这个值没有随 α 缩放

## PhysX 参数优先级层次

```
Actor Shape Properties  >  Asset Properties  >  SimParams (全局默认值)
        ↑                       ↑                    ↑
    最高优先级            中等优先级            最低优先级
```

**关键发现**: 之前的 `apply_curriculum_physics()` 只更新了 `sim_params.physx.contact_offset`，这属于最低优先级，会被 Actor 级别的设置覆盖！

## 修复内容

### 1. 添加 `self.object_handles` 存储
位置: `_create_envs()` 函数

```python
self.object_handles = []  # 存储 object actor handles 以便运行时更新 Shape Properties
# ...
self.object_handles.append(object_handle)  # 存储 handle 以便运行时更新 Shape Properties
```

### 2. 在 `apply_curriculum_physics()` 中更新 Actor Shape Properties
位置: `apply_curriculum_physics()` 函数末尾

新增代码逻辑:
1. 遍历所有环境
2. 获取手部和物体的 Shape Properties
3. 更新 `contact_offset` 和 `rest_offset` 为缩放后的值
4. 跳过 `thumb_joint0` 的特殊碰撞过滤设置 (contact_offset=-1.0)
5. 应用更新后的 Shape Properties

### 3. 添加底层验证打印
读取实际生效的 Shape Properties 并与预期值比对，如有不一致则打印警告。

## 缩放规则

| 参数 | 缩放公式 | 说明 |
|------|----------|------|
| `bounce_threshold_velocity` | `base_value * α` | 线性缩放 |
| `contact_offset` | `base_value * max(α, 0.1)` | 带下限保护 |
| `rest_offset` | `contact_offset * 0.5` | 保持比例 |

## 预期训练输出

修复后，curriculum 变化时应该看到类似输出:

```
  -> PhysX contact_offset (全局 SimParams): 0.00020
  -> PhysX bounce_threshold: 0.0200 (原始: 0.20)
  -> Actor Shape Properties 已更新: 30000 个环境
     contact_offset=0.00020, rest_offset=0.00010
  -> [验证] 实际生效值 - Hand: 0.00020, Object: 0.00020
```

如果看到 `[警告]` 信息，说明设置可能未正确应用。

## 相关文件

- `penspin/tasks/linker_hand_hora.py`: 主要修改
- `penspin/utils/time_warping.py`: `get_scaled_physx_params()` 提供缩放计算
- `configs/task/LinkerHandHora.yaml`: 基础物理参数配置
