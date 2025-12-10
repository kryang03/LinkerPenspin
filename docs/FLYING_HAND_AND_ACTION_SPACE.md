# Flying Hand 和 Action Space 配置说明

本文档详细说明 Flying Hand 和 `disableRingLittleFinger` 两个配置项对观测空间、动作空间的影响，以及相关代码实现。

---

## 1. Flying Hand (6 DoF 浮空底座)

### 1.1 功能说明

Flying Hand 为灵巧手添加了 6 个虚拟 DOF（3个平移 + 3个旋转），使手可以在空间中移动和旋转。

| 配置项 | 描述 | 默认值 |
|--------|------|--------|
| `task.env.flyingHand.enabled` | 是否启用 Flying Hand | `False` (默认关闭，需显式开启) |
| `task.env.flyingHand.defaultHeight` | 默认悬浮高度 | 0.35m |
| `task.env.flyingHand.linearVelocity` | 线速度上限 | 0.1 m/s |
| `task.env.flyingHand.angularVelocity` | 角速度上限 | 2.0 rad/s |

### 1.2 DOF 结构变化

| 配置 | 总 DOF | 结构 |
|------|--------|------|
| **Flying Hand 启用** | 27 | 6 (base) + 21 (hand) |
| **Flying Hand 禁用** | 21 | 21 (hand only) |

Flying Hand 的 6 个虚拟 DOF（在 URDF 中按字母顺序加载，IsaacGym 排序后为 [0-5]）：

| DOF Index | 关节名称 | 类型 | 范围 |
|-----------|----------|------|------|
| 0 | `virtual_px` | 平移 X | ±0.05m |
| 1 | `virtual_py` | 平移 Y | ±0.05m |
| 2 | `virtual_pz` | 平移 Z | 0.30~0.40m |
| 3 | `virtual_rx` | 旋转 Roll | ±1.57 rad |
| 4 | `virtual_ry` | 旋转 Pitch | ±1.57 rad |
| 5 | `virtual_rz` | 旋转 Yaw | ±3.14 rad |

### 1.3 初始姿态 (Grasp Cache 中的值)

```
virtual_px: 0.000000 (约为 0)
virtual_py: 0.000000 (约为 0)
virtual_pz: 0.350000 (默认高度 35cm)
virtual_rx: 0.000000 (约为 0)
virtual_ry: -1.310000 (约 -75°，手掌朝下)
virtual_rz: 0.000000 (约为 0)
```

### 1.4 代码实现

#### URDF 资源选择
```python
# configs/task/LinkerHandHora.yaml
asset:
    handAsset: 'assets/linker_hand/L25_dof_urdf.urdf'          # 默认: 固定底座
    # Flying Hand (启用时改为): 'assets/linker_hand/L25_dof_urdf_flying.urdf'
```

#### 配置加载
```python
# penspin/tasks/linker_hand_hora.py 第 2188-2247 行
def _setup_flying_hand_config(self, env_config):
    flying_config = env_config.get('flyingHand', {})
    self.flying_hand_enabled = flying_config.get('enabled', False)
    if self.flying_hand_enabled:
        self.flying_default_height = flying_config.get('defaultHeight', 0.35)
        # ... 其他参数
```

#### 观测/动作维度计算
```python
# penspin/tasks/linker_hand_hora.py 第 192 行
self.actual_proprio_dim = 2 * self.num_dofs  # num_dofs = 27 (Flying) 或 21

# 观测维度 = 6 * num_dofs (3个时间步 × 2 × num_dofs)
# Flying: 6 × 27 = 162
# 非Flying: 6 × 21 = 126
```

---

## 2. disableRingLittleFinger (禁用无名指和小拇指)

### 2.1 功能说明

通过禁用无名指和小拇指来缩减动作空间，提高 RL 训练效率。

| 配置项 | 描述 | 默认值 |
|--------|------|--------|
| `task.env.actionSpace.disableRingLittleFinger` | 是否禁用无名指和小拇指 | `True` |

### 2.2 动作维度变化

| Flying | disableRingLittleFinger | 策略输出维度 | 仿真 DOF |
|--------|------------------------|-------------|---------|
| True | False | 27 | 27 |
| True | True | **19** | 27 |
| False | False | 21 | 21 |
| False | True | **13** | 21 |

### 2.3 关节索引映射 (IsaacGym 字母顺序)

**21 DOF 手部关节顺序**：
| 索引范围 | 手指 | DOF 数 |
|---------|------|--------|
| [0-3] | index (食指) | 4 |
| [4-7] | little (小拇指) | 4 |
| [8-11] | middle (中指) | 4 |
| [12-15] | ring (无名指) | 4 |
| [16-20] | thumb (拇指) | 5 |

**禁用后激活的关节** (13 DOF):
- index [0-3] + middle [8-11] + thumb [16-20]

**被禁用的关节** (8 DOF):
- little [4-7] + ring [12-15]

### 2.4 代码实现

#### ActionSpaceMapper
```python
# penspin/utils/robot_config.py 第 718-930 行
class ActionSpaceMapper:
    """动作空间映射器，支持 Flying Hand 和手指禁用"""
    
    def __init__(self, disable_ring_little=False, flying_hand_enabled=False):
        if disable_ring_little:
            self.hand_action_dim = 13  # index + middle + thumb
            self.hand_active_indices = [0,1,2,3, 8,9,10,11, 16,17,18,19,20]
            self.hand_disabled_indices = [4,5,6,7, 12,13,14,15]
        else:
            self.hand_action_dim = 21
        
        if flying_hand_enabled:
            self.flying_action_dim = 6
        else:
            self.flying_action_dim = 0
        
        self.action_dim = self.flying_action_dim + self.hand_action_dim
    
    def expand_actions(self, policy_actions, init_pose=None):
        """将策略输出（低维）扩展为仿真 DOF（全维）"""
        # 禁用关节的动作设为 0（保持不变）
        ...
```

#### 配置加载
```python
# penspin/tasks/linker_hand_hora.py 第 2158 行
def _setup_action_space_config(self, env_config):
    action_space_config = env_config.get('actionSpace', {})
    self.disable_ring_little = action_space_config.get('disableRingLittleFinger', False)
    
    self.action_mapper = ActionSpaceMapper(
        disable_ring_little=self.disable_ring_little,
        flying_hand_enabled=self.flying_hand_enabled
    )
    self.actual_action_dim = self.action_mapper.get_action_dim()
```

---

## 3. 观测空间和动作空间总结

### 3.1 当前配置 (Flying + disableRingLittleFinger)

| 空间 | 维度 | 说明 |
|------|------|------|
| **观测空间** | 162 | 6 × 27 DOF (3时间步 × (位置+目标)) |
| **动作空间** | 19 | 6 (flying) + 13 (hand reduced) |
| **仿真 DOF** | 27 | 6 (flying) + 21 (hand) |

### 3.2 关闭 Flying Hand 后

| 空间 | 维度 | 说明 |
|------|------|------|
| **观测空间** | 126 | 6 × 21 DOF |
| **动作空间** | 13 | 13 (hand reduced) |
| **仿真 DOF** | 21 | 21 (hand only) |

---

## 4. 相关文件路径

| 文件 | 用途 |
|------|------|
| `configs/task/LinkerHandHora.yaml` | 配置文件 (flyingHand, actionSpace) |
| `penspin/tasks/linker_hand_hora.py` | 环境主类 |
| `penspin/utils/robot_config.py` | ActionSpaceMapper 和常量定义 |
| `assets/linker_hand/L25_dof_urdf_flying.urdf` | Flying Hand URDF |
| `assets/linker_hand/L25_dof_urdf.urdf` | 普通 Hand URDF |
| `cache/3_30000_61_grasp_cache.npy` | 抓取缓存 (61维格式) |

---

## 5. 注意事项

1. **Grasp Cache 兼容性**: 现有的 `3_30000_61_grasp_cache.npy` 是为 Flying Hand (27 DOF) 生成的，关闭 Flying Hand 后需要使用兼容的缓存或重新生成。

2. **初始姿态**: Flying Hand 禁用后，手的位置和朝向由 `linker_hand_start_pose` 控制（需要设置正确的 Quaternion 旋转）。

3. **奖励函数**: `flying_base_movement_penalty` 在 Flying Hand 禁用后不生效。
