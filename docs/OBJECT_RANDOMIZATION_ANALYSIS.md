# 物体性质随机化分析文档

本文档详细分析了 `LinkerHandHora` 任务中与物体（`cylinder_pencil-5-7`）性质相关的所有随机化设置及其对应的代码实现。

---

## 一、配置文件概述

**配置文件路径**: `configs/task/LinkerHandHora.yaml`

```yaml
baseObjScale: 0.3  # 基础缩放比例

randomization:
    # 质量随机化
    randomizeMass: True
    randomizeMassLower: 0.01
    randomizeMassUpper: 0.02
    
    # 质心(COM)随机化
    randomizeCOM: True
    randomizeCOMLower: -0.001
    randomizeCOMUpper: 0.001
    
    # 摩擦系数随机化
    randomizeFriction: True
    randomizeFrictionLower: 0.3
    randomizeFrictionUpper: 3.0
    
    # 尺寸缩放随机化（当前实际被代码禁用）
    randomizeScale: True
    randomize_hand_scale: False
    scaleListInit: True
    randomizeScaleList: [0.3]
    randomizeScaleLower: 0.25
    randomizeScaleUpper: 0.3
```

---

## 二、物体原始资产信息

### 2.1 URDF 文件
**路径**: `assets/cylinder/pencil-5-7/0000.urdf`

```xml
<robot name="object">
  <link name="object">
    <visual>
      <geometry>
        <cylinder radius="0.04" length="0.4" />
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.04" length="0.4" />
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2" />
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001" />
    </inertial>
  </link>
</robot>
```

### 2.2 NPY 元数据
**路径**: `assets/cylinder/pencil-5-7/0000.npy`
- **Shape**: `(1, 3)`
- **Data**: `[[5.0, 0.04, 1.0]]`
  - `size_info[0] = 5.0` - 长度比例因子（相对于半径的倍数）
  - `size_info[1] = 0.04` - 圆柱体半径（米）
  - `size_info[2] = 1.0` - 保留字段

### 2.3 代码解析尺寸逻辑
**文件**: `penspin/tasks/linker_hand_hora.py` (Line 1646-1648)

```python
size_info = np.load(os.path.join(asset_root, object_asset_file.replace('.urdf', '.npy')))[0]
self.pen_radius = size_info[1]  # 0.04 米
self.pen_length = size_info[0] * (size_info[1] * 2)  # 5.0 * 0.08 = 0.4 米
```

**原始物体几何尺寸（缩放前）**:
| 属性 | 值 |
|------|-----|
| 半径 (pen_radius) | 0.04 m |
| 长度 (pen_length) | 0.4 m |
| URDF 定义质量 | 0.2 kg |

---

## 三、随机化方法详解

### 3.1 尺寸缩放（Scale）

#### 3.1.1 配置参数
```yaml
baseObjScale: 0.3
randomizeScale: True
scaleListInit: True
randomizeScaleList: [0.3]
```

#### 3.1.2 代码实现
**文件**: `penspin/tasks/linker_hand_hora.py` (Line 323-336)

```python
self.obj_scale = self.base_obj_scale  # = 0.3
# Modified，为了躲避scale随机化，这里注释掉

# if self.randomize_scale:
#     num_scales = len(self.randomize_scale_list)
#     self.obj_scale = np.random.uniform(
#         self.randomize_scale_list[i % num_scales] - 0.025,
#         self.randomize_scale_list[i % num_scales] + 0.025
#     )
self.gym.set_actor_scale(env_ptr, object_handle, self.obj_scale)
self._update_priv_buf(env_id=i, name='obj_scale', value=self.obj_scale)
```

#### 3.1.3 当前状态
⚠️ **代码层面禁用了缩放随机化**

虽然配置文件中 `randomizeScale: True`，但代码中的随机化逻辑已被注释掉，所有环境使用固定的 `baseObjScale = 0.3`。

#### 3.1.4 最终缩放效果
| 属性 | 原始值 | 缩放后 (`×0.3`) |
|------|--------|-----------------|
| 半径 | 0.04 m | 0.012 m (12 mm) |
| 长度 | 0.4 m | 0.12 m (120 mm) |

---

### 3.2 质量随机化（Mass）

#### 3.2.1 配置参数
```yaml
randomizeMass: True
randomizeMassLower: 0.01
randomizeMassUpper: 0.02
```

#### 3.2.2 代码实现
**文件**: `penspin/tasks/linker_hand_hora.py` (Line 332-341)

```python
if self.randomize_mass:
    prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
    for p in prop:
        p.mass = np.random.uniform(self.randomize_mass_lower, self.randomize_mass_upper)
    self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
    self._update_priv_buf(env_id=i, name='obj_mass', value=prop[0].mass)
else:
    prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
    self._update_priv_buf(env_id=i, name='obj_mass', value=prop[0].mass)
```

#### 3.2.3 最终效果
✅ **启用**
- **随机范围**: `[0.01, 0.02]` kg
- **分布**: 均匀分布
- URDF 原始质量 (0.2 kg) 被**完全覆盖**

---

### 3.3 质心随机化（Center of Mass）

#### 3.3.1 配置参数
```yaml
randomizeCOM: True
randomizeCOMLower: -0.001
randomizeCOMUpper: 0.001
```

#### 3.3.2 代码实现
**文件**: `penspin/tasks/linker_hand_hora.py` (Line 298-310)

```python
obj_com = [0, 0, 0]
# COM是物体的质心
if self.randomize_com:
    prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
    assert len(prop) == 1
    obj_com = [np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
               np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
               np.random.uniform(self.randomize_com_lower, self.randomize_com_upper)]
    prop[0].com.x, prop[0].com.y, prop[0].com.z = obj_com
    self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
self._update_priv_buf(env_id=i, name='obj_com', value=obj_com)
```

#### 3.3.3 最终效果
✅ **启用**
- **随机范围**: X/Y/Z 各自独立在 `[-0.001, 0.001]` 米内随机
- **分布**: 3D 均匀分布（立方体区域）
- 质心偏移量极小（±1mm）

---

### 3.4 摩擦系数与恢复系数随机化（Friction & Restitution）

#### 3.4.1 配置参数
```yaml
randomizeFriction: True
randomizeFrictionLower: 0.3
randomizeFrictionUpper: 3.0
```

#### 3.4.2 代码实现
**文件**: `penspin/tasks/linker_hand_hora.py` (Line 312-328)

```python
obj_friction = 1.0
obj_restitution = 0.0  # default is 0
# TODO: bad engineering because of urgent modification
if self.randomize_friction:
    rand_friction = np.random.uniform(self.randomize_friction_lower, self.randomize_friction_upper)
    obj_restitution = np.random.uniform(0, 1)

    # 同时设置手部和物体的摩擦/恢复系数
    hand_props = self.gym.get_actor_rigid_shape_properties(env_ptr, hand_actor)
    for p in hand_props:
        p.friction = rand_friction
        p.restitution = obj_restitution
    self.gym.set_actor_rigid_shape_properties(env_ptr, hand_actor, hand_props)

    object_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
    for p in object_props:
        p.friction = rand_friction
        p.restitution = obj_restitution
    self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_props)
    obj_friction = rand_friction
self._update_priv_buf(env_id=i, name='obj_friction', value=obj_friction)
self._update_priv_buf(env_id=i, name='obj_restitution', value=obj_restitution)
```

#### 3.4.3 最终效果
✅ **启用**

| 属性 | 随机范围 | 分布 | 备注 |
|------|----------|------|------|
| 摩擦系数 (friction) | `[0.3, 3.0]` | 均匀 | **同时应用于手部和物体** |
| 恢复系数 (restitution) | `[0.0, 1.0]` | 均匀 | **同时应用于手部和物体** |

⚠️ **注意**: 恢复系数 (restitution) 的随机化是**隐式**的，配置文件中没有单独的开关，当 `randomizeFriction=True` 时自动启用。

---

### 3.5 配置加载代码
**文件**: `penspin/tasks/linker_hand_hora.py` (Line 980-997)

```python
def _setup_domain_rand_config(self, rand_config):
    self.randomize_mass = rand_config['randomizeMass']
    self.randomize_mass_lower = rand_config['randomizeMassLower']
    self.randomize_mass_upper = rand_config['randomizeMassUpper']
    self.randomize_com = rand_config['randomizeCOM']
    self.randomize_com_lower = rand_config['randomizeCOMLower']
    self.randomize_com_upper = rand_config['randomizeCOMUpper']
    self.randomize_friction = rand_config['randomizeFriction']
    self.randomize_friction_lower = rand_config['randomizeFrictionLower']
    self.randomize_friction_upper = rand_config['randomizeFrictionUpper']
    self.randomize_scale = rand_config['randomizeScale']
    self.randomize_hand_scale = rand_config['randomize_hand_scale']
    self.scale_list_init = rand_config['scaleListInit']
    self.randomize_scale_list = rand_config['randomizeScaleList']
    self.randomize_scale_lower = rand_config['randomizeScaleLower']
    # ...
```

---

## 四、最终仿真物体性质汇总

基于上述分析，`cylinder_pencil-5-7` 在仿真中呈现的物体性质如下：

### 4.1 几何尺寸（固定）
| 属性 | 最终值 | 计算方式 |
|------|--------|----------|
| **半径** | **12 mm** | `0.04 × 0.3 = 0.012 m` |
| **长度** | **120 mm** | `0.4 × 0.3 = 0.12 m` |
| **直径** | **24 mm** | `0.012 × 2 = 0.024 m` |

> 💡 这些尺寸与真实铅笔尺寸相近（标准铅笔直径约 7-8mm，长度 170-190mm）

### 4.2 物理属性（随机化）

| 属性 | 分布类型 | 范围 | 默认值（若禁用） |
|------|----------|------|------------------|
| **质量** | 均匀 | `[10, 20] g` | URDF 定义的 0.2 kg |
| **质心偏移 (X)** | 均匀 | `[-1, 1] mm` | `0` |
| **质心偏移 (Y)** | 均匀 | `[-1, 1] mm` | `0` |
| **质心偏移 (Z)** | 均匀 | `[-1, 1] mm` | `0` |
| **摩擦系数** | 均匀 | `[0.3, 3.0]` | `1.0` |
| **恢复系数** | 均匀 | `[0.0, 1.0]` | `0.0` |

### 4.3 物理属性统计特征

假设各属性独立采样：

| 属性 | 均值 | 标准差 |
|------|------|--------|
| 质量 | 15 g | 2.89 g |
| 质心偏移 | 0 mm | 0.58 mm |
| 摩擦系数 | 1.65 | 0.78 |
| 恢复系数 | 0.5 | 0.29 |

---

## 五、特权信息 (Privileged Information) 中的物体属性

在 Teacher 训练过程中，以下物体属性作为特权信息提供给网络：

| 信息名称 | 维度 | 来源 |
|----------|------|------|
| `obj_position` | 3 | 物体位置 |
| `obj_scale` | 1 | 缩放比例 (固定 0.3) |
| `obj_mass` | 1 | 随机化质量 |
| `obj_friction` | 1 | 随机化摩擦系数 |
| `obj_com` | 3 | 随机化质心 |
| `obj_restitution` | 1 | 随机化恢复系数 |

---

## 六、代码中其他与物体相关的参数

### 6.1 物体端点计算（用于观测）
**文件**: `penspin/tasks/linker_hand_hora.py` (Line 562-577)

```python
# 物体端点位置（考虑缩放）
pencil_ends = [
    [0, 0, -(self.pen_length/2) * self.obj_scale],  # 下端点
    [0, 0, (self.pen_length/2) * self.obj_scale]    # 上端点
]
# 实际值：[-0.06, 0, 0] 和 [0.06, 0, 0]（相对于物体中心）

# 添加噪声
pencil_end_1 += (torch.rand(...) - 0.5) * (self.pen_radius*2)
pencil_end_2 += (torch.rand(...) - 0.5) * (self.pen_radius*2)
# 噪声范围：[-0.04, 0.04] × 0.3 = [-0.012, 0.012] m
```

### 6.2 随机力施加
**文件**: `penspin/tasks/linker_hand_hora.py` (Line 926-939)

```python
def update_rigid_body_force(self):
    if self.force_scale > 0.0:
        # 力与质量相关
        obj_mass = [self.gym.get_actor_rigid_body_properties(env, ...)[0].mass for env in self.envs]
        # ...
        self.rb_forces[force_indices, self.object_rb_handles, :] = \
            torch.randn(...) * obj_mass[force_indices, None] * self.force_scale
```

配置中的力相关参数：
```yaml
forceScale: 2.0
randomForceProbScalar: 0.25
forceDecay: 0.9
forceDecayInterval: 0.08
```

---

## 七、总结

### 7.1 启用的随机化
| 随机化类型 | 状态 | 范围 |
|------------|------|------|
| 质量 | ✅ | `[0.01, 0.02]` kg |
| 质心 | ✅ | `[-0.001, 0.001]` m (每轴) |
| 摩擦系数 | ✅ | `[0.3, 3.0]` |
| 恢复系数 | ✅ | `[0.0, 1.0]` |

### 7.2 禁用的随机化
| 随机化类型 | 状态 | 原因 |
|------------|------|------|
| 尺寸缩放 | ⚠️ 配置启用但代码禁用 | 代码注释掉了随机化逻辑 |
| 手部缩放 | ❌ | `randomize_hand_scale: False` |

### 7.3 最终仿真物体特征
- **几何外形**: 圆柱体，半径 12mm，长度 120mm
- **质量**: 10-20g（随机）
- **摩擦**: 0.3-3.0（随机，手部和物体同步）
- **弹性**: 0-1（随机，手部和物体同步）
- **质心**: 几何中心 ± 1mm（随机）

---

*文档生成时间: 2025年11月28日*
*基于代码版本: curriculum 分支*
