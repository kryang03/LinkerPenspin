# 资产加载优化分析

## 概述

本文档分析了当前 LinkerPenspin 项目的资产加载配置，针对精细转笔技巧（thumb_around、triangle_pass）的性能和精度需求，提出优化建议。

## 当前配置分析

### 1. LinkerPen（转笔物体）

**配置：** `assets/cylinder/linkerpen/spinning_pen.urdf`

```xml
<collision>
  <geometry>
    <cylinder radius="0.009" length="0.15"/>
  </geometry>
</collision>
```

**评估：⭐⭐⭐⭐⭐ 最优**

| 指标 | 评分 | 说明 |
|------|------|------|
| 碰撞效率 | 极高 | 圆柱体是 PhysX 原始几何，有专门优化的碰撞算法 |
| 碰撞精度 | 精确 | 数学解析解，无近似误差 |
| 显存占用 | 极低 | 只需存储半径和长度两个参数 |
| 适合度 | 完美 | 转笔就是圆柱形物体，无需 mesh |

**结论：无需修改，已是最佳选择。**

### 2. Linker Hand（机械手）

**配置：** `assets/linker_hand/L25_dof_urdf.urdf`

```xml
<collision>
  <geometry>
    <mesh filename="meshes/little_joint0.STL"/>
  </geometry>
</collision>
```

**评估：⭐⭐⭐ 可优化**

| 指标 | 评分 | 说明 |
|------|------|------|
| 碰撞效率 | 较低 | 每个 mesh 需要凸分解和三角形级碰撞检测 |
| 碰撞精度 | 高 | 保留了原始 CAD 模型细节 |
| 显存占用 | 较高 | 每个 link 都存储完整顶点/面数据 |
| 适合度 | 过度 | 转笔不需要完整手指细节，只需接触面 |

## 优化方案

### 方案 A：简化碰撞几何（推荐）

为 Linker Hand 创建简化版 URDF，使用原始几何体代替 mesh：

```
优化策略：
├── 指尖（与笔接触的关键部位）
│   └── 使用胶囊体(capsule)或球体(sphere)
│       - 胶囊体：cylinder + 两端半球，模拟指腹
│       - 精度足够，碰撞检测极快
│
├── 指节中段
│   └── 使用圆柱体(cylinder)或长方体(box)
│       - 不直接参与转笔，可大幅简化
│
└── 手掌
    └── 使用 box 或简化 mesh
        - 可能需要凸分解，但调用频率低
```

**预期收益：**
- 碰撞检测速度提升 3-5x
- 显存占用减少 30-50%
- 仿真稳定性提升（更少的穿透问题）

### 方案 B：启用 VHACD 凸分解（保守）

取消注释现有代码：

```python
hand_asset_options.vhacd_enabled = True
hand_asset_options.vhacd_params = gymapi.VhacdParams()
hand_asset_options.vhacd_params.resolution = 100000
# 添加更多参数优化
hand_asset_options.vhacd_params.max_convex_hulls = 8  # 限制凸包数量
hand_asset_options.vhacd_params.max_num_vertices_per_ch = 32  # 限制每个凸包顶点数
```

**预期收益：**
- 碰撞检测速度提升 1.5-2x（相比纯 mesh）
- 保留大部分几何细节
- 无需修改 URDF 文件

### 方案 C：双层碰撞几何

使用两套碰撞几何：
1. **粗略层**：用于快速剔除（broad phase）
2. **精细层**：用于指尖区域精确碰撞

需要自定义碰撞过滤器，复杂度较高。

## 针对转笔技巧的特定建议

### Thumb Around（拇指旋转）

**关键接触点：**
- 拇指指腹（推动笔旋转）
- 食指/中指侧面（支撑）

**优化重点：**
- 拇指末端使用高精度碰撞体（胶囊或球）
- 其他手指可使用简化几何

### Triangle Pass（三角传递）

**关键接触点：**
- 食指、中指、无名指的指腹
- 需要准确检测笔与各指的接触时序

**优化重点：**
- 三个参与手指的末端使用相同精度
- 小指和拇指可简化

## 实施建议

### 短期（不修改 URDF）

```python
# 在 _create_object_asset() 中优化 VHACD 参数
hand_asset_options.vhacd_enabled = True
hand_asset_options.vhacd_params = gymapi.VhacdParams()
hand_asset_options.vhacd_params.resolution = 50000  # 降低分辨率
hand_asset_options.vhacd_params.max_convex_hulls = 4  # 限制凸包数
hand_asset_options.vhacd_params.max_num_vertices_per_ch = 16
```

### 长期（修改 URDF）

创建 `L25_dof_simplified_collision.urdf`：
- 保留 visual mesh（渲染用）
- collision 使用原始几何体

## 性能对比预估

| 方案 | 碰撞速度 | 精度 | 实施难度 | 推荐度 |
|------|----------|------|----------|--------|
| 当前 | 1x | 高 | - | - |
| VHACD优化 | 1.5-2x | 中-高 | 低 | ⭐⭐⭐ |
| 简化几何 | 3-5x | 中 | 中 | ⭐⭐⭐⭐⭐ |
| 双层碰撞 | 2-3x | 高 | 高 | ⭐⭐ |

## 当前 LinkerPen 配置验证

LinkerPen 的圆柱体配置已经是最优选择：

1. **碰撞检测**：PhysX 对圆柱体使用解析方法，O(1) 复杂度
2. **无 mesh 开销**：不需要顶点/面存储，不需要凸分解
3. **数值稳定**：圆柱体碰撞没有"卡角"问题
4. **完美匹配**：转笔本身就是圆柱形

**不建议修改 LinkerPen 的碰撞配置。**

## 结论

1. **LinkerPen 资产**：已最优，无需修改
2. **Linker Hand 资产**：主要优化目标，建议采用方案 A（简化碰撞几何）
3. **优先级**：如果显存/速度不是瓶颈，先尝试方案 B（VHACD）

## 附录：IsaacGym 碰撞几何效率排序

```
最快 → 最慢:
1. Sphere（球）       - 解析碰撞，最简单
2. Capsule（胶囊）    - 解析碰撞，适合手指
3. Box（长方体）      - 解析碰撞，SAT 算法
4. Cylinder（圆柱）   - 解析碰撞，稍复杂
5. Convex Hull（凸包）- GJK/EPA 算法
6. Triangle Mesh      - 三角形级碰撞，最慢
```

对于转笔任务，手指使用 Capsule，笔使用 Cylinder 是最佳组合。
