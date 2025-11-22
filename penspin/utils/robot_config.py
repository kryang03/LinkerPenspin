"""
Robot Configuration Constants
用于统一管理所有与机器人自由度、手指数量、传感器维度相关的常量

=============================================================================
维度迁移对照表：Allegro Hand (16 DoF, 4 Fingers) -> Linker Hand (21 DoF, 5 Fingers)
=============================================================================

基础维度变化：
┌────────────────────────┬────────────────┬─────────────────┬──────────────┐
│ 维度名称               │ Allegro (旧)   │ Linker (新)     │ 说明         │
├────────────────────────┼────────────────┼─────────────────┼──────────────┤
│ NUM_DOF                │ 16             │ 21              │ 关节自由度   │
│ NUM_FINGERS            │ 4              │ 5               │ 手指数量     │
│ FINGERTIP_CNT          │ 4              │ 5               │ 指尖数量     │
├────────────────────────┼────────────────┼─────────────────┼──────────────┤
│ PROPRIO_DIM            │ 32 (16×2)      │ 42 (21×2)       │ 本体感觉     │
│ FINGERTIP_POS_DIM      │ 12 (4×3)       │ 15 (5×3)        │ 指尖位置     │
│ CONTACT_DIM            │ 32 (密集传感器)│ 15 (5×3)        │ 触觉维度     │
├────────────────────────┼────────────────┼─────────────────┼──────────────┤
│ STUDENT_PRIV_DIM       │ 12             │ 15              │ 学生特权信息 │
│ TACTILE_FEATURE_DIM    │ 64 (2×32)      │ 30 (2×15)       │ 触觉特征     │
└────────────────────────┴────────────────┴─────────────────┴──────────────┘

关键变化说明：
1. **不变的维度**（在代码中无需修改）：
   - 历史长度: proprio_hist_len = 30, tactile_hist_len = 30
   - 时间步采样: TACTILE_USED_TIMESTEPS = 2
   - 点云: POINT_CLOUD_NUM_POINTS = 100
   - 物体端点: OBJ_ENDS_TOTAL_DIM = 18 (3×6)
   - 网络结构: Actor MLP [512, 256, 128], Priv MLP [256, 128, 8/64]

2. **变化的维度**（已通过常量自动适配）：
   - PROPRIO_DIM: 32 -> 42 (关节数增加)
   - FINGERTIP_POS_DIM: 12 -> 15 (手指数增加)
   - CONTACT_DIM: 32 -> 15 (传感器类型变化)
   - STUDENT_PRIV_DIM: 12 -> 15 (跟随 FINGERTIP_POS_DIM)
   - TACTILE_FEATURE_DIM: 64 -> 30 (跟随 CONTACT_DIM × 2)

3. **网络输入维度变化**：
   Teacher:
     - obs: 96 -> 126 (基础观测维度增加)
     - priv_info: 61 (动态计算，取决于启用的特权信息项)
     - 总输入: 136 = 126 + 8(priv_mlp) + 32(point_mlp) -> 动态
   
   Student:
     - obs: 96 -> 126
     - proprio_feat: 40 (temporal fusion, 不变)
     - extrin_gt: 8 (priv_mlp 输出, 不变)
     - 总输入: 144 = 126 + 40 + 8 -> 动态

4. **priv_info 布局变化**：
   固定前缀（0-9）：不变
     - obj_position(3) + obj_scale(1) + obj_mass(1) + obj_friction(1) + obj_com(3)
   
   动态部分（取决于启用项）：
     - fingertip_position: 12 -> 15 (4×3 -> 5×3)
     - fingertip_orientation: 16 -> 20 (4×4 -> 5×4)
     - fingertip_linvel: 12 -> 15 (4×3 -> 5×3)
     - fingertip_angvel: 12 -> 15 (4×3 -> 5×3)
     - tactile: 32 -> 15 (传感器密度 vs 每指一个)

参考文件：
- shape_logs/*: 记录了原始 Allegro Hand (16 DoF) 的完整网络形状
- FINAL_SUMMARY_REPORT.md: 记录了当前 Linker Hand (21 DoF) 的配置
=============================================================================
"""

# ============================================================
# Linker Hand Configuration (21 DoF, 5 Fingers)
# ============================================================

# 基础配置
NUM_DOF = 21  # 关节自由度数量
NUM_FINGERS = 5  # 手指数量
FINGERTIP_CNT = 5  # 指尖数量（等于手指数量）

# 触觉传感器配置
# Linker Hand: 5个手指末端关节的3维力传感器
CONTACT_DIM = 15  # 5 fingers × 3D force = 15
CONTACT_SENSOR_DIM = 3  # 每个传感器的维度（3D力）

# 本体感觉配置
# 注意：PROPRIO_DIM 包含 current_pos + target_pos，而非 pos + vel
PROPRIO_DIM = 2 * NUM_DOF  # 当前位置 + 目标位置 = 42 (21+21)
FINGERTIP_POS_DIM = FINGERTIP_CNT * 3  # 5 fingertips × 3D position = 15

# 历史缓冲配置
DEFAULT_PROPRIO_HIST_LEN = 30  # 本体感觉历史长度（时间步数）
DEFAULT_OBS_HIST_LEN = 80  # 观测历史长度

# 计算维度
PROPRIO_HIST_DIM = PROPRIO_DIM * DEFAULT_PROPRIO_HIST_LEN  # 42 × 30 = 1260
OBS_WITH_CONTACT_DIM = PROPRIO_DIM + CONTACT_DIM  # 42 + 15 = 57
OBS_WITH_CONTACT_FINGERTIP_DIM = PROPRIO_DIM + CONTACT_DIM + FINGERTIP_POS_DIM  # 42 + 15 + 15 = 72

# ============================================================
# 指尖和接触点名称（从 linker_hand_hora.py 导入）
# ============================================================

FINGERTIP_LINK_NAMES = [
    "little_joint3",
    "ring_joint3",
    "middle_joint3",
    "index_joint3",
    "thumb_joint4"
]
CONTACT_LINK_NAMES = [
    "little_joint3",
    "ring_joint3",
    "middle_joint3",
    "index_joint3",
    "thumb_joint4",
]

# ============================================================
# Privileged Information Dimensions and Layout
# ============================================================

# 特权信息总维度（根据 linker_hand_hora.py:1324-1346）
PRIV_INFO_DIM = 61  # 默认总维度（实际维度取决于启用的项）

# 固定项（总是启用）
PRIV_OBJ_POS_START = 0
PRIV_OBJ_POS_DIM = 3  # 物体位置

PRIV_OBJ_SCALE_START = 3
PRIV_OBJ_SCALE_DIM = 1  # 物体缩放

PRIV_OBJ_MASS_START = 4
PRIV_OBJ_MASS_DIM = 1  # 物体质量

PRIV_OBJ_FRICTION_START = 5
PRIV_OBJ_FRICTION_DIM = 1  # 物体摩擦系数

PRIV_OBJ_COM_START = 6
PRIV_OBJ_COM_DIM = 3  # 物体质心

# 可选项（起始索引为 9，根据启用情况动态添加）
PRIV_DYNAMIC_START = 9

# 各可选项的维度
PRIV_OBJ_ROT_DIM = 4  # 物体旋转（四元数）
PRIV_OBJ_LINVEL_DIM = 3  # 物体线速度
PRIV_OBJ_ANGVEL_DIM = 3  # 物体角速度
PRIV_FINGERTIP_POS_DIM_IN_PRIV = FINGERTIP_CNT * 3  # 指尖位置 5×3=15
PRIV_FINGERTIP_ROT_DIM = FINGERTIP_CNT * 4  # 指尖旋转 5×4=20
PRIV_FINGERTIP_LINVEL_DIM = FINGERTIP_CNT * 3  # 指尖线速度 5×3=15
PRIV_FINGERTIP_ANGVEL_DIM = FINGERTIP_CNT * 3  # 指尖角速度 5×3=15
PRIV_HAND_SCALE_DIM = 1  # 手的缩放
PRIV_OBJ_RESTITUTION_DIM = 1  # 物体弹性系数
PRIV_TACTILE_DIM = CONTACT_DIM  # 触觉信息 (可变)

# ============================================================
# Point Cloud Configuration
# ============================================================

POINT_CLOUD_NUM_POINTS = 100  # 采样点数
POINT_CLOUD_FEATURE_DIM = 3  # Teacher: xyz
POINT_CLOUD_FEATURE_DIM_STUDENT = 6  # Student: xyz + rgb/features

# PointNet 输出维度 (编码后的特征维度)
POINTNET_OUTPUT_DIM = 256  # PointNet encoder 的输出特征维度

# ============================================================
# Object Endpoints Configuration (for vision-based tracking)
# ============================================================

OBJ_ENDS_HIST_LEN = 3  # 物体端点历史长度
OBJ_ENDS_DIM = 6  # 2 endpoints × 3D = 6
OBJ_ENDS_TOTAL_DIM = OBJ_ENDS_HIST_LEN * OBJ_ENDS_DIM  # 3 × 6 = 18

# ============================================================
# Tactile History Configuration
# ============================================================

# Tactile history 默认长度 (与 proprio_hist 相同)
TACTILE_HIST_LEN = 30  # propHistoryLen from config

# StudentActorCritic 使用最后 N 个时间步的 tactile 数据
# 注意: 原始代码有 bug,使用 [:, :-1, :].squeeze(1) 会得到 [batch, 29, 15]
# 这里明确定义使用最后 2 个时间步
TACTILE_USED_TIMESTEPS = 2  # 使用最后 2 个时间步的触觉数据

# StudentActorCritic 中 tactile 特征的维度
# 等于 TACTILE_USED_TIMESTEPS × CONTACT_DIM
TACTILE_FEATURE_DIM = TACTILE_USED_TIMESTEPS * CONTACT_DIM  # 2 × 15 = 30

# ============================================================
# Student Privileged Information (依赖于上面的定义)
# ============================================================

# Student 可见的"特权"信息（用于监督学习）
# 注意：从 Allegro Hand (4指) 迁移到 Linker Hand (5指) 的维度变化：
#   - 旧版（4指）: 4 × 3D = 12
#   - 新版（5指）: 5 × 3D = 15 (FINGERTIP_POS_DIM)
# Student 的 priv info 实际使用 fingertip_position，从 Teacher 的 priv_info 中提取
STUDENT_PRIV_DIM_LEGACY = 12  # 旧版 Allegro Hand 4指 × 3D（已废弃，仅用于文档）
STUDENT_PRIV_DIM = FINGERTIP_POS_DIM  # Linker Hand: 5指 × 3D = 15
STUDENT_PRIV_WITH_TACTILE_DIM = STUDENT_PRIV_DIM + TACTILE_FEATURE_DIM  # 15 + 30 = 45

# ============================================================
# 网络架构相关维度
# ============================================================

# Temporal Transformer 配置
TEMPORAL_FUSION_INPUT_DIM = PROPRIO_DIM  # 42 for Linker Hand
TEMPORAL_FUSION_OUTPUT_DIM = 32  # 经过 temporal 编码后的维度
TEMPORAL_FUSION_FINAL_DIM = 40  # 经过 all_fuse 后的维度

# MLP 隐藏层配置（默认）
DEFAULT_ACTOR_UNITS = [512, 256, 128]
DEFAULT_PRIV_MLP_UNITS = [256, 128, 64]
DEFAULT_POINT_MLP_UNITS = [64, 128, 256]
DEFAULT_CONTACT_MLP_UNITS = [64, 128]

# ============================================================
# 环境配置
# ============================================================

DEFAULT_CONTROL_FREQ_INV = 10  # 控制频率倒数（每10个仿真步一次控制）
DEFAULT_SIM_FREQ = 200  # Hz

# ============================================================
# 辅助函数
# ============================================================

def get_proprio_slice_range():
    """返回 proprio（位置+目标）在 obs_buf 中的切片范围"""
    return slice(0, PROPRIO_DIM)

def get_contact_slice_range():
    """返回 contact 在 obs_buf 中的切片范围"""
    return slice(PROPRIO_DIM, PROPRIO_DIM + CONTACT_DIM)

def get_fingertip_slice_range():
    """返回 fingertip 在 obs_buf 中的切片范围"""
    start = PROPRIO_DIM + CONTACT_DIM
    return slice(start, start + FINGERTIP_POS_DIM)

def get_priv_info_fingertip_slice(enable_obj_linvel=False):
    """
    返回 priv_info 中 fingertip 位置的切片范围
    
    Args:
        enable_obj_linvel: 是否启用了 obj_linvel (默认False,匹配 train_teacher.sh 配置)
    
    注意：实际位置取决于环境配置中启用了哪些项
    
    priv_in组成： 布局（Linker Hand 21 DoF, 5 Fingers）:
    ┌────────────────────────────────────────────────────────────────┐
    │ 固定部分 (0-9):                                                │
    │   [0:3]   obj_position                                         │
    │   [3:4]   obj_scale                                            │
    │   [4:5]   obj_mass                                             │
    │   [5:6]   obj_friction                                         │
    │   [6:9]   obj_com                                              │
    ├────────────────────────────────────────────────────────────────┤
    │ 动态部分 (9+，取决于启用的项):                                │
    │   train_teacher.sh 默认配置:                                   │
    │   [9:13]  obj_orientation (4)                                  │
    │   [13:16] obj_angvel (3)                                       │
    │   [16:31] fingertip_position (15 = 5 fingers × 3D)             │
    │   [31:32] obj_restitution (1)                                  │
    │   [32:47] tactile (15 = 5 fingers × 3D force)                  │
    │                                                                 │
    │   如果启用 obj_linvel，则在 obj_angvel 后插入 3维:            │
    │   [9:13]  obj_orientation (4)                                  │
    │   [13:16] obj_linvel (3)         ← 插入位置                    │
    │   [16:19] obj_angvel (3)                                       │
    │   [19:34] fingertip_position (15)                              │
    │   ...                                                           │
    └────────────────────────────────────────────────────────────────┘
    
    对比旧版 Allegro Hand (16 DoF, 4 Fingers):
      - fingertip_position 维度: 12 (4×3) -> 15 (5×3)
      - tactile 维度: 32 (密集传感器) -> 15 (5×3)
    
    Returns:
        slice: fingertip 位置的切片范围
    """
    # 根据 linker_hand_hora.py 的 priv_info 构建逻辑 (line 1324-1370)
    # 动态部分从索引 9 开始
    fingertip_start = PRIV_DYNAMIC_START  # 9
    
    # 默认 train_teacher.sh 配置中启用的项（按顺序）
    # 1. obj_orientation: 4
    fingertip_start += PRIV_OBJ_ROT_DIM  # 9 + 4 = 13
    
    # 2. obj_linvel: 3 (可选，默认未启用)
    if enable_obj_linvel:
        fingertip_start += PRIV_OBJ_LINVEL_DIM  # 13 + 3 = 16
    
    # 3. obj_angvel: 3
    fingertip_start += PRIV_OBJ_ANGVEL_DIM  # 13 + 3 = 16 (默认，无linvel)
                                             # 或 16 + 3 = 19 (启用linvel)
    
    # 4. fingertip_position 维度
    fingertip_end = fingertip_start + PRIV_FINGERTIP_POS_DIM_IN_PRIV
    # 默认: [16:31] (16 + 15 = 31)
    # 启用linvel: [19:34] (19 + 15 = 34)
    
    return slice(fingertip_start, fingertip_end)

def validate_dimensions():
    """验证维度配置的一致性"""
    assert NUM_FINGERS == FINGERTIP_CNT, f"NUM_FINGERS ({NUM_FINGERS}) != FINGERTIP_CNT ({FINGERTIP_CNT})"
    assert CONTACT_DIM == NUM_FINGERS * CONTACT_SENSOR_DIM, \
        f"CONTACT_DIM ({CONTACT_DIM}) != NUM_FINGERS * CONTACT_SENSOR_DIM ({NUM_FINGERS * CONTACT_SENSOR_DIM})"
    assert PROPRIO_DIM == 2 * NUM_DOF, f"PROPRIO_DIM ({PROPRIO_DIM}) != 2 * NUM_DOF ({2 * NUM_DOF})"
    assert FINGERTIP_POS_DIM == FINGERTIP_CNT * 3, \
        f"FINGERTIP_POS_DIM ({FINGERTIP_POS_DIM}) != FINGERTIP_CNT * 3 ({FINGERTIP_CNT * 3})"
    assert len(FINGERTIP_LINK_NAMES) == FINGERTIP_CNT, \
        f"len(FINGERTIP_LINK_NAMES) ({len(FINGERTIP_LINK_NAMES)}) != FINGERTIP_CNT ({FINGERTIP_CNT})"
    assert len(CONTACT_LINK_NAMES) == NUM_FINGERS, \
        f"len(CONTACT_LINK_NAMES) ({len(CONTACT_LINK_NAMES)}) != NUM_FINGERS ({NUM_FINGERS})"
    
    print("✓ All dimension validations passed!")
    print(f"  NUM_DOF: {NUM_DOF}")
    print(f"  NUM_FINGERS: {NUM_FINGERS}")
    print(f"  CONTACT_DIM: {CONTACT_DIM}")
    print(f"  PROPRIO_DIM: {PROPRIO_DIM}")
    print(f"  FINGERTIP_POS_DIM: {FINGERTIP_POS_DIM}")

# ============================================================
# 兼容性映射（旧代码迁移用）
# ============================================================

# 旧的硬编码值到新常量的映射
LEGACY_MAPPINGS = {
    16: NUM_DOF,  # 旧的 Allegro Hand DoF
    32: PROPRIO_DIM,  # 2 * NUM_DOF
    20: CONTACT_DIM,  # 旧的 4 fingers × 5 sensors
    4: NUM_FINGERS,  # 旧的手指数
}

def warn_legacy_usage(old_value, context=""):
    """警告使用了旧的硬编码值"""
    if old_value in LEGACY_MAPPINGS:
        new_value = LEGACY_MAPPINGS[old_value]
        print(f"⚠️  WARNING: Using legacy hardcoded value {old_value} in {context}")
        print(f"   Should use constant instead. New value: {new_value}")

# ============================================================
# 自动验证（导入时执行）
# ============================================================

if __name__ == '__main__':
    validate_dimensions()
