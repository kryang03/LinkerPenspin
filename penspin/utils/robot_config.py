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
NUM_DOF = 21  # 手部关节自由度数量
NUM_FINGERS = 5  # 手指数量
FINGERTIP_CNT = 5  # 指尖数量（等于手指数量）

# ============================================================
# Flying Hand Configuration (6 DoF Base + 21 DoF Hand)
# ============================================================
# 为转笔任务设计的浮空底座配置
# 通过添加 6 个虚拟自由度允许手腕在空间中移动和旋转
# 但通过严格的速度和位置限制防止"作弊"（利用惯性甩笔）

NUM_FLYING_DOF = 6  # 浮空底座自由度数量 (3平移 + 3旋转)
NUM_TOTAL_DOF_FLYING = NUM_DOF + NUM_FLYING_DOF  # 总 DOF = 27

# Flying base 关节名称（按 URDF 中定义的顺序）
FLYING_DOF_NAMES = [
    "virtual_px",   # 平移 X (prismatic)
    "virtual_py",   # 平移 Y (prismatic)
    "virtual_pz",   # 平移 Z (prismatic)
    "virtual_rx",   # 旋转 Roll (revolute)
    "virtual_ry",   # 旋转 Pitch (revolute)
    "virtual_rz",   # 旋转 Yaw (revolute)
]

# Flying base 的关节索引（在完整 DOF 数组中的位置）
# 在 URDF 中 flying base 在最前面，所以是 [0-5]
FLYING_DOF_INDICES = list(range(0, NUM_FLYING_DOF))  # [0, 1, 2, 3, 4, 5]

# Flying base 的子类别索引
FLYING_TRANSLATION_INDICES = [0, 1, 2]  # px, py, pz
FLYING_ROTATION_INDICES = [3, 4, 5]     # rx, ry, rz

# Flying base 的默认高度（Z 轴初始位置）
# 根据 gen_flying_hand.py 中的设置：z_lower=0.30, z_upper=0.40
FLYING_DEFAULT_HEIGHT = 0.35  # 默认高度 35cm
FLYING_HEIGHT_LOWER = 0.30    # 最低高度 30cm
FLYING_HEIGHT_UPPER = 0.40    # 最高高度 40cm

# Flying base 的位置限制
FLYING_XY_LIMIT = 0.05  # XY 平面移动限制 ±5cm

# Flying base 的速度限制（防作弊配置）
FLYING_LINEAR_VELOCITY_LIMIT = 0.1   # m/s (极慢，防止甩手腕)
FLYING_ANGULAR_VELOCITY_LIMIT = 2.0  # rad/s (约 115°/s)

# ============================================================
# Flying base 相对限位配置 (Relative Limit Configuration)
# ============================================================
# 相对限位: 动作空间相对于初始位置的对称限位
# 确保无论初始位置在哪里，动作空间在初始位置两侧对称
# 最终限位 = max(绝对下限, 初始-相对) ~ min(绝对上限, 初始+相对)
# ============================================================
FLYING_RELATIVE_XY_LIMIT = 0.05     # XY 相对限位 ±5cm（与绝对限位相同）
FLYING_RELATIVE_Z_LIMIT = 0.05      # Z 相对限位 ±5cm
FLYING_RELATIVE_ROT_LIMIT = 0.5     # 旋转相对限位 ±0.5 rad（约±28.6°）

# ============================================================
# 关节索引配置 (Joint Index Configuration)
# ============================================================
# 重要: IsaacGym 加载 URDF 后的关节顺序是按字母顺序排列的！
# 而不是按 URDF 文件中的声明顺序。
#
# 实际 DOF 顺序 (21 DoF 非 Flying 版本):
#   index (食指):    joint 0-3   (4 DoF)  - indices [0, 1, 2, 3]
#   little (小拇指): joint 4-7   (4 DoF)  - indices [4, 5, 6, 7]
#   middle (中指):   joint 8-11  (4 DoF)  - indices [8, 9, 10, 11]
#   ring (无名指):   joint 12-15 (4 DoF)  - indices [12, 13, 14, 15]
#   thumb (拇指):    joint 16-20 (5 DoF)  - indices [16, 17, 18, 19, 20]
#
# Flying Hand (27 DoF) 顺序:
#   virtual (基座):  joint 0-5   (6 DoF)
#   index (食指):    joint 6-9   (4 DoF)
#   little (小拇指): joint 10-13 (4 DoF)
#   middle (中指):   joint 14-17 (4 DoF)
#   ring (无名指):   joint 18-21 (4 DoF)
#   thumb (拇指):    joint 22-26 (5 DoF)
# ============================================================

# 各手指的关节索引范围 (按 IsaacGym 字母顺序)
INDEX_FINGER_INDICES = list(range(0, 4))     # [0, 1, 2, 3] - 食指
LITTLE_FINGER_INDICES = list(range(4, 8))    # [4, 5, 6, 7] - 小拇指
MIDDLE_FINGER_INDICES = list(range(8, 12))   # [8, 9, 10, 11] - 中指
RING_FINGER_INDICES = list(range(12, 16))    # [12, 13, 14, 15] - 无名指
THUMB_FINGER_INDICES = list(range(16, 21))   # [16, 17, 18, 19, 20] - 拇指

# 各手指的DoF数量 (按 IsaacGym 字母顺序)
INDEX_FINGER_DOF = 4   # 食指
LITTLE_FINGER_DOF = 4  # 小拇指
MIDDLE_FINGER_DOF = 4  # 中指
RING_FINGER_DOF = 4    # 无名指
THUMB_FINGER_DOF = 5   # 拇指

# 禁用无名指和小拇指后的动作空间配置
# 激活的手指: index(4) + middle(4) + thumb(5) = 13 DoF
# 注意: 按 IsaacGym 字母顺序，index 在最前面 [0-3]
ACTIVE_FINGER_INDICES_REDUCED = INDEX_FINGER_INDICES + MIDDLE_FINGER_INDICES + THUMB_FINGER_INDICES  # [0-3] + [8-11] + [16-20]
DISABLED_FINGER_INDICES = LITTLE_FINGER_INDICES + RING_FINGER_INDICES  # [4-7] + [12-15]
NUM_DOF_REDUCED = len(ACTIVE_FINGER_INDICES_REDUCED)  # 13

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
# 
# priv_info 布局说明：
# - 固定部分：obj_position(3), obj_mass(1), obj_friction(1), obj_com(3) = 8维
# - 动态部分：根据 yaml 配置启用的项按顺序添加
# 
# 注意：obj_scale 已从固定部分移除（LinkerPen 使用固定尺寸）
# ============================================================

# 固定项配置（总是启用，无需 yaml 开关）
PRIV_FIXED_ITEMS = {
    'obj_position': 3,   # 物体位置 [0:3]
    'obj_mass': 1,       # 物体质量 [3:4]
    'obj_friction': 1,   # 物体摩擦系数 [4:5]
    'obj_com': 3,        # 物体质心 [5:8]
}

# 动态项配置（需要 yaml 开关控制，按此顺序添加到 priv_info）
# 格式：(yaml配置名, 维度)
PRIV_DYNAMIC_ITEMS = [
    ('obj_orientation', 4),                    # 物体姿态（四元数）
    ('obj_linvel', 3),                         # 物体线速度
    ('obj_angvel', 3),                         # 物体角速度
    ('fingertip_position', FINGERTIP_CNT * 3), # 指尖位置 5×3=15
    ('fingertip_orientation', FINGERTIP_CNT * 4), # 指尖姿态 5×4=20
    ('fingertip_linvel', FINGERTIP_CNT * 3),   # 指尖线速度 5×3=15
    ('fingertip_angvel', FINGERTIP_CNT * 3),   # 指尖角速度 5×3=15
    ('hand_scale', 1),                         # 手的缩放
    ('obj_restitution', 1),                    # 物体弹性系数
    ('tactile', CONTACT_DIM),                  # 触觉信息 (15)
]

# 计算固定部分的总维度和起始索引
PRIV_FIXED_DIM = sum(PRIV_FIXED_ITEMS.values())  # 8
PRIV_DYNAMIC_START = PRIV_FIXED_DIM  # 动态部分从索引 8 开始

# 固定项的具体索引（用于硬编码访问）
_idx = 0
PRIV_OBJ_POS_START = _idx
PRIV_OBJ_POS_DIM = PRIV_FIXED_ITEMS['obj_position']
_idx += PRIV_OBJ_POS_DIM

PRIV_OBJ_MASS_START = _idx
PRIV_OBJ_MASS_DIM = PRIV_FIXED_ITEMS['obj_mass']
_idx += PRIV_OBJ_MASS_DIM

PRIV_OBJ_FRICTION_START = _idx
PRIV_OBJ_FRICTION_DIM = PRIV_FIXED_ITEMS['obj_friction']
_idx += PRIV_OBJ_FRICTION_DIM

PRIV_OBJ_COM_START = _idx
PRIV_OBJ_COM_DIM = PRIV_FIXED_ITEMS['obj_com']
_idx += PRIV_OBJ_COM_DIM

assert _idx == PRIV_DYNAMIC_START, f"Fixed items index mismatch: {_idx} != {PRIV_DYNAMIC_START}"

# 各动态项的维度（用于外部访问）
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

# 默认总维度（仅供参考，实际取决于启用的项）
PRIV_INFO_DIM = 64

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


class PrivInfoLayout:
    """
    priv_info 布局管理类
    
    统一管理 priv_info 的索引计算，避免硬编码。
    使用方法：
        layout = PrivInfoLayout(enable_obj_linvel=True, enable_ft_linvel=True, ...)
        fingertip_slice = layout.get_slice('fingertip_position')
        total_dim = layout.total_dim
        
    布局结构（新版，移除了 obj_scale）：
    ┌────────────────────────────────────────────────────────────────┐
    │ 固定部分 (0-8):                                                │
    │   [0:3]   obj_position                                         │
    │   [3:4]   obj_mass                                             │
    │   [4:5]   obj_friction                                         │
    │   [5:8]   obj_com                                              │
    ├────────────────────────────────────────────────────────────────┤
    │ 动态部分 (8+，按以下顺序添加启用的项):                          │
    │   obj_orientation (4)         - 物体姿态                       │
    │   obj_linvel (3)              - 物体线速度 [新增默认启用]       │
    │   obj_angvel (3)              - 物体角速度                     │
    │   fingertip_position (15)     - 指尖位置                       │
    │   fingertip_orientation (20)  - 指尖姿态                       │
    │   fingertip_linvel (15)       - 指尖线速度 [新增默认启用]       │
    │   fingertip_angvel (15)       - 指尖角速度                     │
    │   hand_scale (1)              - 手部缩放                       │
    │   obj_restitution (1)         - 物体弹性系数                   │
    │   tactile (15)                - 触觉力信息                     │
    └────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, 
                 enable_obj_orientation=True,
                 enable_obj_linvel=True,      # 新默认值：True
                 enable_obj_angvel=True,
                 enable_ft_pos=True,
                 enable_ft_orientation=False,
                 enable_ft_linvel=True,       # 新默认值：True
                 enable_ft_angvel=False,
                 enable_hand_scale=False,
                 enable_obj_restitution=True,
                 enable_tactile=True):
        """
        初始化 priv_info 布局
        
        Args:
            enable_obj_orientation: 是否启用物体姿态
            enable_obj_linvel: 是否启用物体线速度 (新默认开启)
            enable_obj_angvel: 是否启用物体角速度
            enable_ft_pos: 是否启用指尖位置
            enable_ft_orientation: 是否启用指尖姿态
            enable_ft_linvel: 是否启用指尖线速度 (新默认开启)
            enable_ft_angvel: 是否启用指尖角速度
            enable_hand_scale: 是否启用手部缩放
            enable_obj_restitution: 是否启用物体弹性
            enable_tactile: 是否启用触觉信息
        """
        # 保存配置
        self.config = {
            'enable_obj_orientation': enable_obj_orientation,
            'enable_obj_linvel': enable_obj_linvel,
            'enable_obj_angvel': enable_obj_angvel,
            'enable_ft_pos': enable_ft_pos,
            'enable_ft_orientation': enable_ft_orientation,
            'enable_ft_linvel': enable_ft_linvel,
            'enable_ft_angvel': enable_ft_angvel,
            'enable_hand_scale': enable_hand_scale,
            'enable_obj_restitution': enable_obj_restitution,
            'enable_tactile': enable_tactile,
        }
        
        # 构建索引映射
        self._build_layout()
    
    def _build_layout(self):
        """构建 priv_info 的索引布局"""
        self.slices = {}
        
        # 固定部分（总是存在）
        idx = 0
        for name, dim in PRIV_FIXED_ITEMS.items():
            self.slices[name] = (idx, idx + dim)
            idx += dim
        
        # 动态部分（根据配置添加）
        config_mapping = {
            'obj_orientation': 'enable_obj_orientation',
            'obj_linvel': 'enable_obj_linvel',
            'obj_angvel': 'enable_obj_angvel',
            'fingertip_position': 'enable_ft_pos',
            'fingertip_orientation': 'enable_ft_orientation',
            'fingertip_linvel': 'enable_ft_linvel',
            'fingertip_angvel': 'enable_ft_angvel',
            'hand_scale': 'enable_hand_scale',
            'obj_restitution': 'enable_obj_restitution',
            'tactile': 'enable_tactile',
        }
        
        for name, dim in PRIV_DYNAMIC_ITEMS:
            config_key = config_mapping.get(name)
            if config_key and self.config.get(config_key, False):
                self.slices[name] = (idx, idx + dim)
                idx += dim
        
        self.total_dim = idx
    
    def get_slice(self, name):
        """
        获取指定项在 priv_info 中的切片
        
        Args:
            name: 项名称，如 'fingertip_position', 'obj_linvel' 等
            
        Returns:
            slice: 可用于索引的切片对象
            
        Raises:
            KeyError: 如果该项未启用或不存在
        """
        if name not in self.slices:
            raise KeyError(f"'{name}' 未在 priv_info 中启用或不存在。"
                          f"已启用的项: {list(self.slices.keys())}")
        start, end = self.slices[name]
        return slice(start, end)
    
    def get_range(self, name):
        """
        获取指定项在 priv_info 中的索引范围
        
        Args:
            name: 项名称
            
        Returns:
            tuple: (start_index, end_index)
        """
        if name not in self.slices:
            raise KeyError(f"'{name}' 未在 priv_info 中启用")
        return self.slices[name]
    
    def is_enabled(self, name):
        """检查某项是否启用"""
        return name in self.slices
    
    def get_enabled_items(self):
        """返回所有启用的项名称列表"""
        return list(self.slices.keys())
    
    def get_priv_info_dict(self):
        """
        返回与 linker_hand_hora.py 中 priv_info_dict 兼容的字典
        
        Returns:
            dict: {name: (start, end), ...}
        """
        return dict(self.slices)
    
    def print_layout(self):
        """打印当前 priv_info 布局"""
        print("\n" + "="*60)
        print("priv_info 布局")
        print("="*60)
        print(f"{'项名称':<25} {'索引范围':<15} {'维度'}")
        print("-"*60)
        for name, (start, end) in self.slices.items():
            dim = end - start
            print(f"{name:<25} [{start:>3}:{end:<3}]        {dim}")
        print("-"*60)
        print(f"{'总维度:':<25} {self.total_dim}")
        print("="*60 + "\n")
    
    @classmethod
    def from_env_config(cls, env):
        """
        从环境对象创建 PrivInfoLayout
        
        Args:
            env: LinkerHandHora 环境对象
            
        Returns:
            PrivInfoLayout: 与环境配置匹配的布局对象
        """
        return cls(
            enable_obj_orientation=getattr(env, 'enable_priv_obj_orientation', False),
            enable_obj_linvel=getattr(env, 'enable_priv_obj_linvel', False),
            enable_obj_angvel=getattr(env, 'enable_priv_obj_angvel', False),
            enable_ft_pos=getattr(env, 'enable_priv_fingertip_position', False),
            enable_ft_orientation=getattr(env, 'enable_priv_fingertip_orientation', False),
            enable_ft_linvel=getattr(env, 'enable_priv_fingertip_linvel', False),
            enable_ft_angvel=getattr(env, 'enable_priv_fingertip_angvel', False),
            enable_hand_scale=getattr(env, 'enable_priv_hand_scale', False),
            enable_obj_restitution=getattr(env, 'enable_priv_obj_restitution', False),
            enable_tactile=getattr(env, 'enable_priv_tactile', False),
        )
    
    @classmethod
    def from_yaml_config(cls, priv_info_config):
        """
        从 YAML 配置字典创建 PrivInfoLayout
        
        Args:
            priv_info_config: YAML 中 privInfo 部分的配置字典
            
        Returns:
            PrivInfoLayout: 与配置匹配的布局对象
        """
        return cls(
            enable_obj_orientation=priv_info_config.get('enable_obj_orientation', False),
            enable_obj_linvel=priv_info_config.get('enable_obj_linvel', False),
            enable_obj_angvel=priv_info_config.get('enable_obj_angvel', False),
            enable_ft_pos=priv_info_config.get('enable_ft_pos', False),
            enable_ft_orientation=priv_info_config.get('enable_ft_orientation', False),
            enable_ft_linvel=priv_info_config.get('enable_ft_linvel', False),
            enable_ft_angvel=priv_info_config.get('enable_ft_angvel', False),
            enable_hand_scale=priv_info_config.get('enable_hand_scale', False),
            enable_obj_restitution=priv_info_config.get('enable_obj_restitution', False),
            enable_tactile=priv_info_config.get('enable_tactile', False),
        )


def get_priv_info_fingertip_slice(
    enable_obj_orientation=True,
    enable_obj_linvel=True,       # 默认值更新为 True
    enable_obj_angvel=True,
    enable_ft_pos=True,
    enable_ft_orientation=False,
    enable_ft_linvel=True,        # 默认值更新为 True
    enable_ft_angvel=False,
    enable_hand_scale=False,
    enable_obj_restitution=True,
    enable_tactile=True
):
    """
    返回 priv_info 中 fingertip 位置的切片范围
    
    注意：此函数为向后兼容保留，推荐使用 PrivInfoLayout 类
    
    新版布局（移除 obj_scale，固定部分从9维减少到8维）：
    ┌────────────────────────────────────────────────────────────────┐
    │ 固定部分 (0-8):                                                │
    │   [0:3]   obj_position                                         │
    │   [3:4]   obj_mass                                             │
    │   [4:5]   obj_friction                                         │
    │   [5:8]   obj_com                                              │
    ├────────────────────────────────────────────────────────────────┤
    │ 动态部分 (8+，取决于启用的项):                                  │
    │   obj_orientation (4)  -> obj_linvel (3)  -> obj_angvel (3)   │
    │   -> fingertip_position (15) -> fingertip_orientation (20)    │
    │   -> fingertip_linvel (15) -> fingertip_angvel (15)           │
    │   -> hand_scale (1) -> obj_restitution (1) -> tactile (15)    │
    │                                                                 │
    │ train_teacher.sh 新默认配置:                                   │
    │   [8:12]  obj_orientation (4)                                  │
    │   [12:15] obj_linvel (3)             ✓ 新增启用                │
    │   [15:18] obj_angvel (3)                                       │
    │   [18:33] fingertip_position (15)    ← 返回此范围             │
    │   [33:48] fingertip_linvel (15)      ✓ 新增启用                │
    │   [48:49] obj_restitution (1)                                  │
    │   [49:64] tactile (15)                                         │
    └────────────────────────────────────────────────────────────────┘
    
    Returns:
        slice: fingertip_position 在 priv_info 中的切片范围
    """
    layout = PrivInfoLayout(
        enable_obj_orientation=enable_obj_orientation,
        enable_obj_linvel=enable_obj_linvel,
        enable_obj_angvel=enable_obj_angvel,
        enable_ft_pos=enable_ft_pos,
        enable_ft_orientation=enable_ft_orientation,
        enable_ft_linvel=enable_ft_linvel,
        enable_ft_angvel=enable_ft_angvel,
        enable_hand_scale=enable_hand_scale,
        enable_obj_restitution=enable_obj_restitution,
        enable_tactile=enable_tactile,
    )
    return layout.get_slice('fingertip_position')


def get_priv_config_from_env(env):
    """
    从环境对象中提取 priv_info 配置
    
    Args:
        env: LinkerHandHora 环境对象，包含 enable_priv_* 属性
        
    Returns:
        dict: 包含所有 priv_info 配置的字典，可直接传递给 PrivInfoLayout 或 get_priv_info_fingertip_slice()
    """
    return {
        'enable_obj_orientation': getattr(env, 'enable_priv_obj_orientation', False),
        'enable_obj_linvel': getattr(env, 'enable_priv_obj_linvel', False),
        'enable_obj_angvel': getattr(env, 'enable_priv_obj_angvel', False),
        'enable_ft_pos': getattr(env, 'enable_priv_fingertip_position', False),
        'enable_ft_orientation': getattr(env, 'enable_priv_fingertip_orientation', False),
        'enable_ft_linvel': getattr(env, 'enable_priv_fingertip_linvel', False),
        'enable_ft_angvel': getattr(env, 'enable_priv_fingertip_angvel', False),
        'enable_hand_scale': getattr(env, 'enable_priv_hand_scale', False),
        'enable_obj_restitution': getattr(env, 'enable_priv_obj_restitution', False),
        'enable_tactile': getattr(env, 'enable_priv_tactile', False),
    }

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
# 动作空间映射类 (Action Space Mapping)
# ============================================================

class ActionSpaceMapper:
    """
    动作空间映射器
    
    用于管理不同配置下的动作空间，支持：
    1. Flying Hand (6 DoF 浮空底座)
    2. 禁用无名指和小拇指
    
    动作空间维度组合：
    ┌─────────────────────────────────────────────────────────────────┐
    │ 配置                           │ 动作维度                        │
    ├─────────────────────────────────────────────────────────────────┤
    │ Flying + 完整手指              │ 6 + 21 = 27 DoF                │
    │ Flying + 禁用 ring/little      │ 6 + 13 = 19 DoF                │
    │ 无 Flying + 完整手指           │ 21 DoF                          │
    │ 无 Flying + 禁用 ring/little   │ 13 DoF                          │
    └─────────────────────────────────────────────────────────────────┘
    
    Flying Hand 动作空间结构 (27 DoF):
    ┌────────────────────────────────────────────────────────────────┐
    │ Flying Base (6 DoF):                                           │
    │   [0]     virtual_px  - 平移 X                                 │
    │   [1]     virtual_py  - 平移 Y                                 │
    │   [2]     virtual_pz  - 平移 Z                                 │
    │   [3]     virtual_rx  - 旋转 Roll                              │
    │   [4]     virtual_ry  - 旋转 Pitch                             │
    │   [5]     virtual_rz  - 旋转 Yaw                               │
    ├────────────────────────────────────────────────────────────────┤
    │ Hand Joints (21 DoF):                                          │
    │   [6-9]   little (小拇指) - 可禁用                              │
    │   [10-13] ring (无名指)   - 可禁用                              │
    │   [14-17] middle (中指)   - 总是激活                            │
    │   [18-21] index (食指)    - 总是激活                            │
    │   [22-26] thumb (拇指)    - 总是激活                            │
    └────────────────────────────────────────────────────────────────┘
    
    使用方法：
        mapper = ActionSpaceMapper(
            disable_ring_little=True,
            flying_hand_enabled=True
        )
        
        # 策略输出 -> 仿真输入
        full_actions = mapper.expand_actions(policy_actions, init_pose)
        
        # 获取动作维度
        action_dim = mapper.get_action_dim()
    """
    
    def __init__(self, disable_ring_little=False, flying_hand_enabled=False):
        """
        初始化动作空间映射器
        
        Args:
            disable_ring_little: 是否禁用无名指和小拇指
            flying_hand_enabled: 是否启用 Flying Hand (6 DoF 浮空底座)
        """
        self.disable_ring_little = disable_ring_little
        self.flying_hand_enabled = flying_hand_enabled
        
        # 计算手部动作维度
        if disable_ring_little:
            self.hand_action_dim = NUM_DOF_REDUCED  # 13
            # 手部激活的关节索引（相对于手部 DOF）
            self.hand_active_indices = ACTIVE_FINGER_INDICES_REDUCED  # [8-20]
            self.hand_disabled_indices = DISABLED_FINGER_INDICES      # [0-7]
        else:
            self.hand_action_dim = NUM_DOF  # 21
            self.hand_active_indices = list(range(NUM_DOF))  # [0-20]
            self.hand_disabled_indices = []
        
        # 计算 Flying base 动作维度
        if flying_hand_enabled:
            self.flying_action_dim = NUM_FLYING_DOF  # 6
            self.flying_indices = FLYING_DOF_INDICES  # [0-5]
        else:
            self.flying_action_dim = 0
            self.flying_indices = []
        
        # 总动作维度（策略网络输出）
        self.action_dim = self.flying_action_dim + self.hand_action_dim
        
        # 总 DOF 维度（仿真环境）
        if flying_hand_enabled:
            self.total_dof = NUM_TOTAL_DOF_FLYING  # 27
        else:
            self.total_dof = NUM_DOF  # 21
        
        self._build_mapping()
    
    def _build_mapping(self):
        """构建动作索引映射"""
        # 从策略动作索引到仿真 DOF 索引的映射
        self.policy_to_sim = {}
        
        policy_idx = 0
        
        # Flying base 部分（如果启用）
        if self.flying_hand_enabled:
            for sim_idx in self.flying_indices:
                self.policy_to_sim[policy_idx] = sim_idx
                policy_idx += 1
        
        # 手部关节部分
        offset = NUM_FLYING_DOF if self.flying_hand_enabled else 0
        for hand_idx in self.hand_active_indices:
            sim_idx = offset + hand_idx
            self.policy_to_sim[policy_idx] = sim_idx
            policy_idx += 1
        
        # 反向映射
        self.sim_to_policy = {v: k for k, v in self.policy_to_sim.items()}
        
        # 计算仿真中被禁用的关节索引
        self.sim_disabled_indices = []
        if self.hand_disabled_indices:
            offset = NUM_FLYING_DOF if self.flying_hand_enabled else 0
            self.sim_disabled_indices = [offset + idx for idx in self.hand_disabled_indices]
    
    def get_action_dim(self):
        """返回策略网络的动作空间维度"""
        return self.action_dim
    
    def get_total_dof(self):
        """返回仿真环境的总 DOF 数量"""
        return self.total_dof
    
    def expand_actions(self, policy_actions, init_pose=None):
        """
        将策略动作扩展为完整的仿真 DOF 动作
        
        Args:
            policy_actions: [batch, action_dim] 策略网络输出的动作
            init_pose: [batch, total_dof] 初始姿态（用于锁定禁用关节，可选）
            
        Returns:
            full_actions: [batch, total_dof] 完整的仿真动作
        """
        import torch
        
        # 如果没有禁用任何关节，直接返回
        if not self.disable_ring_little:
            return policy_actions
        
        batch_size = policy_actions.shape[0]
        device = policy_actions.device
        dtype = policy_actions.dtype
        
        # 创建完整动作张量，初始化为 0
        full_actions = torch.zeros((batch_size, self.total_dof), device=device, dtype=dtype)
        
        # 将策略动作复制到对应的仿真 DOF 位置
        for policy_idx, sim_idx in self.policy_to_sim.items():
            full_actions[:, sim_idx] = policy_actions[:, policy_idx]
        
        # 禁用关节的动作保持为 0
        # （因为 targets = prev_targets + action_scale * actions，动作为 0 表示保持不变）
        
        return full_actions
    
    def get_flying_action_slice(self):
        """
        获取策略动作中 Flying base 部分的切片
        
        Returns:
            slice: Flying base 动作的切片，如果未启用则返回 None
        """
        if not self.flying_hand_enabled:
            return None
        return slice(0, self.flying_action_dim)
    
    def get_hand_action_slice(self):
        """
        获取策略动作中手部关节部分的切片
        
        Returns:
            slice: 手部动作的切片
        """
        start = self.flying_action_dim if self.flying_hand_enabled else 0
        return slice(start, start + self.hand_action_dim)
    
    def get_active_hand_indices_in_sim(self):
        """返回仿真中激活的手部关节索引列表"""
        offset = NUM_FLYING_DOF if self.flying_hand_enabled else 0
        return [offset + idx for idx in self.hand_active_indices]
    
    def get_disabled_indices_in_sim(self):
        """返回仿真中禁用的关节索引列表"""
        return self.sim_disabled_indices.copy()
    
    def get_flying_indices_in_sim(self):
        """返回仿真中 Flying base 的关节索引列表"""
        return self.flying_indices.copy()
    
    def print_config(self):
        """打印当前动作空间配置"""
        print("\n" + "="*70)
        print("动作空间配置 (Action Space Configuration)")
        print("="*70)
        print(f"Flying Hand 启用:     {self.flying_hand_enabled}")
        print(f"禁用无名指和小拇指:   {self.disable_ring_little}")
        print(f"策略动作维度:         {self.action_dim}")
        print(f"仿真 DOF 总数:        {self.total_dof}")
        print("-"*70)
        
        if self.flying_hand_enabled:
            print(f"Flying Base 维度:     {self.flying_action_dim}")
            print(f"  策略索引 [0-{self.flying_action_dim-1}] -> 仿真 DOF [0-{self.flying_action_dim-1}]")
        
        print(f"手部动作维度:         {self.hand_action_dim}")
        
        if self.disable_ring_little:
            offset = self.flying_action_dim
            print(f"  策略索引 [{offset}-{offset+3}]  -> middle [仿真 {self.policy_to_sim[offset]}-{self.policy_to_sim[offset+3]}]")
            print(f"  策略索引 [{offset+4}-{offset+7}]  -> index  [仿真 {self.policy_to_sim[offset+4]}-{self.policy_to_sim[offset+7]}]")
            print(f"  策略索引 [{offset+8}-{offset+12}] -> thumb  [仿真 {self.policy_to_sim[offset+8]}-{self.policy_to_sim[offset+12]}]")
            print(f"禁用的仿真 DOF:       {self.sim_disabled_indices}")
        else:
            offset = self.flying_action_dim
            print(f"  策略索引 [{offset}-{offset+self.hand_action_dim-1}] -> 仿真 DOF [{self.policy_to_sim[offset]}-{self.policy_to_sim[offset+self.hand_action_dim-1]}]")
        
        print("="*70 + "\n")


def get_action_space_config(config):
    """
    从 YAML 配置中获取动作空间配置
    
    Args:
        config: 环境配置字典
        
    Returns:
        dict: 动作空间配置
    """
    action_space_config = config.get('actionSpace', {})
    flying_hand_config = config.get('flyingHand', {})
    return {
        'disable_ring_little': action_space_config.get('disableRingLittleFinger', False),
        'flying_hand_enabled': flying_hand_config.get('enabled', False),
    }


# ============================================================
# 自动验证（导入时执行）
# ============================================================

if __name__ == '__main__':
    validate_dimensions()
    
    # 测试 ActionSpaceMapper
    print("\n测试 ActionSpaceMapper:")
    mapper_full = ActionSpaceMapper(disable_ring_little=False)
    mapper_full.print_config()
    
    mapper_reduced = ActionSpaceMapper(disable_ring_little=True)
    mapper_reduced.print_config()
