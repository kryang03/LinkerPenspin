#!/usr/bin/env python3
"""
Triangle Pass Waypoints 数据文件

用于存储不同相位下的指尖目标位置和手部关节姿态，供强化学习奖励计算和可视化验证使用。

数据结构说明:
- 每个 waypoint 包含:
  - phase: 相位角 (0 ~ π 或 0 ~ 2π)，表示笔长轴在旋转轴垂平面上的投影方向
  - fingertip_pos: 5个指尖相对于手基座的位置 (15维: 5指 × 3D)
    - 顺序: little, ring, middle, index, thumb (与 FINGERTIP_LINK_NAMES 一致)
  - hand_dof: 手指关节角度 (21维，可选，用于 debug_waypoints.py 可视化)
    - 顺序: IsaacGym 字母序 - index(4), little(4), middle(4), ring(4), thumb(5)
  - object_rot: 物体旋转四元数 (xyzw格式，可选，用于验证)

使用方法:
1. 运行 cache/linker_pose/interactive_tune.py
2. 调整笔的姿态到不同相位
3. 按 P 键打印并复制 waypoint 数据
4. 将数据添加到 TP_WAYPOINTS_RAW 列表中
5. 运行脚本生成密集插值数据

180° 对称性说明 (Triangle Pass / Thumb Around):
- 笔具有对称性: 旋转 180° 后指尖位置相同
- 因此只需采集 [0°, 180°) 范围内的 waypoint
- 插值时会自动镜像到 [180°, 360°) 范围
- 如果物体不具有对称性，设置 half_period_symmetric=False

相位计算原理:
- 笔长轴 = 物体局部Z轴经四元数旋转后的世界坐标方向
- 投影向量 = 笔长轴 - (笔长轴·旋转轴) × 旋转轴
- 相位 = atan2(投影向量在垂平面参考坐标系下的y, x)

注意:
- 相位计算暂未考虑 Flying Hand 基座朝向的影响
- 建议采集 4-8 个均匀分布的 waypoint (在半周期内)
- 插值后将生成 360 个密集点 (每度一个)
- hand_dof 字段仅用于 debug_waypoints.py 可视化，不影响奖励计算

当前的插值逻辑通过在 phases_extended 的头部添加 phases_full[-1] - 2π（虚拟头部点）和尾部添加 phases_full[0] + 2π（虚拟尾部点）来正确处理周期性边界，确保了 360°/0° 处的数据由 350° 和 10° 正确插值得出。

Author: Auto-generated for LinkerPenspin project
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional


# ============================================================
# 原始 Waypoint 数据 (人工采集)
# ============================================================
# 请使用 interactive_tune.py 采集并填充以下数据
# 格式: {'phase': float, 'fingertip_pos': list[15], 'hand_dof': list[21], 'object_rot': list[4]}

TP_WAYPOINTS_RAW: List[Dict] = [

# ---------------------------------------------------------------------------
# 笔长轴 (世界坐标): [0.409259, -0.912148, -0.022190]
# 旋转轴: [0.000000, 0.000000, 1.000000]
# 笔长轴在旋转轴垂平面的投影: [0.409360, -0.912373, -0.000000]
# 相位 (phase): 0.421753 rad (24.16°)
#
# 注意: 相位计算暂未考虑Flying Hand基座朝向的影响
# 如需考虑手朝向，需要将笔长轴和旋转轴都变换到手基座坐标系下计算
#
# TP Waypoint 格式 (可直接复制到 TP_waypoints.py):
{
    'phase': 0.421753,  # 24.2°
    'fingertip_pos': [-0.213537, -0.038052, 0.063074, -0.227959, -0.015947, 0.066923, -0.141349, 0.005628, 0.131451, -0.156635, 0.03259, 0.118515, -0.084716, 0.026653, 0.133005],
    'object_rot': [0.118143, -0.70508, 0.676477, -0.176872],
},
# 笔长轴 (世界坐标): [0.554979, -0.831863, -0.001496]
# 旋转轴: [0.000000, 0.000000, 1.000000]
# 笔长轴在旋转轴垂平面的投影: [0.554979, -0.831864, 0.000000]
# 相位 (phase): 0.588338 rad (33.71°)
#
# 注意: 相位计算暂未考虑Flying Hand基座朝向的影响
# 如需考虑手朝向，需要将笔长轴和旋转轴都变换到手基座坐标系下计算
#
# TP Waypoint 格式 (可直接复制到 TP_waypoints.py):
{
    'phase': 0.588338,  # 33.7°
    'fingertip_pos': [-0.213537, -0.038052, 0.063074, -0.227959, -0.015947, 0.066923, -0.141349, 0.005628, 0.131451, -0.159423, 0.036846, 0.115438, -0.088794, 0.035007, 0.128487],
    'object_rot': [0.14378, -0.692875, 0.655192, -0.26453],
},
# ---------------------------------------------------------------------------
# 笔长轴 (世界坐标): [0.975094, -0.221791, -0.001009]
# 旋转轴: [0.000000, 0.000000, 1.000000]
# 笔长轴在旋转轴垂平面的投影: [0.975094, -0.221791, -0.000000]
# 相位 (phase): 1.347146 rad (77.19°)
#
# 注意: 相位计算暂未考虑Flying Hand基座朝向的影响
# 如需考虑手朝向，需要将笔长轴和旋转轴都变换到手基座坐标系下计算
#
# TP Waypoint 格式 (可直接复制到 TP_waypoints.py):
{
    'phase': 1.347146,  # 77.2°
    'fingertip_pos': [-0.213537, -0.038052, 0.063075, -0.227959, -0.015947, 0.066923, -0.137337, 0.005627, 0.129575, -0.157982, 0.037019, 0.124528, -0.066738, 0.030514, 0.133298],
    'object_rot': [0.459877, -0.537604, 0.567086, -0.421792],
},
# 笔长轴 (世界坐标): [0.999427, 0.033628, 0.003954]
# 旋转轴: [0.000000, 0.000000, 1.000000]
# 笔长轴在旋转轴垂平面的投影: [0.999434, 0.033628, -0.000000]
# 相位 (phase): 1.604431 rad (91.93°)
#
# 注意: 相位计算暂未考虑Flying Hand基座朝向的影响
# 如需考虑手朝向，需要将笔长轴和旋转轴都变换到手基座坐标系下计算
#
# TP Waypoint 格式 (可直接复制到 TP_waypoints.py):
{
    'phase': 1.604431,  # 91.9°
    'fingertip_pos': [-0.213537, -0.038053, 0.063074, -0.227959, -0.015947, 0.066923, -0.143419, 0.005629, 0.132237, -0.1579, 0.029276, 0.124506, -0.070549, 0.023055, 0.1359],
    'object_rot': [0.50044, -0.497577, 0.48534, -0.516161],
},
# 笔长轴 (世界坐标): [0.879954, 0.475059, 0.000000]
# 旋转轴: [0.000000, 0.000000, 1.000000]
# 笔长轴在旋转轴垂平面的投影: [0.879954, 0.475059, 0.000000]
# 相位 (phase): 2.065827 rad (118.36°)
#
# 注意: 相位计算暂未考虑Flying Hand基座朝向的影响
# 如需考虑手朝向，需要将笔长轴和旋转轴都变换到手基座坐标系下计算
#
# TP Waypoint 格式 (可直接复制到 TP_waypoints.py):
{
    'phase': 2.065827,  # 118.4°
    'fingertip_pos': [-0.213537, -0.038052, 0.063074, -0.227959, -0.015947, 0.066923, -0.158691, 0.005634, 0.124348, -0.146363, 0.02773, 0.128808, -0.077703, 0.02275, 0.135783],
    'object_rot': [0.60726, -0.362264, 0.362264, -0.60726],
},
# 笔长轴 (世界坐标): [0.656195, 0.754591, -0.000000]
# 旋转轴: [0.000000, 0.000000, 1.000000]
# 笔长轴在旋转轴垂平面的投影: [0.656195, 0.754591, -0.000000]
# 相位 (phase): 2.425827 rad (138.99°)
#
# 注意: 相位计算暂未考虑Flying Hand基座朝向的影响
# 如需考虑手朝向，需要将笔长轴和旋转轴都变换到手基座坐标系下计算
#
# TP Waypoint 格式 (可直接复制到 TP_waypoints.py):
{
    'phase': 2.425827,  # 139.0°
    'fingertip_pos': [-0.213537, -0.038052, 0.063074, -0.227959, -0.015947, 0.066923, -0.168681, 0.005638, 0.131753, -0.142276, 0.027728, 0.128825, -0.075557, 0.026524, 0.13521],
    'object_rot': [0.662305, -0.247694, 0.247694, -0.662305],
},
# 笔长轴 (世界坐标): [0.309294, 0.950965, 0.001782]
# 旋转轴: [0.000000, 0.000000, 1.000000]
# 笔长轴在旋转轴垂平面的投影: [0.309294, 0.950966, -0.000000]
# 相位 (phase): 2.827142 rad (161.98°)
#
# 注意: 相位计算暂未考虑Flying Hand基座朝向的影响
# 如需考虑手朝向，需要将笔长轴和旋转轴都变换到手基座坐标系下计算
#
# TP Waypoint 格式 (可直接复制到 TP_waypoints.py):
{
    'phase': 2.827142,  # 162.0°
    'fingertip_pos': [-0.213538, -0.038052, 0.063074, -0.227959, -0.015947, 0.066923, -0.152016, 0.005632, 0.134319, -0.136644, 0.031881, 0.129265, -0.079252, 0.02697, 0.136327],
    'object_rot': [0.696089, -0.120705, 0.100689, -0.700537],
},
]


# ============================================================
# 密集插值后的 Waypoint 数据
# ============================================================
# 由 generate_dense_waypoints() 函数自动生成

TP_WAYPOINTS_DENSE: Optional[torch.Tensor] = None  # shape: (360, 15)
TP_PHASES_DENSE: Optional[torch.Tensor] = None     # shape: (360,)
TP_HAND_DOF_DENSE: Optional[torch.Tensor] = None   # shape: (360, 21) - 用于可视化


def generate_dense_waypoints(
    raw_waypoints: List[Dict],
    num_interpolation_points: int = 360,
    device: str = 'cuda',
    half_period_symmetric: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    对原始 waypoint 进行密集插值，生成连续的相位-指尖位置映射
    
    支持两种模式:
    1. half_period_symmetric=True (默认): 适用于笔具有180°对称性的情况
       - 原始数据仅包含 [0, π) 范围的点
       - 自动将 [0, π) 的数据镜像到 [π, 2π)
       - 然后进行周期性插值
    
    2. half_period_symmetric=False: 适用于物体不具有对称性的情况
       - 原始数据包含 [0, 2π) 范围的点
       - 直接进行周期性插值
    
    Args:
        raw_waypoints: 原始 waypoint 列表
        num_interpolation_points: 插值点数量 (默认360，即每度一个)
        device: 计算设备
        half_period_symmetric: 是否使用180°对称性扩展
    
    Returns:
        dense_fingertip_pos: 密集插值后的指尖位置 (num_points, 15)
        dense_phases: 对应的相位值 (num_points,)
        dense_hand_dof: 密集插值后的手指关节角度 (num_points, 21) 或 None (如果原始数据无 hand_dof)
    """
    if len(raw_waypoints) < 2:
        raise ValueError(f"需要至少2个原始waypoint进行插值，当前只有 {len(raw_waypoints)} 个")
    
    # 提取相位和指尖位置
    phases = np.array([wp['phase'] for wp in raw_waypoints])
    fingertip_positions = np.array([wp['fingertip_pos'] for wp in raw_waypoints])
    
    # 检查是否有 hand_dof 数据 (可选字段)
    has_hand_dof = all('hand_dof' in wp and wp['hand_dof'] is not None for wp in raw_waypoints)
    if has_hand_dof:
        hand_dof_data = np.array([wp['hand_dof'] for wp in raw_waypoints])
        print(f"[TP_Waypoints] 检测到 hand_dof 数据 (21维)，将同时进行插值")
    else:
        hand_dof_data = None
        print(f"[TP_Waypoints] 未检测到 hand_dof 数据，仅插值 fingertip_pos")
    
    # 按相位排序
    sort_idx = np.argsort(phases)
    phases = phases[sort_idx]
    fingertip_positions = fingertip_positions[sort_idx]
    if has_hand_dof:
        hand_dof_data = hand_dof_data[sort_idx]
    
    if half_period_symmetric:
        # ============================================================
        # 180° 对称性扩展 (适用于 Triangle Pass 等转笔技巧)
        # ============================================================
        # 笔具有对称性: phase 和 phase + π 对应相同的指尖位置
        # 原始数据仅包含 [0, π) 范围，需要镜像到 [π, 2π)
        
        print(f"[TP_Waypoints] 使用 180° 对称性扩展模式")
        print(f"[TP_Waypoints] 原始数据相位范围: [{np.degrees(phases.min()):.1f}°, {np.degrees(phases.max()):.1f}°]")
        
        # 创建镜像点 (phase + π)
        phases_mirrored = phases + np.pi
        
        # 合并原始点和镜像点
        phases_full = np.concatenate([phases, phases_mirrored])
        fingertip_full = np.concatenate([fingertip_positions, fingertip_positions], axis=0)
        if has_hand_dof:
            hand_dof_full = np.concatenate([hand_dof_data, hand_dof_data], axis=0)
        
        # 重新排序
        sort_idx = np.argsort(phases_full)
        phases_full = phases_full[sort_idx]
        fingertip_full = fingertip_full[sort_idx]
        if has_hand_dof:
            hand_dof_full = hand_dof_full[sort_idx]
        
        print(f"[TP_Waypoints] 扩展后数据点数: {len(phases_full)}")
    else:
        phases_full = phases
        fingertip_full = fingertip_positions
        if has_hand_dof:
            hand_dof_full = hand_dof_data
    
    # ============================================================
    # 周期性插值 (考虑 0 和 2π 相连)
    # ============================================================
    
    # 为了正确处理周期边界，需要在两端添加虚拟点
    # 在头部添加最后一个点 (phase - 2π)
    # 在尾部添加第一个点 (phase + 2π)
    phases_extended = np.concatenate([
        [phases_full[-1] - 2*np.pi],  # 虚拟头部点
        phases_full,
        [phases_full[0] + 2*np.pi]    # 虚拟尾部点
    ])
    fingertip_extended = np.concatenate([
        [fingertip_full[-1]],
        fingertip_full,
        [fingertip_full[0]]
    ], axis=0)
    if has_hand_dof:
        hand_dof_extended = np.concatenate([
            [hand_dof_full[-1]],
            hand_dof_full,
            [hand_dof_full[0]]
        ], axis=0)
    
    # 生成密集相位点 [0, 2π)
    dense_phases = np.linspace(0, 2*np.pi, num_interpolation_points, endpoint=False)
    
    # 对每个维度进行线性插值 - fingertip_pos (15维)
    dense_fingertip = np.zeros((num_interpolation_points, 15))
    for dim in range(15):
        dense_fingertip[:, dim] = np.interp(
            dense_phases,
            phases_extended,
            fingertip_extended[:, dim]
        )
    
    # 对每个维度进行线性插值 - hand_dof (21维, 可选)
    dense_hand_dof_tensor = None
    if has_hand_dof:
        dense_hand_dof = np.zeros((num_interpolation_points, 21))
        for dim in range(21):
            dense_hand_dof[:, dim] = np.interp(
                dense_phases,
                phases_extended,
                hand_dof_extended[:, dim]
            )
        dense_hand_dof_tensor = torch.tensor(dense_hand_dof, dtype=torch.float32, device=device)
    
    # 转换为 PyTorch tensor
    dense_phases_tensor = torch.tensor(dense_phases, dtype=torch.float32, device=device)
    dense_fingertip_tensor = torch.tensor(dense_fingertip, dtype=torch.float32, device=device)
    
    return dense_fingertip_tensor, dense_phases_tensor, dense_hand_dof_tensor


def get_target_fingertip_by_phase(
    current_phase: torch.Tensor,
    dense_fingertip_pos: torch.Tensor,
    dense_phases: torch.Tensor
) -> torch.Tensor:
    """
    根据当前相位获取目标指尖位置 (使用最近邻或线性插值)
    
    Args:
        current_phase: 当前相位 (num_envs,)，值域 [0, 2π)
        dense_fingertip_pos: 密集插值后的指尖位置 (num_points, 15)
        dense_phases: 对应的相位值 (num_points,)
    
    Returns:
        target_fingertip_pos: 目标指尖位置 (num_envs, 15)
    """
    num_envs = current_phase.shape[0]
    num_points = dense_phases.shape[0]
    
    # 将相位归一化到 [0, 2π)
    current_phase = torch.fmod(current_phase, 2 * np.pi)
    current_phase[current_phase < 0] += 2 * np.pi
    
    # 计算相位间隔 (假设均匀分布)
    phase_step = 2 * np.pi / num_points
    
    # 找到最近的索引 (向下取整)
    idx_low = (current_phase / phase_step).long()
    idx_low = torch.clamp(idx_low, 0, num_points - 1)
    idx_high = (idx_low + 1) % num_points
    
    # 计算插值权重
    phase_low = dense_phases[idx_low]
    phase_high = dense_phases[idx_high]
    
    # 处理环形边界
    phase_diff = phase_high - phase_low
    phase_diff[phase_diff < 0] += 2 * np.pi
    
    alpha = (current_phase - phase_low) / (phase_diff + 1e-8)
    alpha = torch.clamp(alpha, 0, 1).unsqueeze(-1)  # (num_envs, 1)
    
    # 线性插值
    fingertip_low = dense_fingertip_pos[idx_low]   # (num_envs, 15)
    fingertip_high = dense_fingertip_pos[idx_high]  # (num_envs, 15)
    
    target_fingertip = (1 - alpha) * fingertip_low + alpha * fingertip_high
    
    return target_fingertip


def get_target_hand_dof_by_phase(
    current_phase: torch.Tensor,
    dense_hand_dof: torch.Tensor,
    dense_phases: torch.Tensor
) -> torch.Tensor:
    """
    根据当前相位获取目标手指关节角度 (使用线性插值)
    
    此函数用于 debug_waypoints.py 可视化，不影响奖励计算。
    
    Args:
        current_phase: 当前相位 (num_envs,) 或标量，值域 [0, 2π)
        dense_hand_dof: 密集插值后的手指关节角度 (num_points, 21)
        dense_phases: 对应的相位值 (num_points,)
    
    Returns:
        target_hand_dof: 目标手指关节角度 (num_envs, 21) 或 (21,)
    """
    # 处理标量输入
    is_scalar = current_phase.dim() == 0
    if is_scalar:
        current_phase = current_phase.unsqueeze(0)
    
    num_envs = current_phase.shape[0]
    num_points = dense_phases.shape[0]
    
    # 将相位归一化到 [0, 2π)
    current_phase = torch.fmod(current_phase, 2 * np.pi)
    current_phase[current_phase < 0] += 2 * np.pi
    
    # 计算相位间隔 (假设均匀分布)
    phase_step = 2 * np.pi / num_points
    
    # 找到最近的索引 (向下取整)
    idx_low = (current_phase / phase_step).long()
    idx_low = torch.clamp(idx_low, 0, num_points - 1)
    idx_high = (idx_low + 1) % num_points
    
    # 计算插值权重
    phase_low = dense_phases[idx_low]
    phase_high = dense_phases[idx_high]
    
    # 处理环形边界
    phase_diff = phase_high - phase_low
    phase_diff[phase_diff < 0] += 2 * np.pi
    
    alpha = (current_phase - phase_low) / (phase_diff + 1e-8)
    alpha = torch.clamp(alpha, 0, 1).unsqueeze(-1)  # (num_envs, 1)
    
    # 线性插值
    dof_low = dense_hand_dof[idx_low]   # (num_envs, 21)
    dof_high = dense_hand_dof[idx_high]  # (num_envs, 21)
    
    target_dof = (1 - alpha) * dof_low + alpha * dof_high
    
    if is_scalar:
        return target_dof.squeeze(0)  # (21,)
    return target_dof


def compute_phase_from_object_rotation(
    object_rot: torch.Tensor,
    rotation_axis: torch.Tensor
) -> torch.Tensor:
    """
    根据物体旋转四元数计算笔长轴在旋转轴垂平面上的相位
    
    此函数与 interactive_tune.py 中的 _compute_pen_phase 保持一致
    
    注意: 暂未考虑 Flying Hand 基座朝向的影响
    
    Args:
        object_rot: 物体旋转四元数 (num_envs, 4) - xyzw格式
        rotation_axis: 旋转轴向量 (num_envs, 3) 或 (3,) - 归一化的单位向量
    
    Returns:
        phase: 相位角 (num_envs,)，值域 [0, 2π)
    """
    num_envs = object_rot.shape[0]
    device = object_rot.device
    
    # 1. 笔的长轴是局部 Z 轴
    local_z = torch.zeros((num_envs, 3), device=device)
    local_z[:, 2] = 1.0
    
    # 2. 将局部向量旋转到世界坐标系 (使用四元数旋转)
    pen_long_axis = quat_apply(object_rot, local_z)  # (num_envs, 3)
    
    # 3. 确保旋转轴是归一化的
    if rotation_axis.dim() == 1:
        rotation_axis = rotation_axis.unsqueeze(0).expand(num_envs, -1)
    rot_axis_normalized = rotation_axis / (torch.norm(rotation_axis, dim=-1, keepdim=True) + 1e-8)
    
    # 4. 投影到旋转轴垂直平面
    dot_product = (pen_long_axis * rot_axis_normalized).sum(dim=-1, keepdim=True)
    proj_vec = pen_long_axis - dot_product * rot_axis_normalized
    
    # 归一化投影向量
    proj_norm = torch.norm(proj_vec, dim=-1, keepdim=True) + 1e-8
    proj_vec_normalized = proj_vec / proj_norm
    
    # 5. 建立垂直平面参考坐标系
    # basis_x: 选择不平行于旋转轴的向量做叉乘
    # 使用与 interactive_tune.py 相同的逻辑
    ref_vec = torch.zeros_like(rot_axis_normalized)
    # 根据旋转轴的z分量选择参考向量
    z_dominant = torch.abs(rot_axis_normalized[:, 2]) >= 0.9
    ref_vec[z_dominant, 0] = 1.0  # 使用 x 轴
    ref_vec[~z_dominant, 2] = 1.0  # 使用 z 轴
    
    basis_x = torch.cross(ref_vec, rot_axis_normalized, dim=-1)
    basis_x = basis_x / (torch.norm(basis_x, dim=-1, keepdim=True) + 1e-8)
    basis_y = torch.cross(rot_axis_normalized, basis_x, dim=-1)
    basis_y = basis_y / (torch.norm(basis_y, dim=-1, keepdim=True) + 1e-8)
    
    # 6. 计算投影向量在参考坐标系下的坐标
    x_coord = (proj_vec_normalized * basis_x).sum(dim=-1)
    y_coord = (proj_vec_normalized * basis_y).sum(dim=-1)
    
    # 7. 计算相位角 (atan2 返回 -π ~ π，转换为 0 ~ 2π)
    phase = torch.atan2(y_coord, x_coord)
    phase[phase < 0] += 2 * np.pi
    
    return phase


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    使用四元数旋转向量
    
    Args:
        quat: 四元数 (N, 4) - xyzw格式
        vec: 向量 (N, 3)
    
    Returns:
        rotated_vec: 旋转后的向量 (N, 3)
    """
    # 提取四元数分量
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # 计算 q × v
    t = 2.0 * torch.stack([
        qy * vec[:, 2] - qz * vec[:, 1],
        qz * vec[:, 0] - qx * vec[:, 2],
        qx * vec[:, 1] - qy * vec[:, 0]
    ], dim=-1)
    
    # 计算 v + w*t + q × t
    result = vec + qw.unsqueeze(-1) * t + torch.cross(
        torch.stack([qx, qy, qz], dim=-1), t, dim=-1
    )
    
    return result


def compute_waypoint_tracking_reward(
    current_fingertip_pos: torch.Tensor,
    target_fingertip_pos: torch.Tensor,
    sigma: float = 0.05,
    finger_mask: torch.Tensor = None
) -> torch.Tensor:
    """
    计算 Waypoint 跟踪奖励 (高斯核函数)
    
    奖励公式: r_track = exp(-||q_curr - q_target||^2 / sigma)
    
    Args:
        current_fingertip_pos: 当前指尖位置 (num_envs, 15)
        target_fingertip_pos: 目标指尖位置 (num_envs, 15)
        sigma: 高斯核带宽，越小奖励越"陡峭"
        finger_mask: 手指掩码 (15,)，1表示参与计算，0表示忽略
                     顺序: [little(3), ring(3), middle(3), index(3), thumb(3)]
                     例如禁用无名指和小拇指时，传入 [0,0,0, 0,0,0, 1,1,1, 1,1,1, 1,1,1]
    
    Returns:
        reward: 跟踪奖励 (num_envs,)，值域 [0, 1]
    """
    # 计算位置误差
    pos_diff = current_fingertip_pos - target_fingertip_pos
    
    # 如果提供了 finger_mask，只计算指定手指的距离
    if finger_mask is not None:
        pos_diff = pos_diff * finger_mask.unsqueeze(0)  # (num_envs, 15) * (1, 15)
    
    squared_distance = (pos_diff ** 2).sum(dim=-1)
    
    # 高斯核函数
    reward = torch.exp(-squared_distance / sigma)
    
    return reward


# ============================================================
# 初始化函数
# ============================================================

def initialize_waypoints(
    device: str = 'cuda',
    half_period_symmetric: bool = True
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    初始化 waypoint 数据
    
    如果 TP_WAYPOINTS_RAW 为空，返回 None
    否则生成密集插值数据
    
    Args:
        device: 计算设备
        half_period_symmetric: 是否使用180°对称性扩展 (默认True，适用于TP等转笔技巧)
    
    Returns:
        (dense_fingertip_pos, dense_phases, dense_hand_dof) 或 (None, None, None)
        dense_hand_dof 可能为 None (如果原始数据没有 hand_dof 字段)
    """
    global TP_WAYPOINTS_DENSE, TP_PHASES_DENSE, TP_HAND_DOF_DENSE
    
    if len(TP_WAYPOINTS_RAW) < 2:
        print("[TP_Waypoints] 警告: 原始 waypoint 数据不足，waypoint 奖励将被禁用")
        print("[TP_Waypoints] 请使用 interactive_tune.py 采集至少2个 waypoint")
        return None, None, None
    
    mode_str = "180°对称性扩展" if half_period_symmetric else "完整360°周期"
    print(f"[TP_Waypoints] 正在从 {len(TP_WAYPOINTS_RAW)} 个原始点生成密集插值 ({mode_str})...")
    TP_WAYPOINTS_DENSE, TP_PHASES_DENSE, TP_HAND_DOF_DENSE = generate_dense_waypoints(
        TP_WAYPOINTS_RAW,
        num_interpolation_points=360,
        device=device,
        half_period_symmetric=half_period_symmetric
    )
    print(f"[TP_Waypoints] 已生成 {TP_WAYPOINTS_DENSE.shape[0]} 个密集 waypoint")
    if TP_HAND_DOF_DENSE is not None:
        print(f"[TP_Waypoints] 同时生成了手指关节角度数据 (21维)")
    
    return TP_WAYPOINTS_DENSE, TP_PHASES_DENSE, TP_HAND_DOF_DENSE


# ============================================================
# 测试代码
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Triangle Pass Waypoints 测试")
    print("=" * 60)
    
    # 测试 1: 180° 对称性扩展 (默认模式) - 无 hand_dof
    print("\n--- 测试 1: 180° 对称性扩展 (无 hand_dof) ---")
    test_waypoints_half = [
        {'phase': 0.0, 'fingertip_pos': [0.0]*15, 'object_rot': [0, 0, 0, 1]},
        {'phase': np.pi/3, 'fingertip_pos': [0.1]*15, 'object_rot': [0, 0, 0, 1]},
        {'phase': 2*np.pi/3, 'fingertip_pos': [0.2]*15, 'object_rot': [0, 0, 0, 1]},
    ]
    
    dense_pos_half, dense_phases_half, dense_dof_half = generate_dense_waypoints(
        test_waypoints_half, device='cpu', half_period_symmetric=True
    )
    print(f"密集插值结果:")
    print(f"  - 相位数量: {dense_phases_half.shape[0]}")
    print(f"  - 相位范围: [{dense_phases_half.min():.4f}, {dense_phases_half.max():.4f}]")
    print(f"  - hand_dof: {dense_dof_half}")
    
    # 验证对称性: phase=0 和 phase=π 应该相同
    idx_0 = 0
    idx_180 = 180
    print(f"  - 相位 0° 的指尖值: {dense_pos_half[idx_0, 0]:.4f}")
    print(f"  - 相位 180° 的指尖值: {dense_pos_half[idx_180, 0]:.4f}")
    print(f"  - 差异: {abs(dense_pos_half[idx_0, 0] - dense_pos_half[idx_180, 0]):.6f}")
    
    # 测试 2: 完整 360° 模式
    print("\n--- 测试 2: 完整 360° 模式 ---")
    test_waypoints_full = [
        {'phase': 0.0, 'fingertip_pos': [0.0]*15, 'object_rot': [0, 0, 0, 1]},
        {'phase': np.pi/2, 'fingertip_pos': [0.1]*15, 'object_rot': [0, 0, 0.707, 0.707]},
        {'phase': np.pi, 'fingertip_pos': [0.2]*15, 'object_rot': [0, 0, 1, 0]},
        {'phase': 3*np.pi/2, 'fingertip_pos': [0.1]*15, 'object_rot': [0, 0, 0.707, -0.707]},
    ]
    
    dense_pos_full, dense_phases_full = generate_dense_waypoints(
        test_waypoints_full, device='cpu', half_period_symmetric=False
    )
    print(f"密集插值结果:")
    print(f"  - 相位数量: {dense_phases_full.shape[0]}")
    print(f"  - 相位范围: [{dense_phases_full.min():.4f}, {dense_phases_full.max():.4f}]")
    
    # 测试 3: 相位查询 (边界情况)
    print("\n--- 测试 3: 相位查询 (边界情况) ---")
    test_phase = torch.tensor([0.0, np.pi/4, np.pi, 3*np.pi/2, 2*np.pi - 0.01], dtype=torch.float32)
    target_pos = get_target_fingertip_by_phase(test_phase, dense_pos_full, dense_phases_full)
    for i, p in enumerate(test_phase):
        print(f"  - 相位 {np.degrees(p.item()):.1f}°: 指尖[0]={target_pos[i, 0]:.4f}")
    
    # 测试 4: 使用真实数据
    print("\n--- 测试 4: 真实 TP_WAYPOINTS_RAW 数据 ---")
    if len(TP_WAYPOINTS_RAW) >= 2:
        dense_pos_real, dense_phases_real = generate_dense_waypoints(
            TP_WAYPOINTS_RAW, device='cpu', half_period_symmetric=True
        )
        print(f"真实数据插值结果:")
        print(f"  - 原始点数: {len(TP_WAYPOINTS_RAW)}")
        print(f"  - 密集点数: {dense_pos_real.shape[0]}")
        
        # 打印一些关键相位的值
        for deg in [0, 30, 60, 90, 120, 150, 180, 210, 270, 330]:
            idx = deg % 360
            print(f"  - 相位 {deg}°: 指尖中指[x]={dense_pos_real[idx, 6]:.4f}")
    else:
        print("  - 跳过: 原始数据不足")
    
    # 测试奖励计算
    print("\n--- 测试 5: 奖励计算 ---")
    current_pos = target_pos + torch.randn_like(target_pos) * 0.01  # 添加噪声
    reward = compute_waypoint_tracking_reward(current_pos, target_pos, sigma=0.05)
    print(f"  - 奖励: {reward.tolist()}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
