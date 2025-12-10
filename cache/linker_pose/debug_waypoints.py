#!/usr/bin/env python3
"""
Ghost Hand Visualizer - Waypoint 与 Phase 可视化验证工具

用途:
- 验证 Waypoint 数据和 Phase 计算的正确性
- 显示笔随时间旋转时，各相位对应的目标手部姿态 (Ghost Hand)
- 不运行物理仿真，仅用于可视化验证

使用方法:
    python cache/linker_pose/debug_waypoints.py
    python cache/linker_pose/debug_waypoints.py --speed 0.5   # 慢速播放
    python cache/linker_pose/debug_waypoints.py --pause       # 暂停模式 (按空格步进)

操作:
    Space - 暂停/继续 或 单步前进
    R     - 重置到 phase=0
    +/-   - 加速/减速
    Q/ESC - 退出

可视化内容:
- 笔 (悬浮，无重力): 根据时间 t 计算当前 phase，旋转到对应角度
- Ghost Hand (半透明): 显示当前 phase 对应的目标指尖位置
- 旋转轴 (黄色箭头): 显示旋转轴方向
- Phase 信息 (屏幕文本): 当前相位角度

Author: LinkerPenspin project
"""

import isaacgym
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
import torch
import numpy as np
import os
import math
import time
import argparse
import yaml
import sys

# 添加项目根目录到路径
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)

from penspin.utils.tp_waypoints import (
    TP_WAYPOINTS_RAW,
    generate_dense_waypoints,
    get_target_fingertip_by_phase,
    get_target_hand_dof_by_phase,
    compute_phase_from_object_rotation,
)
from isaacgym.torch_utils import quat_from_euler_xyz, quat_apply


# ==================== 配置 ====================
CONFIG = {
    # 资源路径 (相对于脚本)
    "hand_urdf": "../../assets/linker_hand/L25_dof_urdf.urdf",  # 非 Flying Hand (21 DOF)
    "obj_urdf": "../../assets/cylinder/linkerpen/spinning_pen.urdf",
    
    # 手部基座初始位置 (与 linker_hand_grasp.py 一致)
    # 非 Flying Hand 模式下，这是 URDF 的 Transform
    "hand_base_init_pos": [0.0, 0.0, 0.35],    # 手部基座位置 [px, py, pz]
    "hand_base_init_rot": [0.0, -1.31, 0.0],   # 手部基座旋转 [rx, ry, rz] 欧拉角
    
    # 物体初始位置 (悬浮，关闭重力)
    "obj_init_pos": [-0.11, 0.03, 0.53],
    "obj_init_rot": [0.0, 0.0, 0.0, 1.0],  # 四元数 xyzw
    
    # 旋转轴
    "rotation_axis": [0, 0, 1],  # +Z (CCW)
    
    # 旋转速度 (rad/s)
    "rotation_speed": np.pi / 4,  # 45°/s，即 8 秒转一圈
    
    # 指尖球体半径 (用于可视化)
    "fingertip_sphere_radius": 0.01,
    
    # Waypoint 是否使用 180° 对称性
    "half_period_symmetric": True,
    
    # 手部 PD 控制参数
    "hand_stiffness": 15.0,
    "hand_damping": 0.35,
}

# 指尖名称和颜色
FINGERTIP_NAMES = ["little", "ring", "middle", "index", "thumb"]
FINGERTIP_COLORS = [
    gymapi.Vec3(1.0, 0.0, 0.0),  # little - 红
    gymapi.Vec3(1.0, 0.5, 0.0),  # ring - 橙
    gymapi.Vec3(1.0, 1.0, 0.0),  # middle - 黄
    gymapi.Vec3(0.0, 1.0, 0.0),  # index - 绿
    gymapi.Vec3(0.0, 0.0, 1.0),  # thumb - 蓝
]


class WaypointVisualizer:
    """Waypoint 可视化器"""
    
    def __init__(self, args):
        self.args = args
        self.paused = args.pause
        self.speed_multiplier = args.speed
        self.current_phase = 0.0
        self.current_time = 0.0
        
        # 初始化 Isaac Gym
        self._init_gym()
        
        # 加载资源
        self._load_assets()
        
        # 创建环境
        self._create_env()
        
        # 初始化 Waypoint 数据
        self._init_waypoints()
        
        # 设置相机
        self._setup_camera()
        
        print("\n" + "="*60)
        print("Ghost Hand Visualizer - Waypoint 验证工具")
        print("="*60)
        print(f"  Waypoint 数量: {len(TP_WAYPOINTS_RAW)}")
        print(f"  密集插值点数: {self.dense_fingertip_pos.shape[0]}")
        print(f"  180° 对称性:   {CONFIG['half_period_symmetric']}")
        print(f"  旋转速度:      {CONFIG['rotation_speed']:.2f} rad/s")
        hand_dof_status = "启用" if self.dense_hand_dof is not None else "未配置"
        print(f"  Hand DOF 动画: {hand_dof_status}")
        print("="*60)
        print("\n操作说明:")
        print("  Space - 暂停/继续 或 单步前进")
        print("  R     - 重置到 phase=0")
        print("  +/-   - 加速/减速")
        print("  Q/ESC - 退出")
        print("="*60 + "\n")
    
    def _init_gym(self):
        """初始化 Isaac Gym"""
        self.gym = gymapi.acquire_gym()
        
        # 仿真参数
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)  # 关闭重力
        
        # 使用 CPU pipeline (避免 setRigidBodyStates 警告)
        sim_params.use_gpu_pipeline = False
        self.device = 'cpu'
        
        # PhysX 参数
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.rest_offset = 0.001
        sim_params.physx.use_gpu = True  # PhysX 仍使用 GPU 加速
        
        # 创建模拟
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            raise RuntimeError("Failed to create sim")
        
        # 创建地面
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
    
    def _load_assets(self):
        """加载资源"""
        # 加载手部资产 (非 Flying Hand)
        asset_options_hand = gymapi.AssetOptions()
        asset_options_hand.fix_base_link = True
        asset_options_hand.disable_gravity = True
        asset_options_hand.flip_visual_attachments = False
        asset_options_hand.collapse_fixed_joints = False
        
        # 加载物体资产
        asset_options_obj = gymapi.AssetOptions()
        asset_options_obj.fix_base_link = False  # 物体需要可以移动
        asset_options_obj.disable_gravity = True
        
        hand_path = os.path.normpath(os.path.join(SCRIPT_DIR, CONFIG["hand_urdf"]))
        obj_path = os.path.normpath(os.path.join(SCRIPT_DIR, CONFIG["obj_urdf"]))
        
        print(f"加载手部 URDF: {hand_path}")
        self.hand_asset = self.gym.load_asset(
            self.sim, os.path.dirname(hand_path), os.path.basename(hand_path), asset_options_hand
        )
        
        print(f"加载物体 URDF: {obj_path}")
        self.obj_asset = self.gym.load_asset(
            self.sim, os.path.dirname(obj_path), os.path.basename(obj_path), asset_options_obj
        )
        
        self.num_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)
        print(f"手部 DOF 数量: {self.num_hand_dofs}")
    
    def _create_env(self):
        """创建环境"""
        env_lower = gymapi.Vec3(-0.5, -0.5, 0.0)
        env_upper = gymapi.Vec3(0.5, 0.5, 1.0)
        
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, 1)
        
        # --- 创建手部 actor (非 Flying Hand: 设置正确的 Transform) ---
        hand_pose = gymapi.Transform()
        hand_pose.p = gymapi.Vec3(*CONFIG["hand_base_init_pos"])
        
        # 将欧拉角转换为四元数 (ZYX 顺序)
        rx, ry, rz = CONFIG["hand_base_init_rot"]
        quat_x = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), rx)
        quat_y = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), ry)
        quat_z = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), rz)
        hand_pose.r = quat_z * quat_y * quat_x
        
        self.hand_actor = self.gym.create_actor(self.env, self.hand_asset, hand_pose, "hand", 0, 1)
        
        # 配置手部关节属性
        dof_props = self.gym.get_asset_dof_properties(self.hand_asset)
        dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        dof_props["stiffness"].fill(CONFIG["hand_stiffness"])
        dof_props["damping"].fill(CONFIG["hand_damping"])
        self.gym.set_actor_dof_properties(self.env, self.hand_actor, dof_props)
        
        # --- 创建物体 actor ---
        obj_pose = gymapi.Transform()
        obj_pose.p = gymapi.Vec3(*CONFIG["obj_init_pos"])
        rot = CONFIG["obj_init_rot"]
        obj_pose.r = gymapi.Quat(rot[0], rot[1], rot[2], rot[3])
        
        self.obj_actor = self.gym.create_actor(self.env, self.obj_asset, obj_pose, "pen", 0, 2)
        
        # 获取手部刚体句柄（用于可视化指尖）
        self.hand_rigid_body_count = self.gym.get_actor_rigid_body_count(self.env, self.hand_actor)
        
        # 查找指尖刚体索引
        self.fingertip_indices = []
        fingertip_link_names = [
            "little_tip_link",
            "ring_tip_link", 
            "middle_tip_link",
            "index_tip_link",
            "thumb_tip_link"
        ]
        for name in fingertip_link_names:
            idx = self.gym.find_actor_rigid_body_index(self.env, self.hand_actor, name, gymapi.DOMAIN_ACTOR)
            self.fingertip_indices.append(idx)
            print(f"  指尖 {name}: 刚体索引 {idx}")
        
        # 准备 Tensor 用于更新物体状态
        self._prepare_tensors()
    
    def _init_waypoints(self):
        """初始化 Waypoint 数据"""
        # 生成密集插值 (包括 hand_dof)
        self.dense_fingertip_pos, self.dense_phases, self.dense_hand_dof = generate_dense_waypoints(
            TP_WAYPOINTS_RAW,
            num_interpolation_points=360,
            device=self.device,
            half_period_symmetric=CONFIG["half_period_symmetric"]
        )
        
        print(f"\n密集 Waypoint 数据:")
        print(f"  fingertip_pos Shape: {self.dense_fingertip_pos.shape}")
        print(f"  Phase 范围: [{self.dense_phases.min():.4f}, {self.dense_phases.max():.4f}]")
        
        if self.dense_hand_dof is not None:
            print(f"  hand_dof Shape: {self.dense_hand_dof.shape}")
            self.has_hand_dof = True
        else:
            print(f"  hand_dof: 无 (原始数据中未包含 hand_dof 字段)")
            self.has_hand_dof = False
        
        # 旋转轴
        self.rotation_axis = torch.tensor(CONFIG["rotation_axis"], dtype=torch.float32, device=self.device)
        self.rotation_axis = self.rotation_axis / torch.norm(self.rotation_axis)
        
        print(f"  旋转轴: {self.rotation_axis.numpy()}")
    
    def _prepare_tensors(self):
        """准备 Tensor 用于控制"""
        # 获取 root state Tensor (用于物体位置)
        _root_states = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(_root_states)
        
        # 获取 DOF 状态 Tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states)
        self.dof_pos = self.dof_states[:self.num_hand_dofs, 0]
        self.dof_vel = self.dof_states[:self.num_hand_dofs, 1]
    
    def _setup_camera(self):
        """设置相机"""
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise RuntimeError("Failed to create viewer")
        
        # 设置相机位置 (看向手部基座附近)
        cam_pos = gymapi.Vec3(0.5, 0.5, 0.6)
        cam_target = gymapi.Vec3(
            CONFIG["hand_base_init_pos"][0], 
            CONFIG["hand_base_init_pos"][1], 
            CONFIG["hand_base_init_pos"][2]
        )
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
    
    def _compute_object_rotation(self, phase):
        """根据相位计算物体旋转四元数
        
        笔的长轴是局部 Z 轴，我们需要绕旋转轴旋转使得：
        - phase = 0 时，笔指向某个参考方向
        - phase 增加时，笔绕旋转轴转动
        """
        # 绕 Z 轴旋转 phase 角度
        # 注意：这里简化处理，假设旋转轴是 +Z
        half_angle = phase / 2.0
        axis = self.rotation_axis.numpy()
        
        # 四元数: [sin(θ/2)*axis, cos(θ/2)]
        qx = np.sin(half_angle) * axis[0]
        qy = np.sin(half_angle) * axis[1]
        qz = np.sin(half_angle) * axis[2]
        qw = np.cos(half_angle)
        
        # 需要额外旋转让笔水平放置（笔长轴与旋转轴垂直）
        # 初始时笔长轴指向 +Z，我们需要先让它躺下
        # 绕 Y 轴旋转 90°
        base_rot = gymapi.Quat.from_euler_zyx(0, np.pi/2, 0)
        phase_rot = gymapi.Quat(qx, qy, qz, qw)
        
        # 组合旋转：先 base 再 phase
        final_rot = phase_rot * base_rot
        
        return final_rot
    
    def _get_target_fingertip_pos(self, phase):
        """获取当前相位的目标指尖位置"""
        phase_tensor = torch.tensor([phase], dtype=torch.float32, device=self.device)
        target = get_target_fingertip_by_phase(
            phase_tensor,
            self.dense_fingertip_pos,
            self.dense_phases
        )
        return target[0].numpy()  # (15,)
    
    def _get_target_hand_dof(self, phase):
        """获取当前相位的目标手指关节角度
        
        Returns:
            target_dof: (21,) numpy array 或 None
        """
        if not self.has_hand_dof:
            return None
        
        phase_tensor = torch.tensor(phase, dtype=torch.float32, device=self.device)
        target = get_target_hand_dof_by_phase(
            phase_tensor,
            self.dense_hand_dof,
            self.dense_phases
        )
        return target.numpy()  # (21,)
    
    def _apply_target_hand_dof(self, target_dof):
        """将目标手指关节角度应用到手部 Actor
        
        Args:
            target_dof: (21,) numpy array - 手指关节角度
        """
        if target_dof is None:
            return
        
        # 设置 DOF 目标位置
        # 注意: 非 Flying Hand URDF 只有 21 个 DOF (手指关节)
        target_tensor = torch.tensor(target_dof, dtype=torch.float32, device=self.device)
        self.gym.set_dof_position_target_tensor(
            self.sim,
            gymtorch.unwrap_tensor(target_tensor)
        )
    
    def _draw_fingertip_spheres(self, fingertip_pos):
        """绘制指尖目标位置球体
        
        Args:
            fingertip_pos: 指尖位置 (15,) - 5个指尖 × 3D
        """
        self.gym.clear_lines(self.viewer)
        
        # 手部基座世界位置
        hand_base_pos = np.array(CONFIG["hand_base_init_pos"])
        
        for i, (name, color) in enumerate(zip(FINGERTIP_NAMES, FINGERTIP_COLORS)):
            # 获取当前指尖的相对位置 (相对于手基座)
            pos_rel = fingertip_pos[i*3:(i+1)*3]
            # 转换为世界坐标
            pos_world = hand_base_pos + pos_rel
            
            # 绘制球体（用多条线近似）
            self._draw_sphere(pos_world, CONFIG["fingertip_sphere_radius"], color)
    
    def _draw_sphere(self, center, radius, color, segments=8):
        """用线条绘制球体轮廓"""
        # 绘制三个正交的圆
        for axis in range(3):
            for i in range(segments):
                angle1 = 2 * np.pi * i / segments
                angle2 = 2 * np.pi * (i + 1) / segments
                
                p1 = np.array(center, dtype=np.float32)
                p2 = np.array(center, dtype=np.float32)
                
                if axis == 0:  # YZ 平面
                    p1[1] += radius * np.cos(angle1)
                    p1[2] += radius * np.sin(angle1)
                    p2[1] += radius * np.cos(angle2)
                    p2[2] += radius * np.sin(angle2)
                elif axis == 1:  # XZ 平面
                    p1[0] += radius * np.cos(angle1)
                    p1[2] += radius * np.sin(angle1)
                    p2[0] += radius * np.cos(angle2)
                    p2[2] += radius * np.sin(angle2)
                else:  # XY 平面
                    p1[0] += radius * np.cos(angle1)
                    p1[1] += radius * np.sin(angle1)
                    p2[0] += radius * np.cos(angle2)
                    p2[1] += radius * np.sin(angle2)
                
                self.gym.add_lines(
                    self.viewer, self.env, 1,
                    [p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]],
                    [color.x, color.y, color.z]
                )
    
    def _draw_rotation_axis(self):
        """绘制旋转轴"""
        center = np.array(CONFIG["obj_init_pos"])
        axis = self.rotation_axis.numpy()
        length = 0.1
        
        p1 = center - axis * length / 2
        p2 = center + axis * length / 2
        
        # 黄色轴线
        self.gym.add_lines(
            self.viewer, self.env, 1,
            [p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]],
            [1.0, 1.0, 0.0]  # 黄色
        )
    
    def _update_object_pose(self, phase):
        """更新物体姿态"""
        rot = self._compute_object_rotation(phase)
        
        # 刷新状态
        self.gym.refresh_actor_root_state_tensor(self.sim)
        
        # 物体是第二个 actor (index=1, 因为 hand 是 0)
        obj_idx = 1
        self.root_states[obj_idx, 0] = CONFIG["obj_init_pos"][0]
        self.root_states[obj_idx, 1] = CONFIG["obj_init_pos"][1]
        self.root_states[obj_idx, 2] = CONFIG["obj_init_pos"][2]
        self.root_states[obj_idx, 3] = rot.x
        self.root_states[obj_idx, 4] = rot.y
        self.root_states[obj_idx, 5] = rot.z
        self.root_states[obj_idx, 6] = rot.w
        self.root_states[obj_idx, 7:13] = 0.0  # 清零速度
        
        # 应用状态 (使用 actor root state tensor)
        obj_indices = torch.tensor([obj_idx], dtype=torch.int32, device=self.device)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, 
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(obj_indices),
            1
        )
    
    def _handle_keyboard(self):
        """处理键盘输入"""
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "quit" and evt.value > 0:
                return False
            elif evt.action == "toggle_pause" and evt.value > 0:
                if self.paused:
                    # 暂停时按空格 = 单步
                    self.current_phase += 0.05
                    if self.current_phase >= 2 * np.pi:
                        self.current_phase -= 2 * np.pi
                else:
                    self.paused = True
            elif evt.action == "reset" and evt.value > 0:
                self.current_phase = 0.0
                self.current_time = 0.0
                print("[Reset] Phase = 0")
            elif evt.action == "speed_up" and evt.value > 0:
                self.speed_multiplier *= 1.5
                print(f"[Speed] {self.speed_multiplier:.2f}x")
            elif evt.action == "speed_down" and evt.value > 0:
                self.speed_multiplier /= 1.5
                print(f"[Speed] {self.speed_multiplier:.2f}x")
            elif evt.action == "continue" and evt.value > 0:
                self.paused = False
                print("[Continue]")
        
        return True
    
    def run(self):
        """主循环"""
        # 订阅键盘事件
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "quit")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "quit")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "toggle_pause")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_EQUAL, "speed_up")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_MINUS, "speed_down")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_C, "continue")
        
        dt = 1.0 / 60.0
        last_print_time = 0
        
        while not self.gym.query_viewer_has_closed(self.viewer):
            # 处理键盘
            if not self._handle_keyboard():
                break
            
            # 更新相位
            if not self.paused:
                self.current_time += dt * self.speed_multiplier
                self.current_phase = (CONFIG["rotation_speed"] * self.current_time) % (2 * np.pi)
            
            # 更新物体姿态
            self._update_object_pose(self.current_phase)
            
            # 获取并应用目标手部姿态 (如果有 hand_dof 数据)
            target_hand_dof = self._get_target_hand_dof(self.current_phase)
            self._apply_target_hand_dof(target_hand_dof)
            
            # 获取目标指尖位置并绘制
            target_fingertip = self._get_target_fingertip_pos(self.current_phase)
            self._draw_fingertip_spheres(target_fingertip)
            self._draw_rotation_axis()
            
            # 定期打印信息
            if time.time() - last_print_time > 1.0:
                phase_deg = np.degrees(self.current_phase)
                hand_dof_info = "手部DOF: 已启用" if self.has_hand_dof else "手部DOF: 无数据"
                print(f"Phase: {self.current_phase:.4f} rad ({phase_deg:.1f}°) | "
                      f"Speed: {self.speed_multiplier:.2f}x | "
                      f"{hand_dof_info} | "
                      f"{'PAUSED' if self.paused else 'RUNNING'}")
                last_print_time = time.time()
            
            # 仿真步进
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            
            # 刷新 DOF 状态
            self.gym.refresh_dof_state_tensor(self.sim)
            
            # 渲染
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
        
        # 清理
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        print("\n[Exit] Ghost Hand Visualizer 已退出")


def main():
    parser = argparse.ArgumentParser(description="Ghost Hand Visualizer - Waypoint 验证工具")
    parser.add_argument("--speed", type=float, default=1.0, help="播放速度倍率")
    parser.add_argument("--pause", action="store_true", help="启动时暂停")
    args = parser.parse_args()
    
    # 检查 Waypoint 数据
    if len(TP_WAYPOINTS_RAW) < 2:
        print("="*60)
        print("[错误] Waypoint 数据不足!")
        print(f"当前只有 {len(TP_WAYPOINTS_RAW)} 个点，至少需要 2 个")
        print("请先使用 interactive_tune.py 采集 Waypoint 数据")
        print("="*60)
        return
    
    visualizer = WaypointVisualizer(args)
    visualizer.run()


if __name__ == "__main__":
    main()
