#!/usr/bin/env python3
"""
python cache/linker_pose/interactive_tune.py
交互式姿态调试工具 - 适配 27 DoF Flying Hand (L25_dof_urdf_flying.urdf)

整合 find_pos, find_rot, gravity_sim 功能于一体
支持键盘交互控制基座位置和旋转

URDF 结构 (27 DoF):
-------------------
重要: IsaacGym 加载 URDF 后的关节顺序是按字母顺序排列的！

  基座 (6 DoF): virtual_px, virtual_py, virtual_pz, virtual_rx, virtual_ry, virtual_rz
  手指 (21 DoF, 按字母顺序):
    - index:   index_joint0-3   (4 DoF)  [idx 6-9]   食指
    - little:  little_joint0-3  (4 DoF)  [idx 10-13] 小拇指
    - middle:  middle_joint0-3  (4 DoF)  [idx 14-17] 中指 (注意 joint0 限制为 [0,0])
    - ring:    ring_joint0-3    (4 DoF)  [idx 18-21] 无名指
    - thumb:   thumb_joint0-4   (5 DoF)  [idx 22-26] 拇指

快捷键说明:
-----------
物体位置控制:
  W/S - 前进/后退 (X轴)
  A/D - 左/右移动 (Y轴)
  Q/E - 上升/下降 (Z轴)

物体旋转控制:
  J/L - Yaw 左/右

旋转轴可视化控制:
  V   - 切换旋转轴可视化 (显示/隐藏)
  B   - 循环切换旋转轴方向 (+Z/-Z/+X/-X/+Y/-Y)
  
功能控制:
  Space - 开启/关闭重力
  R     - 重置物体位置
  P     - 打印当前姿态 (可复制到 linker_hand_grasp.py)
  T     - 切换预设手指姿态 (打开/握拳/Triangle Pass)
  +/-   - 增大/减小移动步长
  F     - 切换显示帮助信息

手指关节控制 (按字母顺序: index -> middle -> thumb):
  食指 (index):  1/! - joint0,  2/@ - joint1,  3/# - joint2,  4/$ - joint3
  中指 (middle): 5/% - joint1,  6/^ - joint2,  7/& - joint3 (joint0 被锁定)
  拇指 (thumb):  8/* - joint0,  9/( - joint1,  0/) - joint2,  -/_ - joint3,  =/+ - joint4
  (数字键增加关节值, Shift+数字键减少关节值)

Author: Auto-generated for LinkerPenspin project
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


# ==================== 配置区域 ====================
# 配置文件路径（相对于脚本）
TASK_CONFIG_PATH = "../../configs/task/LinkerHandHora.yaml"

def load_rotation_axis_from_config():
    """从配置文件加载旋转轴设置"""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.normpath(os.path.join(script_dir, TASK_CONFIG_PATH))
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        rot_axis = config.get('env', {}).get('rotation_axis', [0, 0, 1])
        if isinstance(rot_axis, str):
            # 转换旧格式字符串为向量
            sign = 1 if rot_axis[0] == '+' else -1
            axis = rot_axis[1]
            axis_map = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}
            rot_axis = [sign * v for v in axis_map.get(axis, [0, 0, 1])]
        return rot_axis
    return [0, 0, 1]

CONFIG = {
    # --- 资源路径 (相对于当前脚本) ---
    "hand_urdf": "../../assets/linker_hand/L25_dof_urdf_flying.urdf",
    "obj_urdf": "../../assets/cylinder/linkerpen/spinning_pen.urdf",
    # 备选物体 (铅笔)
    "obj_urdf_pencil": "../../assets/cylinder/pencil-5-7/0000.urdf",
    
    # --- 初始物体位置 ---
    "obj_init_pos": [-0.11, 0.03, 0.6],  # 参考 find_pos.py 的位置
    "obj_init_rot": [0.4462, -0.5485, 0.5485, -0.4462],  # 四元数 (x,y,z,w) - 参考 find_pos.py
    "obj_scale": 1.0,  # spinning_pen 不需要缩放；如果用 pencil-5-7，设为 0.3
    
    # --- 手部基座初始位置 ---
    # 根据 URDF 限制: px/py: [-0.05, 0.05], pz: [0.30, 0.40], rx/ry: [-1.57, 1.57], rz: [-3.14, 3.14]
    "hand_base_init_pos": [0.0, 0.0, 0.35],  # 与 virtual_pz 的 limit 匹配
    # 让手平躺，手心朝上: 绕 Y 轴旋转 -75 度 (约 -1.31 弧度)
    "hand_base_init_rot": [0.0, -1.31, 0.0],  # Euler (Roll, Pitch, Yaw) - Pitch = -75°
    
    # --- 旋转轴设置 (从配置文件加载或使用默认值) ---
    "rotation_axis": load_rotation_axis_from_config(),
    # 预定义的旋转轴选项（用于交互式切换）
    "rotation_axis_presets": [
        {"name": "+Z (CCW)", "axis": [0, 0, 1]},
        {"name": "-Z (CW)", "axis": [0, 0, -1]},
        {"name": "+X", "axis": [1, 0, 0]},
        {"name": "-X", "axis": [-1, 0, 0]},
        {"name": "+Y", "axis": [0, 1, 0]},
        {"name": "-Y", "axis": [0, -1, 0]},
    ],
    
    # --- 场景预设 (完整状态: 手部27 DoF + 物体位置旋转) ---
    # 每个预设包含: hand_dof (27个值), object_pos (3个值), object_rot (4个四元数)
    "scene_presets": {
        "default": {
            "hand_dof": [0.0, 0.0, 0.35, 0.0, -1.31, 0.0] + [0.0] * 21,  # 基座 + 张开手指
            "object_pos": [-0.11, 0.03, 0.6],
            "object_rot": [0.4462, -0.5485, 0.5485, -0.4462]
        },
        "triangle_grasp_0": {
            "hand_dof": [0.0, -0.0, 0.35, -0.0, -1.31, 0.0] + [
                0.1, -1.0, -0.6, -0.4,      # index (食指)
                -0.0, -0.0, -0.0, -0.0,      # little (小拇指)
                0.0, -1.15, -0.6, -0.4,      # middle (中指)
                -0.0, -0.0, -0.0, -0.0,      # ring (无名指)
                -0.0, -1.35, -0.35, -0.5, -0.35  # thumb (拇指)
            ],
            "object_pos": [-0.12, 0.034, 0.526001],
            "object_rot": [0.12166, -0.696562, 0.696562, -0.12166]
        },
        "triangle_grasp_1": {
            "hand_dof": [0.0, 0.0, 0.350001, 0.0, -1.310001, 0.0] + [
                0.08, -1.0, -0.6, -0.4,      # index (食指)
                -0.0, -0.0, -0.0, -0.0,      # little (小拇指)
                -0.0, -1.15, -0.65, -0.4,     # middle (中指)
                -0.0, -0.0, -0.0, -0.0,      # ring (无名指)
                0.1, -1.35, -0.35, -0.5, -0.35  # thumb (拇指)
            ],
            "object_pos": [-0.116, 0.022, 0.528001],
            "object_rot": [0.503743, -0.496228, 0.496228, -0.503743]
        },
        "triangle_grasp_grasp_2": {
            "hand_dof": [-0.000000, 0.000000, 0.350000, 0.000000, -1.310000, -0.000000] + [
                0.03, -1.0, -0.6, -0.4,      # index (食指)
                -0.0, -0.0, -0.0, -0.0,      # little (小拇指)
                -0.0, -1.05, -0.65, -0.4,     # middle (中指)
                -0.0, -0.0, -0.0, -0.0,      # ring (无名指)
                0.0, -1.35, -0.15, -0.6, -0.5  # thumb (拇指)
            ],
            "object_pos": [-0.120102, 0.028283, 0.530196],
            "object_rot": [0.60706, -0.361703, 0.361988, -0.607958]
        }
    },
    "default_scene_preset": "triangle_grasp",
    
    # --- 控制参数 ---
    "step_size_pos": 0.002,  # 位置移动步长 (米)
    "step_size_rot": 0.02,   # 旋转步长 (弧度)
    "step_size_finger": 0.05,  # 手指关节步长 (弧度)
    
    # --- 物理参数 ---
    "hand_stiffness": 1000.0,
    "hand_damping": 50.0,
    "obj_density": 100.0,
}
# ==================== 配置区域结束 ====================


class InteractivePoseTuner:
    """交互式姿态调试器"""
    
    def __init__(self):
        self.gym = gymapi.acquire_gym()
        
        # 状态变量
        self.gravity_active = False
        self.show_help = True
        self.current_scene_preset = CONFIG["default_scene_preset"]
        self.step_size_pos = CONFIG["step_size_pos"]
        self.step_size_rot = CONFIG["step_size_rot"]
        
        # 手部基座控制量 (6 DoF)
        self.hand_base_pos = np.array(CONFIG["hand_base_init_pos"], dtype=np.float32)
        self.hand_base_rot = np.array(CONFIG["hand_base_init_rot"], dtype=np.float32)
        
        # 旋转轴可视化相关状态
        self.show_rotation_axis = True  # 是否显示旋转轴
        self.current_rotation_axis = np.array(CONFIG["rotation_axis"], dtype=np.float32)
        self.rotation_axis_preset_idx = 0  # 当前预设索引
        
        # 物体位置控制量 (用于交互调整)
        self.obj_pos = np.array(CONFIG["obj_init_pos"], dtype=np.float32)
        self.obj_base_rot = np.array(CONFIG["obj_init_rot"], dtype=np.float32)  # 物体基础旋转 (四元数)
        self.obj_yaw = 0.0  # 物体绕 Z 轴的增量旋转角度
        
        # 创建仿真
        self.sim = self._create_sim()
        self.viewer = self._create_viewer()
        
        # 加载资产和创建环境
        self._load_assets()
        self._create_env()
        self._prepare_tensors()
        self._subscribe_keyboard_events()
        
        # 打印 DOF 信息
        self._print_dof_info()
        
        print("\n" + "="*60)
        print("交互式姿态调试工具已启动!")
        print("按 F 键显示/隐藏帮助信息")
        print("="*60 + "\n")

    def _get_absolute_path(self, rel_path):
        """获取相对于脚本的绝对路径"""
        script_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.normpath(os.path.join(script_dir, rel_path))

    def _create_sim(self):
        """创建仿真环境"""
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)  # 初始无重力
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        
        # 使用 CPU pipeline (更稳定，适合交互式调试)
        # GPU pipeline 需要更复杂的 tensor 管理
        sim_params.use_gpu_pipeline = False
        self.device = 'cpu'
        
        # PhysX 参数 (仍然使用 GPU 加速物理计算)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 2
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.use_gpu = True  # PhysX GPU 加速
        
        sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if sim is None:
            raise RuntimeError("Failed to create simulation")
        
        # 添加地面 (可选，用于重力测试)
        if CONFIG.get('add_ground', True):
            plane_params = gymapi.PlaneParams()
            plane_params.normal = gymapi.Vec3(0, 0, 1)
            plane_params.distance = 0
            plane_params.static_friction = 1.0
            plane_params.dynamic_friction = 1.0
            self.gym.add_ground(sim, plane_params)
            print("已添加地面平面")
        
        return sim

    def _create_viewer(self):
        """创建查看器"""
        viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError("Failed to create viewer")
        return viewer

    def _load_assets(self):
        """加载手和物体资产"""
        # --- 加载 Flying Hand ---
        hand_path = self._get_absolute_path(CONFIG["hand_urdf"])
        hand_opts = gymapi.AssetOptions()
        hand_opts.fix_base_link = True  # 虚拟关节的根连杆固定
        hand_opts.disable_gravity = True
        hand_opts.flip_visual_attachments = False
        hand_opts.collapse_fixed_joints = False
        hand_opts.convex_decomposition_from_submeshes = True
        
        print(f"Loading hand: {hand_path}")
        self.hand_asset = self.gym.load_asset(
            self.sim, os.path.dirname(hand_path), os.path.basename(hand_path), hand_opts
        )
        self.num_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)
        print(f"  Hand DOFs: {self.num_hand_dofs}")
        
        # --- 加载物体 ---
        obj_path = self._get_absolute_path(CONFIG["obj_urdf"])
        obj_opts = gymapi.AssetOptions()
        obj_opts.fix_base_link = False
        obj_opts.disable_gravity = False
        obj_opts.density = CONFIG["obj_density"]
        obj_opts.angular_damping = 0.5
        obj_opts.linear_damping = 0.5
        
        print(f"Loading object: {obj_path}")
        self.obj_asset = self.gym.load_asset(
            self.sim, os.path.dirname(obj_path), os.path.basename(obj_path), obj_opts
        )
        
    def _create_env(self):
        """创建环境"""
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, 1)
        
        # --- 添加手 ---
        hand_pose = gymapi.Transform()
        hand_pose.p = gymapi.Vec3(0, 0, 0)
        hand_pose.r = gymapi.Quat(0, 0, 0, 1)
        
        self.hand_actor = self.gym.create_actor(
            self.env, self.hand_asset, hand_pose, "hand", 0, 1
        )
        
        # 配置手部关节属性
        dof_props = self.gym.get_asset_dof_properties(self.hand_asset)
        dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        dof_props["stiffness"].fill(CONFIG["hand_stiffness"])
        dof_props["damping"].fill(CONFIG["hand_damping"])
        self.gym.set_actor_dof_properties(self.env, self.hand_actor, dof_props)
        
        # 保存关节限制
        self.dof_lower = dof_props["lower"].copy()
        self.dof_upper = dof_props["upper"].copy()
        
        # --- 添加物体 ---
        obj_pose = gymapi.Transform()
        obj_pose.p = gymapi.Vec3(*CONFIG["obj_init_pos"])
        rot = CONFIG["obj_init_rot"]
        obj_pose.r = gymapi.Quat(rot[0], rot[1], rot[2], rot[3])
        
        self.obj_actor = self.gym.create_actor(
            self.env, self.obj_asset, obj_pose, "object", 0, 2
        )
        self.gym.set_actor_scale(self.env, self.obj_actor, CONFIG["obj_scale"])
        
        # 设置相机视角
        self.gym.viewer_camera_look_at(
            self.viewer, None,
            gymapi.Vec3(0.6, 0.6, 0.6),  # 相机位置
            gymapi.Vec3(0, 0, 0.35)      # 看向的点
        )

    def _prepare_tensors(self):
        """准备 Tensor 用于控制"""
        # 获取状态 Tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states)
        self.dof_pos = self.dof_states[:self.num_hand_dofs, 0]
        self.dof_vel = self.dof_states[:self.num_hand_dofs, 1]
        
        _root_states = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(_root_states)
        
        # 初始化目标 DOF (与 URDF 的实际 DOF 数量匹配)
        self.target_dof_pos = torch.zeros(self.num_hand_dofs, dtype=torch.float32, device=self.device)
        
        # 设置初始基座位置 (使用 float() 转换避免 numpy/torch 类型冲突)
        self.target_dof_pos[0] = float(self.hand_base_pos[0])  # virtual_px
        self.target_dof_pos[1] = float(self.hand_base_pos[1])  # virtual_py
        self.target_dof_pos[2] = float(self.hand_base_pos[2])  # virtual_pz
        self.target_dof_pos[3] = float(self.hand_base_rot[0])  # virtual_rx (Roll)
        self.target_dof_pos[4] = float(self.hand_base_rot[1])  # virtual_ry (Pitch)
        self.target_dof_pos[5] = float(self.hand_base_rot[2])  # virtual_rz (Yaw)
        
        # 设置初始场景姿态
        self._apply_scene_preset(self.current_scene_preset)
        
        # 强制设置初始状态
        self.dof_pos[:] = self.target_dof_pos
        self.dof_vel[:] = 0.0
        self.gym.set_dof_state_tensor(self.sim, _dof_states)

    def _apply_scene_preset(self, preset_name):
        """应用场景预设姿态"""
        if preset_name not in CONFIG["scene_presets"]:
            print(f"Warning: Unknown preset '{preset_name}', using 'default'")
            preset_name = "default"
        
        preset = CONFIG["scene_presets"][preset_name]
        
        # 应用手部 DOF (27个值)
        hand_dof = preset["hand_dof"]
        for i, val in enumerate(hand_dof):
            if i < self.num_hand_dofs:
                # 裁剪到关节限制范围内
                self.target_dof_pos[i] = np.clip(val, self.dof_lower[i], self.dof_upper[i])
        
        # 应用物体位置和旋转
        self.obj_pos = np.array(preset["object_pos"], dtype=np.float32)
        self.obj_base_rot = np.array(preset["object_rot"], dtype=np.float32)  # 更新基础旋转
        self.obj_yaw = 0.0  # 重置 yaw 增量
        
        # 设置物体状态
        self.root_states[1, 0:3] = torch.tensor(self.obj_pos, device=self.device)
        obj_rot = preset["object_rot"]
        self.root_states[1, 3:7] = torch.tensor([obj_rot[0], obj_rot[1], obj_rot[2], obj_rot[3]], device=self.device)
        self.root_states[1, 7:13] = 0.0  # 速度清零
        
        # 应用物体更新
        object_idx = torch.tensor([1], dtype=torch.int32, device=self.device)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(object_idx),
            1
        )
        
        self.current_scene_preset = preset_name

    def _subscribe_keyboard_events(self):
        """订阅键盘事件"""
        # 物体位置控制
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "obj_move_forward")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "obj_move_backward")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "obj_move_left")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "obj_move_right")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "obj_move_up")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "obj_move_down")
        
        # 物体旋转控制 (只有 Yaw)
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_J, "obj_yaw_left")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_L, "obj_yaw_right")
        
        # 功能键
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "toggle_gravity")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset_object")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_P, "print_pose")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_T, "toggle_finger_preset")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F, "toggle_help")
        
        # 旋转轴可视化控制
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_axis_viz")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_B, "cycle_axis")
        
        # 手指关节单独控制 (数字键增加, Shift+数字键减少)
        # IsaacGym 按字母顺序: index[6-9], little[10-13], middle[14-17], ring[18-21], thumb[22-26]
        # 食指 (index): joint0-3 -> DoF 6, 7, 8, 9
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_1, "index_j0_inc")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_2, "index_j1_inc")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_3, "index_j2_inc")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_4, "index_j3_inc")
        # 中指 (middle): joint0 被锁定, joint1-3 -> DoF 15, 16, 17
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_5, "middle_j1_inc")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_6, "middle_j2_inc")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_7, "middle_j3_inc")
        # 拇指 (thumb): joint0-4 -> DoF 22, 23, 24, 25, 26
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_8, "thumb_j0_inc")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_9, "thumb_j1_inc")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_0, "thumb_j2_inc")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_MINUS, "thumb_j3_inc")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_EQUAL, "thumb_j4_inc")
        
        # I/K/U/O 用于减小手指关节值
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_I, "index_j0_dec")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_K, "index_j1_dec")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_U, "index_j2_dec")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_O, "index_j3_dec")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Y, "middle_j1_dec")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_H, "middle_j2_dec")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_N, "middle_j3_dec")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_M, "thumb_j0_dec")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_COMMA, "thumb_j1_dec")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_PERIOD, "thumb_j2_dec")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SLASH, "thumb_j3_dec")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_BACKSLASH, "thumb_j4_dec")
        
    def _print_dof_info(self):
        """打印 DOF 信息"""
        dof_names = self.gym.get_asset_dof_names(self.hand_asset)
        print("\n" + "="*50)
        print(f"DOF 信息 ({self.num_hand_dofs} DoF Flying Hand):")
        print("="*50)
        print("基座关节 (前6个):")
        for i in range(min(6, self.num_hand_dofs)):
            print(f"  [{i}] {dof_names[i]}: [{self.dof_lower[i]:.3f}, {self.dof_upper[i]:.3f}]")
        print(f"\n手指关节 (后{self.num_hand_dofs - 6}个):")
        for i in range(6, len(dof_names)):
            print(f"  [{i}] {dof_names[i]}: [{self.dof_lower[i]:.3f}, {self.dof_upper[i]:.3f}]")
        print("="*50 + "\n")

    def _print_help(self):
        """打印帮助信息"""
        # 获取当前旋转轴信息
        axis_str = f"[{self.current_rotation_axis[0]:.1f}, {self.current_rotation_axis[1]:.1f}, {self.current_rotation_axis[2]:.1f}]"
        help_text = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    交互式姿态调试工具 - 快捷键说明                              ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  物体位置控制:                                                                 ║
║    W/S - 前进/后退 (X轴)    A/D - 左/右 (Y轴)    Q/E - 上升/下降 (Z轴)        ║
║                                                                               ║
║  物体旋转控制:                                                                 ║
║    J/L - Yaw 左/右 (绕Z轴)                                                    ║
║                                                                               ║
║  旋转轴可视化控制:                                                             ║
║    V - 显示/隐藏旋转轴 (当前: {axis_viz})                                      ║
║    B - 循环切换旋转轴方向 (当前: {axis})                                       ║
║                                                                               ║
║  功能键:                                                                       ║
║    Space - 开启/关闭重力    R - 重置物体位置    P - 打印可复制姿态            ║
║    T     - 切换场景预设     F - 显示/隐藏帮助                                  ║
║                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  手指关节控制 (数字键增加/字母键减少):                                         ║
║  ─────────────────────────────────────────────────────────────────────────────║
║  食指 (index):   1/I - j0(侧摆)   2/K - j1   3/U - j2   4/O - j3             ║
║  中指 (middle):  5/Y - j1         6/H - j2   7/N - j3   (j0被锁定)            ║
║  拇指 (thumb):   8/M - j0   9/, - j1   0/. - j2   -// - j3   =/\\ - j4       ║
║                                                                               ║
║  步长: 位置={pos_step:.4f}m  旋转={rot_step:.4f}rad  关节={finger_step:.4f}rad║
╚═══════════════════════════════════════════════════════════════════════════════╝
""".format(pos_step=self.step_size_pos, rot_step=self.step_size_rot, 
           finger_step=CONFIG["step_size_finger"],
           axis_viz="ON" if self.show_rotation_axis else "OFF",
           axis=axis_str)
        print(help_text)

    def _process_events(self):
        """处理键盘事件"""
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.value <= 0:  # 只处理按下事件
                continue
                
            action = evt.action
            
            # --- 物体位置控制 ---
            if action == "obj_move_forward":
                self.obj_pos[0] += self.step_size_pos
                self._update_object_pose()
            elif action == "obj_move_backward":
                self.obj_pos[0] -= self.step_size_pos
                self._update_object_pose()
            elif action == "obj_move_left":
                self.obj_pos[1] += self.step_size_pos
                self._update_object_pose()
            elif action == "obj_move_right":
                self.obj_pos[1] -= self.step_size_pos
                self._update_object_pose()
            elif action == "obj_move_up":
                self.obj_pos[2] += self.step_size_pos
                self._update_object_pose()
            elif action == "obj_move_down":
                self.obj_pos[2] -= self.step_size_pos
                self._update_object_pose()
            
            # --- 物体旋转控制 (只有 Yaw) ---
            elif action == "obj_yaw_left":
                self.obj_yaw += self.step_size_rot
                self._update_object_pose()
            elif action == "obj_yaw_right":
                self.obj_yaw -= self.step_size_rot
                self._update_object_pose()
            
            # --- 功能键 ---
            elif action == "toggle_gravity":
                self._toggle_gravity()
            elif action == "reset_object":
                self._reset_object()
            elif action == "print_pose":
                self._export_pose()
            elif action == "toggle_finger_preset":
                self._cycle_scene_preset()
            elif action == "toggle_help":
                self.show_help = not self.show_help
                if self.show_help:
                    self._print_help()
                else:
                    print("帮助已隐藏 (按 F 重新显示)")
            
            # --- 旋转轴可视化控制 ---
            elif action == "toggle_axis_viz":
                self._toggle_rotation_axis_viz()
            elif action == "cycle_axis":
                self._cycle_rotation_axis()
            
            # --- 手指关节单独控制 ---
            # IsaacGym 按字母顺序: index[6-9], little[10-13], middle[14-17], ring[18-21], thumb[22-26]
            # 食指 (index) 增加
            elif action == "index_j0_inc":
                self._adjust_single_joint(6, CONFIG["step_size_finger"], "index_j0")
            elif action == "index_j1_inc":
                self._adjust_single_joint(7, CONFIG["step_size_finger"], "index_j1")
            elif action == "index_j2_inc":
                self._adjust_single_joint(8, CONFIG["step_size_finger"], "index_j2")
            elif action == "index_j3_inc":
                self._adjust_single_joint(9, CONFIG["step_size_finger"], "index_j3")
            # 中指 (middle) 增加 (joint0 被锁定, 从 joint1 开始)
            elif action == "middle_j1_inc":
                self._adjust_single_joint(15, CONFIG["step_size_finger"], "middle_j1")
            elif action == "middle_j2_inc":
                self._adjust_single_joint(16, CONFIG["step_size_finger"], "middle_j2")
            elif action == "middle_j3_inc":
                self._adjust_single_joint(17, CONFIG["step_size_finger"], "middle_j3")
            # 拇指 (thumb) 增加
            elif action == "thumb_j0_inc":
                self._adjust_single_joint(22, CONFIG["step_size_finger"], "thumb_j0")
            elif action == "thumb_j1_inc":
                self._adjust_single_joint(23, CONFIG["step_size_finger"], "thumb_j1")
            elif action == "thumb_j2_inc":
                self._adjust_single_joint(24, CONFIG["step_size_finger"], "thumb_j2")
            elif action == "thumb_j3_inc":
                self._adjust_single_joint(25, CONFIG["step_size_finger"], "thumb_j3")
            elif action == "thumb_j4_inc":
                self._adjust_single_joint(26, CONFIG["step_size_finger"], "thumb_j4")
            # 食指 (index) 减少
            elif action == "index_j0_dec":
                self._adjust_single_joint(6, -CONFIG["step_size_finger"], "index_j0")
            elif action == "index_j1_dec":
                self._adjust_single_joint(7, -CONFIG["step_size_finger"], "index_j1")
            elif action == "index_j2_dec":
                self._adjust_single_joint(8, -CONFIG["step_size_finger"], "index_j2")
            elif action == "index_j3_dec":
                self._adjust_single_joint(9, -CONFIG["step_size_finger"], "index_j3")
            # 中指 (middle) 减少
            elif action == "middle_j1_dec":
                self._adjust_single_joint(15, -CONFIG["step_size_finger"], "middle_j1")
            elif action == "middle_j2_dec":
                self._adjust_single_joint(16, -CONFIG["step_size_finger"], "middle_j2")
            elif action == "middle_j3_dec":
                self._adjust_single_joint(17, -CONFIG["step_size_finger"], "middle_j3")
            # 拇指 (thumb) 减少
            elif action == "thumb_j0_dec":
                self._adjust_single_joint(22, -CONFIG["step_size_finger"], "thumb_j0")
            elif action == "thumb_j1_dec":
                self._adjust_single_joint(23, -CONFIG["step_size_finger"], "thumb_j1")
            elif action == "thumb_j2_dec":
                self._adjust_single_joint(24, -CONFIG["step_size_finger"], "thumb_j2")
            elif action == "thumb_j3_dec":
                self._adjust_single_joint(25, -CONFIG["step_size_finger"], "thumb_j3")
            elif action == "thumb_j4_dec":
                self._adjust_single_joint(26, -CONFIG["step_size_finger"], "thumb_j4")

    def _update_object_pose(self):
        """更新物体位姿 (根据 obj_pos 和 obj_yaw)"""
        # 设置位置
        self.root_states[1, 0:3] = torch.tensor(self.obj_pos, device=self.device)
        
        # 计算四元数 (只绕 Z 轴旋转 yaw)
        # 基础旋转来自当前预设，再叠加 yaw 增量
        base_rot = self.obj_base_rot  # (x,y,z,w)
        # 使用 Z 轴旋转的四元数: (0, 0, sin(yaw/2), cos(yaw/2))
        half_yaw = self.obj_yaw / 2.0
        yaw_quat = np.array([0, 0, np.sin(half_yaw), np.cos(half_yaw)])
        # 四元数乘法: yaw_quat * base_rot
        combined_rot = self._quat_multiply(yaw_quat, base_rot)
        self.root_states[1, 3:7] = torch.tensor(combined_rot, device=self.device)
        
        # 速度清零
        self.root_states[1, 7:13] = 0.0
        
        # 应用更新
        object_idx = torch.tensor([1], dtype=torch.int32, device=self.device)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(object_idx),
            1
        )
    
    def _quat_multiply(self, q1, q2):
        """四元数乘法 (x,y,z,w 格式)"""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])
    
    def _adjust_single_joint(self, dof_idx, delta, joint_name=""):
        """调整单个关节"""
        if dof_idx >= self.num_hand_dofs:
            print(f"警告: DoF 索引 {dof_idx} 超出范围")
            return
        
        # 检查是否被锁定
        if self.dof_lower[dof_idx] == self.dof_upper[dof_idx]:
            print(f"警告: {joint_name} (DoF {dof_idx}) 被锁定")
            return
        
        old_val = self.target_dof_pos[dof_idx].item()
        new_val = np.clip(old_val + delta, self.dof_lower[dof_idx], self.dof_upper[dof_idx])
        self.target_dof_pos[dof_idx] = new_val
        print(f"{joint_name}: {old_val:.3f} -> {new_val:.3f}")

    def _toggle_gravity(self):
        """切换重力"""
        self.gravity_active = not self.gravity_active
        
        sim_params = self.gym.get_sim_params(self.sim)
        if self.gravity_active:
            sim_params.gravity = gymapi.Vec3(0, 0, -9.81)
            print("重力: 开启 (按 R 重置物体)")
        else:
            sim_params.gravity = gymapi.Vec3(0, 0, 0)
            print("重力: 关闭")
        self.gym.set_sim_params(self.sim, sim_params)

    def _reset_object(self):
        """重置物体位置到当前预设的初始状态"""
        print("重置物体位置到当前预设...")
        
        # 获取当前预设的配置
        preset = CONFIG["scene_presets"].get(self.current_scene_preset, CONFIG["scene_presets"]["default"])
        
        # 重置控制变量到预设值
        self.obj_pos = np.array(preset["object_pos"], dtype=np.float32)
        self.obj_base_rot = np.array(preset["object_rot"], dtype=np.float32)
        self.obj_yaw = 0.0  # 重置 yaw 增量
        
        # 物体是第二个 actor (index=1)
        self.root_states[1, 0:3] = torch.tensor(self.obj_pos, device=self.device)
        self.root_states[1, 3:7] = torch.tensor(self.obj_base_rot, device=self.device)
        self.root_states[1, 7:13] = 0.0  # 速度清零
        
        # 只更新物体 (index=1)
        object_idx = torch.tensor([1], dtype=torch.int32, device=self.device)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(object_idx),
            1
        )

    def _cycle_scene_preset(self):
        """循环切换场景预设"""
        presets = list(CONFIG["scene_presets"].keys())
        current_idx = presets.index(self.current_scene_preset)
        next_idx = (current_idx + 1) % len(presets)
        next_preset = presets[next_idx]
        
        self._apply_scene_preset(next_preset)
        print(f"场景预设切换为: {next_preset}")

    def _toggle_rotation_axis_viz(self):
        """切换旋转轴可视化开关"""
        self.show_rotation_axis = not self.show_rotation_axis
        status = "ON" if self.show_rotation_axis else "OFF"
        axis_str = f"[{self.current_rotation_axis[0]:.2f}, {self.current_rotation_axis[1]:.2f}, {self.current_rotation_axis[2]:.2f}]"
        print(f"旋转轴可视化: {status} (当前轴: {axis_str})")
    
    def _cycle_rotation_axis(self):
        """循环切换旋转轴预设"""
        presets = CONFIG["rotation_axis_presets"]
        self.rotation_axis_preset_idx = (self.rotation_axis_preset_idx + 1) % len(presets)
        preset = presets[self.rotation_axis_preset_idx]
        self.current_rotation_axis = np.array(preset["axis"], dtype=np.float32)
        print(f"旋转轴切换为: {preset['name']} = {preset['axis']}")
        print(f"  提示: 如需保存此轴，请编辑 configs/task/LinkerHandHora.yaml 中的 rotation_axis")
    
    def _draw_rotation_axis(self):
        """绘制旋转轴可视化"""
        if not self.show_rotation_axis:
            return
        
        # 获取物体当前位置作为旋转轴的中心点
        obj_pos = self.root_states[1, 0:3].cpu().numpy()
        
        # 旋转轴线段的长度和颜色
        axis_length = 0.3  # 30cm
        
        # 计算轴线段的两个端点
        axis_dir = self.current_rotation_axis / (np.linalg.norm(self.current_rotation_axis) + 1e-8)
        start_point = obj_pos - axis_dir * axis_length / 2
        end_point = obj_pos + axis_dir * axis_length / 2
        
        # 绘制旋转轴 (黄色虚线)
        # IsaacGym 使用 add_lines API
        lines = np.array([
            [start_point[0], start_point[1], start_point[2], end_point[0], end_point[1], end_point[2]]
        ], dtype=np.float32)
        
        # 颜色: 黄色 (RGB)
        colors = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)
        
        self.gym.add_lines(self.viewer, self.env, 1, lines, colors)
        
        # 在轴的正方向端点绘制一个小箭头指示方向
        # 箭头由两条短线组成
        arrow_length = 0.03  # 3cm
        # 找一个垂直于旋转轴的向量
        if abs(axis_dir[2]) < 0.9:
            perp1 = np.cross(axis_dir, np.array([0, 0, 1]))
        else:
            perp1 = np.cross(axis_dir, np.array([1, 0, 0]))
        perp1 = perp1 / (np.linalg.norm(perp1) + 1e-8)
        perp2 = np.cross(axis_dir, perp1)
        
        # 箭头的两个分支
        arrow_base = end_point - axis_dir * arrow_length
        arrow1 = arrow_base + perp1 * arrow_length * 0.5
        arrow2 = arrow_base - perp1 * arrow_length * 0.5
        
        arrow_lines = np.array([
            [end_point[0], end_point[1], end_point[2], arrow1[0], arrow1[1], arrow1[2]],
            [end_point[0], end_point[1], end_point[2], arrow2[0], arrow2[1], arrow2[2]]
        ], dtype=np.float32)
        arrow_colors = np.array([[1.0, 0.8, 0.0], [1.0, 0.8, 0.0]], dtype=np.float32)
        
        self.gym.add_lines(self.viewer, self.env, 2, arrow_lines, arrow_colors)

    def _adjust_finger_group(self, finger_name, delta):
        """调整指定手指组的弯曲程度"""
        # 重要: IsaacGym 加载 URDF 后的关节顺序是按字母顺序排列的！
        # Flying Hand (27 DoF) 顺序:
        #   virtual: 0-5, index: 6-9, little: 10-13, middle: 14-17, ring: 18-21, thumb: 22-26
        finger_ranges = {
            "index": (6, 10),   # 食指
            "little": (10, 14), # 小拇指
            "middle": (14, 18), # 中指
            "ring": (18, 22),   # 无名指
            "thumb": (22, 27)   # 拇指
        }
        
        if finger_name in finger_ranges:
            start, end = finger_ranges[finger_name]
            for i in range(start, min(end, self.num_hand_dofs)):
                # 跳过被锁定的关节 (如 middle_joint0)
                if self.dof_lower[i] == self.dof_upper[i]:
                    continue
                new_val = self.target_dof_pos[i].item() + delta
                self.target_dof_pos[i] = np.clip(new_val, self.dof_lower[i], self.dof_upper[i])
            print(f"调整 {finger_name} 手指")

    def _export_pose(self):
        """导出当前姿态 (可复制格式)"""
        # 获取当前手部 DOF 状态 (25维: 6 base + 19 fingers)
        current_dof = self.dof_pos.cpu().numpy().tolist()
        
        # 分离基座和手指
        base_dof = current_dof[:6]
        finger_dof = current_dof[6:]
        
        # 获取物体位姿
        obj_pos = self.root_states[1, 0:3].cpu().numpy().tolist()
        obj_rot = self.root_states[1, 3:7].cpu().numpy().tolist()  # xyzw
        
        # 格式化输出
        print("\n" + "="*75)
        print("█████ 可复制的姿态配置 █████")
        print("="*75)
        
        # 格式1: 适用于 linker_hand_grasp.py 的 canonical_pose_dict (21 DoF 固定基座)
        # 注意: 固定基座版本的 URDF 只有 21 个 DOF
        print("\n【格式1】固定基座版本 (21 DoF, 用于 linker_hand_grasp.py):")
        print("-"*75)
        print("# 注意: 固定基座版本需要去掉前 6 个基座关节")
        print("# 物体位置需要根据基座偏移进行调整")
        print(f"'hand': {finger_dof},")
        # 调整物体位置 (考虑基座偏移)
        adjusted_obj_pos = [
            obj_pos[0] - base_dof[0],  # x - px
            obj_pos[1] - base_dof[1],  # y - py  
            obj_pos[2] - base_dof[2],  # z - pz
        ]
        print(f"'object': {adjusted_obj_pos + obj_rot}")
        
        # 格式2: 完整 25 DoF Flying Hand
        print("\n【格式2】Flying Hand 版本 (25 DoF, 用于 flying_hand_grasp.py):")
        print("-"*75)
        print("'flying_hand': [")
        print("    {")
        print(f"        'hand': {current_dof},")
        print(f"        'object': {obj_pos + obj_rot}")
        print("    }")
        print("]")
        
        # 格式3: 详细状态信息
        print("\n【格式3】详细状态:")
        print("-"*75)
        print(f"# 基座状态 (6 DoF):")
        print(f"#   平移 (x,y,z): [{base_dof[0]:.6f}, {base_dof[1]:.6f}, {base_dof[2]:.6f}]")
        print(f"#   旋转 (r,p,y): [{base_dof[3]:.6f}, {base_dof[4]:.6f}, {base_dof[5]:.6f}]")
        print(f"# 手指状态 (19 DoF): {[round(x, 4) for x in finger_dof]}")
        print(f"# 物体世界坐标:")
        print(f"#   位置: {[round(x, 6) for x in obj_pos]}")
        print(f"#   旋转: {[round(x, 6) for x in obj_rot]}")
        print("="*75 + "\n")

    def _update_hand_base(self):
        """更新手部基座关节"""
        # 裁剪到关节限制范围 (使用 float() 转换避免类型问题)
        for i in range(3):
            self.target_dof_pos[i] = float(np.clip(
                self.hand_base_pos[i], self.dof_lower[i], self.dof_upper[i]
            ))
        for i in range(3):
            self.target_dof_pos[3+i] = float(np.clip(
                self.hand_base_rot[i], self.dof_lower[3+i], self.dof_upper[3+i]
            ))

    def run(self):
        """主循环"""
        if self.show_help:
            self._print_help()
        
        last_status_time = time.time()
        status_interval = 2.0  # 每2秒打印一次状态
        
        while not self.gym.query_viewer_has_closed(self.viewer):
            # 处理键盘事件
            self._process_events()
            
            # 更新基座目标
            self._update_hand_base()
            
            # 应用关节目标
            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(self.target_dof_pos)
            )
            
            # 仿真步进
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            
            # 刷新状态
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            
            # 定期打印状态
            current_time = time.time()
            if current_time - last_status_time >= status_interval:
                # 从刷新后的 tensor 读取物体状态
                obj_pos = self.root_states[1, 0:3].cpu().numpy()
                obj_rot = self.root_states[1, 3:7].cpu().numpy()
                # 获取关键手指关节值
                idx_vals = [self.dof_pos[i].item() for i in [6,7,8,9]]
                mid_vals = [self.dof_pos[i].item() for i in [15,16,17]]
                thumb_vals = [self.dof_pos[i].item() for i in [22,23,24,25,26]]
                print(f"[状态] 物体: pos=[{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}], yaw={self.obj_yaw:.3f} | "
                      f"重力={'ON' if self.gravity_active else 'OFF'}")
                print(f"       食指: {[f'{v:.2f}' for v in idx_vals]} | 中指: {[f'{v:.2f}' for v in mid_vals]} | "
                      f"拇指: {[f'{v:.2f}' for v in thumb_vals]}")
                # 打印旋转轴状态
                if self.show_rotation_axis:
                    axis_str = f"[{self.current_rotation_axis[0]:.2f}, {self.current_rotation_axis[1]:.2f}, {self.current_rotation_axis[2]:.2f}]"
                    print(f"       旋转轴: {axis_str} (按V隐藏/B切换)")
                last_status_time = current_time
            
            # 绘制旋转轴可视化
            self._draw_rotation_axis()
            
            # 渲染
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
        
        # 清理
        print("关闭...")
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Interactive Pose Tuner for 25 DoF Flying Hand')
    parser.add_argument('--object', type=str, default='spinning_pen',
                        choices=['spinning_pen', 'pencil'],
                        help='选择物体类型 (default: spinning_pen)')
    parser.add_argument('--preset', type=str, default='triangle_grasp',
                        choices=list(CONFIG['scene_presets'].keys()),
                        help='初始场景预设 (default: triangle_grasp)')
    parser.add_argument('--scale', type=float, default=None,
                        help='物体缩放比例 (spinning_pen 默认 1.0, pencil 默认 0.3)')
    parser.add_argument('--no-ground', action='store_true',
                        help='不添加地面')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║     Interactive Pose Tuner for 25 DoF Flying Hand             ║
    ║                                                               ║
    ║  整合 find_pos + find_rot + gravity_sim 功能                 ║
    ║  支持键盘交互控制基座位置、旋转和手指姿态                    ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # 根据命令行参数更新配置
    if args.object == 'pencil':
        CONFIG['obj_urdf'] = CONFIG['obj_urdf_pencil']
        if args.scale is None:
            CONFIG['obj_scale'] = 0.3  # pencil 默认缩放
        print(f"使用物体: pencil (scale={CONFIG['obj_scale']})")
    else:
        print(f"使用物体: spinning_pen (scale={CONFIG['obj_scale']})")
    
    if args.scale is not None:
        CONFIG['obj_scale'] = args.scale
    
    CONFIG['default_scene_preset'] = args.preset
    CONFIG['add_ground'] = not args.no_ground
    
    tuner = InteractivePoseTuner()
    tuner.run()


if __name__ == '__main__':
    main()
