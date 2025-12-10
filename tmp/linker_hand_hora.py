import os
from typing import Optional
import torch
import omegaconf
import numpy as np
import math

from collections import OrderedDict

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import quat_conjugate, quat_mul, to_torch, quat_apply, tensor_clamp, torch_rand_float, quat_from_euler_xyz

from penspin.utils.point_cloud_prep import sample_cylinder
from .base.vec_task import VecTask
from penspin.utils.misc import tprint
# Import waypoint tracking utilities for Triangle Pass
from penspin.utils.tp_waypoints import (
    initialize_waypoints,
    compute_phase_from_object_rotation,
    get_target_fingertip_by_phase,
    compute_waypoint_tracking_reward,
)
# Import centralized robot dimension constants
from penspin.utils.robot_config import (
    NUM_DOF,
    NUM_DOF_REDUCED,
    PROPRIO_DIM,
    CONTACT_DIM,
    FINGERTIP_CNT,
    FINGERTIP_LINK_NAMES,
    CONTACT_LINK_NAMES,
    FINGERTIP_POS_DIM,
    PRIV_FINGERTIP_ROT_DIM,
    OBS_WITH_CONTACT_FINGERTIP_DIM,
    ACTIVE_FINGER_INDICES_REDUCED,
    ActionSpaceMapper,
    # Flying Hand 相关常量
    NUM_FLYING_DOF,
    NUM_TOTAL_DOF_FLYING,
    FLYING_DOF_NAMES,
    FLYING_DOF_INDICES,
    FLYING_DEFAULT_HEIGHT,
    FLYING_HEIGHT_LOWER,
    FLYING_HEIGHT_UPPER,
    FLYING_XY_LIMIT,
    # Flying Hand 相对限位常量
    FLYING_RELATIVE_XY_LIMIT,
    FLYING_RELATIVE_Z_LIMIT,
    FLYING_RELATIVE_ROT_LIMIT,
)
# Import Time-Warping Orchestrator for Space E Curriculum Learning
from penspin.utils.time_warping import TimeWarpingOrchestrator

# 刚体的 位置 (3) + 姿态 (4, 四元数) + 线速度 (3) + 角速度 (3)
RIGID_BODY_STATES = 13

# CHECKLIST
# 用于查找生成的初始化抓取状态文件夹
NUM_POSE_PER_CACHE = '50k'
# 物体平均位置
OBJ_CANON_POS = [-0.12593473494052887, 0.027405261993408203, 0.5] # 3pose
# [-0.11722512543201447, 0.006986482068896294, 0.1717524379491806] # 4pose
# [-0.12593473494052887, 0.027405261993408203, 0.16902321577072144] # 3pose
# [-0.13216303288936615, 0.022801531478762627, 0.16341765224933624] # 6pose

# Contact上下界
CONTACT_THRESH = 0.02

class LinkerHandHora(VecTask):
    def __init__(self, config, sim_device, graphics_device_id, headless):
        self.config = config
        # before calling init in VecTask, need to do
        # 1. setup randomization
        self._setup_domain_rand_config(config['env']['randomization'])
        # 2. setup privileged information
        self._setup_priv_option_config(config['env']['privInfo'])
        # 3. setup object assets
        self._setup_object_info(config['env']['object'])
        # 4. setup rewards
        self._setup_reward_config(config['env']['reward'])
        # 5. setup flying hand configuration (必须在 action space 之前)
        self._setup_flying_hand_config(config['env'])
        # 6. setup action space (支持 flying hand + 禁用手指)
        self._setup_action_space_config(config['env'])

        self.base_obj_scale = config['env']['baseObjScale']
        # print("缩放比例", self.base_obj_scale)

        self.save_init_pose = config['env']['genGrasps'] #这个参数是为了在genGrasp步骤生成初始化抓取状态

        self.aggregate_mode = self.config['env']['aggregateMode']
        self.up_axis = 'z'
        self.rotation_axis = config['env']['rotation_axis']
        # 早期终止阈值（支持新旧配置键名，向后兼容）
        self.relative_z_drop_threshold = self.config['env'].get(
            'relative_z_drop_threshold',
            self.config['env'].get('reset_height_threshold', 0.06)  # 向后兼容旧配置
        )
        self.pencil_tilt_threshold = self.config['env'].get('pencil_tilt_threshold', 0.12)
        self.grasp_cache_name = self.config['env']['grasp_cache_name']
        self.canonical_pose_category = config['env']['genGraspCategory']
        self.num_pose_per_cache = NUM_POSE_PER_CACHE
        self.with_camera = config['env']['enableCameraSensors']
        self.enable_obj_ends = config['env']['enable_obj_ends']
        self.init_pose_mode = config['env']['initPoseMode']
        # 根据是否启用 Flying Hand 设置 DOF 数量
        if self.flying_hand_enabled:
            self.num_linker_hand_dofs = NUM_TOTAL_DOF_FLYING  # 27 = 6 (base) + 21 (hand)
        else:
            self.num_linker_hand_dofs = self.config['env']['numActions']  # 21
        # Important: map CUDA device IDs to Vulkan ones.
        graphics_device_id = 0

        super().__init__(config, sim_device, graphics_device_id, headless)
        
        # 覆盖动作空间（当动作维度与默认配置不同时）
        # 这包括：Flying Hand、禁用手指、或两者组合
        if self.actual_action_dim != self.config['env']['numActions']:
            from gym import spaces
            self.num_actions = self.actual_action_dim
            self.act_space = spaces.Box(
                np.ones(self.actual_action_dim, dtype=np.float32) * -1., 
                np.ones(self.actual_action_dim, dtype=np.float32) * 1.
            )
            print(f"动作空间已更新: {self.act_space.shape} (策略输出维度: {self.actual_action_dim})")

        # ============================================================
        # 覆盖观察空间（当 DOF 数量与默认配置不同时）
        # ============================================================
        # obs_buf 维度 = 6 * num_dofs（3个时间步 × 2（位置+目标））
        # 默认配置 numObservations=126 (21 DOF × 6)，Flying Hand 需要 162 (27 DOF × 6)
        # 注意：此时 self.num_dofs 尚未设置，使用 num_linker_hand_dofs 替代
        actual_num_obs = 6 * self.num_linker_hand_dofs
        if actual_num_obs != self.config['env']['numObservations']:
            from gym import spaces
            self.num_observations = actual_num_obs
            self.obs_space = spaces.Box(
                np.ones(actual_num_obs, dtype=np.float32) * -np.Inf,
                np.ones(actual_num_obs, dtype=np.float32) * np.Inf
            )
            print(f"观察空间已更新: {self.obs_space.shape} (实际维度: {actual_num_obs})")

        self.eval_done_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.debug_viz = self.config['env']['enableDebugVis']
        self.max_episode_length = self.config['env']['episodeLength']
        self.dt = self.sim_params.dt

        if self.viewer:
            cam_pos = gymapi.Vec3(0.0, 0.4, 1.5)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        # force_sensor = self.gym.acquire_force_sensor_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.linker_hand_default_dof_pos = torch.zeros(self.num_linker_hand_dofs, dtype=torch.float, device=self.device)

        # 通过传引用表示一个轻量级对象，包含了张量在 GPU 或 CPU 内存中的地址、数据类型和形状等信息。它不包含实际的数据，但指向存储数据的内存区域。
        # 将 Gym 的张量描述符 dof_state_tensor 转换为 PyTorch Tensor 对象 self.dof_state
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        print("Contact Tensor Dimension [1, numRigidBody, 3]", self.contact_forces.shape)

        self.linker_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_linker_hand_dofs]
        self.linker_hand_dof_pos = self.linker_hand_dof_state[..., 0] # 关节角（Revolute）
        self.linker_hand_dof_vel = self.linker_hand_dof_state[..., 1] # 关节角速度（Revolute）

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, RIGID_BODY_STATES)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, RIGID_BODY_STATES)

        self._refresh_gym()

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        
        # ============================================================
        # 重新分配观测缓冲区（使用实际的 DOF 数量）
        # ============================================================
        # vec_task.py 中的 _allocate_buffers 使用硬编码的 NUM_DOF=21，
        # 但 Flying Hand 需要 27 DOF。这里根据实际的 num_dofs 重新分配。
        # PROPRIO_DIM = 2 * num_dofs (当前位置 + 目标位置)
        self.actual_proprio_dim = 2 * self.num_dofs  # 保存为实例变量，用于 compute_observations
        if self.config['env']['privInfo']['enable_tactile']:
            obs_buf_lag_dim = self.actual_proprio_dim + CONTACT_DIM + FINGERTIP_POS_DIM
        else:
            obs_buf_lag_dim = self.actual_proprio_dim
        self.obs_buf_lag_history = torch.zeros(
            (self.num_envs, 80, obs_buf_lag_dim), 
            device=self.device, dtype=torch.float
        )
        # obs_buf 使用 6 * num_dofs (3个时间步 × 2 × num_dofs)
        self.obs_buf = torch.zeros(
            (self.num_envs, 6 * self.num_dofs), 
            device=self.device, dtype=torch.float
        )
        # 重新分配 proprio_hist_buf（之前在 _allocate_task_buffer 中使用了 PROPRIO_DIM 常量）
        self.proprio_hist_buf = torch.zeros(
            (self.num_envs, self.prop_hist_len, self.actual_proprio_dim),
            device=self.device, dtype=torch.float
        )
        # 重新分配噪声缓冲区（需要与 num_dofs 匹配，因为它们与 dof_pos 相加）
        self.random_obs_noise_e = torch.zeros(
            (self.num_envs, self.num_dofs), device=self.device, dtype=torch.float
        )
        self.random_action_noise_e = torch.zeros(
            (self.num_envs, self.num_dofs), device=self.device, dtype=torch.float
        )

        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        # object apply random forces parameters
        self.force_scale = self.config['env'].get('forceScale', 0.0)
        self.random_force_prob_scalar = self.config['env'].get('randomForceProbScalar', 0.0)
        self.force_decay = self.config['env'].get('forceDecay', 0.99)
        self.force_decay_interval = self.config['env'].get('forceDecayInterval', 0.08)
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

        self.last_contacts = torch.zeros((self.num_envs, self.num_contacts), dtype=torch.float, device=self.device)

        # ================================================================
        # 加载抓取缓存 (Grasp Cache)
        # ================================================================
        # 缓存命名格式: {num_poses}_{total_samples}_{cache_dim}_grasp_cache.npy
        # 例如: 3_30000_61_grasp_cache.npy
        # 
        # 支持两种缓存格式:
        # - 61维(新): [hand_actual(27) + hand_target(27) + obj_pos(3) + obj_rot(4)]
        # - 34维(旧): [hand_dof(27) + obj_pos(3) + obj_rot(4)]
        # ================================================================
        if not self.save_init_pose:  # 正常训练模式（非缓存生成模式）
            cache_path = f'cache/{self.grasp_cache_name}_grasp_cache.npy'
            if not os.path.exists(cache_path):
                raise FileNotFoundError(f"[LinkerHandHora] 缓存文件不存在: {cache_path}")
            
            self.saved_grasping_states = torch.from_numpy(np.load(cache_path)).float().to(self.device)
            cache_dim = self.saved_grasping_states.shape[1]
            
            # 检测缓存格式: 61维(新) vs 34维(旧)
            self.cache_is_new_format = (cache_dim == self.num_linker_hand_dofs * 2 + 7)  # 27*2 + 7 = 61
            format_str = "61维(actual+target)" if self.cache_is_new_format else "34维(旧格式)"
            
            print(f"[LinkerHandHora] 加载缓存: {cache_path}")
            print(f"  缓存形状: {self.saved_grasping_states.shape}")
            print(f"  缓存格式: {format_str}")

        # ================================================================
        # 旋转轴初始化 (Rotation Axis Initialization)
        # ================================================================
        # 支持两种配置格式:
        # 1. 旧格式 (字符串): '+z', '-z', '+x' 等
        # 2. 新格式 (三维向量): [0, 0, 1], [0, 0, -1], [1, 0, 0] 等
        #
        # 旋转轴定义: 期望笔绕此轴旋转（右手螺旋法则）
        # 正值分量表示逆时针旋转为正奖励，负值分量表示顺时针旋转为正奖励
        # ================================================================
        self.rot_axis_buf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.rot_axis_task = None
        
        # 检查是否为列表/元组格式 (包括 OmegaConf 的 ListConfig)
        is_list_format = hasattr(self.rotation_axis, '__iter__') and not isinstance(self.rotation_axis, str)
        
        if is_list_format:
            # 新格式: 三维向量 [x, y, z]
            rot_axis_vec = torch.tensor(list(self.rotation_axis), device=self.device, dtype=torch.float)
            # 归一化旋转轴
            rot_axis_norm = torch.norm(rot_axis_vec) + 1e-8
            rot_axis_vec = rot_axis_vec / rot_axis_norm
            self.rot_axis_buf[:] = rot_axis_vec.unsqueeze(0).expand(self.num_envs, -1)
            print(f"[旋转轴] 使用三维向量格式: {list(self.rotation_axis)} (归一化后: {rot_axis_vec.tolist()})")
        else:
            # 旧格式: 字符串 '+z', '-z' 等
            sign, axis = self.rotation_axis[0], self.rotation_axis[1]
            axis_index = ['x', 'y', 'z'].index(axis)
            self.rot_axis_buf[:, axis_index] = 1
            self.rot_axis_buf[:, axis_index] = -self.rot_axis_buf[:, axis_index] if sign == '-' else self.rot_axis_buf[:, axis_index]
            print(f"[旋转轴] 使用字符串格式: {self.rotation_axis} -> {self.rot_axis_buf[0].tolist()}")

        # useful buffers
        self.init_pose_buf = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        
        # ============================================================
        # 每个环境独立的相对限位张量 (Per-Environment Relative Limits)
        # ============================================================
        # 这些张量会在 reset_idx 时根据 init_pose_buf 动态更新
        # 确保每个环境的动作空间相对于其初始位置对称
        # ============================================================
        if self.use_relative_limit:
            # 初始化为全局绝对限位（会在 reset 时更新）
            self.env_dof_lower_limits = self.linker_hand_dof_lower_limits.unsqueeze(0).expand(self.num_envs, -1).clone()
            self.env_dof_upper_limits = self.linker_hand_dof_upper_limits.unsqueeze(0).expand(self.num_envs, -1).clone()
        # there is an extra dim [self.control_freq_inv] because we want to get a mean over multiple control steps
        # 注意：torques 是 PD 控制计算的输出，维度应该是 num_dofs（而不是 num_actions）
        self.torques = torch.zeros((self.num_envs, self.control_freq_inv, self.num_dofs), device=self.device, dtype=torch.float)
        self.dof_vel_finite_diff = torch.zeros((self.num_envs, self.control_freq_inv, self.num_dofs), device=self.device, dtype=torch.float)

        # --- calculate velocity at control frequency instead of simulated frequency
        self.object_pos_prev = self.object_pos.clone()
        self.object_rot_prev = self.object_rot.clone()
        # 使用世界坐标系位置用于速度计算
        self.ft_pos_prev_world = self.fingertip_pos_world.clone()
        self.ft_rot_prev = self.fingertip_orientation.clone()
        self.dof_vel_prev = self.dof_vel_finite_diff.clone()
        
        # Flying base 位置历史，用于计算速度惩罚
        if self.flying_hand_enabled:
            self.flying_base_pos_prev = self.linker_hand_dof_pos[:, :NUM_FLYING_DOF].clone()

        self.obj_linvel_at_cf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.obj_angvel_at_cf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.ft_linvel_at_cf = torch.zeros((self.num_envs, FINGERTIP_CNT * 3), device=self.device, dtype=torch.float)
        self.ft_angvel_at_cf = torch.zeros((self.num_envs, FINGERTIP_CNT * 3), device=self.device, dtype=torch.float)
        self.dof_acc = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        # ----

        assert type(self.p_gain) in [int, float] and type(self.d_gain) in [int, float], 'assume p_gain and d_gain are only scalars'
        # 注意：p_gain 和 d_gain 需要与 DOF 维度匹配（而不是 action 维度），
        # 因为在 update_low_level_control 中它们需要与 cur_targets、dof_pos、dof_vel 相乘
        self.p_gain = torch.ones((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float) * self.p_gain
        self.d_gain = torch.ones((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float) * self.d_gain

        # ============================================================
        # Space E 时空扭曲课程学习初始化
        # ============================================================
        # 初始化时空扭曲编排器
        curriculum_config = config['env'].get('curriculum', {})
        self.time_warper = TimeWarpingOrchestrator(curriculum_config)
        
        # 缓存原始物理参数 (Baseline Values)
        # 这些值将作为 alpha=1.0 时的参考，所有缩放都基于这些值计算
        # 必须深拷贝，因为后续我们会直接修改 sim_params
        self.base_gravity = list(config['sim']['gravity'])  # [0, 0, -9.81]
        self.base_pgain = config['env']['controller']['pgain']
        self.base_dgain = config['env']['controller']['dgain']
        self.base_torque_limit = config['env']['controller']['torque_limit']
        
        # 缓存 PhysX 求解器阈值 (用于 Space E 动态缩放)
        # 低 alpha 下速度被缩放为 v_sim = alpha * v_real
        # 如果不缩放阈值，原本应该反弹的碰撞会变成粘性接触
        physx_config = config['sim'].get('physx', {})
        self.base_bounce_threshold = physx_config.get('bounce_threshold_velocity', 0.2)
        self.base_contact_offset = physx_config.get('contact_offset', 0.002)
        
        # Flying Hand 的基础参数缓存
        if self.flying_hand_enabled:
            self.base_flying_pgain = config['env']['flyingHand']['basePgain']
            self.base_flying_dgain = config['env']['flyingHand']['baseDgain']
        else:
            # 非 Flying Hand 模式，设置默认值（不会被使用）
            self.base_flying_pgain = 1000.0
            self.base_flying_dgain = 50.0
        
        # 首次应用物理参数（在 alpha_start 阶段）
        # 注意：此时 envs 和 sim 已经创建完成（super().__init__ 调用后）
        self.apply_curriculum_physics()
        # ============================================================

        # debug and understanding statistics
        self.evaluate = self.config['on_evaluation']
        self.evaluate_cache_name = self.config['eval_cache_name']
        self.stat_sum_rewards = [0 for _ in self.object_type_list]  # all episode reward
        self.stat_sum_episode_length = [0 for _ in self.object_type_list]  # average episode length
        self.stat_sum_rotate_rewards = [0 for _ in self.object_type_list]  # rotate reward, with clipping
        self.stat_sum_rotate_penalty = [0 for _ in self.object_type_list]  # rotate penalty with clipping
        self.stat_sum_unclip_rotate_rewards = [0 for _ in self.object_type_list]  # rotate reward, with clipping
        self.stat_sum_unclip_rotate_penalty = [0 for _ in self.object_type_list]  # rotate penalty with clipping
        self.extrin_log = []
        self.env_evaluated = [0 for _ in self.object_type_list]
        self.evaluate_iter = 0

        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.total_rot_angle = torch.zeros(self.num_envs, device=self.device)  # 累计旋转角度
        self.current_angvel = torch.zeros(self.num_envs, device=self.device)  # 当前角速度（用于early termination）
        
        # ============================================================
        # [Anti-Hacking] EMA 速度平滑和 Jitter 惩罚相关状态变量
        # ============================================================
        self.angvel_ema = torch.zeros(self.num_envs, device=self.device)          # EMA 平滑后的角速度
        self.prev_spin_velocity = torch.zeros(self.num_envs, device=self.device)  # 上一帧的角速度（用于 jitter 惩罚）

        # 添加终止原因记录计数器
        self.termination_counts = {
            'max_episode_length': 0,
            'object_below_threshold': 0,
            'angular_velocity_too_high': 0,
            'pencil_tilt': 0,
            'total_episodes': 0
        }
        
        # 为每个环境记录当前episode的终止原因
        self.current_termination_reason = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # 0: 未终止, 1: max_episode_length, 2: object_below_threshold, 3: pencil_tilt, 4: angular_velocity_too_high
        
        # 初始化物体高度缓冲区，用于相对 relative_z_drop_threshold 计算
        self.init_object_z_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        # 为每个环境记录终止时的实际值（用于调试输出）
        self.termination_actual_values = torch.zeros(self.num_envs, device=self.device)
        
        # ============================================================
        # Waypoint 跟踪奖励初始化 (Triangle Pass)
        # ============================================================
        # 从 tp_waypoints.py 加载密集插值后的 waypoint 数据
        # 如果数据不足，waypoint 奖励将被禁用 (scale=0)
        self.waypoint_sigma = self.config['env']['reward'].get('waypoint_sigma', 0.05)
        self.axial_tilt_threshold = self.config['env']['reward'].get('axial_tilt_threshold', 0.03)
        self.waypoint_half_period_symmetric = self.config['env']['reward'].get('waypoint_half_period_symmetric', True)
        self.dense_waypoint_pos, self.dense_phases = initialize_waypoints(
            device=self.device,
            half_period_symmetric=self.waypoint_half_period_symmetric
        )
        self.waypoint_tracking_enabled = (self.dense_waypoint_pos is not None)
        
        # 创建 waypoint finger mask
        # 指尖顺序: [little(3), ring(3), middle(3), index(3), thumb(3)]
        # 如果禁用了无名指和小拇指，只比较中指、食指、大拇指
        if self.disable_ring_little:
            # 禁用 little 和 ring，只保留 middle, index, thumb (后9维)
            self.waypoint_finger_mask = torch.tensor(
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                dtype=torch.float32, device=self.device
            )
            waypoint_fingers_str = "middle, index, thumb (9维)"
        else:
            # 使用全部5指
            self.waypoint_finger_mask = torch.ones(15, dtype=torch.float32, device=self.device)
            waypoint_fingers_str = "全部5指 (15维)"
        
        if self.waypoint_tracking_enabled:
            print(f"[Waypoint Tracking] 已启用，sigma={self.waypoint_sigma}, 180°对称={self.waypoint_half_period_symmetric}")
            print(f"[Waypoint Tracking] 密集点数: {self.dense_waypoint_pos.shape[0]}, 比较手指: {waypoint_fingers_str}")
        else:
            print("[Waypoint Tracking] 未启用 (waypoint 数据不足)")
        
        # 当前相位缓冲区 (用于 debug 输出)
        self.current_phase_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        # ============================================================
        # [Config Check] 打印实际运行参数（消除硬编码打印）
        # ============================================================
        print("\n" + "="*80)
        print("[Config Check] 实际运行参数 (Python 内存值)")
        print("="*80)
        print(f"  > P-Gain (Base):         {self.base_pgain:.4f}")
        print(f"  > D-Gain (Base):         {self.base_dgain:.4f}")
        print(f"  > Torque Limit:          {self.torque_limit:.4f}")
        print(f"  > Action Scale:          {self.action_scale:.4f}")
        print(f"  > Alpha Start:           {self.time_warper.alpha_start:.4f}")
        print(f"  > Alpha End:             {self.time_warper.alpha_end:.4f}")
        print(f"  > Alpha Current:         {self.time_warper.current_alpha:.4f}")
        print(f"  > Relative Z Drop Thres: {self.relative_z_drop_threshold:.4f}")
        print(f"  > Pencil Tilt Thres:     {self.pencil_tilt_threshold:.4f}")
        print(f"  > Num Envs:              {self.num_envs}")
        print(f"  > Num Actions:           {self.num_actions}")
        print(f"  > Flying Hand:           {self.flying_hand_enabled}")
        print(f"  > Disable Ring/Little:   {self.disable_ring_little}")
        print("="*80 + "\n")

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._create_ground_plane()
        # envSpacing = 0.5，划出1m*1m立方体区域
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self._create_object_asset()
        linker_hand_dof_props = self._parse_hand_dof_props()
        hand_pose, obj_pose = self._init_object_pose()

        # compute aggregate size
        self.num_linker_hand_bodies = self.gym.get_asset_rigid_body_count(self.hand_asset)
        self.num_linker_hand_shapes = self.gym.get_asset_rigid_shape_count(self.hand_asset)
        max_agg_bodies = self.num_linker_hand_bodies + 2
        max_agg_shapes = self.num_linker_hand_shapes + 2

        self.envs = []
        self.vid_record_tensor = None  # Used for record video during training, NOT FOR POLICY OBSERVATION
        self.object_init_state = []

        self.hand_indices = []
        self.hand_actors = []
        self.object_indices = []
        self.object_type_at_env = []

        self.obj_point_clouds = []

        linker_hand_rb_count = self.gym.get_asset_rigid_body_count(self.hand_asset)
        object_rb_count = 1
        self.object_rb_handles = list(range(linker_hand_rb_count, linker_hand_rb_count + object_rb_count))

        # 打印显存相关的关键参数
        print("\n" + "="*80)
        print("环境初始化 - 显存相关参数")
        print("="*80)
        print(f"num_envs (环境数量):        {num_envs}")
        print(f"环境间距 (spacing):         {spacing}")
        print(f"每行环境数 (num_per_row):   {num_per_row}")
        print(f"手部刚体数量:               {self.num_linker_hand_bodies}")
        print(f"物体刚体数量:               {object_rb_count}")
        print(f"总刚体数/环境:              {self.num_linker_hand_bodies + object_rb_count}")
        print("="*80)
        print(f"开始创建 {num_envs} 个并行环境...")
        print("="*80 + "\n")

        for i in range(num_envs):
            # 只在特定进度点打印（0%, 25%, 50%, 75%, 100%）
            if i == 0 or i == num_envs - 1 or i % (num_envs // 4) == 0:
                progress = (i / num_envs) * 100
                tprint(f'环境创建进度: {i}/{num_envs} ({progress:.1f}%)')
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies * 20, max_agg_shapes * 20, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            hand_actor = self.gym.create_actor(env_ptr, self.hand_asset, hand_pose, 'hand', i, 0, 1)
            self.gym.set_actor_dof_properties(env_ptr, hand_actor, linker_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)
            self.hand_actors.append(hand_actor)

            # ============ 碰撞过滤器设置 ============
            # 获取手部 actor 的所有刚体形状属性
            hand_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, hand_actor)
            
            # 获取刚体名称和形状索引映射
            hand_body_names = self.gym.get_asset_rigid_body_names(self.hand_asset)
            body_shape_indices = self.gym.get_asset_rigid_body_shape_indices(self.hand_asset)
            
            for body_idx, body_name in enumerate(hand_body_names):
                start = body_shape_indices[body_idx].start
                count = body_shape_indices[body_idx].count
            for body_idx, body_name in enumerate(hand_body_names):
                start = body_shape_indices[body_idx].start
                count = body_shape_indices[body_idx].count
                
                if count > 0:
                # 【特殊处理】：如果你真的非常想让 thumb_joint0 不撞任何东西
                    if "thumb_joint0" in body_name:
                        # 设置一个非常特殊的 filter ID (比如 128)
                        # 在 Isaac Gym 默认机制下，如果其他物体 filter 是 0
                        # 0 会撞一切，所以单纯改 filter 不足以完全屏蔽。
                        # 但通常我们只需要它不撞特定的手指。
                        
                        # 更好的方法是使用 contact_offset 欺骗物理引擎
                        for k in range(start, start + count):
                            # 将接触偏移设为极小负数，使其极难触发碰撞接触生成
                            # 意思是：只有当物体陷入它内部 1米 深时才算碰撞
                            # 实际上这让它变成了“幽灵”
                            hand_shape_props[k].contact_offset = -1.0 
                            hand_shape_props[k].rest_offset = -1.0
                    else:
                        # 其他部件正常设置
                        for k in range(start, start + count):
                            hand_shape_props[k].filter = 0 # 设为 0 以解决穿模
                            hand_shape_props[k].restitution = 0.0
                            hand_shape_props[k].friction = 0.8
                            # 确保正常的 contact_offset
                            hand_shape_props[k].contact_offset = 0.002                

            # 将修改后的属性应用回去
            self.gym.set_actor_rigid_shape_properties(env_ptr, hand_actor, hand_shape_props)
            # ========================================

            # add object
            eval_object_type = self.config['env']['object']['evalObjectType']
            if eval_object_type is None:
                object_type_id = np.random.choice(len(self.object_type_list), p=self.object_type_prob)
            else:
                object_type_id = self.object_type_list.index(eval_object_type)

            self.object_type_at_env.append(object_type_id)
            object_asset = self.object_asset_list[object_type_id]

            object_handle = self.gym.create_actor(env_ptr, object_asset, obj_pose, 'object', i, 0, 2)

            # 这里的object init state作为物体的初始状态是无关紧要的，因为
            # 后续会通过self.root_state_tensor[self.object_indices[s_ids]]进行姿态赋值
            self.object_init_state.append([
                obj_pose.p.x, obj_pose.p.y, obj_pose.p.z,
                obj_pose.r.x, obj_pose.r.y, obj_pose.r.z, obj_pose.r.w,
                0, 0, 0, 0, 0, 0
            ])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            
            # LinkerPen 使用实际尺寸，不需要缩放
            self.obj_scale = 1.0
            self.gym.set_actor_scale(env_ptr, object_handle, self.obj_scale)
            # 注：obj_scale 已从 priv_info 中移除（固定为 1.0）

            obj_com = [0, 0, 0]
            # COM是物体的质心 - 对于 LinkerPen，质心已在 URDF 中精确定义
            # 这里仍保留微小扰动以增加鲁棒性
            if self.randomize_com:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                # LinkerPen 有多个 link，只修改第一个（主体）
                obj_com = [np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                           np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                           np.random.uniform(self.randomize_com_lower, self.randomize_com_upper)]
                prop[0].com.x, prop[0].com.y, prop[0].com.z = obj_com
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
            self._update_priv_buf(env_id=i, name='obj_com', value=obj_com)

            obj_friction = 1.0
            obj_restitution = 0.0  # default is 0
            
            # 摩擦系数随机化
            if self.randomize_friction:
                rand_friction = np.random.uniform(self.randomize_friction_lower, self.randomize_friction_upper)
                obj_friction = rand_friction
            
            # 恢复系数随机化
            if self.randomize_restitution:
                obj_restitution = np.random.uniform(self.randomize_restitution_lower, self.randomize_restitution_upper)
            
            # 应用摩擦和恢复系数到手和物体
            if self.randomize_friction or self.randomize_restitution:
                hand_props = self.gym.get_actor_rigid_shape_properties(env_ptr, hand_actor)
                for p in hand_props:
                    p.friction = obj_friction
                    p.restitution = obj_restitution
                self.gym.set_actor_rigid_shape_properties(env_ptr, hand_actor, hand_props)

                object_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
                for p in object_props:
                    p.friction = obj_friction
                    p.restitution = obj_restitution
                self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_props)
            
            self._update_priv_buf(env_id=i, name='obj_friction', value=obj_friction)
            self._update_priv_buf(env_id=i, name='obj_restitution', value=obj_restitution)

            # 质量处理
            # URDF 中精确定义了 LinkerPen 质量（约 40.9g）
            # 如果启用质量随机化，则在 URDF 基础上应用均匀扰动
            prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
            urdf_total_mass = sum([p.mass for p in prop])
            
            if self.randomize_mass:
                # 随机生成目标质量
                target_mass = np.random.uniform(self.randomize_mass_lower, self.randomize_mass_upper)
                # 计算缩放因子并应用到所有刚体
                mass_scale = target_mass / urdf_total_mass if urdf_total_mass > 0 else 1.0
                for p in prop:
                    p.mass *= mass_scale
                    # 同时缩放惯量张量
                    p.inertia.x *= mass_scale
                    p.inertia.y *= mass_scale
                    p.inertia.z *= mass_scale
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
                total_mass = target_mass
            else:
                total_mass = urdf_total_mass
            
            self._update_priv_buf(env_id=i, name='obj_mass', value=total_mass)
            
            # 在第一个环境中打印物体的完整物理属性（随机化后的最终值）
            if i == 0:
                print("\n" + "="*70)
                print("物体物理属性（随机化后的最终值）")
                print("="*70)
                print(f"随机化配置:")
                print(f"  - 质量随机化: {self.randomize_mass} [{self.randomize_mass_lower}, {self.randomize_mass_upper}]")
                print(f"  - 质心随机化: {self.randomize_com} [{self.randomize_com_lower}, {self.randomize_com_upper}]")
                print(f"  - 摩擦随机化: {self.randomize_friction} [{self.randomize_friction_lower}, {self.randomize_friction_upper}]")
                print(f"  - 恢复系数随机化: {self.randomize_restitution} [{self.randomize_restitution_lower}, {self.randomize_restitution_upper}]")
                print(f"本环境实际值:")
                print(f"  - 总质量: {total_mass:.4f} kg ({total_mass*1000:.2f} g)")
                print(f"  - 摩擦系数: {obj_friction:.4f}")
                print(f"  - 恢复系数: {obj_restitution:.4f}")
                print(f"  - 质心偏移: {obj_com}")
                self._print_object_properties(env_ptr, object_handle)

            if self.point_cloud_sampled_dim > 0:
                # 不再乘以 obj_scale，因为已经是 1.0
                self.obj_point_clouds.append(self.asset_point_clouds[object_type_id])

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # for training record, visualized in tensorboard
            # FIX: Only create camera for the first environment to prevent freeze/OOM
            if self.with_camera and i == 0:
                self.vid_record_tensor = self._create_camera(env_ptr)

            self.envs.append(env_ptr)

        sensor_handles = [self.gym.find_actor_rigid_body_handle(
            env_ptr, hand_actor, sensor_name
        ) for sensor_name in self.contact_sensor_names]
        self.sensor_handle_indices = to_torch(sensor_handles, dtype=torch.int64)
        self.obj_point_clouds = to_torch(np.array(self.obj_point_clouds), device=self.device, dtype=torch.float)
        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, RIGID_BODY_STATES)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.object_type_at_env = to_torch(self.object_type_at_env, dtype=torch.long, device=self.device)

    def _create_camera(self, env_ptr) -> torch.Tensor:
        """Create a camera in a particular environment. Should be called in _create_envs."""
        camera_props = gymapi.CameraProperties()
        camera_props.width = 256
        camera_props.height = 256
        camera_props.enable_tensors = True
        
        camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)

        cam_pos = gymapi.Vec3(0.0, 0.2, 0.75)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.5)

        self.gym.set_camera_location(camera_handle, env_ptr, cam_pos, cam_target)
        # obtain camera tensor
        vid_record_tensor = self.gym.get_camera_image_gpu_tensor(
            self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR
        )
        # wrap camera tensor in a pytorch tensor
        vid_record_tensor.device = 0
        torch_vid_record_tensor = gymtorch.wrap_tensor(vid_record_tensor)
        assert torch_vid_record_tensor.shape == (camera_props.height, camera_props.width, 4)

        return torch_vid_record_tensor

    def apply_curriculum_physics(self):
        """
        将当前的时空扭曲因子 α 应用到 Physics Engine。
        
        此函数执行以下物理参数更新：
        1. 全局重力 (Global Gravity): g' = α² × g
        2. 手部 PD 刚度 (Stiffness/Kp): Kp' = α² × Kp
        3. 手部 PD 阻尼 (Damping/Kd): Kd' = α × Kd
        4. 力矩限制 (Effort Limit): τ' = α² × τ
        5. Flying Base PD 参数（当前直接取消flying base）
        
        注意：
        - 此函数开销较大（需要暂停物理线程重写属性）
        - 只应在 alpha 发生显著变化时调用（由 PPO 训练循环控制）
        - 不应在 step() 中调用
        
        关于 Flying Hand 的处理：
        - Flying Base 的 Kp/Kd 必须参与缩放
        - 否则在 Space E (α=0.1) 中，物体变轻 100 倍
        - 但 Flying Base 相对物体是"无限硬"的，会导致动量守恒失配
        """
        alpha = self.time_warper.current_alpha
        
        # 如果是 SpaceA 模式 (α=1.0)，跳过更新（保持原始参数）
        if self.time_warper.mode == 'SpaceA' and alpha == 1.0:
            return
        
        print(f"\n[Space E Curriculum] 更新物理参数 Alpha = {alpha:.3f} "
              f"(Progress: {self.time_warper.get_progress()*100:.1f}%)")
        
        # 获取缩放系数
        g_scale = self.time_warper.gravity_scale
        kp_scale = self.time_warper.stiffness_scale
        kd_scale = self.time_warper.damping_scale
        eff_scale = self.time_warper.effort_scale
        
        # ============================================================
        # 1. 更新全局重力 (Global Gravity)
        # ============================================================
        # 获取当前的 sim params
        sim_params = self.gym.get_sim_params(self.sim)
        
        # 计算新的重力向量
        new_gravity = gymapi.Vec3(
            self.base_gravity[0] * g_scale,
            self.base_gravity[1] * g_scale,
            self.base_gravity[2] * g_scale
        )
        sim_params.gravity = new_gravity
        
        # 应用到 Sim
        # 注意: set_sim_params 在 PhysX backend 上通常支持 gravity 实时变更
        self.gym.set_sim_params(self.sim, sim_params)
        print(f"  -> 重力: {new_gravity.z:.4f} m/s² (原始: {self.base_gravity[2]:.2f}, 缩放: {g_scale:.4f})")
        
        # ============================================================
        # 2. 更新 Actor DOF 属性 (Stiffness, Damping, Effort)
        # ============================================================
        # 注意: 我们必须基于"原始"属性缩放，而不是"当前"属性
        # 这样可以防止累积误差
        
        for i in range(self.num_envs):
            hand_actor = self.hand_actors[i]
            dof_props = self.gym.get_actor_dof_properties(self.envs[i], hand_actor)
            
            # 遍历所有 DOF
            for j in range(self.num_linker_hand_dofs):
                # 判断是 Flying Base 还是 Hand Joint
                is_flying_joint = (self.flying_hand_enabled and j < NUM_FLYING_DOF)
                
                # --- Stiffness (Kp) ---
                if is_flying_joint:
                    base_kp = self.base_flying_pgain
                elif self.torque_control:
                    base_kp = 0.0  # 力控模式 Kp=0
                else:
                    base_kp = self.base_pgain
                
                dof_props['stiffness'][j] = base_kp * kp_scale
                
                # --- Damping (Kd) ---
                if is_flying_joint:
                    base_kd = self.base_flying_dgain
                elif self.torque_control:
                    base_kd = 0.0
                else:
                    base_kd = self.base_dgain
                
                dof_props['damping'][j] = base_kd * kd_scale
                
                # --- Effort Limit ---
                if is_flying_joint:
                    base_eff = 100000.0  # Flying base 使用极大力矩
                else:
                    base_eff = self.base_torque_limit
                
                dof_props['effort'][j] = base_eff * eff_scale
            
            # 应用属性到该环境的 Actor
            self.gym.set_actor_dof_properties(self.envs[i], hand_actor, dof_props)
        
        # ============================================================
        # 3. 更新类内部用于 PD 计算的 Tensor
        # ============================================================
        # 在 update_low_level_control 中会用到 self.p_gain 和 self.d_gain
        if self.torque_control:
            # 手部 PD（力矩模式下通常 Kp=0，但也可能非0）
            self.p_gain[:] = self.base_pgain * kp_scale
            self.d_gain[:] = self.base_dgain * kd_scale
            
            # Flying Base 的 PD（始终是位置控制）
            if self.flying_hand_enabled:
                self.p_gain[:, :NUM_FLYING_DOF] = self.base_flying_pgain * kp_scale
                self.d_gain[:, :NUM_FLYING_DOF] = self.base_flying_dgain * kd_scale
            
            # 更新力矩限制
            self.torque_limit = self.base_torque_limit * eff_scale
        
        print(f"  -> DOF 属性已更新: Kp_scale={kp_scale:.4f}, Kd_scale={kd_scale:.4f}, τ_scale={eff_scale:.4f}")
        print(f"  -> 手部 Kp: {self.base_pgain * kp_scale:.4f}, Kd: {self.base_dgain * kd_scale:.4f}")
        if self.flying_hand_enabled:
            print(f"  -> Flying Base Kp: {self.base_flying_pgain * kp_scale:.2f}, Kd: {self.base_flying_dgain * kd_scale:.2f}")
        
        # ============================================================
        # 4. 更新 PhysX 求解器阈值 (通过 time_warper 统一管理)
        # ============================================================
        # 获取当前 sim_params (包含之前设置的重力)
        sim_params = self.gym.get_sim_params(self.sim)
        
        # 从 time_warper 获取缩放后的 PhysX 参数
        physx_params = self.time_warper.get_scaled_physx_params(
            self.base_bounce_threshold,
            self.base_contact_offset
        )
        
        # 应用 bounce_threshold_velocity
        sim_params.physx.bounce_threshold_velocity = physx_params['bounce_threshold_velocity']
        
        # 可以考虑仅在低 alpha 时更新 contact_offset，当前逻辑是始终更新
        # if physx_params['needs_contact_offset_update']:
        sim_params.physx.contact_offset = physx_params['contact_offset']
        print(f"  -> PhysX contact_offset: {physx_params['contact_offset']:.5f} (当前逻辑是始终缩放)")
        
        # 应用更新后的 sim_params
        self.gym.set_sim_params(self.sim, sim_params)
        print(f"  -> PhysX bounce_threshold: {physx_params['bounce_threshold_velocity']:.4f} (原始: {self.base_bounce_threshold:.2f})")

    def reset_idx(self, env_ids):
        if self.randomize_pd_gains:
            # ================= [Fix Start] =================
            # 获取当前的缩放系数
            kp_scale = self.time_warper.stiffness_scale  # alpha^2
            kd_scale = self.time_warper.damping_scale    # alpha
            
            # 对随机化范围进行缩放
            p_lower = self.randomize_p_gain_lower * kp_scale
            p_upper = self.randomize_p_gain_upper * kp_scale
            d_lower = self.randomize_d_gain_lower * kd_scale
            d_upper = self.randomize_d_gain_upper * kd_scale
            
            # 使用缩放后的范围进行随机化
            self.p_gain[env_ids] = torch_rand_float(
                p_lower, p_upper, (len(env_ids), self.num_dofs),
                device=self.device).squeeze(1)
            self.d_gain[env_ids] = torch_rand_float(
                d_lower, d_upper, (len(env_ids), self.num_dofs),
                device=self.device).squeeze(1)
            # ================= [Fix End] =================
        # 这两个噪声变量 random_obs_noise_e 和 random_action_noise_e 的物理量纲是 位置（Position） 或 角度（Rotation, radians）
        # 几何类的噪声不需要随 Alpha 缩放
        self.random_obs_noise_e[env_ids] = torch.normal(0, self.random_obs_noise_e_scale, size=(len(env_ids), self.num_dofs), device=self.device, dtype=torch.float)
        self.random_action_noise_e[env_ids] = torch.normal(0, self.random_action_noise_e_scale, size=(len(env_ids), self.num_dofs), device=self.device, dtype=torch.float)
        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        # ================================================================
        # 从缓存中采样初始姿态
        # ================================================================
        sampled_pose_idx = np.random.randint(self.saved_grasping_states.shape[0], size=len(env_ids))
        sampled_pose = self.saved_grasping_states[sampled_pose_idx].clone()
        
        # ================================================================
        # 解析缓存数据（支持新旧两种格式）
        # ================================================================
        # 新格式 (61维): [hand_actual(27) + hand_target(27) + obj_pos(3) + obj_rot(4)]
        #   - hand_actual: 仿真稳定后的实际位置（用于物理状态，避免穿模）
        #   - hand_target: 原始目标位置（用于PD控制，产生抓握力）
        # 旧格式 (34维): [hand_dof(27) + obj_pos(3) + obj_rot(4)]
        #   - 物理位置 = 控制目标（零力矩陷阱）
        # ================================================================
        if self.cache_is_new_format:
            # 新格式: 分别提取 actual 和 target
            hand_actual = sampled_pose[:, :self.num_linker_hand_dofs]
            hand_target = sampled_pose[:, self.num_linker_hand_dofs:self.num_linker_hand_dofs*2]
            obj_pose_start = self.num_linker_hand_dofs * 2
        else:
            # 旧格式: actual = target（向后兼容）
            hand_actual = sampled_pose[:, :self.num_linker_hand_dofs]
            hand_target = hand_actual.clone()
            obj_pose_start = self.num_linker_hand_dofs
        
        # 物体位姿（添加位置噪声）
        object_pose_noise = torch.normal(0, self.random_pose_noise, size=(sampled_pose.shape[0], 7), device=self.device, dtype=torch.float)
        object_pose_noise[:, 3:] = 0  # disable rotation noise
        self.root_state_tensor[self.object_indices[env_ids], :7] = sampled_pose[:, obj_pose_start:obj_pose_start+7] + object_pose_noise
        self.root_state_tensor[self.object_indices[env_ids], 7:RIGID_BODY_STATES] = 0
        
        # ================================================================
        # 分别设置物理位置和控制目标
        # ================================================================
        # 物理位置: 使用 actual（避免穿模爆炸）
        self.linker_hand_dof_pos[env_ids, :] = hand_actual
        self.linker_hand_dof_vel[env_ids, :] = 0
        
        # 控制目标: 使用 target（产生抓握力）
        # PD 误差 = target - actual ≠ 0，产生持续夹紧力矩
        self.prev_targets[env_ids, :self.num_linker_hand_dofs] = hand_target
        self.cur_targets[env_ids, :self.num_linker_hand_dofs] = hand_target
        self.init_pose_buf[env_ids, :] = hand_target  # 用 target 作为初始参考
        
        # ============================================================
        # 更新每个环境的相对限位
        # ============================================================
        if self.use_relative_limit:
            # 计算相对限位: init_pose ± relative_range (基于 target)
            relative_lower = hand_target - self.dof_relative_range.unsqueeze(0)
            relative_upper = hand_target + self.dof_relative_range.unsqueeze(0)
            # 与绝对限位取交集
            self.env_dof_lower_limits[env_ids] = torch.max(relative_lower, self.linker_hand_dof_lower_limits.unsqueeze(0))
            self.env_dof_upper_limits[env_ids] = torch.min(relative_upper, self.linker_hand_dof_upper_limits.unsqueeze(0))

        object_indices = torch.unique(self.object_indices[env_ids]).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(object_indices), len(object_indices))
        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        
        # ============================================================
        # Flying Hand 混合控制模式下的 Reset：
        # - Flying base 使用 DOF_MODE_POS，需要设置位置目标
        # - 手部使用 DOF_MODE_EFFORT，由力矩控制（reset时不需要额外操作）
        # ============================================================
        # 无论是否 torque_control，Flying base 都需要设置位置目标
        # 因为它始终是 DOF_MODE_POS 模式
        if self.flying_hand_enabled:
            self.gym.set_dof_position_target_tensor_indexed(
                self.sim, 
                gymtorch.unwrap_tensor(self.prev_targets), 
                gymtorch.unwrap_tensor(hand_indices), 
                len(env_ids)
            )
        elif not self.torque_control:
            # 非 Flying Hand 且非力矩控制模式
            self.gym.set_dof_position_target_tensor_indexed(
                self.sim, 
                gymtorch.unwrap_tensor(self.prev_targets), 
                gymtorch.unwrap_tensor(hand_indices), 
                len(env_ids)
            )
        
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        # 记录初始物体高度用于相对 relative_z_drop_threshold
        self.init_object_z_buf[env_ids] = self.root_state_tensor[self.object_indices[env_ids], 2].clone()

        # reset tactile
        self.last_contacts[env_ids] = 0.0

        self.progress_buf[env_ids] = 0
        self.obs_buf[env_ids] = 0
        self.rb_forces[env_ids] = 0
        self.priv_info_buf[env_ids, 0:3] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.tactile_hist_buf[env_ids] = 0
        self.noisy_quaternion_buf[env_ids] = 0
        self.dof_vel_finite_diff[:] = 0
        self.at_reset_buf[env_ids] = 1

    def compute_observations(self):
        """
        计算当前时间步的环境观测（Observations）。
        这个方法是RL环境中每个时间步的核心部分，用于生成Agent的输入。
        """
        self._refresh_gym()
        # 从仿真器刷新最新的物理状态，确保后续读取的数据是最新的
        # 具体实现依赖于所使用的仿真后端（Isaac Gym）

        # ==============================================================
        # Space E: 观测逆缩放 (Observation Inverse Scaling)
        # ==============================================================
        # 目的: 使 Agent 感知到"真实世界尺度"的速度和力
        # 原理: 仿真中 v_sim = α * v_real, F_sim = α² * F_real
        #       逆缩放: v_obs = v_sim / α, F_obs = F_sim / α²
        # 这样 Agent 总是感知到 α=1 时的真实物理量级
        v_scale = self.time_warper.obs_velocity_scale  # = 1/α
        f_scale = self.time_warper.obs_force_scale     # = 1/α²
        
        # 对速度类变量应用逆缩放 (用于后续特权信息)
        # 注意：这些是 view/reference，缩放会影响原始数据
        # 我们创建缩放后的副本用于观测，不修改原始仿真数据
        self.object_linvel_scaled = self.object_linvel * v_scale
        self.object_angvel_scaled = self.object_angvel * v_scale
        self.fingertip_linvel_scaled = self.fingertip_linvel * v_scale
        self.fingertip_angvel_scaled = self.fingertip_angvel * v_scale
        
        # 对力类变量应用逆缩放 (用于触觉处理)
        # contact_forces_scaled 将在触觉处理中使用
        self.contact_forces_scaled = self.contact_forces * f_scale

        # --------------------------------------------------------------
        # 1. 关节位置观测噪声 (Joint Position Observation Noise)
        # --------------------------------------------------------------
        # 生成服从均值0、标准差self.random_obs_noise_t_scale 的高斯噪声
        # 形状与关节位置的shape、device 和 dtype 与张量self.linker_hand_dof_pos相同
        random_obs_noise_t = torch.normal(0, self.random_obs_noise_t_scale, size=self.linker_hand_dof_pos.shape, device=self.device, dtype=torch.float)
        # 将高斯噪声 random_obs_noise_t 和另一个常数噪声 random_obs_noise_e 加到当前的关节位置上
        # random_obs_noise_e 可能是用于模拟传感器偏置 (bias) 的常数噪声
        noisy_joint_pos = random_obs_noise_t + self.random_obs_noise_e + self.linker_hand_dof_pos
        # noisy_joint_pos shape: torch.Size([1, num_linker_hand_dofs])

        # --------------------------------------------------------------
        # --------------------------------------------------------------
        # 触觉传感处理 (Tactile Sensing Processing) - 基于分量阈值筛选版
        # 基于每个力分量和对应的阈值进行筛选，输出缩放后的展平三维力向量
        # --------------------------------------------------------------
        # 注意: 使用统一的 CONTACT_THRESH 常量进行阈值过滤

        if self.config['env']['privInfo']['enable_tactile']:
            # 获取并克隆当前的总接触力张量，形状 (num_envs, num_bodies, 3)
            # Space E: 使用逆缩放后的力 (F_obs = F_sim / α²)
            contacts = self.contact_forces_scaled.clone() # (N, num_bodies, 3)
            
            # 2. 提取传感器数据，保持 (N, M, 3) 形状，不要急着 flatten
            contacts_at_sensors = contacts[:, self.sensor_handle_indices, :] # (N, M, 3)
            
            # ================= [New Logic Start] =================
            
            # 3. 计算力的模长 (Magnitude)
            # shape: (N, M, 1)
            contact_magnitudes = torch.norm(contacts_at_sensors, dim=-1, keepdim=True)
            
            # 4. 基于模长的阈值过滤 (Magnitude Thresholding)
            # 修复了分量过滤导致的斜向力丢失问题
            # 使用统一的 CONTACT_THRESH 常量
            mask = (contact_magnitudes >= CONTACT_THRESH) # (N, M, 1)
            
            # 应用掩码：小于阈值的力全零，大于阈值的保留原始向量方向和大小
            filtered_contacts = contacts_at_sensors * mask.float()
            
            # 5. 对数缩放 (Logarithmic Scaling)
            # log_scale_beta 用于控制灵敏度。
            # beta=1.0 时: 0.1N -> 0.095, 1.0N -> 0.69, 4.0N -> 1.6
            # beta=10.0 时: 0.1N -> 0.69 (极度敏感), 4.0N -> 3.7
            # 推荐使用 beta=1.0 或 2.0，既保持小力线性，又压缩大力
            log_scale_beta = self.config['env']['reward']['log_scale_beta']
            
            # 变换公式: sign(x) * log(1 + beta * |x|)
            # 这种变换保留了正负号（方向），同时压缩了幅度
            scaled_contacts = torch.sign(filtered_contacts) * torch.log1p(log_scale_beta * torch.abs(filtered_contacts))
            
            # 6. 展平 (Flatten)
            # 最终输出形状 (N, M*3)
            self.sensed_contacts = torch.flatten(scaled_contacts, start_dim=1)
            # 如果启用了可视化，复制感知数据到 CPU 用于调试显示
            if self.viewer:
                # debug_contacts 的形状是 (num_envs, num_sensor_handles, 3)
                self.debug_contacts = filtered_contacts.detach().cpu().numpy() 
       
        # --------------------------------------------------------------
        # 3. 物体端点跟踪 (Object End Points Tracking)
        # --------------------------------------------------------------
        
        # 定义物体的本地坐标系下的端点位置（LinkerPen 的两端）
        # pen_length = 0.18m，无需缩放
        pencil_ends = [
            [0, 0, -self.pen_length / 2],  # -0.09m
            [0, 0, self.pen_length / 2]    # +0.09m
        ]
        # 端点相对于机械臂根部的位置
        pencil_end_1 = self.object_pos + quat_apply(
            self.object_rot, to_torch(pencil_ends[0], device=self.device)[None].repeat(self.num_envs, 1)
        ) - self.root_state_tensor[self.hand_indices, :3]
        # 端点相对于机械臂根部的位置
        pencil_end_2 = self.object_pos + quat_apply(
            self.object_rot, to_torch(pencil_ends[1], device=self.device)[None].repeat(self.num_envs, 1)
        ) - self.root_state_tensor[self.hand_indices, :3]
        # 对计算出的物体端点位置添加均匀分布噪声，噪声范围与 pen_radius * 2 相关
        # (torch.rand(...) - 0.5) * (self.pen_radius*2) 生成范围在 [-pen_radius, pen_radius] 的均匀噪声
        pencil_end_1 += (torch.rand(pencil_end_1.shape[0], 3).to(self.device) - 0.5) * (self.pen_radius * 2)
        pencil_end_2 += (torch.rand(pencil_end_2.shape[0], 3).to(self.device) - 0.5) * (self.pen_radius * 2)

        # 将两个端点位置拼接起来，形状为 (num_envs, 6)
        # unsqueeze(1) 添加一个维度，形状变为 (num_envs, 1, 6)，便于后续拼接到历史缓冲区
        cur_obj_ends = torch.cat([pencil_end_1, pencil_end_2], dim=-1).unsqueeze(1)
        # 将当前时间步的物体端点信息添加到历史缓冲区的末尾，同时丢弃最旧的一个时间步
        prev_obj_ends = self.obj_ends_history[:, 1:].clone()
        self.obj_ends_history[:] = torch.cat([prev_obj_ends, cur_obj_ends], dim=1)
        # obj_ends_history shape:(num_envs, history_len_obj_ends=3, 6)
        
        # --------------------------------------------------------------
        # 4. 更新主观测缓冲区 (Main Observation Buffer Update)
        # --------------------------------------------------------------
        # 提取 obs_buf_lag_history 中最近的三个时间步的obs历史 (前 obs_buf.shape[1]//3=42 部分)
        # 形状变化：(num_envs, history_len, obs_dim) -> (num_envs, 3, obs_dim_per_step//3) -> (num_envs, 3 * obs_dim_per_step//3)

        t_buf = (self.obs_buf_lag_history[:, -3:, :self.actual_proprio_dim].reshape(self.num_envs, -1)).clone()
                # t_buf shape: torch.Size([1, 3*actual_proprio_dim])

        # 将提取的历史部分赋值给主观测缓冲区的开头部分
        self.obs_buf[:, :t_buf.shape[1]] = t_buf  # [1, 126]
        # obs_buf只要了obs_buf_lag_history最后一维度的前42个，也就是只有joint pos和target。一共往前要了三个时间步的obs

        # 处理正常的滑动窗口更新
        # 复制 obs_buf_lag_history 中除了最旧时间步之外的所有数据
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
                # prev_obs_buf shape: torch.Size([1, 79, 21+21+15+15]),obs_buf只要了最后一维度的前42个，也就是只有joint pos和target

        # 获取当前时间步的带噪声关节位置，并添加一个维度，形状变为 (num_envs, 1, self.num_linker_hand_dofs)
        cur_obs_buf = noisy_joint_pos.clone().unsqueeze(1)  # [1, 1, self.num_linker_hand_dofs]
        # 获取当前时间步的目标关节位置，并添加一个维度，形状变为 (num_envs, 1, num_linker_hand_dofs)
        cur_tar_buf = self.cur_targets[:, None]  # [1, 1, self.num_linker_hand_dofs]
        # 将当前关节位置和目标位置拼接，形状变为 (num_envs, 1, PROPRIO_DIM)
        cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)  # [1, 1, PROPRIO_DIM]
        
        # 如果启用了触觉，则将感知到的触觉信息拼接到当前观测中
        if self.config['env']['privInfo']['enable_tactile']:
            # 拼接 sensed_contacts (形状 num_envs, CONTACT_DIM) -> unsqueeze(1) -> (num_envs, 1, CONTACT_DIM)
            # 这里拼接的是sensor_contact的维度
            cur_obs_buf = torch.cat([cur_obs_buf, self.sensed_contacts.unsqueeze(1)], dim=-1) # [1, 1, PROPRIO_DIM+CONTACT_DIM]
            # 拼接指尖位置 fingertip_pos (形状 num_envs, FINGERTIP_POS_DIM) -> unsqueeze(1) -> (num_envs, 1, FINGERTIP_POS_DIM)
            cur_obs_buf = torch.cat([cur_obs_buf, self.fingertip_pos.clone().unsqueeze(1)], dim=-1) # [1, 1, PROPRIO_DIM+CONTACT_DIM+FINGERTIP_POS_DIM]
          
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)
        # obs_buf_lag_history shape: torch.Size([1, 80, PROPRIO_DIM+CONTACT_DIM+FINGERTIP_POS_DIM])

        # --------------------------------------------------------------
        # 5. 环境重置时特殊处理 (Reset Handling)
        # --------------------------------------------------------------
        # 找到所有刚刚发生重置的环境实例的索引
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # 对于重置的环境，用初始姿态 self.init_pose_buf 填充其观测历史缓冲区的本体感知部分
        # 这确保重置后的历史观测从初始状态开始
        # 注意：初始目标位置也用初始姿态填充，这可能是为了在episode开始时agent目标就是保持初始姿态
        self.obs_buf_lag_history[at_reset_env_ids, :, 0:self.num_linker_hand_dofs] = self.init_pose_buf[at_reset_env_ids].unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, self.num_linker_hand_dofs:self.actual_proprio_dim] = self.init_pose_buf[at_reset_env_ids].unsqueeze(1)
        # 对于重置的环境，用当前的物体端点位置填充其历史缓冲区
        self.obj_ends_history[at_reset_env_ids, :, :] = cur_obj_ends[at_reset_env_ids]

        # 如果启用了触觉，则对于重置的环境，将其观测历史缓冲区中触觉相关部分清零
        # 范围是 actual_proprio_dim 到 actual_proprio_dim+CONTACT_DIM
        if self.config['env']['privInfo']['enable_tactile']:
            self.obs_buf_lag_history[at_reset_env_ids, :, self.actual_proprio_dim:self.actual_proprio_dim+CONTACT_DIM] = torch.zeros((len(at_reset_env_ids),80,CONTACT_DIM),device=self.device)
            # 对于重置的环境，用当前的指尖位置填充其观测历史缓冲区中指尖位置部分
            # 范围是 actual_proprio_dim+CONTACT_DIM 到 actual_proprio_dim+CONTACT_DIM+FINGERTIP_POS_DIM
            self.obs_buf_lag_history[at_reset_env_ids, :, self.actual_proprio_dim+CONTACT_DIM:self.actual_proprio_dim+CONTACT_DIM+FINGERTIP_POS_DIM] = self.fingertip_pos[at_reset_env_ids].unsqueeze(1)

        # 重置相关的速度信息缓冲
        # 在接触或碰撞时记录的物体和指尖线速度/角速度，在重置时也需要清零或用当前值填充
        # Space E: 使用逆缩放后的速度 (v_obs = v_sim / α)
        self.obj_linvel_at_cf[at_reset_env_ids] = self.object_linvel_scaled[at_reset_env_ids]
        self.obj_angvel_at_cf[at_reset_env_ids] = self.object_angvel_scaled[at_reset_env_ids]
        self.ft_linvel_at_cf[at_reset_env_ids] = self.fingertip_linvel_scaled[at_reset_env_ids]
        self.ft_angvel_at_cf[at_reset_env_ids] = self.fingertip_angvel_scaled[at_reset_env_ids]

        # 将重置标志位 at_reset_buf 设置为 0，表示这些环境已经完成重置处理
        self.at_reset_buf[at_reset_env_ids] = 0
        # 为物体的初始姿态添加观测噪声 (roll, pitch, yaw 欧拉角噪声)
        rand_rpy = torch.normal(0, self.noisy_rpy_scale, size=(self.num_envs, 3), device=self.device, dtype=torch.float)
        # 将欧拉角噪声转换为四元数
        rand_quat = quat_from_euler_xyz(rand_rpy[:, 0], rand_rpy[:, 1], rand_rpy[:, 2])
        # 将噪声四元数与物体真实姿态相乘，得到带噪声的姿态观测
        noisy_quat = quat_mul(rand_quat, self.object_rot)
        # 为物体初始位置添加高斯噪声
        noisy_position = torch.normal(0, self.noisy_pos_scale, size=(self.num_envs, 3), device=self.device, dtype=torch.float) + self.object_pos
        # 将带噪声的物体姿态和位置存储到历史缓冲区，仅针对重置的环境实例
        self.noisy_quaternion_buf[at_reset_env_ids, :, :4] = noisy_quat[at_reset_env_ids].unsqueeze(1)
        self.noisy_quaternion_buf[at_reset_env_ids, :, 4:] = noisy_position[at_reset_env_ids].unsqueeze(1)
        # 更新带噪声的物体姿态和位置历史缓冲区，将当前时间步的数据添加到末尾，移除最旧的数据
        self.noisy_quaternion_buf[:] = torch.cat([
            self.noisy_quaternion_buf[:, 1:].clone(), # 移除最旧时间步
            torch.cat([noisy_quat.unsqueeze(1), noisy_position.unsqueeze(1)], dim=-1) # 添加当前时间步带噪声的数据
        ], dim=1)
        # noisy_quaternion_buf shape: torch.Size([1, prop_hist_len=30, pos+rot=7])


        # --------------------------------------------------------------
        # 6. 提取特定历史缓冲区 (Extract Specific History Buffers)
        # --------------------------------------------------------------
        # 从主观测历史 obs_buf_lag_history 中提取最近 prop_hist_len 个时间步的本体感知信息 (关节位置和目标)
        # 范围是 0 到 actual_proprio_dim，因为只使用关节位置和目标位置
        self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.prop_hist_len:, :self.actual_proprio_dim]  # [1, 30, actual_proprio_dim] - 示例形状
        # 如果启用了触觉，从主观测历史中提取最近 prop_hist_len 个时间步的触觉信息
        # 范围是 actual_proprio_dim 到 actual_proprio_dim + CONTACT_DIM
        if self.config['env']['privInfo']['enable_tactile']:
            self.tactile_hist_buf[:] = self.obs_buf_lag_history[:, -self.prop_hist_len:, self.actual_proprio_dim:self.actual_proprio_dim + CONTACT_DIM]

        # --------------------------------------------------------------
        # 7. 更新特权信息缓冲区 (Update Privileged Information Buffer)
        # --------------------------------------------------------------
        # 特权信息是只有训练时可用的“地面真值”信息，不包含噪声和延迟，用于辅助训练
        # _update_priv_buf 是一个辅助方法，用于更新特权信息字典或缓冲区
        self._update_priv_buf(env_id=range(self.num_envs), name='obj_position', value=self.object_pos.clone()) # 物体真实位置
        self._update_priv_buf(env_id=range(self.num_envs), name='obj_orientation', value=self.object_rot.clone()) # 物体真实姿态
        self._update_priv_buf(env_id=range(self.num_envs), name='obj_linvel', value=self.obj_linvel_at_cf.clone()) # 物体在接触时的真实线速度
        self._update_priv_buf(env_id=range(self.num_envs), name='obj_angvel', value=self.obj_angvel_at_cf.clone()) # 物体在接触时的真实角速度
        self._update_priv_buf(env_id=range(self.num_envs), name='fingertip_position', value=self.fingertip_pos.clone()) # 指尖真实位置
        self._update_priv_buf(env_id=range(self.num_envs), name='fingertip_orientation', value=self.fingertip_orientation.clone()) # 指尖真实姿态
        self._update_priv_buf(env_id=range(self.num_envs), name='fingertip_linvel', value=self.ft_linvel_at_cf.clone()) # 指尖在接触时的真实线速度
        self._update_priv_buf(env_id=range(self.num_envs), name='fingertip_angvel', value=self.ft_angvel_at_cf.clone()) # 指尖在接触时的真实角速度
        # 如果启用了触觉，更新真实的触觉信息（这里用的是经过延迟和噪声处理的 sensed_contacts，可能是为了某种特定的PrivIL方法）
        if self.config['env']['privInfo']['enable_tactile']:
            self._update_priv_buf(env_id=range(self.num_envs), name='tactile', value=self.sensed_contacts.clone())

        # --------------------------------------------------------------
        # 8. 更新 Critic 观测 (Critic Observation)
        # --------------------------------------------------------------
        # Critic 网络可能需要额外的、更全面的信息来评估当前状态的价值
        # note: the critic will receive normal observation, privileged info, and critic info
        # Critic的完整输入通常是：agent的观测 + 特权信息 + critic特有的信息
        # deprecated - 这行注释可能表示下面的 critic_info_buf 构建方式是旧的或即将废弃
        self.critic_info_buf[:, 0:4] = self.object_rot # 物体真实姿态（四元数）
        self.critic_info_buf[:, 4:7] = self.obj_linvel_at_cf # 物体在接触时的真实线速度
        self.critic_info_buf[:, 7:10] = self.obj_angvel_at_cf # 物体在接触时的真实角速度
        # 3,7,11,15,20 are fingertip indexes for rigid body states,即self.fingertip_handles
        # RIGID_BODY_STATES：每个刚体的状态向量包含 位置 (3) + 姿态 (4, 四元数) + 线速度 (3) + 角速度 (3) = 3+4+3+3=13 个分量
        # self.rigid_body_states 形状可能是 (num_envs, num_bodies, RIGID_BODY_STATES)
        # 提取指尖的刚体状态
        fingertip_states = self.rigid_body_states[:, self.fingertip_handles].clone()
        # Space E: 对 fingertip_states 中的速度分量进行逆缩放 (与 priv_buf 保持一致)
        # RIGID_BODY_STATES = 13: pos(3) + rot(4) + linvel(3) + angvel(3)
        # 速度分量在索引 7:13 (linvel 7:10, angvel 10:13)
        fingertip_states[:, :, 7:13] = fingertip_states[:, :, 7:13] * self.time_warper.obs_velocity_scale
        # 将指尖的刚体状态展平并存入 critic_info_buf
        # 形状变为 (num_envs, len(self.fingertip_handles) * RIGID_BODY_STATES)
        self.critic_info_buf[:, 10:10 + RIGID_BODY_STATES * len(self.fingertip_handles)] = fingertip_states.reshape(self.num_envs, -1)
        # critic_info_buf shape: torch.Size([1, 100])，这里的100是在config里定义的critic_info_dim

        # --------------------------------------------------------------
        # 9. 点云处理 (Point Cloud Processing)
        # --------------------------------------------------------------
        # 如果点云采样维度大于 0 (表示启用了点云观测)
        # point_cloud_sampled_dim 通常是采样的点数
        if self.point_cloud_sampled_dim > 0:
            # 点云主要用于收集行为克隆 (Behavior Cloning, BC) 数据
            # 将物体的本地点云 obj_point_clouds 旋转到世界坐标系下
            # self.object_rot[:, None].repeat(1, self.point_cloud_sampled_dim, 1) 将物体姿态复制 num_points 次
            # 然后加上物体在世界坐标系下的位置，得到点云在世界坐标系下的位置
            # 形状变为 (num_envs, point_cloud_sampled_dim=100, 3)
            self.point_cloud_buf[:, :self.point_cloud_sampled_dim] = quat_apply(
                self.object_rot[:, None].repeat(1, self.point_cloud_sampled_dim, 1), self.obj_point_clouds
            ) + self.object_pos[:, None]  
            # point_cloud_buf shape: torch.Size([1, self.point_cloud_sampled_dim=100, pos=3])
    
    def _get_reward_scale_by_name(self, name):
        env_steps = (self.gym.get_frame_count(self.sim) * len(self.envs))
        agent_steps = env_steps // self.control_freq_inv
    # 2. 从预定义的字典中获取该奖励项的尺度参数
    #    init_scale: 初始权重
    #    final_scale: 最终权重
    #    curr_start: 开始调整权重的智能体步数
    #    curr_end: 结束调整权重的智能体步数
        init_scale, final_scale, curr_start, curr_end = self.reward_scale_dict[name]
            # 3. 计算当前进度
        if curr_end > 0: # 如果 curr_end > 0，意味着权重是动态变化的
            # 计算当前在 [curr_start, curr_end] 区间内的进度百分比
            curr_progress = (agent_steps - curr_start) / (curr_end - curr_start)
            # 将进度限制在 [0, 1] 之间
            curr_progress = min(max(curr_progress, 0), 1)
            # 将连续的进度离散化到 [0, 0.05, 0.1, ..., 1.0]
            # 这样做的目的是在批量收集数据时，避免因奖励尺度微小连续变化导致学习不稳定
            curr_progress = round(curr_progress * 20) / 20
        else: # 如果 curr_end <= 0 (通常设为0或-1)，意味着权重是固定的，直接使用最终权重
            curr_progress = 1

        # 4. 如果处于评估模式，则直接使用最终权重
        if self.evaluate:
            curr_progress = 1

        # 5. 根据进度线性插值计算当前的奖励权重
        return init_scale + (final_scale - init_scale) * curr_progress

    def compute_reward(self, actions):
        # torque penalty
        torque_penalty = (self.torques[:, -1] ** 2).sum(-1)
        
        # ================================================================
        # Triangle Pass 旋转奖励计算（基于笔长轴投影的扫过角度）
        # ================================================================
        # 原理说明：
        # - 原逻辑计算的是刚体角速度在旋转轴上的投影，无法区分"公转"和"自转"
        # - 策略会利用"搓笔"（自转）来 Hack 高奖励，而不是真正的 Triangle Pass
        # - 新逻辑计算笔的长轴在旋转轴垂直平面上的投影向量的扫过角度
        # - 如果只是自转（搓笔），投影向量不会变化，奖励为 0
        # ================================================================
        
        # 1. 获取笔的长轴向量 (蓝线 = 局部 Z 轴 [0, 0, 1])
        local_axis_z = torch.zeros((self.num_envs, 3), device=self.device)
        local_axis_z[:, 2] = 1.0  # 笔的长轴是局部 Z 轴
        
        # 将局部向量旋转到世界坐标系，self.object_rot是物体当前帧的旋转四元数
        current_long_axis = quat_apply(self.object_rot, local_axis_z)      # 当前帧笔的长轴方向
        prev_long_axis = quat_apply(self.object_rot_prev, local_axis_z)    # 上一帧笔的长轴方向
        
        # 2. 将长轴投影到旋转轴的垂直平面
        # 投影公式: v_proj = v - (v · n) * n，其中 n 是旋转轴的单位向量
        # rot_axis_buf 已经在初始化时归一化
        rot_axis_normalized = self.rot_axis_buf / (torch.norm(self.rot_axis_buf, dim=-1, keepdim=True) + 1e-8)
        
        # 计算当前帧投影
        dot_curr = (current_long_axis * rot_axis_normalized).sum(dim=-1, keepdim=True)
        vec_curr_3d = current_long_axis - dot_curr * rot_axis_normalized
        
        # 计算上一帧投影
        dot_prev = (prev_long_axis * rot_axis_normalized).sum(dim=-1, keepdim=True)
        vec_prev_3d = prev_long_axis - dot_prev * rot_axis_normalized
        
        # 3. 归一化投影向量（关键步骤，防止长度变化影响叉乘结果）
        # 添加 epsilon 防止除零（当笔与旋转轴平行时）
        vec_curr_norm = torch.norm(vec_curr_3d, dim=-1, keepdim=True) + 1e-6
        vec_prev_norm = torch.norm(vec_prev_3d, dim=-1, keepdim=True) + 1e-6
        vec_curr_3d = vec_curr_3d / vec_curr_norm
        vec_prev_3d = vec_prev_3d / vec_prev_norm
        
        # 4. 计算 3D 叉乘获取旋转角
        # cross = prev × curr，结果向量与旋转轴平行时表示绕该轴的旋转
        # 叉乘结果与旋转轴的点积 = |prev||curr|sin(theta) * sign
        # 由于 prev 和 curr 已归一化，结果 ≈ sin(theta)
        cross_product = torch.cross(vec_prev_3d, vec_curr_3d, dim=-1)
        # 与旋转轴点积得到有符号的角度变化（正值=与旋转轴同向旋转）
        angle_delta = (cross_product * rot_axis_normalized).sum(dim=-1)
        
        # 5. 计算这一帧的"公转"速度 (rad/s)
        spin_velocity = angle_delta / (self.control_freq_inv * self.dt)
        
        # ==============================================================
        # Space E: 奖励输入还原缩放 (Reward Input Restoration)
        # ==============================================================
        # 目的: 将仿真中的角速度还原到"真实世界尺度"
        # 原理: ω_sim = α * ω_real (仿真中时间放慢α倍，角速度变为α倍)
        #       还原: ω_real = ω_sim / α = ω_sim * reward_input_scale
        # 这样 rotate_reward 的阈值和clip值在所有α下保持一致
        reward_input_scale = self.time_warper.reward_input_scale  # = 1/α
        spin_velocity_scaled = spin_velocity * reward_input_scale
        
        # 6. 计算奖励
        # 旋转轴的模长表示期望的旋转方向强度（已归一化为1）
        # angle_delta 已经包含了方向信息（与旋转轴同向为正）
        vec_dot = spin_velocity_scaled  # 与旋转轴同向旋转时为正（已还原到真实世界尺度）
        
        # ==============================================================
        # [Anti-Hacking] EMA 速度平滑 (Exponential Moving Average)
        # ==============================================================
        # 目的: 用于速度门控的平滑信号，防止高频振荡 exploit waypoint 奖励
        # 机制: ema_t = α * v_curr + (1-α) * ema_{t-1}
        # 注意: 主奖励/惩罚使用瞬时速度 vec_dot，EMA 仅用于速度门控
        # ==============================================================
        self.angvel_ema = self.ema_alpha * vec_dot + (1.0 - self.ema_alpha) * self.angvel_ema
        
        # ==============================================================
        # [Anti-Hacking] Jitter 惩罚 (速度变化率惩罚)
        # ==============================================================
        # 目的: 惩罚速度的急剧变化，鼓励平稳旋转
        # 公式: jitter_penalty = (v_t - v_{t-1})^2
        # 注意: 使用瞬时速度计算，外部 scale 由 jitter_penalty_scale 控制 (负值)
        # ==============================================================
        velocity_diff = vec_dot - self.prev_spin_velocity
        jitter_penalty = velocity_diff ** 2
        # 更新上一帧速度
        self.prev_spin_velocity = vec_dot.clone()
        
        # ==============================================================
        # [Anti-Hacking] 反向旋转惩罚 (Reverse Rotation Penalty)
        # ==============================================================
        # 目的: 惩罚与目标方向相反的旋转
        # 公式: reverse_penalty = |v| (当 v < 0 时)
        # 注意: 使用瞬时速度计算，外部 scale 由 reverse_penalty_scale 控制 (负值)
        # ==============================================================
        reverse_penalty = torch.where(
            vec_dot < 0,
            torch.abs(vec_dot),
            torch.zeros_like(vec_dot)
        )
        
        # 保存当前角速度用于early termination判断
        self.current_angvel = vec_dot.clone()
        
        # 同时计算刚体角速度用于日志输出（保留原逻辑用于对比分析）
        angdiff = quat_to_axis_angle(quat_mul(self.object_rot, quat_conjugate(self.object_rot_prev)))
        object_angvel = angdiff / (self.control_freq_inv * self.dt)
        
        # ============================================================
        # 旋转奖励计算 (支持两种模式)
        # ============================================================
        # 注意: 使用瞬时速度 vec_dot 进行奖励计算
        if self.use_gaussian_angvel_reward:
            # Gaussian Kernel 模式: r = exp(-||ω - ω_target||² / σ²)
            # 优点: 平滑、有明确的目标速度、不会奖励过快旋转
            # 适合需要精确速度控制的任务 (如 Triangle Pass)
            angvel_error = vec_dot - self.target_angvel
            rotate_reward = torch.exp(-(angvel_error ** 2) / (self.angvel_sigma ** 2))
            # 惩罚反向旋转 (负角速度) - 高斯核对负值也给予奖励，需要额外处理
            rotate_reward = torch.where(vec_dot < 0, torch.zeros_like(rotate_reward), rotate_reward)
            # Gaussian 模式下不使用额外的 rotate_penalty (反向旋转由 reverse_penalty 处理)
            rotate_penalty = torch.zeros_like(rotate_reward)
        else:
            # Clip-based 模式 (原逻辑): r = clip(ω, min, max)
            # 奖励只针对与期望方向一致的旋转，且不超过设定的最大速度
            rotate_reward = torch.clip(vec_dot, max=self.angvel_clip_max, min=self.angvel_clip_min)
            # vec_dot 的值低于 angvel_penalty_threshold_low或高于angvel_penalty_threshold_high时，才会开始施加惩罚
            penalty_overspeed = torch.relu(vec_dot - self.angvel_penalty_threshold_high)
            penalty_reverse_rotation = torch.relu(self.angvel_penalty_threshold_low - vec_dot)
            rotate_penalty = penalty_overspeed + penalty_reverse_rotation
        # 累计旋转角度（用于统计圈数）
        # 使用 angle_delta（投影角度变化量）
        # angle_delta 已经包含方向信息（与旋转轴同向为正）
        rot_angle = angle_delta  # 与旋转轴同向旋转时为正
        self.total_rot_angle += rot_angle
        
        # ============================================================
        # Waypoint 跟踪奖励计算 (Triangle Pass)
        # ============================================================
        # 基于当前相位获取目标指尖位置，使用高斯核函数计算奖励
        # 如果 waypoint 数据不足，奖励为 0
        if self.waypoint_tracking_enabled:
            # 计算当前相位 (复用已有的投影向量 vec_curr_3d)
            # 方法: 使用 tp_waypoints.py 中的 compute_phase_from_object_rotation
            # 注意: 该函数与此处的投影逻辑一致
            self.current_phase_buf = compute_phase_from_object_rotation(
                self.object_rot, self.rot_axis_buf
            )
            
            # 根据相位获取目标指尖位置
            target_fingertip_pos = get_target_fingertip_by_phase(
                self.current_phase_buf,
                self.dense_waypoint_pos,
                self.dense_phases
            )
            
            # 获取当前指尖位置 (相对于手基座，展平为15维)
            # fingertip_pos 已经是相对于手基座的位置（与 waypoint 采集时保持一致）
            current_fingertip_pos = self.fingertip_pos.reshape(self.num_envs, -1)  # (num_envs, 15)
            
            # 计算高斯核奖励
            # 如果禁用了无名指和小拇指，只比较中指、食指、大拇指 (后9维)
            raw_tracking_reward = compute_waypoint_tracking_reward(
                current_fingertip_pos,
                target_fingertip_pos,
                sigma=self.waypoint_sigma,
                finger_mask=self.waypoint_finger_mask
            )
            
            # 5. [Anti-Hacking] 高斯速度门控 (Gaussian Velocity Gating)
            # -------------------------------------------------------------
            # 目的: 只有在转速接近 target_angvel 时才给予姿态奖励。
            # 机制: Gate = exp(- (v_ema - v_target)^2 / sigma^2)
            # 使用 EMA 平滑后的速度，防止高频振荡 exploit waypoint 奖励
            # -------------------------------------------------------------
            
            # angvel_ema 是 EMA 平滑后的公转速度 (rad/s)，Space E 已将其还原为真实世界尺度
            # 计算速度误差
            gate_vel_error = self.angvel_ema - self.target_angvel
            
            # 计算高斯门控值 (范围 0.0 ~ 1.0)
            # 使用 self.angvel_sigma 确保与主旋转奖励的宽容度一致
            velocity_gate = torch.exp(-(gate_vel_error ** 2) / (self.angvel_sigma ** 2))
            
            # 强制约束: 如果反向旋转或静止 (angvel_ema <= 0)，门控强制为 0
            # 虽然高斯在 0 处已经很小 (exp(-3.14^2)≈0)，但硬截断更安全
            velocity_gate = torch.where(self.angvel_ema <= 1e-2, torch.zeros_like(velocity_gate), velocity_gate)
            
            # 应用门控
            waypoint_tracking_reward = raw_tracking_reward * velocity_gate
            
            # [Optional Debug] 偶尔打印门控状态，确认没有被死锁
            # if self.agent_steps % 5000 == 0:
            #     print(f"[Gate] Avg EMA Vel: {self.angvel_ema.mean():.2f}, Target: {self.target_angvel:.2f}, Avg Gate: {velocity_gate.mean():.3f}")
        else:
            waypoint_tracking_reward = torch.zeros(self.num_envs, device=self.device)
        
        # ============================================================
        # DEBUG: total_rad 计算逻辑调试输出（已更新为新的投影角度算法）
        # ============================================================
        if self.viewer is not None and hasattr(self, '_debug_spin_counter'):
            self._debug_spin_counter += 1
            # 每50步输出一次，避免刷屏
            if self._debug_spin_counter % 50 == 0:
                print("\n" + "="*75)
                print(f"[DEBUG spin_check] Step {self._debug_spin_counter} (新算法: 任意轴投影)")
                print("="*75)
                print(f"  total_rot_angle (rad): {self.total_rot_angle[0].item():.4f}")
                print(f"  total_rot_angle (deg): {self.total_rot_angle[0].item() * 180 / 3.14159:.2f}")
                print(f"  total_rot_angle (圈): {self.total_rot_angle[0].item() / (2*3.14159):.3f}")
                print("-"*75)
                print(f"  当前帧角度变化 angle_delta: {angle_delta[0].item():.6f} rad")
                print(f"  当前帧有效角度 rot_angle: {rot_angle[0].item():.6f} rad")
                print(f"  spin_velocity (投影公转速度): {spin_velocity[0].item():.4f} rad/s")
                print(f"  vec_dot (奖励用): {vec_dot[0].item():.4f} rad/s")
                print("-"*75)
                print(f"  旋转轴 rot_axis_buf: [{self.rot_axis_buf[0, 0].item():.3f}, {self.rot_axis_buf[0, 1].item():.3f}, {self.rot_axis_buf[0, 2].item():.3f}]")
                print(f"  笔长轴 (世界坐标): [{current_long_axis[0, 0].item():.4f}, {current_long_axis[0, 1].item():.4f}, {current_long_axis[0, 2].item():.4f}]")
                print(f"  笔长轴投影 (当前): [{vec_curr_3d[0, 0].item():.4f}, {vec_curr_3d[0, 1].item():.4f}, {vec_curr_3d[0, 2].item():.4f}]")
                print(f"  笔长轴投影 (上帧): [{vec_prev_3d[0, 0].item():.4f}, {vec_prev_3d[0, 1].item():.4f}, {vec_prev_3d[0, 2].item():.4f}]")
                print("-"*75)
                print(f"  [对比] 刚体角速度 (3D): [{object_angvel[0, 0].item():.4f}, {object_angvel[0, 1].item():.4f}, {object_angvel[0, 2].item():.4f}]")
                print(f"  [对比] 原 vec_dot (会被自转污染): {(object_angvel * self.rot_axis_buf).sum(-1)[0].item():.4f}")
                print(f"  rotate_reward: {rotate_reward[0].item():.4f}")
                print("="*75)
        # elif self.viewer is not None and not hasattr(self, '_debug_spin_counter'):
        #     self._debug_spin_counter = 0
        #     print("\n[DEBUG] spin_check 调试模式已启用（新算法: 任意轴投影），将每50步输出一次详细信息")
        
        # 计算物体线速度惩罚，这里不使用self.object_linvel，而是用位置差分计算
        # 在仿真中，物理计算频率（Physics Freq, e.g., 1000Hz）通常远高于控制频率（Control Freq, e.g., 50Hz）。 self.object_linvel 只是这 20 个物理步中最后一步的速度。如果物体刚好在那一步发生了碰撞（Contact），瞬时速度可能会剧烈抖动（高频噪声），而位置差分则平滑了这一过程。
        object_linvel = ((self.object_pos - self.object_pos_prev) / (self.control_freq_inv * self.dt)).clone()
        object_linvel_penalty = torch.norm(object_linvel, p=1, dim=-1)
        # TODO: move this to a more appropriate place
        # Space E: 对速度应用逆缩放 (v_obs = v_sim / α)，使 Agent 感知真实世界尺度
        v_scale = self.time_warper.obs_velocity_scale
        self.obj_angvel_at_cf = object_angvel * v_scale
        self.obj_linvel_at_cf = object_linvel * v_scale
        # 对旋转四元数的乘法
        ft_angdiff = quat_to_axis_angle(quat_mul(self.fingertip_orientation.reshape(-1, 4), quat_conjugate(self.ft_rot_prev.reshape(-1, 4)))).reshape(-1, 3*FINGERTIP_CNT)
        self.ft_angvel_at_cf = ft_angdiff / (self.control_freq_inv * self.dt) * v_scale
        # 指尖线速度计算使用世界坐标系位置（速度是位置的变化率，与参考系无关）
        self.ft_linvel_at_cf = ((self.fingertip_pos_world - self.ft_pos_prev_world) / (self.control_freq_inv * self.dt)) * v_scale
        # ============================================================
        # 轴向倾斜惩罚 (Axial Tilt Penalty)
        # ============================================================
        # 惩罚物体在旋转轴方向的高度差（投影长度差）
        # 当笔与旋转轴平行时，投影长度最大；当笔垂直于旋转轴时，投影长度最小
        # 这个惩罚鼓励笔保持与旋转轴垂直的姿态
        if self.point_cloud_sampled_dim > 0:
            # 获取点云数据 (num_envs, num_points, 3)
            point_cloud = self.point_cloud_buf[:, :self.point_cloud_sampled_dim, :]
            
            # 将点云投影到旋转轴方向
            # rot_axis_buf: (num_envs, 3) - 已归一化的旋转轴
            rot_axis_normalized = self.rot_axis_buf / (torch.norm(self.rot_axis_buf, dim=-1, keepdim=True) + 1e-8)
            
            # 计算每个点在旋转轴方向的投影值
            # point_cloud: (num_envs, num_points, 3), rot_axis: (num_envs, 1, 3)
            # 结果: (num_envs, num_points)
            axial_projection = (point_cloud * rot_axis_normalized.unsqueeze(1)).sum(dim=-1)
            
            # 计算旋转轴方向的高度差（最大投影 - 最小投影）
            axial_tilt_penalty = axial_projection.max(dim=1)[0] - axial_projection.min(dim=1)[0]
            
            # 设置阈值：只有当倾斜超过一定程度时才惩罚
            axial_tilt_penalty[axial_tilt_penalty <= self.axial_tilt_threshold] = 0
        else:
            axial_tilt_penalty = torch.zeros(self.num_envs, device=self.device)

        # 惩罚在z轴上的位置偏离
        position_penalty = (self.object_pos[:, 2] - OBJ_CANON_POS[2]) ** 2
        # 未使用，惩罚指尖与物体的距离（使用世界坐标系）
        finger_obj_penalty = ((self.fingertip_pos_world - self.object_pos.repeat(1, FINGERTIP_CNT)) ** 2).sum(-1)

        # ============================================================
        # Flying base 移动惩罚
        # ============================================================
        # 惩罚 Flying base 的频繁移动，鼓励策略依赖手指技巧而非手腕运动
        # 计算方式：基于位置差分计算速度 L1 范数
        # - 线速度惩罚 (前3个 DOF: px, py, pz)
        # - 角速度惩罚 (后3个 DOF: rx, ry, rz)
        # ============================================================
        if self.flying_hand_enabled:
            flying_base_vel = (self.linker_hand_dof_pos[:, :NUM_FLYING_DOF] - self.flying_base_pos_prev) / (self.control_freq_inv * self.dt)
            # 分别计算线速度和角速度的 L1 范数
            flying_linear_vel = torch.abs(flying_base_vel[:, :3]).sum(dim=-1)   # px, py, pz
            flying_angular_vel = torch.abs(flying_base_vel[:, 3:6]).sum(dim=-1)  # rx, ry, rz
            # 综合惩罚：线速度 + 角速度（可以考虑不同权重）
            flying_base_movement_penalty = flying_linear_vel + flying_angular_vel
        else:
            flying_base_movement_penalty = torch.zeros(self.num_envs, device=self.device)

        self.rew_buf[:] = compute_hand_reward(
            object_linvel_penalty, self._get_reward_scale_by_name('obj_linvel_penalty'),
            rotate_reward, self._get_reward_scale_by_name('rotate_reward'),
            torque_penalty, self._get_reward_scale_by_name('torque_penalty'),
            axial_tilt_penalty, self._get_reward_scale_by_name('axial_tilt_penalty'),
            position_penalty, self._get_reward_scale_by_name('position_penalty'),
            rotate_penalty, self._get_reward_scale_by_name('rotate_penalty'),
            flying_base_movement_penalty, self._get_reward_scale_by_name('flying_base_movement_penalty'),
            waypoint_tracking_reward, self._get_reward_scale_by_name('waypoint_tracking_reward'),
            jitter_penalty, self._get_reward_scale_by_name('jitter_penalty'),
            reverse_penalty, self._get_reward_scale_by_name('reverse_penalty')
        )
        
        # ==============================================================
        # Space E: 奖励输出缩放 (Reward Output Scaling / Time Dilation)
        # ==============================================================
        # 目的: 补偿慢时间世界中"多收集奖励"的偏差
        # 原理: 当 α < 1 时，仿真时间变慢，同样真实时间内收集的奖励变多
        #       缩放: r_final = r_raw * α (α越小，单步奖励越小)
        # 这样总奖励与真实时间成正比，保持价值函数估计的一致性
        reward_output_scale = self.time_warper.reward_output_scale  # = α
        self.rew_buf[:] = self.rew_buf * reward_output_scale
        
        self.reset_buf[:] = self.check_termination(self.object_pos)
        
        #mean都是对envs维度，compute_reward 函数本身计算的是当前这个时间步获得的即时奖励
        #PPO中奖励的累加实现在ppo.py的play_step()中，self.current_rewards += rewards
        #Tensorboard的奖励累加实现在mean_rewards = self.episode_rewards.get_mean()，mean也是对env维度

        # extras部分是传入ppo中的infos
        # _get_reward_scale_by_name()要更改 configs/task/LinkerHandHora.yaml中的scale
        self.extras['timestep_reward_sum'] = self.rew_buf.mean() # rew_buf 已经是各项加权求和后的总奖励，所以这里不变
        
        # === 奖励 (reward, 正值 scale) ===
        self.extras['reward/rotation_reward'] = (rotate_reward * self._get_reward_scale_by_name('rotate_reward')).mean()
        self.extras['reward/waypoint_tracking_reward'] = (waypoint_tracking_reward * self._get_reward_scale_by_name('waypoint_tracking_reward')).mean()
        
        # === 惩罚 (penalty, 负值 scale) ===
        self.extras['penalty/object_linvel_penalty'] = (object_linvel_penalty * self._get_reward_scale_by_name('obj_linvel_penalty')).mean()
        self.extras['penalty/torques'] = (torque_penalty * self._get_reward_scale_by_name('torque_penalty')).mean()
        self.extras['penalty/axial_tilt_penalty'] = (axial_tilt_penalty * self._get_reward_scale_by_name('axial_tilt_penalty')).mean()
        self.extras['penalty/object_position_penalty'] = (position_penalty * self._get_reward_scale_by_name('position_penalty')).mean()
        self.extras['penalty/rotate_penalty'] = (rotate_penalty * self._get_reward_scale_by_name('rotate_penalty')).mean()
        self.extras['penalty/flying_base_movement_penalty'] = (flying_base_movement_penalty * self._get_reward_scale_by_name('flying_base_movement_penalty')).mean()
        self.extras['penalty/jitter_penalty'] = (jitter_penalty * self._get_reward_scale_by_name('jitter_penalty')).mean()
        self.extras['penalty/reverse_penalty'] = (reverse_penalty * self._get_reward_scale_by_name('reverse_penalty')).mean()
        
        # === 原始值 (未加权) ===
        self.extras['raw/jitter_penalty'] = jitter_penalty.mean()
        self.extras['raw/reverse_penalty'] = reverse_penalty.mean()
        self.extras['raw/angvel_ema'] = self.angvel_ema.mean()
        
        self.extras['finger_obj_penalty(NOT USED)'] = finger_obj_penalty.mean()
        self.extras['vel/roll_angvel'] = torch.abs(object_angvel[:, 0]).mean()
        self.extras['vel/pitch_angvel'] = torch.abs(object_angvel[:, 1]).mean()
        self.extras['vel/yaw_angvel(NEED)'] = torch.abs(object_angvel[:, 2]).mean()
        # 新增：投影角度算法相关日志（用于对比分析）
        self.extras['vel/spin_velocity(NEW)'] = spin_velocity.mean()  # 新算法：投影公转速度
        self.extras['vel/old_vec_dot(HACK)'] = (object_angvel * self.rot_axis_buf).sum(-1).mean()  # 原算法：会被自转污染
        # sparse，不能在每个时间步直接对所有环境取平均值
        self.extras['rot_angle'] = rot_angle
        # Waypoint tracking 额外日志
        self.extras['reward/waypoint_tracking_reward_per_env'] = waypoint_tracking_reward * self._get_reward_scale_by_name('waypoint_tracking_reward')  # 每个环境的值
        self.extras['phase/current_phase'] = self.current_phase_buf.mean()  # 平均相位 (0~2π)
        self.extras['phase/current_phase_deg'] = (self.current_phase_buf * 180 / 3.14159).mean()  # 平均相位 (度)
        
        # 添加终止原因统计信息

        self.extras['termination/total_episodes'] = self.termination_counts['total_episodes']
        self.extras['termination/max_episode_length_count'] = self.termination_counts['max_episode_length']
        self.extras['termination/object_below_threshold_count'] = self.termination_counts['object_below_threshold']
        self.extras['termination/pencil_tilt_count'] = self.termination_counts['pencil_tilt']
        self.extras['termination/angular_velocity_too_high_count'] = self.termination_counts['angular_velocity_too_high']
        # 输出当前终止原因张量，供 ppo_rl_teacher 计算 survival_rate
        # 0: 未终止, 1: max_episode_length, 2: object_below_threshold, 3: pencil_tilt, 4: angular_velocity_too_high
        self.extras['termination_reason'] = self.current_termination_reason.clone()

        if self.evaluate:
            for i in range(len(self.object_type_list)):
                env_ids = torch.where(self.object_type_at_env == i)
                if len(env_ids[0]) > 0:
                    running_mask = 1 - self.eval_done_buf[env_ids]
                    self.stat_sum_rewards[i] += (running_mask * self.rew_buf[env_ids]).sum()
                    self.stat_sum_episode_length[i] += running_mask.sum()
                    self.stat_sum_rotate_rewards[i] += (running_mask * rotate_reward[env_ids]).sum()
                    self.stat_sum_unclip_rotate_rewards[i] += (running_mask * vec_dot[env_ids]).sum()

                    # Update eval_done_buf when evaluating just one object. This will
                    # stop tracking statistics after environment resets.
                    if self.config['env']['object']['evalObjectType'] is not None:
                        flip = running_mask * self.reset_buf[env_ids]
                        self.env_evaluated[i] += flip.sum()
                        self.eval_done_buf[env_ids] += flip

                    info = f'Progress: {self.evaluate_iter} / {self.max_episode_length}'
                    tprint(info)
            self.evaluate_iter += 1

    def post_physics_step(self):
        self.progress_buf += 1
        self.reset_buf[:] = 0
        self._refresh_gym()
        # cur* but need for reward is here
        self.compute_reward(self.actions)

        #这里的env_ids是指当前处于重置状态的环境实例的索引
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0: # 对于处于重置状态的环境实例
            # 在演示状态下输出圈数和终止原因
            if self.viewer:
                for eid in env_ids:
                    turns = float(self.total_rot_angle[eid].item()) / (2 * math.pi)
                    reason_code = int(self.current_termination_reason[eid].item())
                    actual_value = float(self.termination_actual_values[eid].item())
                    
                    if reason_code == 1:
                        reason_text = f"达到最大步数(阈值={self.max_episode_length}步, 实际={int(actual_value)}步)"
                    elif reason_code == 2:
                        init_z = float(self.init_object_z_buf[eid].item())
                        reason_text = f"物体高度偏离超过阈值(允许偏离={self.relative_z_drop_threshold:.3f}m, 实际偏离={actual_value:.3f}m, 初始高度={init_z:.3f}m)"
                    elif reason_code == 3:
                        reason_text = f"铅笔倾倒(高度差阈值={self.pencil_tilt_threshold:.2f}m, 实际={actual_value:.3f}m)"
                    elif reason_code == 4:
                        threshold = 10.0 * self.angvel_penalty_threshold_high
                        reason_text = f"角速度过大(阈值={threshold:.2f}rad/s, 实际={abs(actual_value):.2f}rad/s)"
                    else:
                        reason_text = f"未知原因({reason_code})"
                    
                    print(f"[演示] 环境{eid}本轮累计转笔圈数: {turns:.2f}, 终止原因: {reason_text}")
            self.total_rot_angle[env_ids] = 0.0
            # [Anti-Hacking] 重置 EMA 和 jitter 惩罚相关状态
            self.angvel_ema[env_ids] = 0.0
            self.prev_spin_velocity[env_ids] = 0.0
            self.reset_idx(env_ids)
        self.compute_observations()

        self.debug_viz = False
        # 仿真 Viewer（也就是说是否打开了图形界面）
        if self.viewer and self.config['env']['privInfo']['enable_tactile']:
            for env in range(len(self.envs)):
                for i, contact_idx in enumerate(list(self.sensor_handle_indices)):

                    # 用于Viewer界面的可视化，对于所有sensor_handles，可视化接触力范数 > CONTACT_THRESH的刚体
                    contact_norm = np.linalg.norm(self.debug_contacts[env, i])
                    if contact_norm > CONTACT_THRESH:
                        fx = self.debug_contacts[env, i, 0]
                        fy = self.debug_contacts[env, i, 1]
                        fz = self.debug_contacts[env, i, 2]
                        # print(f"Fx: {fx:.4f}, Fy: {fy:.4f}, Fz: {fz:.4f}, Norm: {contact_norm:.4f}")
                        self.gym.set_rigid_body_color(self.envs[env], self.hand_actors[env],
                                                      contact_idx, gymapi.MESH_VISUAL_AND_COLLISION,
                                                      gymapi.Vec3(0.0, 1.0, 0.0)) # RGB 颜色向量 (R, G, B)，这里是绿色
                    else:
                        self.gym.set_rigid_body_color(self.envs[env], self.hand_actors[env],
                                                      contact_idx, gymapi.MESH_VISUAL_AND_COLLISION,
                                                      gymapi.Vec3(1.0, 0.0, 0.0)) # RGB 颜色向量 (R, G, B)，这里是红色
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

# vec_task在step()中调用了pre_physics_step()，而vec_task的step()在子类linker_hand_hora的step()被调用被重写
# linker_hand_hora的step()最终会作为env.step在demon.py和ppo.py中被调用
    def pre_physics_step(self, actions):
        # ====================================================================
        # 动作空间映射
        # ====================================================================
        # 策略网络输出的动作可能是：
        # - 27 维 (Flying Hand + 全部手指)
        # - 19 维 (Flying Hand + 禁用 ring/little)
        # - 21 维 (无 Flying + 全部手指)
        # - 13 维 (无 Flying + 禁用 ring/little)
        #
        # 需要将策略动作映射到仿真的完整 DOF 空间
        # ====================================================================
        
        # 如果动作维度与仿真 DOF 不同，需要扩展
        if self.disable_ring_little:
            # 将缩减的策略动作扩展为完整的仿真 DOF 动作
            expanded_actions = self.action_mapper.expand_actions(actions, None)
            self.actions = expanded_actions.clone().to(self.device)
        else:
            self.actions = actions.clone().to(self.device)
        
        targets = self.prev_targets + self.action_scale * self.actions
        # targets = self.actions.clone()
        
        # ============================================================
        # Flying base 速度限制
        # ============================================================
        # 限制 Flying base 每一步的最大位移，防止过快移动
        # - 线速度限制 (前3个 DOF: px, py, pz) -> 最大位移 = linearVelocity * control_dt
        # - 角速度限制 (后3个 DOF: rx, ry, rz) -> 最大位移 = angularVelocity * control_dt
        # ============================================================
        if self.flying_hand_enabled:
            control_dt = self.control_freq_inv * self.dt  # 控制步长时间
            max_linear_disp = self.flying_linear_velocity * control_dt  # 最大线位移
            max_angular_disp = self.flying_angular_velocity * control_dt  # 最大角位移
            
            # 计算 Flying base 的位移 (相对于上一步的目标)
            flying_base_delta = targets[:, :NUM_FLYING_DOF] - self.prev_targets[:, :NUM_FLYING_DOF]
            
            # 分别限制线位移和角位移
            flying_base_delta[:, :3] = torch.clamp(flying_base_delta[:, :3], -max_linear_disp, max_linear_disp)
            flying_base_delta[:, 3:6] = torch.clamp(flying_base_delta[:, 3:6], -max_angular_disp, max_angular_disp)
            
            # 应用限制后的位移
            targets[:, :NUM_FLYING_DOF] = self.prev_targets[:, :NUM_FLYING_DOF] + flying_base_delta
        
        # ============================================================
        # 使用每个环境独立的限位（相对限位）
        # ============================================================
        if self.use_relative_limit:
            # 每个环境有独立的限位，根据初始位置计算
            self.cur_targets[:] = torch.max(torch.min(targets, self.env_dof_upper_limits), self.env_dof_lower_limits)
        else:
            # 使用全局绝对限位
            self.cur_targets[:] = tensor_clamp(targets, self.linker_hand_dof_lower_limits, self.linker_hand_dof_upper_limits)
        # get prev* buffer here
        self.prev_targets[:] = self.cur_targets
        self.object_rot_prev[:] = self.object_rot
        self.object_pos_prev[:] = self.object_pos
        self.ft_rot_prev[:] = self.fingertip_orientation
        # 使用世界坐标系位置用于速度计算
        self.ft_pos_prev_world[:] = self.fingertip_pos_world
        self.dof_vel_prev[:] = self.dof_vel_finite_diff
        # 保存 Flying base 的当前位置用于下一步速度计算
        if self.flying_hand_enabled:
            self.flying_base_pos_prev[:] = self.linker_hand_dof_pos[:, :NUM_FLYING_DOF]

    def reset(self):
        super().reset() # 直接对所有env调用了self.reset_idx(env_ids)
        self.obs_dict['priv_info'] = self.priv_info_buf.to(self.rl_device)
        self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.rl_device)
        self.obs_dict['tactile_hist'] = self.tactile_hist_buf.to(self.rl_device)
        self.obs_dict['noisy_quaternion'] = self.noisy_quaternion_buf.to(self.rl_device)
        # observation buffer for critic
        self.obs_dict['critic_info'] = self.critic_info_buf.to(self.rl_device)
        self.obs_dict['point_cloud_info'] = self.point_cloud_buf.to(self.rl_device)
        self.obs_dict['rot_axis_buf'] = self.rot_axis_buf.to(self.rl_device)
        if self.enable_obj_ends:
            self.obs_dict['obj_ends'] = self.obj_ends_history.to(self.rl_device)
        # one-time shape summary for debugging
        if self.debug_shape_summary and not self._shape_summary_logged_once:
            try:
                print("[Shape Summary][reset]")
                print(f"  priv_info_buf:          {tuple(self.priv_info_buf.shape)}")
                print(f"  proprio_hist_buf:       {tuple(self.proprio_hist_buf.shape)}")
                print(f"  tactile_hist_buf:       {tuple(self.tactile_hist_buf.shape)}")
                print(f"  point_cloud_buf:        {tuple(self.point_cloud_buf.shape)}")
                print(f"  critic_info_buf:        {tuple(self.critic_info_buf.shape)}")
                if self.enable_obj_ends:
                    print(f"  obj_ends_history:       {tuple(self.obj_ends_history.shape)}")
                print(f"  numObservations (obs):  {self.config['env']['numObservations']}")
                print(f"  CONTACT_DIM:            {CONTACT_DIM}, FINGERTIP_POS_DIM: {FINGERTIP_POS_DIM}, PROPRIO_DIM: {PROPRIO_DIM}")
            finally:
                self._shape_summary_logged_once = True
        return self.obs_dict

    def step(self, actions, extrin_record: Optional[torch.Tensor] = None):
        # Save extrinsics if evaluating on just one object.
        if extrin_record is not None and self.config['env']['object']['evalObjectType'] is not None:
            # Put a (z vectors, is done) tuple into the log.
            self.extrin_log.append(
                (extrin_record.detach().cpu().numpy().copy(), self.eval_done_buf.detach().cpu().numpy().copy())
            )

        super().step(actions)
        self.obs_dict['priv_info'] = self.priv_info_buf.to(self.rl_device)
        # stage 2 buffer
        self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.rl_device)
        self.obs_dict['tactile_hist'] = self.tactile_hist_buf.to(self.rl_device)
        self.obs_dict['noisy_quaternion'] = self.noisy_quaternion_buf.to(self.rl_device)
        # observation buffer for critic
        self.obs_dict['critic_info'] = self.critic_info_buf.to(self.rl_device)
        self.obs_dict['point_cloud_info'] = self.point_cloud_buf.to(self.rl_device)
        self.obs_dict['rot_axis_buf'] = self.rot_axis_buf.to(self.rl_device)
        if self.enable_obj_ends:
            self.obs_dict['obj_ends'] = self.obj_ends_history.to(self.rl_device)
        # one-time shape summary for debugging (if not printed in reset path)
        if self.debug_shape_summary and not self._shape_summary_logged_once:
            try:
                print("[Shape Summary][step]")
                print(f"  priv_info_buf:          {tuple(self.priv_info_buf.shape)}")
                print(f"  proprio_hist_buf:       {tuple(self.proprio_hist_buf.shape)}")
                print(f"  tactile_hist_buf:       {tuple(self.tactile_hist_buf.shape)}")
                print(f"  point_cloud_buf:        {tuple(self.point_cloud_buf.shape)}")
                print(f"  critic_info_buf:        {tuple(self.critic_info_buf.shape)}")
                if self.enable_obj_ends:
                    print(f"  obj_ends_history:       {tuple(self.obj_ends_history.shape)}")
                print(f"  numObservations (obs):  {self.config['env']['numObservations']}")
                print(f"  CONTACT_DIM:            {CONTACT_DIM}, FINGERTIP_POS_DIM: {FINGERTIP_POS_DIM}, PROPRIO_DIM: {PROPRIO_DIM}")
            finally:
                self._shape_summary_logged_once = True
        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras

    def capture_frame(self) -> np.ndarray:
        assert self.enable_camera_sensors  # camera sensors should be enabled
        assert self.vid_record_tensor is not None
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        frame = self.vid_record_tensor.cpu().numpy()
        self.gym.end_access_image_tensors(self.sim)

        return frame

    def update_low_level_control(self, step_id):
        previous_dof_pos = self.linker_hand_dof_pos.clone()
        self._refresh_gym()
        random_action_noise_t = torch.normal(0, self.random_action_noise_t_scale, size=self.linker_hand_dof_pos.shape, device=self.device, dtype=torch.float)
        noise_action = self.cur_targets + self.random_action_noise_e + random_action_noise_t
        
        # Flying Hand: 前 6 个 DOF 是虚拟基座，不应添加噪声
        # 它们需要精确执行位控指令
        if self.flying_hand_enabled:
            noise_action[:, :NUM_FLYING_DOF] = self.cur_targets[:, :NUM_FLYING_DOF]
        
        if self.torque_control:
            dof_pos = self.linker_hand_dof_pos
            dof_vel = (dof_pos - previous_dof_pos) / self.dt
            self.dof_vel_finite_diff[:, step_id] = dof_vel.clone()
            torques = self.p_gain * (noise_action - dof_pos) - self.d_gain * dof_vel
            torques = torch.clip(torques, -self.torque_limit, self.torque_limit).clone()
            self.torques[:, step_id] = torques
            
            # ============================================================
            # Flying Hand 混合控制模式：
            # - Flying base (前6个DOF): DOF_MODE_POS，使用 IsaacGym 内置 PD
            # - 手部关节 (后21个DOF): DOF_MODE_EFFORT，使用自定义力矩
            # ============================================================
            if self.flying_hand_enabled:
                # Flying base: 通过 set_dof_position_target_tensor 设置位置目标
                # IsaacGym 内置 PD 控制器会自动执行位置跟踪
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))
                
                # 手部关节: Flying base 的力矩设为 0（它们由位置控制器驱动）
                torques[:, :NUM_FLYING_DOF] = 0.0
            
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
        else:
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(noise_action))

    def update_rigid_body_force(self):
        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)
            # apply new forces
            obj_mass = [self.gym.get_actor_rigid_body_properties(env, self.gym.find_actor_handle(env, 'object'))[0].mass for env in self.envs]
            obj_mass = to_torch(obj_mass, device=self.device)
            prob = self.random_force_prob_scalar
            force_indices = (torch.less(torch.rand(self.num_envs, device=self.device), prob)).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape,
                device=self.device) * obj_mass[force_indices, None] * self.force_scale
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.ENV_SPACE)

    def check_termination(self, object_pos):
        term_by_max_eps = torch.greater_equal(self.progress_buf, self.max_episode_length)
        # 使用相对高度阈值：当物体偏离初始高度超过阈值时终止（上升或下降都算）
        # relative_z_drop_threshold 表示允许偏离的最大距离（正值）
        z_deviation = torch.abs(self.init_object_z_buf - object_pos[:, -1])  # 偏离了多少（绝对值）
        reset_z = torch.greater(z_deviation, self.relative_z_drop_threshold)  # 偏离超过阈值
        
        # 新增：角速度过大的early termination（超过最高转速阈值的10倍）
        # 用于过滤初始化时发生碰撞导致的异常高速旋转
        angvel_too_high = torch.greater(torch.abs(self.current_angvel), 10.0 * self.angvel_penalty_threshold_high)
        
        resets = reset_z
        
        # 重置终止原因记录
        self.current_termination_reason.fill_(0)
        self.termination_actual_values.fill_(0)
        
        # 记录各种终止原因
        term_by_max_eps_envs = torch.where(term_by_max_eps)[0]
        reset_z_envs = torch.where(reset_z)[0]
        angvel_too_high_envs = torch.where(angvel_too_high)[0]
        
        # 设置终止原因 (优先级: 角速度过大 > 物体高度偏离 > 达到最大长度)
        # 同时记录实际值用于调试
        self.current_termination_reason[reset_z_envs] = 2  # object_z_deviation
        self.termination_actual_values[reset_z_envs] = z_deviation[reset_z_envs]  # 记录实际偏离距离
        
        self.current_termination_reason[term_by_max_eps_envs] = 1  # max_episode_length
        self.termination_actual_values[term_by_max_eps_envs] = self.progress_buf[term_by_max_eps_envs].float()
        
        self.current_termination_reason[angvel_too_high_envs] = 4  # angular_velocity_too_high
        self.termination_actual_values[angvel_too_high_envs] = self.current_angvel[angvel_too_high_envs]
        
        resets = torch.logical_or(resets, term_by_max_eps)
        resets = torch.logical_or(resets, angvel_too_high)

        if self.canonical_pose_category == 'pencil':
            # LinkerPen 端点计算，无需缩放
            pencil_ends = [
                [0, 0, -self.pen_length / 2],  # -0.09m
                [0, 0, self.pen_length / 2]    # +0.09m
            ]
            pencil_end_1 = self.object_pos + quat_apply(self.object_rot, to_torch(pencil_ends[0], device=self.device)[None].repeat(self.num_envs, 1))
            pencil_end_2 = self.object_pos + quat_apply(self.object_rot, to_torch(pencil_ends[1], device=self.device)[None].repeat(self.num_envs, 1))
            pencil_z_min = torch.min(pencil_end_1, pencil_end_2)[:, -1]
            pencil_z_max = torch.max(pencil_end_1, pencil_end_2)[:, -1]
            # LinkerPen 倾倒判定：高度差超过阈值（笔长 0.18m，约 45度倾斜）
            pencil_tilt = torch.greater(pencil_z_max - pencil_z_min, 0.12)  # pencil fall threshold 0.12m

            # 记录铅笔倾倒的环境 (最高优先级)
            pencil_tilt_envs = torch.where(pencil_tilt)[0]
            self.current_termination_reason[pencil_tilt_envs] = 3  # pencil_tilt
            # 记录实际的高度差值
            pencil_height_diff = pencil_z_max - pencil_z_min
            self.termination_actual_values[pencil_tilt_envs] = pencil_height_diff[pencil_tilt_envs]
            
            resets = torch.logical_or(resets, pencil_tilt)

        # 统计终止原因
        reset_envs = torch.where(resets)[0]
        for env_id in reset_envs:
            reason = self.current_termination_reason[env_id].item()
            if reason == 1:
                self.termination_counts['max_episode_length'] += 1
            elif reason == 2:
                self.termination_counts['object_below_threshold'] += 1
            elif reason == 3:
                self.termination_counts['pencil_tilt'] += 1
            elif reason == 4:
                self.termination_counts['angular_velocity_too_high'] += 1
            self.termination_counts['total_episodes'] += 1
        
        return resets
    def reset_termination_counts(self):
        """
        手动重置终止原因计数器。
        通常在 PPO 的每个 Epoch 结束时调用，以便统计当前 Epoch 的分布。
        """
        self.termination_counts = {
            'max_episode_length': 0,
            'object_below_threshold': 0,
            'angular_velocity_too_high': 0,
            'pencil_tilt': 0,
            'total_episodes': 0
        }
    def _refresh_gym(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:RIGID_BODY_STATES]
        self.fingertip_states = self.rigid_body_states[:, self.fingertip_handles]
        # 世界坐标系下的指尖位置 (保留用于速度计算等)
        self.fingertip_pos_world = self.fingertip_states[:, :, :3].reshape(self.num_envs, -1)
        
        # ============================================================
        # 计算相对于手部基座的指尖位置 (Base-Relative Frame)
        # ============================================================
        # hand_base_pos: 手部基座在世界坐标系中的位置 (num_envs, 3)
        # 对于 Flying Hand: 这是 root_state_tensor 中的位置
        # 对于固定基座: 同样使用 root_state_tensor（虽然固定，但保持一致性）
        self.hand_base_pos = self.root_state_tensor[self.hand_indices, :3]  # (num_envs, 3)
        
        # 将每个指尖位置减去手部基座位置，得到相对位置
        # fingertip_pos_world: (num_envs, 15) = (num_envs, 5指 * 3维)
        # hand_base_pos: (num_envs, 3) -> repeat 5次 -> (num_envs, 15)
        self.fingertip_pos = self.fingertip_pos_world - self.hand_base_pos.repeat(1, FINGERTIP_CNT)
        
        # 保持其他属性不变（使用世界坐标系）
        self.fingertip_orientation = self.fingertip_states[:, :, 3:7].reshape(self.num_envs, -1)
        self.fingertip_linvel = self.fingertip_states[:, :, 7:10].reshape(self.num_envs, -1)
        self.fingertip_angvel = self.fingertip_states[:, :, 10:RIGID_BODY_STATES].reshape(self.num_envs, -1)

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
        # 恢复系数随机化配置
        self.randomize_restitution = rand_config.get('randomizeRestitution', True)
        self.randomize_restitution_lower = rand_config.get('randomizeRestitutionLower', 0.1)
        self.randomize_restitution_upper = rand_config.get('randomizeRestitutionUpper', 0.5)
        self.randomize_scale = rand_config['randomizeScale']
        self.randomize_hand_scale = rand_config['randomize_hand_scale']
        self.scale_list_init = rand_config['scaleListInit']
        self.randomize_scale_list = rand_config['randomizeScaleList']
        self.randomize_scale_lower = rand_config['randomizeScaleLower']
        self.randomize_scale_upper = rand_config['randomizeScaleUpper']
        self.randomize_pd_gains = rand_config['randomizePDGains']
        self.randomize_p_gain_lower = rand_config['randomizePGainLower']
        self.randomize_p_gain_upper = rand_config['randomizePGainUpper']
        self.randomize_d_gain_lower = rand_config['randomizeDGainLower']
        self.randomize_d_gain_upper = rand_config['randomizeDGainUpper']
        self.random_obs_noise_e_scale = rand_config['obs_noise_e_scale']
        self.random_obs_noise_t_scale = rand_config['obs_noise_t_scale']
        self.random_pose_noise = rand_config['pose_noise_scale']
        self.random_action_noise_e_scale = rand_config['action_noise_e_scale']
        self.random_action_noise_t_scale = rand_config['action_noise_t_scale']
        # stage 2 specific
        self.noisy_rpy_scale = rand_config['noisy_rpy_scale']
        self.noisy_pos_scale = rand_config['noisy_pos_scale']

        self.sensor_thresh = 1.0
        self.sensor_noise = 0.1
        self.latency = 0.2

    def _setup_priv_option_config(self, p_config):
        # 读取所有可选特权信息的配置
        self.enable_priv_obj_orientation = p_config.get('enable_obj_orientation', False)
        self.enable_priv_obj_linvel = p_config.get('enable_obj_linvel', False)
        self.enable_priv_obj_angvel = p_config.get('enable_obj_angvel', False)
        self.enable_priv_fingertip_position = p_config.get('enable_ft_pos', False)
        self.enable_priv_fingertip_orientation = p_config.get('enable_ft_orientation', False)
        self.enable_priv_fingertip_linvel = p_config.get('enable_ft_linvel', False)
        self.enable_priv_fingertip_angvel = p_config.get('enable_ft_angvel', False)
        self.enable_priv_hand_scale = p_config.get('enable_hand_scale', False)
        self.enable_priv_obj_restitution = p_config['enable_obj_restitution']
        self.enable_priv_tactile = p_config['enable_tactile']

        hand_asset_file = self.config['env']['asset']['handAsset']
        print(f'Hand asset file: {hand_asset_file}')
        self.num_contacts = 3 * FINGERTIP_CNT
        if not self.config['env']['privInfo']['enable_tactile']:
            self.num_contacts = 0

        # priv_info 布局（新版，移除 obj_scale）
        # 固定部分：obj_position(3) + obj_mass(1) + obj_friction(1) + obj_com(3) = 8维
        # 注意：obj_scale 已移除，因为 LinkerPen 使用固定尺寸
        self.priv_info_dict = {
            'obj_position': (0, 3),
            'obj_mass': (3, 4),
            'obj_friction': (4, 5),
            'obj_com': (5, 8),
        }
        start_index = 8  # 从索引 8 开始添加动态部分

        priv_dims = OrderedDict()
        priv_dims['obj_orientation'] = 4
        priv_dims['obj_linvel'] = 3
        priv_dims['obj_angvel'] = 3
        priv_dims['fingertip_position'] = 3 * FINGERTIP_CNT
        priv_dims['fingertip_orientation'] = 4 * FINGERTIP_CNT
        priv_dims['fingertip_linvel'] = FINGERTIP_POS_DIM
        priv_dims['fingertip_angvel'] = FINGERTIP_POS_DIM
        priv_dims['hand_scale'] = 1
        priv_dims['obj_restitution'] = 1
        priv_dims['tactile'] = self.num_contacts
        for name, dim in priv_dims.items():
            # 使用hasattr安全检查属性是否存在，避免访问已删除的未使用参数
            if hasattr(self, f'enable_priv_{name}') and eval(f'self.enable_priv_{name}'):
                self.priv_info_dict[name] = (start_index, start_index + dim) # 在这里对priv_info进行了更新，并在后面读取这个表获得priv_info_dim，传给PPO
                start_index += dim

    def _update_priv_buf(self, env_id, name, value):
        # normalize to -1, 1
        # 使用hasattr安全检查属性是否存在
        # 固定部分的启用开关（总是启用）
        self.enable_priv_obj_position = True
        self.enable_priv_obj_mass = True
        self.enable_priv_obj_friction = True
        self.enable_priv_obj_com = True
        
        if hasattr(self, f'enable_priv_{name}') and eval(f'self.enable_priv_{name}'):
            s, e = self.priv_info_dict[name]
            if type(value) is list:
                value = to_torch(value, dtype=torch.float, device=self.device)
            self.priv_info_buf[env_id, s:e] = value

    def _setup_object_info(self, o_config):
        """设置物体信息 - 简化版，仅支持 linkerpen 转笔"""
        self.object_type = o_config['type']
        raw_prob = o_config['sampleProb']
        assert (sum(raw_prob) == 1)

        print('---- Object Setup (LinkerPen) ----')
        print(f'Object type: {self.object_type}')
        
        # 简化：仅支持 cylinder_linkerpen
        self.object_type_prob = [1.0]
        self.object_type_list = ['linkerpen']
        self.asset_files_dict = {
            'linkerpen': 'assets/cylinder/linkerpen/spinning_pen.urdf',
        }
        
        print('---- Object List ----')
        print(f'using {len(self.object_type_list)} training objects: {self.object_type_list}')

    def _allocate_task_buffer(self, num_envs):
        # extra buffers for observe randomized params
        self.prop_hist_len = self.config['env']['hora']['propHistoryLen']
        self.priv_info_dim = max([v[1] for k, v in self.priv_info_dict.items()])
        self.critic_obs_dim = self.config['env']['hora']['critic_obs_dim']
        self.point_cloud_sampled_dim = self.config['env']['hora']['point_cloud_sampled_dim']
        self.point_cloud_buffer_dim = self.point_cloud_sampled_dim
        self.priv_info_buf = torch.zeros((num_envs, self.priv_info_dim), device=self.device, dtype=torch.float)
        self.critic_info_buf = torch.zeros((num_envs, self.critic_obs_dim), device=self.device, dtype=torch.float)
        # for collecting bc data
        self.point_cloud_buf = torch.zeros((num_envs, self.point_cloud_sampled_dim, 3), device=self.device, dtype=torch.float)
        # fixed noise per-episode, for different hardware have different this value
        self.random_obs_noise_e = torch.zeros((num_envs, self.config['env']['numActions']), device=self.device, dtype=torch.float)
        self.random_action_noise_e = torch.zeros((num_envs, self.config['env']['numActions']), device=self.device, dtype=torch.float)
        # ---- stage 2 buffers
        # stage 2 related buffers
        self.proprio_hist_buf = torch.zeros((num_envs, self.prop_hist_len, PROPRIO_DIM), device=self.device, dtype=torch.float)
        self.tactile_hist_buf = torch.zeros((num_envs, self.prop_hist_len, CONTACT_DIM), device=self.device, dtype=torch.float)
        # a bit unintuitive: first 4 is quaternion and last 3 is position, due to development order
        self.noisy_quaternion_buf = torch.zeros((num_envs, self.prop_hist_len, 7), device=self.device, dtype=torch.float)
        # debug and verification controls
        self.debug_shape_summary = self.config['env'].get('debug_shape_summary', False)
        self.enable_strict_dim_assertions = self.config['env'].get('enable_strict_dim_assertions', False)
        self._shape_summary_logged_once = False

        # final sanity check for priv_info layout and dimension
        try:
            total_span = 0
            for name, (s, e) in self.priv_info_dict.items():
                # basic non-overlap and ordering assumptions
                assert e > s, f"priv_info slice for {name} must have positive length"
                total_span += (e - s)
                if name == 'tactile' and self.enable_priv_tactile:
                    assert (e - s) == CONTACT_DIM, f"tactile dim mismatch: {(e - s)} vs CONTACT_DIM={CONTACT_DIM}"
                if name == 'fingertip_position' and self.enable_priv_fingertip_position:
                    assert (e - s) == FINGERTIP_POS_DIM, f"fingertip_position dim mismatch: {(e - s)} vs FINGERTIP_POS_DIM={FINGERTIP_POS_DIM}"
                if name == 'obj_restitution' and self.enable_priv_obj_restitution:
                    assert (e - s) == 1, "obj_restitution must be scalar"
            if self.enable_strict_dim_assertions:
                assert total_span == self.priv_info_dim, f"priv_info_dim mismatch: total_span={total_span} vs priv_info_dim={self.priv_info_dim}"
        except AssertionError as ae:
            print(f"[PrivInfo Verify] Assertion failed: {ae}")
            if self.enable_strict_dim_assertions:
                raise
        else:
            print(f"[PrivInfo Verify] OK: priv_info_dim={self.priv_info_dim}, total_span={total_span}, slices={len(self.priv_info_dict)}")
        
        # 打印所有启用的特权信息及其维度
        print("\n" + "="*80)
        print("启用的特权信息 (Privileged Information) 配置:")
        print("="*80)
        print(f"{'名称':<30} {'索引范围':<15} {'维度':<8} {'状态'}")
        print("-"*80)
        
        # 固定部分（始终启用）
        fixed_privs = ['obj_position', 'obj_scale', 'obj_mass', 'obj_friction', 'obj_com']
        for name in fixed_privs:
            if name in self.priv_info_dict:
                s, e = self.priv_info_dict[name]
                print(f"{name:<30} [{s:>2}:{e:>2}]{'':>8} {e-s:<8} [固定启用]")
        
        print("-"*80)
        
        # 可选部分（根据配置启用）
        optional_privs = ['obj_orientation', 'obj_linvel', 'obj_angvel', 
                         'fingertip_position', 'fingertip_orientation', 
                         'fingertip_linvel', 'fingertip_angvel', 
                         'hand_scale', 'obj_restitution', 'tactile']
        
        enabled_count = len(fixed_privs)
        disabled_list = []
        
        for name in optional_privs:
            if name in self.priv_info_dict:
                s, e = self.priv_info_dict[name]
                print(f"{name:<30} [{s:>2}:{e:>2}]{'':>8} {e-s:<8} ✓ 已启用")
                enabled_count += 1
            else:
                disabled_list.append(name)
        
        if disabled_list:
            print("-"*80)
            print("未启用的可选特权信息:")
            for name in disabled_list:
                print(f"{name:<30} {'':>15} {'':>8} ✗ 未启用")
        
        print("-"*80)
        print(f"总计: {enabled_count} 项启用, 总维度 = {total_span}")
        print(f"priv_info_dim (缓冲区大小) = {self.priv_info_dim}")
        print("="*80 + "\n")

    def _setup_reward_config(self, r_config):
        # the list
        self.reward_scale_dict = {}
        for k, v in r_config.items():
            if 'scale' in k:
                if type(v) is not omegaconf.listconfig.ListConfig:
                    v = [v, v, 0, 0]
                else:
                    assert len(v) == 4
                self.reward_scale_dict[k.replace('_scale', '')] = v
        self.angvel_clip_min = r_config['angvelClipMin']
        self.angvel_clip_max = r_config['angvelClipMax']
        self.angvel_penalty_threshold_high = r_config['angvelPenaltyThresHigh']
        self.angvel_penalty_threshold_low  = r_config['angvelPenaltyThresLow']
        
        # Gaussian kernel 角速度奖励参数
        self.use_gaussian_angvel_reward = r_config.get('use_gaussian_angvel_reward', False)
        self.target_angvel = r_config.get('target_angvel', 3.14)  # 目标角速度 (rad/s)
        self.angvel_sigma = r_config.get('angvel_sigma', 1.0)     # 高斯核带宽
        
        # [Anti-Hacking] EMA 平滑参数 (仅用于速度门控)
        self.ema_alpha = r_config.get('ema_alpha', 0.15)  # EMA 平滑系数 (较小值=更平滑)
        
        if self.use_gaussian_angvel_reward:
            print(f"\n[角速度奖励] 使用 Gaussian Kernel 模式:")
            print(f"  目标角速度: {self.target_angvel:.2f} rad/s ({self.target_angvel/(2*3.14159):.3f} Hz)")
            print(f"  高斯核带宽 σ: {self.angvel_sigma:.2f}")
            print(f"\n[Anti-Hacking] EMA 速度门控:")
            print(f"  EMA α: {self.ema_alpha:.3f}")
        else:
            print(f"\n[角速度奖励] 使用 Clip-based 模式:")
            print(f"  Clip 范围: [{self.angvel_clip_min:.2f}, {self.angvel_clip_max:.2f}]")

    def _setup_action_space_config(self, env_config):
        """
        设置动作空间配置
        
        支持两个独立的配置选项：
        1. Flying Hand: 添加 6 DoF 浮空底座
        2. 禁用无名指和小拇指: 缩减手部动作空间
        
        动作维度组合：
        ┌─────────────────────────────────────────────────────────────────┐
        │ 配置                           │ 动作维度                        │
        ├─────────────────────────────────────────────────────────────────┤
        │ Flying + 完整手指              │ 6 + 21 = 27 DoF                │
        │ Flying + 禁用 ring/little      │ 6 + 13 = 19 DoF                │
        │ 无 Flying + 完整手指           │ 21 DoF                          │
        │ 无 Flying + 禁用 ring/little   │ 13 DoF                          │
        └─────────────────────────────────────────────────────────────────┘
        
        Args:
            env_config: 环境配置字典
        """
        action_space_config = env_config.get('actionSpace', {})
        self.disable_ring_little = action_space_config.get('disableRingLittleFinger', False)
        
        # 注意：flying_hand_enabled 需要在此方法调用之前设置（由 _setup_flying_hand_config 设置）
        # 如果还未设置，默认为 False
        if not hasattr(self, 'flying_hand_enabled'):
            self.flying_hand_enabled = False
        
        # 创建动作空间映射器（包含 Flying Hand 和手指禁用的配置）
        self.action_mapper = ActionSpaceMapper(
            disable_ring_little=self.disable_ring_little,
            flying_hand_enabled=self.flying_hand_enabled
        )
        
        # 设置实际动作维度（策略网络输出维度）
        self.actual_action_dim = self.action_mapper.get_action_dim()
        
        # 打印配置信息
        self.action_mapper.print_config()

    def _setup_flying_hand_config(self, env_config):
        """
        设置 Flying Hand 配置
        
        Flying Hand 是带有 6 DoF 浮空底座的灵巧手，用于转笔任务。
        底座允许手在空间中移动和旋转，但通过严格的速度和位置限制
        防止利用惯性"作弊"（甩手腕转笔）。
        
        Args:
            env_config: 环境配置字典
        """
        flying_config = env_config.get('flyingHand', {})
        
        # 是否启用 Flying Hand
        self.flying_hand_enabled = flying_config.get('enabled', False)
        
        if self.flying_hand_enabled:
            # 从配置读取参数，使用默认值作为后备
            self.flying_default_height = flying_config.get('defaultHeight', FLYING_DEFAULT_HEIGHT)
            self.flying_height_lower = flying_config.get('heightLower', FLYING_HEIGHT_LOWER)
            self.flying_height_upper = flying_config.get('heightUpper', FLYING_HEIGHT_UPPER)
            self.flying_xy_limit = flying_config.get('xyLimit', FLYING_XY_LIMIT)
            self.flying_linear_velocity = flying_config.get('linearVelocity', 0.1)
            self.flying_angular_velocity = flying_config.get('angularVelocity', 2.0)
            # Flying base PD 增益需要足够高以实现精确位控
            # 参考 interactive_tune.py: stiffness=1000, damping=50
            self.flying_base_pgain = flying_config.get('basePgain', 1000.0)
            self.flying_base_dgain = flying_config.get('baseDgain', 50.0)
            
            # ============================================================
            # 相对限位配置 (Relative Limit Configuration)
            # ============================================================
            # 动作空间相对于初始位置的对称限位，避免初始化后动作空间不对称
            # 例如: 如果初始 Pitch = -1.31 rad，绝对限位 [-1.57, 1.57]，
            #       则正向可移动 2.88 rad，负向只能 0.26 rad（不对称）
            # 使用相对限位后: 正负方向都限制在 relative_limit 范围内
            # ============================================================
            relative_config = flying_config.get('relativeLimit', {})
            self.use_relative_limit = relative_config.get('enabled', True)  # 默认启用
            # Flying base 相对限位范围 (使用 robot_config 中的常量作为默认值)
            self.flying_relative_xy_limit = relative_config.get('xyLimit', FLYING_RELATIVE_XY_LIMIT)
            self.flying_relative_z_limit = relative_config.get('zLimit', FLYING_RELATIVE_Z_LIMIT)
            self.flying_relative_rot_limit = relative_config.get('rotLimit', FLYING_RELATIVE_ROT_LIMIT)
            
            print("\n" + "="*70)
            print("Flying Hand 配置 (6 DoF Floating Base)")
            print("="*70)
            print(f"  启用状态: {self.flying_hand_enabled}")
            print(f"  默认高度: {self.flying_default_height}m")
            print(f"  高度范围: [{self.flying_height_lower}, {self.flying_height_upper}]m")
            print(f"  XY 限制 (绝对): ±{self.flying_xy_limit}m")
            print(f"  线速度限制: {self.flying_linear_velocity} m/s")
            print(f"  角速度限制: {self.flying_angular_velocity} rad/s")
            print(f"  PD 增益: P={self.flying_base_pgain}, D={self.flying_base_dgain}")
            print("-"*70)
            print(f"  相对限位启用: {self.use_relative_limit}")
            if self.use_relative_limit:
                print(f"    XY 相对限位: ±{self.flying_relative_xy_limit}m")
                print(f"    Z 相对限位: ±{self.flying_relative_z_limit}m")
                print(f"    旋转相对限位: ±{self.flying_relative_rot_limit} rad")
            print("="*70 + "\n")
        else:
            # 设置默认值（即使未启用也需要这些属性存在）
            self.flying_default_height = FLYING_DEFAULT_HEIGHT
            self.flying_height_lower = FLYING_HEIGHT_LOWER
            self.flying_height_upper = FLYING_HEIGHT_UPPER
            self.flying_xy_limit = FLYING_XY_LIMIT
            self.flying_linear_velocity = 0.1
            self.flying_angular_velocity = 2.0
            self.flying_base_pgain = 1000.0
            self.flying_base_dgain = 50.0
            # 非 Flying Hand 模式下禁用相对限位
            self.use_relative_limit = False
            self.flying_relative_xy_limit = 0.0
            self.flying_relative_z_limit = 0.0
            self.flying_relative_rot_limit = 0.0

    def _create_object_asset(self):
        # object file to asset
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
        hand_asset_file = self.config['env']['asset']['handAsset']
        # load hand asset
        hand_asset_options = gymapi.AssetOptions()
        hand_asset_options.flip_visual_attachments = False
        # ====================================================================
        # Flying Hand 配置
        # ====================================================================
        # 对于 Flying Hand (带 6 DoF 浮空底座)：
        #   - fix_base_link = True: 固定 world_virtual 到世界坐标系
        #   - 手仍然可以通过 6 个虚拟关节在空间中移动/旋转
        # 对于普通 Hand (固定底座)：
        #   - fix_base_link = True: 固定 base_link 到世界坐标系
        # ====================================================================
        hand_asset_options.fix_base_link = True  # 始终固定最顶层的 base link
        hand_asset_options.collapse_fixed_joints = False
        hand_asset_options.disable_gravity = True
        hand_asset_options.thickness = 0.001
        hand_asset_options.angular_damping = 0.01
        # IsaacGym没有这个参数hand_asset_options.enable_self_collisions = True
        # Flying Hand 特有配置：打印加载信息
        if self.flying_hand_enabled:
            print("\n" + "="*60)
            print("Flying Hand 模式已启用")
            print("="*60)
            print(f"  URDF 文件: {hand_asset_file}")
            print(f"  总 DOF: {NUM_TOTAL_DOF_FLYING} = {NUM_FLYING_DOF} (base) + {NUM_DOF} (hand)")
            print(f"  默认高度: {self.flying_default_height}m")
            print(f"  高度范围: {self.flying_height_lower}m ~ {self.flying_height_upper}m")
            print(f"  XY 限制: ±{self.flying_xy_limit}m")
            print("="*60 + "\n")
        
        # ====================================================================
        # 碰撞几何优化配置
        # ====================================================================
        # 对于精细转笔技巧（thumb_around, triangle_pass），碰撞检测的精度和速度都很重要
        # 选项 1: convex_decomposition_from_submeshes（当前使用）
        #         - 从 mesh 子部件自动生成凸包
        #         - 精度高，但速度较慢
        # 选项 2: VHACD 凸分解（推荐用于性能优化）
        #         - 体积层次近似凸分解
        #         - 可调节精度/速度平衡
        # ====================================================================
        
        # 默认使用 submesh 凸分解（保持兼容性）
        use_vhacd = self.config['env'].get('useVHACD', False)
        
        if use_vhacd:
            # VHACD 凸分解 - 更快但精度略低
            hand_asset_options.convex_decomposition_from_submeshes = False
            hand_asset_options.vhacd_enabled = True
            hand_asset_options.vhacd_params = gymapi.VhacdParams()
            # 降低分辨率以提升速度，同时保持足够精度
            hand_asset_options.vhacd_params.resolution = 50000  # 默认 100000
            hand_asset_options.vhacd_params.max_convex_hulls = 8  # 限制每个 mesh 的凸包数
            hand_asset_options.vhacd_params.max_num_vertices_per_ch = 32  # 限制每个凸包的顶点数
            print("[Asset] Using VHACD convex decomposition for hand collision")
        else:
            # Submesh 凸分解 - 更精确
            hand_asset_options.convex_decomposition_from_submeshes = True
            print("[Asset] Using submesh convex decomposition for hand collision")
        if self.torque_control:
            hand_asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
        else:
            hand_asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        self.hand_asset = self.gym.load_asset(self.sim, asset_root, hand_asset_file, hand_asset_options)
        
        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(self.hand_asset, name) for name in
                                  FINGERTIP_LINK_NAMES]
        # 这里不通过在urdf中加入刚体作为传感器，而是直接读取指尖所受的net_force
        self.contact_sensor_names = CONTACT_LINK_NAMES 

        self.object_asset_list = []
        self.asset_point_clouds = []
        for object_type in self.object_type_list:
            object_asset_file = self.asset_files_dict[object_type]
            object_asset_options = gymapi.AssetOptions()
            # 设置 collapse_fixed_joints=True 以合并 fixed joint 连接的刚体
            object_asset_options.collapse_fixed_joints = True
            
            # If we've specified a specific eval object, we only need to load that object.
            eval_object_type = self.config['env']['object']['evalObjectType']
            if eval_object_type is not None and object_type != eval_object_type:
                self.object_asset_list.append(None)
                self.asset_point_clouds.append(None)
                continue

            object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
            self.object_asset_list.append(object_asset)
            
            # LinkerPen 几何参数（直接定义，无需从 npy 文件读取）
            # 来自 pen_gen.py: L_BODY=0.15, L_TIP=0.015×2, R_BODY_OUT=0.009
            self.pen_radius = 0.009  # 9mm 半径
            self.pen_length = 0.18   # 总长度 150mm + 15mm×2 = 180mm
            print(f"LinkerPen loaded: radius={self.pen_radius}m, length={self.pen_length}m (no scale applied)")
            
            # 点云采样（如果需要）
            if self.point_cloud_sampled_dim > 0:
                # 使用长度/直径比例进行采样
                length_ratio = self.pen_length / (self.pen_radius * 2)  # 0.18 / 0.018 = 10
                self.asset_point_clouds.append(sample_cylinder(length_ratio) * self.pen_radius * 2)

        assert any([x is not None for x in self.object_asset_list])
        # assert any([x is not None for x in self.asset_point_clouds])

    def _print_object_properties(self, env_ptr, object_handle):
        """
        打印 IsaacGym 加载后的物体完整物理属性，用于验证 URDF 读取正确性。
        包括：质量、质心、惯量、摩擦、恢复系数等所有相关量。
        """
        print("\n" + "=" * 100)
        print("IsaacGym 物体物理属性验证 (Object Physical Properties Verification)")
        print("=" * 100)
        
        # 获取刚体属性 (Rigid Body Properties)
        rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        print(f"\n--- 刚体属性 (Rigid Body Properties) ---")
        print(f"刚体数量: {len(rb_props)}")
        
        total_mass = 0.0
        for idx, prop in enumerate(rb_props):
            total_mass += prop.mass
            print(f"\n  [Link {idx}]")
            print(f"    质量 (mass):           {prop.mass:.6f} kg ({prop.mass * 1000:.2f} g)")
            print(f"    质心 (COM):            ({prop.com.x:.6f}, {prop.com.y:.6f}, {prop.com.z:.6f}) m")
            print(f"    惯量 (inertia):")
            print(f"      Ixx: {prop.inertia.x.x:.8e}, Ixy: {prop.inertia.x.y:.8e}, Ixz: {prop.inertia.x.z:.8e}")
            print(f"      Iyx: {prop.inertia.y.x:.8e}, Iyy: {prop.inertia.y.y:.8e}, Iyz: {prop.inertia.y.z:.8e}")
            print(f"      Izx: {prop.inertia.z.x:.8e}, Izy: {prop.inertia.z.y:.8e}, Izz: {prop.inertia.z.z:.8e}")
            print(f"    flags:                 {prop.flags}")
        
        print(f"\n  总质量 (Total Mass): {total_mass:.6f} kg ({total_mass * 1000:.2f} g)")
        
        # 获取形状属性 (Shape Properties) - 包含摩擦、恢复系数等
        shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
        print(f"\n--- 形状属性 (Shape Properties) ---")
        print(f"形状数量: {len(shape_props)}")
        
        for idx, prop in enumerate(shape_props):
            print(f"\n  [Shape {idx}]")
            print(f"    摩擦系数 (friction):       {prop.friction:.4f}")
            print(f"    恢复系数 (restitution):    {prop.restitution:.4f}")
            print(f"    滚动摩擦 (rolling_friction): {prop.rolling_friction:.4f}")
            print(f"    扭转摩擦 (torsion_friction): {prop.torsion_friction:.4f}")
            print(f"    接触偏移 (contact_offset):  {prop.contact_offset:.6f}")
            print(f"    静止偏移 (rest_offset):     {prop.rest_offset:.6f}")
            print(f"    合规性 (compliance):       {prop.compliance:.6f}")
            print(f"    厚度 (thickness):          {prop.thickness:.6f}")
            # 注意：filter 是碰撞过滤器
        
        # 获取 DOF 属性（如果物体有关节）
        dof_count = self.gym.get_actor_dof_count(env_ptr, object_handle)
        print(f"\n--- 自由度属性 (DOF Properties) ---")
        print(f"自由度数量: {dof_count}")
        
        if dof_count > 0:
            dof_props = self.gym.get_actor_dof_properties(env_ptr, object_handle)
            for idx in range(dof_count):
                print(f"\n  [DOF {idx}]")
                print(f"    damping: {dof_props['damping'][idx]:.6f}")
                print(f"    stiffness: {dof_props['stiffness'][idx]:.6f}")
                print(f"    friction: {dof_props['friction'][idx]:.6f}")
        
        # 打印预期值对比（来自 pen_gen.py）
        print(f"\n--- 预期值对比 (Expected from pen_gen.py) ---")
        print(f"  预期尼龙主体质量:  24.39 g")
        print(f"  预期铝头质量 (x2): 8.27 g × 2 = 16.54 g")
        print(f"  预期总质量:        40.93 g")
        print(f"  预期半径:          9 mm")
        print(f"  预期总长度:        180 mm (150 + 15×2)")


    def _parse_hand_dof_props(self):
        self.num_linker_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)
        linker_hand_dof_props = self.gym.get_asset_dof_properties(self.hand_asset)

        self.linker_hand_dof_lower_limits = []
        self.linker_hand_dof_upper_limits = []

        # 手部关节的限制（21 DoF）
        # 重要: 顺序按 IsaacGym 加载后的字母顺序: index -> little -> middle -> ring -> thumb
        # index (食指): joint0 侧摆 [-0.18, 0], joint1-3 弯曲 [-1.57, 0]
        # little (小拇指): joint0 侧摆 [0, 0.18], joint1-3 弯曲 [-1.57, 0]
        # middle (中指): joint0 锁定 [0, 0], joint1-3 弯曲 [-1.57, 0]
        # ring (无名指): joint0 侧摆 [-0.18, 0], joint1-3 弯曲 [-1.57, 0]
        # thumb (拇指): joint0 展开 [-0.61, 0.61], joint1 弯曲 [-1.43, 0], joint2-4 弯曲 [-1.57, 0]
        hand_dof_lower_limits = [
            -0.18, -1.57, -1.57, -1.57,  # index (4)
            0., -1.57, -1.57, -1.57,     # little (4)
            0., -1.57, -1.57, -1.57,     # middle (4, joint0锁定)
            -0.18, -1.57, -1.57, -1.57,  # ring (4)
            -0.61, -1.43, -1.57, -1.57, -1.57  # thumb (5)
        ]
        hand_dof_upper_limits = [
            0., 0., 0., 0.,    # index (4)
            0.18, 0., 0., 0.,  # little (4)
            0., 0., 0., 0.,    # middle (4, joint0锁定)
            0., 0., 0., 0.,    # ring (4)
            0.61, 0., 0., 0., 0.  # thumb (5)
        ]

        # ====================================================================
        # Flying Hand DOF 配置
        # ====================================================================
        # 如果启用了 Flying Hand，前 6 个 DOF 是虚拟底座关节：
        #   [0] virtual_px: 平移 X
        #   [1] virtual_py: 平移 Y
        #   [2] virtual_pz: 平移 Z
        #   [3] virtual_rx: 旋转 Roll
        #   [4] virtual_ry: 旋转 Pitch
        #   [5] virtual_rz: 旋转 Yaw
        # 后 21 个 DOF 是原始手部关节
        # ====================================================================
        
        if self.flying_hand_enabled:
            # Flying base 的限制（从 gen_flying_hand.py 中获取）
            flying_dof_lower_limits = [
                -self.flying_xy_limit,       # px: -0.05
                -self.flying_xy_limit,       # py: -0.05
                self.flying_height_lower,    # pz: 0.30
                -1.57,                       # rx: -90°
                -1.57,                       # ry: -90°
                -3.14,                       # rz: -180°
            ]
            flying_dof_upper_limits = [
                self.flying_xy_limit,        # px: 0.05
                self.flying_xy_limit,        # py: 0.05
                self.flying_height_upper,    # pz: 0.40
                1.57,                        # rx: 90°
                1.57,                        # ry: 90°
                3.14,                        # rz: 180°
            ]
            # 合并 flying base + hand DOF 限制
            all_dof_lower_limits = flying_dof_lower_limits + hand_dof_lower_limits
            all_dof_upper_limits = flying_dof_upper_limits + hand_dof_upper_limits
        else:
            all_dof_lower_limits = hand_dof_lower_limits
            all_dof_upper_limits = hand_dof_upper_limits

        for i in range(self.num_linker_hand_dofs):
            linker_hand_dof_props['lower'][i] = all_dof_lower_limits[i]
            linker_hand_dof_props['upper'][i] = all_dof_upper_limits[i]
            
            self.linker_hand_dof_lower_limits.append(linker_hand_dof_props['lower'][i])
            self.linker_hand_dof_upper_limits.append(linker_hand_dof_props['upper'][i])
            
            # Flying base 使用位置控制模式（PD 控制器）
            if self.flying_hand_enabled and i < NUM_FLYING_DOF:
                # 极高力矩限制以支持刚性位控（匹配高P增益）
                linker_hand_dof_props['effort'][i] = 100000.0
                linker_hand_dof_props['stiffness'][i] = self.flying_base_pgain
                linker_hand_dof_props['damping'][i] = self.flying_base_dgain
                linker_hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
                linker_hand_dof_props['friction'][i] = 0.0
                linker_hand_dof_props['armature'][i] = 0.001  # 极小惯量，提高响应速度
                # 设置速度限制
                if i < 3:  # 平移关节
                    linker_hand_dof_props['velocity'][i] = self.flying_linear_velocity
                else:  # 旋转关节
                    linker_hand_dof_props['velocity'][i] = self.flying_angular_velocity
            else:
                # 手部关节使用原有配置
                linker_hand_dof_props['effort'][i] = self.torque_limit
                if self.torque_control:
                    linker_hand_dof_props['stiffness'][i] = 0.
                    linker_hand_dof_props['damping'][i] = 0.
                    linker_hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
                else:
                    linker_hand_dof_props['stiffness'][i] = self.config['env']['controller']['pgain']
                    linker_hand_dof_props['damping'][i] = self.config['env']['controller']['dgain']
                linker_hand_dof_props['friction'][i] = 0.01
                linker_hand_dof_props['armature'][i] = 0.001

        self.linker_hand_dof_lower_limits = to_torch(self.linker_hand_dof_lower_limits, device=self.device)
        self.linker_hand_dof_upper_limits = to_torch(self.linker_hand_dof_upper_limits, device=self.device)
        
        # ============================================================
        # 相对限位范围张量 (Relative Limit Range Tensor)
        # ============================================================
        # 定义每个 DOF 相对于初始位置的对称移动范围
        # 最终限位 = max(absolute_lower, init - relative) ~ min(absolute_upper, init + relative)
        # ============================================================
        if self.flying_hand_enabled and self.use_relative_limit:
            # Flying base 的相对限位范围 (6 DoF)
            flying_relative_range = [
                self.flying_relative_xy_limit,   # px
                self.flying_relative_xy_limit,   # py
                self.flying_relative_z_limit,    # pz
                self.flying_relative_rot_limit,  # rx (Roll)
                self.flying_relative_rot_limit,  # ry (Pitch)
                self.flying_relative_rot_limit,  # rz (Yaw)
            ]
            # 手部关节使用较大的范围（基本不限制，由绝对限位控制）
            hand_relative_range = [10.0] * len(hand_dof_lower_limits)  # 足够大，实际由绝对限位控制
            all_relative_range = flying_relative_range + hand_relative_range
        else:
            # 不使用相对限位时，范围设为足够大（由绝对限位控制）
            all_relative_range = [10.0] * self.num_linker_hand_dofs
        
        self.dof_relative_range = to_torch(all_relative_range, device=self.device)
        
        # 打印 DOF 配置信息
        if self.flying_hand_enabled:
            print("\n" + "-"*60)
            print("Flying Hand DOF 配置:")
            print("-"*60)
            for i in range(NUM_FLYING_DOF):
                abs_lower = all_dof_lower_limits[i]
                abs_upper = all_dof_upper_limits[i]
                rel_range = all_relative_range[i]
                print(f"  [{i}] {FLYING_DOF_NAMES[i]}: 绝对[{abs_lower:.3f}, {abs_upper:.3f}], 相对±{rel_range:.3f}")
            print(f"  [6-26] 手部关节: 原有配置 (相对限位不限制)")
            print("-"*60 + "\n")
        
        return linker_hand_dof_props

    def _init_object_pose(self):
        linker_hand_start_pose = gymapi.Transform()
        
        # ====================================================================
        # 手部初始 Transform 设置
        # ====================================================================
        # Flying Hand 模式：手的位置和旋转完全由 DOF 控制（前 6 个 DOF）
        #   - DOF[0-2]: virtual_px, virtual_py, virtual_pz (位置)
        #   - DOF[3-5]: virtual_rx, virtual_ry, virtual_rz (旋转，欧拉角)
        #   - Transform 必须是单位变换，否则 DOF 控制的旋转会叠加到 Transform 上
        #   - 这与 interactive_tune.py 保持一致
        # 普通模式：从配置文件读取手部基座初始位置
        #   - handBaseInit.position: [px, py, pz]
        #   - handBaseInit.rotation: [rx, ry, rz] 欧拉角
        # ====================================================================
        if self.flying_hand_enabled:
            # Flying Hand: 使用单位变换，所有变换由 DOF 控制
            linker_hand_start_pose.p = gymapi.Vec3(0, 0, 0)
            linker_hand_start_pose.r = gymapi.Quat(0, 0, 0, 1)
        else:
            # 普通模式: 从配置读取手部基座位置
            hand_base_init = self.config['env'].get('handBaseInit', {})
            base_pos = hand_base_init.get('position', [0.0, 0.0, 0.35])
            base_rot = hand_base_init.get('rotation', [0.0, -1.31, 0.0])  # 欧拉角 [rx, ry, rz]
            
            linker_hand_start_pose.p = gymapi.Vec3(base_pos[0], base_pos[1], base_pos[2])
            
            # 将欧拉角转换为四元数 (ZYX 顺序)
            rx, ry, rz = base_rot
            quat_x = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), rx)
            quat_y = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), ry)
            quat_z = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), rz)
            # 组合旋转: Z * Y * X
            linker_hand_start_pose.r = quat_z * quat_y * quat_x
        
        # ====================================================================
        # 物体初始位置设置
        # ====================================================================
        # 注意：物体的初始位置只是占位用，实际位置在 reset_idx 中设置
        # ====================================================================
        if self.flying_hand_enabled:
            # Flying Hand: 物体放置在手的默认高度附近
            pose_dx, pose_dy, pose_dz = 0.00, -0.04, self.flying_default_height
        else:
            # 普通模式: 使用原来的设置
            pose_dx, pose_dy, pose_dz = 0.00, -0.04, 0.15
        
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = linker_hand_start_pose.p.x + pose_dx
        object_start_pose.p.y = linker_hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = linker_hand_start_pose.p.z + pose_dz

        object_start_pose.p.y = linker_hand_start_pose.p.y - 0.01
        
        # ====================================================================
        # 物体初始 Z 高度设置
        # ====================================================================
        # 说明：
        # - 对于 RL Policy：不重要。Policy 看到的永远是 reset_idx 后从 Cache 取出的状态
        # - 对于代码运行：重要。防止 create_env 时的物理碰撞报错
        # 
        # 注意：relative_z_drop_threshold 现在是相对阈值（允许下降的最大距离）
        # Termination 检查是基于 (init_object_z_buf - current_z) > threshold
        # 因此这里只需要设置一个合理的初始高度，不需要依赖 threshold 值
        # 
        # 重要：物体初始位置要避开手的初始位置，防止碰撞导致物体飞出
        # Flying Hand 模式下手悬浮在 flying_default_height，所以物体放在更高处
        # ====================================================================
        if self.flying_hand_enabled:
            # Flying Hand: 物体放在手上方 0.3m 处，避免与手碰撞
            # 手悬浮在 ~0.35m，物体放在 0.65m 安全位置
            object_start_pose.p.z = self.flying_default_height + 0.1
        else:
            # 普通模式: 物体放在手上方，使用配置中的手部高度
            hand_base_init = self.config['env'].get('handBaseInit', {})
            base_pos = hand_base_init.get('position', [0.0, 0.0, 0.35])
            object_start_pose.p.z = base_pos[2] + 0.2  # 手部 Z + 0.2m
        
        return linker_hand_start_pose, object_start_pose


def compute_hand_reward(
    object_linvel_penalty, object_linvel_penalty_scale: float,
    rotate_reward, rotate_reward_scale: float,
    torque_penalty, torque_pscale: float,
    axial_tilt_penalty, axial_tilt_penalty_scale: float,
    position_penalty, position_penalty_scale: float,
    rotate_penalty, rotate_penalty_scale: float,
    flying_base_movement_penalty, flying_base_movement_penalty_scale: float,
    waypoint_tracking_reward, waypoint_tracking_reward_scale: float,
    jitter_penalty, jitter_penalty_scale: float,
    reverse_penalty, reverse_penalty_scale: float
):
    reward = rotate_reward_scale * rotate_reward
    reward = reward + object_linvel_penalty * object_linvel_penalty_scale
    reward = reward + torque_penalty * torque_pscale
    reward = reward + axial_tilt_penalty * axial_tilt_penalty_scale
    reward = reward + position_penalty * position_penalty_scale
    reward = reward + rotate_penalty * rotate_penalty_scale
    reward = reward + flying_base_movement_penalty * flying_base_movement_penalty_scale
    reward = reward + waypoint_tracking_reward * waypoint_tracking_reward_scale
    # [Anti-Hacking] Jitter 和 Reverse 惩罚
    # jitter_penalty_scale 和 reverse_penalty_scale 应为负值
    reward = reward + jitter_penalty * jitter_penalty_scale
    reward = reward + reverse_penalty * reverse_penalty_scale
    return reward


def quat_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Adapted from PyTorch3D:
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#quaternion_to_axis_angle

    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., :3], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., 3:])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., :3] / sin_half_angles_over_angles
