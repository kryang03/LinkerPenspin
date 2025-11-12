# --------------------------------------------------------
# Lessons from Learning to Spin “Pens”
# Written by Paper Authors
# Copyright (c) 2024 All Authors
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
from typing import Optional
import torch
import omegaconf
import numpy as np
import math

from glob import glob
from collections import OrderedDict

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import quat_conjugate, quat_mul, to_torch, quat_apply, tensor_clamp, torch_rand_float, quat_from_euler_xyz

from penspin.utils.point_cloud_prep import sample_cylinder, sample_cuboid
from .base.vec_task import VecTask
from penspin.utils.misc import tprint
# Import centralized robot dimension constants
from penspin.utils.robot_config import (
    NUM_DOF,
    PROPRIO_DIM,
    CONTACT_DIM,
    FINGERTIP_CNT,
    FINGERTIP_LINK_NAMES,
    CONTACT_LINK_NAMES,
    FINGERTIP_POS_DIM,
    PRIV_FINGERTIP_ROT_DIM,
    OBS_WITH_CONTACT_FINGERTIP_DIM
)

# 刚体的 位置 (3) + 姿态 (4, 四元数) + 线速度 (3) + 角速度 (3)
RIGID_BODY_STATES = 13

# CHECKLIST
# 用于查找生成的初始化抓取状态文件夹
NUM_POSE_PER_CACHE = '3k'
# 物体平均位置
OBJ_CANON_POS = [-0.12593473494052887, 0.027405261993408203, 0.16902321577072144] # 3pose
# [-0.11722512543201447, 0.006986482068896294, 0.1717524379491806] # 4pose
# [-0.12593473494052887, 0.027405261993408203, 0.16902321577072144] # 3pose
# [-0.13216303288936615, 0.022801531478762627, 0.16341765224933624] # 6pose
WAYPOINTS = [{'hand': [0.09058664739131927, -1.1684951782226562, -0.3523995876312256, -0.5262272953987122, -0.13673382997512817, -1.0198190212249756, -0.9558137059211731, -1.2512168884277344, -3.8383831224564346e-07, -1.3491917848587036, -0.5089983940124512, -0.3240099549293518, -0.05116431415081024, -0.9247583150863647, -1.2445602416992188, -0.9097212553024292, -0.020588409155607224, -1.1557494401931763, -0.7035709023475647, -0.2180664837360382, -0.001244630548171699],
            'object': [-0.11762521415948868, 0.023072263225913048, 0.18054848909378052, 0.708649218082428, 0.021436162292957306, 0.03619061037898064, -0.7043059468269348]},
            {'hand': [0.17996101081371307, -1.2075750827789307, -0.3885318338871002, -0.4472368657588959, -0.17998147010803223, -1.030931830406189, -0.910559356212616, -1.1463258266448975, -1.9397666051190754e-07, -1.2313501834869385, -0.6338013410568237, -0.4614178240299225, -0.08347102254629135, -0.8618580102920532, -1.1939127445220947, -0.8964090943336487, 0.17496797442436218, -1.3870528936386108, -0.6740223169326782, -0.21953974664211273, -0.0016594172921031713],
            'object': [-0.10668071359395981, 0.021574808284640312, 0.17717039585113525, 0.21597789227962494, -0.6708014011383057, 0.2683640122413635, -0.6567798256874084]},
            {'hand': [0.12977235019207, -0.9924959540367126, -0.3606451451778412, -0.4871084988117218, -0.17996357381343842, -1.1815704107284546, -1.0074188709259033, -1.063124179840088, -1.3303166213063378e-07, -1.1750483512878418, -0.5964861512184143, -0.7107267379760742, -0.0033535510301589966, -0.8607388734817505, -1.2455166578292847, -0.8664445281028748, -0.02050342597067356, -1.1381360292434692, -0.7652751803398132, -0.32268959283828735, -0.03381062299013138],
            'object': [-0.1360439658164978, 0.02780270390212536, 0.16825813055038452, -0.3405953049659729, -0.6889249086380005, 0.5953956842422485, -0.23426729440689087]}]
HAND_SIMILARITY_SCALE_FACTOR = 0.5 
ORIENTATION_SIMILARITY_THRESHOLD = 0.9

# Contact上下界
CONTACT_THRESH = 0.02
TACTILE_FORCE_MAX = 4.0
# Reward 缩放比例
REWARD_SCALE_DICT = {
    'obj_linvel_penalty': 1.0,
    'rotate_reward': 0.7,
    'waypoint_sparse_reward': 100,
    'torque_penalty': 0.1,
    'work_penalty': 0.05,
    'pencil_z_dist_penalty': 3.0,
    'position_penalty': 20000.0,
    'rotate_penalty': 4.0,
    'hand_pose_consistency_penalty': 80.0
}
#    由于Python 3.7+ dict 保持插入顺序，.values() 返回的视图中的值顺序将与定义时一致
REWARD_SCALE = list(REWARD_SCALE_DICT.values())

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

        # 这个参数暂未使用
        self.obs_with_binary_contact = config['env']['obs_with_binary_contact']
        self.base_obj_scale = config['env']['baseObjScale']
        # print("缩放比例", self.base_obj_scale)

        self.save_init_pose = config['env']['genGrasps'] #这个参数是为了在genGrasp步骤生成初始化抓取状态

        self.aggregate_mode = self.config['env']['aggregateMode']
        self.up_axis = 'z'
        self.rotation_axis = config['env']['rotation_axis']
        self.reset_z_threshold = self.config['env']['reset_height_threshold']
        self.grasp_cache_name = self.config['env']['grasp_cache_name']
        self.canonical_pose_category = config['env']['genGraspCategory']
        self.num_pose_per_cache = NUM_POSE_PER_CACHE
        self.with_camera = config['env']['enableCameraSensors']
        self.enable_obj_ends = config['env']['enable_obj_ends']
        self.init_pose_mode = config['env']['initPoseMode']
        self.num_linker_hand_dofs = self.config['env']['numActions']
        # Important: map CUDA device IDs to Vulkan ones.
        graphics_device_id = 0

        super().__init__(config, sim_device, graphics_device_id, headless)

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
        self.contact_thresh = torch.zeros((self.num_envs, self.num_contacts), dtype=torch.float, device=self.device)

        if self.randomize_scale and self.scale_list_init:
            self.saved_grasping_states = {}
            for s in self.randomize_scale_list:
                cache_name = '_'.join([self.grasp_cache_name, 'grasp', self.canonical_pose_category,
                                       self.num_pose_per_cache, f's{str(s).replace(".", "")}'])
                cache_name_tmp = '/'.join([self.grasp_cache_name, self.canonical_pose_category,
                                           f's{str(s).replace(".", "")}_{self.num_pose_per_cache}'])
                print(cache_name_tmp)
                if os.path.exists(f'cache/{cache_name_tmp}.npy'):
                    self.saved_grasping_states[str(s)] = torch.from_numpy(np.load(f'cache/{cache_name_tmp}.npy')).float().to(self.device)
                    print(cache_name_tmp)
                else:
                    self.saved_grasping_states[str(s)] = torch.from_numpy(np.load(f'cache/{cache_name}.npy')).float().to(self.device)
                    print(cache_name)
        else:
            assert self.save_init_pose

        self.rot_axis_buf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.rot_axis_task = None
        sign, axis = self.rotation_axis[0], self.rotation_axis[1]
        axis_index = ['x', 'y', 'z'].index(axis)
        self.rot_axis_buf[:, axis_index] = 1
        self.rot_axis_buf[:, axis_index] = -self.rot_axis_buf[:, axis_index] if sign == '-' else self.rot_axis_buf[:, axis_index]

        # useful buffers
        self.init_pose_buf = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        # there is an extra dim [self.control_freq_inv] because we want to get a mean over multiple control steps
        self.torques = torch.zeros((self.num_envs, self.control_freq_inv, self.num_actions), device=self.device, dtype=torch.float)
        self.dof_vel_finite_diff = torch.zeros((self.num_envs, self.control_freq_inv, self.num_dofs), device=self.device, dtype=torch.float)

        # --- calculate velocity at control frequency instead of simulated frequency
        self.object_pos_prev = self.object_pos.clone()
        self.object_rot_prev = self.object_rot.clone()
        self.ft_pos_prev = self.fingertip_pos.clone()
        self.ft_rot_prev = self.fingertip_orientation.clone()
        self.dof_vel_prev = self.dof_vel_finite_diff.clone()

        self.obj_linvel_at_cf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.obj_angvel_at_cf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.ft_linvel_at_cf = torch.zeros((self.num_envs, FINGERTIP_CNT * 3), device=self.device, dtype=torch.float)
        self.ft_angvel_at_cf = torch.zeros((self.num_envs, FINGERTIP_CNT * 3), device=self.device, dtype=torch.float)
        self.dof_acc = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        # ----

        assert type(self.p_gain) in [int, float] and type(self.d_gain) in [int, float], 'assume p_gain and d_gain are only scalars'
        self.p_gain = torch.ones((self.num_envs, self.num_actions), device=self.device, dtype=torch.float) * self.p_gain
        self.d_gain = torch.ones((self.num_envs, self.num_actions), device=self.device, dtype=torch.float) * self.d_gain

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

        # 初始化为全0，表示所有环境开始时都没有达成任何参考帧
        self.waypoint_achievement_mask = torch.zeros(
            (self.num_envs, len(WAYPOINTS)), dtype=torch.bool, device=self.device
        )

        # 添加终止原因记录计数器
        self.termination_counts = {
            'max_episode_length': 0,
            'object_below_threshold': 0,
            'pencil_fall': 0,
            'total_episodes': 0
        }
        
        # 为每个环境记录当前episode的终止原因
        self.current_termination_reason = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # 0: 未终止, 1: max_episode_length, 2: object_below_threshold, 3: pencil_fall

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

        for i in range(num_envs):
            tprint(f'{i} / {num_envs} num_envs')
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies * 20, max_agg_shapes * 20, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            hand_actor = self.gym.create_actor(env_ptr, self.hand_asset, hand_pose, 'hand', i, -1, 1)
            self.gym.set_actor_dof_properties(env_ptr, hand_actor, linker_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)
            self.hand_actors.append(hand_actor)

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
            self.obj_scale = self.base_obj_scale
            # Modified，为了躲避scale随机化，这里注释掉

            # if self.randomize_scale:
            #     num_scales = len(self.randomize_scale_list)
            #     self.obj_scale = np.random.uniform(
            #         self.randomize_scale_list[i % num_scales] - 0.025,
            #         self.randomize_scale_list[i % num_scales] + 0.025
            #     )
            self.gym.set_actor_scale(env_ptr, object_handle, self.obj_scale)
            self._update_priv_buf(env_id=i, name='obj_scale', value=self.obj_scale)
            # print("缩放比例", self.obj_scale)

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

            obj_friction = 1.0
            obj_restitution = 0.0  # default is 0
            # TODO: bad engineering because of urgent modification
            if self.randomize_friction:
                rand_friction = np.random.uniform(self.randomize_friction_lower, self.randomize_friction_upper)
                obj_restitution = np.random.uniform(0, 1)

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

            if self.randomize_mass:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                for p in prop:
                    p.mass = np.random.uniform(self.randomize_mass_lower, self.randomize_mass_upper)
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
                self._update_priv_buf(env_id=i, name='obj_mass', value=prop[0].mass)
            else:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                self._update_priv_buf(env_id=i, name='obj_mass', value=prop[0].mass)

            if self.point_cloud_sampled_dim > 0:
                self.obj_point_clouds.append(self.asset_point_clouds[object_type_id] * self.obj_scale)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # for training record, visualized in tensorboard
            if self.with_camera:
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

    def reset_idx(self, env_ids):
        if self.randomize_pd_gains:
            self.p_gain[env_ids] = torch_rand_float(
                self.randomize_p_gain_lower, self.randomize_p_gain_upper, (len(env_ids), self.num_actions),
                device=self.device).squeeze(1)
            self.d_gain[env_ids] = torch_rand_float(
                self.randomize_d_gain_lower, self.randomize_d_gain_upper, (len(env_ids), self.num_actions),
                device=self.device).squeeze(1)

        self.random_obs_noise_e[env_ids] = torch.normal(0, self.random_obs_noise_e_scale, size=(len(env_ids), self.num_dofs), device=self.device, dtype=torch.float)
        self.random_action_noise_e[env_ids] = torch.normal(0, self.random_action_noise_e_scale, size=(len(env_ids), self.num_dofs), device=self.device, dtype=torch.float)
        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        num_scales = len(self.randomize_scale_list)
        for n_s in range(num_scales):
            s_ids = env_ids[(env_ids % num_scales == n_s).nonzero(as_tuple=False).squeeze(-1)]
            if len(s_ids) == 0:
                continue
            obj_scale = self.randomize_scale_list[n_s]
            scale_key = str(obj_scale)
            # single object (category) case:
            sampled_pose_idx = np.random.randint(self.saved_grasping_states[scale_key].shape[0], size=len(s_ids))
            sampled_pose = self.saved_grasping_states[scale_key][sampled_pose_idx].clone()
            object_pose_noise = torch.normal(0, self.random_pose_noise, size=(sampled_pose.shape[0], 7), device=self.device, dtype=torch.float)
            object_pose_noise[:, 3:] = 0  # disable rotation noise
            self.root_state_tensor[self.object_indices[s_ids], :7] = sampled_pose[:, self.num_linker_hand_dofs:] + object_pose_noise
            self.root_state_tensor[self.object_indices[s_ids], 7:RIGID_BODY_STATES] = 0
            pos = sampled_pose[:, :self.num_linker_hand_dofs]
            self.linker_hand_dof_pos[s_ids, :] = pos
            self.linker_hand_dof_vel[s_ids, :] = 0
            self.prev_targets[s_ids, :self.num_linker_hand_dofs] = pos
            self.cur_targets[s_ids, :self.num_linker_hand_dofs] = pos
            self.init_pose_buf[s_ids, :] = pos

        object_indices = torch.unique(self.object_indices[env_ids]).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(object_indices), len(object_indices))
        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        if not self.torque_control:
            self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets), gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        # reset tactile
        self.contact_thresh[env_ids] = CONTACT_THRESH
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

        self.waypoint_achievement_mask[env_ids, :] = False

    def compute_observations(self):
        """
        计算当前时间步的环境观测（Observations）。
        这个方法是RL环境中每个时间步的核心部分，用于生成Agent的输入。
        """
        self._refresh_gym()
        # 从仿真器刷新最新的物理状态，确保后续读取的数据是最新的
        # 具体实现依赖于所使用的仿真后端（Isaac Gym）

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
        # 注意: self.contact_thresh 现在预期是与展平后的触觉数据形状兼容的张量
        # 例如，如果 num_envs = N, num_sensor_handles = 5，那么 self.contact_thresh 预期形状是 (N, 15)

        if self.config['env']['privInfo']['enable_tactile']:
            # 获取并克隆当前的总接触力张量，形状 (num_envs, num_bodies, 3)
            contacts = self.contact_forces.clone()

            # 提取与触觉传感器句柄对应的接触力，形状变为 (num_envs, num_sensor_handles, 3)
            contacts_at_sensors = contacts[:, self.sensor_handle_indices, :]

            # **修改点 1: 先展平 contacts_at_sensors，然后进行分量阈值比较**
            # 展平提取到的三维力向量，形状变为 (num_envs, num_sensor_handles * 3)
            # 这是为了与形状为 (num_envs, num_sensor_handles * 3) 的 self.contact_thresh 进行逐元素比较
            flattened_contacts = torch.flatten(contacts_at_sensors, start_dim=1) # 形状: (N, M*3)

            # **修改点 2: 基于每个力分量和对应的阈值进行筛选**
            # 创建一个布尔型掩码：flattened_contacts 中每个分量是否 >= self.contact_thresh 中对应分量的阈值
            # 这里的 self.contact_thresh 预期形状与 flattened_contacts 兼容，例如 (num_envs, num_sensor_handles * 3)
            component_threshold_mask = (flattened_contacts >= self.contact_thresh) # 形状: (N, M*3)

            # 应用分量阈值掩码：如果分量小于其对应的阈值，则将其置零
            # 将布尔掩码转换为浮点型进行乘法
            filtered_flattened_forces = flattened_contacts * component_threshold_mask.float() # 形状: (N, M*3)
            # 防止除以零或缩放参考值过小的情况 ，TACTILE_FORCE_MAX是预期的最大力大小，用于将分量大致映射到 [-1, 1]
            if TACTILE_FORCE_MAX <= 1e-6:
                # 如果最大力参考值接近零，所有非零的力都应该变得非常大或无穷大
                # 实际处理中，如果参考值接近零，且过滤后的力也接近零，结果就是零
                scaled_flattened_forces = torch.zeros_like(filtered_flattened_forces)
            else:
                # 对过滤后的展平力向量的每个分量应用缩放
                # 范围 [-TACTILE_FORCE_MAX, TACTILE_FORCE_MAX] 大致映射到 [-1, 1]
                scaled_flattened_forces = filtered_flattened_forces / TACTILE_FORCE_MAX
                # 限制缩放后的值在 [-1, 1] 范围内，提高数值鲁棒性
                scaled_flattened_forces = torch.clamp(scaled_flattened_forces, -1.0, 1.0)

            # 最终 agent 感知到的触觉信息是缩放后的展平三维力向量
            # self.sensed_contacts 的形状是 (num_envs, num_sensor_handles * 3)
            # 其中包含 num_sensor_handles 个传感器的 Fx, Fy, Fz 值依次排列，小于对应分量阈值的已置零
            self.sensed_contacts = scaled_flattened_forces

            # 如果启用了可视化，复制感知数据到 CPU 用于调试显示
            if self.viewer:
                # debug_contacts 的形状是 (num_envs, num_sensor_handles, 3)
                # 表现的是每个分力在被TACTILE_FORCE_MAX缩放之前的状态
                self.debug_contacts = filtered_flattened_forces.reshape(self.num_envs, -1, 3).detach().cpu().numpy() 
       
        # --------------------------------------------------------------
        # 3. 物体端点跟踪 (Object End Points Tracking)
        # --------------------------------------------------------------
        
        # 定义物体的本地坐标系下的端点位置（例如铅笔的头部和尾部）
        # 这里假设物体中心在本地坐标系原点，长度为 pen_length * obj_scale
        pencil_ends = [
            [0, 0, -(self.pen_length/2) * self.obj_scale],
            [0, 0, (self.pen_length/2) * self.obj_scale]
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
        pencil_end_1 += (torch.rand(pencil_end_1.shape[0], 3).to(self.device) - 0.5) * (self.pen_radius*2)
        pencil_end_2 += (torch.rand(pencil_end_2.shape[0], 3).to(self.device) - 0.5) * (self.pen_radius*2)

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

        t_buf = (self.obs_buf_lag_history[:, -3:, :PROPRIO_DIM].reshape(self.num_envs, -1)).clone()
                # t_buf shape: torch.Size([1, 3*42])

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
        self.obs_buf_lag_history[at_reset_env_ids, :, self.num_linker_hand_dofs:PROPRIO_DIM] = self.init_pose_buf[at_reset_env_ids].unsqueeze(1)
        # 对于重置的环境，用当前的物体端点位置填充其历史缓冲区
        self.obj_ends_history[at_reset_env_ids, :, :] = cur_obj_ends[at_reset_env_ids]

        # 如果启用了触觉，则对于重置的环境，将其观测历史缓冲区中触觉相关部分清零
        # 范围是 PROPRIO_DIM 到 PROPRIO_DIM+CONTACT_DIM
        if self.config['env']['privInfo']['enable_tactile']:
            self.obs_buf_lag_history[at_reset_env_ids, :, PROPRIO_DIM:PROPRIO_DIM+CONTACT_DIM] = torch.zeros((len(at_reset_env_ids),80,CONTACT_DIM),device=self.device)
            # 对于重置的环境，用当前的指尖位置填充其观测历史缓冲区中指尖位置部分
            # 范围是 PROPRIO_DIM+CONTACT_DIM 到 PROPRIO_DIM+CONTACT_DIM+FINGERTIP_POS_DIM
            self.obs_buf_lag_history[at_reset_env_ids, :, PROPRIO_DIM+CONTACT_DIM:PROPRIO_DIM+CONTACT_DIM+FINGERTIP_POS_DIM] = self.fingertip_pos[at_reset_env_ids].unsqueeze(1)

        # 重置相关的速度信息缓冲
        # 在接触或碰撞时记录的物体和指尖线速度/角速度，在重置时也需要清零或用当前值填充
        self.obj_linvel_at_cf[at_reset_env_ids] = self.object_linvel[at_reset_env_ids]
        self.obj_angvel_at_cf[at_reset_env_ids] = self.object_angvel[at_reset_env_ids]
        self.ft_linvel_at_cf[at_reset_env_ids] = self.fingertip_linvel[at_reset_env_ids]
        self.ft_angvel_at_cf[at_reset_env_ids] = self.fingertip_angvel[at_reset_env_ids]

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
        # 范围是 0 到 PROPRIO_DIM，因为只使用关节位置和目标位置
        self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.prop_hist_len:, :PROPRIO_DIM]  # [1, 30, PROPRIO_DIM] - 示例形状
        # 如果启用了触觉，从主观测历史中提取最近 prop_hist_len 个时间步的触觉信息
        # 范围是 PROPRIO_DIM 到 PROPRIO_DIM + CONTACT_DIM
        if self.config['env']['privInfo']['enable_tactile']:
            self.tactile_hist_buf[:] = self.obs_buf_lag_history[:, -self.prop_hist_len:, PROPRIO_DIM:PROPRIO_DIM + CONTACT_DIM]

        # --------------------------------------------------------------
        # 7. 更新私有信息缓冲区 (Update Privileged Information Buffer)
        # --------------------------------------------------------------
        # 私有信息是只有训练时可用的“地面真值”信息，不包含噪声和延迟，用于辅助训练
        # _update_priv_buf 是一个辅助方法，用于更新私有信息字典或缓冲区
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
        # Critic的完整输入通常是：agent的观测 + 私有信息 + critic特有的信息
        # deprecated - 这行注释可能表示下面的 critic_info_buf 构建方式是旧的或即将废弃
        self.critic_info_buf[:, 0:4] = self.object_rot # 物体真实姿态（四元数）
        self.critic_info_buf[:, 4:7] = self.obj_linvel_at_cf # 物体在接触时的真实线速度
        self.critic_info_buf[:, 7:10] = self.obj_angvel_at_cf # 物体在接触时的真实角速度
        # 3,7,11,15,20 are fingertip indexes for rigid body states,即self.fingertip_handles
        # RIGID_BODY_STATES：每个刚体的状态向量包含 位置 (3) + 姿态 (4, 四元数) + 线速度 (3) + 角速度 (3) = 3+4+3+3=13 个分量
        # self.rigid_body_states 形状可能是 (num_envs, num_bodies, RIGID_BODY_STATES)
        # 提取指尖的刚体状态
        fingertip_states = self.rigid_body_states[:, self.fingertip_handles].clone()
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

    def _compute_waypoint_reward(self):
        """
        计算并返回当前时间步基于路径点（waypoint）的稀疏奖励。

        奖励机制设计核心：
        1. 奖励是稀疏的：仅在环境“新”达成一个路径点时才可能触发。
        2. 奖励是连续的：奖励大小与姿态和手部位置的相似度成正比。
        3. 豁免初始状态：对于初始状态（满足的第一个路径点），不会给予奖励。
        4. 支持周期任务：当一个环境完成所有路径点后，除最近完成的的所有路径点状态会被重置以开始下一轮。
        """
        # 初始化一个零张量，用于存储本步中每个环境获得的稀疏奖励
        # shape: [num_envs]
        waypoint_sparse_reward_this_step = torch.zeros(self.num_envs, device=self.device)

        # 遍历每一个预先定义的路径点（waypoint）
        for wp_idx, wp_data in enumerate(WAYPOINTS):

            # === 步骤 1 & 2: 筛选姿态合格的环境 ===
            # 目标: 找到所有当前物体姿态与当前路径点目标姿态足够接近的环境。

            # 从路径点数据中提取目标物体姿态（四元数），并增加一个维度以进行广播计算
            # shape: [1, 4]
            wp_obj_rot = to_torch(wp_data['object'][3:7], device=self.device).unsqueeze(0)

            # 计算当前所有环境的物体姿态与目标姿态的相似度。
            # 对于四元数 q 和 -q 代表相同旋转，因此我们用点积的绝对值来衡量相似性。
            # 相似度范围 [0, 1]，值越接近 1 表示姿态越接近。
            # self.object_rot shape: [num_envs, 4], wp_obj_rot shape: [1, 4] -> 广播后乘积 shape: [num_envs, 4]
            # shape: [num_envs]
            orientation_similarity = torch.abs(torch.sum(self.object_rot * wp_obj_rot, dim=1))

            # 创建一个布尔掩码，标记那些姿态相似度超过阈值的环境
            # shape: [num_envs]
            orientation_eligible_mask = (orientation_similarity > ORIENTATION_SIMILARITY_THRESHOLD)

            # 优化：如果没有任何环境的姿态合格，则直接跳到下一个路径点的检查
            if not torch.any(orientation_eligible_mask):
                continue

            # 获取所有姿态合格环境的索引
            # shape: [num_orientation_eligible], 其中 num_orientation_eligible <= num_envs
            orientation_eligible_indices = torch.where(orientation_eligible_mask)[0]

            # === 步骤 3: 计算手部姿态相似度 ===
            # 目标: 对于姿态合格的环境，进一步计算它们当前的手部姿态与目标姿态的相似度。

            num_orientation_eligible = len(orientation_eligible_indices)

            # 提取路径点定义的目标手部姿态，并扩展以匹配姿态合格环境的数量
            # to_torch(...) shape: [N] -> unsqueeze(0) shape: [1, N] -> expand(...) shape: [num_orientation_eligible, N]
            wp_hand_pos = to_torch(wp_data['hand'], device=self.device).unsqueeze(0).expand(num_orientation_eligible, -1)

            # 筛选出姿态合格环境的当前手部姿态
            # shape: [num_orientation_eligible, N]
            eligible_hand_pos = self.linker_hand_dof_pos[orientation_eligible_indices]

            # 计算当前手部姿态与目标姿态之间的欧氏距离
            # (eligible_hand_pos - wp_hand_pos) shape: [num_orientation_eligible, N]
            # shape: [num_orientation_eligible]
            hand_pos_diff = torch.norm(eligible_hand_pos - wp_hand_pos, dim=1)

            # 使用指数衰减函数将距离转换为相似度分数，范围 (0, 1]
            # shape: [num_orientation_eligible]
            hand_similarity = torch.exp(-hand_pos_diff / HAND_SIMILARITY_SCALE_FACTOR)

            # === 步骤 4: 识别“新达成”的路径点并更新状态 ===
            # 目标: 在姿态合格的环境中，找到那些在本轮周期内首次达成此路径点的环境，并立即更新它们的成就状态。

            # 检查在这些姿态合格的环境中，哪些在成就掩码中仍为 False，即“新达成”
            # self.waypoint_achievement_mask[orientation_eligible_indices, wp_idx] shape: [num_orientation_eligible]
            # shape: [num_orientation_eligible] (布尔类型)
            is_wp_idx_newly_achieved_mask = (self.waypoint_achievement_mask[orientation_eligible_indices, wp_idx] == False)

            # 优化：如果在姿态合格的环境中，没有一个是新达成的，则跳到下一个路径点
            if not torch.any(is_wp_idx_newly_achieved_mask):
                continue

            # 获取所有“新达成”此路径点的环境的全局索引
            # shape: [num_newly_achieved], 其中 num_newly_achieved <= num_orientation_eligible
            newly_achieved_indices = orientation_eligible_indices[is_wp_idx_newly_achieved_mask]

            # === 步骤 5: 筛选并计算应得的奖励 ===
            # 目标: 在“新达成”的环境中，排除掉那些处于“初始状态”的环境，然后为剩下的环境计算并累加奖励。

            # a. 检查新达成的环境是否是首次达成 *任何* 路径点
            # 计算在达成此 wp_idx 之前，每个环境已达成的路径点总数，若某个环境已达成的路径点总数为0，它不应该被给予奖励。
            # self.waypoint_achievement_mask[newly_achieved_indices, :] shape: [num_newly_achieved, num_waypoints]
            # torch.sum(...) shape: [num_newly_achieved]
            # shape: [num_newly_achieved] (布尔类型)
            is_initial_achievement_mask = (torch.sum(self.waypoint_achievement_mask[newly_achieved_indices, :].int(), dim=1) == 0) 

            # b. 筛选出真正应该给予奖励的环境（即非初始状态达成的环境）
            # shape: [num_newly_achieved] (布尔类型)
            should_grant_reward_mask = ~is_initial_achievement_mask

            # 【关键逻辑】无论后续是否给予奖励，只要是新达成的，就必须立刻更新其状态掩码。
            # 防止一直停滞在初始状态
            self.waypoint_achievement_mask[newly_achieved_indices, wp_idx] = True

            if torch.any(should_grant_reward_mask):
                # 获取最终应该被奖励的环境的全局索引
                # shape: [num_to_reward], 其中 num_to_reward <= num_newly_achieved
                final_grant_reward_indices = newly_achieved_indices[should_grant_reward_mask]

                # --- 代码可读性优化 ---
                # 从 `hand_similarity` 中筛选出需要奖励的部分
                # 原来的实现虽然功能正确，但可读性较差，这里分解为更清晰的步骤：
                # 1. 先从所有姿态合格的环境中，筛选出“新达成”环境的手部相似度
                # shape: [num_newly_achieved]
                hand_sim_for_newly_achieved = hand_similarity[is_wp_idx_newly_achieved_mask]
                # 2. 再从“新达成”的环境中，筛选出“应该被奖励”的环境的手部相似度
                # shape: [num_to_reward]
                hand_sim_for_reward = hand_sim_for_newly_achieved[should_grant_reward_mask]
                # --- 结束优化 ---

                # 同样地，筛选出对应环境的姿态相似度分数
                # shape: [num_to_reward]
                orientation_sim_for_reward = orientation_similarity[final_grant_reward_indices]

                # 计算最终奖励值：姿态相似度 * 手部姿态相似度
                # shape: [num_to_reward]
                reward_values = orientation_sim_for_reward * hand_sim_for_reward

                # 将计算出的奖励累加到对应环境的奖励张量中
                waypoint_sparse_reward_this_step[final_grant_reward_indices] += reward_values

            # === 步骤 6: 周期性重置 ===
            # 目标: 检查是否有环境完成了所有路径点，如果有，则重置其成就掩码以开始新的周期。
            # 这个检查应该对所有“新达成”的环境进行。
            for env_k_idx in newly_achieved_indices:
                # 如果一个环境的成就掩码全部为 True，说明它已完成一轮任务
                if torch.all(self.waypoint_achievement_mask[env_k_idx, :]):
                    # 重置该环境的所有成就记录为 False
                    self.waypoint_achievement_mask[env_k_idx, :] = False
                    # 【关键逻辑】将当前刚刚达成的路径点重新设置为 True。
                    # 这可以防止智能体在重置后的下一步因为还停留在当前状态而立即获得一次“作弊”的奖励。
                    # 它确保了任务周期的无缝衔接。
                    self.waypoint_achievement_mask[env_k_idx, wp_idx] = True

        return waypoint_sparse_reward_this_step

    def compute_reward(self, actions):
        # 计算waypoint稀疏奖励
        waypoint_sparse_reward = torch.zeros(self.num_envs, device=self.device)
        # work and torque penalty
        torque_penalty = (self.torques[:, -1] ** 2).sum(-1)
        work_penalty = (((torch.abs(self.torques[:, -1]) * torch.abs(self.dof_vel_finite_diff[:, -1])).sum(-1)) ** 2)
        # Compute offset in radians. Radians -> radians / sec
        angdiff = quat_to_axis_angle(quat_mul(self.object_rot, quat_conjugate(self.object_rot_prev)))
        object_angvel = angdiff / (self.control_freq_inv * self.dt)
        # vec_dot > 0 表示与期望方向一致的旋转
        # vec_dot < 0 表示与期望方向相反的旋转 (逆向旋转)   
        vec_dot = (object_angvel * self.rot_axis_buf).sum(-1)
        # 奖励只针对与期望方向一致的旋转，且不超过设定的最大速度
        rotate_reward = torch.clip(vec_dot, max=self.angvel_clip_max, min=0.0)
        # vec_dot 的值低于 self.angvel_clip_min或高于self.angvel_penalty_threshold时，才会开始施加惩罚
        penalty_overspeed = torch.relu(vec_dot - self.angvel_penalty_threshold)
        penalty_reverse_rotation = torch.relu(self.angvel_clip_min-vec_dot)
        rotate_penalty = penalty_overspeed + penalty_reverse_rotation
        # 累计旋转角度（用于统计圈数）
        rot_angle = torch.abs(vec_dot) * self.dt * self.control_freq_inv # 当前时间步旋转角度
        self.total_rot_angle += rot_angle
        # linear velocity: use position difference instead of self.object_linvel
        object_linvel = ((self.object_pos - self.object_pos_prev) / (self.control_freq_inv * self.dt)).clone()
        object_linvel_penalty = torch.norm(object_linvel, p=1, dim=-1)
        # TODO: move this to a more appropriate place
        self.obj_angvel_at_cf = object_angvel
        self.obj_linvel_at_cf = object_linvel
        # 对旋转四元数的乘法
        ft_angdiff = quat_to_axis_angle(quat_mul(self.fingertip_orientation.reshape(-1, 4), quat_conjugate(self.ft_rot_prev.reshape(-1, 4)))).reshape(-1, 3*FINGERTIP_CNT)
        self.ft_angvel_at_cf = ft_angdiff / (self.control_freq_inv * self.dt)
        self.ft_linvel_at_cf = ((self.fingertip_pos - self.ft_pos_prev) / (self.control_freq_inv * self.dt))
        # 惩罚物体的最高点-最低点的高度差
        if self.point_cloud_sampled_dim > 0:
            point_cloud_z = self.point_cloud_buf[:, :self.point_cloud_sampled_dim, -1]
            z_dist_penalty = point_cloud_z.max(axis=1)[0] - point_cloud_z.min(axis=1)[0]
            z_dist_penalty[z_dist_penalty <= 0.03] = 0
        else:
            z_dist_penalty = to_torch([0], device=self.device)

        # penalize large deviation of cube
        # position_penalty = (self.object_pos[:, 0] - OBJ_CANON_POS[0]) ** 2 + (self.object_pos[:, 1] - OBJ_CANON_POS[1]) ** 2 \
        #     + (self.object_pos[:, 2] - OBJ_CANON_POS[2]) ** 2
        position_penalty = (self.object_pos[:, 2] - OBJ_CANON_POS[2]) ** 2
        # finger obj deviation penalty
        finger_obj_penalty = ((self.fingertip_pos - self.object_pos.repeat(1, FINGERTIP_CNT)) ** 2).sum(-1)

        # 新增奖励：当每圈旋转180-360度时，根据手部姿态与初始状态的差异给出惩罚
        hand_pose_consistency_penalty = self._compute_hand_pose_consistency_penalty()

        self.rew_buf[:] = compute_hand_reward(
            object_linvel_penalty, self._get_reward_scale_by_name('obj_linvel_penalty')*REWARD_SCALE[0],
            rotate_reward, self._get_reward_scale_by_name('rotate_reward')*REWARD_SCALE[1],
            waypoint_sparse_reward, self._get_reward_scale_by_name('waypoint_sparse_reward')*REWARD_SCALE[2],
            torque_penalty, self._get_reward_scale_by_name('torque_penalty')*REWARD_SCALE[3],
            work_penalty, self._get_reward_scale_by_name('work_penalty')*REWARD_SCALE[4],
            z_dist_penalty, self._get_reward_scale_by_name('pencil_z_dist_penalty')*REWARD_SCALE[5],
            position_penalty, self._get_reward_scale_by_name('position_penalty')*REWARD_SCALE[6],
            rotate_penalty, self._get_reward_scale_by_name('rotate_penalty')*REWARD_SCALE[7],
            hand_pose_consistency_penalty, self._get_reward_scale_by_name('hand_pose_consistency_penalty')*REWARD_SCALE[8]
        )
        self.reset_buf[:] = self.check_termination(self.object_pos)
        
        #mean都是对envs维度，compute_reward 函数本身计算的是当前这个时间步获得的即时奖励
        #PPO中奖励的累加实现在ppo.py的play_step()中，self.current_rewards += rewards
        #Tensorboard的奖励累加实现在mean_rewards = self.episode_rewards.get_mean()，mean也是对env维度

        # extras部分是传入ppo中的infos
        # _get_reward_scale_by_name()要更改 configs/task/LinkerHandHora.yaml中的scale
        self.extras['timestep_reward_sum'] = self.rew_buf.mean() # rew_buf 已经是各项加权求和后的总奖励，所以这里不变
        self.extras['penalty/object_linvel_penalty'] = (object_linvel_penalty * self._get_reward_scale_by_name('obj_linvel_penalty') * REWARD_SCALE[0]).mean()
        self.extras['rotation_reward'] = (rotate_reward * self._get_reward_scale_by_name('rotate_reward') * REWARD_SCALE[1]).mean()
        self.extras['penalty/torques'] = (torque_penalty * self._get_reward_scale_by_name('torque_penalty') * REWARD_SCALE[3]).mean()
        self.extras['penalty/work_done'] = (work_penalty * self._get_reward_scale_by_name('work_penalty') * REWARD_SCALE[4]).mean()
        self.extras['penalty/z_dist_penalty'] = (z_dist_penalty * self._get_reward_scale_by_name('pencil_z_dist_penalty') * REWARD_SCALE[5]).mean()
        self.extras['penalty/object_position_penalty'] = (position_penalty * self._get_reward_scale_by_name('position_penalty') * REWARD_SCALE[6]).mean()
        self.extras['penalty/rotate_penalty'] = (rotate_penalty * self._get_reward_scale_by_name('rotate_penalty') * REWARD_SCALE[7]).mean()
        self.extras['penalty/hand_pose_consistency_penalty'] = (hand_pose_consistency_penalty * self._get_reward_scale_by_name('hand_pose_consistency_penalty') * REWARD_SCALE[8]).mean()
        self.extras['finger_obj_penalty(NOT USED)'] = finger_obj_penalty.mean()
        self.extras['vel/roll_angvel'] = torch.abs(object_angvel[:, 0]).mean()
        self.extras['vel/pitch_angvel'] = torch.abs(object_angvel[:, 1]).mean()
        self.extras['vel/yaw_angvel(NEED)'] = torch.abs(object_angvel[:, 2]).mean()
        # sparse，不能在每个时间步直接对所有环境取平均值
        self.extras['rot_angle'] = rot_angle
        self.extras['reward/waypoint_sparse_reward'] = waypoint_sparse_reward * self._get_reward_scale_by_name('waypoint_sparse_reward') * REWARD_SCALE[2]
        
        # 添加终止原因统计信息

        self.extras['termination/total_episodes'] = self.termination_counts['total_episodes']
        self.extras['termination/max_episode_length_count'] = self.termination_counts['max_episode_length']
        self.extras['termination/object_below_threshold_count'] = self.termination_counts['object_below_threshold']
        self.extras['termination/pencil_fall_count'] = self.termination_counts['pencil_fall']

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

    def _compute_hand_pose_consistency_penalty(self):
        """
        计算手部姿态一致性惩罚。
        当每圈旋转180-360度时，根据当前手部姿态与初始手部姿态的差异给出惩罚。
        
        Returns:
            hand_pose_consistency_penalty: 形状为 (num_envs,) 的张量，表示每个环境的手部姿态一致性惩罚
        """
        # 计算当前累计旋转角度在一个完整圈内的相对位置
        # total_rot_angle 是累计旋转角度，取模得到当前圈内的旋转角度
        angle_in_circle = torch.fmod(self.total_rot_angle, 2 * math.pi)
        
        # 判断是否在惩罚区间内（180-360度，即 π 到 2π 弧度）
        pi = math.pi
        penalty_mask = (angle_in_circle >= pi) & (angle_in_circle <= 2 * pi)
        
        # 计算权重：在180-360度区间内，权重从0线性增长到1
        # 当角度为π时权重为0，当角度为2π时权重为1
        weight = torch.zeros_like(angle_in_circle)

        # 使用一个平滑的凸函数（如二次函数）来计算权重，使得在180度时权重为0，360度时权重为1
        # 并且在180-360度区间内平滑增长
        normalized_angle = (angle_in_circle - pi) / pi
        weight[penalty_mask] = normalized_angle[penalty_mask] ** 2
        
        # 计算当前手部姿态与初始手部姿态的差异
        # self.init_pose_buf 存储了每个环境的初始手部姿态
        # self.linker_hand_dof_pos 是当前的手部关节位置
        hand_pose_diff = torch.norm(self.linker_hand_dof_pos - self.init_pose_buf, dim=1)

        # 应用权重，只在指定角度范围内给出惩罚
        hand_pose_consistency_penalty = weight * hand_pose_diff
        
        return hand_pose_consistency_penalty

    def post_physics_step(self):
        self.progress_buf += 1
        self.reset_buf[:] = 0
        self._refresh_gym()
        # cur* but need for reward is here
        self.compute_reward(self.actions)

        #这里的env_ids是指当前处于重置状态的环境实例的索引
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0: # 对于处于重置状态的环境实例
            # 在演示状态下输出圈数
            if self.viewer:
                for eid in env_ids:
                    turns = float(self.total_rot_angle[eid].item()) / (2 * math.pi)
                    print(f"[演示] 环境{eid}本轮累计转笔圈数: {turns:.2f}")
            self.total_rot_angle[env_ids] = 0.0
            self.reset_idx(env_ids)
        self.compute_observations()

        self.debug_viz = False
        # 仿真 Viewer（也就是说是否打开了图形界面）
        if self.viewer and self.config['env']['privInfo']['enable_tactile']:
            for env in range(len(self.envs)):
                for i, contact_idx in enumerate(list(self.sensor_handle_indices)):

                    # 用于Viewer界面的可视化，对于所有sensor_handles，可视化接触力范数 > CONTACT_THRESH的刚体
                    # debug_contacts表现的是每个分力在被TACTILE_FORCE_MAX缩放之前的状态
                    contact_norm = np.linalg.norm(self.debug_contacts[env, i])
                    if contact_norm > CONTACT_THRESH:
                        fx = self.debug_contacts[env, i, 0]
                        fy = self.debug_contacts[env, i, 1]
                        fz = self.debug_contacts[env, i, 2]
                        print(f"Fx: {fx:.4f}, Fy: {fy:.4f}, Fz: {fz:.4f}, Norm: {contact_norm:.4f}")
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
        self.actions = actions.clone().to(self.device)
        targets = self.prev_targets + self.action_scale * self.actions
        # targets = self.actions.clone()
        self.cur_targets[:] = tensor_clamp(targets, self.linker_hand_dof_lower_limits, self.linker_hand_dof_upper_limits)
        # get prev* buffer here
        self.prev_targets[:] = self.cur_targets
        self.object_rot_prev[:] = self.object_rot
        self.object_pos_prev[:] = self.object_pos
        self.ft_rot_prev[:] = self.fingertip_orientation
        self.ft_pos_prev[:] = self.fingertip_pos
        self.dof_vel_prev[:] = self.dof_vel_finite_diff

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
        if self.torque_control:
            dof_pos = self.linker_hand_dof_pos
            dof_vel = (dof_pos - previous_dof_pos) / self.dt
            self.dof_vel_finite_diff[:, step_id] = dof_vel.clone()
            torques = self.p_gain * (noise_action - dof_pos) - self.d_gain * dof_vel
            torques = torch.clip(torques, -self.torque_limit, self.torque_limit).clone()
            self.torques[:, step_id] = torques
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
        # default option
        reset_z = torch.less(object_pos[:, -1], self.reset_z_threshold)
        resets = reset_z
        
        # 重置终止原因记录
        self.current_termination_reason.fill_(0)
        
        # 记录各种终止原因
        term_by_max_eps_envs = torch.where(term_by_max_eps)[0]
        reset_z_envs = torch.where(reset_z)[0]
        
        # 设置终止原因 (优先级: 物体低于阈值 > 达到最大长度)
        self.current_termination_reason[reset_z_envs] = 2  # object_below_threshold
        self.current_termination_reason[term_by_max_eps_envs] = 1  # max_episode_length
        
        resets = torch.logical_or(resets, term_by_max_eps)

        if self.canonical_pose_category == 'pencil':
            pencil_ends = [
                [0, 0, -(self.pen_length/2) * self.obj_scale],
                [0, 0, (self.pen_length/2) * self.obj_scale]
            ]
            pencil_end_1 = self.object_pos + quat_apply(self.object_rot, to_torch(pencil_ends[0], device=self.device)[None].repeat(self.num_envs, 1))
            pencil_end_2 = self.object_pos + quat_apply(self.object_rot, to_torch(pencil_ends[1], device=self.device)[None].repeat(self.num_envs, 1))
            pencil_z_min = torch.min(pencil_end_1, pencil_end_2)[:, -1]
            pencil_z_max = torch.max(pencil_end_1, pencil_end_2)[:, -1]
            # Modified
            pencil_fall = torch.greater(pencil_z_max - pencil_z_min, 0.05)

            # 记录铅笔掉落的环境 (最高优先级)
            pencil_fall_envs = torch.where(pencil_fall)[0]
            self.current_termination_reason[pencil_fall_envs] = 3  # pencil_fall
            
            resets = torch.logical_or(resets, pencil_fall)

        # 统计终止原因
        reset_envs = torch.where(resets)[0]
        for env_id in reset_envs:
            reason = self.current_termination_reason[env_id].item()
            if reason == 1:
                self.termination_counts['max_episode_length'] += 1
            elif reason == 2:
                self.termination_counts['object_below_threshold'] += 1
            elif reason == 3:
                self.termination_counts['pencil_fall'] += 1
            self.termination_counts['total_episodes'] += 1
        print(self.termination_counts)
        return resets

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
        self.fingertip_pos = self.fingertip_states[:, :, :3].reshape(self.num_envs, -1)
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
        self.enable_priv_obj_position = p_config['enableObjPos']
        self.enable_priv_obj_mass = p_config['enableObjMass']
        self.enable_priv_obj_scale = p_config['enableObjScale']
        self.enable_priv_obj_com = p_config['enableObjCOM']
        self.enable_priv_obj_friction = p_config['enableObjFriction']
        self.contact_input_dim = p_config['contact_input_dim']
        self.contact_form = p_config['contact_form']
        self.contact_input = p_config['contact_input']
        self.contact_binarize_threshold = p_config['contact_binarize_threshold']
        self.enable_priv_obj_orientation = p_config['enable_obj_orientation']
        self.enable_priv_obj_linvel = p_config['enable_obj_linvel']
        self.enable_priv_obj_angvel = p_config['enable_obj_angvel']
        self.enable_priv_fingertip_position = p_config['enable_ft_pos']
        self.enable_priv_fingertip_orientation = p_config['enable_ft_orientation']
        self.enable_priv_fingertip_linvel = p_config['enable_ft_linvel']
        self.enable_priv_fingertip_angvel = p_config['enable_ft_angvel']
        self.enable_priv_hand_scale = p_config['enable_hand_scale']
        self.enable_priv_obj_restitution = p_config['enable_obj_restitution']
        self.enable_priv_tactile = p_config['enable_tactile']

        hand_asset_file = self.config['env']['asset']['handAsset']
        if hand_asset_file == "assets/round_tip/allegro_hand_right_fsr_round_dense.urdf":
            self.num_contacts = 5 * FINGERTIP_CNT + 12
        else:# if hand_asset_file == "assets/round_tip/allegro_hand_right_fsr_round.urdf":
            self.num_contacts = 3 * FINGERTIP_CNT
        if not self.config['env']['privInfo']['enable_tactile']:
            self.num_contacts = 0

        self.priv_info_dict = {
            'obj_position': (0, 3),
            'obj_scale': (3, 4),
            'obj_mass': (4, 5),
            'obj_friction': (5, 6),
            'obj_com': (6, 9),
        }
        start_index = 9

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
            if eval(f'self.enable_priv_{name}'):
                self.priv_info_dict[name] = (start_index, start_index + dim) # 在这里对priv_info进行了更新，并在后面读取这个表获得priv_info_dim，传给PPO
                start_index += dim

    def _update_priv_buf(self, env_id, name, value):
        # normalize to -1, 1
        if eval(f'self.enable_priv_{name}'):
            s, e = self.priv_info_dict[name]
            if type(value) is list:
                value = to_torch(value, dtype=torch.float, device=self.device)
            self.priv_info_buf[env_id, s:e] = value

    def _setup_object_info(self, o_config):
        self.object_type = o_config['type']
        raw_prob = o_config['sampleProb']
        assert (sum(raw_prob) == 1)

        primitive_list = self.object_type.split('+')
        print('---- Primitive List ----')
        print(primitive_list)
        self.object_type_prob = []
        self.object_type_list = []
        self.asset_files_dict = {
            'simple_tennis_ball': 'assets/ball.urdf',
            'simple_cube': 'assets/cube.urdf',
            'simple_cylin4cube': 'assets/cylinder4cube.urdf',
        }
        for p_id, prim in enumerate(primitive_list):
            if 'cuboid' in prim:
                subset_name = self.object_type.split('_')[-1]
                cuboids = sorted(glob(f'assets/cuboid/{subset_name}/*.urdf'))
                cuboid_list = [f'cuboid_{i}' for i in range(len(cuboids))]
                self.object_type_list += cuboid_list
                for i, name in enumerate(cuboids):
                    self.asset_files_dict[f'cuboid_{i}'] = name.replace('../assets/', '')
                self.object_type_prob += [raw_prob[p_id] / len(cuboid_list) for _ in cuboid_list]
            elif 'cylinder' in prim:
                subset_name = self.object_type.split('_')[-1]
                cylinders = sorted(glob(f'assets/cylinder/{subset_name}/*.urdf'))
                cylinder_list = [f'cylinder_{i}' for i in range(len(cylinders))]
                self.object_type_list += cylinder_list
                for i, name in enumerate(cylinders):
                    self.asset_files_dict[f'cylinder_{i}'] = name.replace('../assets/', '')
                self.object_type_prob += [raw_prob[p_id] / len(cylinder_list) for _ in cylinder_list]
            else:
                self.object_type_list += [prim]
                self.object_type_prob += [raw_prob[p_id]]
        print('---- Object List ----')
        print(f'using {len(self.object_type_list)} training objects')
        assert (len(self.object_type_list) == len(self.object_type_prob))

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
            print(f"[PrivInfo Verify] OK: priv_info_dim={self.priv_info_dim}, slices={len(self.priv_info_dict)}")

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
        self.angvel_penalty_threshold = r_config['angvelPenaltyThres']

    def _create_object_asset(self):
        # object file to asset
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
        hand_asset_file = self.config['env']['asset']['handAsset']
        # load hand asset
        hand_asset_options = gymapi.AssetOptions()
        hand_asset_options.flip_visual_attachments = False
        hand_asset_options.fix_base_link = True
        hand_asset_options.collapse_fixed_joints = False
        hand_asset_options.disable_gravity = True
        hand_asset_options.thickness = 0.001
        hand_asset_options.angular_damping = 0.01
        hand_asset_options.convex_decomposition_from_submeshes = True
        # 这里需要用Vhacd让碰撞体积更精确
        # hand_asset_options.vhacd_enabled = True
        # hand_asset_options.vhacd_params = gymapi.VhacdParams()
        # hand_asset_options.vhacd_params.resolution = 100000
        if self.torque_control:
            hand_asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
        else:
            hand_asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        self.hand_asset = self.gym.load_asset(self.sim, asset_root, hand_asset_file, hand_asset_options)
        
        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(self.hand_asset, name) for name in
                                  FINGERTIP_LINK_NAMES]
        # 这里不通过在urdf中加入刚体作为传感器，而是直接读取指尖所受的net_force
        self.contact_sensor_names = CONTACT_LINK_NAMES 
        # urdf中，<link name="link_1.0_fsr">:代表了一个 FSR (Force-Sensitive Resistor) 触觉传感器物理部分
        # self.contact_sensor_names = ["link_1.0_fsr", "link_2.0_fsr", "link_5.0_fsr",
        #                              "link_6.0_fsr", "link_9.0_fsr", "link_10.0_fsr",
        #                              "link_14.0_fsr", "link_15.0_fsr", "link_0.0_fsr", 
        #                              "link_4.0_fsr", "link_8.0_fsr", "link_13.0_fsr"]
        # for tip_name in ['3.0', '15.0', '7.0', '11.0']:
        #     # 5*num_linker_dofs的指尖传感器
        #     if hand_asset_file == "assets/round_tip/linker_hand_right_fsr_round_dense.urdf":
        #         tip_fsr_range = [2, 5, 8, 11, 13]
        #     else:
        #         tip_fsr_range = []
        #     for i in tip_fsr_range:
        #         self.contact_sensor_names.append("link_{}_tip_fsr_{}".format(tip_name, str(i)))

        # load object asset
        self.object_asset_list = []
        self.asset_point_clouds = []
        for object_type in self.object_type_list:
            object_asset_file = self.asset_files_dict[object_type]
            object_asset_options = gymapi.AssetOptions()
            # If we've specified a specific eval object, we only need to load that object.
            eval_object_type = self.config['env']['object']['evalObjectType']
            if eval_object_type is not None and object_type != eval_object_type:
                self.object_asset_list.append(None)
                self.asset_point_clouds.append(None)
                continue

            object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
            self.object_asset_list.append(object_asset)
            if 'cylin4cube' in object_type and self.point_cloud_sampled_dim > 0:
                pcs = sample_cylinder(1) * 0.08
                pcs[:, :2] *= 1.2
                self.asset_point_clouds.append(pcs)
            else:
                if 'cylinder' in object_type:
                    # dim 0 is cylinder height, 1 = 0.08m
                    # dim 1,2 are cylinder diameter, 1 = 0.08m [radius 4cm] 
                    size_info = np.load(os.path.join(asset_root, object_asset_file.replace('.urdf', '.npy')))[0]
                    self.pen_radius = size_info[1]
                    self.pen_length = size_info[0] * (size_info[1] * 2)
                    print("loading", os.path.join(asset_root, object_asset_file.replace('.urdf', '.npy')), "radius", self.pen_radius, "length", self.pen_length,"BEFORE scale")
                    if self.point_cloud_sampled_dim > 0:
                        self.asset_point_clouds.append(sample_cylinder(size_info[0]) * self.pen_radius * 2)
                elif ('cube' in object_type or 'cuboid' in object_type) and self.point_cloud_sampled_dim > 0:
                    size_info = np.load(os.path.join(asset_root, object_asset_file.replace('.urdf', '.npy')))[0]
                    self.asset_point_clouds.append(sample_cuboid(size_info[0] * 0.08, size_info[1] * 0.08, size_info[2] * 0.08))

        assert any([x is not None for x in self.object_asset_list])
        # assert any([x is not None for x in self.asset_point_clouds])

    def _parse_hand_dof_props(self):
        self.num_linker_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)
        linker_hand_dof_props = self.gym.get_asset_dof_properties(self.hand_asset)

        self.linker_hand_dof_lower_limits = []
        self.linker_hand_dof_upper_limits = []

        for i in range(self.num_linker_hand_dofs):
            # another option, just do it for now, parse directly from Nvidia's Calibrated Value
            # avoid frequently or adding another URDF
            linker_hand_dof_lower_limits = [0., -1.57, -1.57, -1.57, -0.18, -1.57, -1.57, -1.57, 0., -1.57, -1.57, -1.57, -0.18, -1.57, -1.57, -1.57, -0.61, -1.43, -1.57, -1.57, -1.57]
            linker_hand_dof_upper_limits = [0.18, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.61, 0., 0., 0., 0.]
            linker_hand_dof_props['lower'][i] = linker_hand_dof_lower_limits[i]
            linker_hand_dof_props['upper'][i] = linker_hand_dof_upper_limits[i]
            
            self.linker_hand_dof_lower_limits.append(linker_hand_dof_props['lower'][i])
            self.linker_hand_dof_upper_limits.append(linker_hand_dof_props['upper'][i])
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
        return linker_hand_dof_props

    def _init_object_pose(self):
        linker_hand_start_pose = gymapi.Transform()
        
        linker_hand_start_pose.p = gymapi.Vec3(0, 0, 0)
        linker_hand_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), math.radians(-75))
        # TODO 为什么要这样？待验证。这应该也只是加载物体，真正初始化在
        pose_dx, pose_dy, pose_dz = 0.00, -0.04, 0.15
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = linker_hand_start_pose.p.x
        object_start_pose.p.x = linker_hand_start_pose.p.x + pose_dx
        object_start_pose.p.y = linker_hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = linker_hand_start_pose.p.z + pose_dz

        object_start_pose.p.y = linker_hand_start_pose.p.y - 0.01
        # TODO: this weird thing is an unknown issue
        if self.save_init_pose:
            object_start_pose.p.z = (self.reset_z_threshold + 0.015)
        else:
            object_start_pose.p.z = (self.reset_z_threshold + 0.005)
        return linker_hand_start_pose, object_start_pose


def compute_hand_reward(
    object_linvel_penalty, object_linvel_penalty_scale: float,
    rotate_reward, rotate_reward_scale: float,
    waypoint_sparse_reward, waypoint_sparse_reward_scale: float,
    torque_penalty, torque_pscale: float,
    work_penalty, work_pscale: float,
    z_dist_penalty, z_dist_penalty_scale: float,
    position_penalty, position_penalty_scale: float,
    rotate_penalty, rotate_penalty_scale: float,
    hand_pose_consistency_penalty, hand_pose_consistency_penalty_scale: float
):
    reward = rotate_reward_scale * rotate_reward
    reward = reward + object_linvel_penalty * object_linvel_penalty_scale
    reward = reward + waypoint_sparse_reward * waypoint_sparse_reward_scale
    reward = reward + torque_penalty * torque_pscale
    reward = reward + work_penalty * work_pscale
    reward = reward + z_dist_penalty * z_dist_penalty_scale
    reward = reward + position_penalty * position_penalty_scale
    reward = reward + rotate_penalty * rotate_penalty_scale
    reward = reward + hand_pose_consistency_penalty * hand_pose_consistency_penalty_scale
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
