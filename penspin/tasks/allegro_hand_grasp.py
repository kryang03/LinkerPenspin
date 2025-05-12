# --------------------------------------------------------
# Lessons from Learning to Spin “Pens”
# Written by Paper Authors
# Copyright (c) 2024 All Authors
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import os  # 导入操作系统接口模块，用于文件路径操作
import torch  # 导入 PyTorch 库
import numpy as np  # 导入 NumPy 库，用于数值计算
from isaacgym import gymtorch  # 导入 Isaac Gym PyTorch 辅助库
from isaacgym.torch_utils import torch_rand_float, to_torch, quat_apply  # 从 Isaac Gym 的 PyTorch 工具中导入特定函数
from penspin.tasks.allegro_hand_hora import AllegroHandHora  # 从 penspin 项目中导入 AllegroHandHora 基类


# 定义 AllegroHandGrasp 类，继承自 AllegroHandHora
# 这个类专门用于实现 Allegro Hand 的抓取生成任务
class AllegroHandGrasp(AllegroHandHora):
    # 初始化函数
    def __init__(self, config, sim_device, graphics_device_id, headless):
        # 调用父类 AllegroHandHora 的初始化方法
        super().__init__(config, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)
        
        # 初始化一个空的 Tensor，用于存储成功生成的抓取状态（手部关节角度 + 物体位姿）
        # 维度为 (0, 23)，其中 23 = 16 (手部DOF) + 7 (物体位姿: 3位置 + 4四元数)
        self.saved_grasping_states = torch.zeros((0, 23), dtype=torch.float, device=self.device)

        # 定义一个字典，存储不同物体类别的"标准"抓取姿态
        # 这些姿态作为生成抓取时的初始参考点或平均姿态
        self.canonical_pose_dict = {
            # 针对铅笔 ('pencil') 类别的抓取姿态
            # 由于铅笔的特性（例如旋转对称性），初始化可能更困难，因此定义了多种标准姿态
            'pencil': [
                # 标准姿态 1: 手部关节角度 (16个) 和 物体位姿 (7个)
                {'hand': [0.0344, 1.0537, 0.5184, 0.2920, 1.4615, 0.2352, 0.7243, -0.0436,
                          0.0974, 1.1978, 0.3126, 0.2979, 0.1340, 1.0228, 0.6625, 0.2327],
                 'object': [-0.01, -0.01, 0.62, 0.0000, -0.7071, -0.0000,  0.7071]},  # 描述: 拇指 | 食指 + 中指 + 无名指
                # 标准姿态 2
                {'hand': [-0.0578, 0.8993, 0.7449, 0.4767, 1.3089, 0.5699, 0.7037, 0.3857,
                          0.0299, 1.2735, 0.1415, 0.2507, 0.1457, 1.0319, 1.0376, 0.3283],
                 'object': [-0.01, -0.01, 0.62, 0.0000, -0.7071, -0.0000,  0.7071]},  # 描述: 拇指 | 中指 + 无名指
                # 标准姿态 3
                {'hand': [0.0031, 1.0689, 0.8180, 0.1929, 1.2988, 0.7557, 0.5802, 0.3827,
                          0.0077, 1.2703, 0.1356, 0.3160, 0.1375, 1.3213, 0.5802, 0.2302],
                 'object': [-0.01, -0.01, 0.62, 0.2706, -0.6533, 0.2706, 0.6533]},  # 描述: 拇指 + 食指 | 中指 + 无名指
                # 标准姿态 4
                {'hand': [-0.1146, 1.1154, 0.4979, 0.1732, 0.8928, 1.1395, 0.5224, 1.1077,
                          -0.1250, 1.2618, 0.1522, 0.7024, 0.0226, 1.2307, 0.7128, 0.1524],
                 'object': [-0.01, -0.01, 0.62, 0.5, -0.5, 0.5, 0.5]},  # 描述: 食指 | 中指
                # 标准姿态 5
                {'hand': [-0.0804, 1.3213, 0.3743, 0.0876, 1.2907, 0.7441, 0.4251, 0.3748,
                          -0.1412, 1.3882, 0.4888, 0.1689, -0.0802, 1.2987, 0.6577, 0.1455],
                 'object': [-0.01, -0.01, 0.62, 0.653281, -0.270598, 0.653281, 0.270598]},  # 描述: 食指 | 拇指 + 中指
                # 标准姿态 6
                {'hand': [-0.1346, 1.1048, 0.5053, 0.1029, 1.2654, 0.6378, 0.8787, 0.0611,
                          -0.0573, 1.0228, 1.1360, 0.2587, 0.0253, 1.0228, 1.0845, 0.3828],
                 'object': [-0.01, -0.01, 0.62, 0.0000, -0.7071, -0.0000,  0.7071]},  # 描述: 拇指 | 食指
            ]}

        # 根据配置选择当前使用的标准姿态类别 只支持pencil
        self.canonical_pose = self.canonical_pose_dict[self.canonical_pose_category]
        # 初始化一个 Tensor，用于存储每个环境采样得到的初始姿态（手+物体）
        self.sampled_init_pose = torch.zeros((len(self.envs), 23), dtype=torch.float, device=self.device)

        
        # 定义铅笔的两个端点相对于其中心的局部坐标
        # 0.76 可能是铅笔的长度， self.base_obj_scale 是物体的缩放比例
        self.pencil_ends = [
            [0, 0, -0.76 / 2 * self.base_obj_scale],
            [0, 0, 0.76 / 2 * self.base_obj_scale]
        ]
        # 初始化重置步骤计数器
        self.reset_steps = 0

    # 重置指定环境的函数
    def reset_idx(self, env_ids):
        # 如果配置了随机化 PD 控制器增益
        if self.randomize_pd_gains:
            # 为指定环境随机生成 P 增益
            self.p_gain[env_ids] = torch_rand_float(
                self.randomize_p_gain_lower, self.randomize_p_gain_upper, (len(env_ids), self.num_actions),
                device=self.device).squeeze(1)
            # 为指定环境随机生成 D 增益
            self.d_gain[env_ids] = torch_rand_float(
                self.randomize_d_gain_lower, self.randomize_d_gain_upper, (len(env_ids), self.num_actions),
                device=self.device).squeeze(1)

        # 生成随机浮点数，用于后续的姿态随机化
        # 维度: (环境数量, 手部DOF*2 + 5)，具体用途看后续代码
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_allegro_hand_dofs * 2 + 5), device=self.device)

        # 重置指定环境的刚体力为 0
        self.rb_forces[env_ids, :, :] = 0.0
        # 判断这些重置的环境是否是因为达到了最大 episode 长度而成功的
        success = self.progress_buf[env_ids] == self.max_episode_length
        # 获取当前这些环境的手部关节角度和物体位姿
        all_states = torch.cat([
            self.allegro_hand_dof_pos, self.root_state_tensor[self.object_indices, :7]
        ], dim=1)

        # 将成功完成 episode 的环境的最终状态 (抓取姿态) 保存到 saved_grasping_states 中
        self.saved_grasping_states = torch.cat([self.saved_grasping_states, all_states[env_ids][success]])
        # 打印当前收集到的抓取姿态数量
        print('current cache size:', self.saved_grasping_states.shape[0])

        # 从配置中读取每个缓存文件期望包含的姿态数量 (例如 '50k' -> 50000)
        pose_threshold = int(eval(self.num_pose_per_cache[:-1]) * 1e3)
        # 每隔 200 个重置步骤，保存一次当前的抓取状态缓存 (用于备份或中间结果)
        if self.reset_steps % 200 == 0:
            # 构建缓存文件保存路径
            cache_dir = '/'.join(['cache', self.grasp_cache_name, self.canonical_pose_category])
            os.makedirs(cache_dir, exist_ok=True) # 确保目录存在
            # 构建缓存文件名，包含物体缩放比例和当前已保存姿态的数量
            cache_name = f's{str(self.base_obj_scale).replace(".", "")}_{len(self.saved_grasping_states)}'
            cache_name = f'{cache_dir}/{cache_name}.npy'
            # 保存 Tensor (需转到 CPU 并转为 NumPy 数组)
            np.save(cache_name, self.saved_grasping_states.cpu().numpy())
        # 如果收集到的抓取姿态数量达到了预设阈值
        if len(self.saved_grasping_states) >= pose_threshold:
            # 构建最终的缓存文件保存路径
            cache_dir = '/'.join(['cache', self.grasp_cache_name, self.canonical_pose_category])
            os.makedirs(cache_dir, exist_ok=True)
            # 构建最终缓存文件名，包含物体缩放比例和目标姿态数量 (e.g., '50k')
            cache_name = f's{str(self.base_obj_scale).replace(".", "")}_{self.num_pose_per_cache}'
            cache_name = f'{cache_dir}/{cache_name}.npy'
            # 保存达到阈值数量的抓取姿态
            np.save(cache_name, self.saved_grasping_states[:pose_threshold].cpu().numpy())
            # 退出程序，因为抓取生成任务已完成
            exit()

        # --- 重置物体状态 ---
        # 将指定环境的物体状态重置为初始状态
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        # 确保 x, y 坐标和高度 (up_axis_idx) 被正确重置
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, 0:2]
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[env_ids, self.up_axis_idx]

        # --- 设置随机化幅度 ---
        # 根据物体类别确定手部姿态的随机化幅度
        hand_randomize_amount = {
            'pencil': 0.05, 'thinpencil': 0.05,
        }[self.canonical_pose_category]
        # 根据物体类别确定物体位置的随机化幅度 [x, y, z]
        obj_randomize_amount = {
            'pencil': [0.0, 0.0, 0.0], 'thinpencil': [0.01, 0.01, 0.0],
        }[self.canonical_pose_category]

        # --- 初始化手和物体姿态 ---
        # 为每个需要重置的环境随机选择一个标准姿态的索引
        pose_ids = np.random.randint(0, len(self.canonical_pose), size=len(env_ids))
        # 获取选定的标准手部姿态，并转换为 Tensor
        hand_pose = to_torch([self.canonical_pose[pose_id]['hand'] for pose_id in pose_ids], device=self.device)
        # 应用随机扰动到手部姿态
        hand_pose += hand_randomize_amount * rand_floats[:, 5:5 + self.num_allegro_hand_dofs]
        # 获取选定的标准物体位姿，并转换为 Tensor
        object_pose = to_torch([self.canonical_pose[pose_id]['object'] for pose_id in pose_ids], device=self.device)
        # 将标准物体位姿设置给对应环境的物体
        self.root_state_tensor[self.object_indices[env_ids], 0:7] = object_pose[:, 0:7]
        # 应用随机扰动到物体位置 (x, y, z)
        for i in range(3):
            self.root_state_tensor[self.object_indices[env_ids], i] += (obj_randomize_amount[i] * torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device))[:, 0]
        # 将采样得到的初始手部和物体姿态存储起来 (用于可能的调试或分析)
        self.sampled_init_pose[env_ids] = torch.cat([hand_pose, self.root_state_tensor[self.object_indices[env_ids], :7]], dim=-1)

        # 将物体的线速度和角速度重置为 0
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])
        # 获取需要更新状态的物体 actor 的索引 (去重)
        object_indices = torch.unique(self.object_indices[env_ids]).to(torch.int32)
        # 将更新后的物体根状态张量应用到仿真环境中
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # --- 初始化手部状态 ---
        # 设置手部关节的初始位置
        self.allegro_hand_dof_pos[env_ids, :] = hand_pose
        # 设置手部关节的初始速度为 0
        self.allegro_hand_dof_vel[env_ids, :] = 0
        # 初始化上一帧和当前帧的控制目标为初始手部姿态
        self.prev_targets[env_ids, :self.num_allegro_hand_dofs] = hand_pose
        self.cur_targets[env_ids, :self.num_allegro_hand_dofs] = hand_pose

        # 获取需要更新状态的手部 actor 的索引
        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        # 如果不是力矩控制模式 (即位置控制)
        if not self.torque_control:
            # 将初始的手部姿态设置为 DOF 位置控制目标
            self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets), gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        # 将包含初始位置和速度的 DOF 状态张量应用到仿真环境中
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        # --- 重置其他状态缓冲区 ---
        # 重置环境的进度计数器
        self.progress_buf[env_ids] = 0
        # 重置观测缓冲区 (虽然这里设为0，但通常会在之后立刻计算新的观测)
        self.obs_buf[env_ids] = 0
        # 重置施加在刚体上的力
        self.rb_forces[env_ids] = 0
        # 重置特权信息缓冲区 (如果使用的话)
        self.priv_info_buf[env_ids, 0:3] = 0
        # 重置本体感觉历史缓冲区 (如果使用的话)
        self.proprio_hist_buf[env_ids] = 0

        # 标记这些环境刚刚被重置
        self.at_reset_buf[env_ids] = 1
        # 增加重置步骤计数
        self.reset_steps += 1

    # 计算奖励函数 (在这个抓取生成任务中，主要用于判断是否满足抓取条件，触发重置)
    def compute_reward(self, actions):
        # 内部辅助函数，用于计算特定环境中有多少个指尖与物体发生接触
        def list_intersect(li, hash_num):
            # 物体的刚体索引通常是最后一个
            obj_id = self.rigid_body_states.shape[1] - 1
            # 构建查询列表: [物体ID * hash + 指尖1ID, 物体ID * hash + 指尖2ID, ...]
            # hash_num 用于将两个 ID 组合成一个唯一的数字
            query_list = [obj_id * hash_num + self.fingertip_handles[0], obj_id * hash_num + self.fingertip_handles[1], obj_id * hash_num + self.fingertip_handles[2], obj_id * hash_num + self.fingertip_handles[3]]
            # 计算查询列表与实际接触对列表 (li) 的交集大小，即接触数量
            return len(np.intersect1d(query_list, li))

        # 断言检查，确保奖励计算在 CPU 上进行 (get_env_rigid_contacts 通常是 CPU 操作)
        assert self.device == 'cpu'
        # 获取每个环境中的刚体接触信息
        contacts = [self.gym.get_env_rigid_contacts(env) for env in self.envs]
        # 对每个环境，计算指尖与物体的接触数量
        # np.unique([c[2] * 10000 + c[3] for c in contact]) 创建接触对的唯一哈希值列表
        contact_list = [list_intersect(np.unique([c[2] * 10000 + c[3] for c in contact]), 10000) for contact in contacts]
        # 将接触数量列表转换为 PyTorch Tensor
        contact_condition = to_torch(contact_list, device=self.device)

        # 获取物体的位置
        obj_pos = self.rigid_body_states[:, [-1], :3]
        # 获取所有指尖的位置
        finger_pos = self.rigid_body_states[:, self.fingertip_handles, :3]

        # --- 定义抓取成功的条件 ---
        # 条件 1: 所有指尖都靠近物体 (距离小于 0.1)
        # 对于铅笔类别，这个条件暂时放宽 (可能因为铅笔细长，指尖不一定都能同时靠近)
        if self.canonical_pose_category == 'pencil':
            cond1 = torch.ones(1, device=self.device) # 暂时假设条件满足
        else:
            # 计算每个指尖到物体的距离，判断是否所有指尖都小于阈值
            cond1 = (torch.sqrt(((obj_pos - finger_pos) ** 2).sum(-1)) < 0.1).all(-1)
        # 条件 2: 至少有两个指尖与物体接触
        cond2 = contact_condition >= 2
        # 条件 3: 物体没有掉落 (Z 轴坐标大于重置高度阈值)
        cond3 = torch.greater(obj_pos[:, -1, -1], self.reset_z_threshold)
        # 综合所有条件 (逻辑与)
        cond = cond1.float() * cond2.float() * cond3.float() # 结果为 1 表示满足，0 表示不满足

        # 针对铅笔类别的额外条件 (检查铅笔的高度范围)
        if self.canonical_pose_category == 'pencil':
            # 计算铅笔两个端点在世界坐标系下的位置
            pencil_end_1 = self.object_pos + quat_apply(self.object_rot, to_torch(self.pencil_ends[0], device=self.device)[None].repeat(self.num_envs, 1))
            pencil_end_2 = self.object_pos + quat_apply(self.object_rot, to_torch(self.pencil_ends[1], device=self.device)[None].repeat(self.num_envs, 1))
            # 获取铅笔在 Z 轴上的最低点和最高点
            pencil_z_min = torch.min(pencil_end_1, pencil_end_2)[:, -1]
            pencil_z_max = torch.max(pencil_end_1, pencil_end_2)[:, -1]
            # 根据配置的初始姿态模式 (高/低) 检查铅笔是否在预定高度范围内
            if self.init_pose_mode == "high":
                cond4 = torch.logical_and(pencil_z_min > 0.62, pencil_z_max < 0.65)
            elif self.init_pose_mode == "low":
                cond4 = torch.logical_and(pencil_z_min > 0.60, pencil_z_max < 0.63)
            else:
                raise NotImplementedError # 如果模式未知则报错
            # 将高度条件加入总条件
            cond = cond * cond4.float()

        # --- 触发重置 ---
        # 如果任何一个条件不满足 (cond < 1)，则标记该环境需要重置
        self.reset_buf[cond < 1] = 1
        # 如果环境运行时间达到最大 episode 长度，也标记需要重置 (这种情况视为成功，状态已在 reset_idx 中保存)
        self.reset_buf[self.progress_buf >= self.max_episode_length] = 1