# --------------------------------------------------------
# Lessons from Learning to Spin “Pens”
# Written by Paper Authors
# Copyright (c) 2024 All Authors
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import os
import torch
import numpy as np
from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float, to_torch, quat_apply
from penspin.tasks.linker_hand_hora import LinkerHandHora

# 定义 LinkerHandGrasp 类，继承自 LinkerHandHora
# 这个类专门用于实现 Linker Hand 的抓取生成任务
class LinkerHandGrasp(LinkerHandHora):
    # 初始化函数
    def __init__(self, config, sim_device, graphics_device_id, headless):
        # 调用父类 LinkerHandHora 的初始化方法
        super().__init__(config, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)
        
        # 初始化一个空的 Tensor，用于存储成功生成的抓取状态（手部关节角度 + 物体位姿）
        self.saved_grasping_states = torch.zeros((0, 28), dtype=torch.float, device=self.device)
           # 定义一个字典，存储不同物体类别的"标准"抓取姿态
        self.canonical_pose_dict = {
            'pencil': [{'hand': [0.09058664739131927, -1.1684951782226562, -0.3523995876312256, -0.5262272953987122, -0.13673382997512817, -1.0198190212249756, -0.9558137059211731, -1.2512168884277344, -3.8383831224564346e-07, -1.3491917848587036, -0.5089983940124512, -0.3240099549293518, -0.05116431415081024, -0.9247583150863647, -1.2445602416992188, -0.9097212553024292, -0.020588409155607224, -1.1557494401931763, -0.7035709023475647, -0.2180664837360382, -0.001244630548171699],
            'object': [-0.11762521415948868, 0.023072263225913048, 0.18054848909378052, 0.708649218082428, 0.021436162292957306, 0.03619061037898064, -0.7043059468269348]},
            {'hand': [0.17996101081371307, -1.2075750827789307, -0.3885318338871002, -0.4472368657588959, -0.17998147010803223, -1.030931830406189, -0.910559356212616, -1.1463258266448975, -1.9397666051190754e-07, -1.2313501834869385, -0.6338013410568237, -0.4614178240299225, -0.08347102254629135, -0.8618580102920532, -1.1939127445220947, -0.8964090943336487, 0.17496797442436218, -1.3870528936386108, -0.6740223169326782, -0.21953974664211273, -0.0016594172921031713],
            'object': [-0.10668071359395981, 0.021574808284640312, 0.17717039585113525, 0.21597789227962494, -0.6708014011383057, 0.2683640122413635, -0.6567798256874084]},
            {'hand': [0.12977235019207, -0.9924959540367126, -0.3606451451778412, -0.4871084988117218, -0.17996357381343842, -1.1815704107284546, -1.0074188709259033, -1.063124179840088, -1.3303166213063378e-07, -1.1750483512878418, -0.5964861512184143, -0.7107267379760742, -0.0033535510301589966, -0.8607388734817505, -1.2455166578292847, -0.8664445281028748, -0.02050342597067356, -1.1381360292434692, -0.7652751803398132, -0.32268959283828735, -0.03381062299013138],
            'object': [-0.1360439658164978, 0.02780270390212536, 0.16825813055038452, -0.3405953049659729, -0.6889249086380005, 0.5953956842422485, -0.23426729440689087]}]
        }
        # 跟踪每个 canonical pose 的成功次数，确保最终结果均匀
        self.pose_success_counts = torch.zeros(len(self.canonical_pose_dict[self.canonical_pose_category]), dtype=torch.long, device=self.device)
        self.pose_id_tracker = torch.zeros(len(self.envs), dtype=torch.long, device=self.device)  # 跟踪每个环境当前使用的 pose ID

     
        self.canonical_pose = self.canonical_pose_dict[self.canonical_pose_category]
        self.sampled_init_pose = torch.zeros((len(self.envs), 28), dtype=torch.float, device=self.device)
        
        # 定义铅笔的两个端点相对于其中心的局部坐标
        self.pencil_ends = to_torch([
            [0, 0, -self.pen_length / 2 * self.base_obj_scale],
            [0, 0, self.pen_length / 2 * self.base_obj_scale]
        ], device=self.device)

        self.reset_steps = 0

    def reset_idx(self, env_ids):
        if self.randomize_pd_gains:
            self.p_gain[env_ids] = torch_rand_float(
                self.randomize_p_gain_lower, self.randomize_p_gain_upper, (len(env_ids), self.num_actions),
                device=self.device).squeeze(1)
            self.d_gain[env_ids] = torch_rand_float(
                self.randomize_d_gain_lower, self.randomize_d_gain_upper, (len(env_ids), self.num_actions),
                device=self.device).squeeze(1)

        self.rb_forces[env_ids, :, :] = 0.0
        
        # 判断这些重置的环境是否是因为达到了最大 episode 长度而成功的
        success = self.progress_buf[env_ids] >= self.max_episode_length
        # 获取当前这些环境的手部关节角度和物体位姿
        all_states = torch.cat([
            self.linker_hand_dof_pos, self.root_state_tensor[self.object_indices, :7]
        ], dim=1)

        # 提取成功完成 episode 的环境的最终状态 (抓取姿态)
        successful_states = all_states[env_ids][success]
        
        if successful_states.shape[0] > 0:
            # 获取成功环境对应的 pose ID
            successful_pose_ids = self.pose_id_tracker[env_ids][success]
            
            # 检查是否需要平衡不同 pose 的成功次数
            min_success_count = self.pose_success_counts.min().item()
            max_success_count = self.pose_success_counts.max().item()
            
            # 只保存那些成功次数还没有达到最大值的 pose 的状态
            balanced_states = []
            for i, pose_id in enumerate(successful_pose_ids):
                if self.pose_success_counts[pose_id] < max_success_count or max_success_count - min_success_count <= 1:
                    balanced_states.append(successful_states[i])
                    self.pose_success_counts[pose_id] += 1
            
            if balanced_states:
                balanced_states = torch.stack(balanced_states)
                self.saved_grasping_states = torch.cat([self.saved_grasping_states, balanced_states])
            #     print(f'初始化成功! 当前缓存数量: {self.saved_grasping_states.shape[0]}, Pose分布: {self.pose_success_counts.cpu().numpy()}')
            # else:
            #     print(f'跳过保存 - 为了保持平衡, Pose分布: {self.pose_success_counts.cpu().numpy()}')

        # 如果收集到的抓取姿态数量达到了预设阈值
        pose_threshold = int(eval(self.num_pose_per_cache[:-1]) * 1e3)
        if len(self.saved_grasping_states) >= pose_threshold:
            cache_dir = os.path.join('cache', self.grasp_cache_name, self.canonical_pose_category)
            os.makedirs(cache_dir, exist_ok=True)
            cache_name = f's{str(self.base_obj_scale).replace(".", "")}_{self.num_pose_per_cache}'
            cache_path = os.path.join(cache_dir, f'{cache_name}.npy')
            print(f"已收集到 {pose_threshold} 个抓取姿态，保存至 {cache_path} 并退出。")
            np.save(cache_path, self.saved_grasping_states[:pose_threshold].cpu().numpy())
            exit()

        # --- 重置物体状态 ---
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        
        # --- 设置随机化幅度 ---
        hand_randomize_amount = 0.1
        obj_randomize_amount = [0.01, 0.01, 0.0]

        # --- 初始化手和物体姿态 ---
        # 确保每个 canonical pose 被均匀分配给环境
        num_poses = len(self.canonical_pose)
        num_envs = len(env_ids)
        
        # 智能分配策略：优先分配成功次数较少的 pose
        pose_success_counts_cpu = self.pose_success_counts.cpu().numpy()
        min_success_count = pose_success_counts_cpu.min()
        
        # 找出成功次数最少的 pose
        underrepresented_poses = np.where(pose_success_counts_cpu == min_success_count)[0]
        
        # 如果所有 pose 的成功次数相等，则随机分配
        if len(underrepresented_poses) == num_poses:
            # 计算每个 pose 应该分配的环境数量
            poses_per_env = num_envs // num_poses
            remaining_envs = num_envs % num_poses
            
            # 创建均匀分布的 pose ID 列表
            pose_ids = []
            for pose_idx in range(num_poses):
                # 每个 pose 至少分配 poses_per_env 个环境
                pose_ids.extend([pose_idx] * poses_per_env)
            
            # 将剩余的环境随机分配给前 remaining_envs 个 pose
            if remaining_envs > 0:
                remaining_pose_ids = np.random.choice(num_poses, remaining_envs, replace=False)
                pose_ids.extend(remaining_pose_ids.tolist())
        else:
            # 优先分配给成功次数较少的 pose
            pose_ids = []
            
            # 首先尽可能多地分配给最少成功次数的 pose
            envs_per_underrepresented = max(1, num_envs // len(underrepresented_poses))
            allocated_envs = 0
            
            for pose_idx in underrepresented_poses:
                allocation = min(envs_per_underrepresented, num_envs - allocated_envs)
                pose_ids.extend([pose_idx] * allocation)
                allocated_envs += allocation
                if allocated_envs >= num_envs:
                    break
            
            # 如果还有剩余环境，随机分配给所有 pose
            while allocated_envs < num_envs:
                remaining_pose_id = np.random.choice(num_poses)
                pose_ids.append(remaining_pose_id)
                allocated_envs += 1
        
        # 打乱顺序以避免固定的分配模式
        np.random.shuffle(pose_ids)
        pose_ids = np.array(pose_ids)
        
        # 更新 pose ID 跟踪器
        self.pose_id_tracker[env_ids] = torch.tensor(pose_ids, dtype=torch.long, device=self.device)
        
        # 添加调试信息确认分配均匀
        if self.reset_steps % 100 == 0:  # 每100步打印一次统计信息
            unique, counts = np.unique(pose_ids, return_counts=True)
            pose_distribution = dict(zip(unique, counts))
            # print(f"Step {self.reset_steps}: Reset分布 {pose_distribution}, 成功分布: {self.pose_success_counts.cpu().numpy()}")

        hand_pose = to_torch([self.canonical_pose[pose_id]['hand'] for pose_id in pose_ids], device=self.device)
        hand_pose += hand_randomize_amount * torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_linker_hand_dofs), device=self.device)
        
        object_pose = to_torch([self.canonical_pose[pose_id]['object'] for pose_id in pose_ids], device=self.device)
        self.root_state_tensor[self.object_indices[env_ids], 0:7] = object_pose[:, 0:7]
        
        # 应用随机扰动到物体位置 (x, y, z)
        pos_rand = (torch_rand_float(-1.0, 1.0, (len(env_ids), 3), device=self.device) * to_torch(obj_randomize_amount, device=self.device))
        self.root_state_tensor[self.object_indices[env_ids], 0:3] += pos_rand

        self.sampled_init_pose[env_ids] = torch.cat([hand_pose, self.root_state_tensor[self.object_indices[env_ids], :7]], dim=-1)

        # 将物体的线速度和角速度重置为 0
        self.root_state_tensor[self.object_indices[env_ids], 7:13].zero_()

        object_indices = torch.unique(self.object_indices[env_ids]).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # --- 初始化手部状态 ---
        self.linker_hand_dof_pos[env_ids, :] = hand_pose
        self.linker_hand_dof_vel[env_ids, :].zero_()
        self.prev_targets[env_ids, :self.num_linker_hand_dofs] = hand_pose
        self.cur_targets[env_ids, :self.num_linker_hand_dofs] = hand_pose

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        if not self.torque_control:
            self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets), gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        # --- 重置其他状态缓冲区 ---
        self.progress_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1
        self.reset_steps += 1

    # 在 GPU 上高效地判断抓取是否失败
    def compute_reward(self, actions):
        # --- 获取必要的物理状态 (全部在 GPU 上) ---
        obj_pos = self.root_state_tensor[self.object_indices, 0:3]
        obj_rot = self.root_state_tensor[self.object_indices, 3:7]
        obj_lin_vel = self.root_state_tensor[self.object_indices, 7:10]
        # obj_ang_vel = self.root_state_tensor[self.object_indices, 10:13]

        # 刷新并获取接触力张量
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # contact_forces = self.net_contact_force_tensor
        # fingertip_forces = contact_forces.view(self.num_envs, -1, 3)[:, self.fingertip_handles, :]

        # --- 定义抓取失败的条件 (所有计算均为并行的张量操作) ---

        # 条件1: 物体掉落 (Z 轴坐标低于阈值)
        cond_dropped = obj_pos[:, self.up_axis_idx] < self.reset_z_threshold

        # 条件2: 物体不稳定 (线速度或角速度过大)
        cond_unstable = torch.linalg.norm(obj_lin_vel, dim=1) > 1.0
        # cond_unstable = torch.any(
        #     torch.stack([
        #         torch.linalg.norm(obj_lin_vel, dim=1) > 1.0,
        #         torch.linalg.norm(obj_ang_vel, dim=1) > 2.0
        #     ]), dim=0
        # )

        # # 条件3: 接触不足 (接触的指尖数量少于2个)
        # # 通过检查接触力大小来判断是否接触，阈值可以避免传感器噪声
        # contact_force_magnitudes = torch.linalg.norm(fingertip_forces, dim=-1)
        # num_contacts = torch.sum(contact_force_magnitudes > 1.0, dim=1) # 接触力大于1N视为有效接触
        # cond_no_contact = num_contacts < 2

        # 条件4 (铅笔专用): 笔身过于倾斜 (姿态不佳)
        # 计算铅笔两端在世界坐标系下的位置
        pencil_end_1 = obj_pos + quat_apply(obj_rot, to_torch(self.pencil_ends[0], device=self.device)[None].repeat(self.num_envs, 1))
        pencil_end_2 = obj_pos + quat_apply(obj_rot, to_torch(self.pencil_ends[1], device=self.device)[None].repeat(self.num_envs, 1))
        # 计算 Z 轴上的高度差
        pencil_height_diff = torch.abs(pencil_end_1[:, self.up_axis_idx] - pencil_end_2[:, self.up_axis_idx])
        cond_tilted = pencil_height_diff > 0.03

        # --- 综合所有失败条件 ---
        # 任何一个失败条件满足，都认为抓取失败，触发重置
        failed_buf = cond_dropped | cond_unstable | cond_tilted # | cond_no_contact 
        
        # --- 触发重置 ---
        # 1. 抓取失败的环境需要重置
        self.reset_buf = torch.where(failed_buf, torch.ones_like(self.reset_buf), self.reset_buf)
        # 2. 达到最大 episode 长度 (视为成功) 的环境也需要重置
        self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length, torch.ones_like(self.reset_buf), self.reset_buf)