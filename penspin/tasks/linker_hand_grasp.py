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
from penspin.tasks.linker_hand_hora import LinkerHandHora  # 从 penspin 项目中导入 LinkerHandHora 基类


# 定义 LinkerHandGrasp 类，继承自 LinkerHandHora
# 这个类专门用于实现 Linker Hand 的抓取生成任务
class LinkerHandGrasp(LinkerHandHora):
    # 初始化函数
    def __init__(self, config, sim_device, graphics_device_id, headless):
        # 调用父类 LinkerHandHora 的初始化方法
        super().__init__(config, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)
        
        # 初始化一个空的 Tensor，用于存储成功生成的抓取状态（手部关节角度 + 物体位姿）
        # 维度为 (0, 28)，其中 28 = 21 (手部DOF) + 7 (物体位姿: 3位置 + 4四元数)
        self.saved_grasping_states = torch.zeros((0, 28), dtype=torch.float, device=self.device)

        # 定义一个字典，存储不同物体类别的"标准"抓取姿态
        # 这些姿态作为生成抓取时的初始参考点或平均姿态
        # 从 linker_pose/initialization.md 加载
        self.canonical_pose_dict = {
            'pencil': [
                {'hand': [0.12453551590442657, -1.3983227014541626, -0.03729098662734032, -0.6903106570243835, -0.12674453854560852, -1.3707561492919922, -0.35772815346717834, -0.5420524477958679, -1.03192556721865e-08, -1.3883659839630127, -0.16317471861839294, -0.4754868745803833, -0.024602990597486496, -1.4065502882003784, 1.0915914572251495e-06, -0.6129568219184875, -0.30466559529304504, -1.4296414852142334, -0.23149807751178741, -0.5501845479011536, -0.34258005023002625],
                 'object': [-0.11171458661556244, -0.003700472181662917, 0.1771288514137268, -0.6131685376167297, -0.011775919236242771, 0.10527353733778, 0.7828174233436584]},
                {'hand': [0.17996101081371307, -1.2075750827789307, -0.3885318338871002, -0.4472368657588959, -0.17998147010803223, -1.030931830406189, -0.910559356212616, -1.1463258266448975, -1.9397666051190754e-07, -1.2313501834869385, -0.6338013410568237, -0.4614178240299225, -0.08347102254629135, -0.8618580102920532, -1.1939127445220947, -0.8964090943336487, 0.17496797442436218, -1.3870528936386108, -0.6740223169326782, -0.21953974664211273, -0.0016594172921031713],
                 'object': [-0.10668071359395981, 0.021574808284640312, 0.17717039585113525, 0.21597789227962494, -0.6708014011383057, 0.2683640122413635, -0.6567798256874084]},
                {'hand': [0.12453551590442657, -1.3983227014541626, -0.03729098662734032, -0.6903106570243835, -0.12674453854560852, -1.3707561492919922, -0.35772815346717834, -0.5420524477958679, -1.03192556721865e-08, -1.3883659839630127, -0.16317471861839294, -0.4754868745803833, -0.024602990597486496, -1.4065502882003784, 1.0915914572251495e-06, -0.6129568219184875, -0.30466559529304504, -1.4296414852142334, -0.23149807751178741, -0.5501845479011536, -0.34258005023002625],
                 'object': [-0.11171458661556244, -0.003700472181662917, 0.1771288514137268, -0.6131685376167297, -0.011775919236242771, 0.10527353733778, 0.7828174233436584]},
                {'hand': [0.18000002205371857, -1.0464586019515991, -0.41268664598464966, -0.41177529096603394, -0.17681774497032166, -1.3315761089324951, -0.3797590732574463, -0.5883009433746338, -2.8795298590011953e-07, -1.0698927640914917, -0.7280430793762207, -0.5705047249794006, -0.0002597713319119066, -0.9266747832298279, -0.26815065741539, -0.3203774094581604, 0.1648491472005844, -1.0737873315811157, -0.6846374273300171, -0.41734376549720764, -0.5779910683631897],
                 'object': [-0.13613660633563995, 0.01893787458539009, 0.16901230812072754, -0.5389097929000854, -0.5710364580154419, 0.5835363864898682, -0.20731335878372192]},
            ]}

        # 根据配置选择当前使用的标准姿态类别 只支持pencil
        self.canonical_pose = self.canonical_pose_dict[self.canonical_pose_category]
        # 初始化一个 Tensor，用于存储每个环境采样得到的初始姿态（手+物体）
        self.sampled_init_pose = torch.zeros((len(self.envs), 28), dtype=torch.float, device=self.device)

        
        # 定义铅笔的两个端点相对于其中心的局部坐标
        # 0.4 是铅笔的长度， self.base_obj_scale 是物体的缩放比例
        self.pencil_ends = [
            [0, 0, -self.pen_length / 2 * self.base_obj_scale],
            [0, 0, self.pen_length / 2 * self.base_obj_scale]
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
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_linker_hand_dofs * 2 + 5), device=self.device)

        # 重置指定环境的刚体力为 0
        self.rb_forces[env_ids, :, :] = 0.0
        # 判断这些重置的环境是否是因为达到了最大 episode 长度而成功的
        success = self.progress_buf[env_ids] == self.max_episode_length
        # 获取当前这些环境的手部关节角度和物体位姿
        all_states = torch.cat([
            self.linker_hand_dof_pos, self.root_state_tensor[self.object_indices, :7]
        ], dim=1)

        # 提取成功完成 episode 的环境的最终状态 (抓取姿态)
        successful_states = all_states[env_ids][success]

        # 如果有新的成功状态被添加
        if successful_states.shape[0] > 0:
            # 将成功状态添加到缓存中
            self.saved_grasping_states = torch.cat([self.saved_grasping_states, successful_states])
            # 只在第一次打印成功姿态
            if not hasattr(self, 'first_pose_printed') or not self.first_pose_printed:
                for i in range(successful_states.shape[0]):
                    state = successful_states[i]
                    # 将 Tensor 转换为 NumPy 数组以便打印
                    state_np = state.cpu().numpy()
                    hand_pose = state_np[:self.num_linker_hand_dofs]
                    object_pose = state_np[self.num_linker_hand_dofs:]

                    # print("{'hand': " + str(hand_pose.tolist()) + ",")
                    # print(" 'object': " + str(object_pose.tolist()) + "},")
                #self.first_pose_printed = True

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
            'pencil': 0.1, 'thinpencil': 0.05,
        }[self.canonical_pose_category]
        # 根据物体类别确定物体位置的随机化幅度 [x, y, z]
        obj_randomize_amount = {
            'pencil': [0.01, 0.01, 0.0], 'thinpencil': [0.01, 0.01, 0.0],
        }[self.canonical_pose_category]

        # --- 初始化手和物体姿态 ---
        # 为每个需要重置的环境随机选择一个标准姿态的索引
        pose_ids = np.random.randint(0, len(self.canonical_pose), size=len(env_ids))
        # 获取选定的标准手部姿态，并转换为 Tensor
        hand_pose = to_torch([self.canonical_pose[pose_id]['hand'] for pose_id in pose_ids], device=self.device)
        # 应用随机扰动到手部姿态
        hand_pose += hand_randomize_amount * rand_floats[:, 5:5 + self.num_linker_hand_dofs]
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
        self.linker_hand_dof_pos[env_ids, :] = hand_pose
        # 设置手部关节的初始速度为 0
        self.linker_hand_dof_vel[env_ids, :] = 0
        # 初始化上一帧和当前帧的控制目标为初始手部姿态
        self.prev_targets[env_ids, :self.num_linker_hand_dofs] = hand_pose
        self.cur_targets[env_ids, :self.num_linker_hand_dofs] = hand_pose

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

    # 计算奖励函数 (在这个抓取生成任务中，主要用于判断**是否**满足抓取条件，触发重置)
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
        # 条件 2: 至少有两个指尖与物体接触 TODO 改为取消这一条件
        cond2 = contact_condition >= 0
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
            # 检查铅笔 Z 轴最高点和最低点的距离是否小于0.03 笔长度为0.4*0.3=0.12
            cond4 = (pencil_z_max - pencil_z_min) < 0.03
            # print('cond4:', pencil_z_max - pencil_z_min)
            
            # 将高度条件加入总条件
            cond = cond * cond4.float()

        # --- 触发重置 ---
        # 如果任何一个条件不满足 (cond < 1)，则标记该环境需要重置
        self.reset_buf[cond < 1] = 1

        # Print values of individual conditions for environments where overall cond < 1
        # These are the environments that will be reset due to failure conditions.
        # failed_env_indices = torch.where(cond < 1)[0]
        # if len(failed_env_indices) > 0:
        #     # 获取不满足条件的环境的条件值
        #     for env_idx_tensor in failed_env_indices:
        #         env_idx = env_idx_tensor.item()
        #     print(f"  cond2 (min_2_contacts): {cond2[env_idx].item()}")
        #     print(f"  cond3 (object_not_dropped): {cond3[env_idx].item()}")
        #     print(f"  cond4 (pencil_height_range): {cond4[env_idx].item()}")
        # 如果环境运行时间达到最大 episode 长度，也标记需要重置 (这种情况视为成功，状态已在 reset_idx 中保存)
        
        self.reset_buf[self.progress_buf >= self.max_episode_length] = 1