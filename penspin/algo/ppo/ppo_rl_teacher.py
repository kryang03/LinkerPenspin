# --------------------------------------------------------
# Lessons from Learning to Spin “Pens”
# Written by Paper Authors
# Copyright (c) 2024 All Authors
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: RLGames
# Copyright (c) 2019 Denys88
# Licence under MIT License
# https://github.com/Denys88/rl_games/
# --------------------------------------------------------

# 导入所需的库
import os
import time
import torch
import numpy as np
from isaacgym import gymapi
# [移除] CosineAnnealingLR - Curriculum Learning 场景下不适合时间衰减 LR
# from torch.optim.lr_scheduler import CosineAnnealingLR

# 从 penspin.algo.ppo 模块导入 ExperienceBuffer，用于存储智能体与环境交互的数据
from penspin.algo.ppo.experience import ExperienceBuffer
# 从 penspin.algo.models 模块导入 TeacherActorCritic 模型
from penspin.algo.models.models import TeacherActorCritic
# 从 penspin.algo.models 模块导入 RunningMeanStd，用于对输入进行标准化处理
from penspin.algo.models.running_mean_std import RunningMeanStd

# 从 penspin.utils.misc 模块导入 AverageScalarMeter，用于计算标量值的平均值
from penspin.utils.misc import AverageScalarMeter

# 导入 tensorboardX 中的 SummaryWriter，用于记录训练过程中的日志和可视化数据
from tensorboardX import SummaryWriter

# 导入轨迹录制器，用于在课程学习更新时录制完整轨迹
from penspin.utils.trajectory_recorder import TrajectoryRecorder

# 导入统一的机器人配置常量
from penspin.utils.robot_config import (
    NUM_DOF, NUM_FINGERS, FINGERTIP_CNT, CONTACT_DIM, PROPRIO_DIM
)

# 定义 PPOTeacher 类，实现 Proximal Policy Optimization 算法用于Teacher训练
class PPOTeacher(object):
    # 构造函数，初始化 PPO 智能体
    def __init__(self, env, output_dif, full_config):
        # 设置设备（CPU 或 GPU），从配置中读取
        self.device = full_config['rl_device']
        # 获取网络配置
        self.network_config = full_config.train.network
        # 获取 PPO 算法相关的配置
        self.ppo_config = full_config.train.ppo
        # ---- 构建环境 ----
        self.env = env
        # 获取环境中的并行智能体数量（num_actors）
        self.num_actors = self.ppo_config['num_actors']
        # 获取动作空间
        action_space = self.env.action_space
        # 获取动作的数量
        self.actions_num = action_space.shape[0]
        # 获取动作空间的下界，并转换为 PyTorch Tensor 并移动到指定设备
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.device)
        # 获取动作空间的上界，并转换为 PyTorch Tensor 并移动到指定设备
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.device)
        # 获取观察空间
        self.observation_space = self.env.observation_space
        # 获取观察的形状
        self.obs_shape = self.observation_space.shape
        # **特权信息**
        # 获取特权信息的维度，通常指只有智能体能获取的环境信息，例如机器人关节速度、触觉信息等
        self.priv_info_dim = self.env.priv_info_dim
        # Teacher模型固定使用特权信息
        self.priv_info = True
        self.proprio_adapt = self.ppo_config['proprio_adapt']
        # ---- Critic 信息 ----
        # 配置是否使用非对称 Actor-Critic 结构，即 Actor 和 Critic 使用不同的输入或网络结构
        self.asymm_actor_critic = self.ppo_config['asymm_actor_critic']
        # 获取 Critic 使用的额外信息的维度
        self.critic_info_dim = self.ppo_config['critic_info_dim']
        # ---- 点云信息 (Point Cloud Info) ----
        # 获取点云 buffer 的维度
        self.point_cloud_buffer_dim = self.env.point_cloud_buffer_dim
        # 本体感觉模式
        self.proprio_mode = self.ppo_config['proprio_mode']
        # 输入模式
        self.input_mode = self.ppo_config['input_mode']
        # 本体感觉历史信息的长度
        self.proprio_len = self.ppo_config['proprio_len']
        # Teacher模型固定使用点云信息
        self.use_point_cloud_info = True
        # ---- 输出目录 (Output Dir) ----
        # 指定实验结果保存的根目录
        self.output_dir = output_dif
        # 模型保存目录（Teacher使用teacher_nn）
        self.nn_dir = os.path.join(self.output_dir, 'teacher_nn')
        # TensorBoard 日志保存目录（Teacher使用teacher_tb）
        self.tb_dif = os.path.join(self.output_dir, 'teacher_tb')
        # 创建模型保存目录，如果已存在则不报错
        os.makedirs(self.nn_dir, exist_ok=True)
        # 创建 TensorBoard 日志保存目录，如果已存在则不报错
        os.makedirs(self.tb_dif, exist_ok=True)
        # ---- 模型 (Model) ----
        # 构建 Actor-Critic 模型的配置字典
        net_config = {
            'actor_units': self.network_config.mlp.units, # Actor MLP 的层单元数
            'priv_mlp_units': self.network_config.priv_mlp.units, # 特权信息 MLP 的层单元数
            'actions_num': self.actions_num, # 动作数量
            'input_shape': self.obs_shape, # 输入（观察）的形状
            'priv_info_dim': self.priv_info_dim, # 特权信息的维度
            'critic_info_dim': self.critic_info_dim, # Critic 信息的维度
            'asymm_actor_critic': self.asymm_actor_critic, # 是否使用非对称 Actor-Critic
            'point_mlp_units': self.network_config.point_mlp.units, # 点云 MLP 的层单元数
            'use_point_transformer': self.network_config.use_point_transformer, # 是否使用 Point Transformer 处理点云
        }
        # 实例化 Teacher Actor-Critic 模型 (用于 PPO 训练)
        self.model = TeacherActorCritic(net_config)
        # 将模型移动到指定设备
        self.model.to(self.device)

        # 初始化 RunningMeanStd 用于标准化观察空间输入
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        # 初始化 RunningMeanStd 用于标准化特权信息
        self.priv_mean_std = RunningMeanStd(self.priv_info_dim).to(self.device)
        # 本体感觉维度：从环境动态获取（支持 Flying Hand 等不同 DOF 配置）
        # 优先使用环境的 actual_proprio_dim，如果不存在则回退到默认 PROPRIO_DIM
        self.proprio_dim = getattr(self.env, 'actual_proprio_dim', PROPRIO_DIM)
        # Teacher模型使用3维点云标准化
        self.point_cloud_mean_std = RunningMeanStd(3,).to(self.device)
        # 初始化 RunningMeanStd 用于标准化价值函数输出
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)
        # ---- 优化器 (Optim) ----
        # 获取初始学习率
        self.last_lr = float(self.ppo_config['learning_rate'])
        # 获取权重衰减系数，如果未配置则默认为 0.0
        self.weight_decay = self.ppo_config.get('weight_decay', 0.0)
        # 初始化 AdamW 优化器，优化模型的参数
        self.optimizer = torch.optim.AdamW(self.model.parameters(), self.last_lr, weight_decay=self.weight_decay)
        
        # ---- 学习率调度策略 (Learning Rate Scheduling) ----
        # [设计说明] 移除 CosineAnnealingLR，完全依赖 KL 自适应调度
        # 原因：Curriculum Learning 是非平稳 (Non-Stationary) 过程
        # - 每当 α 改变，物理环境就变了（相当于换了一个新游戏）
        # - CosineAnnealing 的假设是任务静态，随时间需微调，故 LR 单调递减
        # - Space E 的现实：当 α 从 0.3 变到 0.4 时，Agent 需要较大 LR 适应新动力学
        # - 如果此时 LR 已被 Cosine 压到 1e-5，Agent 无法适应新环境，导致卡死
        # 
        # 新策略：LR 仅由策略稳定性 (KL 散度) 决定，不随时间强制衰减
        # - 当 KL 过大：Agent 在剧烈探索，降低 LR 稳定训练
        # - 当 KL 过小：Agent 陷入局部，提高 LR 鼓励探索
        self.max_steps = self.ppo_config['max_agent_steps']
        
        # ---- PPO 训练参数 (PPO Train Param) ----
        # PPO 裁剪范围的 epsilon 值
        self.e_clip = self.ppo_config['e_clip']
        # 价值函数裁剪范围
        self.clip_value = self.ppo_config['clip_value']
        # 熵系数，用于鼓励策略探索
        self.entropy_coef = self.ppo_config['entropy_coef']
        # Critic 损失系数
        self.critic_coef = self.ppo_config['critic_coef']
        # 边界损失系数，用于限制动作在合法范围内
        self.bounds_loss_coef = self.ppo_config['bounds_loss_coef']
        # 折扣因子 gamma
        self.gamma = self.ppo_config['gamma']
        # GAE (Generalized Advantage Estimation) 参数 tau
        self.tau = self.ppo_config['tau']
        # 是否裁剪梯度
        self.truncate_grads = self.ppo_config['truncate_grads']
        # 梯度裁剪的范数阈值
        self.grad_norm = self.ppo_config['grad_norm']
        # 是否使用 Value Bootstrap (通常在 episode 未结束时用于估计最后一个状态的价值)
        self.value_bootstrap = self.ppo_config['value_bootstrap']
        # 是否标准化 Advantage
        self.normalize_advantage = self.ppo_config['normalize_advantage']
        # 是否标准化输入观察
        self.normalize_input = self.ppo_config['normalize_input']
        # 是否标准化价值函数输出
        self.normalize_value = self.ppo_config['normalize_value']
        # 是否标准化特权信息
        self.normalize_priv = self.ppo_config['normalize_priv']
        # 是否标准化点云信息
        self.normalize_point_cloud = self.ppo_config['normalize_point_cloud']
        # ---- PPO 收集参数 (PPO Collect Param) ----
        # 每个 rollout 的步数（horizon length）
        self.horizon_length = self.ppo_config['horizon_length']
        # 总的 batch size，等于 num_actors * horizon_length
        self.batch_size = self.horizon_length * self.num_actors
        # 训练时每个 minibatch 的大小
        self.minibatch_size = self.ppo_config['minibatch_size']
        # 每个 epoch 的训练 mini-epoch 数量
        self.mini_epochs_num = self.ppo_config['mini_epochs']
        # 确保 batch_size 可以被 minibatch_size 整除，除非处于测试模式
        assert self.batch_size % self.minibatch_size == 0 or full_config.test
        # ---- 调度器 (scheduler) ----
        # KL 散度阈值，用于自适应调整学习率
        self.kl_threshold = self.ppo_config['kl_threshold']
        # KL 自适应因子 (默认 1.15)
        self.kl_adaptive_factor = self.ppo_config.get('kl_adaptive_factor', 1.15)
        # 最小学习率 (默认 1e-6)
        self.min_lr = self.ppo_config.get('min_lr', 1e-6)
        # 初始化自适应学习率调度器，传入初始学习率作为 max_lr
        self.scheduler = AdaptiveScheduler(
            kl_threshold=self.kl_threshold,
            min_lr=self.min_lr,
            max_lr=self.last_lr,
            adaptive_factor=self.kl_adaptive_factor
        )
        # ---- 快照 (Snapshot) ----
        # 模型保存频率
        self.save_freq = self.ppo_config['save_frequency']
        # 在达到多少 agent steps 后开始保存最优模型
        self.save_best_after = self.ppo_config['save_best_after']
        # ---- Tensorboard 日志 (Tensorboard Logger) ----
        # 用于存储额外信息的字典
        self.extra_info = {}
        # 初始化 SummaryWriter
        writer = SummaryWriter(self.tb_dif)
        self.writer = writer

        # ---- Rollout GIFs ----
        # GIF 帧计数器
        self.gif_frame_counter = 0
        # 对于 Horizon Length = 16，建议设置：每 50 Epoch 录制一次：$16 \times 50 = 800$每 100 Epoch 录制一次：$16 \times 100 = 1600$
        # 每隔多少步保存一次 GIF
        self.gif_save_every_n = 800
        # GIF 的帧长度
        self.gif_save_length = 80
        # 用于存储 GIF 帧的列表
        self.gif_frames = []

        # 初始化 AverageScalarMeter 用于记录 episode 奖励
        self.episode_rewards = AverageScalarMeter(20000)
        # 初始化 AverageScalarMeter 用于记录 episode 长度
        self.episode_lengths = AverageScalarMeter(20000)
        self.total_rot_angle = AverageScalarMeter(20000)
        self.total_waypoint_tracking_reward = AverageScalarMeter(20000)
        # 追踪成功案例（旋转角度>阈值）的数量和比例
        self.success_count = 0
        self.total_episodes = 0
        self.best_success_rate = 0.0
        # 追踪失败案例（用于计算 survival_rate）
        self.fall_count = 0  # 掉落 + 倾斜
        self.drop_count = 0  # 仅掉落
        self.tilt_count = 0  # 仅倾斜
        # 存储当前观察
        self.obs = None
        # 当前 epoch 计数
        self.epoch_num = 0
        # 实例化 ExperienceBuffer，用于存储 rollout 数据
        self.storage = ExperienceBuffer(
            self.num_actors, self.horizon_length, self.batch_size, self.minibatch_size, self.obs_shape[0],
            self.actions_num, self.priv_info_dim, self.critic_info_dim, self.point_cloud_buffer_dim, self.device,
            self.proprio_dim,self.proprio_len # 添加 proprio_dim 和 proprio_len 参数
        )

        # ================================================================================
        # 打印显存相关的关键配置参数
        # ================================================================================
        print("\n" + "="*80)
        print("显存相关配置参数 (Memory-Critical Configuration)")
        print("="*80)
        print(f"环境数量 (num_actors):        {self.num_actors:,}")
        print(f"Horizon 长度:                 {self.horizon_length}")
        print(f"Batch Size:                   {self.batch_size:,} = {self.num_actors:,} × {self.horizon_length}")
        print(f"Minibatch Size:               {self.minibatch_size:,}")
        print(f"Mini Epochs:                  {self.mini_epochs_num}")
        print(f"观察维度 (obs_shape):         {self.obs_shape[0]}")
        print(f"动作维度 (actions_num):       {self.actions_num}")
        print(f"特权信息维度 (priv_info):     {self.priv_info_dim}")
        print(f"Critic信息维度:               {self.critic_info_dim}")
        print(f"点云缓冲维度:                 {self.point_cloud_buffer_dim}")
        print(f"本体感觉历史长度:             {self.proprio_len}")

        print("="*80 + "\n")

        # 初始化当前 episode 奖励和长度
        batch_size = self.num_actors
        current_rewards_shape = (batch_size, 1)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_rot_angle = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_waypoint_tracking_reward = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        # 初始化 dones 标志，最初所有环境都标记为 done
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)
        # 初始化智能体总步数计数器
        self.agent_steps = 0
        # 最大智能体步数
        self.max_agent_steps = self.ppo_config['max_agent_steps']
        # 记录迄今为止最优的 episode 平均奖励
        self.best_rewards = -10000
        # ---- 时间统计 (Timing) ----
        # 数据收集总时间
        self.data_collect_time = 0
        # RL 训练总时间
        self.rl_train_time = 0
        # 所有时间
        self.all_time = 0
        
        # ---- Curriculum 配置 (在轨迹录制器初始化之前需要) ----
        # 获取 curriculum 配置用于自适应成功阈值
        self.success_rot_threshold = 10.0  # 默认值
        self.use_adaptive_threshold = False
        if hasattr(self.env, 'config') and 'env' in self.env.config:
            curriculum_config = self.env.config['env'].get('curriculum', {})
            self.success_rot_threshold = curriculum_config.get('success_rot_threshold', 10.0)
            self.use_adaptive_threshold = curriculum_config.get('use_adaptive_threshold', False)
        
        # ---- 轨迹录制器 (Trajectory Recorder) ----
        # 用于在课程学习更新时录制完整的成功轨迹
        # 只在主进程且启用相机时初始化
        self.trajectory_recorder = None
        is_main_process = int(os.getenv('LOCAL_RANK', '0')) == 0
        camera_enabled = getattr(self.env, 'with_camera', False)
        
        print("\n" + "="*80)
        print("轨迹录制器初始化状态 (Trajectory Recorder Status)")
        print("="*80)
        print(f"  主进程 (LOCAL_RANK=0):  {is_main_process}")
        print(f"  相机已启用:             {camera_enabled}")
        
        if is_main_process and camera_enabled:
            # 使用自适应成功阈值逻辑，与 effective_threshold 保持一致
            # 但轨迹录制需要更宽松的阈值来捕获轨迹，所以使用 0.6 系数
            success_threshold = self.success_rot_threshold * 0.6
            if self.use_adaptive_threshold and hasattr(self.env, 'time_warper'):
                success_threshold = self.success_rot_threshold * self.env.time_warper.current_alpha * 0.6
            
            self.trajectory_recorder = TrajectoryRecorder(
                env=self.env,
                num_record_envs=4,  # 追踪 4 个环境
                max_episode_length=1000,  # 最大 episode 长度
                success_threshold=success_threshold,
                min_trajectories_to_export=1,  # 只要有1条成功轨迹就导出
                keep_best_k=5,  # 缓冲区保留 Top-5 最佳轨迹
                device=self.device
            )
            print(f"  轨迹录制器:             ✓ 已初始化")
            print(f"  成功阈值:               {success_threshold:.2f} rad")
        else:
            print(f"  轨迹录制器:             ✗ 未初始化")
            if not is_main_process:
                print(f"    原因: 非主进程 (LOCAL_RANK != 0)")
            if not camera_enabled:
                print(f"    原因: 相机未启用 (enableCameraSensors=False)")
                print(f"    解决方案: 在启动命令中添加 task.env.enableCameraSensors=True")
        print("="*80 + "\n")

    # 写入统计数据到 TensorBoard
    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls, grad_norms):
        # 记录 RL 训练的 FPS (Frames Per Second)，以 agent steps / RL 训练时间 计算
        self.writer.add_scalar('performance/RLTrainFPS', self.agent_steps / self.rl_train_time, self.agent_steps)
        # 记录环境交互的 FPS，以 agent steps / 数据收集时间 计算
        self.writer.add_scalar('performance/EnvStepFPS', self.agent_steps / self.data_collect_time, self.agent_steps)

        # 记录 Actor 损失的平均值
        self.writer.add_scalar('losses/actor_loss', torch.mean(torch.stack(a_losses)).item(), self.agent_steps)
        # 记录边界损失的平均值
        self.writer.add_scalar('losses/bounds_loss', torch.mean(torch.stack(b_losses)).item(), self.agent_steps)
        # 记录 Critic 损失的平均值
        self.writer.add_scalar('losses/critic_loss', torch.mean(torch.stack(c_losses)).item(), self.agent_steps)
        # 记录熵的平均值
        self.writer.add_scalar('losses/entropy', torch.mean(torch.stack(entropies)).item(), self.agent_steps)

        # 记录当前学习率
        self.writer.add_scalar('info/last_lr', self.last_lr, self.agent_steps)
        # 记录 PPO 裁剪范围 epsilon
        self.writer.add_scalar('info/e_clip', self.e_clip, self.agent_steps)
        # 记录 KL 散度的平均值
        self.writer.add_scalar('info/kl', torch.mean(torch.stack(kls)).item(), self.agent_steps)
        # 记录梯度范数的平均值
        self.writer.add_scalar('info/grad_norms', torch.mean(torch.stack(grad_norms)    ).item(), self.agent_steps)

        # 记录额外信息中的标量值
        for k, v in self.extra_info.items():
            if isinstance(v, torch.Tensor) and len(v.shape) != 0:
                continue
            self.writer.add_scalar(f'{k}', v, self.agent_steps)

    # 设置模型为评估模式
    def set_eval(self):
        self.model.eval()
        # 如果配置了输入标准化，则将 RunningMeanStd 也设置为评估模式
        if self.normalize_input:
            self.running_mean_std.eval()
        # 如果配置了特权信息标准化，则将 priv_mean_std 也设置为评估模式
        if self.normalize_priv:
            self.priv_mean_std.eval()
        # 如果配置了点云标准化，则将 point_cloud_mean_std 也设置为评估模式
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.eval()
        # 如果配置了价值标准化，则将 value_mean_std 也设置为评估模式
        if self.normalize_value:
            self.value_mean_std.eval()

    # 设置模型为训练模式
    def set_train(self):
        self.model.train()
        # 如果配置了输入标准化，则将 RunningMeanStd 也设置为训练模式
        if self.normalize_input:
            self.running_mean_std.train()
        # 如果配置了特权信息标准化，则将 priv_mean_std 也设置为训练模式
        if self.normalize_priv:
            self.priv_mean_std.train()
        # 如果配置了点云标准化，则将 point_cloud_mean_std 也设置为训练模式
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.train()
        # 如果配置了价值标准化，则将 value_mean_std 也设置为训练模式
        if self.normalize_value:
            self.value_mean_std.train()

    # 模型动作预测，用于数据收集阶段
    def model_act(self, obs_dict):
        # 对观察进行标准化处理
        processed_obs = self.running_mean_std(obs_dict['obs'])
        # 获取特权信息
        priv_info = obs_dict['priv_info']
        # 如果配置了特权信息标准化，则进行标准化处理
        if self.normalize_priv:
            priv_info = self.priv_mean_std(obs_dict['priv_info'])

        # 如果配置了点云标准化，则进行标准化处理
        if self.normalize_point_cloud:
            # 将点云数据reshape后进行标准化，再reshape回原形状
            point_cloud = self.point_cloud_mean_std(
                obs_dict['point_cloud_info'].reshape(-1, 3)
            ).reshape((processed_obs.shape[0], -1, 3))
        else:
            # 否则直接使用原始点云信息
            point_cloud = obs_dict['point_cloud_info']

        # 构建输入字典，包含所有模型所需的输入数据
        input_dict = {
            'obs': processed_obs, # 标准化后的观察
            'priv_info': priv_info, # 标准化或原始的特权信息
            'critic_info': obs_dict['critic_info'], # Critic 信息
            'point_cloud_info': point_cloud, # 标准化或原始的点云信息
            'proprio_hist': obs_dict['proprio_hist'], # 本体感觉历史信息
            'tactile_hist': obs_dict['tactile_hist'], # 触觉历史信息
            'obj_ends': obs_dict['obj_ends'], # 物体末端信息
        }
        # 调用模型进行动作预测（actor）和价值估计（critic）
        res_dict = self.model.act(input_dict)
        # 对预测的价值进行逆标准化（True 表示逆标准化）
        res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        # 返回结果字典，包含预测的动作、价值、对数概率等
        return res_dict

    # 主训练循环
    def train(self):
        # 记录开始时间
        _t = time.time()
        _last_t = time.time()
        # 重置环境，获取初始观察
        self.obs = self.env.reset()
        
        # ==============================================================
        # Space E: 初始化 Curriculum Learning 物理参数
        # ==============================================================
        # 在训练开始时应用初始的 curriculum 物理参数 (α = alpha_start)
        # 这确保从第一个 episode 开始就使用正确的慢时间物理设置
        if hasattr(self.env, 'time_warper') and hasattr(self.env, 'apply_curriculum_physics'):
            print(f"[Space E] Initializing curriculum physics with α = {self.env.time_warper.current_alpha:.4f}")
            self.env.apply_curriculum_physics()
        
        # 初始化 agent steps，每个 epoch 开始时增加一个 batch_size
        self.agent_steps = self.batch_size
        self.success_count = 0
        self.total_episodes = 0
        self.fall_count = 0
        self.drop_count = 0
        self.tilt_count = 0
        self.env.reset_termination_counts()
        
        # ============================================================
        # 累积统计缓冲区 (Accumulated Statistics Buffer)
        # ============================================================
        # 解决低 α 阶段的"分母消失"问题：
        # 当 episode 很长（高存活率）时，一个 Epoch 内完成的 episode 数量
        # 可能非常少，导致 survival_rate 等指标统计失真。
        # 
        # 方案：累积多个 Epoch 的数据，直到样本量足够才进行门控判断。
        # ============================================================
        self.accumulated_stats = {
            'total_episodes': 0,
            'success_count': 0,
            'fail_count': 0,
            'rot_angle_sum': 0.0,
            'reward_sum': 0.0,
        }
        # 最小样本量阈值（建议设为并行环境数的 5%~10% 或固定值 500）
        self.min_episodes_for_curriculum = 500
        
        # 循环直到达到最大 agent steps
        while self.agent_steps < self.max_agent_steps:
            # 增加 epoch 计数
            self.epoch_num += 1
            
            # ---- 轨迹录制器：动态调整成功阈值 ----
            # 在 Space E 中，低 alpha 阶段成功旋转的弧度较小
            # 需要根据当前 alpha 动态调整阈值，否则初期无法录制到轨迹
            if self.trajectory_recorder is not None and hasattr(self.env, 'time_warper'):
                # 动态阈值 = 基础阈值 × alpha × 宽松系数(0.6)
                # alpha=0.1 时阈值为 0.6 rad，alpha=1.0 时为 6.0 rad
                adaptive_threshold = self.success_rot_threshold * self.env.time_warper.current_alpha * 0.6
                self.trajectory_recorder.success_threshold = adaptive_threshold
            
            # 执行一个训练 epoch（包括数据收集和模型更新）
            a_losses, c_losses, b_losses, entropies, kls, grad_norms = self.train_epoch()
            
            # 清空 storage 中的数据，准备下一个 epoch 的收集
            self.storage.data_dict = None

            # 更新额外信息字典，只保留标量值
            for k, v in self.extra_info.items():
                # only log scalars
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.extra_info[k] = v

            # 计算整体 FPS 和最近一个 epoch 的 FPS
            # 平均每秒执行的 Agent step数。这是一个衡量系统整体长期运行效率的指标
            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()


            # 将统计数据写入 TensorBoard
            self.write_stats(a_losses, c_losses, b_losses, entropies, kls, grad_norms)
            # 获取 episode 奖励和长度的平均值
            mean_rewards = self.episode_rewards.get_mean()
            mean_lengths = self.episode_lengths.get_mean()
            mean_rot_angle = self.total_rot_angle.get_mean()
            mean_waypoint_tracking_reward = self.total_waypoint_tracking_reward.get_mean()
            # 计算成功率（旋转角度>阈值）
            current_success_rate = self.success_count / max(self.total_episodes, 1)
            if current_success_rate > self.best_success_rate:
                self.best_success_rate = current_success_rate
            
            # ==============================================================
            # Space E: Curriculum Learning - Time Warping 更新
            # ==============================================================
            # 在每个 epoch 结束后更新 curriculum 进度
            # 如果 α 发生显著变化，则重新应用物理参数
            if hasattr(self.env, 'time_warper'):
                # ------------------------------------------------------------
                # 累积统计：解决低 α 阶段的"分母消失"问题
                # ------------------------------------------------------------
                # 将当前 Epoch 的数据累加到缓冲区
                self.accumulated_stats['total_episodes'] += self.total_episodes
                self.accumulated_stats['success_count'] += self.success_count
                self.accumulated_stats['fail_count'] += self.fall_count
                self.accumulated_stats['rot_angle_sum'] += mean_rot_angle * self.total_episodes  # 加权累加
                self.accumulated_stats['reward_sum'] += mean_rewards * self.total_episodes
                
                # 检查累积样本量是否足够
                accumulated_episodes = self.accumulated_stats['total_episodes']
                has_enough_samples = accumulated_episodes >= self.min_episodes_for_curriculum
                
                # 基于累积统计计算更鲁棒的 metrics
                if has_enough_samples:
                    # 用累积数据计算 survival_rate（更稳健）
                    accumulated_survival_rate = 1.0 - (
                        self.accumulated_stats['fail_count'] / max(accumulated_episodes, 1)
                    )
                    accumulated_success_rate = (
                        self.accumulated_stats['success_count'] / max(accumulated_episodes, 1)
                    )
                    accumulated_mean_rot = (
                        self.accumulated_stats['rot_angle_sum'] / max(accumulated_episodes, 1)
                    )
                    accumulated_mean_reward = (
                        self.accumulated_stats['reward_sum'] / max(accumulated_episodes, 1)
                    )
                    
                    # 构建 metrics 字典传递给 time_warper
                    curriculum_metrics = {
                        'success_rate': accumulated_success_rate,
                        'survival_rate': accumulated_survival_rate,
                        'mean_rot_angle': accumulated_mean_rot,
                        'mean_reward': accumulated_mean_reward,
                        'mean_entropy': torch.mean(torch.stack(entropies)).item() if entropies else 0.0,
                        # 附加信息用于调试
                        'accumulated_episodes': accumulated_episodes,
                    }
                else:
                    # 样本量不足时，使用当前 Epoch 的瞬时值（但不触发 curriculum update）
                    current_survival_rate = 1.0 - (self.fall_count / max(self.total_episodes, 1))
                    curriculum_metrics = {
                        'success_rate': current_success_rate,
                        'survival_rate': current_survival_rate,
                        'mean_rot_angle': mean_rot_angle,
                        'mean_reward': mean_rewards,
                        'mean_entropy': torch.mean(torch.stack(entropies)).item() if entropies else 0.0,
                        'accumulated_episodes': accumulated_episodes,
                        # 标记样本量不足，time_warper 可据此跳过门控判断
                        '_insufficient_samples': True,
                    }
                
                # [Bug Fix] 先保存更新前的 alpha，用于视频标签
                # 因为 time_warper.update() 会修改 current_alpha
                alpha_before_update = self.env.time_warper.current_alpha
                
                needs_update = self.env.time_warper.update(self.agent_steps, curriculum_metrics)
                
                # [修复] 无论是否触发课程更新，只要样本量足够并进行了一次检查，
                # 就清空累积缓冲区，开始新一轮统计。
                # 这避免了历史失败数据拖累当前表现，导致死锁。
                if has_enough_samples:
                    self.accumulated_stats = {
                        'total_episodes': 0,
                        'success_count': 0,
                        'fail_count': 0,
                        'rot_angle_sum': 0.0,
                        'reward_sum': 0.0,
                    }
                
                if needs_update:
                    # ---- 轨迹录制器：先导出旧 alpha 下的轨迹视频 ----
                    # 重要：使用 alpha_before_update 确保视频标签与内容一致
                    if self.trajectory_recorder is not None:
                        self.trajectory_recorder.export_on_curriculum_update(
                            writer=self.writer,
                            agent_steps=self.agent_steps,
                            current_alpha=alpha_before_update,  # [Fix] 使用更新前的 alpha
                            output_dir=self.output_dir
                        )
                    # 然后再应用新的物理参数（逻辑更顺畅：先总结过去，再开启未来）
                    self.env.apply_curriculum_physics()
                # 记录 curriculum 相关指标到 TensorBoard
                self.writer.add_scalar('curriculum/alpha', self.env.time_warper.current_alpha, self.agent_steps)
                self.writer.add_scalar('curriculum/progress', self.env.time_warper.progress, self.agent_steps)
                self.writer.add_scalar('curriculum/gravity_scale', self.env.time_warper.gravity_scale, self.agent_steps)
                # 使用累积统计的 survival_rate（如果样本足够）
                survival_rate_to_log = curriculum_metrics.get('survival_rate', 0.0)
                self.writer.add_scalar('curriculum/survival_rate', survival_rate_to_log, self.agent_steps)
                # 额外记录累积样本量，便于调试
                self.writer.add_scalar('curriculum/accumulated_episodes', accumulated_episodes, self.agent_steps)
            
            # ========================================================================================
            # [Enhanced Logging] Space E Curriculum Diagnostic & Tensorboard
            # ========================================================================================
            
            # 1. 获取课程学习的核心状态
            curriculum_active = hasattr(self.env, 'time_warper')
            if curriculum_active:
                tw = self.env.time_warper
                current_alpha = tw.current_alpha
                alpha_progress = tw.progress
                
                # 获取累积统计数据（用于判断是否满足门控）
                # 注意：这些是在本次 loop 中刚刚累积或清空的
                # 如果刚刚发生了 update，accumulated_stats 已经被清空，这里显示的是新一轮的起点
                acc_episodes = curriculum_metrics.get('accumulated_episodes', 0)
                # 使用 curriculum_metrics 中的值，因为它们可能是累积值
                curr_survival_rate = curriculum_metrics.get('survival_rate', 0.0)
                curr_success_rate = curriculum_metrics.get('success_rate', 0.0)
                
                # 判断当前未更新的原因
                update_status_msg = "N/A"
                if needs_update:
                    update_status_msg = f"✅ UPDATE TRIGGERED (Alpha -> {current_alpha:.3f})"
                else:
                    # 分析由于什么原因卡住了
                    reasons = []
                    # 原因 A: 样本不足
                    if acc_episodes < self.min_episodes_for_curriculum:
                        reasons.append(f"Insufficient Samples ({acc_episodes}/{self.min_episodes_for_curriculum})")
                    else:
                        # 原因 B: 门控未过
                        if current_alpha < 0.5:
                            if curr_survival_rate < 0.8:
                                reasons.append(f"Low Survival ({curr_survival_rate:.2f} < 0.8)")
                        else:
                            if curr_success_rate < 0.7:
                                reasons.append(f"Low Success ({curr_success_rate:.2f} < 0.7)")
                        
                        # 原因 C: 增量太小 (虽然门控过了，但 alpha 增长还没达到物理刷新阈值)
                        # 这是 time_warper 内部逻辑，通常不显示，但如果上面都过了，说明是这个原因
                        if not reasons:
                            reasons.append("Accumulating Alpha Increment (Ratio Threshold)")
                    
                    update_status_msg = f"❌ HOLD: {', '.join(reasons)}"

            # 2. 打印详细日志 (每 100 epoch 或 课程更新时)
            if self.epoch_num % 100 == 0 or (curriculum_active and needs_update):
                # 计算当前有效的成功阈值
                effective_threshold = self.success_rot_threshold
                if self.use_adaptive_threshold and hasattr(self.env, 'time_warper'):
                    effective_threshold = self.success_rot_threshold * self.env.time_warper.current_alpha
                
                print("\n" + "="*80)
                print(f"Space E Curriculum Status [Epoch {self.epoch_num}]")
                print("="*80)
                if curriculum_active:
                    print(f"  Current Alpha:      {current_alpha:.4f} (Gravity: {current_alpha**2:.4f}g)")
                    print(f"  Curriculum Status:  {update_status_msg}")
                    print(f"  Metrics (Accum.):   Survival={curr_survival_rate:.4f}, Success={curr_success_rate:.4f}")
                    print(f"  Sample Buffer:      {acc_episodes} episodes")
                    print("-"*80)
                
                print(f"  Episode Reward:     {mean_rewards:.2f}")
                print(f"  Episode Length:     {mean_lengths:.2f}")
                print(f"  Mean Rot Angle:     {mean_rot_angle:.2f} rad")
                print(f"  Current Success:    {current_success_rate:.4f} ({self.success_count}/{max(self.total_episodes, 1)}) rot>{effective_threshold:.1f}")
                
                # 打印终止原因分布
                term_counts = self.env.termination_counts
                total_terms = max(term_counts['total_episodes'], 1)
                print(f"  Termination Dist:   Total={term_counts['total_episodes']}")
                print(f"    - Max Steps:      {term_counts['max_episode_length']} ({term_counts['max_episode_length']/total_terms*100:.1f}%)")
                print(f"    - Drop (Low Z):   {term_counts['object_below_threshold']} ({term_counts['object_below_threshold']/total_terms*100:.1f}%)")
                print(f"    - Tilt (Fall):    {term_counts['pencil_tilt']} ({term_counts['pencil_tilt']/total_terms*100:.1f}%)")
                print(f"    - Overspeed:      {term_counts['angular_velocity_too_high']} ({term_counts['angular_velocity_too_high']/total_terms*100:.1f}%)")
                print("="*80 + "\n")
            
            # ========================================================================================
            # [周期性轨迹导出] 每 100 Epoch 导出最佳轨迹
            # ========================================================================================
            # 注意：这与 curriculum update 触发的导出是独立的
            # curriculum update 时会导出 "curriculum" 标签的视频，这里导出 "periodic" 标签的视频
            # 如果两者在同一 epoch 发生，也不冲突（缓冲区独立管理）
            if self.epoch_num % 100 == 0 and self.trajectory_recorder is not None:
                # 获取当前 alpha（如果有 time_warper）
                periodic_alpha = 1.0
                if hasattr(self.env, 'time_warper'):
                    periodic_alpha = self.env.time_warper.current_alpha
                
                # 尝试导出最佳轨迹
                self.trajectory_recorder.export(
                    writer=self.writer,
                    agent_steps=self.agent_steps,
                    current_alpha=periodic_alpha,
                    tag_prefix="periodic",
                    output_dir=self.output_dir
                )

            # 3. Tensorboard 记录
            self.writer.add_scalar('episode_rewards/step', mean_rewards, self.agent_steps)
            self.writer.add_scalar('episode_lengths/step', mean_lengths, self.agent_steps)
            self.writer.add_scalar('total_rot_angle(rad)/step', mean_rot_angle, self.agent_steps)
            self.writer.add_scalar('total_waypoint_tracking_reward/step', mean_waypoint_tracking_reward, self.agent_steps)
            self.writer.add_scalar('success_rate/step', current_success_rate, self.agent_steps) # 当前 epoch 的瞬时成功率
            self.writer.add_scalar('success_count/step', self.success_count, self.agent_steps)
            
            # [Add] Space E 特有指标
            if curriculum_active:
                # 记录累积的 survival rate (这才是决定课程进度的关键)
                self.writer.add_scalar('curriculum/accumulated_survival_rate', curr_survival_rate, self.agent_steps)
                self.writer.add_scalar('curriculum/accumulated_success_rate', curr_success_rate, self.agent_steps)
                # 记录导致失败的主要原因占比
                total_fails = self.fall_count # 这里用当前 epoch 的计数
                total_eps = max(self.total_episodes, 1)
                self.writer.add_scalar('failure/drop_rate', self.drop_count / total_eps, self.agent_steps)
                self.writer.add_scalar('failure/tilt_rate', self.tilt_count / total_eps, self.agent_steps)

            # 构建 checkpoint 文件名
            checkpoint_name = f'ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}m_reward_{mean_rewards:.2f}'

            # 如果设置了保存频率且当前平均奖励不高于历史最优，则保存模型
            if self.save_freq > 0:
                if (self.epoch_num % self.save_freq == 0) and (mean_rewards <= self.best_rewards):
                    self.save(os.path.join(self.nn_dir, checkpoint_name))
                    self.save(os.path.join(self.nn_dir, f'last'))

            # 如果当前平均奖励高于历史最优，且达到开始保存最优模型的步数，则保存最优模型
            if mean_rewards > self.best_rewards and self.agent_steps >= self.save_best_after:
                # print(f'save current best reward: {mean_rewards:.2f}')
                # 删除之前的最优模型文件
                prev_best_ckpt = os.path.join(self.nn_dir, f'best_reward_{self.best_rewards:.2f}.pth')
                if os.path.exists(prev_best_ckpt):
                    os.remove(prev_best_ckpt)
                # 更新历史最优奖励
                self.best_rewards = mean_rewards
                # 保存当前最优模型
                self.save(os.path.join(self.nn_dir, f'best_reward_{mean_rewards:.2f}'))
            self.env.reset_termination_counts()
            self.success_count = 0
            self.total_episodes = 0
            self.fall_count = 0
            self.drop_count = 0
            self.tilt_count = 0
        # 达到最大步数时打印信息
        print('max steps achieved')
        print(f'Final best reward: {self.best_rewards:.2f}')
        print(f'Final success rate: {self.best_success_rate:.4f}')
        print(f'Final mean rotation angle (rad): {mean_rot_angle:.4f}')
        # 返回三个独立指标供Optuna使用
        # best_reward: 最佳episode奖励
        # best_success_rate: 最佳成功率（旋转角度>6的比例）
        # mean_rot_angle: 最终旋转角度均值（弧度）
        return {
            'best_reward': self.best_rewards,
            'success_rate': self.best_success_rate,
            'mean_rot_angle': mean_rot_angle
        }

    # 保存模型权重和标准化统计信息
    def save(self, name):
        # 创建包含模型状态字典的 weights 字典
        weights = {
            'model': self.model.state_dict(),
        }
        # 如果使用了标准化，则将相应的 RunningMeanStd 状态字典也添加到 weights 字典
        if self.running_mean_std:
            weights['running_mean_std'] = self.running_mean_std.state_dict()
        if self.normalize_priv:
            weights['priv_mean_std'] = self.priv_mean_std.state_dict()
        if self.normalize_point_cloud:
            weights['point_cloud_mean_std'] = self.point_cloud_mean_std.state_dict()
        if self.value_mean_std:
            weights['value_mean_std'] = self.value_mean_std.state_dict()
        # 保存 weights 字典到文件
        torch.save(weights, f'{name}.pth')

    # 恢复训练过程，加载模型权重和标准化统计信息
    def restore_train(self, fn):
        # 如果文件路径为空，则直接返回
        if not fn:
            return
        print("restore_train: loading checkpoint from path", fn)
        # 加载 checkpoint 文件
        checkpoint = torch.load(fn, weights_only=False)
        # 加载模型状态字典
        self.model.load_state_dict(checkpoint['model'])
        # 加载标准化统计信息状态字典
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        if self.normalize_priv:
            self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.load_state_dict(checkpoint['point_cloud_mean_std'])

    # 恢复测试过程，加载模型权重和标准化统计信息
    def restore_test(self, fn):
        # 加载 checkpoint 文件
        checkpoint = torch.load(fn, weights_only=False)
        # 加载模型状态字典
        self.model.load_state_dict(checkpoint['model'])
        # 如果使用了标准化，则加载相应的 RunningMeanStd 状态字典
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        if self.normalize_priv:
            self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.load_state_dict(checkpoint['point_cloud_mean_std'])

    # 测试智能体性能
    # 测试智能体性能
    def test(self):
        # 设置模型为评估模式
        self.set_eval()
        # 重置环境，获取初始观察
        obs_dict = self.env.reset()

        while True:
            # Teacher模式：处理点云标准化
            if self.normalize_point_cloud:
                point_cloud = self.point_cloud_mean_std(
                    obs_dict['point_cloud_info'].reshape(-1, 3)
                ).reshape((obs_dict['obs'].shape[0], -1, 3))
            else:
                point_cloud = obs_dict['point_cloud_info']
            
            # 构建输入字典（Teacher使用完整特权信息）
            input_dict = {
                'obs': self.running_mean_std(obs_dict['obs']),
                'priv_info': self.priv_mean_std(obs_dict['priv_info']) if self.normalize_priv else obs_dict['priv_info'],
                'proprio_hist': obs_dict['proprio_hist'],
                'point_cloud_info': point_cloud,
            }
            
            # 调用模型进行推理模式的动作预测
            mu, extrin, extrin_gt = self.model.act_inference(input_dict)
            
            # 将预测的动作裁剪到 [-1, 1] 范围
            mu = torch.clamp(mu, -1.0, 1.0)
            
            # 在环境中执行动作，获取新的观察、奖励、done 标志和信息
            obs_dict, r, done, info = self.env.step(mu, extrin_record=extrin)

    def train_epoch(self):
        # 收集 minibatch 数据
        _t = time.time()
        # 设置为评估模式进行数据收集
        self.set_eval()
        # 执行 rollout 收集数据
        self.play_steps()
        # 累加数据收集时间
        self.data_collect_time += (time.time() - _t)
        # 更新网络
        _t = time.time()
        # 设置为训练模式进行模型更新
        self.set_train()
        # 初始化用于存储损失和统计数据的列表
        a_losses, b_losses, c_losses = [], [], []
        entropies, kls, grad_norms = [], [], []
        # 循环进行 mini-epoch 训练
        for _ in range(0, self.mini_epochs_num):
            ep_kls = [] # 用于存储当前 mini-epoch 的 KL 散度
            # 遍历 storage 中的 minibatch 数据
            for i in range(len(self.storage)):
                # 从 storage 中获取一个 minibatch 的数据
                value_preds, old_action_log_probs, advantage, old_mu, old_sigma, \
                    returns, actions, obs, priv_info, critic_info, point_cloud_info, proprio_hist, tactile_hist, obj_ends = self.storage[i]

                # 对观察进行标准化
                obs = self.running_mean_std(obs)
                # 如果配置了点云标准化，则进行标准化处理
                if self.normalize_point_cloud:
                    point_cloud_info = self.point_cloud_mean_std(point_cloud_info.reshape(-1, 3)).reshape((obs.shape[0], -1, 3))

                # 构建 batch 输入字典
                batch_dict = {
                    'prev_actions': actions, # 上一步的动作
                    'obs': obs, # 标准化后的观察
                    'priv_info': self.priv_mean_std(priv_info) if self.normalize_priv else priv_info, # 标准化或原始的特权信息
                    'critic_info': critic_info, # Critic 信息
                    'point_cloud_info': point_cloud_info, # 标准化或原始的点云信息
                    'obj_ends': obj_ends, # 物体末端信息
                    'proprio_hist': proprio_hist, # 本体感觉历史信息
                }
                # 调用模型进行前向传播，计算当前策略下的动作对数概率、价值、熵、均值和标准差
                res_dict = self.model(batch_dict)
                action_log_probs = res_dict['prev_neglogp'] # 当前策略的动作对数概率
                values = res_dict['values'] # 当前 Critic 估计的价值
                entropy = res_dict['entropy'] # 当前策略的熵
                mu = res_dict['mus'] # 当前策略的动作均值
                sigma = res_dict['sigmas'] # 当前策略的动作标准差

                # Actor 损失计算
                # 计算重要性采样比率 (ratio)
                ratio = torch.exp(old_action_log_probs - action_log_probs)
                # PPO 裁剪的第一个项： advantage * ratio
                surr1 = advantage * ratio
                # PPO 裁剪的第二个项： advantage * clipped_ratio
                surr2 = advantage * torch.clamp(ratio, 1.0 - self.e_clip, 1.0 + self.e_clip)
                # Actor 损失为 -min(surr1, surr2)
                a_loss = torch.max(-surr1, -surr2)
                # Critic 损失计算
                # 对价值预测进行裁剪
                value_pred_clipped = value_preds + (values - value_preds).clamp(-self.e_clip, self.e_clip)
                # 计算未裁剪的价值损失 (均方误差)
                value_losses = (values - returns) ** 2
                # 计算裁剪后的价值损失 (均方误差)
                value_losses_clipped = (value_pred_clipped - returns) ** 2
                # Critic 损失为 max(value_losses, value_losses_clipped)
                c_loss = torch.max(value_losses, value_losses_clipped)
                # 边界损失计算
                if self.bounds_loss_coef > 0:
                    # 定义软边界
                    soft_bound = 1.1
                    # 计算动作均值超出上软边界的惩罚
                    mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
                    # 计算动作均值超出下软边界的惩罚
                    mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
                    # 边界损失为上下边界惩罚之和
                    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
                else:
                    # 如果边界损失系数为 0，则边界损失为 0
                    b_loss = torch.zeros_like(mu)
                # 计算损失的平均值
                a_loss, c_loss, entropy, b_loss = [torch.mean(loss) for loss in [a_loss, c_loss, entropy, b_loss]]
                # 计算总损失：Actor 损失 + Critic 损失 * Critic 系数 - 熵 * 熵系数 + 边界损失 * 边界损失系数
                loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
                # 清空优化器梯度
                self.optimizer.zero_grad()
                # 反向传播计算梯度
                loss.backward()

                # 计算模型参数的梯度范数
                grad_norms.append(torch.norm(torch.cat([p.reshape(-1) for p in self.model.parameters()])))
                # 如果配置了梯度裁剪，则进行梯度裁剪
                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                # 优化器 step，更新模型参数
                self.optimizer.step()

                # 计算新旧策略之间的 KL 散度，用于自适应学习率调整
                with torch.no_grad():
                    kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)

                # 记录当前 minibatch 的 KL 散度、Actor 损失、Critic 损失、熵和边界损失
                kl = kl_dist
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                # 如果边界损失系数不为 None，则记录边界损失
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                # 更新 storage 中的 mu 和 sigma 为当前策略的值，用于下一次 minibatch 的 KL 计算
                self.storage.update_mu_sigma(mu.detach(), sigma.detach())

            # 计算当前 mini-epoch 的平均 KL 散度
            av_kls = torch.mean(torch.stack(ep_kls))
            kls.append(av_kls)

            # 使用自适应调度器更新学习率
            # [重构] 移除 CosineAnnealing 的约束，LR 仅由 KL 自适应控制
            self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())

        # [移除] 不再使用全局余弦退火调度器
        # self.global_scheduler.step()
        # global_lr = self.global_scheduler.get_last_lr()[0]
        # self.scheduler.max_lr = global_lr
        # self.last_lr = min(self.last_lr, global_lr)
        
        # 更新优化器的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.last_lr

        # 累加 RL 训练时间
        self.rl_train_time += (time.time() - _t)
        # 返回训练过程中收集的损失和统计数据
        return a_losses, c_losses, b_losses, entropies, kls, grad_norms

    # 执行 rollout 收集数据
    def play_steps(self):
        # 在 horizon length 步内与环境交互
        for n in range(self.horizon_length):

            # **使用当前模型预测动作和价值**
            res_dict = self.model_act(self.obs)

            # 收集当前观察 o_t 到 storage
            self.storage.update_data('obses', n, self.obs['obs'])
            # 收集特权信息到 storage
            self.storage.update_data('priv_info', n, self.obs['priv_info'])
            # 收集 critic 信息到 storage
            self.storage.update_data('critic_info', n, self.obs['critic_info'])
            # 收集点云信息到 storage
            self.storage.update_data('point_cloud_info', n, self.obs['point_cloud_info'])
            # 收集本体感觉历史信息到 storage
            self.storage.update_data('proprio_hist', n, self.obs['proprio_hist'])
            # 收集触觉历史信息到 storage
            self.storage.update_data('tactile_hist', n, self.obs['tactile_hist'])
            # 收集物体末端信息到 storage
            self.storage.update_data('obj_ends', n, self.obs['obj_ends'])
            # 收集动作、动作对数概率、价值、动作均值和标准差到 storage
            for k in ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']:
                self.storage.update_data(k, n, res_dict[k])
            # 在环境中执行动作
            # 将预测的动作裁剪到 [-1, 1] 范围
            actions = torch.clamp(res_dict['actions'], -1.0, 1.0)

            # render() is called during env.step() # 注释
            # to save time, save gif only per gif_save_every_n steps # 注释
            # 1 step = #gpu * #envs agent steps # 注释
            # 判断是否需要记录当前帧用于生成 GIF
            record_frame = False
            if self.gif_frame_counter >= self.gif_save_every_n and self.gif_frame_counter % self.gif_save_every_n < self.gif_save_length:
                record_frame = True
            # 只有在主进程且满足条件时才记录帧
            record_frame = record_frame and int(os.getenv('LOCAL_RANK', '0')) == 0
            # 根据是否记录帧来启用/禁用环境的相机传感器
            self.env.enable_camera_sensors = record_frame
            # 增加 GIF 帧计数器
            self.gif_frame_counter += 1

            # 在环境中执行动作，获取新的观察、奖励、done 标志和信息
            self.obs, rewards, self.dones, infos = self.env.step(actions)

            # 如果需要记录帧且环境支持相机，则捕捉帧并添加到 GIF 帧列表
            captured_frame = None
            if record_frame and self.env.with_camera:
                captured_frame = self.env.capture_frame()
                self.gif_frames.append(captured_frame)
                # add frame to GIF # 注释
                # 如果 GIF 帧列表达到指定长度，则将帧列表写入 TensorBoard 作为视频
                if len(self.gif_frames) == self.gif_save_length:
                    frame_array = np.array(self.gif_frames)[None]  # add batch axis # 添加 batch 维度
                    self.writer.add_video(
                        'rollout_gif', frame_array, global_step=self.agent_steps,
                        dataformats='NTHWC', fps=20,
                    )
                    # 刷新 writer，确保数据写入
                    self.writer.flush()
                    # 清空 GIF 帧列表
                    self.gif_frames.clear()

            # ---- 轨迹录制器：记录当前步 ----
            # 在课程学习中，追踪指定环境的完整轨迹
            if self.trajectory_recorder is not None:
                self.trajectory_recorder.record_step(
                    dones=self.dones,
                    infos=infos,
                    frame=captured_frame  # 复用已捕获的帧
                )

            # 将奖励 reshape 为 (batch_size, 1)
            rewards = rewards.unsqueeze(1)
            rot_angle = infos['rot_angle'].unsqueeze(1)
            waypoint_tracking_reward = infos['reward/waypoint_tracking_reward_per_env'].unsqueeze(1)
            # update dones and rewards after env step # 注释
            # 收集 done 标志到 storage
            self.storage.update_data('dones', n, self.dones)
            # 将奖励移动到指定设备
            rewards = rewards.to(self.device)
            rot_angle = rot_angle.to(self.device)
            waypoint_tracking_reward = waypoint_tracking_reward.to(self.device)
            # 计算 shaped rewards，这里简单乘以 0.01，可能根据具体环境设计 reward shaping
            shaped_rewards = 0.01 * rewards.clone()
            # 如果使用 value bootstrap 且 info 中包含 time_outs 信息，则在 shaped rewards 中加入 bootstrap 项
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * infos['time_outs'].unsqueeze(1).float()
            # 收集 shaped rewards 到 storage
            self.storage.update_data('rewards', n, shaped_rewards)

            # 累加当前 episode 的奖励和长度
            self.current_rewards += rewards
            self.current_rot_angle += rot_angle
            self.current_waypoint_tracking_reward += waypoint_tracking_reward
            self.current_lengths += 1
            # 找到 episode 结束的环境索引
            done_indices = self.dones.nonzero(as_tuple=False)
            # 更新 episode 奖励和长度的 AverageScalarMeter
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])
            self.total_rot_angle.update(self.current_rot_angle[done_indices])
            self.total_waypoint_tracking_reward.update(self.current_waypoint_tracking_reward[done_indices])
            # 统计成功案例（旋转角度>阈值，支持自适应阈值）
            if len(done_indices) > 0:
                # 计算有效阈值：如果使用自适应阈值，则乘以当前 alpha
                effective_threshold = self.success_rot_threshold
                if self.use_adaptive_threshold and hasattr(self.env, 'time_warper'):
                    effective_threshold = self.success_rot_threshold * self.env.time_warper.current_alpha
                
                success_mask = self.current_rot_angle[done_indices] > effective_threshold
                self.success_count += success_mask.sum().item()
                self.total_episodes += len(done_indices)
                
                # 统计失败案例（用于 survival_rate）
                # 从 infos 获取终止原因或从 env 的 termination_counts 获取
                if 'termination_reason' in infos:
                    # 每个 done 环境的终止原因
                    term_reasons = infos['termination_reason'][done_indices.squeeze(-1)]
                    # 2: object_below_threshold (掉落), 3: pencil_tilt (倾斜)
                    drop_mask = (term_reasons == 2)
                    tilt_mask = (term_reasons == 3)
                    self.drop_count += drop_mask.sum().item()
                    self.tilt_count += tilt_mask.sum().item()
                    self.fall_count += (drop_mask | tilt_mask).sum().item()
            # 确保 infos 是字典类型
            assert isinstance(infos, dict), 'Info Should be a Dict'
            # 更新额外信息字典，储存上一个时间步的信息
            self.extra_info = infos

            # 计算 not_dones 标志，用于重置已结束环境的奖励和长度计数器
            not_dones = (1.0 - self.dones.float()).to(self.device)

            # 重置已结束环境的当前奖励和长度计数器
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_rot_angle = self.current_rot_angle * not_dones.unsqueeze(1)
            self.current_waypoint_tracking_reward = self.current_waypoint_tracking_reward * not_dones.unsqueeze(1)    
            self.current_lengths = self.current_lengths * not_dones

        # rollout 结束后，使用最后一个状态的价值估计来计算 GAE 和 Returns
        res_dict = self.model_act(self.obs)
        last_values = res_dict['values']

        # 增加总 agent steps
        self.agent_steps = self.agent_steps + self.batch_size
        # 计算 GAE (advantage) 和 Returns
        self.storage.compute_return(last_values, self.gamma, self.tau)
        # 准备数据用于训练（例如，将数据展平或进行其他预处理）
        self.storage.prepare_training()

        # 获取计算好的 Returns 和 Values
        returns = self.storage.data_dict['returns']
        values = self.storage.data_dict['values']
        # 如果配置了价值标准化，则对 Returns 和 Values 进行标准化
        if self.normalize_value:
            self.value_mean_std.train() # 设置为训练模式以更新统计信息
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval() # 设置回评估模式
        # 将标准化后的 Values 和 Returns 更新回 storage
        self.storage.data_dict['values'] = values
        self.storage.data_dict['returns'] = returns


# 计算两个高斯分布之间的 KL 散度 (D_KL(p0 || p1))
# 其中 p0 是旧策略 (mu, sigma)，p1 是新策略 (p1_mu, p1_sigma)
def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    # 计算 KL 散度的第一项: log(sigma_1 / sigma_0)
    # 添加一个小的 epsilon 防止除以零或 log(0)
    c1 = torch.log(p1_sigma/p0_sigma + 1e-5)
    # 计算 KL 散度的第二项: (sigma_0^2 + (mu_1 - mu_0)^2) / (2 * sigma_1^2)
    # 添加一个小的 epsilon 防止除以零
    c2 = (p0_sigma ** 2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma ** 2 + 1e-5))
    # KL 散度的第三项: -0.5
    c3 = -1.0 / 2.0
    # 计算每个动作维度的 KL 散度
    kl = c1 + c2 + c3
    # 对所有动作维度求和，得到每个样本的 KL 散度
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions # 对所有步长的 KL 散度求和
    # 返回所有样本的平均 KL 散度
    return kl.mean()


# 从 https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/algorithms/ppo.py 引用
# 自适应学习率调度器，根据 KL 散度调整学习率
class AdaptiveScheduler(object):
    """自适应学习率调度器
    
    根据 KL 散度动态调整学习率:
    - KL > 2 * threshold: LR /= factor (策略变化太大，减小步幅)
    - KL < 0.5 * threshold: LR *= factor (策略变化太小，增大步幅)
    
    Args:
        kl_threshold: KL 散度阈值
        min_lr: 最小学习率 (防止死锁)
        max_lr: 最大学习率
        adaptive_factor: 调整因子 (默认 1.15)
    """
    def __init__(self, kl_threshold=0.01, min_lr=1e-6, max_lr=1e-2, adaptive_factor=1.15):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.kl_threshold = kl_threshold
        self.adaptive_factor = adaptive_factor

    def update(self, current_lr, kl_dist):
        """ 根据 KL 散度更新学习率 """
        lr = current_lr
        # 如果 KL 散度大于阈值的两倍，则降低学习率
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / self.adaptive_factor, self.min_lr)
        # 如果 KL 散度小于阈值的一半，则增加学习率
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * self.adaptive_factor, self.max_lr)
        return lr