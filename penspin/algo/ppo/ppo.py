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

# 从 penspin.algo.ppo 模块导入 ExperienceBuffer，用于存储智能体与环境交互的数据
from penspin.algo.ppo.experience import ExperienceBuffer
# 从 penspin.algo.models 模块导入重构后的 ActorCritic 模型
from penspin.algo.models.models import TeacherActorCritic, StudentActorCritic, create_actor_critic
# 从 penspin.algo.models 模块导入 RunningMeanStd，用于对输入进行标准化处理
from penspin.algo.models.running_mean_std import RunningMeanStd

# 从 penspin.utils.misc 模块导入 AverageScalarMeter，用于计算标量值的平均值
from penspin.utils.misc import AverageScalarMeter

# 导入 tensorboardX 中的 SummaryWriter，用于记录训练过程中的日志和可视化数据
from tensorboardX import SummaryWriter

# 导入统一的机器人配置常量
from penspin.utils.robot_config import (
    NUM_DOF, NUM_FINGERS, FINGERTIP_CNT, CONTACT_DIM, PROPRIO_DIM
)

# 定义 PPO 类，实现 Proximal Policy Optimization 算法
class PPO(object):
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
        # ---- 私有信息 (Priv Info) ---- **特权信息**
        # 获取私有信息的维度，通常指只有智能体能获取的环境信息，例如机器人关节速度、触觉信息等
        self.priv_info_dim = self.env.priv_info_dim
        # 配置是否使用私有信息
        self.priv_info = self.ppo_config['priv_info']
        # 配置是否使用本体感觉信息进行适应
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
        # 配置是否使用点云信息
        self.use_point_cloud_info = self.ppo_config['use_point_cloud_info']
        # ---- 输出目录 (Output Dir) ----
        # 指定实验结果保存的根目录
        self.output_dir = output_dif
        # 模型保存目录
        self.nn_dir = os.path.join(self.output_dir, 'stage1_nn')
        # TensorBoard 日志保存目录
        self.tb_dif = os.path.join(self.output_dir, 'stage1_tb')
        # 创建模型保存目录，如果已存在则不报错
        os.makedirs(self.nn_dir, exist_ok=True)
        # 创建 TensorBoard 日志保存目录，如果已存在则不报错
        os.makedirs(self.tb_dif, exist_ok=True)
        # ---- 模型 (Model) ----
        # 构建 Actor-Critic 模型的配置字典
        net_config = {
            'actor_units': self.network_config.mlp.units, # Actor MLP 的层单元数
            'priv_mlp_units': self.network_config.priv_mlp.units, # 私有信息 MLP 的层单元数
            'actions_num': self.actions_num, # 动作数量
            'input_shape': self.obs_shape, # 输入（观察）的形状
            'priv_info_dim': self.priv_info_dim, # 私有信息的维度
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
        # 初始化 RunningMeanStd 用于标准化私有信息
        self.priv_mean_std = RunningMeanStd(self.priv_info_dim).to(self.device)
        # 本体感觉维度（使用配置常量）
        self.proprio_dim = PROPRIO_DIM
        # 如果是 student 模型，点云标准化使用 6 维，否则使用 3 维
        if self.ppo_config.distill:
            self.point_cloud_mean_std = RunningMeanStd(6,).to(self.device)
        else:
            self.point_cloud_mean_std = RunningMeanStd(3,).to(self.device)
        # 初始化 RunningMeanStd 用于标准化价值函数输出
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)
        # ---- 优化器 (Optim) ----
        # 获取初始学习率
        self.last_lr = float(self.ppo_config['learning_rate'])
        # 获取权重衰减系数，如果未配置则默认为 0.0
        self.weight_decay = self.ppo_config.get('weight_decay', 0.0)
        # 初始化 Adam 优化器，优化模型的参数
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.last_lr, weight_decay=self.weight_decay)
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
        # 蒸馏损失系数
        self.distill_loss_coef = self.ppo_config['distill_loss_coef']
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
        # 是否标准化私有信息
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
        # 初始化自适应学习率调度器
        self.scheduler = AdaptiveScheduler(self.kl_threshold)
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
        # 每隔多少步保存一次 GIF
        self.gif_save_every_n = 7500
        # GIF 的帧长度
        self.gif_save_length = 600
        # 用于存储 GIF 帧的列表
        self.gif_frames = []

        # 初始化 AverageScalarMeter 用于记录 episode 奖励
        self.episode_rewards = AverageScalarMeter(20000)
        # 初始化 AverageScalarMeter 用于记录 episode 长度
        self.episode_lengths = AverageScalarMeter(20000)
        self.total_rot_angle = AverageScalarMeter(20000)
        self.total_sparse_reward = AverageScalarMeter(20000)
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

        # 初始化当前 episode 奖励和长度
        batch_size = self.num_actors
        current_rewards_shape = (batch_size, 1)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_rot_angle = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_sparse_reward = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
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
        # 如果配置了私有信息标准化，则将 priv_mean_std 也设置为评估模式
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
        # 如果配置了私有信息标准化，则将 priv_mean_std 也设置为训练模式
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
        # 获取私有信息
        priv_info = obs_dict['priv_info']
        # 如果配置了私有信息标准化，则进行标准化处理
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
            'priv_info': priv_info, # 标准化或原始的私有信息
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
        # 初始化 agent steps，每个 epoch 开始时增加一个 batch_size
        self.agent_steps = self.batch_size

        # 循环直到达到最大 agent steps
        while self.agent_steps < self.max_agent_steps:
            # 增加 epoch 计数
            self.epoch_num += 1
            # 执行一个训练 epoch（包括数据收集和模型更新）
            a_losses, c_losses, b_losses, entropies, kls, grad_norms = self.train_epoch()
            # 清空 storage 中的数据，准备下一个 epoch 的收集
            self.storage.data_dict = None

            # 更新额外信息字典，只保留标量值
            for k, v in self.extra_info.items():
                # only log scalars
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.extra_info[k] = v
                    # 打印奖励信息
                    print(f'{k}: {v}')

            # 计算整体 FPS 和最近一个 epoch 的 FPS
            # 平均每秒执行的 Agent step数。这是一个衡量系统整体长期运行效率的指标
            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            # 打印训练信息
            info_string = f'Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | ' \
                          f'Last FPS: {last_fps:.1f} | ' \
                          f'Collect Time: {self.data_collect_time / 60:.1f} min | ' \
                          f'Train RL Time: {self.rl_train_time / 60:.1f} min | ' \
                          f'Current Best: {self.best_rewards:.2f}'
            print(info_string)

            # 将统计数据写入 TensorBoard
            self.write_stats(a_losses, c_losses, b_losses, entropies, kls, grad_norms)
            # 获取 episode 奖励和长度的平均值
            mean_rewards = self.episode_rewards.get_mean()
            mean_lengths = self.episode_lengths.get_mean()
            mean_rot_angle = self.total_rot_angle.get_mean()
            mean_sparse_reward = self.total_sparse_reward.get_mean()
            print('-----------------------')
            print(f'episode rewards: {mean_rewards:.2f} | episode lengths: {mean_lengths:.2f}')
            print(f'episode rot angle(rad): {mean_rot_angle:.2f}')
            print(f'epsode sparse reward: {mean_sparse_reward:.2f}')
            # 记录 episode 奖励和长度的平均值到 TensorBoard
            self.writer.add_scalar('episode_rewards/step', mean_rewards, self.agent_steps)
            self.writer.add_scalar('episode_lengths/step', mean_lengths, self.agent_steps)
            self.writer.add_scalar('total_rot_angle(rad)/step', mean_rot_angle, self.agent_steps)
            self.writer.add_scalar('total_sparse_reward/step', mean_sparse_reward, self.agent_steps)

            # 构建 checkpoint 文件名
            checkpoint_name = f'ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}m_reward_{mean_rewards:.2f}'

            # 如果设置了保存频率且当前平均奖励不高于历史最优，则保存模型
            if self.save_freq > 0:
                if (self.epoch_num % self.save_freq == 0) and (mean_rewards <= self.best_rewards):
                    self.save(os.path.join(self.nn_dir, checkpoint_name))
                    self.save(os.path.join(self.nn_dir, f'last'))

            # 如果当前平均奖励高于历史最优，且达到开始保存最优模型的步数，则保存最优模型
            if mean_rewards > self.best_rewards and self.agent_steps >= self.save_best_after:
                print(f'save current best reward: {mean_rewards:.2f}')
                # 删除之前的最优模型文件
                prev_best_ckpt = os.path.join(self.nn_dir, f'best_reward_{self.best_rewards:.2f}.pth')
                if os.path.exists(prev_best_ckpt):
                    os.remove(prev_best_ckpt)
                # 更新历史最优奖励
                self.best_rewards = mean_rewards
                # 保存当前最优模型
                self.save(os.path.join(self.nn_dir, f'best_reward_{mean_rewards:.2f}'))

        # 达到最大步数时打印信息
        print('max steps achieved')

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
        checkpoint = torch.load(fn)
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
        checkpoint = torch.load(fn)
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
    def test(self):
        # 设置模型为评估模式
        self.set_eval()
        # 重置环境，获取初始观察
        obs_dict = self.env.reset()
        # import pickle # 注释掉的 pickle 导入和使用代码
        # num_frames = 0
        # 进入测试循环
        while True:
            # with open("replay_round2030_obs.pkl", "ab") as f: # 注释掉的代码
            #     pickle.dump(obs_dict['obs'], f) # 注释掉的代码
            # 如果不是蒸馏模式
            if not self.ppo_config.distill:
                # 如果配置了点云标准化，则进行标准化处理
                if self.normalize_point_cloud:
                    point_cloud = self.point_cloud_mean_std(
                        obs_dict['point_cloud_info'].reshape(-1, 3)
                    ).reshape((obs_dict['obs'].shape[0], -1, 3))
                else:
                    # 否则直接使用原始点云信息
                    point_cloud = obs_dict['point_cloud_info']
            # 如果是蒸馏模式，断言 Not ImplementedError，并使用 student_pc_info
            if self.ppo_config.distill:
                assert NotImplementedError # 在蒸馏模式下，此处需要实现具体的逻辑
                student_pc = obs_dict['student_pc_info']
                # temporary w/o one-hot # 注释掉的代码
                # student_pc = self.point_cloud_mean_std( # 注释掉的代码
                #             student_pc.reshape(-1, 6)[..., :3] # 注释掉的代码
                #             ).reshape((obs_dict['obs'].shape[0], -1, 3)) # 注释掉的代码
                # 构建输入字典（蒸馏模式）
                input_dict = {
                    'obs': self.running_mean_std(obs_dict['obs']), # 标准化后的观察
                    'student_pc_info': student_pc, # 学生点云信息
                }
            else:
                # 构建输入字典（非蒸馏模式）
                input_dict = {
                    'obs': self.running_mean_std(obs_dict['obs']), # 标准化后的观察
                    'priv_info': self.priv_mean_std(obs_dict['priv_info']) if self.normalize_priv else obs_dict['priv_info'], # 标准化或原始的私有信息
                    'proprio_hist': obs_dict['proprio_hist'], # 本体感觉历史信息
                    'point_cloud_info': point_cloud, # 标准化或原始的点云信息
                }
            # 调用模型进行推理模式的动作预测
            mu, extrin, extrin_gt = self.model.act_inference(input_dict)
            # assert extrin is not None # 注释掉的代码

            # 将预测的动作裁剪到 [-1, 1] 范围
            mu = torch.clamp(mu, -1.0, 1.0)
            # print(mu) # 注释掉的代码
            # with open("replay_round2030_action.pkl", "ab") as f: # 注释掉的代码
            #     pickle.dump(mu, f) # 注释掉的代码
            # 在环境中执行动作，获取新的观察、奖励、done 标志和信息
            obs_dict, r, done, info = self.env.step(mu, extrin_record=extrin)
            # num_frames += 1 # 注释掉的代码
            # print(num_frames) # 注释掉的代码
            # print(done.item()) # 注释掉的代码
            # if done.item(): # 注释掉的代码
            #     exit() # 注释掉的代码

    # 执行一个训练 epoch
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
                    'priv_info': self.priv_mean_std(priv_info) if self.normalize_priv else priv_info, # 标准化或原始的私有信息
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
            self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())
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
            # 收集私有信息到 storage
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
            if record_frame and self.env.with_camera:
                self.gif_frames.append(self.env.capture_frame())
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

            # 将奖励 reshape 为 (batch_size, 1)
            rewards = rewards.unsqueeze(1)
            rot_angle = infos['rot_angle'].unsqueeze(1)
            sparse_reward = infos['reward/waypoint_sparse_reward'].unsqueeze(1)
            # update dones and rewards after env step # 注释
            # 收集 done 标志到 storage
            self.storage.update_data('dones', n, self.dones)
            # 将奖励移动到指定设备
            rewards = rewards.to(self.device)
            rot_angle = rot_angle.to(self.device)
            sparse_reward = sparse_reward.to(self.device)
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
            self.current_sparse_reward += sparse_reward
            self.current_lengths += 1
            # 找到 episode 结束的环境索引
            done_indices = self.dones.nonzero(as_tuple=False)
            # 更新 episode 奖励和长度的 AverageScalarMeter
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])
            self.total_rot_angle.update(self.current_rot_angle[done_indices])
            self.total_sparse_reward.update(self.current_sparse_reward[done_indices])
            # 确保 infos 是字典类型
            assert isinstance(infos, dict), 'Info Should be a Dict'
            # 更新额外信息字典，储存上一个时间步的信息
            self.extra_info = infos

            # 计算 not_dones 标志，用于重置已结束环境的奖励和长度计数器
            not_dones = (1.0 - self.dones.float()).to(self.device)

            # 重置已结束环境的当前奖励和长度计数器
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_rot_angle = self.current_rot_angle * not_dones.unsqueeze(1)
            self.current_sparse_reward = self.current_sparse_reward * not_dones.unsqueeze(1)    
            self.current_lengths = self.current_lengths * not_dones

        # rollout 结束后，使用最后一个状态的价值估计来计算 GAE 和 Returns
        res_dict = self.model_act(self.obs)
        last_values = res_dict['values']

        # 增加总 agent steps
        self.agent_steps = self.agent_steps + self.batch_size
        # 计算 GAE (advantage) 和 Returns
        self.storage.computer_return(last_values, self.gamma, self.tau)
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
    # 构造函数，初始化调度器参数
    def __init__(self, kl_threshold=0.008):
        super().__init__()
        # 最小学习率
        self.min_lr = 1e-6
        # 最大学习率
        self.max_lr = 1e-2
        # KL 散度阈值
        self.kl_threshold = kl_threshold

    # 更新学习率
    def update(self, current_lr, kl_dist):
        # 当前学习率
        lr = current_lr
        # 如果 KL 散度大于阈值的两倍，则降低学习率
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        # 如果 KL 散度小于阈值的一半，则增加学习率
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        # 返回更新后的学习率
        return lr