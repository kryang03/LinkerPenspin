# --------------------------------------------------------
# PPO Teacher Trainer - Pure RL with Privileged Information
# 功能1: 直接通过纯PPO强化学习在仿真中用全部信息（包括特权信息）训练模型
# --------------------------------------------------------
# Based on: RLGames
# Copyright (c) 2019 Denys88
# Licence under MIT License
# https://github.com/Denys88/rl_games/
# --------------------------------------------------------

import os
import time
import torch
import numpy as np

from penspin.algo.ppo.experience import ExperienceBuffer
from penspin.algo.models.models import TeacherActorCritic
from penspin.algo.models.running_mean_std import RunningMeanStd
from penspin.utils.misc import AverageScalarMeter
from tensorboardX import SummaryWriter

from penspin.utils.robot_config import (
    NUM_DOF, NUM_FINGERS, FINGERTIP_CNT, CONTACT_DIM, PROPRIO_DIM
)


class PPOTeacher(object):
    """
    纯PPO训练器 - 使用完整特权信息训练Teacher模型
    
    特点：
    - 使用完整的特权信息 (priv_info: 47维，包括物体状态、指尖位置、触觉等)
    - 使用点云信息 (point_cloud: 100×3)
    - 纯强化学习，没有BC loss
    - 适合在仿真环境中训练expert策略
    """
    
    def __init__(self, env, output_dir, full_config):
        self.device = full_config['rl_device']
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        
        # ---- 环境设置 ----
        self.env = env
        self.num_actors = self.ppo_config['num_actors']
        action_space = self.env.action_space
        self.actions_num = action_space.shape[0]
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.device)
        self.observation_space = self.env.observation_space
        self.obs_shape = self.observation_space.shape
        
        # ---- 特权信息 ----
        self.priv_info_dim = self.env.priv_info_dim
        self.priv_info = True  # Teacher 始终使用特权信息
        
        # ---- Critic信息 ----
        self.asymm_actor_critic = self.ppo_config['asymm_actor_critic']
        self.critic_info_dim = self.ppo_config['critic_info_dim']
        
        # ---- 点云信息 ----
        self.point_cloud_buffer_dim = self.env.point_cloud_buffer_dim
        self.use_point_cloud_info = True  # Teacher 使用点云
        
        # ---- 本体感觉信息 ----
        self.proprio_dim = PROPRIO_DIM
        self.proprio_len = self.ppo_config['proprio_len']
        
        # ---- 输出目录 ----
        self.output_dir = output_dir
        self.nn_dir = os.path.join(self.output_dir, 'teacher_nn')
        self.tb_dir = os.path.join(self.output_dir, 'teacher_tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        
        # ---- Teacher模型 ----
        net_config = {
            'actor_units': self.network_config.mlp.units,
            'priv_mlp_units': self.network_config.priv_mlp.units,
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'priv_info_dim': self.priv_info_dim,
            'critic_info_dim': self.critic_info_dim,
            'asymm_actor_critic': self.asymm_actor_critic,
            'point_mlp_units': self.network_config.point_mlp.units,
            'use_point_transformer': self.network_config.use_point_transformer,
        }
        self.model = TeacherActorCritic(net_config)
        self.model.to(self.device)
        
        # ---- 标准化 ----
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.priv_mean_std = RunningMeanStd(self.priv_info_dim).to(self.device)
        self.point_cloud_mean_std = RunningMeanStd(3).to(self.device)
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)
        
        self.normalize_input = self.ppo_config['normalize_input']
        self.normalize_priv = self.ppo_config['normalize_priv']
        self.normalize_point_cloud = self.ppo_config['normalize_point_cloud']
        self.normalize_value = self.ppo_config['normalize_value']
        
        # ---- 优化器 ----
        self.last_lr = float(self.ppo_config['learning_rate'])
        self.weight_decay = self.ppo_config.get('weight_decay', 0.0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.last_lr, 
                                         weight_decay=self.weight_decay)
        
        # ---- PPO超参数 ----
        self.e_clip = self.ppo_config['e_clip']
        self.clip_value = self.ppo_config['clip_value']
        self.entropy_coef = self.ppo_config['entropy_coef']
        self.critic_coef = self.ppo_config['critic_coef']
        self.bounds_loss_coef = self.ppo_config['bounds_loss_coef']
        self.gamma = self.ppo_config['gamma']
        self.tau = self.ppo_config['tau']
        self.truncate_grads = self.ppo_config['truncate_grads']
        self.grad_norm = self.ppo_config['grad_norm']
        self.value_bootstrap = self.ppo_config['value_bootstrap']
        self.normalize_advantage = self.ppo_config['normalize_advantage']
        
        # ---- 数据收集参数 ----
        self.horizon_length = self.ppo_config['horizon_length']
        self.batch_size = self.horizon_length * self.num_actors
        self.minibatch_size = self.ppo_config['minibatch_size']
        self.mini_epochs_num = self.ppo_config['mini_epochs']
        assert self.batch_size % self.minibatch_size == 0 or full_config.test
        
        # ---- 学习率调度 ----
        self.kl_threshold = self.ppo_config['kl_threshold']
        self.scheduler = AdaptiveScheduler(self.kl_threshold)
        
        # ---- 保存设置 ----
        self.save_freq = self.ppo_config['save_frequency']
        self.save_best_after = self.ppo_config['save_best_after']
        
        # ---- TensorBoard ----
        self.extra_info = {}
        self.writer = SummaryWriter(self.tb_dir)
        
        # ---- 统计信息 ----
        self.episode_rewards = AverageScalarMeter(20000)
        self.episode_lengths = AverageScalarMeter(20000)
        self.total_rot_angle = AverageScalarMeter(20000)
        
        # ---- 经验缓冲 ----
        self.storage = ExperienceBuffer(
            self.num_actors, self.horizon_length, self.batch_size, self.minibatch_size,
            self.obs_shape[0], self.actions_num, self.priv_info_dim, self.critic_info_dim,
            self.point_cloud_buffer_dim, self.device, self.proprio_dim, self.proprio_len
        )
        
        # ---- 训练状态 ----
        batch_size = self.num_actors
        current_rewards_shape = (batch_size, 1)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_rot_angle = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)
        
        self.obs = None
        self.epoch_num = 0
        self.agent_steps = 0
        self.max_agent_steps = self.ppo_config['max_agent_steps']
        self.best_rewards = -10000
        
        # ---- 计时 ----
        self.data_collect_time = 0
        self.rl_train_time = 0
        
        print("=" * 80)
        print("PPO Teacher Trainer 初始化完成")
        print(f"模型: TeacherActorCritic")
        print(f"特权信息维度: {self.priv_info_dim}")
        print(f"点云信息: 100×3")
        print(f"训练模式: 纯PPO强化学习")
        print("=" * 80)
    
    def set_eval(self):
        """设置为评估模式"""
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()
        if self.normalize_priv:
            self.priv_mean_std.eval()
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.eval()
        if self.normalize_value:
            self.value_mean_std.eval()
    
    def set_train(self):
        """设置为训练模式"""
        self.model.train()
        if self.normalize_input:
            self.running_mean_std.train()
        if self.normalize_priv:
            self.priv_mean_std.train()
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.train()
        if self.normalize_value:
            self.value_mean_std.train()
    
    def model_act(self, obs_dict):
        """模型动作采样"""
        # 标准化输入
        processed_obs = self.running_mean_std(obs_dict['obs']) if self.normalize_input else obs_dict['obs']
        priv_info = self.priv_mean_std(obs_dict['priv_info']) if self.normalize_priv else obs_dict['priv_info']
        
        if self.normalize_point_cloud:
            point_cloud = self.point_cloud_mean_std(
                obs_dict['point_cloud_info'].reshape(-1, 3)
            ).reshape((processed_obs.shape[0], -1, 3))
        else:
            point_cloud = obs_dict['point_cloud_info']
        
        input_dict = {
            'obs': processed_obs,
            'priv_info': priv_info,
            'point_cloud_info': point_cloud,
            'critic_info': obs_dict['critic_info'],
            'proprio_hist': obs_dict['proprio_hist'],
        }
        
        res_dict = self.model.act(input_dict)
        
        # 标准化value
        if self.normalize_value:
            res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        
        return res_dict
    
    def train(self):
        """主训练循环"""
        _t = time.time()
        _last_t = time.time()
        
        # 初始化环境
        self.obs = self.env.reset()
        self.obs['proprio_hist'] = self.obs['proprio_hist'][..., -self.proprio_len:, :self.proprio_dim]
        self.agent_steps = self.batch_size
        
        print("\n开始训练 PPO Teacher...")
        
        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            a_losses, c_losses, b_losses, entropies, kls, grad_norms = self.train_epoch()
            self.storage.data_dict = None
            
            # 计算FPS
            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            
            # 打印信息
            mean_rewards = self.episode_rewards.get_mean()
            mean_lengths = self.episode_lengths.get_mean()
            
            info_string = (f'Epoch: {self.epoch_num:04} | Steps: {int(self.agent_steps // 1e6):04}M | '
                          f'FPS: {all_fps:.1f} | Last FPS: {last_fps:.1f} | '
                          f'Reward: {mean_rewards:.2f} | Best: {self.best_rewards:.2f} | '
                          f'Collect: {self.data_collect_time / 60:.1f}min | '
                          f'Train: {self.rl_train_time / 60:.1f}min')
            print(info_string)
            
            # 写入TensorBoard
            self.write_stats(a_losses, c_losses, b_losses, entropies, kls, grad_norms)
            self.writer.add_scalar('episode_rewards/step', mean_rewards, self.agent_steps)
            self.writer.add_scalar('episode_lengths/step', mean_lengths, self.agent_steps)
            
            # 保存模型
            if self.save_freq > 0 and self.epoch_num % self.save_freq == 0:
                checkpoint_name = f'ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}m_reward_{mean_rewards:.2f}'
                self.save(os.path.join(self.nn_dir, checkpoint_name))
                self.save(os.path.join(self.nn_dir, 'last'))
            
            # 保存最佳模型
            if mean_rewards > self.best_rewards and self.agent_steps >= self.save_best_after:
                print(f'  ✓ 新的最佳奖励: {mean_rewards:.2f} (之前: {self.best_rewards:.2f})')
                self.best_rewards = mean_rewards
                self.save(os.path.join(self.nn_dir, f'best_reward_{mean_rewards:.2f}'))
        
        print('训练完成！达到最大步数。')
    
    def train_epoch(self):
        """训练一个epoch"""
        # 数据收集
        _t = time.time()
        self.set_eval()
        self.play_steps()
        self.data_collect_time += (time.time() - _t)
        
        # 更新网络
        _t = time.time()
        self.set_train()
        
        a_losses, c_losses, b_losses, entropies, kls, grad_norms = [], [], [], [], [], []
        
        for _ in range(self.mini_epochs_num):
            for i in range(len(self.storage)):
                (value_preds, old_action_log_probs, advantage, old_mu, old_sigma, returns, actions, obs,
                 priv_info, critic_info, point_cloud_info, proprio_hist, tactile_hist, obj_ends) = self.storage[i]
                
                # 准备输入
                batch_dict = {
                    'obs': self.running_mean_std(obs) if self.normalize_input else obs,
                    'priv_info': self.priv_mean_std(priv_info) if self.normalize_priv else priv_info,
                    'point_cloud_info': self.point_cloud_mean_std(point_cloud_info.reshape(-1, 3)).reshape(
                        (obs.shape[0], -1, 3)) if self.normalize_point_cloud else point_cloud_info,
                    'critic_info': critic_info,
                    'proprio_hist': proprio_hist,
                }
                
                # 前向传播
                mu, sigma, value, extrin, extrin_gt = self.model._actor_critic(batch_dict)
                
                # 计算PPO损失
                a_loss, c_loss, b_loss, entropy, kl, grad_norm = self.calc_ppo_loss(
                    old_action_log_probs, value_preds, advantage, returns, actions,
                    obs, mu, sigma, value, old_mu, old_sigma
                )
                
                # 反向传播
                self.optimizer.zero_grad()
                loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + self.bounds_loss_coef * b_loss
                loss.backward()
                
                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                
                self.optimizer.step()
                
                # 记录统计信息
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                b_losses.append(b_loss)
                entropies.append(entropy)
                kls.append(kl)
                grad_norms.append(grad_norm)
                
                # 更新存储的mu和sigma
                self.storage.update_mu_sigma(mu.detach(), sigma.detach())
        
        # 更新学习率
        av_kls = torch.mean(torch.stack(kls))
        self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.last_lr
        
        self.rl_train_time += (time.time() - _t)
        return a_losses, c_losses, b_losses, entropies, kls, grad_norms
    
    def play_steps(self):
        """收集rollout数据"""
        for n in range(self.horizon_length):
            res_dict = self.model_act(self.obs)
            
            # 存储数据
            self.storage.update_data('obses', n, self.obs['obs'])
            self.storage.update_data('priv_info', n, self.obs['priv_info'])
            self.storage.update_data('point_cloud_info', n, self.obs['point_cloud_info'])
            self.storage.update_data('proprio_hist', n, self.obs['proprio_hist'])
            self.storage.update_data('critic_info', n, self.obs['critic_info'])
            
            for k in ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']:
                self.storage.update_data(k, n, res_dict[k])
            
            # 执行动作
            actions = torch.clamp(res_dict['actions'], -1.0, 1.0)
            self.obs, rewards, self.dones, infos = self.env.step(actions)
            self.obs['proprio_hist'] = self.obs['proprio_hist'][..., -self.proprio_len:, :self.proprio_dim]
            
            rewards = rewards.unsqueeze(1)
            
            # 更新done和reward
            self.storage.update_data('dones', n, self.dones)
            self.storage.update_data('rewards', n, rewards)
            
            # 更新episode统计
            self.current_rewards += rewards
            self.current_lengths += 1
            
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])
            
            self.extra_info = infos
            
            not_dones = (1.0 - self.dones.float()).to(self.device)
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
        
        # 计算最后的value
        res_dict = self.model_act(self.obs)
        last_values = res_dict['values']
        
        self.agent_steps += self.batch_size
        
        # 准备训练数据
        self.storage.compute_return(last_values, self.gamma, self.tau)
        self.storage.prepare_training()
    
    def calc_ppo_loss(self, old_action_log_probs, value_preds, advantage, returns, actions,
                      obs, mu, sigma, value, old_mu, old_sigma):
        """计算PPO损失"""
        # Actor loss
        log_std = torch.log(sigma)
        action_log_probs = gaussian_log_prob(actions, mu, log_std)
        ratio = torch.exp(action_log_probs - old_action_log_probs)
        
        if self.normalize_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - self.e_clip, 1.0 + self.e_clip)
        a_loss = -torch.min(surr1, surr2).mean()
        
        # Critic loss
        if self.clip_value:
            value_pred_clipped = value_preds + torch.clamp(value - value_preds, 
                                                            -self.e_clip, self.e_clip)
            value_losses = (value - returns).pow(2)
            value_losses_clipped = (value_pred_clipped - returns).pow(2)
            c_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        else:
            c_loss = 0.5 * (returns - value).pow(2).mean()
        
        # Entropy
        entropy = -(log_std + 0.5 * np.log(2 * np.pi * np.e)).sum(dim=-1).mean()
        
        # Bounds loss
        if self.bounds_loss_coef > 0:
            soft_bound = 1.0
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0).pow(2)
            mu_loss_low = torch.clamp_min(-soft_bound - mu, 0.0).pow(2)
            b_loss = (mu_loss_high + mu_loss_low).sum(axis=-1).mean()
        else:
            b_loss = torch.tensor(0.0, device=self.device)
        
        # KL
        kl = policy_kl(old_mu, old_sigma, mu, sigma)
        
        # Grad norm
        grad_norm = torch.norm(torch.cat([p.reshape(-1) for p in self.model.parameters()]))
        
        return a_loss, c_loss, b_loss, entropy, kl, grad_norm
    
    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls, grad_norms):
        """写入TensorBoard统计信息"""
        self.writer.add_scalar('performance/RLTrainFPS', self.agent_steps / self.rl_train_time, self.agent_steps)
        self.writer.add_scalar('performance/EnvStepFPS', self.agent_steps / self.data_collect_time, self.agent_steps)
        
        self.writer.add_scalar('losses/actor_loss', torch.mean(torch.stack(a_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/bounds_loss', torch.mean(torch.stack(b_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/critic_loss', torch.mean(torch.stack(c_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/entropy', torch.mean(torch.stack(entropies)).item(), self.agent_steps)
        
        self.writer.add_scalar('info/last_lr', self.last_lr, self.agent_steps)
        self.writer.add_scalar('info/e_clip', self.e_clip, self.agent_steps)
        self.writer.add_scalar('info/kl', torch.mean(torch.stack(kls)).item(), self.agent_steps)
        self.writer.add_scalar('info/grad_norms', torch.mean(torch.stack(grad_norms)).item(), self.agent_steps)
        
        for k, v in self.extra_info.items():
            if isinstance(v, (float, int)) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                self.writer.add_scalar(f'{k}', v, self.agent_steps)
    
    def save(self, name):
        """保存模型"""
        weights = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch_num,
            'agent_steps': self.agent_steps,
        }
        if self.normalize_input:
            weights['running_mean_std'] = self.running_mean_std.state_dict()
        if self.normalize_priv:
            weights['priv_mean_std'] = self.priv_mean_std.state_dict()
        if self.normalize_point_cloud:
            weights['point_cloud_mean_std'] = self.point_cloud_mean_std.state_dict()
        if self.normalize_value:
            weights['value_mean_std'] = self.value_mean_std.state_dict()
        
        torch.save(weights, f'{name}.pth')
    
    def restore_train(self, fn):
        """训练模式加载模型"""
        if not fn:
            return
        print(f"加载训练checkpoint: {fn}")
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch_num = checkpoint.get('epoch', 0)
        self.agent_steps = checkpoint.get('agent_steps', 0)
        
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        if self.normalize_priv and 'priv_mean_std' in checkpoint:
            self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])
        if self.normalize_point_cloud and 'point_cloud_mean_std' in checkpoint:
            self.point_cloud_mean_std.load_state_dict(checkpoint['point_cloud_mean_std'])
        if self.normalize_value and 'value_mean_std' in checkpoint:
            self.value_mean_std.load_state_dict(checkpoint['value_mean_std'])
    
    def restore_test(self, fn):
        """测试模式加载模型"""
        if not fn:
            return
        print(f"加载测试checkpoint: {fn}")
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint['model'])
        
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        if self.normalize_priv and 'priv_mean_std' in checkpoint:
            self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])
        if self.normalize_point_cloud and 'point_cloud_mean_std' in checkpoint:
            self.point_cloud_mean_std.load_state_dict(checkpoint['point_cloud_mean_std'])
        if self.normalize_value and 'value_mean_std' in checkpoint:
            self.value_mean_std.load_state_dict(checkpoint['value_mean_std'])
    
    def test(self):
        """测试模式"""
        self.set_eval()
        obs_dict = self.env.reset()
        obs_dict['proprio_hist'] = obs_dict['proprio_hist'][..., -self.proprio_len:, :self.proprio_dim]
        
        while True:
            processed_obs = self.running_mean_std(obs_dict['obs']) if self.normalize_input else obs_dict['obs']
            priv_info = self.priv_mean_std(obs_dict['priv_info']) if self.normalize_priv else obs_dict['priv_info']
            
            if self.normalize_point_cloud:
                point_cloud = self.point_cloud_mean_std(
                    obs_dict['point_cloud_info'].reshape(-1, 3)
                ).reshape((obs_dict['obs'].shape[0], -1, 3))
            else:
                point_cloud = obs_dict['point_cloud_info']
            
            input_dict = {
                'obs': processed_obs,
                'priv_info': priv_info,
                'point_cloud_info': point_cloud,
                'critic_info': obs_dict['critic_info'],
                'proprio_hist': obs_dict['proprio_hist'],
            }
            
            mu, extrin, extrin_gt = self.model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            
            obs_dict, r, done, info = self.env.step(mu, extrin_record=extrin)
            obs_dict['proprio_hist'] = obs_dict['proprio_hist'][..., -self.proprio_len:, :self.proprio_dim]


# ---- 辅助函数 ----

def gaussian_log_prob(x, mu, log_std):
    """计算高斯分布的对数概率"""
    return -0.5 * (((x - mu) / (torch.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    """计算两个高斯策略之间的KL散度"""
    c1 = torch.log(p1_sigma / p0_sigma + 1e-5)
    c2 = (p0_sigma ** 2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma ** 2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)
    return kl.mean()


class AdaptiveScheduler(object):
    """自适应学习率调度器"""
    def __init__(self, kl_threshold=0.008):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr
