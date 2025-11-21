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

import os
import time
import torch
import torch.distributed as dist
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingLR

from penspin.algo.ppo.experience import ExperienceBuffer
from penspin.algo.models.models import TeacherActorCritic, StudentActorCritic, create_actor_critic
from penspin.algo.models.running_mean_std import RunningMeanStd

from penspin.utils.misc import AverageScalarMeter

from tensorboardX import SummaryWriter

# 导入统一的机器人配置常量
from penspin.utils.robot_config import (
    NUM_DOF, NUM_FINGERS, FINGERTIP_CNT, CONTACT_DIM, PROPRIO_DIM
)

class PPO_RL_BC_Teacher(object):
    def __init__(self, env, output_dif, full_config):
        self.device = full_config['rl_device']
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        self.demon_path = full_config.train.demon_path
        # ---- build environment ----
        self.env = env
        self.num_actors = self.ppo_config['num_actors']
        action_space = self.env.action_space
        self.actions_num = action_space.shape[0]
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.device)
        self.observation_space = self.env.observation_space
        self.obs_shape = self.observation_space.shape
        # ---- Priv Info ----
        self.priv_info_dim = self.env.priv_info_dim
        self.enable_latent_loss = self.ppo_config['enable_latent_loss']
        self.use_l1 = self.ppo_config['use_l1']
        self.priv_info = self.ppo_config['priv_info']
        self.proprio_adapt = self.ppo_config['proprio_adapt']
        # ---- Critic Info
        self.asymm_actor_critic = self.ppo_config['asymm_actor_critic']
        self.critic_info_dim = self.ppo_config['critic_info_dim']
        # ---- Point Cloud Info
        self.point_cloud_buffer_dim = self.env.point_cloud_buffer_dim
        self.proprio_mode = self.ppo_config['proprio_mode']
        self.input_mode = self.ppo_config['input_mode']
        self.proprio_len = self.ppo_config['proprio_len']
        # ---- Output Dir ----
        # allows us to specify a folder where all experiments will reside
        self.output_dir = output_dif
        self.nn_dir = os.path.join(self.output_dir, 'stage1_nn')
        self.tb_dif = os.path.join(self.output_dir, 'stage1_tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dif, exist_ok=True)
        # ---- Demonstration Model (Teacher) ----
        demon_net_config = {
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
        self.proprio_dim = PROPRIO_DIM  # 使用配置常量

        # 创建 Teacher 模型
        self.demon_model = TeacherActorCritic(demon_net_config)
        self.demon_model.to(self.device)
        self.running_mean_std_demon = RunningMeanStd(self.obs_shape).to(self.device)
        self.priv_mean_std_demon = RunningMeanStd(self.priv_info_dim).to(self.device)
        self.proprio_mean_std_demon = RunningMeanStd(self.proprio_len*self.proprio_dim).to(self.device)
        self.point_cloud_mean_std_demon = RunningMeanStd(3,).to(self.device)
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)
        self.demon_load(self.demon_path)

        # ---- Student Model ----
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
            'proprio_len': self.proprio_len,
            'input_mode': self.input_mode,
            'use_point_cloud_info': True,  # Student uses point cloud
        }
        # 创建 Student 模型
        self.model = StudentActorCritic(net_config)
        self.model.to(self.device)

        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.priv_mean_std = RunningMeanStd(self.priv_info_dim).to(self.device)
        self.proprio_mean_std = RunningMeanStd(self.proprio_len*self.proprio_dim).to(self.device)
        self.point_cloud_mean_std = RunningMeanStd(3,).to(self.device)
        
        # ---- Optim ----
        self.last_lr = float(self.ppo_config['learning_rate'])
        self.weight_decay = self.ppo_config.get('weight_decay', 0.0)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), self.last_lr, weight_decay=self.weight_decay)
        # ---- PPO Train Param ----
        self.e_clip = self.ppo_config['e_clip']
        self.clip_value = self.ppo_config['clip_value']
        self.entropy_coef = self.ppo_config['entropy_coef']
        self.critic_coef = self.ppo_config['critic_coef']
        self.bounds_loss_coef = self.ppo_config['bounds_loss_coef']
        self.distill_loss_coef = self.ppo_config['distill_loss_coef']
        
        # ---- Optim ----
        self.last_lr = float(self.ppo_config['learning_rate'])
        self.weight_decay = self.ppo_config.get('weight_decay', 0.0)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), self.last_lr, weight_decay=self.weight_decay)
        
        # ---- Global Learning Rate Scheduler ----
        self.max_steps = self.ppo_config['max_agent_steps']
        # 估算总共的 epoch 数量
        total_epochs = self.max_steps // (self.ppo_config['horizon_length'] * self.ppo_config['num_actors'])
        # 使用余弦退火调度器，让学习率在整个训练过程中平滑下降
        self.global_scheduler = CosineAnnealingLR(self.optimizer, T_max=total_epochs, eta_min=1e-6)
        
        # ---- PPO Train Param ----
        self.e_clip = self.ppo_config['e_clip']
        self.clip_value = self.ppo_config['clip_value']
        self.entropy_coef = self.ppo_config['entropy_coef']
        self.critic_coef = self.ppo_config['critic_coef']
        self.bounds_loss_coef = self.ppo_config['bounds_loss_coef']
        self.bc_loss_coef = self.ppo_config.get('bc_loss_coef', 1.0)  # 添加默认值
        self.gamma = self.ppo_config['gamma']
        self.tau = self.ppo_config['tau']
        self.truncate_grads = self.ppo_config['truncate_grads']
        self.grad_norm = self.ppo_config['grad_norm']
        self.value_bootstrap = self.ppo_config['value_bootstrap']
        self.normalize_advantage = self.ppo_config['normalize_advantage']
        self.normalize_input = self.ppo_config['normalize_input']
        self.normalize_value = self.ppo_config['normalize_value']
        self.normalize_priv = self.ppo_config['normalize_priv']
        self.normalize_point_cloud = self.ppo_config['normalize_point_cloud']
        # ---- PPO Collect Param ----
        self.horizon_length = self.ppo_config['horizon_length']
        self.batch_size = self.horizon_length * self.num_actors
        self.minibatch_size = self.ppo_config['minibatch_size']
        self.mini_epochs_num = self.ppo_config['mini_epochs']
        assert self.batch_size % self.minibatch_size == 0 or full_config.test
        # ---- scheduler ----
        self.kl_threshold = self.ppo_config['kl_threshold']
        self.scheduler = AdaptiveScheduler(self.kl_threshold, min_lr=1e-6, max_lr=self.last_lr)
        # ---- Snapshot
        self.save_freq = self.ppo_config['save_frequency']
        self.save_best_after = self.ppo_config['save_best_after']
        # ---- Tensorboard Logger ----
        self.extra_info = {}
        writer = SummaryWriter(self.tb_dif)
        self.writer = writer

        self.episode_rewards = AverageScalarMeter(20000)
        self.episode_lengths = AverageScalarMeter(20000)
        self.obs = None
        self.epoch_num = 0
        self.storage = ExperienceBuffer(
            self.num_actors, self.horizon_length, self.batch_size, self.minibatch_size, self.obs_shape[0],
            self.actions_num, self.priv_info_dim, self.critic_info_dim, self.point_cloud_buffer_dim, self.device,
            self.proprio_dim, self.proprio_len
        )

        batch_size = self.num_actors
        current_rewards_shape = (batch_size, 1)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = self.ppo_config['max_agent_steps']
        self.best_rewards = -10000
        # ---- Timing
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0

    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls, grad_norms, latent_losses):
        self.writer.add_scalar('performance/RLTrainFPS', self.agent_steps / self.rl_train_time, self.agent_steps)
        self.writer.add_scalar('performance/EnvStepFPS', self.agent_steps / self.data_collect_time, self.agent_steps)

        self.writer.add_scalar('losses/actor_loss', torch.mean(torch.stack(a_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/critic_loss', torch.mean(torch.stack(c_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/bc_loss', torch.mean(torch.stack(b_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/latent_loss', torch.mean(torch.stack(latent_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/entropy', torch.mean(torch.stack(entropies)).item(), self.agent_steps)
        self.writer.add_scalar('info/last_lr', self.last_lr, self.agent_steps)
        self.writer.add_scalar('info/e_clip', self.e_clip, self.agent_steps)
        self.writer.add_scalar('info/kl', torch.mean(torch.stack(kls)).item(), self.agent_steps)
        self.writer.add_scalar('info/grad_norms', torch.mean(torch.stack(grad_norms)).item(), self.agent_steps)

        for k, v in self.extra_info.items():
            if isinstance(v, torch.Tensor) and len(v.shape) != 0:
                continue
            self.writer.add_scalar(f'{k}', v, self.agent_steps)

    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()
        if self.normalize_priv:
            self.priv_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.running_mean_std.train()
        if self.normalize_priv:
            self.priv_mean_std.train()

    def model_act(self, obs_dict, is_demon=True):
        processed_obs = self.running_mean_std_demon(obs_dict['obs']) if is_demon else self.running_mean_std(obs_dict['obs'])
        priv_info = obs_dict['priv_info']
        proprio_hist = obs_dict['proprio_hist']
        if self.normalize_priv:
            priv_info = self.priv_mean_std_demon(obs_dict['priv_info']) if is_demon else self.priv_mean_std(obs_dict['priv_info'])
        if self.normalize_point_cloud:
            point_cloud = self.point_cloud_mean_std_demon(
                obs_dict['point_cloud_info'].reshape(-1, 3)
            ).reshape((processed_obs.shape[0], -1, 3)) if is_demon else self.point_cloud_mean_std(
                obs_dict['point_cloud_info'].reshape(-1, 3)
            ).reshape((processed_obs.shape[0], -1, 3))
        else:
            point_cloud = obs_dict['point_cloud_info']
        input_dict = {
            'obs': processed_obs,
            'priv_info': priv_info,
            'rot_axis_buf': obs_dict['rot_axis_buf'],
            'critic_info': obs_dict['critic_info'],
            'proprio_hist': proprio_hist,
            'tactile_hist': obs_dict['tactile_hist'],
            'obj_ends': obs_dict['obj_ends'],
            'point_cloud_info': point_cloud,
        }
        res_dict = self.model.act(input_dict)
        res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        return res_dict

    def train(self):
        _t = time.time()
        _last_t = time.time()
        self.obs = self.env.reset()
        self.obs['proprio_hist'] = self.obs['proprio_hist'][..., -self.proprio_len:, :self.proprio_dim]
        self.agent_steps = self.batch_size

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            a_losses, c_losses, b_losses, entropies, kls, grad_norms, latent_losses = self.train_epoch()
            self.storage.data_dict = None

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            info_string = f'Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | ' \
                          f'Last FPS: {last_fps:.1f} | ' \
                          f'Collect Time: {self.data_collect_time / 60:.1f} min | ' \
                          f'Train RL Time: {self.rl_train_time / 60:.1f} min | ' \
                          f'Current Best Reward: {self.best_rewards:.2f}'
            print(info_string)

            self.write_stats(a_losses, c_losses, b_losses, entropies, kls, grad_norms, latent_losses)
            mean_rewards = self.episode_rewards.get_mean()
            mean_lengths = self.episode_lengths.get_mean()
            self.writer.add_scalar('episode_rewards/step', mean_rewards, self.agent_steps)
            self.writer.add_scalar('episode_lengths/step', mean_lengths, self.agent_steps)
            checkpoint_name = f'ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}m_reward_{mean_rewards:.2f}'

            if self.save_freq > 0:
                if self.epoch_num % self.save_freq == 0:
                    self.save(os.path.join(self.nn_dir, checkpoint_name))
                    self.save(os.path.join(self.nn_dir, 'last'))

            if mean_rewards > self.best_rewards and self.agent_steps >= self.save_best_after:
                print(f'save current best reward: {mean_rewards:.2f}')
                self.best_rewards = mean_rewards
                self.save(os.path.join(self.nn_dir, 'best'))

        print('max steps achieved')

    def save(self, name):
        weights = {
            'model': self.model.state_dict(),
        }
        if self.normalize_priv:
            weights['priv_mean_std'] = self.priv_mean_std.state_dict()
        if self.normalize_value:
            weights['value_mean_std'] = self.value_mean_std.state_dict()
        if self.normalize_point_cloud:
            weights['point_cloud_mean_std'] = self.point_cloud_mean_std.state_dict()

        torch.save(weights, f'{name}.pth')

    def restore_train(self, fn):
        if not fn:
            return
        print("loading checkpoint from path", fn)
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint['model'])
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        if self.normalize_priv:
            self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])
        if self.normalize_value:
            self.value_mean_std.load_state_dict(checkpoint['value_mean_std'])
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.load_state_dict(checkpoint['point_cloud_mean_std'])

    def demon_load(self, path):
        print("loading demonstration checkpoint from path", path)
        checkpoint = torch.load(path)
        self.demon_model.load_state_dict(checkpoint['model'])
        self.running_mean_std_demon.load_state_dict(checkpoint['running_mean_std'])
        
        self.priv_mean_std_demon.load_state_dict(checkpoint['priv_mean_std'])
        self.priv_mean_std_demon.eval()

        # self.proprio_mean_std_demon.load_state_dict(checkpoint['proprio_mean_std'])
        # self.proprio_mean_std_demon.eval()
        
        self.point_cloud_mean_std_demon.load_state_dict(checkpoint['point_cloud_mean_std'])
        self.point_cloud_mean_std_demon.eval()
        self.demon_model.eval()
        self.running_mean_std_demon.eval()
    
    def restore_test(self, fn):
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        if self.normalize_priv:
            self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.load_state_dict(checkpoint['point_cloud_mean_std'])
        if self.normalize_value:
            self.value_mean_std.load_state_dict(checkpoint['value_mean_std'])

    def test(self):
        self.set_eval()
        obs_dict = self.env.reset()
        obs_dict['proprio_hist'] = obs_dict['proprio_hist'][..., -self.proprio_len:, :self.proprio_dim]
        while True:
            point_cloud = obs_dict['point_cloud_info']
            if self.normalize_point_cloud:
                point_cloud = self.point_cloud_mean_std(
                    obs_dict['point_cloud_info'].reshape(-1, 3)
                ).reshape((obs_dict['obs'].shape[0], -1, 3))

            input_dict = {
                'obs': self.running_mean_std(obs_dict['obs']),
                'priv_info': self.priv_mean_std(obs_dict['priv_info']) if self.normalize_priv else obs_dict['priv_info'],
                'proprio_hist': obs_dict['proprio_hist'],
                'point_cloud_info': point_cloud,
            }
            mu, extrin, extrin_gt = self.model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu, extrin_record=extrin)
            obs_dict['proprio_hist'] = obs_dict['proprio_hist'][..., -self.proprio_len:, :self.proprio_dim]

    def recon_criterion(self, out, target):
        if self.use_l1:
            return torch.abs(out - target)
        else:
            return (out - target).pow(2)

    def train_epoch(self):
        # collect minibatch data
        _t = time.time()
        self.set_eval()
        self.play_steps()
        self.data_collect_time += (time.time() - _t)
        # update network
        _t = time.time()
        self.set_train()
        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []
        grad_norms = []
        latent_losses = []

        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.storage)):
                (value_preds, old_action_log_probs, advantage, old_mu, old_sigma, returns, actions, obs,
                 priv_info, critic_info, point_cloud_info, proprio_hist, tactile_hist, obj_ends) = self.storage[i]

                batch_dict = {
                    'obs': self.running_mean_std(obs),
                    'priv_info': self.priv_mean_std(priv_info) if self.normalize_priv else priv_info,
                    'critic_info': critic_info,
                    'proprio_hist': proprio_hist,
                    'tactile_hist': tactile_hist,
                    'obj_ends': obj_ends,
                    'point_cloud_info': point_cloud_info,
                }
                res_dict = self.model(batch_dict)
                mu = res_dict['mus']
                sigma = res_dict['sigmas']
                e = res_dict['extrin']
                e_gt = res_dict['extrin_gt']
                values = res_dict['values']

                # PPO loss
                action_log_probs = self.model.get_action_log_probs(actions, mu, sigma)
                ratio = torch.exp(action_log_probs - old_action_log_probs)
                surr1 = advantage * ratio
                surr2 = advantage * torch.clamp(ratio, 1.0 - self.e_clip, 1.0 + self.e_clip)
                a_loss = torch.max(-surr1, -surr2).mean()

                # critic loss
                value_clipped = value_preds + (values - value_preds).clamp(-self.e_clip, self.e_clip)
                c_loss = torch.max((values - returns) ** 2, (value_clipped - returns) ** 2).mean()

                # entropy loss
                entropy = self.model.get_entropy(sigma).mean()

                # demonstration
                with torch.no_grad():
                    demon_batch_dict = {
                        'obs': self.running_mean_std_demon(obs),
                        'priv_info': self.priv_mean_std_demon(priv_info),
                        'point_cloud_info': self.point_cloud_mean_std_demon(point_cloud_info.reshape(-1, 3)).reshape((obs.shape[0], -1, 3)),
                        'proprio_hist': proprio_hist,
                    }
                    mu_demon, e_demon, e_gtdemon = self.demon_model.act_inference(demon_batch_dict)

                # behavior cloning loss
                bc_loss = self.recon_criterion(torch.clamp(mu, -1, 1), torch.clamp(mu_demon, -1, 1)).mean()

                # latent alignment loss
                latent_loss = ((e_gt - e_gtdemon.detach()) ** 2).mean()

                # total loss
                loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + self.bc_loss_coef * bc_loss
                if self.enable_latent_loss:
                    loss += self.distill_loss_coef * latent_loss

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    kl_dist = policy_kl(old_mu, old_sigma, mu, sigma)
                    ep_kls.append(kl_dist)

                a_losses.append(a_loss)
                c_losses.append(c_loss)
                b_losses.append(bc_loss)
                entropies.append(entropy)
                grad_norms.append(grad_norm)
                latent_losses.append(latent_loss)

            kls.append(torch.mean(torch.stack(ep_kls)))
            self.last_lr = self.scheduler.update(self.last_lr, kl_dist)

        # 更新全局学习率（余弦退火）
        self.global_scheduler.step()
        global_lr = self.global_scheduler.get_last_lr()[0]
        
        # 更新 AdaptiveScheduler 的 max_lr，使其受全局调度器约束
        self.scheduler.max_lr = global_lr
        
        # 更新最终学习率，确保不超过全局上限
        self.last_lr = min(self.last_lr, global_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.last_lr

        self.rl_train_time += (time.time() - _t)
        return a_losses, c_losses, b_losses, entropies, kls, grad_norms, latent_losses

    def play_steps(self):
        for n in range(self.horizon_length):
            res_dict = self.model_act(self.obs, is_demon=False)
            # collect o_t
            self.storage.update_data('obses', n, self.obs['obs'])
            self.storage.update_data('priv_info', n, self.obs['priv_info'])
            self.storage.update_data('critic_info', n, self.obs['critic_info'])
            self.storage.update_data('point_cloud_info', n, self.obs['point_cloud_info'])
            self.storage.update_data('proprio_hist', n, self.obs['proprio_hist'])
            self.storage.update_data('tactile_hist', n, self.obs['tactile_hist'])
            self.storage.update_data('obj_ends', n, self.obs['obj_ends'])
            for k in ['values', 'mus', 'sigmas']:
                self.storage.update_data(k, n, res_dict[k])
            # do env step
            actions = torch.clamp(res_dict['mus'], -1.0, 1.0)
            self.storage.update_data('actions', n, actions)

            self.obs, rewards, self.dones, infos = self.env.step(actions)
            self.obs['proprio_hist'] = self.obs['proprio_hist'][..., -self.proprio_len:, :self.proprio_dim]

            rewards = rewards.unsqueeze(1)
            # update dones and rewards after env step
            self.storage.update_data('dones', n, self.dones)
            rewards = rewards.to(self.device)
            self.storage.update_data('rewards', n, rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])

            assert isinstance(infos, dict), 'Info Should be a Dict'
            self.extra_info = infos
            not_dones = (1.0 - self.dones.float()).to(self.device)

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        res_dict = self.model_act(self.obs, is_demon=False)
        last_values = res_dict['values']

        self.agent_steps = self.agent_steps + self.batch_size
        self.storage.compute_returns(last_values, self.gamma, self.tau)
        self.storage.prepare_training()
        if self.normalize_advantage:
            self.storage.normalize_advantage()


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    c1 = torch.log(p1_sigma/p0_sigma + 1e-5)
    c2 = (p0_sigma ** 2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma ** 2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    return kl.mean()


# from https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/algorithms/ppo.py
class AdaptiveScheduler(object):
    def __init__(self, kl_threshold=0.008, min_lr=1e-6, max_lr=1e-2):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr  # 可动态更新的最大学习率
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)  # 使用动态的 self.max_lr
        return lr
