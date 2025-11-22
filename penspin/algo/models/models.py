# --------------------------------------------------------
# Refactored Model Architecture for Pen Spinning
# Separates Teacher and Student Networks
# 
# =============================================================================
# 网络架构详解：从 Allegro Hand (16 DoF) 到 Linker Hand (21 DoF)
# =============================================================================
#
# 一、Teacher Network (专家网络 - 在仿真中训练)
# -----------------------------------------------
# 输入：
#   - obs: [batch, 126]  基础观测 (3个时间步 × 42 proprio, 其中42=21 current_pos + 21 target_pos)
#   - priv_info: [batch, 47] 特权信息 (动态维度，取决于启用项)
#   - point_cloud_info: [batch, 100, 3]  点云 (xyz)
#   - critic_info: [batch, 100]  (如果 asymm_actor_critic=True)
#
# 编码流程：
#   1. priv_info -> env_mlp [256,128,8] -> [batch, 8] (特权编码)
#   2. point_cloud -> point_mlp [32,32,32] -> max_pool -> [batch, 32] (点云编码)
#   3. extrin = tanh(concat([priv_encoded, pc_encoded])) -> [batch, 40]
#   4. obs_input = concat([obs, extrin]) -> [batch, 126+40=166]
#
# Actor-Critic：
#   - actor_mlp: [batch, 166] -> [512,256,128] -> [batch, 128]
#   - mu: [batch, 128] -> Linear -> [batch, 21]  (动作均值)
#   - sigma: Parameter [21]  (动作标准差)
#   - value: [batch, 128] -> Linear -> [batch, 1]  (如果对称AC)
#            或 concat([obs, critic_info]) -> mlp -> [batch, 1]  (如果非对称AC)
#
# 输出：
#   - mu: [batch, 21]
#   - sigma: [21]
#   - value: [batch, 1]
#   - extrin: [batch, 40]  (编码后的特征，用于蒸馏)
#   - extrin_gt: [batch, 40]  (与extrin相同，teacher没有区分)
#
# 旧版 (Allegro Hand) 对比：
#   - obs: 96 -> 126  (3 timesteps × (PROPRIO_DIM: 32->42))
#   - mu: 16 -> 21  (NUM_DOF: 16->21)
#   - priv_info: 61 -> 47 (维度相似，但内部结构变化)
#   - extrin: 40 (不变，因为priv_mlp和point_mlp输出维度相同)
#   
# 说明：obs=126 来自 3个历史时间步 × 42维 proprio
#      其中 PROPRIO_DIM=42 = 21维当前位置 + 21维目标位置
#
# -----------------------------------------------
# 二、Student Network (学生网络 - 用于真实机器人部署)
# -----------------------------------------------
# 输入：
#   - obs: [batch, 126]  基础观测 (3 timesteps × 42 proprio)
#   - proprio_hist: [batch, 30, 42]  本体感觉历史 (42 = 21 current_pos + 21 target_pos)
#   - obj_ends: [batch, 3, 6]  物体端点 (视觉跟踪，如SAM)
#   - student_pc_info: [batch, 100, 6]  点云+特征 (xyz+rgb)
#   - tactile_hist: [batch, 30, 15]  触觉历史 (5 fingers × 3D)
#   - priv_info: [batch, 47]  (仅训练时，提取fingertip用于监督)
#
# 编码流程：
#   1. proprio_hist -> TemporalTransformer -> [batch, 32] -> Linear -> [batch, 40]
#   2. obj_ends[:,:,:3] -> end_mlp [6,6,6] -> [batch, 3, 6]
#      obj_ends[:,:,3:] -> end_mlp [6,6,6] -> [batch, 3, 6]
#      max_pool -> [batch, 18]  (flatten)
#   3. student_pc_info -> PointNet -> [batch, 256]
#   4. fingertip_pos (from priv_info[16:31]) + noise -> [batch, 15]
#      if "tactile" in mode: concat(tactile_hist[:,-2:,:]) -> [batch, 15+30=45]
#      -> env_mlp [256,128,64] -> extrin_gt [batch, 64]
#   5. obs_input = concat([obs, proprio_feat, obj_ends_feat, pc_feat, extrin_gt])
#                = [batch, 126+40+18+256+64=504]  (如果全部启用)
#
# Actor-Critic：
#   - actor_mlp: [batch, 504] -> [512,256,128] -> [batch, 128]
#   - mu: [batch, 128] -> Linear -> [batch, 21]
#   - sigma: Parameter [21]
#   - value: 同 Teacher
#
# 输出：
#   - mu: [batch, 21]
#   - sigma: [21]
#   - value: [batch, 1]
#   - extrin: [batch, 40]  (proprio temporal encoding)
#   - extrin_gt: [batch, 64]  (fingertip+tactile encoding，用于监督学习)
#
# 旧版 (Allegro Hand) 对比：
#   - proprio_hist: [30, 32] -> [30, 42]  (32=16pos+16target -> 42=21pos+21target)
#   - tactile_hist: [30, 32] -> [30, 15]  (传感器类型变化)
#   - fingertip: 12 -> 15  (4指 -> 5指)
#   - tactile feature: 64 -> 30  (2×32 -> 2×15)
#   - mu: 16 -> 21
#   - obs_input总维度变化: 取决于启用的输入模式，但单项维度已适配
#
# 说明：PROPRIO_DIM=42 = 21维当前位置 + 21维目标位置（非速度）
#
# -----------------------------------------------
# 三、Teacher 和 Student 的主要差异
# -----------------------------------------------
# 1. **输入来源**：
#    - Teacher: 仿真提供的完整状态 (priv_info, clean point cloud)
#    - Student: 可部署传感器 (视觉跟踪的端点, 带噪声的点云)
#
# 2. **特权信息使用**：
#    - Teacher: 完整 priv_info (47维) 包含物体物理属性、指尖状态等
#    - Student: 仅使用 fingertip_position (15维) + tactile (30维)
#
# 3. **训练模式**：
#    - Teacher: RL (PPO) 在仿真中学习
#    - Student: 行为克隆 (BC) + 潜在蒸馏 (可选) 学习 Teacher 的策略
#
# 4. **部署模式**：
#    - Teacher: 不可部署 (依赖仿真特权信息)
#    - Student: 可部署 (priv_info=False, 仅用可见传感器)
#
# 5. **RL Loss**：
#    - Teacher: 完整的 PPO loss (policy loss + value loss + entropy)
#    - Student 训练阶段: BC loss + latent loss (optional)
#    - Student 部署阶段: 无 loss，纯推理
#
# 6. **Asymmetric Actor-Critic**：
#    - 两者都支持，但通常 Teacher 更可能使用 (有 critic_info 可用)
#    - Student 在部署时通常不使用 critic (critic_info 不可获取)
#
# -----------------------------------------------
# 四、功能扩展接口
# -----------------------------------------------
# 1. **Student 启用 priv_info**：
#    - 设置 ppo.priv_info=True (训练时)
#    - 用于监督学习，提取 fingertip + tactile
#    - 部署时必须设置 ppo.priv_info=False
#
# 2. **Student 启用 RL loss**：
#    - 当前架构支持，需要在 PPO trainer 中启用
#    - 可用于 sim-to-real 微调或混合训练
#    - 需要完整的 value 和 entropy 计算
#
# 3. **input_mode 配置**：
#    - 'proprio': 仅本体感觉历史
#    - 'proprio-ends': + 物体端点
#    - 'proprio-tactile': + 触觉历史
#    - 'proprio-ends-tactile': + 端点 + 触觉 (最常用)
#
# 4. **点云配置**：
#    - use_point_cloud_info=True: 启用点云编码
#    - use_point_transformer=True: 使用 Transformer 而非 MLP
#
# =============================================================================
# --------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from .pointnets import PointNet
from .block import TemporalConv, TemporalTransformer

# 导入统一的机器人配置
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from penspin.utils.robot_config import (
    NUM_DOF, NUM_FINGERS, FINGERTIP_CNT, CONTACT_DIM,
    PROPRIO_DIM, FINGERTIP_POS_DIM, TEMPORAL_FUSION_OUTPUT_DIM,
    TEMPORAL_FUSION_FINAL_DIM, OBJ_ENDS_TOTAL_DIM, 
    TACTILE_FEATURE_DIM, TACTILE_USED_TIMESTEPS,
    POINT_CLOUD_FEATURE_DIM, POINT_CLOUD_FEATURE_DIM_STUDENT,
    POINTNET_OUTPUT_DIM,
    get_priv_info_fingertip_slice
)


class MLP(nn.Module):
    """Multi-Layer Perceptron with ELU activation"""
    def __init__(self, units, input_size, with_last_activation=True):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        if not with_last_activation:
            layers.pop()
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class BaseActorCritic(nn.Module):
    """Base class for Actor-Critic models with shared components"""
    
    def __init__(self, kwargs):
        super(BaseActorCritic, self).__init__()
        
        # Basic parameters
        self.actions_num = kwargs.get('actions_num')
        self.units = kwargs.get('actor_units')
        self.asymm_actor_critic = kwargs['asymm_actor_critic']
        self.critic_info_dim = kwargs['critic_info_dim']
        
        # Initialize actor/critic heads (will be set by subclasses)
        self.actor_mlp = None
        self.value = None
        self.mu = None
        self.sigma = None
        
    def _init_actor_critic_heads(self, policy_input_dim):
        """Initialize actor and critic network heads"""
        out_size = self.units[-1]
        
        # Actor network
        self.actor_mlp = MLP(units=self.units, input_size=policy_input_dim)
        
        # Critic network (asymmetric or shared)
        if self.asymm_actor_critic:
            critic_input_dim = policy_input_dim + self.critic_info_dim
            self.value = MLP(units=self.units + [1], input_size=critic_input_dim)
        else:
            self.value = nn.Linear(out_size, 1)
        
        # Action distribution parameters
        self.mu = nn.Linear(out_size, self.actions_num)
        self.sigma = nn.Parameter(
            torch.zeros(self.actions_num, requires_grad=True, dtype=torch.float32),
            requires_grad=True
        )
        
        # Weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
        
        if self.sigma is not None:
            nn.init.constant_(self.sigma, 0)
    
    @torch.no_grad()
    def act(self, obs_dict):
        """Sample action during training (with exploration)"""
        mu, logstd, value, extrin, extrin_gt = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        
        return {
            'neglogpacs': -distr.log_prob(selected_action).sum(1),
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
            'extrin': extrin,
            'extrin_gt': extrin_gt,
        }
    
    @torch.no_grad()
    def act_inference(self, obs_dict):
        """Deterministic action for testing/deployment"""
        mu, logstd, value, extrin, extrin_gt = self._actor_critic(obs_dict)
        return mu, extrin, extrin_gt
    
    def forward(self, input_dict):
        """Full forward pass for training"""
        prev_actions = input_dict.get('prev_actions', None)
        mu, logstd, value, extrin, extrin_gt = self._actor_critic(input_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        
        prev_neglogp = None
        if prev_actions is not None:
            prev_neglogp = -distr.log_prob(prev_actions).sum(1)
            prev_neglogp = torch.squeeze(prev_neglogp)
        
        return {
            'prev_neglogp': prev_neglogp,
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
            'extrin': extrin,
            'extrin_gt': extrin_gt,
        }
    
    def get_dimension_info(self):
        """
        返回模型的维度信息,用于调试和验证
        
        Returns:
            dict: 包含各个维度的字典
        """
        if hasattr(self, '_dim_info'):
            return self._dim_info
        else:
            return {}
    
    def print_dimension_info(self):
        """
        打印模型的维度信息，包括从旧版本(Allegro Hand)到新版本(Linker Hand)的对比
        """
        dim_info = self.get_dimension_info()
        if not dim_info:
            print("维度信息不可用")
            return
        
        model_type = "Teacher" if isinstance(self, TeacherActorCritic) else "Student"
        print(f"\n{'='*70}")
        print(f"  {model_type} Network 维度信息 (Linker Hand - 21 DoF, 5 Fingers)")
        print(f"{'='*70}")
        
        # 显示基础维度
        print(f"\n【基础维度】")
        print(f"  NUM_DOF: {NUM_DOF}  (Allegro: 16 -> Linker: 21)")
        print(f"  NUM_FINGERS: {NUM_FINGERS}  (Allegro: 4 -> Linker: 5)")
        print(f"  PROPRIO_DIM: {PROPRIO_DIM}  (Allegro: 32 -> Linker: 42)")
        print(f"  CONTACT_DIM: {CONTACT_DIM}  (Allegro: 32 -> Linker: 15)")
        print(f"  FINGERTIP_POS_DIM: {FINGERTIP_POS_DIM}  (Allegro: 12 -> Linker: 15)")
        
        # 显示网络输入维度
        print(f"\n【网络输入维度】")
        for key, value in dim_info.items():
            if key != 'total_policy_input_dim':
                print(f"  {key}: {value}")
        
        # 显示总输入维度
        if 'total_policy_input_dim' in dim_info:
            print(f"\n【总输入维度】")
            print(f"  total_policy_input_dim: {dim_info['total_policy_input_dim']}")
        
        # 显示Actor-Critic配置
        print(f"\n【Actor-Critic 配置】")
        print(f"  actions_num: {self.actions_num}")
        print(f"  actor_units: {self.units}")
        print(f"  asymm_actor_critic: {self.asymm_actor_critic}")
        if self.asymm_actor_critic:
            print(f"  critic_info_dim: {self.critic_info_dim}")
        
        # 显示模型特定信息
        if isinstance(self, TeacherActorCritic):
            print(f"\n【Teacher 特定配置】")
            print(f"  priv_info_dim: {self.priv_info_dim}")
            print(f"  priv_mlp_units: {self.priv_mlp_units}")
            print(f"  point_mlp_units: {self.point_mlp_units}")
            print(f"  use_point_transformer: {self.use_point_transformer}")
        elif isinstance(self, StudentActorCritic):
            print(f"\n【Student 特定配置】")
            print(f"  input_mode: {self.input_mode}")
            print(f"  proprio_len: {self.proprio_len}")
            print(f"  use_point_cloud: {self.use_point_cloud}")
            if 'new_priv_dim' in dim_info:
                print(f"  new_priv_dim: {dim_info['new_priv_dim']}  (fingertip + tactile)")
        
        print(f"{'='*70}\n")
    
    def validate_dimensions(self):
        """
        验证维度配置的一致性
        
        Returns:
            bool: 如果所有维度检查通过返回True
        """
        try:
            dim_info = self.get_dimension_info()
            
            # 基础检查
            assert self.actions_num == NUM_DOF, \
                f"actions_num ({self.actions_num}) != NUM_DOF ({NUM_DOF})"
            
            if isinstance(self, TeacherActorCritic):
                # Teacher 特定检查
                assert self.priv_info_dim > 0, \
                    f"priv_info_dim must be positive, got {self.priv_info_dim}"
                
                # 检查 priv_mlp 输出维度
                priv_encoded_dim = self.priv_mlp_units[-1]
                assert priv_encoded_dim == dim_info.get('priv_info_encoded_dim'), \
                    f"priv_mlp output mismatch"
            
            elif isinstance(self, StudentActorCritic):
                # Student 特定检查
                assert self.proprio_dim == PROPRIO_DIM, \
                    f"proprio_dim ({self.proprio_dim}) != PROPRIO_DIM ({PROPRIO_DIM})"
                
                # 检查 new_priv_dim 范围
                new_priv_dim = dim_info.get('new_priv_dim', 0)
                assert FINGERTIP_POS_DIM <= new_priv_dim <= FINGERTIP_POS_DIM + TACTILE_FEATURE_DIM, \
                    f"new_priv_dim ({new_priv_dim}) out of expected range"
            
            print(f"✓ {self.__class__.__name__} 维度验证通过!")
            return True
            
        except AssertionError as e:
            print(f"✗ {self.__class__.__name__} 维度验证失败: {e}")
            return False

        print(f"\n{'='*60}")
        print(f"{self.__class__.__name__} 维度信息:")
        print(f"{'='*60}")
        for key, value in dim_info.items():
            print(f"  {key:30s}: {value}")
        print(f"{'='*60}\n")
    
    def _actor_critic(self, obs_dict):
        """To be implemented by subclasses"""
        raise NotImplementedError


class TeacherActorCritic(BaseActorCritic):
    """
    Teacher network for PPO training with privileged information.
    
    Inputs:
        - obs: proprioceptive observations (joint pos/vel)
        - priv_info: privileged information (object state, fingertip state, etc.)
        - point_cloud_info: sampled point cloud from object
        - critic_info: additional info for value function (if asymmetric)
    """
    
    def __init__(self, kwargs):
        super(TeacherActorCritic, self).__init__(kwargs)
        
        # Teacher-specific parameters
        policy_input_dim = kwargs.get('input_shape')[0]
        self.priv_mlp_units = kwargs.get('priv_mlp_units')
        self.priv_info_dim = kwargs['priv_info_dim']
        self.point_mlp_units = kwargs.get('point_mlp_units')
        self.use_point_transformer = kwargs.get('use_point_transformer')
        
        # ========== 维度验证 ==========
        # 验证 priv_info_dim 是否合理 (应该包含完整的特权信息)
        assert self.priv_info_dim > 0, \
            f"TeacherActorCritic: priv_info_dim 必须大于0, 当前值: {self.priv_info_dim}"
        
        # 验证 policy_input_dim 是否合理
        assert policy_input_dim > 0, \
            f"TeacherActorCritic: input_shape[0] 必须大于0, 当前值: {policy_input_dim}"
        
        # 验证 priv_mlp_units 配置
        assert self.priv_mlp_units and len(self.priv_mlp_units) > 0, \
            f"TeacherActorCritic: priv_mlp_units 不能为空"
        
        # 验证 point_mlp_units 配置
        assert self.point_mlp_units and len(self.point_mlp_units) > 0, \
            f"TeacherActorCritic: point_mlp_units 不能为空"
        
        # Privileged information encoder
        self.env_mlp = MLP(
            units=self.priv_mlp_units,
            input_size=self.priv_info_dim,
            with_last_activation=False
        )
        policy_input_dim += self.priv_mlp_units[-1]
        
        # Point cloud encoder
        # Teacher 使用 3D 点云 (xyz)
        if self.use_point_transformer:
            self.point_mlp = TemporalTransformer(
                32, 2, 1, 32, use_pe=False, pre_ffn=True, input_dim=POINT_CLOUD_FEATURE_DIM
            )
        else:
            self.point_mlp = MLP(units=self.point_mlp_units, input_size=POINT_CLOUD_FEATURE_DIM)
        policy_input_dim += self.point_mlp_units[-1]
        
        # ========== 最终维度验证和记录 ==========
        # 验证最终的 policy_input_dim 是否合理
        assert policy_input_dim > kwargs.get('input_shape')[0], \
            f"TeacherActorCritic: policy_input_dim ({policy_input_dim}) 应该大于基础 obs 维度 ({kwargs.get('input_shape')[0]})"
        
        # 记录维度信息 (用于调试)
        self._dim_info = {
            'base_obs_dim': kwargs.get('input_shape')[0],
            'priv_info_encoded_dim': self.priv_mlp_units[-1],
            'point_cloud_dim': self.point_mlp_units[-1],
            'total_policy_input_dim': policy_input_dim,
        }
        
        # Initialize actor-critic heads
        self._init_actor_critic_heads(policy_input_dim)
    
    def _actor_critic(self, obs_dict):
        """
        Forward pass for teacher network.
        
        Args:
            obs_dict: Dictionary containing:
                - obs: [batch, obs_dim]
                - priv_info: [batch, priv_info_dim]
                - point_cloud_info: [batch, num_points, 3]
                - critic_info: [batch, critic_info_dim] (if asymmetric)
        
        Returns:
            mu, sigma, value, extrin, extrin_gt
        """
        obs = obs_dict['obs']
        
        # Encode privileged information
        extrin = self.env_mlp(obs_dict['priv_info'])
        
        # Encode point cloud
        if self.use_point_transformer:
            pcs = self.point_mlp(obs_dict['point_cloud_info'])
        else:
            pcs = self.point_mlp(obs_dict['point_cloud_info'])
            pcs = torch.max(pcs, dim=1)[0]  # Max pooling
        
        # Concatenate features
        extrin = torch.cat([extrin, pcs], dim=-1)
        extrin = torch.tanh(extrin)
        extrin_gt = extrin  # For teacher, extrin and extrin_gt are the same
        
        # Actor forward
        obs_input = torch.cat([obs, extrin], dim=-1)
        x = self.actor_mlp(obs_input)
        
        # Critic forward
        if self.asymm_actor_critic:
            critic_obs = torch.cat([obs, obs_dict['critic_info']], dim=-1)
            value = self.value(critic_obs)
        else:
            value = self.value(x)
        
        # Action distribution
        mu = self.mu(x)
        sigma = self.sigma
        
        return mu, sigma, value, extrin, extrin_gt


class StudentActorCritic(BaseActorCritic):
    """
    Student network for distillation and real robot deployment.
    
    Inputs:
        - obs: proprioceptive observations (joint pos/vel)
        - proprio_hist: history of proprioceptive observations [batch, time, proprio_dim]
        - obj_ends: object endpoints tracked by vision (e.g., SAM) [batch, time, 6]
        - student_pc_info: point cloud with features [batch, num_points, 6]
        - tactile_hist: tactile history (optional) [batch, time, contact_dim]
        - priv_info: privileged info for supervised learning (training only)
    """
    
    def __init__(self, kwargs):
        super(StudentActorCritic, self).__init__(kwargs)
        
        # Student-specific parameters
        policy_input_dim = kwargs.get('input_shape')[0]
        self.priv_mlp_units = kwargs.get('priv_mlp_units')
        self.proprio_len = kwargs.get('proprio_len', 30)
        self.proprio_dim = PROPRIO_DIM  # 使用配置中的 PROPRIO_DIM (42 for Linker Hand)
        self.input_mode = kwargs.get('input_mode', 'proprio-ends')
        self.use_point_cloud = kwargs.get('use_point_cloud_info', True)
        
        # Priv info configuration (for extracting fingertip positions)
        self.priv_config = kwargs.get('priv_config', {})
        
        # ========== 维度验证 ==========
        # 验证基础维度配置
        assert policy_input_dim > 0, \
            f"StudentActorCritic: input_shape[0] 必须大于0, 当前值: {policy_input_dim}"
        
        assert self.proprio_len > 0, \
            f"StudentActorCritic: proprio_len 必须大于0, 当前值: {self.proprio_len}"
        
        assert self.proprio_dim == PROPRIO_DIM, \
            f"StudentActorCritic: proprio_dim 不匹配! 期望: {PROPRIO_DIM}, 实际: {self.proprio_dim}"
        
        # 验证 input_mode 的有效性
        valid_modes = ['proprio', 'proprio-ends', 'proprio-tactile', 'proprio-ends-tactile', 
                       'proprio-tactile-ends', 'proprio-tactile-ends-fingertip']
        assert any(mode in self.input_mode for mode in ['proprio']), \
            f"StudentActorCritic: input_mode 必须包含 'proprio', 当前值: {self.input_mode}"
        
        # 验证 priv_mlp_units 配置
        assert self.priv_mlp_units and len(self.priv_mlp_units) > 0, \
            f"StudentActorCritic: priv_mlp_units 不能为空"
        
        # Proprioception history encoder (temporal)
        # Note: Input is [batch, time, PROPRIO_DIM], output is [batch, TEMPORAL_FUSION_OUTPUT_DIM]
        self.adapt_tconv = TemporalTransformer(
            self.proprio_dim, 2, 2, TEMPORAL_FUSION_OUTPUT_DIM, use_pe=True
        )
        self.all_fuse = nn.Linear(TEMPORAL_FUSION_OUTPUT_DIM, TEMPORAL_FUSION_FINAL_DIM)
        proprio_feat_dim = TEMPORAL_FUSION_FINAL_DIM
        policy_input_dim += proprio_feat_dim
        
        # Object endpoints encoder
        if "ends" in self.input_mode:
            policy_input_dim += OBJ_ENDS_TOTAL_DIM  # 使用配置常量: 3 timesteps * 2 endpoints * 3 features = 18
            self.end_feat_extractor = MLP(
                units=[6, 6, 6], input_size=3, with_last_activation=False
            )
        
        # Point cloud encoder (vision-based)
        # Student 使用 6D 点云 (xyz + rgb/features)
        if self.use_point_cloud:
            policy_input_dim += POINTNET_OUTPUT_DIM  # PointNet 输出维度
            self.pc_encoder = PointNet(point_channel=POINT_CLOUD_FEATURE_DIM_STUDENT)
        
        # Privileged info encoder (for supervised training phase)
        # Uses fingertip positions + optional tactile
        new_priv_dim = FINGERTIP_POS_DIM  # 15 for Linker Hand (5 fingertips × 3D)
        if "tactile" in self.input_mode:
            # 使用最后 TACTILE_USED_TIMESTEPS 个时间步的触觉数据
            # TACTILE_FEATURE_DIM = TACTILE_USED_TIMESTEPS * CONTACT_DIM
            new_priv_dim += TACTILE_FEATURE_DIM
        
        policy_input_dim += self.priv_mlp_units[-1]
        self.env_mlp = MLP(
            units=self.priv_mlp_units,
            input_size=new_priv_dim,
            with_last_activation=False
        )
        
        # ========== 最终维度验证和记录 ==========
        # 验证 priv_mlp 输入维度的合理性
        expected_min_priv_dim = FINGERTIP_POS_DIM  # 最少包含 fingertip
        expected_max_priv_dim = FINGERTIP_POS_DIM + TACTILE_FEATURE_DIM  # 最多包含 fingertip + tactile
        
        assert expected_min_priv_dim <= new_priv_dim <= expected_max_priv_dim, \
            f"StudentActorCritic: new_priv_dim 超出预期范围! " \
            f"范围: [{expected_min_priv_dim}, {expected_max_priv_dim}], 实际: {new_priv_dim}"
        
        # 验证最终的 policy_input_dim 是否合理
        assert policy_input_dim > kwargs.get('input_shape')[0], \
            f"StudentActorCritic: policy_input_dim ({policy_input_dim}) 应该大于基础 obs 维度 ({kwargs.get('input_shape')[0]})"
        
        # 记录维度信息 (用于调试)
        self._dim_info = {
            'base_obs_dim': kwargs.get('input_shape')[0],
            'proprio_hist_dim': TEMPORAL_FUSION_FINAL_DIM,
            'obj_ends_dim': OBJ_ENDS_TOTAL_DIM if "ends" in self.input_mode else 0,
            'point_cloud_dim': POINTNET_OUTPUT_DIM if self.use_point_cloud else 0,
            'priv_info_dim': self.priv_mlp_units[-1],
            'new_priv_dim': new_priv_dim,
            'total_policy_input_dim': policy_input_dim,
        }
        
        # Initialize actor-critic heads
        self._init_actor_critic_heads(policy_input_dim)
    
    def _actor_critic(self, obs_dict):
        """
        Forward pass for student network.
        
        Args:
            obs_dict: Dictionary containing:
                - obs: [batch, obs_dim]
                - proprio_hist: [batch, time, proprio_dim]
                - obj_ends: [batch, time, 6] (if input_mode contains 'ends')
                - student_pc_info: [batch, num_points, 6] (if use_point_cloud)
                - tactile_hist: [batch, time, contact_dim] (if 'tactile' in input_mode)
                - priv_info: [batch, priv_dim] (training only, for extrin_gt)
        
        Returns:
            mu, sigma, value, extrin, extrin_gt
        """
        obs = obs_dict['obs']
        feature_list = [obs]
        
        # 1. Encode proprioception history
        proprio_feat = self.adapt_tconv(obs_dict['proprio_hist'])
        proprio_feat = self.all_fuse(proprio_feat)
        proprio_feat = torch.tanh(proprio_feat)
        feature_list.append(proprio_feat)
        extrin = proprio_feat  # For return value
        
        # 2. Encode object endpoints (vision-based tracking)
        if "ends" in self.input_mode:
            point_feat_1 = self.end_feat_extractor(obs_dict['obj_ends'][..., :3]).unsqueeze(-1)
            point_feat_2 = self.end_feat_extractor(obs_dict['obj_ends'][..., 3:]).unsqueeze(-1)
            point_feat = torch.cat([point_feat_1, point_feat_2], dim=-1)
            point_feat = torch.max(point_feat, dim=-1)[0]  # Max pooling over endpoints
            feature_list.append(point_feat.view(obs.shape[0], -1))
        
        # 3. Encode point cloud (if available)
        if self.use_point_cloud:
            # 兼容键名：优先使用 student_pc_info；若缺失则回退到 point_cloud_info
            pc_key = 'student_pc_info' if 'student_pc_info' in obs_dict else 'point_cloud_info'
            pc_embedding, self.point_indices = self.pc_encoder(obs_dict[pc_key])
            feature_list.append(pc_embedding)
        
        # 4. Encode "privileged" info for student (fingertip + tactile)
        # Note: This is used during supervised training to match teacher's features
        fingertip_slice = get_priv_info_fingertip_slice(**self.priv_config)
        fingertip_pos = obs_dict['priv_info'][..., fingertip_slice]  # Extract fingertip positions
        # Add noise for robustness
        new_priv = fingertip_pos + (torch.rand_like(fingertip_pos) - 0.5) * 0.02
        
        if "tactile" in self.input_mode:
            # 取最后 TACTILE_USED_TIMESTEPS 个时间步的触觉数据
            # tactile_hist: [batch, 30, 15] -> [-TACTILE_USED_TIMESTEPS:, :] -> [batch, 2, 15]
            # 然后 flatten: [batch, 2*15=30]
            tactile = obs_dict['tactile_hist'][:, -TACTILE_USED_TIMESTEPS:, :].reshape(obs.shape[0], -1)
            new_priv = torch.cat([new_priv, tactile], dim=-1)
        
        new_priv = self.env_mlp(new_priv)
        extrin_gt = torch.tanh(new_priv)
        feature_list.append(extrin_gt)
        
        # 5. Concatenate all features
        obs_input = torch.cat(feature_list, dim=-1)
        
        # Actor forward
        x = self.actor_mlp(obs_input)
        
        # Critic forward
        if self.asymm_actor_critic:
            critic_obs = torch.cat([obs, obs_dict['critic_info']], dim=-1)
            value = self.value(critic_obs)
        else:
            value = self.value(x)
        
        # Action distribution
        mu = self.mu(x)
        sigma = self.sigma
        
        return mu, sigma, value, extrin, extrin_gt


# Factory function for easy model creation
def create_actor_critic(model_type, **kwargs):
    """
    Factory function to create appropriate ActorCritic model.
    
    Args:
        model_type: "teacher" or "student"
        **kwargs: model configuration parameters
    
    Returns:
        ActorCritic model instance
    """
    if model_type.lower() == "teacher":
        return TeacherActorCritic(kwargs)
    elif model_type.lower() == "student":
        return StudentActorCritic(kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'teacher' or 'student'.")
