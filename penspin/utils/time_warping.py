"""
Time-Warping Orchestrator for Space E Curriculum Learning

基于动力学同构（Isomorphic Dynamics）原理，构建的时空扭曲编排器。
用于实现 Space A/B/C/D/E 课程学习。

Space E 核心原理：
- 时间缩放: t' = t / α (流逝变慢)
- 长度不变: L' = L (几何同构)
- 重力缩放: g' = α² g
- 速度缩放: v' = α v
- 加速度缩放: a' = α² a
- 力/力矩缩放: F' = α² F

其中 α ∈ (0, 1] 是时间缩放因子：
- α = 0.1: 世界慢 10 倍 (训练初期)
- α = 1.0: 真实世界速度 (训练结束)

关键洞察：
1. Froude Number 守恒：Fr = v / √(gL) 在缩放前后保持不变
2. 仿真步长 dt 保持不变 = 用同样的计算资源模拟更慢的过程 = 更高精度积分
3. 全局同步课程：所有环境必须处于同一 α 阶段（因为 gravity 是全局参数）

Curriculum 更新逻辑 (重要)：
============================
调用关系:
  ppo_rl_teacher.train()
    -> 每个 epoch 结束后调用 time_warper.update(agent_steps, metrics)
    -> update() 返回 True 时调用 env.apply_curriculum_physics()

更新触发条件 - 增量式自适应 (Incremental Adaptive):
  1. 门控检查：根据 survival_rate (低α) 或 success_rate (高α) 决定是否允许前进
  2. 增量计算：target_alpha = current_alpha * (1 + ratio_threshold)
     - 关键：基于当前 alpha 增量，而非基于时间线性插值
     - 防止"大坝决堤"效应：门控打开后不会跳变到时间对应的位置
  3. 限速器：target_alpha 不能超过时间进度对应的理论上限
  4. 只有物理参数需要更新时 (需调用昂贵的 IsaacGym API)，才返回 True

curriculum_steps 的角色：
  - 作为"限速器"的参考，防止课程跑得太快
  - 即使 Agent 表现优秀，alpha 也不能超过 (steps / curriculum_steps) 对应的上限
  - 不再直接决定 target_alpha，避免积累式跳变

自适应门控指标 (Adaptive Gating):
  - survival_rate: 存活率 (1 - 掉落率)，低 α 阶段的核心指标
  - success_rate: 成功率 (旋转角度 > 阈值)，高 α 阶段的核心指标
  - mean_rot_angle: 平均旋转角度，质量指标
  - mean_reward: 平均奖励，综合指标
  - mean_entropy: 平均熵，探索/收敛指标

使用方法：
    orchestrator = TimeWarpingOrchestrator(config)
    
    # 在训练循环中
    metrics = {
        'success_rate': current_success_rate,
        'survival_rate': current_survival_rate,
        'mean_rot_angle': mean_rot_angle,
        'mean_reward': mean_rewards,
        'mean_entropy': mean_entropy
    }
    if orchestrator.update(global_agent_steps, metrics):
        env.apply_curriculum_physics()  # 只在 alpha 变化时调用
    
    # 获取缩放系数
    gravity_scale = orchestrator.gravity_scale
    obs_vel_scale = orchestrator.obs_velocity_scale
    reward_scale = orchestrator.reward_output_scale
"""

import numpy as np


class TimeWarpingOrchestrator:
    """
    时空扭曲编排器
    
    负责计算 Space A-E 课程学习中的所有物理缩放系数。
    
    支持的模式：
    - 'SpaceA': Baseline，无任何缩放 (α ≡ 1.0)
    - 'SpaceD': 仅物理缩放，无观测/奖励逆缩放
    - 'SpaceE': 完整的动力学同构缩放（推荐）
    
    Attributes:
        mode (str): 课程模式
        alpha_start (float): 初始时间缩放因子
        alpha_end (float): 最终时间缩放因子
        total_steps (int): 课程持续的总步数
        current_alpha (float): 当前的 α 值
    """
    
    def __init__(self, config):
        """
        初始化时空扭曲编排器
        
        Args:
            config (dict): 课程配置，包含以下键：
                - mode: 'SpaceA', 'SpaceD', 或 'SpaceE'
                - alpha_start: 初始 α 值 (默认 0.1)
                - alpha_end: 最终 α 值 (默认 1.0)
                - curriculum_steps: 课程总步数 (默认 1e8)
                - ratio_threshold: 相对变化阈值 (默认 0.05, 即 5%)
        """
        self.mode = config.get('mode', 'SpaceA')
        self.alpha_start = config.get('alpha_start', 0.3)  # 默认从 0.3 开始，更快进入有效训练阶段
        self.alpha_end = config.get('alpha_end', 1.0)
        self.total_steps = config.get('curriculum_steps', 1e8)
        self.ratio_threshold = config.get('ratio_threshold', 0.05)  # 5% 相对变化
        
        # 当前状态
        if self.mode == 'SpaceA':
            self.current_alpha = 1.0
        else:
            self.current_alpha = self.alpha_start
        self.current_step = 0
        
        # 上一次更新时的 alpha 值（用于检测变化）
        # 重要：只有在 apply_curriculum_physics() 被调用后才更新此值
        self._last_physics_alpha = self.current_alpha
        
        # 用于日志的历史指标
        self.last_metrics = {}
        
        # 打印初始化信息
        self._print_init_info()
    
    def _print_init_info(self):
        """打印初始化信息"""
        print("\n" + "="*70)
        print("Space E 时空扭曲编排器 (Time-Warping Orchestrator)")
        print("="*70)
        print(f"  模式:           {self.mode}")
        print(f"  Alpha 范围:     {self.alpha_start:.2f} -> {self.alpha_end:.2f}")
        print(f"  课程步数:       {self.total_steps:.2e}")
        print(f"  相对变化阈值:   {self.ratio_threshold*100:.1f}%")
        print(f"  当前 Alpha:     {self.current_alpha:.3f}")
        print("-"*70)
        
        if self.mode == 'SpaceA':
            print("  [SpaceA] Baseline 模式，无任何物理缩放")
        elif self.mode == 'SpaceD':
            print("  [SpaceD] 仅物理缩放，无观测/奖励逆缩放")
            print("  警告: SpaceD 可能导致 Value Function 漂移")
        elif self.mode == 'SpaceE':
            print("  [SpaceE] 完整动力学同构缩放（推荐）")
            print("  物理层: g'=α²g, Kp'=α²Kp, Kd'=αKd, τ'=α²τ")
            print("  观测层: v_obs=v_sim/α, F_obs=F_sim/α²")
            print("  奖励层: r_final=r_raw×α")
        print("="*70 + "\n")
    
    def update(self, global_agent_steps, metrics=None):
        """
        根据全局步数和训练指标更新 alpha
        
        ============================================================
        Curriculum 更新逻辑 - 增量式自适应 (Incremental Adaptive)
        ============================================================
        
        核心思想：
        - 低 α 阶段 (α < 0.5)：重力小，物体不易掉落，主要看 survival_rate
        - 高 α 阶段 (α >= 0.5)：重力大，要求旋转达标，看 success_rate
        
        关键修正 - 防止"大坝决堤"效应：
        - target_alpha 基于 current_alpha 增量计算，而非基于全局时间
        - 每次只允许增加当前 alpha 的一小步 (ratio_threshold)
        - 避免门控打开瞬间的剧烈跳变
        
        门控条件：
        1. 低 α 阶段：survival_rate >= 0.8 才能增加 α
        2. 高 α 阶段：success_rate >= 0.7 才能增加 α (使用自适应阈值)
        
        流程:
        1. 检查门控条件是否满足（决定"是否允许向前走"）
        2. 基于 current_alpha 计算增量目标（而非基于时间）
        3. 使用基于比率的阈值判断是否需要更新物理参数
        4. 只有需要更新物理时才返回 True
        ============================================================
        
        Args:
            global_agent_steps (int): 当前的全局 Agent 步数
            metrics (dict, optional): 训练指标，包含:
                - success_rate: 成功率 (基于自适应阈值)
                - survival_rate: 存活率 (1 - 掉落/倾倒比例)
                - mean_rot_angle: 平均旋转角度
                - mean_reward: 平均奖励
                - mean_entropy: 平均熵
            
        Returns:
            bool: 如果 alpha 发生显著变化需要更新物理参数，返回 True
        """
        if self.mode == 'SpaceA':
            # SpaceA 模式下 alpha 恒为 1.0，永不更新物理
            return False
        
        self.current_step = global_agent_steps
        
        # 保存指标用于日志
        if metrics:
            self.last_metrics = metrics.copy()
        
        # ============================================================
        # 0. 统计显著性检查 - 样本量不足时跳过门控判断
        # ============================================================
        # 解决低 α 阶段的"分母消失"问题：
        # 当累积样本量不足时，survival_rate 等指标不可靠，
        # 此时不进行门控判断，等待更多数据
        if metrics and metrics.get('_insufficient_samples', False):
            # 样本量不足，跳过本次 curriculum 更新
            return False
        
        # ============================================================
        # 1. 门控检查 (Gating) - 决定"是否允许向前走"
        # ============================================================
        can_advance = False
        
        if metrics is None:
            # 如果没有指标，默认允许推进（回退到纯增量模式）
            can_advance = True
        else:
            survival_rate = metrics.get('survival_rate', 1.0)
            success_rate = metrics.get('success_rate', 0.0)
            
            if self.current_alpha < 0.5:
                # 低 α 阶段: 主要看存活率
                # 重力小 (g' = α²g ≈ 0.09g)，物体不易掉落
                # 需要 survival_rate >= 0.8 才能增加难度
                if survival_rate >= 0.8:
                    can_advance = True
            else:
                # 高 α 阶段: 必须要求旋转达标
                # 重力接近正常，需要真正学会旋转
                # success_rate 基于自适应阈值 (threshold * alpha) 计算
                if success_rate >= 0.7:
                    can_advance = True
        
        if not can_advance:
            # 没达标，原地踏步，不更新 alpha
            return False
        
        # ============================================================
        # 2. 增量式目标计算 (Incremental Target) - 防止"大坝决堤"
        # ============================================================
        # 关键修正：target_alpha 基于 current_alpha 增量计算
        # 而非基于全局时间，避免门控打开瞬间的剧烈跳变
        #
        # 每次增加当前值的 ratio_threshold 比例
        # 例如 ratio_threshold=0.05 时：
        #   - α=0.10 -> 增量 0.005 -> target=0.105
        #   - α=0.50 -> 增量 0.025 -> target=0.525
        #   - α=0.90 -> 增量 0.045 -> target=0.945
        # ============================================================
        
        increment = self.current_alpha * self.ratio_threshold
        target_alpha = self.current_alpha + increment
        
        # 可选：限速器 - 防止跑得太快（基于时间的上限）
        # 即使 Agent 表现很好，也不能超过时间进度对应的理论上限
        # time_based_progress = np.clip(global_agent_steps / self.total_steps, 0.0, 1.0)
        # time_based_limit = self.alpha_start + time_based_progress * (self.alpha_end - self.alpha_start)
        # target_alpha = min(target_alpha, time_based_limit)
        
        # 确保不超过最大值
        target_alpha = np.clip(target_alpha, self.alpha_start, self.alpha_end)
        
        # ============================================================
        # 3. 更新触发 (Update Trigger)
        # ============================================================
        # 检查增量是否足够大 (避免频繁微小更新 physics)
        # 对比的是 _last_physics_alpha，确保物理更新是阶梯状的
        # ============================================================
        
        relative_change = (target_alpha - self._last_physics_alpha) / (self._last_physics_alpha + 1e-8)
        
        if relative_change >= self.ratio_threshold:
            # 更新 alpha 并标记需要更新物理
            self.current_alpha = target_alpha
            self._last_physics_alpha = target_alpha
            return True
        
        # 不需要更新物理，current_alpha 保持不变
        # 重要: 不在这里更新 current_alpha，保证物理一致性
        return False
    
    def get_progress(self):
        """
        获取当前课程进度
        
        Returns:
            float: 进度百分比 [0, 1]
        """
        return np.clip(self.current_step / self.total_steps, 0.0, 1.0)
    
    @property
    def progress(self):
        """
        当前课程进度（属性版本）
        
        Returns:
            float: 进度百分比 [0, 1]
        """
        return self.get_progress()
    
    # ================== 物理参数缩放 (Physics Scaling) ==================
    # 遵循 Isomorphic Dynamics: F ~ α², v ~ α
    
    @property
    def gravity_scale(self):
        """
        重力缩放系数
        
        g' = α² × g
        
        原理: 在时间缩放后的世界中，自由落体需要保持相似性。
        如果时间变慢 1/α 倍，物体下落相同距离需要 1/α 倍时间。
        由 s = ½gt²，有 g' = g × (t'/t)² = g × α²
        
        Returns:
            float: 重力缩放系数 α²
        """
        return self.current_alpha ** 2
    
    @property
    def stiffness_scale(self):
        """
        PD 刚度 (Kp) 缩放系数
        
        Kp' = α² × Kp
        
        原理: F_pd = Kp × (pos_target - pos)
        我们需要 F' = α² × F (力缩放)
        位置误差不变，所以 Kp' = α² × Kp
        
        Returns:
            float: 刚度缩放系数 α²
        """
        return self.current_alpha ** 2
    
    @property
    def damping_scale(self):
        """
        PD 阻尼 (Kd) 缩放系数
        
        Kd' = α × Kd
        
        原理: F_damp = Kd × v
        我们需要 F' = α² × F，且 v' = α × v
        所以 α² × F = Kd' × (α × v)
        解得 Kd' = α × Kd
        
        Returns:
            float: 阻尼缩放系数 α
        """
        return self.current_alpha
    
    @property
    def effort_scale(self):
        """
        力矩限制 (Effort Limit) 缩放系数
        
        τ_limit' = α² × τ_limit
        
        原理: 力矩与力同比例缩放
        
        Returns:
            float: 力矩限制缩放系数 α²
        """
        return self.current_alpha ** 2
    
    # ================== PhysX 求解器阈值缩放 (Solver Threshold Scaling) ==================
    # 解决低 alpha 下的"物理粘滞"问题
    
    @property
    def bounce_threshold_scale(self):
        """
        PhysX bounce_threshold_velocity 缩放系数
        
        threshold' = α × threshold
        
        问题分析：
        - bounce_threshold_velocity: 当相对碰撞速度 < 此阈值时，PhysX 认为是
          "静止接触"，强制恢复系数为 0（完全非弹性碰撞 = 粘在一起）
        - Space E 中：v_sim = α × v_real
          当 α=0.1 时，真实 1.0 m/s 碰撞变成仿真中 0.1 m/s
          由于 0.1 < 0.2 (默认阈值)，所有接触变成粘性接触
        
        解决方案：动态缩放阈值，保持物理行为一致性
        threshold' = α × threshold
        
        Returns:
            float: bounce threshold 缩放系数 α
        """
        return self.current_alpha
    
    @property
    def contact_offset_scale(self):
        """
        PhysX contact_offset 缩放系数
        
        逻辑修正：
        原本 max(alpha, 0.5) 在极低 alpha 下会导致 offset 相对过大，产生"幽灵碰撞"。
        现在允许线性缩放至 0.1，对应 alpha=0.1。
        例如 base_offset=0.02, alpha=0.25 -> new_offset=0.005 (5mm -> 1.25mm)
        这能显著减少切向的"隔空摩擦"。
        """
        return max(self.current_alpha, 0.1)  # 修改下限为 0.1
    
    def get_scaled_physx_params(self, base_bounce_threshold, base_contact_offset):
        """
        获取缩放后的 PhysX 求解器参数
        
        Args:
            base_bounce_threshold (float): 基础 bounce_threshold_velocity
            base_contact_offset (float): 基础 contact_offset
            
        Returns:
            dict: 缩放后的 PhysX 参数
                - bounce_threshold_velocity: 缩放后的反弹阈值 (下限 0.01)
                - contact_offset: 缩放后的接触偏移 (仅低 alpha 时缩放)
                - needs_contact_offset_update: 是否需要更新 contact_offset
        """
        # 计算缩放后的 bounce threshold，设置下限防止数值问题
        scaled_bounce = max(base_bounce_threshold * self.bounce_threshold_scale, 0.01)
        
        # contact_offset 是否应该仅在低 alpha 时缩放？当前逻辑是始终缩放
        scaled_contact = base_contact_offset * self.contact_offset_scale
        
        return {
            'bounce_threshold_velocity': scaled_bounce,
            'contact_offset': scaled_contact,
            'needs_contact_offset_update': True,
        }
    
    # ================== 观测逆缩放 (Observation Inverse Scaling) ==================
    # 遵循 Space E 定义: Agent 看到的必须是 "Real World Scale"
    
    @property
    def obs_velocity_scale(self):
        """
        观测速度逆缩放系数
        
        v_real = v_sim / α
        
        原理: 仿真中的速度是 v_sim = α × v_real (变慢了)
        Agent 需要看到真实世界的速度，所以要除以 α
        
        Returns:
            float: 速度逆缩放系数 1/α
        """
        if self.mode == 'SpaceD':
            return 1.0  # SpaceD 不做观测逆缩放
        return 1.0 / (self.current_alpha + 1e-8)
    
    @property
    def obs_force_scale(self):
        """
        观测力逆缩放系数
        
        F_real = F_sim / α²
        
        原理: 仿真中的力是 F_sim = α² × F_real (变小了)
        Agent 需要看到真实世界的力，所以要除以 α²
        
        Returns:
            float: 力逆缩放系数 1/α²
        """
        if self.mode == 'SpaceD':
            return 1.0  # SpaceD 不做观测逆缩放
        return 1.0 / (self.current_alpha ** 2 + 1e-8)
    
    # ================== 奖励缩放 (Reward Scaling) ==================
    
    @property
    def reward_input_scale(self):
        """
        奖励输入还原系数
        
        用于将仿真物理量还原到真实尺度后计算奖励。
        
        v_real = v_sim × (1/α)
        ω_real = ω_sim × (1/α)
        
        注意: 如果观测已经做过逆缩放，这里不需要再做。
        此属性主要用于 compute_reward 中直接从仿真读取的值。
        
        Returns:
            float: 输入还原系数 1/α
        """
        return 1.0 / (self.current_alpha + 1e-8)
    
    @property
    def reward_output_scale(self):
        """
        奖励输出缩放系数
        
        r_final = r_raw × α
        
        原理: 在时间缩放的世界中，单步 dt 只代表真实时间的 α×dt。
        为了保持累积奖励（Return）的期望不变，需要将单步奖励乘以 α。
        
        这确保了 Value Function 在不同 α 阶段的量级稳定。
        
        Returns:
            float: 输出缩放系数 α
        """
        if self.mode == 'SpaceD':
            return 1.0  # SpaceD 不做奖励缩放（可能导致 Value 漂移）
        return self.current_alpha
    
    # ================== 辅助方法 ==================
    
    def get_scaled_physics_params(self, base_gravity, base_pgain, base_dgain, base_torque_limit):
        """
        获取缩放后的物理参数
        
        Args:
            base_gravity (list): 基础重力 [gx, gy, gz]
            base_pgain (float): 基础 PD P 增益
            base_dgain (float): 基础 PD D 增益
            base_torque_limit (float): 基础力矩限制
            
        Returns:
            dict: 缩放后的物理参数
        """
        return {
            'gravity': [g * self.gravity_scale for g in base_gravity],
            'pgain': base_pgain * self.stiffness_scale,
            'dgain': base_dgain * self.damping_scale,
            'torque_limit': base_torque_limit * self.effort_scale,
        }
    
    def __repr__(self):
        return (f"TimeWarpingOrchestrator(mode={self.mode}, "
                f"alpha={self.current_alpha:.3f}, "
                f"progress={self.get_progress()*100:.1f}%)")
