#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optuna超参数优化脚本 - RL Teacher PPO训练
基于 optuna/RL_TEACHER_PARAMETERS.md 中的重要参数总结
CUDA_VISIBLE_DEVICES=0 python train.py task=LinkerHandHora headless=True seed=42 train.ppo.output_name=test_mem_info train.algo=PPOTeacher task.env.grasp_cache_name=3pose task.env.initPoseMode=low task.env.reset_height_threshold=0.12 task.env.numEnvs=8192 train.ppo.horizon_length=32 train.ppo.minibatch_size=32768
优化策略：
1. 以最终reward为主要评判标准
2. 旋转角度>10的成功案例会获得巨大加成（权重1000）
3. 综合评分 = best_reward + 1000 * success_rate

使用方法：
    # SpaceA 模式 (Baseline)
    python optuna/tune_teacher.py --gpu 0 --n_trials 50 --max_steps 100000000 --space_mode SpaceA
    
    # SpaceE 模式 (Curriculum Learning)
    python optuna/tune_teacher.py --gpu 0 --n_trials 50 --max_steps 100000000 --space_mode SpaceE

参数说明：
    --gpu: GPU ID
    --n_trials: 优化试验次数
    --max_steps: 每次试验的最大训练步数（建议100M-300M以加快迭代）
    --storage: Optuna数据库路径（默认：sqlite:///optuna/hpo_teacher.db）
    --study_name: Study名称（默认：teacher_ppo_hpo）
    --space_mode: 训练模式 SpaceA (Baseline) 或 SpaceE (Curriculum Learning)
    --alpha_start: SpaceE 模式的初始 alpha 值（默认 0.1）
    --alpha_end: SpaceE 模式的最终 alpha 值（默认 1.0）
    --curriculum_steps: SpaceE 模式的 curriculum 总步数（默认 100000000）
"""

import os
import sys
import argparse
import subprocess
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# 获取项目根目录（optuna/tune_teacher.py的父目录的父目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def objective(trial: optuna.trial.Trial, args) -> float:
    """
    Optuna优化目标函数
    
    Args:
        trial: Optuna trial对象
        args: 命令行参数
        
    Returns:
        composite_score: 综合评分（reward为主 + 成功率巨大加成）
    """
    
    # =====================================================
    # 1. 固定的基础参数（与 scripts/train_rl_teacher.sh 保持一致）
    # =====================================================
    base_overrides = [
        "task=LinkerHandHora",
        "headless=True",
        "train.algo=PPOTeacher",
        f"train.ppo.max_agent_steps={args.max_steps}",
        # Grasp cache 配置（非 Flying Hand）
        "task.env.grasp_cache_name='3_30000_49_nofly'",
        "task.env.initPoseMode=low",
        # 早期终止阈值
        "task.env.relative_z_drop_threshold=0.05",
        "task.env.pencil_tilt_threshold=0.06",
        # 动作空间配置：禁用无名指和小拇指 (21 -> 13 DoF)
        "task.env.actionSpace.disableRingLittleFinger=True",
        # 默认关闭 Flying Hand（固定底座 21 DoF）
        "task.env.flyingHand.enabled=False",
        "task.env.asset.handAsset='assets/linker_hand/L25_dof_urdf.urdf'",
        "task.env.numActions=21",
        "task.env.numObservations=126",
    ]
    
    # =====================================================
    # Space E 配置 (统一架构: SpaceA 是 alpha_start=1.0 的特例)
    # =====================================================
    # 核心思想: 不再区分 SpaceA 和 SpaceE
    # - alpha_start 作为 Optuna 搜索参数
    # - alpha_start = 1.0 等效于原来的 SpaceA (无物理缩放)
    # - alpha_start < 1.0 启用 Space E 课程学习
    
    # 搜索最佳初始 Alpha
    # 范围: 0.25 (慢) ~ 1.0 (真实物理)
    # alpha=1.0 等效于 SpaceA baseline
    alpha_start = trial.suggest_float("alpha_start", 0.25, 1.0)
    
    base_overrides.extend([
        "task.env.curriculum.mode=SpaceE",  # 统一使用 SpaceE 模式
        f"task.env.curriculum.alpha_start={alpha_start}",
        "task.env.curriculum.alpha_end=1.0",  # 强制 alpha_end 为 1.0
        "task.env.curriculum.ratio_threshold=0.05",
        "task.env.curriculum.success_rot_threshold=10.0",
        "task.env.curriculum.use_adaptive_threshold=True",
    ])
    
    print(f"[Space E] Alpha: {alpha_start:.3f} -> 1.0")
    if alpha_start >= 0.99:
        print("  (等效于 SpaceA Baseline: 无物理缩放)")
    
    # =====================================================
    # 2. 定义超参数搜索空间（基于RL_TEACHER_PARAMETERS.md）
    # =====================================================
    hpo_overrides = []
    # --- A. PPO核心算法参数 ---
    
    # 学习率 (Best: ~0.002)
    # 原范围 5e-5~5e-3。最佳值偏向高端，且较大。
    # 调整策略：缩小范围，聚焦于 1e-4 到 1e-3
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    hpo_overrides.append(f"train.ppo.learning_rate={lr}")
    
    # 权重衰减 (Best: ~4.9e-5)
    # 原范围 1e-5~1e-3。最佳值偏小。
    # 调整策略：聚焦于 1e-5 到 1e-4
    weight_decay = trial.suggest_float("weight_decay", 5e-6, 5e-5, log=True)
    hpo_overrides.append(f"train.ppo.weight_decay={weight_decay}")
    
    # 折扣因子 (Best: 0.995)
    # 当前动作对未来累积奖励的影响程度
    gamma = trial.suggest_categorical("gamma", [0.985, 0.99, 0.992, 0.995])
    hpo_overrides.append(f"train.ppo.gamma={gamma}")
    
    # GAE lambda (Best: 0.95)
    # GAE (Generalized Advantage Estimation) 的平滑系数，用于权衡偏差（Bias）和方差（Variance）
    tau = trial.suggest_categorical("tau", [0.85, 0.90, 0.95, 0.98])
    hpo_overrides.append(f"train.ppo.tau={tau}")
    
    # PPO裁剪范围 (Best: 0.3)
    # 限制了新策略 $\pi_{new}$ 和旧策略 $\pi_{old}$ 之间的差异幅度
    e_clip = trial.suggest_categorical("e_clip", [0.1, 0.2, 0.3, 0.4])
    hpo_overrides.append(f"train.ppo.e_clip={e_clip}")
    
    # 熵系数 (Best: ~0.005)
    # 数值越大，智能体越倾向于采取随机动作；数值越小，策略越容易收敛到确定性行为
    entropy_coef = trial.suggest_float("entropy_coef", 0.001, 0.008)
    hpo_overrides.append(f"train.ppo.entropy_coef={entropy_coef}")
    
    # Critic损失系数 (Best: 1.0)
    # Loss = Loss_{Actor} + critic_coef * Loss_{Critic}，相对较高的权重意味着在这个任务配置中，准确估计当前状态的价值（V值）非常重要
    critic_coef = trial.suggest_categorical("critic_coef", [0.5, 1.0, 2.0])
    hpo_overrides.append(f"train.ppo.critic_coef={critic_coef}")
    
    # KL散度阈值 (Best: ~0.049)
    # 计算新旧策略之间的 KL 散度，太剧烈降低学习率，太保守增大学习率
    # 扩大范围以允许更大的更新步幅，帮助跳出局部最优
    kl_threshold = trial.suggest_float("kl_threshold", 0.008, 0.08)
    hpo_overrides.append(f"train.ppo.kl_threshold={kl_threshold}")
    
    # KL 自适应因子 (原 1.15)
    # 控制学习率调整的幅度: LR *= factor 或 LR /= factor
    # 较大的因子允许更激进的调整，有助于逃离死锁
    kl_adaptive_factor = trial.suggest_categorical("kl_adaptive_factor", [1.1, 1.15, 1.2, 1.5])
    hpo_overrides.append(f"train.ppo.kl_adaptive_factor={kl_adaptive_factor}")
    
    # 最小学习率下限 (防止死锁)
    # 即使 KL 很大，LR 也不会低于此值，保持一定探索能力
    min_lr = trial.suggest_float("min_lr", 1e-6, 1e-4, log=True)
    hpo_overrides.append(f"train.ppo.min_lr={min_lr}")
    
    # --- B. PPO数据收集参数 ---
    
    # Mini-epochs
    mini_epochs = trial.suggest_int("mini_epochs", 4, 10)
    hpo_overrides.append(f"train.ppo.mini_epochs={mini_epochs}")
    
    # Minibatch大小
    minibatch_size = trial.suggest_categorical("minibatch_size", [8192, 16384, 32768])
    hpo_overrides.append(f"train.ppo.minibatch_size={minibatch_size}")
    
    # --- C. 梯度优化参数 ---
    
    # 梯度裁剪 (Best: 2.0)
    # 原范围 [0.5, 1.0, 2.0]。最佳值触顶。
    # 调整策略：移除 0.5，增加 4.0
    grad_norm = trial.suggest_categorical("grad_norm", [0.25, 0.5, 1.0, 2.0])
    hpo_overrides.append(f"train.ppo.grad_norm={grad_norm}")
    
    # --- D. 环境与数据收集规模参数 ---
    
    # 环境数量 (默认: 8192)
    # 影响并行采样效率和样本多样性
    # 范围：4096, 8192
    num_envs = trial.suggest_categorical("num_envs", [8192])
    hpo_overrides.append(f"task.env.numEnvs={num_envs}")
    
    # Horizon长度 (默认: 16)
    # 影响每次采样的轨迹长度和更新频率
    # 范围：16, 24, 32
    horizon_length = trial.suggest_categorical("horizon_length", [16, 24 ,32])
    hpo_overrides.append(f"train.ppo.horizon_length={horizon_length}")
    
    # --- E. 物理控制器参数 (Physics Controller) ---
    # 核心痛点：低 Alpha 下手太软。让 Optuna 寻找最佳刚度。
    
    # 1. P-Gain (刚度)
    # 原值 15.0。范围扩大到 [8.0, 25.0]
    p_gain = trial.suggest_float("controller_pgain", 12.0, 25.0)
    hpo_overrides.append(f"task.env.controller.pgain={p_gain}")
    
    # 2. D-Gain (阻尼)
    # 原值 0.35。通常随 P 增大。范围 [0.15, 0.6]
    # 经验公式: K_d ≈ ratio * K_p, ratio ∈ [0.02, 0.03]
    d_gain = trial.suggest_float("controller_dgain", 0.15, 0.6)
    hpo_overrides.append(f"task.env.controller.dgain={d_gain}")
    
    # 3. Torque Limit (力矩限制)
    # 原值 4.0。必须随 P 增大。范围 [2.0, 8.0]
    torque_limit = trial.suggest_float("controller_torque_limit", 5.0, 8.0)
    hpo_overrides.append(f"task.env.controller.torque_limit={torque_limit}")
    
    # 4. Action Scale (动作缩放)
    # 原值 0.1。影响灵敏度。范围 [0.05, 0.15]
    action_scale = trial.suggest_float("controller_action_scale", 0.08, 0.15)
    hpo_overrides.append(f"task.env.controller.action_scale={action_scale}")
    
    # --- F. 奖励参数 (基于 Best Values 重新中心化) ---
    
    # 1. 角速度相关
    # - Gaussian-kernel: 使用 target_angvel 和 angvel_sigma
    
    # 使用 Gaussian kernel 角速度奖励
    hpo_overrides.append("task.env.reward.use_gaussian_angvel_reward=True")
    
    # Gaussian kernel 参数
    # 目标角速度: π rad/s ≈ 0.5 圈/s 是一个合理的起点
    target_angvel = trial.suggest_float("target_angvel", 6, 10)  # 4.0 ~ 6.0 rad/s
    hpo_overrides.append(f"task.env.reward.target_angvel={target_angvel}")
    
    # 高斯核带宽 σ: 越小奖励越陡峭
    angvel_reward_sigma = trial.suggest_float("angvel_sigma", 1.0, 2.0)
    hpo_overrides.append(f"task.env.reward.angvel_sigma={angvel_reward_sigma}")
    
    # 2. 奖励权重相关 (Reward Scales)
    
    # 旋转奖励 (Best: 1.7) -> 偏向高值 (原范围 0.5~2.0)
    # 调整：1.0 ~ 2.5，鼓励进一步加大旋转奖励比重
    rotate_reward_scale = trial.suggest_float("rotate_reward_scale", 1.5, 2.5)
    hpo_overrides.append(f"task.env.reward.rotate_reward_scale={rotate_reward_scale}")
    
    # 物体线速度惩罚 (Best: -0.11) -> 偏向低惩罚 (原范围 -0.6 ~ -0.1)
    # 调整：-0.2 ~ -0.05 (允许更小的惩罚)
    obj_linvel_penalty_scale = trial.suggest_float("obj_linvel_penalty_scale", -0.2, -0.05)
    hpo_overrides.append(f"task.env.reward.obj_linvel_penalty_scale={obj_linvel_penalty_scale}")
    
    # 力矩惩罚 (Best: -0.02) -> 中间值
    # 调整：-0.04 ~ -0.01
    torque_penalty_scale = trial.suggest_float("torque_penalty_scale", -0.04, -0.01)
    hpo_overrides.append(f"task.env.reward.torque_penalty_scale={torque_penalty_scale}")
    
    # 旋转惩罚 (Best: -0.005) -> 非常小，几乎忽略不计
    # 调整：-0.1 ~ 0.0
    rotate_penalty_scale = trial.suggest_float("rotate_penalty_scale", -0.15, 0.0)
    hpo_overrides.append(f"task.env.reward.rotate_penalty_scale={rotate_penalty_scale}")
    
    # 轴向倾斜惩罚 (Best: -1.5) -> 中间值
    # 调整：-2.0 ~ -1.0
    axial_tilt_penalty_scale = trial.suggest_float("axial_tilt_penalty_scale", -2.5, -1.5)
    hpo_overrides.append(f"task.env.reward.axial_tilt_penalty_scale={axial_tilt_penalty_scale}")
    
    # 轴向倾斜阈值（米），低于此值不惩罚
    # 调整：0.02 ~ 0.05
    axial_tilt_threshold = trial.suggest_float("axial_tilt_threshold", 0.01, 0.03)
    hpo_overrides.append(f"task.env.reward.axial_tilt_threshold={axial_tilt_threshold}")
    
    # 位置惩罚 (Best: -0.22) -> 中间偏高
    # 调整：-0.3 ~ -0.1
    position_penalty_scale = trial.suggest_float("position_penalty_scale", -0.3, -0.1)
    hpo_overrides.append(f"task.env.reward.position_penalty_scale={position_penalty_scale}")
    
    # Flying base 移动惩罚（Flying Hand 关闭时此项几乎不生效）
    # 鼓励策略依赖手指技巧而非手腕运动
    # flying_base_penalty_scale = trial.suggest_float("flying_base_movement_penalty_scale", -0.2, -0.05)
    # hpo_overrides.append(f"task.env.reward.flying_base_movement_penalty_scale={flying_base_penalty_scale}")
    
    # Waypoint 跟踪奖励 (Triangle Pass)
    # 默认禁用，仅在填充 waypoint 数据后启用
    # 调整：0.0 ~ 1.0
    waypoint_tracking_reward_scale = trial.suggest_float("waypoint_tracking_reward_scale", 5.0, 10.0)
    hpo_overrides.append(f"task.env.reward.waypoint_tracking_reward_scale={waypoint_tracking_reward_scale}")
    
    # Waypoint 高斯核带宽
    # 调整：0.03 ~ 0.1
    waypoint_sigma = trial.suggest_float("waypoint_sigma", 0.03, 0.4)
    hpo_overrides.append(f"task.env.reward.waypoint_sigma={waypoint_sigma}")
    
    # 对数缩放灵敏度控制参数
    # 范围: 0.5 ~ 5.0，越小越线性，越大大力压缩更强
    log_scale_beta = trial.suggest_float("log_scale_beta", 0.5, 7.0)
    hpo_overrides.append(f"task.env.reward.log_scale_beta={log_scale_beta}")
    
    # --- 4. [Anti-Hacking] EMA 平滑和惩罚参数 ---
    
    # EMA 平滑系数 (仅用于速度门控，较小值=更平滑)
    # 范围: 0.1 ~ 0.3，越小越平滑但响应越慢
    ema_alpha = trial.suggest_float("ema_alpha", 0.1, 0.3)
    hpo_overrides.append(f"task.env.reward.ema_alpha={ema_alpha}")
    
    # Jitter 惩罚权重 (负值因为是惩罚)
    # 范围: -1.0 ~ -0.1
    jitter_penalty_scale = trial.suggest_float("jitter_penalty_scale", -1.5, -0.5)
    hpo_overrides.append(f"task.env.reward.jitter_penalty_scale={jitter_penalty_scale}")
    
    # 反向旋转惩罚权重 (负值因为是惩罚)
    # 范围: -2.0 ~ -0.5
    reverse_penalty_scale = trial.suggest_float("reverse_penalty_scale", -2.0, -0.5)
    hpo_overrides.append(f"task.env.reward.reverse_penalty_scale={reverse_penalty_scale}")
    
    # --- 5. 创建唯一输出目录 ---
    # 使用 study_name 作为输出目录的基础
    output_dir = f"{args.study_name}/optuna_trial_{trial.number:04d}"
    hpo_overrides.append(f"train.ppo.output_name={output_dir}")
    
    # 固定种子以保证可复现性（可选）
    seed = 42 + trial.number
    hpo_overrides.append(f"seed={seed}")
    
    # =====================================================
    # 4. 运行训练（在独立子进程中，避免Isaac Gym重复初始化问题）
    # =====================================================
    
    # 构建完整的命令行参数
    cmd_args = ["python", "train.py"] + base_overrides + hpo_overrides
    
    print("\n" + "="*80)
    print(f"[Optuna Trial {trial.number}] 开始训练")
    print("="*80)
    print(f"试验参数:")
    for key, value in trial.params.items():
        print(f"  {key:30s} = {value}")
    print(f"输出目录: outputs/{output_dir}")
    print("="*80 + "\n")
    
    try:
        # 使用subprocess在独立进程中运行训练
        # 这样可以避免Isaac Gym Foundation对象重复创建的问题
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        
        # 运行训练并捕获输出
        result = subprocess.run(
            cmd_args,
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=None  # 不设置超时
        )
        
        # 提取并打印 priv_info 配置信息（在训练开始时输出）
        try:
            priv_info_section = []
            in_priv_section = False
            equals_count = 0
            for line in result.stdout.split('\n'):
                if '特权信息 (Privileged Information) 维度配置:' in line:
                    in_priv_section = True
                    # 从前一个分隔线开始
                    if priv_info_section and priv_info_section[-1].startswith('='*80):
                        priv_info_section = [priv_info_section[-1]]
                
                if in_priv_section:
                    priv_info_section.append(line)
                    # 计数等号分隔线，需要至少2个才是完整的section
                    if line.startswith('='*80):
                        equals_count += 1
                        if equals_count >= 3:  # 开始、标题后、结束
                            break
            
            if priv_info_section and len(priv_info_section) > 3:
                print("\n" + '\n'.join(priv_info_section))
        except Exception as priv_error:
            print(f"警告: 提取priv_info配置失败: {priv_error}")
        
        # 检查返回码
        if result.returncode != 0:
            print(f"\n[错误] Trial {trial.number} 训练失败")
            print("STDERR:")
            print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
            raise optuna.exceptions.TrialPruned()
        
        # 从输出中提取三个独立指标
        # 训练脚本应该输出: "OPTUNA_METRICS: reward=X.XX success_rate=X.XX mean_rot_angle=X.XX"
        best_reward = None
        success_rate = None
        mean_rot_angle = None
        
        for line in result.stdout.split('\n'):
            if line.startswith('OPTUNA_METRICS:'):
                try:
                    # 解析格式: "OPTUNA_METRICS: reward=X.XX success_rate=X.XX mean_rot_angle=X.XX"
                    metrics_str = line.split('OPTUNA_METRICS:')[1].strip()
                    for pair in metrics_str.split():
                        key, value = pair.split('=')
                        if key == 'reward':
                            best_reward = float(value)
                        elif key == 'success_rate':
                            success_rate = float(value)
                        elif key == 'mean_rot_angle':
                            mean_rot_angle = float(value)
                    break
                except (IndexError, ValueError) as e:
                    print(f"警告: 无法解析指标: {line}, 错误: {e}")
        
        # 检查是否成功解析所有指标
        if best_reward is None or success_rate is None or mean_rot_angle is None:
            print(f"警告: Trial {trial.number} 未返回完整指标，使用默认值")
            print("最后100行输出:")
            print('\n'.join(result.stdout.split('\n')[-100:]))
            best_reward = best_reward or -100.0
            success_rate = success_rate or 0.0
            mean_rot_angle = mean_rot_angle or 0.0
        
        # 打印三个独立指标
        print(f"\n训练指标:")
        print(f"  Best Reward:         {best_reward:.4f}")
        print(f"  Success Rate:        {success_rate:.4f}")
        print(f"  Mean Rot Angle(rad): {mean_rot_angle:.4f}")
        
        # 计算综合评分:重要性依次递增
        # 权重设计:reward (1x) < success_rate (100x) < mean_rot_angle (1000x)
        # 这样mean_rot_angle对评分影响最大,success_rate次之,reward影响最小
        try:
            composite_score = (
                1.0 * best_reward +           # 基础奖励权重
                100.0 * success_rate +        # 成功率权重(中等重要)
                100.0 * mean_rot_angle       # 平均旋转角度权重(最重要)
            )
            print(f"  Composite Score:     {composite_score:.2f}")
            print(f"    = 1.0×{best_reward:.2f} + 100.0×{success_rate:.4f} + 100.0×{mean_rot_angle:.4f}")
        except Exception as calc_error:
            print(f"\n[错误] 计算综合评分失败: {calc_error}")
            print(f"  best_reward={best_reward}, success_rate={success_rate}, mean_rot_angle={mean_rot_angle}")
            raise optuna.exceptions.TrialPruned()
            
    except subprocess.TimeoutExpired:
        print(f"\n[错误] Trial {trial.number} 训练超时")
        raise optuna.exceptions.TrialPruned()
        
    except Exception as e:
        print(f"\n[错误] Trial {trial.number} 训练失败")
        print(f"异常类型: {type(e).__name__}")
        print(f"异常信息: {str(e)}")
        import traceback
        print("\n完整调用栈:")
        traceback.print_exc()
        
        # 尝试输出更多调试信息
        try:
            if 'result' in locals():
                print(f"\n进程返回码: {result.returncode}")
                print(f"STDOUT最后500字符:")
                print(result.stdout[-500:] if result.stdout else "(无输出)")
                print(f"\nSTDERR最后500字符:")
                print(result.stderr[-500:] if result.stderr else "(无错误)")
        except:
            pass
            
        raise optuna.exceptions.TrialPruned()
    
    print("\n" + "="*80)
    print(f"[Optuna Trial {trial.number}] 训练完成")
    print("="*80 + "\n")
    
    return composite_score


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RL Teacher PPO 超参数优化")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID")
    parser.add_argument("--n_trials", type=int, default=50, help="优化试验次数")
    parser.add_argument("--max_steps", type=int, default=300_000_000, 
                       help="每次试验的最大训练步数（建议100M-300M）")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna/hpo_teacher.db",
                       help="Optuna数据库存储路径")
    parser.add_argument("--study_name", type=str, default="teacher_ppo_hpo",
                       help="Study名称（也用作输出目录名）")
    parser.add_argument("--load_if_exists", action="store_true",
                       help="如果study已存在则加载并继续")
    # 注意: alpha_start 现在作为 Optuna 搜索参数，不再从命令行传入
    # SpaceA 是 SpaceE 在 alpha_start=1.0 时的特例
    
    args = parser.parse_args()
    
    # 设置GPU环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # 确保数据库目录存在
    db_dir = os.path.dirname(args.storage.replace("sqlite:///", ""))
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    
    # 创建或加载Study
    print("\n" + "="*80)
    print("Optuna 超参数优化 - RL Teacher PPO (统一架构)")
    print("="*80)
    print(f"GPU ID:           {args.gpu}")
    print(f"试验次数:         {args.n_trials}")
    print(f"每次训练步数:     {args.max_steps:,}")
    print(f"数据库:           {args.storage}")
    print(f"Study名称:        {args.study_name}")
    print(f"加载已有study:    {args.load_if_exists}")
    print(f"架构说明:         SpaceE 统一架构 (alpha_start 作为搜索参数)")
    print(f"  alpha_start:    [0.25, 1.0] (Optuna 搜索)")
    print(f"  alpha_end:      1.0 (固定)")
    print(f"  注: alpha_start=1.0 等效于原 SpaceA Baseline")
    print("="*80 + "\n")
    
    # 使用TPE采样器和Median剪枝器
    sampler = TPESampler(seed=42, n_startup_trials=10)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.load_if_exists,
        direction="maximize",  # 最大化综合评分
        sampler=sampler,
        pruner=pruner,
    )
    
    print(f"Study状态: {'已加载' if args.load_if_exists else '新建'}")
    if args.load_if_exists and len(study.trials) > 0:
        print(f"已完成的试验数: {len(study.trials)}")
        # 只有在有成功完成的试验时才显示最佳值
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed_trials:
            print(f"当前最佳综合评分: {study.best_value:.2f}")
            print(f"当前最佳参数:")
            for key, value in study.best_params.items():
                print(f"  {key:30s} = {value}")
        else:
            print("警告: 没有成功完成的试验，将从头开始")
    print("\n开始优化...\n")
    
    # 启动优化
    try:
        study.optimize(
            lambda trial: objective(trial, args),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[中断] 用户手动停止优化")
    
    # 打印最终结果
    print("\n" + "="*80)
    print("优化完成！")
    print("="*80)
    print(f"完成的试验数:     {len(study.trials)}")
    
    # 检查是否有成功完成的试验
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if completed_trials:
        print(f"成功完成的试验:   {len(completed_trials)}")
        print(f"最佳综合评分:     {study.best_value:.2f}")
        print(f"最佳试验编号:     {study.best_trial.number}")
        print("\n最佳超参数:")
        print("-"*80)
        for key, value in study.best_params.items():
            print(f"  {key:30s} = {value}")
        print("="*80)
        
        # 保存最佳参数到文件（使用 study_name 作为标识）
        best_params_file = f"optuna/best_params_{args.study_name}.txt"
        with open(best_params_file, "w") as f:
            f.write(f"Best Trial Number: {study.best_trial.number}\n")
            f.write(f"Best Composite Score: {study.best_value:.2f}\n")
            f.write(f"\nBest Hyperparameters:\n")
            for key, value in study.best_params.items():
                f.write(f"  {key}: {value}\n")
        print(f"\n最佳参数已保存到: {best_params_file}")
    else:
        print("警告: 没有成功完成的试验！")
        print("请检查训练日志并解决问题后重试。")
    
    # 生成可视化（需要安装plotly）
    if completed_trials:
        try:
            import optuna.visualization as vis
            
            # 参数重要性
            fig = vis.plot_param_importances(study)
            fig.write_html(f"optuna/param_importances_{args.study_name}.html")
            print(f"参数重要性图已保存到: optuna/param_importances_{args.study_name}.html")
            
            # 优化历史
            fig = vis.plot_optimization_history(study)
            fig.write_html(f"optuna/optimization_history_{args.study_name}.html")
            print(f"优化历史图已保存到: optuna/optimization_history_{args.study_name}.html")
            
        except ImportError as e:
            print(f"\n提示: 无法导入可视化库: {e}")
            print("  pip install plotly kaleido")
        except Exception as e:
            print(f"\n警告: 生成可视化图表时出错: {e}")
            print("  这不影响优化结果，可以忽略此错误")


if __name__ == "__main__":
    main()