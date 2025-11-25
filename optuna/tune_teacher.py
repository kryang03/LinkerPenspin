#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optuna超参数优化脚本 - RL Teacher PPO训练
基于 optuna/RL_TEACHER_PARAMETERS.md 中的重要参数总结

优化策略：
1. 以最终reward为主要评判标准
2. 旋转角度>10的成功案例会获得巨大加成（权重1000）
3. 综合评分 = best_reward + 1000 * success_rate

使用方法：
    python optuna/tune_teacher.py --gpu 0 --n_trials 50 --max_steps 100000000

参数说明：
    --gpu: GPU ID
    --n_trials: 优化试验次数
    --max_steps: 每次试验的最大训练步数（建议100M-300M以加快迭代）
    --storage: Optuna数据库路径（默认：sqlite:///optuna/hpo_teacher.db）
    --study_name: Study名称（默认：teacher_ppo_hpo）
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
    # 1. 固定的基础参数
    # =====================================================
    base_overrides = [
        "task=LinkerHandHora",
        "headless=True",
        "train.algo=PPOTeacher",
        f"train.ppo.max_agent_steps={args.max_steps}",
        "task.env.grasp_cache_name=3pose",
        "task.env.initPoseMode=low",
        "task.env.reset_height_threshold=0.12",
    ]
    
    # =====================================================
    # 2. 定义超参数搜索空间（基于RL_TEACHER_PARAMETERS.md）
    # =====================================================
    hpo_overrides = []
    # --- A. PPO核心算法参数 ---
    
    # 学习率 (Best: ~0.002)
    # 原范围 5e-5~5e-3。最佳值偏向高端，且较大。
    # 调整策略：缩小范围，聚焦于 5e-4 到 5e-3
    lr = trial.suggest_float("learning_rate", 5e-4, 5e-3, log=True)
    hpo_overrides.append(f"train.ppo.learning_rate={lr}")
    
    # 权重衰减 (Best: ~4.9e-5)
    # 原范围 1e-5~1e-3。最佳值偏小。
    # 调整策略：聚焦于 1e-5 到 1e-4
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-4, log=True)
    hpo_overrides.append(f"train.ppo.weight_decay={weight_decay}")
    
    # 折扣因子 (Best: 0.995)
    # 原范围 [0.98, 0.99, 0.995]。最佳值触顶。
    # 调整策略：移除 0.98，增加 0.999 尝试更长远视界
    gamma = trial.suggest_categorical("gamma", [0.99, 0.995, 0.999])
    hpo_overrides.append(f"train.ppo.gamma={gamma}")
    
    # GAE lambda (Best: 0.95)
    # 保持原样，0.95 是非常标准的 PPO 参数
    tau = trial.suggest_categorical("tau", [0.90, 0.95, 0.97])
    hpo_overrides.append(f"train.ppo.tau={tau}")
    
    # PPO裁剪范围 (Best: 0.3)
    # 原范围 [0.1, 0.2, 0.3]。最佳值触顶。
    # 调整策略：尝试更大的裁剪范围，增加 0.4
    e_clip = trial.suggest_categorical("e_clip", [0.2, 0.3, 0.4])
    hpo_overrides.append(f"train.ppo.e_clip={e_clip}")
    
    # 熵系数 (Best: ~0.005)
    # 位于中间，说明原范围 0.0~0.01 合理。
    # 调整策略：微调为 0.001 ~ 0.008
    entropy_coef = trial.suggest_float("entropy_coef", 0.001, 0.008)
    hpo_overrides.append(f"train.ppo.entropy_coef={entropy_coef}")
    
    # Critic损失系数 (Best: 1.0)
    # 保持原样
    critic_coef = trial.suggest_categorical("critic_coef", [0.5, 1.0, 2.0])
    hpo_overrides.append(f"train.ppo.critic_coef={critic_coef}")
    
    # KL散度阈值 (Best: ~0.049)
    # 原范围 0.01~0.05。最佳值严重触顶 (0.049 接近 0.05)。
    # 调整策略：大幅提升上限，探索允许更大更新步幅的可能性
    kl_threshold = trial.suggest_float("kl_threshold", 0.03, 0.08)
    hpo_overrides.append(f"train.ppo.kl_threshold={kl_threshold}")
    
    # --- B. PPO数据收集参数 ---
    
    # Mini-epochs (Best: 7)
    # 原范围 3-8。最佳值偏大。
    # 调整策略：改为 5-10
    mini_epochs = trial.suggest_int("mini_epochs", 5, 10)
    hpo_overrides.append(f"train.ppo.mini_epochs={mini_epochs}")
    
    # Minibatch大小 (Best: 8192)
    # 选择了最小的 Batch。这通常意味着更频繁的更新对当前任务有利。
    # 调整策略：尝试更小的 4096 (如果显存允许)，保留 8192
    minibatch_size = trial.suggest_categorical("minibatch_size", [4096, 8192, 16384])
    hpo_overrides.append(f"train.ppo.minibatch_size={minibatch_size}")
    
    # --- C. 梯度优化参数 ---
    
    # 梯度裁剪 (Best: 2.0)
    # 原范围 [0.5, 1.0, 2.0]。最佳值触顶。
    # 调整策略：移除 0.5，增加 4.0
    grad_norm = trial.suggest_categorical("grad_norm", [1.0, 2.0, 4.0])
    hpo_overrides.append(f"train.ppo.grad_norm={grad_norm}")
    
    # --- E. 环境与奖励参数 (基于 Best Values 重新中心化) ---
    
    # 1. 角速度相关
    # Clip Min (Best: -0.22) -> Range: -0.4 ~ -0.1
    angvel_clip_min = trial.suggest_float("angvelClipMin", -0.4, -0.1)
    hpo_overrides.append(f"task.env.reward.angvelClipMin={angvel_clip_min}")
    
    # Clip Max (Best: 0.73) -> Range: 0.5 ~ 0.9
    angvel_clip_max = trial.suggest_float("angvelClipMax", 0.5, 0.9)
    hpo_overrides.append(f"task.env.reward.angvelClipMax={angvel_clip_max}")
    
    # Penalty High (Best: 0.805) -> 原范围 0.8~1.5，触底！
    # 必须降低下限
    angvel_penalty_thres_high = trial.suggest_float("angvelPenaltyThresHigh", 0.5, 1.0)
    hpo_overrides.append(f"task.env.reward.angvelPenaltyThresHigh={angvel_penalty_thres_high}")
    
    # Penalty Low (Best: -0.67) -> Range: -0.8 ~ -0.5
    angvel_penalty_thres_low = trial.suggest_float("angvelPenaltyThresLow", -0.8, -0.5)
    hpo_overrides.append(f"task.env.reward.angvelPenaltyThresLow={angvel_penalty_thres_low}")
    
    # 2. 奖励权重相关 (Reward Scales)
    
    # 旋转奖励 (Best: 1.7) -> 偏向高值 (原范围 0.5~2.0)
    # 调整：1.0 ~ 2.5，鼓励进一步加大旋转奖励比重
    rotate_reward_scale = trial.suggest_float("rotate_reward_scale", 1.0, 2.5)
    hpo_overrides.append(f"task.env.reward.rotate_reward_scale={rotate_reward_scale}")
    
    # 物体线速度惩罚 (Best: -0.11) -> 偏向低惩罚 (原范围 -0.6 ~ -0.1)
    # 调整：-0.2 ~ -0.05 (允许更小的惩罚)
    obj_linvel_penalty_scale = trial.suggest_float("obj_linvel_penalty_scale", -0.2, -0.05)
    hpo_overrides.append(f"task.env.reward.obj_linvel_penalty_scale={obj_linvel_penalty_scale}")
    
    # 力矩惩罚 (Best: -0.02) -> 中间值
    # 调整：-0.04 ~ -0.01
    torque_penalty_scale = trial.suggest_float("torque_penalty_scale", -0.04, -0.01)
    hpo_overrides.append(f"task.env.reward.torque_penalty_scale={torque_penalty_scale}")
    
    # 姿态一致性惩罚 (Best: -0.03) -> 偏向小惩罚
    hand_pose_consistency_penalty_scale = trial.suggest_float("hand_pose_consistency_penalty_scale", -0.06, -0.01)
    hpo_overrides.append(f"task.env.reward.hand_pose_consistency_penalty_scale={hand_pose_consistency_penalty_scale}")
    
    # 旋转惩罚 (Best: -0.005) -> 非常小，几乎忽略不计
    # 调整：-0.1 ~ 0.0
    rotate_penalty_scale = trial.suggest_float("rotate_penalty_scale", -0.1, 0.0)
    hpo_overrides.append(f"task.env.reward.rotate_penalty_scale={rotate_penalty_scale}")
    
    # 铅笔高度差惩罚 (Best: -1.5) -> 中间值
    # 调整：-2.0 ~ -1.0
    pencil_z_dist_penalty_scale = trial.suggest_float("pencil_z_dist_penalty_scale", -2.0, -1.0)
    hpo_overrides.append(f"task.env.reward.pencil_z_dist_penalty_scale={pencil_z_dist_penalty_scale}")
    
    # 位置惩罚 (Best: -0.22) -> 中间偏高
    # 调整：-0.3 ~ -0.1
    position_penalty_scale = trial.suggest_float("position_penalty_scale", -0.3, -0.1)
    hpo_overrides.append(f"task.env.reward.position_penalty_scale={position_penalty_scale}")
    
    # --- 3. 创建唯一输出目录 ---
    output_dir = f"LinkerHandHora/optuna_trial_{trial.number:04d}"
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
                       help="Study名称")
    parser.add_argument("--load_if_exists", action="store_true",
                       help="如果study已存在则加载并继续")
    
    args = parser.parse_args()
    
    # 设置GPU环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # 确保数据库目录存在
    db_dir = os.path.dirname(args.storage.replace("sqlite:///", ""))
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    
    # 创建或加载Study
    print("\n" + "="*80)
    print("Optuna 超参数优化 - RL Teacher PPO")
    print("="*80)
    print(f"GPU ID:           {args.gpu}")
    print(f"试验次数:         {args.n_trials}")
    print(f"每次训练步数:     {args.max_steps:,}")
    print(f"数据库:           {args.storage}")
    print(f"Study名称:        {args.study_name}")
    print(f"加载已有study:    {args.load_if_exists}")
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
        
        # 保存最佳参数到文件
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