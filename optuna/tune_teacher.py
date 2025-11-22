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
    # 学习率（对数空间搜索）
    # 合理范围：包含默认值5e-3，允许探索更大和更小的值
    lr = trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True)
    hpo_overrides.append(f"train.ppo.learning_rate={lr}")
    
    # 权重衰减
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    hpo_overrides.append(f"train.ppo.weight_decay={weight_decay}")
    
    # 折扣因子
    gamma = trial.suggest_categorical("gamma", [0.98, 0.99, 0.995])
    hpo_overrides.append(f"train.ppo.gamma={gamma}")
    
    # GAE lambda参数
    tau = trial.suggest_categorical("tau", [0.90, 0.95, 0.97])
    hpo_overrides.append(f"train.ppo.tau={tau}")
    
    # PPO裁剪范围
    e_clip = trial.suggest_categorical("e_clip", [0.1, 0.2, 0.3])
    hpo_overrides.append(f"train.ppo.e_clip={e_clip}")
    
    # 熵系数（鼓励探索）
    entropy_coef = trial.suggest_float("entropy_coef", 0.0, 0.01)
    hpo_overrides.append(f"train.ppo.entropy_coef={entropy_coef}")
    
    # Critic损失系数
    critic_coef = trial.suggest_categorical("critic_coef", [0.5, 1.0, 2.0])
    hpo_overrides.append(f"train.ppo.critic_coef={critic_coef}")
    
    # KL散度阈值
    kl_threshold = trial.suggest_float("kl_threshold", 0.01, 0.05)
    hpo_overrides.append(f"train.ppo.kl_threshold={kl_threshold}")
    
    # --- B. PPO数据收集参数 ---
    # 每个mini-epoch的训练轮数
    mini_epochs = trial.suggest_int("mini_epochs", 3, 8)
    hpo_overrides.append(f"train.ppo.mini_epochs={mini_epochs}")
    
    # Minibatch大小（必须能整除batch_size=8192*12=98304）
    minibatch_size = trial.suggest_categorical("minibatch_size", [8192, 16384, 32768])
    hpo_overrides.append(f"train.ppo.minibatch_size={minibatch_size}")
    
    # --- C. 梯度优化参数 ---
    # 梯度裁剪范数
    grad_norm = trial.suggest_categorical("grad_norm", [0.5, 1.0, 2.0])
    hpo_overrides.append(f"train.ppo.grad_norm={grad_norm}")
    
    # --- E. 环境与奖励参数 ---
    
    # == 角速度参数 ==
    # 角速度裁剪下限（默认: -0.5）
    angvel_clip_min = trial.suggest_float("angvelClipMin", -1.0, -0.2)
    hpo_overrides.append(f"task.env.reward.angvelClipMin={angvel_clip_min}")
    
    # 角速度裁剪上限（默认: 0.5）
    angvel_clip_max = trial.suggest_float("angvelClipMax", 0.3, 1.0)
    hpo_overrides.append(f"task.env.reward.angvelClipMax={angvel_clip_max}")
    
    # 角速度惩罚阈值上限（默认: 1.0）
    angvel_penalty_thres_high = trial.suggest_float("angvelPenaltyThresHigh", 0.8, 1.5)
    hpo_overrides.append(f"task.env.reward.angvelPenaltyThresHigh={angvel_penalty_thres_high}")
    
    # 角速度惩罚阈值下限（默认: -0.5）
    angvel_penalty_thres_low = trial.suggest_float("angvelPenaltyThresLow", -1.0, -0.2)
    hpo_overrides.append(f"task.env.reward.angvelPenaltyThresLow={angvel_penalty_thres_low}")
    
    # == 奖励权重参数 ==
    # 旋转奖励权重（默认: 1.0）
    rotate_reward_scale = trial.suggest_float("rotate_reward_scale", 0.5, 2.0)
    hpo_overrides.append(f"task.env.reward.rotate_reward_scale={rotate_reward_scale}")
    
    # 物体线速度惩罚权重（默认: -0.3）
    obj_linvel_penalty_scale = trial.suggest_float("obj_linvel_penalty_scale", -0.6, -0.1)
    hpo_overrides.append(f"task.env.reward.obj_linvel_penalty_scale={obj_linvel_penalty_scale}")
    
    # # 航点稀疏奖励权重（默认: 0.0）
    # waypoint_sparse_reward_scale = trial.suggest_float("waypoint_sparse_reward_scale", 0.0, 0.5)
    # hpo_overrides.append(f"task.env.reward.waypoint_sparse_reward_scale={waypoint_sparse_reward_scale}")
    
    # 力矩惩罚权重（默认: -0.01）
    torque_penalty_scale = trial.suggest_float("torque_penalty_scale", -0.05, -0.001)
    hpo_overrides.append(f"task.env.reward.torque_penalty_scale={torque_penalty_scale}")
    
    # 手部姿态一致性惩罚权重（默认: -0.05）
    hand_pose_consistency_penalty_scale = trial.suggest_float("hand_pose_consistency_penalty_scale", -0.1, -0.01)
    hpo_overrides.append(f"task.env.reward.hand_pose_consistency_penalty_scale={hand_pose_consistency_penalty_scale}")
    
    # 旋转惩罚权重（逆向/超速）（默认: 0.0）
    rotate_penalty_scale = trial.suggest_float("rotate_penalty_scale", -0.5, 0.0)
    hpo_overrides.append(f"task.env.reward.rotate_penalty_scale={rotate_penalty_scale}")
    
    # 铅笔高度差惩罚权重（默认: -1.5）
    pencil_z_dist_penalty_scale = trial.suggest_float("pencil_z_dist_penalty_scale", -3.0, -0.5)
    hpo_overrides.append(f"task.env.reward.pencil_z_dist_penalty_scale={pencil_z_dist_penalty_scale}")
    
    # 位置惩罚权重（默认: -0.1）
    position_penalty_scale = trial.suggest_float("position_penalty_scale", -0.3, -0.01)
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
        
        # 检查返回码
        if result.returncode != 0:
            print(f"\n[错误] Trial {trial.number} 训练失败")
            print("STDERR:")
            print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
            raise optuna.exceptions.TrialPruned()
        
        # 从输出中提取综合评分
        # 训练脚本应该在最后一行输出: "OPTUNA_SCORE: <score>"
        composite_score = None
        for line in result.stdout.split('\n'):
            if line.startswith('OPTUNA_SCORE:'):
                try:
                    composite_score = float(line.split(':')[1].strip())
                    break
                except (IndexError, ValueError) as e:
                    print(f"警告: 无法解析评分: {line}, 错误: {e}")
        
        if composite_score is None:
            print(f"警告: Trial {trial.number} 未返回评分，使用默认值-10000")
            print("最后100行输出:")
            print('\n'.join(result.stdout.split('\n')[-100:]))
            composite_score = -10000
            
    except subprocess.TimeoutExpired:
        print(f"\n[错误] Trial {trial.number} 训练超时")
        raise optuna.exceptions.TrialPruned()
        
    except Exception as e:
        print(f"\n[错误] Trial {trial.number} 训练失败: {e}")
        import traceback
        traceback.print_exc()
        raise optuna.exceptions.TrialPruned()
    
    print("\n" + "="*80)
    print(f"[Optuna Trial {trial.number}] 训练完成")
    print(f"综合评分 (用于优化): {composite_score:.2f}")
    print("="*80 + "\n")
    
    return composite_score


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RL Teacher PPO 超参数优化")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID")
    parser.add_argument("--n_trials", type=int, default=50, help="优化试验次数")
    parser.add_argument("--max_steps", type=int, default=500_000_000, 
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