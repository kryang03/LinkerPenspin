#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
清理Optuna数据库中失败的试验
"""

import argparse
import optuna

def main():
    parser = argparse.ArgumentParser(description="清理失败的Optuna试验")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna/hpo_teacher.db",
                       help="Optuna数据库路径")
    parser.add_argument("--study_name", type=str, default="teacher_ppo_hpo",
                       help="Study名称")
    parser.add_argument("--dry_run", action="store_true",
                       help="仅显示将要删除的试验，不实际删除")
    
    args = parser.parse_args()
    
    # 加载study
    study = optuna.load_study(
        study_name=args.study_name,
        storage=args.storage
    )
    
    print(f"Study: {args.study_name}")
    print(f"总试验数: {len(study.trials)}\n")
    
    # 统计各状态的试验
    state_counts = {}
    failed_trials = []
    
    for trial in study.trials:
        state = trial.state
        state_counts[state] = state_counts.get(state, 0) + 1
        
        if state in [optuna.trial.TrialState.FAIL, optuna.trial.TrialState.PRUNED]:
            failed_trials.append(trial)
    
    print("试验状态统计:")
    for state, count in state_counts.items():
        print(f"  {state.name:12s}: {count}")
    
    if failed_trials:
        print(f"\n找到 {len(failed_trials)} 个失败/被剪枝的试验:")
        for trial in failed_trials:
            print(f"  Trial {trial.number}: {trial.state.name}")
        
        if args.dry_run:
            print("\n[模拟模式] 不会删除任何试验")
        else:
            confirm = input(f"\n确认删除这 {len(failed_trials)} 个失败的试验? (yes/no): ")
            if confirm.lower() == 'yes':
                for trial in failed_trials:
                    study._storage.delete_trial(trial._trial_id)
                print(f"已删除 {len(failed_trials)} 个失败的试验")
            else:
                print("取消删除")
    else:
        print("\n没有失败的试验需要清理")

if __name__ == "__main__":
    main()
