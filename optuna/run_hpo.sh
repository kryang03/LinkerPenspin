#!/bin/bash
# ========================================
# RL Teacher 超参数优化 - 快速启动脚本
# ========================================
# 使用方法:
#   bash optuna/run_hpo.sh 
# ========================================

set -e

# ========================================
# 可配置参数
# ========================================
STUDY_NAME="teacher_reproduce_trial_04"  # Study名称（同时用作输出目录）

GPU_ID=${1:-0}
N_TRIALS=${2:-40}
MAX_STEPS=${3:-250000000}

echo "========================================"
echo "RL Teacher 超参数优化"
echo "========================================"
echo "GPU ID:       $GPU_ID"
echo "试验次数:     $N_TRIALS"
echo "每次训练步数: $MAX_STEPS"
echo "Study名称:    $STUDY_NAME (同时用作输出目录)"
echo "数据库:       sqlite:///optuna/hpo_${STUDY_NAME}.db"
echo "输出目录:     outputs/${STUDY_NAME}/optuna_trial_*"
echo "========================================"
echo ""

# 确保optuna目录存在
mkdir -p optuna

LOG_FILE="optuna/hpo_${STUDY_NAME}.log"
echo "日志文件:     $LOG_FILE"

# 运行优化（后台运行）
nohup python -u optuna/tune_teacher.py \
    --gpu "$GPU_ID" \
    --n_trials "$N_TRIALS" \
    --max_steps "$MAX_STEPS" \
    --storage "sqlite:///optuna/hpo_${STUDY_NAME}.db" \
    --study_name "$STUDY_NAME" \
    --load_if_exists > "${LOG_FILE}" 2>&1 &

PID=$!
echo "HPO 已在后台启动，PID: $PID"
echo "查看日志: tail -f $LOG_FILE"

echo ""
echo "========================================"
echo "优化任务已提交后台"
echo "========================================"
echo ""
echo "查看结果:"
echo "  1. 数据库: optuna/hpo_${STUDY_NAME}.db"
echo "  2. 最佳参数: optuna/best_params_${STUDY_NAME}.txt"
echo "  3. 可视化图表: optuna/param_importances_${STUDY_NAME}.html"
echo "  4. 优化历史: optuna/optimization_history_${STUDY_NAME}.html"
echo "  5. TensorBoard: tensorboard --logdir outputs/${STUDY_NAME}/optuna_trial_*"
echo ""
echo "停止任务: kill $PID ; 还要 ps aux | grep train.py 杀死所有子进程"
echo ""
