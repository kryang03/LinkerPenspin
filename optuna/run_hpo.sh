#!/bin/bash
# ========================================
# RL Teacher 超参数优化 - 快速启动脚本
# ========================================
# 使用方法:
#   bash optuna/run_hpo.sh <GPU_ID> [N_TRIALS] [MAX_STEPS]
#
# 示例:
#   bash optuna/run_hpo.sh 0 80 300000000
#   bash optuna/run_hpo.sh 1 30 500000000
# ========================================

set -e

GPU_ID=${1:-0}
N_TRIALS=${2:-80}
MAX_STEPS=${3:-300000000}

echo "========================================"
echo "RL Teacher 超参数优化"
echo "========================================"
echo "GPU ID:       $GPU_ID"
echo "试验次数:     $N_TRIALS"
echo "每次训练步数: $MAX_STEPS"
echo "数据库:       sqlite:///optuna/hpo_teacher.db"
echo "Study名称:    teacher_reproduce"
echo "========================================"
echo ""

# 确保optuna目录存在
mkdir -p optuna

# 运行优化（使用默认的storage和study_name）
python optuna/tune_teacher.py \
    --gpu "$GPU_ID" \
    --n_trials "$N_TRIALS" \
    --max_steps "$MAX_STEPS" \
    --storage "sqlite:///optuna/hpo_teacher_reproduce.db" \
    --study_name "teacher_reproduce" \
    --load_if_exists

echo ""
echo "========================================"
echo "优化完成！"
echo "========================================"
echo ""
echo "查看结果:"
echo "  1. 数据库: optuna/hpo_teacher.db"
echo "  2. 最佳参数: optuna/best_params_teacher_reproduce.txt"
echo "  3. 可视化图表: optuna/param_importances_teacher_reproduce.html"
echo "  4. 优化历史: optuna/optimization_history_teacher_reproduce.html"
echo "  5. TensorBoard: tensorboard --logdir outputs/LinkerHandHora/optuna_trial_*"
echo ""
echo "继续优化（累加更多试验）:"
echo "  bash optuna/run_hpo.sh $GPU_ID <更多试验次数> $MAX_STEPS"
echo ""
