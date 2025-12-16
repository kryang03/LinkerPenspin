#!/bin/bash
# ========================================
# RL Teacher 超参数优化 - 快速启动脚本 (统一架构)
# ========================================
# 使用方法:
#   bash optuna/run_hpo.sh [GPU_ID] [N_TRIALS] [MAX_STEPS] [STUDY_NAME]
# 例如:
#   bash optuna/run_hpo.sh
# ========================================
# 统一架构说明：
# - SpaceA 已合并为 SpaceE 的特例 (alpha_start=1.0)
# - alpha_start 作为 Optuna 搜索参数，范围 [0.25, 1.0]
# - Controller 参数 (pgain, dgain, torque_limit, action_scale) 加入搜索
# ========================================
# 固定配置：
# - grasp_cache_name: 3_30000_49_nofly
# - 动作空间: disableRingLittleFinger=True (21 -> 13 DoF)
# - Flying Hand: 默认关闭
# ========================================

set -e

# ========================================
# 可配置参数
# ========================================
GPU_ID=${1:-0}
N_TRIALS=${2:-40}
MAX_STEPS=${3:-400000000}
STUDY_NAME=${4:-TP_V5}  # 统一架构版本

echo "========================================"
echo "RL Teacher 超参数优化 (统一架构)"
echo "========================================"
echo "GPU ID:       $GPU_ID"
echo "试验次数:     $N_TRIALS"
echo "每次训练步数: $MAX_STEPS"
echo "Study名称:    $STUDY_NAME"
echo "数据库:       sqlite:///optuna/hpo_${STUDY_NAME}.db"
echo "输出目录:     outputs/${STUDY_NAME}/optuna_trial_*"
echo ""
echo "统一架构 (SpaceE):"
echo "  alpha_start:    [0.25, 1.0] (Optuna 搜索)"
echo "  alpha_end:      1.0 (固定)"
echo "  注: alpha_start=1.0 等效于原 SpaceA Baseline"
echo ""
echo "Controller 参数搜索:"
echo "  pgain:          [8.0, 25.0]"
echo "  dgain:          [0.15, 0.6]"
echo "  torque_limit:   [2.0, 8.0]"
echo "  action_scale:   [0.05, 0.15]"
echo "========================================"
echo ""

# 确保optuna目录存在
mkdir -p optuna

LOG_FILE="optuna/hpo_${STUDY_NAME}.log"
echo "日志文件:     $LOG_FILE"

# 构建命令参数 (简化：不再需要 space_mode 和 alpha 参数)
CMD_ARGS=(
    --gpu "$GPU_ID"
    --n_trials "$N_TRIALS"
    --max_steps "$MAX_STEPS"
    --storage "sqlite:///optuna/hpo_${STUDY_NAME}.db"
    --study_name "$STUDY_NAME"
    --load_if_exists
)

# 运行优化（后台运行）
nohup python -u optuna/tune_teacher.py "${CMD_ARGS[@]}" > "${LOG_FILE}" 2>&1 &

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
