#!/bin/bash
# ========================================
# 测试超参数优化功能
# ========================================
# 这个脚本会运行一个快速测试来验证HPO系统是否正常工作
# 使用极少的训练步数（1M步）来快速验证
# ========================================
# 使用方法:
#   bash optuna/test_hpo.sh [SPACE_MODE]
# 例如:
#   bash optuna/test_hpo.sh SpaceA    # 测试 Baseline 模式
#   bash optuna/test_hpo.sh SpaceE    # 测试 Curriculum Learning 模式
# ========================================

SPACE_MODE=${1:-SpaceA}  # 默认 SpaceA
STUDY_NAME="test_hpo_${SPACE_MODE}"

echo "========================================"
echo "测试 RL Teacher 超参数优化系统"
echo "========================================"
echo ""
echo "注意：这是一个快速测试，使用极少的训练步数"
echo "仅用于验证系统功能，不用于实际优化"
echo ""
echo "Space模式: $SPACE_MODE"
echo "Study名称: $STUDY_NAME"
echo ""
echo "开始测试..."
echo ""

# 清理之前的测试数据
rm -f "optuna/${STUDY_NAME}.db"
rm -f "optuna/best_params_${STUDY_NAME}.txt"

# 构建命令参数
CMD_ARGS=(
    --gpu 0
    --n_trials 4
    --max_steps 1000000
    --storage "sqlite:///optuna/${STUDY_NAME}.db"
    --study_name "$STUDY_NAME"
    --space_mode "$SPACE_MODE"
)

# SpaceE 特有参数
if [ "$SPACE_MODE" = "SpaceE" ]; then
    CMD_ARGS+=(
        --alpha_start 0.1
        --alpha_end 1.0
        --curriculum_steps 1000000
    )
fi

# 运行4次快速试验，每次只训练1M步
python optuna/tune_teacher.py "${CMD_ARGS[@]}"

echo ""
echo "========================================"
echo "测试完成！"
echo "========================================"
echo ""

# 检查是否生成了必要的文件
if [ -f "optuna/${STUDY_NAME}.db" ]; then
    echo "✅ 数据库文件已生成"
else
    echo "❌ 数据库文件未生成"
fi

if [ -f "optuna/best_params_${STUDY_NAME}.txt" ]; then
    echo "✅ 最佳参数文件已生成"
    echo ""
    echo "最佳参数内容："
    cat "optuna/best_params_${STUDY_NAME}.txt"
else
    echo "❌ 最佳参数文件未生成"
fi

echo ""
echo "如果看到上述✅标记，说明HPO系统工作正常！"
echo ""
echo "现在可以运行真正的优化："
echo "  bash optuna/run_hpo.sh 0 50 250000000 SpaceA   # Baseline"
echo "  bash optuna/run_hpo.sh 0 50 250000000 SpaceE   # Curriculum"
echo ""
