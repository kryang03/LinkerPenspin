#!/bin/bash
# ========================================
# 测试超参数优化功能
# ========================================
# 这个脚本会运行一个快速测试来验证HPO系统是否正常工作
# 使用极少的训练步数（1M步）来快速验证
# ========================================

echo "========================================"
echo "测试 RL Teacher 超参数优化系统"
echo "========================================"
echo ""
echo "注意：这是一个快速测试，使用极少的训练步数"
echo "仅用于验证系统功能，不用于实际优化"
echo ""
echo "开始测试..."
echo ""

# 清理之前的测试数据
rm -f optuna/test_hpo.db
rm -f optuna/best_params_test_hpo.txt

# 运行10次快速试验，每次只训练1M步
python optuna/tune_teacher.py \
    --gpu 0 \
    --n_trials 10 \
    --max_steps 1000000 \
    --storage "sqlite:///optuna/test_hpo.db" \
    --study_name "test_hpo"

echo ""
echo "========================================"
echo "测试完成！"
echo "========================================"
echo ""

# 检查是否生成了必要的文件
if [ -f "optuna/test_hpo.db" ]; then
    echo "✅ 数据库文件已生成"
else
    echo "❌ 数据库文件未生成"
fi

if [ -f "optuna/best_params_test_hpo.txt" ]; then
    echo "✅ 最佳参数文件已生成"
    echo ""
    echo "最佳参数内容："
    cat optuna/best_params_test_hpo.txt
else
    echo "❌ 最佳参数文件未生成"
fi

echo ""
echo "如果看到上述✅标记，说明HPO系统工作正常！"
echo ""
echo "现在可以运行真正的优化："
echo "  bash optuna/run_hpo.sh 0 50"
echo ""
