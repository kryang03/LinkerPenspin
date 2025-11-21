#!/bin/bash
# ========================================
# 快速验证HPO系统是否正常工作
# ========================================

echo "========================================"
echo "验证 HPO 系统配置"
echo "========================================"
echo ""

# 1. 检查configs/__init__.py
echo "1. 检查 configs/__init__.py..."
if [ -f "configs/__init__.py" ]; then
    echo "   ✅ configs/__init__.py 存在"
else
    echo "   ❌ configs/__init__.py 不存在，正在创建..."
    touch configs/__init__.py
    echo "   ✅ 已创建 configs/__init__.py"
fi
echo ""

# 2. 检查optuna目录
echo "2. 检查 optuna 目录..."
if [ -d "optuna" ]; then
    echo "   ✅ optuna 目录存在"
else
    echo "   ❌ optuna 目录不存在，正在创建..."
    mkdir -p optuna
    echo "   ✅ 已创建 optuna 目录"
fi
echo ""

# 3. 检查关键文件
echo "3. 检查关键文件..."
files=(
    "train.py"
    "optuna/tune_teacher.py"
    "optuna/run_hpo.sh"
    "configs/config.yaml"
    "configs/task/LinkerHandHora.yaml"
    "configs/train/LinkerHandHora.yaml"
)

all_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file 不存在！"
        all_exist=false
    fi
done
echo ""

# 4. 检查Python依赖
echo "4. 检查 Python 依赖..."
python -c "import optuna; print('   ✅ optuna:', optuna.__version__)" 2>/dev/null || echo "   ❌ optuna 未安装"
python -c "import hydra; print('   ✅ hydra-core 已安装')" 2>/dev/null || echo "   ❌ hydra-core 未安装"
python -c "import torch; print('   ✅ torch:', torch.__version__)" 2>/dev/null || echo "   ❌ torch 未安装"
echo ""

# 5. 测试Hydra配置（可选，需要时间）
if [ "$1" == "--full" ]; then
    echo "5. 测试 Hydra 配置（完整测试）..."
    python test_hydra_config.py
    echo ""
fi

# 总结
echo "========================================"
if [ "$all_exist" = true ]; then
    echo "✅ 基础验证通过！"
    echo ""
    echo "下一步："
    echo "  1. 清理失败的试验（如果有）："
    echo "     python optuna/clean_failed_trials.py --dry_run"
    echo ""
    echo "  2. 运行快速测试（约10分钟）："
    echo "     bash optuna/test_hpo.sh"
    echo ""
    echo "  3. 开始正式优化："
    echo "     bash optuna/run_hpo.sh 0 50"
else
    echo "❌ 验证失败，请检查缺失的文件"
fi
echo "========================================"
