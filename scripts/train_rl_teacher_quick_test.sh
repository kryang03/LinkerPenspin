#!/bin/bash
# ========================================
# 功能: RL Teacher 快速可视化测试 (Quick Test)
# 作用: 验证环境物理、奖励函数逻辑、Reset机制
# ========================================
# 使用方法:
# scripts/train_rl_teacher_quick_test.sh <GPU_ID> <SEED> [EXTRA_ARGS...]
# 例如: scripts/train_rl_teacher_quick_test.sh 0 42
# ========================================

set -e

GPUS=${1:-0}        # GPU ID
SEED=${2:-42}  # 随机种子 (默认42)

# 获取额外参数
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}

# 创建临时输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_NAME="quick_test_${TIMESTAMP}"
OUTPUT_DIR="outputs/LinkerHandHora/${OUTPUT_NAME}"
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "启动 Quick Test (可视化模式)"
echo "=========================================="
echo "GPU ID:   ${GPUS}"
echo "环境数:   64 (为满足PPO Batch要求)"
echo "步数限制: 20,000"
echo "=========================================="

# ========================================
# 构建参数 (强制覆盖用于测试的参数)
# ========================================
ARGS=(
    "task=LinkerHandHora"
    "headless=False"                        # 强制开启可视化窗口
    "seed=${SEED}"
    "train.ppo.output_name=LinkerHandHora/${OUTPUT_NAME}"
    "train.algo=PPOTeacher"
    
    # === 关键测试参数 ===
    "task.env.numEnvs=64"                   # 设置为64以避免 minibatch size 不匹配报错
    "train.ppo.max_agent_steps=20000"       # 跑完约 1-2 个迭代即停止
    "train.ppo.minibatch_size=768"          # batch_size = 64 * 12 = 768，必须整除
    "task.env.grasp_cache_name='3_30000_61'"
    "task.env.initPoseMode=low"             # 初始姿态模式
    
    # === 阈值设置 ===
    "task.env.relative_z_drop_threshold=0.15"
    "task.env.pencil_tilt_threshold=0.08"
    
    # === 动作空间 & 奖励 ===
    "task.env.actionSpace.disableRingLittleFinger=True"
    "task.env.reward.rotate_reward_scale=1.0"
    "task.env.reward.obj_linvel_penalty_scale=-0.3"
    "task.env.reward.pencil_z_dist_penalty_scale=-1.5"
)

# 添加用户自定义参数
if [ -n "${EXTRA_ARGS}" ]; then
    ARGS+=("${EXTRA_ARGS}")
fi

# ========================================
# 执行训练 (前台阻塞模式)
# ========================================
# -u: 禁用输出缓存，实时显示打印信息
env CUDA_VISIBLE_DEVICES=${GPUS} python -u train.py "${ARGS[@]}"

echo "=========================================="
echo "测试结束。日志目录: ${OUTPUT_DIR}"
echo "=========================================="