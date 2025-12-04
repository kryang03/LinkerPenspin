#!/bin/bash
# ========================================
# 功能: 纯PPO强化学习训练Teacher模型
# 使用全部信息（包括特权信息）在仿真中训练
# ========================================
# 使用方法:
# scripts/TP_train_rl_teacher.sh <GPU_ID> <SEED> <OUTPUT_NAME> [EXTRA_ARGS...]
# 例如: scripts/TP_train_rl_teacher.sh 0 42 debug_teacher
# ========================================

set -e  # 遇到错误立即退出

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                         可调参数配置区（常用）                              ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# --- 奖励权重 ---
REWARD_ROTATE=1.0                      # 旋转奖励（主奖励）
REWARD_OBJ_LINVEL_PENALTY=-0.3         # 物体线速度惩罚
REWARD_TORQUE_PENALTY=-0.01            # 力矩惩罚
REWARD_ROTATE_PENALTY=0.0              # 旋转惩罚（逆向/超速）
REWARD_AXIAL_TILT_PENALTY=-1.5         # 轴向倾斜惩罚
REWARD_POSITION_PENALTY=-0.1           # 位置惩罚
REWARD_FLYING_BASE_PENALTY=-0.1        # Flying base 移动惩罚

# --- Waypoint 跟踪奖励 (Triangle Pass) ---
# 建议值: 0.1~0.5（作为辅助奖励，不主导训练）
REWARD_WAYPOINT_TRACKING=0.2           # Waypoint 跟踪奖励权重
WAYPOINT_SIGMA=0.05                    # 高斯核带宽（越小要求越精确）
WAYPOINT_HALF_PERIOD_SYMMETRIC=True    # 笔具有180°对称性

# --- 轴向倾斜阈值 ---
AXIAL_TILT_THRESHOLD=0.03              # 轴向倾斜阈值（米），低于此值不惩罚

# --- 早期终止阈值 ---
RELATIVE_Z_DROP_THRESHOLD=${RELATIVE_Z_DROP_THRESHOLD:-0.04}  # 物体下降阈值（米）
PENCIL_TILT_THRESHOLD=${PENCIL_TILT_THRESHOLD:-0.04}          # 铅笔倾倒阈值（米）

# --- 角速度参数 ---
ANGVEL_CLIP_MIN=-0.5                   # 角速度裁剪下限
ANGVEL_CLIP_MAX=0.5                    # 角速度裁剪上限
ANGVEL_PENALTY_THRES_HIGH=1.0          # 角速度惩罚阈值（上限）
ANGVEL_PENALTY_THRES_LOW=-0.5          # 角速度惩罚阈值（下限）

# --- 动作空间配置 ---
DISABLE_RING_LITTLE=True               # 禁用无名指和小拇指 (21 -> 13 DoF)

# --- Flying base 配置 ---
FLYING_LINEAR_VELOCITY=0.1             # 线速度上限 (m/s)
FLYING_ANGULAR_VELOCITY=2.0            # 角速度上限 (rad/s)

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                              脚本主体                                       ║
# ╚════════════════════════════════════════════════════════════════════════════╝

GPUS=${1:-0}        # GPU ID
SEED=${2:-42}  # 随机种子 (默认42)
OUTPUT_NAME=${3:-debug_teacher} # 输出目录名称

# 获取额外参数
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

# ========================================
# 创建唯一输出目录（带时间戳）
# ========================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
UNIQUE_OUTPUT_NAME="${OUTPUT_NAME}_${TIMESTAMP}"
OUTPUT_DIR="outputs/LinkerHandHora/${UNIQUE_OUTPUT_NAME}"
LOG_FILE="${OUTPUT_DIR}/training.log"
CONFIG_FILE="${OUTPUT_DIR}/run_config.txt"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "RL Teacher 训练启动"
echo "=========================================="
echo "GPU ID:          ${GPUS}"
echo "随机种子:        ${SEED}"
echo "输出目录:        ${OUTPUT_DIR}"
echo "日志文件:        ${LOG_FILE}"
echo "=========================================="

# ========================================
# 保存运行配置
# ========================================
cat > ${CONFIG_FILE} << EOF
========================================
RL Teacher 训练配置
========================================
启动时间:        $(date +"%Y-%m-%d %H:%M:%S")
GPU ID:          ${GPUS}
随机种子:        ${SEED}
输出目录:        ${OUTPUT_DIR}
输出名称:        ${UNIQUE_OUTPUT_NAME}
额外参数:        ${EXTRA_ARGS}

========================================
训练参数配置
========================================
算法:            PPOTeacher
任务:            LinkerHandHora
Grasp Cache:     3pose
初始化模式:      low
重置高度阈值:    0.12
最大训练步数:    10000000000

角速度参数:
  angvelClipMin:           ${ANGVEL_CLIP_MIN}
  angvelClipMax:           ${ANGVEL_CLIP_MAX}
  angvelPenaltyThresHigh:  ${ANGVEL_PENALTY_THRES_HIGH}
  angvelPenaltyThresLow:   ${ANGVEL_PENALTY_THRES_LOW}

奖励权重（合并后的最终值）:
  rotate_reward:          ${REWARD_ROTATE}
  obj_linvel_penalty:     ${REWARD_OBJ_LINVEL_PENALTY}
  torque_penalty:         ${REWARD_TORQUE_PENALTY}
  rotate_penalty:         ${REWARD_ROTATE_PENALTY}
  axial_tilt_penalty:     ${REWARD_AXIAL_TILT_PENALTY}
  axial_tilt_threshold:   ${AXIAL_TILT_THRESHOLD}
  position_penalty:       ${REWARD_POSITION_PENALTY}
  waypoint_tracking:      ${REWARD_WAYPOINT_TRACKING}
  waypoint_sigma:         ${WAYPOINT_SIGMA}
  waypoint_symmetric:     ${WAYPOINT_HALF_PERIOD_SYMMETRIC}
  flying_base_penalty:    ${REWARD_FLYING_BASE_PENALTY}

Flying base 配置:
  linearVelocity:       ${FLYING_LINEAR_VELOCITY}
  angularVelocity:      ${FLYING_ANGULAR_VELOCITY}

========================================
EOF

# ========================================
# 保存启动命令
# ========================================
{
    echo "=========================================="
    echo "启动命令"
    echo "=========================================="
    echo "完整命令:"
    echo "CUDA_VISIBLE_DEVICES=${GPUS} \\"
    echo "python train.py task=LinkerHandHora headless=True seed=${SEED} \\"
    echo "  train.ppo.output_name=LinkerHandHora/${UNIQUE_OUTPUT_NAME} \\"
    echo "  train.algo=PPOTeacher \\"
    echo "  task.env.grasp_cache_name=3_30000_61 \\"
    echo "  train.ppo.max_agent_steps=10000000000 \\"
    echo "  task.env.initPoseMode=low \\"
    echo "  task.env.relative_z_drop_threshold=${RELATIVE_Z_DROP_THRESHOLD} \\"
    echo "  task.env.pencil_tilt_threshold=${PENCIL_TILT_THRESHOLD} \\"
    echo "  ${EXTRA_ARGS}"
    echo "=========================================="
    echo ""
} | tee ${LOG_FILE}

# ========================================
# 启动 TensorBoard（后台运行）
# ========================================
TB_PORT=6006
TB_LOGDIR="${OUTPUT_DIR}/teacher_tb"

# 检查端口是否已被占用
if command -v lsof >/dev/null 2>&1 && lsof -Pi :${TB_PORT} -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    # 获取占用端口的进程 PID 列表
    PIDS=$(lsof -t -i :${TB_PORT} -sTCP:LISTEN)
    echo "警告: TensorBoard 端口 ${TB_PORT} 已被占用 (PIDs: ${PIDS})"
    echo "将尝试停止占用端口的进程，以便本次 TensorBoard 启动成功..."

    # 优雅终止占用端口的进程（SIGTERM），等待短暂时间，若未退出则强制结束（SIGKILL）
    for PID in ${PIDS}; do
        if [ -z "${PID}" ]; then
            continue
        fi
        # 如果有保存的旧 PID 文件且匹配，输出提示
        if [ -f "${OUTPUT_DIR}/tensorboard.pid" ]; then
            OLD_PID=$(cat "${OUTPUT_DIR}/tensorboard.pid" 2>/dev/null || true)
            if [ "${OLD_PID}" = "${PID}" ]; then
                echo "发现旧的 TensorBoard PID 文件 (${OLD_PID}) 与监听的 PID ${PID} 匹配，优先尝试优雅终止..."
            fi
        fi

        if kill ${PID} >/dev/null 2>&1; then
            echo "已发送 SIGTERM 到 PID ${PID}，等待退出..."
        else
            echo "无法发送 SIGTERM 给 PID ${PID}（可能没有权限或进程已退出），将继续尝试强制终止..."
        fi
    done

    # 等待最多 5 秒观察进程是否退出
    timeout=5
    while [ ${timeout} -gt 0 ] && lsof -Pi :${TB_PORT} -sTCP:LISTEN -t >/dev/null 2>&1; do
        sleep 1
        timeout=$((timeout-1))
    done

    # 如果进程仍然存在，则发送 SIGKILL
    if lsof -Pi :${TB_PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
        PIDS_LEFT=$(lsof -t -i :${TB_PORT} -sTCP:LISTEN)
        echo "端口 ${TB_PORT} 仍被占用 (PIDs: ${PIDS_LEFT})，将发送 SIGKILL 强制终止..."
        for PID in ${PIDS_LEFT}; do
            if kill -9 ${PID} >/dev/null 2>&1; then
                echo "已强制终止 PID ${PID}"
            else
                echo "无法强制终止 PID ${PID}，请手动处理后重试（可能需要 sudo）。"
            fi
        done
    fi

    # 给系统一点时间释放端口
    sleep 1
else
    # 检查 tensorboard 是否已安装
    if python -c "import tensorboard" 2>/dev/null; then
        echo "启动 TensorBoard 监控..."
        echo "TensorBoard 端口: ${TB_PORT}"
        echo "TensorBoard 日志目录: ${TB_LOGDIR}"
        
        # 在后台启动 TensorBoard (使用 python -m 避免路径冲突)
        nohup python -m tensorboard.main --logdir=${TB_LOGDIR} --port=${TB_PORT} --bind_all > ${OUTPUT_DIR}/tensorboard.log 2>&1 &
        TB_PID=$!
        echo "TensorBoard PID: ${TB_PID}"
        echo "访问地址: http://localhost:${TB_PORT}"
        echo ""
        
        # 保存 TensorBoard PID
        echo ${TB_PID} > ${OUTPUT_DIR}/tensorboard.pid
    else
        echo "警告: TensorBoard 未安装，跳过启动"
        echo "安装命令: pip install tensorboard"
        echo "如需查看 TensorBoard，请手动运行:"
        echo "  python -m tensorboard.main --logdir=${TB_LOGDIR} --port=${TB_PORT}"
    fi
fi

# ========================================
# 构建训练参数数组
# ========================================
ARGS=(
    "task=LinkerHandHora"
    "headless=True"
    "seed=${SEED}"
    "train.ppo.output_name=LinkerHandHora/${UNIQUE_OUTPUT_NAME}"
    "train.algo=PPOTeacher"
    "task.env.grasp_cache_name='3_30000_61'"
    "train.ppo.max_agent_steps=500000000"
    "task.env.initPoseMode=low"
    # 早期终止阈值（新命名）
    "task.env.relative_z_drop_threshold=${RELATIVE_Z_DROP_THRESHOLD}"
    "task.env.pencil_tilt_threshold=${PENCIL_TILT_THRESHOLD}"
    # 动作空间配置
    "task.env.actionSpace.disableRingLittleFinger=${DISABLE_RING_LITTLE}"
    # Flying base 配置
    "task.env.flyingHand.linearVelocity=${FLYING_LINEAR_VELOCITY}"
    "task.env.flyingHand.angularVelocity=${FLYING_ANGULAR_VELOCITY}"
    # 角速度参数
    "task.env.reward.angvelClipMin=${ANGVEL_CLIP_MIN}"
    "task.env.reward.angvelClipMax=${ANGVEL_CLIP_MAX}"
    "task.env.reward.angvelPenaltyThresHigh=${ANGVEL_PENALTY_THRES_HIGH}"
    "task.env.reward.angvelPenaltyThresLow=${ANGVEL_PENALTY_THRES_LOW}"
    # 奖励权重参数
    "task.env.reward.rotate_reward_scale=${REWARD_ROTATE}"
    "task.env.reward.obj_linvel_penalty_scale=${REWARD_OBJ_LINVEL_PENALTY}"
    "task.env.reward.torque_penalty_scale=${REWARD_TORQUE_PENALTY}"
    "task.env.reward.rotate_penalty_scale=${REWARD_ROTATE_PENALTY}"
    "task.env.reward.axial_tilt_penalty_scale=${REWARD_AXIAL_TILT_PENALTY}"
    "task.env.reward.axial_tilt_threshold=${AXIAL_TILT_THRESHOLD}"
    "task.env.reward.position_penalty_scale=${REWARD_POSITION_PENALTY}"
    # Waypoint 跟踪奖励 (Triangle Pass)
    "task.env.reward.waypoint_tracking_reward_scale=${REWARD_WAYPOINT_TRACKING}"
    "task.env.reward.waypoint_sigma=${WAYPOINT_SIGMA}"
    "task.env.reward.waypoint_half_period_symmetric=${WAYPOINT_HALF_PERIOD_SYMMETRIC}"
    "task.env.reward.flying_base_movement_penalty_scale=${REWARD_FLYING_BASE_PENALTY}"
)

# 添加额外参数
if [ -n "${EXTRA_ARGS}" ]; then
    ARGS+=("${EXTRA_ARGS}")
fi

# ========================================
# 执行训练 (后台运行)
# ========================================
echo "启动训练进程..."
echo "日志将写入文件: ${LOG_FILE}"

# 使用 nohup 后台运行
nohup env CUDA_VISIBLE_DEVICES=${GPUS} python -u train.py "${ARGS[@]}" >> ${LOG_FILE} 2>&1 &
PID=$!

# 保存 PID 到文件
echo ${PID} > ${OUTPUT_DIR}/train.pid

echo "训练已在后台启动，PID: ${PID}"
echo "查看实时日志: tail -f ${LOG_FILE}"
echo "访问地址: http://localhost:${TB_PORT}"
echo "=========================================="
echo ""

# ========================================
# 提示
# ========================================
echo "提示:"
echo "1. 查看实时日志: tail -f ${LOG_FILE}"
echo "2. 查看 TensorBoard: http://localhost:${TB_PORT}"
echo "3. 停止训练: kill ${PID}"
echo "4. 停止 TensorBoard: kill \$(cat ${OUTPUT_DIR}/tensorboard.pid)"
echo ""

exit 0
