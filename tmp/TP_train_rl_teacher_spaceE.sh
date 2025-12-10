#!/bin/bash
# ========================================
# 功能: Space E Curriculum Learning - 时空扭曲课程学习
# Space E: 完整动力学同构缩放（推荐）
#   - 物理层: g'=α²g, Kp'=α²Kp, Kd'=αKd, τ'=α²τ
#   - 观测层: v_obs=v_sim/α, F_obs=F_sim/α²
#   - 奖励层: r_final=r_raw×α
# ========================================
# 使用方法:
# scripts/TP_train_rl_teacher_spaceE.sh <GPU_ID> <SEED> <OUTPUT_NAME> [EXTRA_ARGS...]
# 例如: scripts/TP_train_rl_teacher_spaceE.sh 0 42 spaceE_curriculum_waypointfixed_0.25_1.5-1.15
# ========================================

set -e  # 遇到错误立即退出

# ╭════════════════════════════════════════════════════════════════════════════╮
# ║                            可调参数配置区（常用）                              ║
# ╰════════════════════════════════════════════════════════════════════════════╯

# --- Space E Curriculum 配置 ---
# 注: SpaceA 是 SpaceE 在 alpha_start=1.0 时的特例
CURRICULUM_MODE='SpaceE'               # 统一使用 SpaceE 模式
CURRICULUM_ALPHA_START=0.25            # 初始 α 值 (0.25 = 世界慢 4 倍)
CURRICULUM_ALPHA_END=1.0               # 最终 α 值 (1.0 = 真实世界速度)
# 注: curriculum_steps 已移除，Agent 自动决定进度
CURRICULUM_RATIO_THRESHOLD=0.05        # 物理更新触发阈值 (5% 相对变化)
SUCCESS_ROT_THRESHOLD=10.0             # 成功旋转角度阈值 (rad), 约 1.6 圈
USE_ADAPTIVE_THRESHOLD=True            # 是否使用自适应阈值 (threshold * alpha)

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
REWARD_WAYPOINT_TRACKING=3.0           # Waypoint 跟踪奖励权重
WAYPOINT_SIGMA=0.05                    # 高斯核带宽（越小要求越精确）
WAYPOINT_HALF_PERIOD_SYMMETRIC=True    # 笔具有180°对称性

# --- 轴向倾斜阈值 ---
AXIAL_TILT_THRESHOLD=0.03              # 轴向倾斜阈值（米），低于此值不惩罚

# --- 早期终止阈值 ---
RELATIVE_Z_DROP_THRESHOLD=${RELATIVE_Z_DROP_THRESHOLD:-0.04}  # 物体下降阈值（米）
PENCIL_TILT_THRESHOLD=${PENCIL_TILT_THRESHOLD:-0.04}          # 铅笔倾倒阈值（米）

# --- 视频录制配置 ---
# 启用相机传感器（用于录制 GIF/视频到 TensorBoard）
# 注意：headless 模式下可能不工作，需要使用 headless=False 或配置虚拟显示
ENABLE_CAMERA_SENSORS=${ENABLE_CAMERA_SENSORS:-False}

# --- 角速度参数 ---
# 两种模式:
# 1. Clip-based (默认): 使用 angvelClipMin/Max 和 Penalty 阈值
# 2. Gaussian-kernel: 使用 target_angvel 和 angvel_sigma
USE_GAUSSIAN_ANGVEL_REWARD=True        # 启用高斯核角速度奖励
TARGET_ANGVEL=3.14                     # 目标角速度 (rad/s), π rad/s ≈ 0.5 Hz
ANGVEL_REWARD_SIGMA=1.0                # 高斯核带宽 σ, 越小奖励越陡峭

# Clip-based 参数 (USE_GAUSSIAN_ANGVEL_REWARD=False 时使用)
ANGVEL_CLIP_MIN=-0.5                   # 角速度裁剪下限
ANGVEL_CLIP_MAX=0.5                    # 角速度裁剪上限
ANGVEL_PENALTY_THRES_HIGH=1.0          # 角速度惩罚阈值（上限）
ANGVEL_PENALTY_THRES_LOW=-0.5          # 角速度惩罚阈值（下限）

# --- [Anti-Hacking] EMA 平滑和惩罚参数 ---
EMA_ALPHA=0.15                         # EMA 平滑系数 (0.1~0.2, 越小越平滑，仅用于速度门控)
JITTER_PENALTY_SCALE=-0.5              # Jitter 惩罚权重 (负值)
REVERSE_PENALTY_SCALE=-1.0             # 反向旋转惩罚权重 (负值)
LOG_SCALE_BETA=1.0                     # 对数缩放灵敏度控制参数

# --- Controller 参数 (PD 控制器) ---
# 核心痛点：低 Alpha 下手太软，需要调整这些参数
CONTROLLER_PGAIN=15.0                  # P-Gain (刚度)
CONTROLLER_DGAIN=0.35                  # D-Gain (阻尼)
CONTROLLER_TORQUE_LIMIT=4.0            # 力矩限制
CONTROLLER_ACTION_SCALE=0.1            # 动作缩放

# --- 动作空间配置 ---
DISABLE_RING_LITTLE=True               # 禁用无名指和小拇指 (21 -> 13 DoF)

# --- Flying Hand 配置 (默认关闭) ---
FLYING_HAND_ENABLED=False              # 默认禁用 Flying Hand (使用固定底座)
FLYING_LINEAR_VELOCITY=0.1             # 线速度限制 (m/s) - 仅当 Flying Hand 启用时生效
FLYING_ANGULAR_VELOCITY=2.0            # 角速度限制 (rad/s) - 仅当 Flying Hand 启用时生效

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
Grasp Cache:     3_30000_49_nofly
初始化模式:      low
最大训练步数:    500000000

早期终止阈值:
  relative_z_drop_threshold:  ${RELATIVE_Z_DROP_THRESHOLD}
  pencil_tilt_threshold:      ${PENCIL_TILT_THRESHOLD}

Space E Curriculum 配置 (SpaceA 是 alpha_start=1.0 的特例):
  mode:                     ${CURRICULUM_MODE}
  alpha_start:              ${CURRICULUM_ALPHA_START}
  alpha_end:                ${CURRICULUM_ALPHA_END}
  ratio_threshold:          ${CURRICULUM_RATIO_THRESHOLD}
  success_rot_threshold:    ${SUCCESS_ROT_THRESHOLD}
  use_adaptive_threshold:   ${USE_ADAPTIVE_THRESHOLD}

Controller 参数 (PD 控制器):
  pgain:                      ${CONTROLLER_PGAIN}
  dgain:                      ${CONTROLLER_DGAIN}
  torque_limit:               ${CONTROLLER_TORQUE_LIMIT}
  action_scale:               ${CONTROLLER_ACTION_SCALE}

角速度参数:
  use_gaussian_angvel_reward: ${USE_GAUSSIAN_ANGVEL_REWARD}
  target_angvel:              ${TARGET_ANGVEL}
  angvel_sigma:               ${ANGVEL_REWARD_SIGMA}

Anti-Hacking 参数:
  ema_alpha:                  ${EMA_ALPHA}
  jitter_penalty_scale:       ${JITTER_PENALTY_SCALE}
  reverse_penalty_scale:      ${REVERSE_PENALTY_SCALE}

奖励权重（合并后的最终值）:
  [奖励 - 正数 scale]
  rotate_reward:          ${REWARD_ROTATE}
  waypoint_tracking:      ${REWARD_WAYPOINT_TRACKING}
    waypoint_sigma:         ${WAYPOINT_SIGMA}
    waypoint_symmetric:     ${WAYPOINT_HALF_PERIOD_SYMMETRIC}
  
  [惩罚 - 负数 scale]
  obj_linvel_penalty:     ${REWARD_OBJ_LINVEL_PENALTY}
  torque_penalty:         ${REWARD_TORQUE_PENALTY}
  rotate_penalty:         ${REWARD_ROTATE_PENALTY}
  axial_tilt_penalty:     ${REWARD_AXIAL_TILT_PENALTY}
    axial_tilt_threshold:   ${AXIAL_TILT_THRESHOLD}
  position_penalty:       ${REWARD_POSITION_PENALTY}
  flying_base_penalty:    ${REWARD_FLYING_BASE_PENALTY} (ignored when Flying Hand disabled)
  jitter_penalty:         ${JITTER_PENALTY_SCALE}
  reverse_penalty:        ${REVERSE_PENALTY_SCALE}

Flying Hand: disabled (fixed base, 21 DoF)

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
    echo "  task.env.grasp_cache_name=3_30000_49_nofly \\"
    echo "  task.env.flyingHand.enabled=False \\"
    echo "  task.env.asset.handAsset='assets/linker_hand/L25_dof_urdf.urdf' \\"
    echo "  task.env.numActions=21 \\"
    echo "  task.env.numObservations=126 \\"
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
    "task.env.grasp_cache_name='3_30000_49_nofly'"
    "task.env.initPoseMode=low"
    "train.ppo.max_agent_steps=500000000"
    # 早期终止阈值
    "task.env.relative_z_drop_threshold=${RELATIVE_Z_DROP_THRESHOLD}"
    "task.env.pencil_tilt_threshold=${PENCIL_TILT_THRESHOLD}"
    # Controller 参数 (PD 控制器)
    "task.env.controller.pgain=${CONTROLLER_PGAIN}"
    "task.env.controller.dgain=${CONTROLLER_DGAIN}"
    "task.env.controller.torque_limit=${CONTROLLER_TORQUE_LIMIT}"
    "task.env.controller.action_scale=${CONTROLLER_ACTION_SCALE}"
    # 动作空间配置
    "task.env.actionSpace.disableRingLittleFinger=${DISABLE_RING_LITTLE}"
    # Flying Hand 配置 (默认关闭)
    "task.env.flyingHand.enabled=${FLYING_HAND_ENABLED}"
    "task.env.flyingHand.linearVelocity=${FLYING_LINEAR_VELOCITY}"
    "task.env.flyingHand.angularVelocity=${FLYING_ANGULAR_VELOCITY}"
    # 角速度参数 (Gaussian kernel)
    "task.env.reward.use_gaussian_angvel_reward=${USE_GAUSSIAN_ANGVEL_REWARD}"
    "task.env.reward.target_angvel=${TARGET_ANGVEL}"
    "task.env.reward.angvel_sigma=${ANGVEL_REWARD_SIGMA}"
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
    "task.env.reward.log_scale_beta=${LOG_SCALE_BETA}"
    # Space E Curriculum 配置
    "task.env.curriculum.mode=${CURRICULUM_MODE}"
    "task.env.curriculum.alpha_start=${CURRICULUM_ALPHA_START}"
    "task.env.curriculum.alpha_end=${CURRICULUM_ALPHA_END}"
    "task.env.curriculum.ratio_threshold=${CURRICULUM_RATIO_THRESHOLD}"
    "task.env.curriculum.success_rot_threshold=${SUCCESS_ROT_THRESHOLD}"
    "task.env.curriculum.use_adaptive_threshold=${USE_ADAPTIVE_THRESHOLD}"
    # [Anti-Hacking] EMA 平滑和惩罚参数
    "task.env.reward.ema_alpha=${EMA_ALPHA}"
    "task.env.reward.jitter_penalty_scale=${JITTER_PENALTY_SCALE}"
    "task.env.reward.reverse_penalty_scale=${REVERSE_PENALTY_SCALE}"
    # 视频录制配置
    "task.env.enableCameraSensors=${ENABLE_CAMERA_SENSORS}"
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
