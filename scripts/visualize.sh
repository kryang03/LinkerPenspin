#!/bin/bash
# chmod -R u+x ../../Code

# ============================================================
# 用法:
#   scripts/visualize.sh <checkpoint_dir> [alpha] [extra_args...]
#
# 参数:
#   checkpoint_dir: 模型检查点目录 (必需)
#   alpha:          时间缩放因子 (可选, 默认 1.0)
#                   - 0.3: 慢 3.3 倍，重力约 9%
#                   - 0.5: 慢 2 倍，重力约 25%
#                   - 1.0: 真实世界速度
#   extra_args:     额外的 Hydra 参数 (可选)
#
# 示例:
# 使用“空命令” (:) 配合 Here Document 来实现类似的效果。这告诉 Shell：“把这段内容当作输入传给空命令（即什么都不做）”
# BLOCK 可以替换成任何你喜欢的结束标识符（如 EOF, COMMENT 等），但必须保证开始和结束的标识符一致，且结束标识符必须顶格写
: <<'BLOCK'
scripts/visualize.sh outputs/TP_V3/optuna_trial_0000/teacher_nn 0.4
scripts/visualize.sh outputs/TP_V3/optuna_trial_0001/teacher_nn 1.0
============================================================
BLOCK
# CHECKLIST
# 1. 命令的最后一个参数指向output文件夹的名称，三维力信息是否重定向到正确的文件夹
# 2. 检查checkpoint的名称
# 3. relative_z_drop_threshold 和 pencil_tilt_threshold 早期终止阈值
# 4. grasp_cache_name对应canonical pose的cache名称
# 5. linker_hand_hora.py 中的CHECKLIST
# 6. episodeLength 训练时为400
# 7. 动作空间: 如果训练时禁用了无名指小拇指，可视化时也需要禁用
#    添加: task.env.actionSpace.disableRingLittleFinger=True
# 8. Flying base 配置需与训练时一致
# 9. Alpha 参数: 用于测试低 alpha 下的物理环境
#
# 重要说明：pose3_50k_cfg2 模型训练时使用47维特权信息
# 当前默认配置只有25维，需要显式启用以下选项来匹配：
# task.env.privInfo.enable_obj_orientation=True task.env.privInfo.enable_obj_angvel=True task.env.privInfo.enable_ft_pos=True \
#
# 动作空间说明：
# - 默认使用21维动作空间（所有手指）
# - 如需禁用无名指和小拇指，添加: task.env.actionSpace.disableRingLittleFinger=True
# - 使用13维动作空间时，策略只控制中指、食指、拇指

# 第一个参数: checkpoint 目录 (必需)
CACHE=$1
if [ -z "$CACHE" ]; then
    echo "Error: checkpoint directory is required"
    echo "Usage: $0 <checkpoint_dir> [alpha] [extra_args...]"
    exit 1
fi

# 第二个参数: alpha 值 (可选, 默认 1.0)
ALPHA=${2:-1.0}

# 验证 alpha 是否为有效的浮点数
if ! [[ "$ALPHA" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    echo "Warning: '$ALPHA' is not a valid number, treating as extra argument"
    ALPHA=1.0
    # 如果第二个参数不是数字，则作为 extra_args 的一部分
    array=( $@ )
    len=${#array[@]}
    EXTRA_ARGS=${array[@]:1:$len}
else
    # 第三个及之后的参数作为 extra_args
    array=( $@ )
    len=${#array[@]}
    EXTRA_ARGS=${array[@]:2:$len}
fi
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

# 根据 alpha 值决定 curriculum 模式
if [ "$ALPHA" == "1.0" ] || [ "$ALPHA" == "1" ]; then
    # alpha=1.0 时使用 SpaceA (无缩放)
    CURRICULUM_MODE="SpaceA"
    echo "[Visualize] Using SpaceA mode (no physics scaling, alpha=1.0)"
else
    # 其他 alpha 值使用 SpaceE (完整缩放)
    CURRICULUM_MODE="SpaceE"
    echo "[Visualize] Using SpaceE mode with alpha=$ALPHA"
    echo "  - Gravity scale: $(echo "scale=3; $ALPHA * $ALPHA" | bc)"
    echo "  - Time feels $(echo "scale=1; 1 / $ALPHA" | bc)x slower"
fi

python train.py task=LinkerHandHora headless=False \
train.algo=PPOTeacher \
task.env.numEnvs=1 test=True checkpoint="${CACHE}"/best*.pth \
task.env.episodeLength=4000 \
"task.env.grasp_cache_name='3_30000_49_nofly'" \
task.env.initPoseMode=low \
task.env.relative_z_drop_threshold=0.05 \
task.env.pencil_tilt_threshold=0.06 \
task.env.actionSpace.disableRingLittleFinger=True \
task.env.flyingHand.linearVelocity=0.1 \
task.env.flyingHand.angularVelocity=2.0 \
task.env.privInfo.enable_obj_orientation=True task.env.privInfo.enable_obj_angvel=True task.env.privInfo.enable_ft_pos=True \
task.env.curriculum.mode=${CURRICULUM_MODE} \
task.env.curriculum.alpha_start=${ALPHA} \
task.env.curriculum.alpha_end=${ALPHA} \
${EXTRA_ARGS}
