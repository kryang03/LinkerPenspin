#!/bin/bash
# chmod -R u+x ../../Code

# scripts/visualize.sh outputs/LinkerHandHora/debug_teacher_20251130_220235_copy/teacher_nn

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

# test=True 是用来load weight进行测试的
# 
# 重要说明：pose3_50k_cfg2 模型训练时使用47维特权信息
# 当前默认配置只有25维，需要显式启用以下选项来匹配：
# task.env.privInfo.enable_obj_orientation=True task.env.privInfo.enable_obj_angvel=True task.env.privInfo.enable_ft_pos=True \
#
# 动作空间说明：
# - 默认使用21维动作空间（所有手指）
# - 如需禁用无名指和小拇指，添加: task.env.actionSpace.disableRingLittleFinger=True
# - 使用13维动作空间时，策略只控制中指、食指、拇指

CACHE=$1

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}
python train.py task=LinkerHandHora headless=False \
train.algo=PPOTeacher \
task.env.numEnvs=1 test=True checkpoint="${CACHE}"/best*.pth \
task.env.episodeLength=4000 \
"task.env.grasp_cache_name='3_30000_61'" \
task.env.initPoseMode=low \
task.env.relative_z_drop_threshold=0.12 \
task.env.pencil_tilt_threshold=0.12 \
task.env.actionSpace.disableRingLittleFinger=True \
task.env.flyingHand.linearVelocity=0.1 \
task.env.flyingHand.angularVelocity=2.0 \
task.env.privInfo.enable_obj_orientation=True task.env.privInfo.enable_obj_angvel=True task.env.privInfo.enable_ft_pos=True \
${EXTRA_ARGS}
