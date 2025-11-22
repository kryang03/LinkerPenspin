#!/bin/bash
# chmod -R u+x ../../Code
# scripts/visualize.sh pose4_50k_cfg2 > contacts/pose4_50k_cfg2.txt
# scripts/visualize.sh pose6_50k_cfg4 > contacts/pose6_50k_cfg4.txt
# scripts/visualize.sh pose3_50k_cfg7_200_3 > contacts/pose3_50k_cfg7_200_3.txt
# scripts/visualize.sh pose3_50k_cfg9 > contacts/pose3_50k_cfg9.txt
# scripts/visualize.sh pose3_3k_cfg_c > contacts/pose3_3k_cfg_c.txt
# scripts/visualize.sh outputs/LinkerHandHora/optuna_trial_0011/teacher_nn > tmp_debug.txt
#best:
# scripts/visualize.sh outputs/LinkerHandHora/pose3_50k_cfg2/stage1_nn > contacts/pose3_50k_cfg2.txt
# CHECKLIST
# 1. 命令的最后一个参数指向output文件夹的名称，三维力信息是否重定向到正确的文件夹
# 2. 检查checkpoint的名称
# 3. reset_height_threshold 是否=0.14
# 4. grasp_cache_name对应canonical pose的cache名称
# 5. linker_hand_hora.py 中的CHECKLIST
# 6. episodeLength 训练时为400

# test=True 是用来load weight进行测试的
# 
# 重要说明：pose3_50k_cfg2 模型训练时使用47维特权信息
# 当前默认配置只有25维，需要显式启用以下选项来匹配：
# task.env.privInfo.enable_obj_orientation=True task.env.privInfo.enable_obj_angvel=True task.env.privInfo.enable_ft_pos=True \

CACHE=$1

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}
python train.py task=LinkerHandHora headless=False \
train.algo=PPOTeacher \
task.env.numEnvs=1 test=True checkpoint="${CACHE}"/best*.pth \
task.env.episodeLength=4000 \
task.env.grasp_cache_name=3pose \
task.env.initPoseMode=low \
task.env.reset_height_threshold=0.12 \
task.env.privInfo.enable_obj_orientation=True task.env.privInfo.enable_obj_angvel=True task.env.privInfo.enable_ft_pos=True \
${EXTRA_ARGS}
