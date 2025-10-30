#!/bin/bash
# chmod -R u+x ../../Code
# scripts/vis_teacher.sh pose4_50k_cfg2 > contacts/pose4_50k_cfg2.txt
# scripts/vis_teacher.sh pose6_50k_cfg4 > contacts/pose6_50k_cfg4.txt
# scripts/vis_teacher.sh pose3_50k_cfg7_200_3 > contacts/pose3_50k_cfg7_200_3.txt
# scripts/vis_teacher.sh pose3_50k_cfg9 > contacts/pose3_50k_cfg9.txt
# scripts/vis_teacher.sh pose3_3k_cfg_c > contacts/pose3_3k_cfg_c.txt

#best:scripts/vis_teacher.sh pose3_50k_cfg2 > contacts/pose3_50k_cfg2.txt
# CHECKLIST
# 1. 命令的最后一个参数指向output文件夹的名称，三维力信息是否重定向到正确的文件夹
# 2. 检查checkpoint的名称
# 3. reset_height_threshold 是否=0.14
# 4. grasp_cache_name对应canonical pose的cache名称
# 5. linker_hand_hora.py 中的CHECKLIST
# 6. episodeLength 训练时为400

# test=True 是用来load weight进行测试的 
CACHE=$1

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

python train.py task=LinkerHandHora headless=False \
task.env.numEnvs=1 test=True checkpoint=outputs/LinkerHandHora/"${CACHE}"/stage1_nn/best*.pth \
task.env.object.type=cylinder_pencil-5-7 \
task.env.episodeLength=4000 \
task.env.randomForceProbScalar=0.25 train.algo=PPO \
task.env.rotation_axis=+z \
task.env.genGraspCategory=pencil task.env.privInfo.enable_obj_orientation=True \
task.env.privInfo.enable_ft_pos=True task.env.privInfo.enable_obj_angvel=True \
task.env.randomization.randomizeScaleList=[0.3] task.env.grasp_cache_name=3pose \
task.env.asset.handAsset=assets/linker_hand/L25_dof_urdf.urdf \
task.env.privInfo.enable_tactile=True train.ppo.priv_info=True task.env.hora.point_cloud_sampled_dim=100 \
task.env.numObservations=126 task.env.initPoseMode=low task.env.reset_height_threshold=0.14 \
task.env.reward.angvelClipMax=0.5 task.env.forceScale=2.0 \
task.env.reward.angvelPenaltyThres=1.0 \
task.env.enable_obj_ends=True \
wandb_activate=False \
${EXTRA_ARGS}
