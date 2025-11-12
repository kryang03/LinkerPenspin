#!/bin/bash
# ========================================
# 功能1: 纯PPO强化学习训练Teacher模型
# 使用全部信息（包括特权信息）在仿真中训练
# ========================================
# chmod -R u+x ../../Code
# nohup scripts/train_rl_teacher.sh 0 42 teacher_rl_test > teacher_rl_test.txt &

# CHECKLIST: 
# 1. 命令的最后一个参数对应OUTPUT_NAME，测reward时用tmp，开训时标注信息
# 2. task.env.grasp_cache_name = 所使用的canonical pose文件名
# 3. 确认最低高度task.env.reset_height_threshold
# 4. 确定命令的GPU参数
# 5. 剪切rotation reward的 angvelClipMax 和产生rotation penalty的 angvelPenaltyThres
# 6. 确认train.algo=PPOTeacher

# 参数说明
GPUS=$1        # GPU ID
SEED=$2        # 随机种子
OUTPUT_NAME=$3 # 输出目录名称

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=LinkerHandHora headless=True seed=${SEED} \
train.ppo.output_name=LinkerHandHora/${OUTPUT_NAME} \
train.algo=PPOTeacher \
task.env.grasp_cache_name=3pose \
task.env.reward.angvelClipMax=3.0 \
task.env.reward.angvelPenaltyThres=3.5 \
task.env.reward.angvelClipMin=-0.1 \
train.ppo.minibatch_size=16384 \
train.ppo.horizon_length=12 \
task.env.numEnvs=8192 \
task.env.object.type=cylinder_pencil-5-7 \
task.env.randomForceProbScalar=0.25 \
task.env.rotation_axis=+z \
task.env.genGraspCategory=pencil \
task.env.privInfo.enable_obj_orientation=True \
task.env.privInfo.enable_ft_pos=True \
task.env.privInfo.enable_obj_angvel=True \
task.env.privInfo.enable_tactile=True \
train.ppo.max_agent_steps=10000000000 \
task.env.randomization.randomizeScaleList=[0.3] \
train.ppo.priv_info=True \
task.env.hora.point_cloud_sampled_dim=100 \
task.env.numObservations=126 \
task.env.initPoseMode=low \
task.env.reset_height_threshold=0.12 \
task.env.forceScale=2.0 \
task.env.enable_obj_ends=True \
wandb_activate=False \
${EXTRA_ARGS}
