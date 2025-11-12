#!/bin/bash
# ========================================
# 功能3: RL+BC训练Student模型
# 只使用真机可获得的信息（proprio history），结合Teacher模型的回放
# ========================================
# chmod +x scripts/train_rl_bc_student.sh
# nohup scripts/train_rl_bc_student.sh 0 42 student_rl_bc_test outputs/LinkerHandHora/teacher_rl_test/teacher_nn/best_reward_xxx.pth > student_rl_bc_test.txt &

# 参数说明
GPUS=$1            # GPU ID
SEED=$2            # 随机种子
OUTPUT_NAME=$3     # 输出目录名称
DEMON_PATH=$4      # Teacher模型路径 (必须提供)

# 检查是否提供了demon_path
if [ -z "$DEMON_PATH" ]; then
    echo "错误: 必须提供DEMON_PATH参数 (Teacher模型路径)"
    echo "用法: $0 <GPUS> <SEED> <OUTPUT_NAME> <DEMON_PATH>"
    exit 1
fi

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=LinkerHandHora headless=True seed=${SEED} \
train.ppo.output_name=LinkerHandHora/${OUTPUT_NAME} \
train.algo=PPO_RL_BC_Student \
train.demon_path=${DEMON_PATH} \
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
train.ppo.priv_info=False \
train.ppo.proprio_mode=True \
train.ppo.input_mode=proprio \
train.ppo.proprio_len=30 \
train.ppo.bc_loss_coef=1.0 \
train.ppo.enable_latent_loss=False \
train.ppo.use_l1=True \
train.ppo.learning_rate=1e-3 \
task.env.hora.point_cloud_sampled_dim=100 \
task.env.numObservations=126 \
task.env.initPoseMode=low \
task.env.reset_height_threshold=0.12 \
task.env.forceScale=2.0 \
task.env.enable_obj_ends=True \
wandb_activate=False \
${EXTRA_ARGS}
