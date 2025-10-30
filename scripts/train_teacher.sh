#!/bin/bash
# chmod -R u+x ../../Code
# nohup scripts/train_teacher.sh 3 42 pose3_50k_cfg7_200_3 > pose3_50k_cfg7_200_3.txt &                   生成可以关闭终端的训练任务，需要通过kill ID1 ID2关闭    

# CHECKLIST: 
# 1. 命令的最后一个参数对应OUTPUT_NAME，测reward时用tmp，开训时标注信息
# 2. task.env.grasp_cache_name = 所使用的canonical pose文件名
# 3. 确认最低高度task.env.reset_height_threshold
# 4. 确定命令的GPU参数
# 5. 剪切rotation reward的 angvelClipMax 和产生rotation penalty的 angvelPenaltyThres
# 6. 检查checkpoint的名称，是否为 Cache对应的best*，即 checkpoint=outputs/LinkerHandHora/"${CACHE}"/stage1_nn/best*.pth
#    抑或是从头训练，将 checkpoint 行删掉
# 7. 去linker_hand_hora.py中修改CHECKLIST


# 下面的换行部分就不能插入注释了；注意换行前要有空格
# 不能删除随机化，因为加载dof缓存的条件是if self.randomize_scale and self.scale_list_init:
# numObservations = self.num_obs = joint target pos = 3(history) * 2 * num_dofs 
# initPoseMode是用来解决高度问题的（现在的实现绕过了这个部分）
# test=True 是用来load weight进行测试的 

GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=LinkerHandHora headless=True seed=${SEED} \
train.ppo.output_name=LinkerHandHora/${CACHE} task.env.grasp_cache_name=3pose \
task.env.reward.angvelClipMax=3.0 task.env.reward.angvelPenaltyThres=3.5 task.env.reward.angvelClipMin=-0.1 \
train.ppo.minibatch_size=16384 train.ppo.horizon_length=12 task.env.numEnvs=8192 \
task.env.object.type=cylinder_pencil-5-7 \
experiment=rl \
task.env.randomForceProbScalar=0.25 train.algo=PPO train.ppo.proprio_adapt=False \
task.env.rotation_axis=+z \
task.env.genGraspCategory=pencil task.env.privInfo.enable_obj_orientation=True \
task.env.privInfo.enable_ft_pos=True task.env.privInfo.enable_obj_angvel=True \
train.ppo.max_agent_steps=10000000000 \
task.env.randomization.randomizeScaleList=[0.3] task.env.privInfo.enable_tactile=True \
train.ppo.priv_info=True task.env.hora.point_cloud_sampled_dim=100 \
task.env.numObservations=126 task.env.initPoseMode=low task.env.reset_height_threshold=0.12 \
task.env.forceScale=2.0 \
task.env.enable_obj_ends=True wandb_activate=False \
${EXTRA_ARGS}
