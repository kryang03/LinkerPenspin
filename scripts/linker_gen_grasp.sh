#!/bin/bash
# ./scripts/linker_gen_grasp.sh <GPU_ID> <task.env.baseObjScale>
# ./scripts/linker_gen_grasp.sh 0 0.3
 
GPUS=$1
SCALE=$2

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python gen_grasp.py task=LinkerHandGrasp test=True \
task.env.numEnvs=8192 headless=True pipeline=cpu task.env.grasp_cache_name=4pose \
task.env.controller.controlFrequencyInv=8 task.env.episodeLength=40 \
task.env.controller.torque_control=False task.env.genGrasps=True task.env.baseObjScale="${SCALE}" \
task.env.genGraspCategory=pencil \
task.env.object.type=cylinder_pencil-5-7 \
task.env.randomization.randomizeMass=True task.env.randomization.randomizeMassLower=0.18 task.env.randomization.randomizeMassUpper=0.22 \
task.env.randomization.randomizeCOM=False \
task.env.randomization.randomizeFriction=False \
task.env.randomization.randomizePDGains=False \
task.env.randomization.randomizeScale=False \
train.ppo.priv_info=True \
task.env.reset_height_threshold=0.16 \
${EXTRA_ARGS}