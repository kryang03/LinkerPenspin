#!/bin/bash
# ------------------------------------------------------------------------------------
# 功能: 启动 Isaac Gym 进行 Linker Hand 抓取姿态生成任务
# 用法: ./scripts/linker_gen_grasp.sh <GPU_ID> <task.env.baseObjScale>
# 示例: ./scripts/linker_gen_grasp.sh 0 0.3
# ------------------------------------------------------------------------------------
 
# --- 脚本参数 ---
GPUS=$1           # 要使用的 GPU ID
SCALE=$2          # 物体的缩放比例

# --- 额外参数处理 ---
# 允许传入额外的 Hydra 命令行参数
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo "使用 GPU: ${GPUS}, 物体缩放: ${SCALE}"
echo "额外参数: ${EXTRA_ARGS}"

# --- 启动命令 ---
CUDA_VISIBLE_DEVICES=${GPUS} \
python gen_grasp.py \
    task=LinkerHandGrasp \
    test=True \
    headless=True \
    pipeline=gpu \
    task.env.numEnvs=1024 \
    task.env.grasp_cache_name=3pose \
    task.env.controller.controlFrequencyInv=8 \
    task.env.episodeLength=40 \
    task.env.controller.torque_control=False \
    task.env.genGrasps=True \
    task.env.baseObjScale="${SCALE}" \
    task.env.genGraspCategory=pencil \
    task.env.object.type=cylinder_pencil-5-7 \
    task.env.randomization.randomizeMass=True \
    task.env.randomization.randomizeMassLower=0.18 \
    task.env.randomization.randomizeMassUpper=0.22 \
    task.env.randomization.randomizeCOM=False \
    task.env.randomization.randomizeFriction=False \
    task.env.randomization.randomizePDGains=False \
    task.env.randomization.randomizeScale=False \
    train.ppo.priv_info=True \
    task.env.reset_height_threshold=0.16 \
    ${EXTRA_ARGS}