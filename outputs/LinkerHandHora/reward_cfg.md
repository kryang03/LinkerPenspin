cfg0: REWARD_SCALE = [1,1,0.5,0.1,0.01,1,1]
cfg1: REWARD_SCALE = [1,1,0.05,0.1,0.01,1,1] lessPosePenalty
cfg2: REWARD_SCALE = [1,1,0.05,0.1,0.05,1.5,1] lessPosePenalty More_zDistPenalty
      CONTACT_THRESH = 0.02
      TACTILE_FORCE_MAX = 4.0

---------------将tensorboard改为按比例缩放后显示，对REWARD_SCALE append了reward penalty的scale------------------      

cfg3: cfg3用于重训pose3，加速旋转
      
      rotate奖励/惩罚来源：
      vec_dot = (object_angvel * self.rot_axis_buf).sum(-1)
      rotate_reward = torch.clip(vec_dot, max=self.angvel_clip_max, min=self.angvel_clip_min)
      rotate_penalty = torch.where(vec_dot > self.angvel_penalty_threshold, vec_dot - self.angvel_penalty_threshold, 0)
      
      更改部分：
      task.env.reward.angvelClipMax=0.5 task.env.reward.angvelPenaltyThres=1.0 
      改为task.env.reward.angvelClipMax=5.0 task.env.reward.angvelPenaltyThres=5.0 
      REWARD_SCALE_DICT = {
      'obj_linvel_penalty': 1.0,
      'rotate_reward': 1.0,
      'pose_diff_penalty': 0, 改为0
      'torque_penalty': 0.1,
      'work_penalty': 0.05,
      'pencil_z_dist_penalty': 1.5,
      'position_penalty': 1.0,
      'rotate_penalty': 1.0 只是append，没变化
      }
cfg4：cfg4用于重训3pose、4pose、6pose
      task.env.reward.angvelClipMax=0.5 task.env.reward.angvelPenaltyThres=1.0 
      改为task.env.reward.angvelClipMax=2.0 task.env.reward.angvelPenaltyThres=3.0 
      REWARD_SCALE_DICT = {
      'obj_linvel_penalty': 1.0,
      'rotate_reward': 1.0,
      'pose_diff_penalty': 0, 改为0
      'torque_penalty': 0.1,
      'work_penalty': 0.05,
      'pencil_z_dist_penalty': 1.5,
      'position_penalty': 1.0,
      'rotate_penalty': 1.0 只是append，没变化
      }

---------------将旋转角度加入tensorboard，更改了rot reward逻辑，只奖励(0,clipMax)，惩罚(,clipMin)和(thresh,)----------------- 
cfg5:
      task.env.reward.angvelClipMax=5.0 task.env.reward.angvelPenaltyThres=5.0 task.env.reward.angvelClipMin=-0.1 \
      REWARD_SCALE_DICT = {
      'obj_linvel_penalty': 1.0,
      'rotate_reward': 1.0,
      'pose_diff_penalty': 0,
      'torque_penalty': 0.1,
      'work_penalty': 0.05,
      'pencil_z_dist_penalty': 1.5,
      'position_penalty': 1.0,
      'rotate_penalty': 4.0 改为4
      }
-------------------------------- 
1. 当前没用上position_penalty，由于数量级是0.04^2 *0.1，完全忽略不计；考虑将它改成掉落的sparse reward
2. 改了周期的
REWARD_SCALE_DICT = {
    'obj_linvel_penalty': 1.0,
    'rotate_reward': 0.7,
    'waypoint_sparse_reward': 90,
    'torque_penalty': 0.1,
    'work_penalty': 0.05,
    'pencil_z_dist_penalty': 1.5,
    'position_penalty': 1.0,
    'rotate_penalty': 4.0
}
REWARD_SCALE_DICT = {
    'obj_linvel_penalty': 1.0,
    'rotate_reward': 0.7,
    'waypoint_sparse_reward': 100,
    'torque_penalty': 0.1,
    'work_penalty': 0.05,
    'pencil_z_dist_penalty': 2.0,
    'position_penalty': 1.0,
    'rotate_penalty': 4.0
}
REWARD_SCALE_DICT = {
    'obj_linvel_penalty': 1.0,
    'rotate_reward': 0.7,
    'waypoint_sparse_reward': 200,
    'torque_penalty': 0.1,
    'work_penalty': 0.05,
    'pencil_z_dist_penalty': 2.0,
    'position_penalty': 1.0,
    'rotate_penalty': 4.0
}
REWARD_SCALE_DICT = {
    'obj_linvel_penalty': 1.0,
    'rotate_reward': 0.7,
    'waypoint_sparse_reward': 200,
    'torque_penalty': 0.1,
    'work_penalty': 0.05,
    'pencil_z_dist_penalty': 3.0,
    'position_penalty': 1.0,
    'rotate_penalty': 4.0
}
-------------------------------- 
# 将position_penalty改为height penalty，惩罚和初始状态距离的二范数，并调整了数量级1->1000
# 将waypoint_sparse_reward中的hand_similarity改为hand_similarity*orientation_similarity
REWARD_SCALE_DICT = {
    'obj_linvel_penalty': 1.0,
    'rotate_reward': 0.7,
    'waypoint_sparse_reward': 200,
    'torque_penalty': 0.1,
    'work_penalty': 0.05,
    'pencil_z_dist_penalty': 3.0,
    'position_penalty': 1000.0,
    'rotate_penalty': 4.0
}