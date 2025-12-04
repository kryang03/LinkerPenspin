"""
python cache/linker_hand_grasp.py
LinkerHandGrasp - 生成基于Flying Hand的抓取初始化缓存

使用interactive_tune.py中的triangle_grasp预设生成61维缓存:
- 27维 hand_actual_dof: 手部实际物理位置（用于设置初始物理状态，避免穿模爆炸）
- 27维 hand_target_dof: 手部目标位置（用于设置PD控制目标，产生抓握力）
- 3维 object_pos: 物体位置
- 4维 object_rot: 物体旋转

关键设计说明（零力矩陷阱解决方案）:
- 物理位置(actual): 仿真稳定后的手指实际位置，被笔挤开后的状态
- 控制目标(target): 初始意图位置，手指试图捏合的目标
- 加载时: 物理状态设为actual（平静），PD目标设为target（产生力矩）
- 效果: PD误差 = target - actual ≠ 0，产生持续抓握力

支持Flying Hand 6自由度虚拟基座和禁用ring/little手指

优化策略："优胜劣汰，即时补充"进化策略
- 一旦某个环境失败（物体掉落/倾倒），立刻 reset 为新的随机抓取姿态
- 如果某个环境坚持到了目标步数，保存它，然后 reset 换新姿态
- 所有环境永远处于"尝试抓取"状态，没有空转等待，大幅提升生成效率
"""

# 添加项目根目录到Python路径
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 命令行参数解析
import argparse

# 重要参数配置 - 可通过命令行覆盖
DEFAULT_CONFIG = {
    # 早期终止阈值
    'relative_z_drop_threshold': 0.012,  # 物体下降超过此值则重置（米）这个值较小，因为Flying Hand抓取比较精细，要避免差的初始化
    'pencil_tilt_threshold': 0.015,      # LinkerPen两端高度差超过此值视为倾倒（米）
    
    # 仿真配置
    'headless': True,                   # 是否无头模式（无GUI）
    'num_envs': 8192,                      # 并行环境数量
    
    # 生成配置
    'target_samples': 30000,            # 目标生成样本数
    'stability_steps': 100,              # 需要坚持的稳定步数
    'cache_filename': 'grasp_cache.npy', # 输出缓存文件名
    
    # 随机种子
    'seed': 42,                         # 随机种子
    
    # 物体配置
    'pen_length': 0.18,                 # LinkerPen长度（米）
    
    # 姿态配置
    'num_canonical_poses': 3,           # 标准姿态数量
}

# 使用示例:
# python cache/linker_hand_grasp.py --target-samples 1000 --stability-steps 30 --num-envs 8
# python cache/linker_hand_grasp.py --no-headless --relative-z-drop-threshold 0.08 --pencil-tilt-threshold 0.15
# python cache/linker_hand_grasp.py --help  # 查看所有可用参数

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LinkerHand 抓取缓存生成器')
    
    # 早期终止阈值
    parser.add_argument('--relative-z-drop-threshold', type=float, 
                       default=DEFAULT_CONFIG['relative_z_drop_threshold'],
                       help=f'物体下降超过此值则重置（米），默认: {DEFAULT_CONFIG["relative_z_drop_threshold"]}')
    parser.add_argument('--pencil-tilt-threshold', type=float,
                       default=DEFAULT_CONFIG['pencil_tilt_threshold'],
                       help=f'LinkerPen两端高度差超过此值视为倾倒（米），默认: {DEFAULT_CONFIG["pencil_tilt_threshold"]}')
    
    # 仿真配置
    parser.add_argument('--headless', action='store_true', default=DEFAULT_CONFIG['headless'],
                       help='启用无头模式（无GUI）')
    parser.add_argument('--no-headless', action='store_true', 
                       help='禁用无头模式（显示GUI）')
    parser.add_argument('--num-envs', type=int, default=DEFAULT_CONFIG['num_envs'],
                       help=f'并行环境数量，默认: {DEFAULT_CONFIG["num_envs"]}')
    
    # 生成配置
    parser.add_argument('--target-samples', type=int, default=DEFAULT_CONFIG['target_samples'],
                       help=f'目标生成样本数，默认: {DEFAULT_CONFIG["target_samples"]}')
    parser.add_argument('--stability-steps', type=int, default=DEFAULT_CONFIG['stability_steps'],
                       help=f'需要坚持的稳定步数，默认: {DEFAULT_CONFIG["stability_steps"]}')
    parser.add_argument('--cache-filename', type=str, default=DEFAULT_CONFIG['cache_filename'],
                       help=f'输出缓存文件名，默认: {DEFAULT_CONFIG["cache_filename"]}')
    
    # 随机种子
    parser.add_argument('--seed', type=int, default=DEFAULT_CONFIG['seed'],
                       help=f'随机种子，默认: {DEFAULT_CONFIG["seed"]}')
    
    # 物体配置
    parser.add_argument('--pen-length', type=float, default=DEFAULT_CONFIG['pen_length'],
                       help=f'LinkerPen长度（米），默认: {DEFAULT_CONFIG["pen_length"]}')
    
    args = parser.parse_args()
    
    # 处理互斥参数
    if args.no_headless:
        args.headless = False
    
    return args

# 解析命令行参数
args = parse_args()

# isaacgym必须在torch之前导入
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_apply

import hydra
import torch
import numpy as np
from omegaconf import OmegaConf

# 注册自定义 resolver (需要在 hydra.main 之前)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)

from penspin.tasks.linker_hand_hora import LinkerHandHora
from penspin.utils.robot_config import (
    NUM_TOTAL_DOF_FLYING,
    FLYING_DOF_INDICES,
)

# 物体状态维度: 3 位置 + 4 旋转 = 7
NUM_OBJECT_DIMS = 7


class LinkerHandGrasp(LinkerHandHora):
    """
    基于Flying Hand的抓取姿态生成器
    
    生成的缓存格式: [hand_dof(27), object_pos(3), object_rot(4)] = 34维
    """
    
    def __init__(self, config, sim_device, graphics_device_id, headless):
        # 修改配置以跳过缓存加载（因为这个脚本就是用来生成缓存的）
        # genGrasps=True 启用抓取生成模式，跳过缓存加载
        config['env']['genGrasps'] = True
        config['env']['randomization']['randomizeScale'] = False  # 禁用缩放随机化
        config['env']['randomization']['scaleListInit'] = False   # 禁用缓存加载
        config['env']['numEnvs'] = args.num_envs  # 使用命令行参数
        # 禁用手指禁用功能以避免动作维度不匹配问题
        # （缓存生成需要完整的27 DOF）
        if 'actionSpace' in config['env']:
            config['env']['actionSpace']['disableRingLittleFinger'] = False
        
        super().__init__(config, sim_device, graphics_device_id, headless)
        
        # [逻辑修正]
        self.num_canonical_poses = 3
        target_total = config.get('target_samples', 10000)
        # 强制向上取整，确保每类都有明确目标
        self.target_per_pose = int(np.ceil(target_total / self.num_canonical_poses))
        
        self.saved_pose_counts = torch.zeros(self.num_canonical_poses, dtype=torch.long, device=self.device)
        self.env_current_pose_id = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        
        # 保存的抓取状态缓存: 61维 
        # [hand_actual(27) + hand_target(27) + object_pos(3) + object_rot(4)]
        # - hand_actual: 仿真后实际位置（用于物理初始化，避免穿模）
        # - hand_target: 原始目标位置（用于PD控制，产生抓握力）
        self.cache_dim = NUM_TOTAL_DOF_FLYING * 2 + NUM_OBJECT_DIMS  # 27*2 + 7 = 61
        self.saved_grasping_states = torch.zeros(
            (0, self.cache_dim), 
            dtype=torch.float32, 
            device=self.device
        )
        
        # 从interactive_tune.py提取的triangle_grasp预设 (已验证正确)
        # 格式: {'hand_dof': [27维], 'object_pos': [3维], 'object_rot': [4维 xyzw]}
        # 注意: IsaacGym 四元数顺序是 (x, y, z, w)
        self.canonical_pose_dict = {
            'pencil': [
                # triangle_grasp_0 - 基础三角抓取姿态 (来自 interactive_tune.py)
                {
                    'hand_dof': [
                        # Flying base: px, py, pz, rx, ry, rz (6)
                        0.0, -0.0, 0.35, -0.0, -1.31, 0.0,
                        # Index finger: base_joint, MCP, PIP, DIP (4)
                        0.1, -1.0, -0.6, -0.4,
                        # Little finger (4) - 保持零位
                        -0.0, -0.0, -0.0, -0.0,
                        # Middle finger: base_joint, MCP, PIP, DIP (4)
                        0.0, -1.15, -0.6, -0.4,
                        # Ring finger (4) - 保持零位
                        -0.0, -0.0, -0.0, -0.0,
                        # Thumb: CMC_abd, CMC_flex, MCP, IP, DIP (5)
                        -0.0, -1.35, -0.35, -0.5, -0.35,
                    ],
                    'object_pos': [-0.12, 0.034, 0.526001],
                    'object_rot': [0.12166, -0.696562, 0.696562, -0.12166],  # xyzw
                },
                # triangle_grasp_1 - 第二个三角抓取姿态
                {
                    'hand_dof': [
                        # Flying base: px, py, pz, rx, ry, rz
                        0.0, 0.0, 0.350001, 0.0, -1.310001, 0.0,
                        # Index finger
                        0.08, -1.0, -0.6, -0.4,
                        # Little finger
                        -0.0, -0.0, -0.0, -0.0,
                        # Middle finger
                        -0.0, -1.15, -0.65, -0.4,
                        # Ring finger
                        -0.0, -0.0, -0.0, -0.0,
                        # Thumb
                        0.1, -1.35, -0.35, -0.5, -0.35,
                    ],
                    'object_pos': [-0.116, 0.022, 0.528001],
                    'object_rot': [0.503743, -0.496228, 0.496228, -0.503743],  # xyzw
                },
                # triangle_grasp_2 - 第三个抓取姿态
                {
                    'hand_dof': [
                        # Flying base: px, py, pz, rx, ry, rz
                        -0.000000, 0.000000, 0.350000, 0.000000, -1.310000, -0.000000,
                        # Index finger
                        0.03, -1.0, -0.6, -0.4,
                        # Little finger
                        -0.0, -0.0, -0.0, -0.0,
                        # Middle finger
                        -0.0, -1.05, -0.65, -0.4,
                        # Ring finger
                        -0.0, -0.0, -0.0, -0.0,
                        # Thumb
                        0.0, -1.35, -0.15, -0.6, -0.5,
                    ],
                    'object_pos': [-0.120102, 0.028283, 0.530196],
                    'object_rot': [0.60706, -0.361703, 0.361988, -0.607958],  # xyzw
                },
            ]
        }
        
        self.grasp_counter = 0
        self.canonical_poses = None
        
    def is_cache_complete(self):
        """检查是否所有姿态都达到了目标配额"""
        # 只有当所有类别的计数都 >= 目标值时，才算完成
        # 这样会强制程序等待最难的那个 Pose 跑完
        return torch.all(self.saved_pose_counts >= self.target_per_pose)
        
    def _setup_canonical_poses(self):
        """设置规范姿态用于环境重置"""
        object_type = self.config["env"].get("object_type", "pencil")
        poses = self.canonical_pose_dict.get(object_type, self.canonical_pose_dict['pencil'])
        
        # 转换为tensor格式: [num_poses, 34]
        pose_list = []
        for pose in poses:
            hand_dof = torch.tensor(pose['hand_dof'], dtype=torch.float32, device=self.device)
            obj_pos = torch.tensor(pose['object_pos'], dtype=torch.float32, device=self.device)
            obj_rot = torch.tensor(pose['object_rot'], dtype=torch.float32, device=self.device)
            full_pose = torch.cat([hand_dof, obj_pos, obj_rot])
            pose_list.append(full_pose)
            
        self.canonical_poses = torch.stack(pose_list)  # [num_poses, 34]
        self.num_canonical_poses = len(poses)
        # 重新计算目标数量
        target_total = self.config.get('target_samples', 10000)
        self.target_per_pose = int(np.ceil(target_total / self.num_canonical_poses))
        
        print(f"[LinkerHandGrasp] 加载了 {len(poses)} 个规范姿态, 维度: {self.canonical_poses.shape}, 目标每姿态: {self.target_per_pose}")
        
    def reset_idx(self, env_ids):
        """重置指定环境，优先选择样本数不足的姿态"""
        if self.canonical_poses is None:
            self._setup_canonical_poses()
            # 更新实际的姿态数量
            self.num_canonical_poses = len(self.canonical_poses)
            
        num_envs_to_reset = len(env_ids)
        if num_envs_to_reset == 0:
            return
            
        # [修改] 智能选择姿态 ID
        # 1. 找出哪些姿态还没有达到目标数量
        remaining_counts = self.target_per_pose - self.saved_pose_counts
        # 获取需要补充的姿态索引 (valid_pose_ids)
        valid_pose_ids = torch.where(remaining_counts > 0)[0]
        
        if len(valid_pose_ids) == 0:
            # 如果所有姿态都存满了（也就是任务即将结束），则随机选一个继续跑，防止报错
            valid_pose_ids = torch.arange(self.num_canonical_poses, device=self.device)
        
        # 2. 为每个需要重置的环境，从 valid_pose_ids 中随机选一个
        # torch.randint 无法直接在非连续索引中采样，所以我们先采下标，再映射回去
        rand_indices = torch.randint(0, len(valid_pose_ids), (num_envs_to_reset,), device=self.device)
        selected_pose_ids = valid_pose_ids[rand_indices]  # [num_envs_to_reset]
        
        # 3. 记录下来，以便 save 时知道是谁成功了
        self.env_current_pose_id[env_ids] = selected_pose_ids
        
        # 4. 根据 ID 获取具体的姿态数据
        selected_poses = self.canonical_poses[selected_pose_ids]  # [num_envs, 34]
        
        # 分解姿态
        hand_dof = selected_poses[:, :NUM_TOTAL_DOF_FLYING]  # [num_envs, 27]
        object_pos = selected_poses[:, NUM_TOTAL_DOF_FLYING:NUM_TOTAL_DOF_FLYING+3]  # [num_envs, 3]
        object_rot = selected_poses[:, NUM_TOTAL_DOF_FLYING+3:]  # [num_envs, 4]
        
        # 添加随机扰动
        if self.config["env"].get("randomize_init", True):
            # 手部DOF扰动 (Flying base位置更大扰动)
            hand_noise = torch.zeros_like(hand_dof)
            # Flying base位置扰动
            hand_noise[:, 0:3] = torch.randn_like(hand_noise[:, 0:3]) * 0.01
            # Flying base旋转扰动
            hand_noise[:, 3:6] = torch.randn_like(hand_noise[:, 3:6]) * 0.05
            # 手指关节扰动 - 只对active手指 (index, middle, thumb)
            # Active DOF: index(6-9), middle(14-17), thumb(22-26)
            # Skip little(10-13) and ring(18-21) as they are disabled
            hand_noise[:, 6:10] = torch.randn_like(hand_noise[:, 6:10]) * 0.05  # index
            hand_noise[:, 14:18] = torch.randn_like(hand_noise[:, 14:18]) * 0.05  # middle
            hand_noise[:, 22:27] = torch.randn_like(hand_noise[:, 22:27]) * 0.05  # thumb
            hand_dof = hand_dof + hand_noise
            
            # 物体位置扰动
            pos_noise = torch.randn_like(object_pos) * 0.005
            object_pos = object_pos + pos_noise
            
        # 应用手部DOF
        self.linker_hand_dof_pos[env_ids, :NUM_TOTAL_DOF_FLYING] = hand_dof
        self.linker_hand_dof_vel[env_ids, :] = 0.0
        
        # 设置目标位置
        self.prev_targets[env_ids, :NUM_TOTAL_DOF_FLYING] = hand_dof
        self.cur_targets[env_ids, :NUM_TOTAL_DOF_FLYING] = hand_dof
        self.init_pose_buf[env_ids, :NUM_TOTAL_DOF_FLYING] = hand_dof
        
        # 应用物体状态
        object_indices = self.object_indices[env_ids]
        self.root_state_tensor[object_indices, 0:3] = object_pos
        self.root_state_tensor[object_indices, 3:7] = object_rot
        self.root_state_tensor[object_indices, 7:13] = 0.0  # 清零速度
        
        # 记录初始物体高度用于相对 relative_z_drop_threshold
        self.init_object_z_buf[env_ids] = object_pos[:, 2]
        
        # 更新仿真状态
        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        object_indices = object_indices.to(torch.int32)
        
        self.gym.set_dof_state_tensor_indexed(
            self.sim, 
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(hand_indices), 
            len(hand_indices)
        )
        
        # 关键：设置 PD 控制器的目标位置
        # 在 GPU pipeline 中，必须显式设置目标，否则 PD 控制器会使用默认目标（0）
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.cur_targets),
            gymtorch.unwrap_tensor(hand_indices),
            len(hand_indices)
        )
        
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(object_indices),
            len(object_indices)
        )
        
        # 重置其他状态
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1
        
    def save_grasping_state(self, env_ids):
        """保存状态，实施严格的配额过滤"""
        if len(env_ids) == 0:
            return
            
        # 1. 获取这些环境对应的 Pose ID
        candidate_pose_ids = self.env_current_pose_id[env_ids]
        
        # 2. [核心修正] 构建过滤器 mask，决定哪些能存
        # 默认为 False (不保存)
        save_mask = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        
        # 遍历每种 Pose，检查配额
        for pose_id in range(self.num_canonical_poses):
            # 找出当前这批里属于 pose_id 的索引
            is_this_pose = (candidate_pose_ids == pose_id)
            num_candidates = int(torch.sum(is_this_pose).item())
            
            if num_candidates > 0:
                # 计算还需要多少个
                current_count = self.saved_pose_counts[pose_id].item()
                needed = max(0, self.target_per_pose - current_count)
                
                if needed > 0:
                    # 找出对应的局部索引
                    local_indices = torch.where(is_this_pose)[0]
                    # 如果候选数量 > 需要数量，截断
                    num_to_take = min(num_candidates, needed)
                    indices_to_take = local_indices[:num_to_take]
                    
                    # 标记为可保存
                    save_mask[indices_to_take] = True
                    
                    # 立即更新计数器 (防止同批次后续逻辑误判)
                    self.saved_pose_counts[pose_id] += num_to_take
        
        # 3. 如果没有通过过滤的，直接返回
        num_accepted = int(torch.sum(save_mask).item())
        if num_accepted == 0:
            return

        # 4. 只筛选出通过 mask 的环境 ID
        final_env_ids = env_ids[save_mask]
        
        # 5. 执行保存 (对筛选后的 ID)
        # ================================================================
        # [关键设计] 同时保存「实际位置」和「目标位置」
        # ================================================================
        # 问题背景:
        # - 只保存实际位置 → 零力矩陷阱（滑落）
        #   加载时 PD 误差 = 0，没有抓握力
        # - 只保存目标位置 → 穿模爆炸
        #   物理位置设在笔内部，第 0 帧碰撞排斥力巨大
        #
        # 解决方案（双位置缓存）:
        # - hand_actual: 仿真稳定后的实际位置（被笔挤开）
        #   用于设置物理状态，避免穿模
        # - hand_target: 原始目标/意图位置（试图捏合）
        #   用于设置 PD 控制目标，产生抓握力
        # 
        # 加载时:
        #   set_dof_state_tensor(..., actual)  # 物理位置：平静
        #   set_dof_position_target_tensor(..., target)  # 控制目标：有力
        # ================================================================
        
        # 实际位置: 仿真稳定后的手指位置（被物体挤开）
        hand_actual = self.linker_hand_dof_pos[final_env_ids, :NUM_TOTAL_DOF_FLYING].clone()
        
        # 目标位置: 原始意图（手指试图达到的位置，产生夹紧力）
        hand_target = self.init_pose_buf[final_env_ids, :NUM_TOTAL_DOF_FLYING].clone()

        object_indices = self.object_indices[final_env_ids]
        object_pos = self.root_state_tensor[object_indices, 0:3].clone()
        object_rot = self.root_state_tensor[object_indices, 3:7].clone()
        
        # 拼接: [actual(27) + target(27) + obj_pos(3) + obj_rot(4)] = 61维
        grasping_states = torch.cat([hand_actual, hand_target, object_pos, object_rot], dim=-1)
        
        self.saved_grasping_states = torch.cat([
            self.saved_grasping_states, 
            grasping_states
        ], dim=0)
        
        self.grasp_counter += num_accepted
        
        # 打印日志
        # print(f"[LinkerHandGrasp] 接受 +{num_accepted} (抛弃 {len(env_ids)-num_accepted}), 总数: {self.grasp_counter}")
        # 打印当前详细状态
        counts_list = self.saved_pose_counts.cpu().numpy().tolist()
        # print(f"   >>> 当前分布: {counts_list} (目标: {self.target_per_pose})")
        
    def export_cache(self, filename: str):
        """导出缓存到文件"""
        if self.saved_grasping_states.shape[0] == 0:
            print("[LinkerHandGrasp] 警告: 没有保存的状态可导出")
            return
            
        states_np = self.saved_grasping_states.cpu().numpy()
        np.save(filename, states_np)
        print(f"[LinkerHandGrasp] 导出 {states_np.shape[0]} 个状态到 {filename}")
        print(f"[LinkerHandGrasp] 缓存维度: {states_np.shape}")


@hydra.main(version_base="1.2", config_path="../configs", config_name="config")
def main(config):
    """
    生成抓取缓存的主函数
    
    采用"优胜劣汰，即时补充"进化策略：
    - 每个环境独立维护自己的步数计数器
    - 失败的环境立刻重置为新的随机姿态
    - 成功坚持到目标步数的环境保存后也重置
    - 所有环境永远处于"尝试抓取"状态，没有空转
    """
    from penspin.utils.misc import set_seed
    from penspin.utils.reformat import omegaconf_to_dict
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 转换配置
    cfg_task = omegaconf_to_dict(config.task)
    
    # 使用命令行参数覆盖配置
    cfg_task['env']['relative_z_drop_threshold'] = args.relative_z_drop_threshold
    cfg_task['env']['pencil_tilt_threshold'] = args.pencil_tilt_threshold
    cfg_task['env']['numEnvs'] = args.num_envs
    cfg_task['target_samples'] = args.target_samples
    cfg_task['stability_steps'] = args.stability_steps
    cfg_task['cache_filename'] = args.cache_filename
    
    # 创建环境
    env = LinkerHandGrasp(
        config=cfg_task,
        sim_device=config.sim_device,
        graphics_device_id=config.graphics_device_id,
        headless=args.headless,
    )
    
    print(f"[main] 创建了 {env.num_envs} 个环境")
    print(f"[main] 缓存维度: {env.cache_dim}")
    print(f"[main] 早期终止阈值: z_drop={env.relative_z_drop_threshold:.3f}m, tilt={env.pencil_tilt_threshold:.3f}m")
    
    # ================================================================
    # 配置参数（使用命令行参数）
    # ================================================================
    target_samples = args.target_samples
    stability_steps = args.stability_steps
    pen_length = args.pen_length
    
    # 预计算笔端点偏移（本地坐标系）
    pencil_end_offset_neg = torch.tensor([0, 0, -pen_length / 2], device=env.device)
    pencil_end_offset_pos = torch.tensor([0, 0, pen_length / 2], device=env.device)
    
    # ================================================================
    # 进化策略状态跟踪
    # ================================================================
    # 每个环境的独立计数器
    env_step_counters = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    # 统计信息
    total_attempts = 0
    success_count = 0
    fail_z_drop = 0
    fail_tilt = 0
    
    # 初始化：重置所有环境
    all_env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(all_env_ids)
    total_attempts += env.num_envs
    
    print(f"[main] 开始进化策略生成...")
    print(f"[main] 目标: 每类 {env.target_per_pose} 个样本 (总计约 {target_samples})")
    
    step_count = 0
    
    # [逻辑修正] 循环直到缓存彻底完成（所有类别都达标）
    # 原来是: while env.grasp_counter < target_samples:
    while not env.is_cache_complete():
        step_count += 1
        
        # 检查 Viewer 是否仍然运行（仅在非 headless 模式下）
        if not args.headless and env.viewer is not None:
            if env.gym.query_viewer_has_closed(env.viewer):
                print("[main] Viewer 已关闭，退出...")
                break
        
        # ============================================================
        # 物理仿真步进
        # ============================================================
        # 零动作（保持姿态）
        actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
        env.actions = actions
        
        # 更新目标（保持当前位置）
        env.cur_targets[:, :env.num_linker_hand_dofs] = env.prev_targets[:, :env.num_linker_hand_dofs]
        
        # 物理仿真
        for _ in range(env.control_freq_inv):
            env.update_low_level_control(0)
            env.gym.simulate(env.sim)
            env.gym.fetch_results(env.sim, True)
        
        # 刷新状态
        env._refresh_gym()
        
        # Viewer 渲染（仅在非 headless 模式且 viewer 存在时）
        if not args.headless and env.viewer is not None:
            env.gym.step_graphics(env.sim)
            env.gym.draw_viewer(env.viewer, env.sim, True)
            env.gym.sync_frame_time(env.sim)
        
        # ============================================================
        # 更新计数器
        # ============================================================
        env_step_counters += 1
        
        # ============================================================
        # 检测失败/成功条件
        # ============================================================
        object_pos = env.root_state_tensor[env.object_indices, 0:3]
        object_rot = env.root_state_tensor[env.object_indices, 3:7]
        
        # 1. 物体高度偏离超过阈值（上升或下降都算）
        z_deviation = torch.abs(env.init_object_z_buf - object_pos[:, 2])
        failed_z_deviation = torch.greater(z_deviation, env.relative_z_drop_threshold)
        
        # 2. LinkerPen 端点高度差（检测倾倒）
        pencil_end_1 = object_pos + quat_apply(object_rot, pencil_end_offset_neg.unsqueeze(0).expand(env.num_envs, -1))
        pencil_end_2 = object_pos + quat_apply(object_rot, pencil_end_offset_pos.unsqueeze(0).expand(env.num_envs, -1))
        z_diff = torch.abs(pencil_end_1[:, 2] - pencil_end_2[:, 2])
        failed_tilt = torch.greater(z_diff, env.pencil_tilt_threshold)
        
        # 3. 成功：坚持到目标步数
        succeeded = torch.greater_equal(env_step_counters, stability_steps)
        
        # ============================================================
        # 处理成功的环境：保存并重置
        # ============================================================
        success_ids = torch.where(succeeded)[0]
        if len(success_ids) > 0:
            env.save_grasping_state(success_ids)
            success_count += len(success_ids)
            # 重置成功的环境
            env.reset_idx(success_ids)
            env_step_counters[success_ids] = 0
            total_attempts += len(success_ids)
        
        # ============================================================
        # 处理失败的环境：立即重置为新姿态
        # ============================================================
        # 排除已经处理的成功环境
        failed_any = torch.logical_or(failed_z_deviation, failed_tilt)
        failed_any[success_ids] = False  # 不重复处理成功的
        
        failed_ids = torch.where(failed_any)[0]
        if len(failed_ids) > 0:
            # 统计失败原因
            fail_z_drop += int(torch.sum(failed_z_deviation[failed_ids]).item())
            fail_tilt += int(torch.sum(failed_tilt[failed_ids] & ~failed_z_deviation[failed_ids]).item())
            
            # 立即重置失败的环境（即时补充）
            env.reset_idx(failed_ids)
            env_step_counters[failed_ids] = 0
            total_attempts += len(failed_ids)
        
        if step_count % 100 == 0:
            success_rate = success_count / total_attempts * 100 if total_attempts > 0 else 0
            counts = env.saved_pose_counts.cpu().numpy()
            # 计算完成度最低的那个
            min_progress = np.min(counts)
            print(f"[main] 步数: {step_count:5d} | 最小进度: {min_progress}/{env.target_per_pose} | "
                  f"分布: {counts} | "
                  f"成功率: {success_rate:.1f}%")
    
    # ================================================================
    # 完成：输出统计和导出缓存
    # ================================================================
    success_rate = success_count / total_attempts * 100 if total_attempts > 0 else 0
    print(f"\n[main] ========== 生成完成 ==========")
    print(f"[main] 总尝试次数: {total_attempts}")
    print(f"[main] 成功样本数: {success_count}")
    print(f"[main] 成功率: {success_rate:.2f}%")
    print(f"[main] 失败原因统计: z_drop={fail_z_drop}, tilt={fail_tilt}")
    
    # 导出缓存
    cache_filename = f"cache/{env.num_canonical_poses}_{env.target_per_pose * env.num_canonical_poses}_{env.cache_dim}_grasp_cache.npy"
    env.export_cache(cache_filename)
    
    print("[main] 完成!")


if __name__ == "__main__":
    main()
