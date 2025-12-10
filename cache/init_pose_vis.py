"""
python cache/init_pose_vis.py
python cache/init_pose_vis.py --alpha 0.3  # 在 alpha=0.3 的物理环境下可视化

初始化姿态可视化 - 读取 grasp_cache 并在 Viewer 中显示初始化姿态

支持三种缓存格式:
- 61维格式 (Flying Hand): [hand_actual(27) + hand_target(27) + obj_pos(3) + obj_rot(4)]
- 49维格式 (非 Flying Hand): [hand_actual(21) + hand_target(21) + obj_pos(3) + obj_rot(4)]
- 34维格式 (旧版): [hand_dof(27) + obj_pos(3) + obj_rot(4)]

用于检验抓取缓存中的初始化姿态是否会因为高度差或碰撞导致第一帧仿真失败。

Space E 支持：
- 使用 --alpha 参数指定时间缩放因子（与训练 alpha_start 一致）
- alpha < 1.0 时自动应用 Space E 物理缩放（重力、PD 增益、PhysX 阈值等）
- 确保可视化环境与训练环境的物理参数一致
"""

# 添加项目根目录到Python路径
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 命令行参数解析
import argparse

# 重要参数配置 - 可通过命令行覆盖
DEFAULT_CONFIG = {
    # 缓存文件配置
    'cache_file': 'cache/3_30000_49_nofly_grasp_cache.npy',  # 默认缓存文件路径 (非 Flying Hand)
    
    # 仿真配置
    'headless': False,                  # 默认显示GUI（用于可视化）
    'num_envs': 8,                      # 显示的环境数量
    
    # 检验配置
    'stability_steps': 200,             # 运行仿真的步数，用于检查稳定性
    
    # 早期终止阈值（与训练配置一致）
    'relative_z_drop_threshold': 0.15,  # 物体高度偏离阈值（米）
    'pencil_tilt_threshold': 0.08,      # LinkerPen倾斜阈值（米）
    
    # Space E 课程学习配置
    'alpha': 0.3,                       # 时间缩放因子 (0.1-1.0)，默认 1.0 (标准物理)
    
    # 随机种子
    'seed': 42,
    
    # 物体配置
    'pen_length': 0.18,                 # LinkerPen长度（米）
}

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='抓取缓存初始化姿态可视化工具')
    
    # 缓存文件
    parser.add_argument('--cache-file', type=str, default=DEFAULT_CONFIG['cache_file'],
                       help=f'缓存文件路径，默认: {DEFAULT_CONFIG["cache_file"]}')
    
    # 仿真配置
    parser.add_argument('--headless', action='store_true', default=DEFAULT_CONFIG['headless'],
                       help='启用无头模式（无GUI）')
    parser.add_argument('--num-envs', type=int, default=DEFAULT_CONFIG['num_envs'],
                       help=f'显示的环境数量，默认: {DEFAULT_CONFIG["num_envs"]}')
    
    # 检验配置
    parser.add_argument('--stability-steps', type=int, default=DEFAULT_CONFIG['stability_steps'],
                       help=f'运行仿真的步数，默认: {DEFAULT_CONFIG["stability_steps"]}')
    
    # 早期终止阈值
    parser.add_argument('--relative-z-drop-threshold', type=float, 
                       default=DEFAULT_CONFIG['relative_z_drop_threshold'],
                       help=f'物体高度偏离阈值（米），默认: {DEFAULT_CONFIG["relative_z_drop_threshold"]}')
    parser.add_argument('--pencil-tilt-threshold', type=float,
                       default=DEFAULT_CONFIG['pencil_tilt_threshold'],
                       help=f'LinkerPen倾斜阈值（米），默认: {DEFAULT_CONFIG["pencil_tilt_threshold"]}')
    
    # Space E 配置
    parser.add_argument('--alpha', type=float, default=DEFAULT_CONFIG['alpha'],
                       help=f'Space E 时间缩放因子 (0.1-1.0)，默认: {DEFAULT_CONFIG["alpha"]}')
    
    # 随机种子
    parser.add_argument('--seed', type=int, default=DEFAULT_CONFIG['seed'],
                       help=f'随机种子，默认: {DEFAULT_CONFIG["seed"]}')
    
    # 物体配置
    parser.add_argument('--pen-length', type=float, default=DEFAULT_CONFIG['pen_length'],
                       help=f'LinkerPen长度（米），默认: {DEFAULT_CONFIG["pen_length"]}')
    
    args = parser.parse_args()
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
    NUM_DOF,
    NUM_TOTAL_DOF_FLYING,
    FLYING_DOF_INDICES,
)

# 物体状态维度: 3 位置 + 4 旋转 = 7
NUM_OBJECT_DIMS = 7


class InitPoseVisualizer(LinkerHandHora):
    """
    初始化姿态可视化器
    
    从 grasp_cache 中随机加载姿态并在 Viewer 中显示
    
    支持 Space E 时间缩放：使用 --alpha 参数在指定的物理环境下可视化
    """
    
    def __init__(self, config, sim_device, graphics_device_id, headless, cache_data):
        # 修改配置
        config['env']['genGrasps'] = True  # 跳过默认缓存加载
        config['env']['randomization']['randomizeScale'] = False
        config['env']['randomization']['scaleListInit'] = False
        config['env']['numEnvs'] = args.num_envs
        
        # 禁用手指禁用功能以避免动作维度不匹配
        if 'actionSpace' in config['env']:
            config['env']['actionSpace']['disableRingLittleFinger'] = False
        
        # ============================================================
        # Space E 课程学习配置：使用指定的 alpha 可视化
        # ============================================================
        # 设置 curriculum 配置，让父类 LinkerHandHora 使用正确的物理参数
        # 这确保可视化时的物理环境与训练初期一致
        alpha = args.alpha
        if 'curriculum' not in config['env']:
            config['env']['curriculum'] = {}
        config['env']['curriculum']['mode'] = 'SpaceE' if alpha < 1.0 else 'SpaceA'
        config['env']['curriculum']['alpha_start'] = alpha
        config['env']['curriculum']['alpha_end'] = alpha  # 可视化时保持固定 alpha
        print(f"[InitPoseVisualizer] Space E 配置: alpha={alpha:.2f}, mode={config['env']['curriculum']['mode']}")
        
        super().__init__(config, sim_device, graphics_device_id, headless)
        
        # 如果 alpha < 1.0，应用 Space E 物理缩放
        if alpha < 1.0:
            print(f"[InitPoseVisualizer] 应用 Space E 物理缩放...")
            self.apply_curriculum_physics()
        
        # 缓存数据
        self.cache_data = torch.tensor(cache_data, dtype=torch.float32, device=self.device)
        self.cache_dim = cache_data.shape[1]
        self.num_cached_poses = cache_data.shape[0]
        
        # 检测缓存格式:
        # - 61维 (Flying Hand 新格式): 27*2 + 7 = 61
        # - 49维 (非 Flying Hand 新格式): 21*2 + 7 = 49
        # - 34维 (旧格式 Flying Hand): 27 + 7 = 34
        # - 28维 (旧格式 非 Flying Hand): 21 + 7 = 28
        self.is_new_format = (self.cache_dim == self.num_linker_hand_dofs * 2 + NUM_OBJECT_DIMS)
        if self.is_new_format:
            format_str = f"{self.cache_dim}维(actual+target, DOF={self.num_linker_hand_dofs})"
        else:
            format_str = f"{self.cache_dim}维(旧格式)"
        
        print(f"[InitPoseVisualizer] 加载缓存: {self.num_cached_poses} 个姿态, 维度: {self.cache_dim} ({format_str})")
        print(f"[InitPoseVisualizer] Flying Hand: {self.flying_hand_enabled}, DOF: {self.num_linker_hand_dofs}")
        
        # 当前显示的姿态索引
        self.current_pose_indices = None
        
    def reset_with_cache(self, env_ids, pose_indices=None):
        """
        使用缓存中的姿态重置环境
        
        与 linker_hand_hora.py 中的 _init_object_pose 保持一致的初始化逻辑
        """
        num_envs_to_reset = len(env_ids)
        if num_envs_to_reset == 0:
            return
        
        # 选择姿态
        if pose_indices is None:
            pose_indices = torch.randint(0, self.num_cached_poses, (num_envs_to_reset,), device=self.device)
        
        self.current_pose_indices = pose_indices
        selected_poses = self.cache_data[pose_indices]  # [num_envs, cache_dim]
        
        # 解析缓存数据（支持新旧两种格式，以及 Flying / 非 Flying Hand）
        hand_dof_dim = self.num_linker_hand_dofs  # 27 (Flying) 或 21 (非 Flying)
        
        if self.is_new_format:
            # 新格式: [hand_actual(N) + hand_target(N) + obj_pos(3) + obj_rot(4)]
            # N = 27 (Flying) 或 21 (非 Flying)
            hand_actual = selected_poses[:, :hand_dof_dim].clone()
            hand_target = selected_poses[:, hand_dof_dim:hand_dof_dim*2].clone()
            object_pos = selected_poses[:, hand_dof_dim*2:hand_dof_dim*2+3].clone()
            object_rot = selected_poses[:, hand_dof_dim*2+3:].clone()
        else:
            # 旧格式: [hand_dof(N) + obj_pos(3) + obj_rot(4)]
            hand_actual = selected_poses[:, :hand_dof_dim].clone()
            hand_target = hand_actual.clone()  # 旧格式：actual = target
            object_pos = selected_poses[:, hand_dof_dim:hand_dof_dim+3].clone()
            object_rot = selected_poses[:, hand_dof_dim+3:].clone()
        
        # ====================================================================
        # 应用手部 DOF 状态（区分物理位置和控制目标）
        # ====================================================================
        # 物理位置: 使用 actual（避免穿模爆炸）
        self.linker_hand_dof_pos[env_ids, :hand_dof_dim] = hand_actual
        self.linker_hand_dof_vel[env_ids, :] = 0.0
        
        # 控制目标: 使用 target（产生抓握力）
        self.prev_targets[env_ids, :hand_dof_dim] = hand_target
        self.cur_targets[env_ids, :hand_dof_dim] = hand_target
        self.init_pose_buf[env_ids, :hand_dof_dim] = hand_target
        
        # ====================================================================
        # 应用物体状态
        # ====================================================================
        object_indices = self.object_indices[env_ids]
        self.root_state_tensor[object_indices, 0:3] = object_pos
        self.root_state_tensor[object_indices, 3:7] = object_rot
        self.root_state_tensor[object_indices, 7:13] = 0.0  # 清零速度
        
        # 记录初始物体高度（用于稳定性检查）
        self.init_object_z_buf[env_ids] = object_pos[:, 2]
        
        # ====================================================================
        # 更新仿真状态（与 _init_object_pose 逻辑一致）
        # ====================================================================
        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        object_indices = object_indices.to(torch.int32)
        
        self.gym.set_dof_state_tensor_indexed(
            self.sim, 
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(hand_indices), 
            len(hand_indices)
        )
        
        # 设置 PD 控制器目标位置（GPU pipeline 必须）
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
        
        return pose_indices


@hydra.main(version_base="1.2", config_path="../configs", config_name="config")
def main(config):
    """
    初始化姿态可视化主函数
    
    从 grasp_cache 中随机加载姿态并运行仿真检查稳定性
    """
    from penspin.utils.misc import set_seed
    from penspin.utils.reformat import omegaconf_to_dict
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载缓存数据
    cache_file = args.cache_file
    if not Path(cache_file).exists():
        print(f"[Error] 缓存文件不存在: {cache_file}")
        return
    
    cache_data = np.load(cache_file)
    print(f"[main] 加载缓存文件: {cache_file}")
    print(f"[main] 缓存形状: {cache_data.shape}")
    
    # 转换配置
    cfg_task = omegaconf_to_dict(config.task)
    
    # 使用命令行参数覆盖配置（与 yaml 配置中的阈值一致）
    cfg_task['env']['relative_z_drop_threshold'] = args.relative_z_drop_threshold
    cfg_task['env']['pencil_tilt_threshold'] = args.pencil_tilt_threshold
    cfg_task['env']['numEnvs'] = args.num_envs
    
    # 创建环境
    env = InitPoseVisualizer(
        config=cfg_task,
        sim_device=config.sim_device,
        graphics_device_id=config.graphics_device_id,
        headless=args.headless,
        cache_data=cache_data,
    )
    
    print(f"[main] 创建了 {env.num_envs} 个环境")
    print(f"[main] 早期终止阈值: z_drop={env.relative_z_drop_threshold:.3f}m, tilt={env.pencil_tilt_threshold:.3f}m")
    print(f"[main] Space E Alpha: {args.alpha:.2f} (重力缩放: {args.alpha**2:.4f})")
    
    # ================================================================
    # 配置参数
    # ================================================================
    stability_steps = args.stability_steps
    pen_length = args.pen_length
    
    # 预计算笔端点偏移（本地坐标系）
    pencil_end_offset_neg = torch.tensor([0, 0, -pen_length / 2], device=env.device)
    pencil_end_offset_pos = torch.tensor([0, 0, pen_length / 2], device=env.device)
    
    # ================================================================
    # 初始化：重置所有环境到随机缓存姿态
    # ================================================================
    all_env_ids = torch.arange(env.num_envs, device=env.device)
    pose_indices = env.reset_with_cache(all_env_ids)
    
    print(f"\n[main] 已加载 {env.num_envs} 个随机姿态")
    print(f"[main] 姿态索引: {pose_indices.cpu().numpy()}")
    print(f"[main] 开始运行 {stability_steps} 步仿真检查稳定性...")
    print(f"[main] 按 ESC 或关闭窗口退出\n")
    
    # ================================================================
    # 状态跟踪
    # ================================================================
    env_step_counters = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    fail_counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    step_count = 0
    while step_count < stability_steps:
        step_count += 1
        
        # 检查 Viewer 是否关闭
        if not args.headless and env.viewer is not None:
            if env.gym.query_viewer_has_closed(env.viewer):
                print("[main] Viewer 已关闭，退出...")
                break
        
        # ============================================================
        # 物理仿真步进（零动作，保持姿态）
        # ============================================================
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
        
        # Viewer 渲染
        if not args.headless and env.viewer is not None:
            env.gym.step_graphics(env.sim)
            env.gym.draw_viewer(env.viewer, env.sim, True)
            env.gym.sync_frame_time(env.sim)
        
        # ============================================================
        # 更新计数器
        # ============================================================
        env_step_counters += 1
        
        # ============================================================
        # 检测失败条件（与训练时相同的逻辑）
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
        
        # 记录失败
        failed_any = torch.logical_or(failed_z_deviation, failed_tilt)
        fail_counts += failed_any.long()
        
        # ============================================================
        # 输出进度
        # ============================================================
        if step_count % 50 == 0 or step_count == 1:
            num_failed = (fail_counts > 0).sum().item()
            num_stable = env.num_envs - num_failed
            print(f"[main] 步数: {step_count:4d}/{stability_steps} | "
                  f"稳定: {num_stable}/{env.num_envs} | "
                  f"失败: {num_failed} (z_dev: {failed_z_deviation.sum().item()}, tilt: {failed_tilt.sum().item()})")
    
    # ================================================================
    # 完成：输出统计
    # ================================================================
    print(f"\n[main] ========== 检验完成 ==========")
    num_failed_total = (fail_counts > 0).sum().item()
    num_stable_total = env.num_envs - num_failed_total
    print(f"[main] 稳定姿态数: {num_stable_total}/{env.num_envs}")
    print(f"[main] 失败姿态数: {num_failed_total}/{env.num_envs}")
    
    if num_failed_total > 0:
        failed_indices = torch.where(fail_counts > 0)[0]
        failed_pose_ids = pose_indices[failed_indices]
        print(f"[main] 失败的缓存索引: {failed_pose_ids.cpu().numpy()}")
    
    print("[main] 完成!")


if __name__ == "__main__":
    main()
