"""
python cache/init_pose_check.py

初始化姿态成功率检测 - Headless 模式批量验证 grasp_cache 质量

支持三种缓存格式:
- 61维格式 (Flying Hand): [hand_actual(27) + hand_target(27) + obj_pos(3) + obj_rot(4)]
- 49维格式 (非 Flying Hand): [hand_actual(21) + hand_target(21) + obj_pos(3) + obj_rot(4)]
- 34维格式 (旧版): [hand_dof(27) + obj_pos(3) + obj_rot(4)]

用于批量检验抓取缓存中的初始化姿态成功率，输出详细统计信息。
与 init_pose_vis.py 使用相同的检测逻辑，但运行在 headless 模式下，适合大规模验证。

Space E 支持：
- 使用 --alpha 参数指定时间缩放因子（与训练 alpha_start 一致）
- alpha < 1.0 时自动应用 Space E 物理缩放（重力、PD 增益、PhysX 阈值等）
- 确保检测环境与训练环境的物理参数一致
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
    #'cache_file': 'cache/SLIP_3_30000_34_grasp_cache.npy',
    #'cache_file': 'cache/HIT_3_30000_34_grasp_cache.npy',
    # 仿真配置
    'num_envs': 8192,                    # 并行环境数量
    'num_batches': 0,                   # 批次数量，0 表示自动计算以覆盖整个缓存
    
    # 检验配置
    'stability_steps': 200,             # 运行仿真的步数，用于检查稳定性
    
    # 早期终止阈值（与训练配置一致）
    'relative_z_drop_threshold': 0.15,  # 物体高度偏离阈值（米）
    'pencil_tilt_threshold': 0.08,      # LinkerPen倾斜阈值（米）
    
    # Space E 课程学习配置
    'alpha': 1.0,                       # 时间缩放因子 (0.1-1.0)，默认 1.0 (标准物理)
    
    # 随机种子
    'seed': 42,
    
    # 物体配置
    'pen_length': 0.18,                 # LinkerPen长度（米）
    
    # 输出配置
    'verbose': False,                   # 是否输出每个批次的详细信息
}

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='抓取缓存初始化姿态成功率检测（Headless）')
    
    # 缓存文件
    parser.add_argument('--cache-file', type=str, default=DEFAULT_CONFIG['cache_file'],
                       help=f'缓存文件路径，默认: {DEFAULT_CONFIG["cache_file"]}')
    
    # 仿真配置
    parser.add_argument('--num-envs', type=int, default=DEFAULT_CONFIG['num_envs'],
                       help=f'并行环境数量，默认: {DEFAULT_CONFIG["num_envs"]}')
    parser.add_argument('--num-batches', type=int, default=DEFAULT_CONFIG['num_batches'],
                       help=f'批次数量（0=自动覆盖整个缓存），默认: {DEFAULT_CONFIG["num_batches"]}')
    
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
    
    # 输出配置
    parser.add_argument('--verbose', '-v', action='store_true', default=DEFAULT_CONFIG['verbose'],
                       help='输出每个批次的详细信息')
    
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


class InitPoseChecker(LinkerHandHora):
    """
    初始化姿态成功率检测器（Headless）
    
    从 grasp_cache 中加载姿态并批量验证成功率
    
    支持 Space E 时间缩放：使用 --alpha 参数在指定的物理环境下检测
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
        # Space E 课程学习配置：使用指定的 alpha 检测
        # ============================================================
        # 设置 curriculum 配置，让父类 LinkerHandHora 使用正确的物理参数
        # 这确保检测时的物理环境与训练初期一致
        alpha = args.alpha
        if 'curriculum' not in config['env']:
            config['env']['curriculum'] = {}
        config['env']['curriculum']['mode'] = 'SpaceE' if alpha < 1.0 else 'SpaceA'
        config['env']['curriculum']['alpha_start'] = alpha
        config['env']['curriculum']['alpha_end'] = alpha  # 检测时保持固定 alpha
        print(f"[InitPoseChecker] Space E 配置: alpha={alpha:.2f}, mode={config['env']['curriculum']['mode']}")
        
        super().__init__(config, sim_device, graphics_device_id, headless)
        
        # 如果 alpha < 1.0，应用 Space E 物理缩放
        if alpha < 1.0:
            print(f"[InitPoseChecker] 应用 Space E 物理缩放...")
            self.apply_curriculum_physics()
        
        # 缓存数据
        self.cache_data = torch.tensor(cache_data, dtype=torch.float32, device=self.device)
        self.cache_dim = cache_data.shape[1]
        self.num_cached_poses = cache_data.shape[0]
        
        # 检测缓存格式:
        # - 61维 (Flying Hand 新格式): 27*2 + 7 = 61
        # - 49维 (非 Flying Hand 新格式): 21*2 + 7 = 49
        # - 34维 (旧格式 Flying Hand): 27 + 7 = 34
        self.is_new_format = (self.cache_dim == self.num_linker_hand_dofs * 2 + NUM_OBJECT_DIMS)
        if self.is_new_format:
            format_str = f"{self.cache_dim}维(actual+target, DOF={self.num_linker_hand_dofs})"
        else:
            format_str = f"{self.cache_dim}维(旧格式)"
        
        print(f"[InitPoseChecker] 加载缓存: {self.num_cached_poses} 个姿态, 维度: {self.cache_dim} ({format_str})")
        print(f"[InitPoseChecker] Flying Hand: {self.flying_hand_enabled}, DOF: {self.num_linker_hand_dofs}")
        
        # 当前显示的姿态索引
        self.current_pose_indices = None
        
    def reset_with_cache(self, env_ids, pose_indices):
        """
        使用缓存中的姿态重置环境
        
        与 linker_hand_hora.py 中的 _init_object_pose 保持一致的初始化逻辑
        """
        num_envs_to_reset = len(env_ids)
        if num_envs_to_reset == 0:
            return
        
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
    初始化姿态成功率检测主函数（Headless）
    
    从 grasp_cache 中按顺序/随机加载姿态并批量验证成功率
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
    
    # 创建环境（Headless 模式）
    env = InitPoseChecker(
        config=cfg_task,
        sim_device=config.sim_device,
        graphics_device_id=config.graphics_device_id,
        headless=True,  # 强制 headless
        cache_data=cache_data,
    )
    
    print(f"[main] 创建了 {env.num_envs} 个环境 (Headless 模式)")
    print(f"[main] 早期终止阈值: z_drop={env.relative_z_drop_threshold:.3f}m, tilt={env.pencil_tilt_threshold:.3f}m")
    print(f"[main] Space E Alpha: {args.alpha:.2f} (重力缩放: {args.alpha**2:.4f})")
    
    # ================================================================
    # 配置参数
    # ================================================================
    stability_steps = args.stability_steps
    pen_length = args.pen_length
    num_envs = env.num_envs
    num_cached_poses = env.num_cached_poses
    
    # 计算批次数量
    if args.num_batches > 0:
        num_batches = args.num_batches
    else:
        # 自动计算以覆盖整个缓存
        num_batches = int(np.ceil(num_cached_poses / num_envs))
    
    total_samples = min(num_batches * num_envs, num_cached_poses)
    
    print(f"[main] 检测配置:")
    print(f"       - 稳定步数: {stability_steps}")
    print(f"       - 批次数量: {num_batches}")
    print(f"       - 每批环境: {num_envs}")
    print(f"       - 总样本数: {total_samples}")
    
    # 预计算笔端点偏移（本地坐标系）
    pencil_end_offset_neg = torch.tensor([0, 0, -pen_length / 2], device=env.device)
    pencil_end_offset_pos = torch.tensor([0, 0, pen_length / 2], device=env.device)
    
    # ================================================================
    # 统计信息
    # ================================================================
    total_tested = 0
    total_success = 0
    total_fail_z_drop = 0
    total_fail_tilt = 0
    
    # 失败的姿态索引记录
    failed_pose_indices = []
    
    # 用于记录每个姿态首次失败的步数
    first_fail_steps = []
    
    all_env_ids = torch.arange(env.num_envs, device=env.device)
    
    print(f"\n[main] 开始批量检测...")
    print("=" * 70)
    
    for batch_idx in range(num_batches):
        # 计算本批次的姿态索引
        start_idx = batch_idx * num_envs
        end_idx = min(start_idx + num_envs, num_cached_poses)
        batch_size = end_idx - start_idx
        
        if batch_size == 0:
            break
        
        # 生成姿态索引
        pose_indices = torch.arange(start_idx, end_idx, device=env.device)
        
        # 如果批次不满，用随机索引填充（但不计入统计）
        if batch_size < num_envs:
            padding_indices = torch.randint(0, num_cached_poses, (num_envs - batch_size,), device=env.device)
            pose_indices = torch.cat([pose_indices, padding_indices])
        
        # 重置环境
        env.reset_with_cache(all_env_ids, pose_indices)
        
        # 状态跟踪
        env_step_counters = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        failed_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        first_fail_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        
        # 运行仿真
        for step in range(stability_steps):
            # 物理仿真步进（零动作，保持姿态）
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
            
            # 更新计数器
            env_step_counters += 1
            
            # 检测失败条件
            object_pos = env.root_state_tensor[env.object_indices, 0:3]
            object_rot = env.root_state_tensor[env.object_indices, 3:7]
            
            # 1. 物体高度偏离超过阈值
            z_deviation = torch.abs(env.init_object_z_buf - object_pos[:, 2])
            failed_z = torch.greater(z_deviation, env.relative_z_drop_threshold)
            
            # 2. LinkerPen 端点高度差（检测倾倒）
            pencil_end_1 = object_pos + quat_apply(object_rot, pencil_end_offset_neg.unsqueeze(0).expand(env.num_envs, -1))
            pencil_end_2 = object_pos + quat_apply(object_rot, pencil_end_offset_pos.unsqueeze(0).expand(env.num_envs, -1))
            z_diff = torch.abs(pencil_end_1[:, 2] - pencil_end_2[:, 2])
            failed_tilt = torch.greater(z_diff, env.pencil_tilt_threshold)
            
            # 记录首次失败步数
            newly_failed = (failed_z | failed_tilt) & ~failed_mask
            first_fail_step[newly_failed] = step + 1
            
            # 更新失败 mask
            failed_mask |= failed_z | failed_tilt
        
        # 统计本批次结果（只统计有效样本）
        valid_mask = torch.arange(env.num_envs, device=env.device) < batch_size
        valid_failed = failed_mask[:batch_size]
        valid_success = ~valid_failed
        
        batch_success = int(valid_success.sum().item())
        batch_fail = int(valid_failed.sum().item())
        
        # 统计失败原因
        batch_fail_z = 0
        batch_fail_tilt = 0
        for i in range(batch_size):
            if failed_mask[i]:
                # 检查最终状态的失败原因
                if failed_z[i]:
                    batch_fail_z += 1
                elif failed_tilt[i]:
                    batch_fail_tilt += 1
        
        # 记录失败的姿态索引
        failed_in_batch = torch.where(valid_failed)[0]
        for idx in failed_in_batch:
            failed_pose_indices.append(pose_indices[idx].item())
            first_fail_steps.append(first_fail_step[idx].item())
        
        # 累计统计
        total_tested += batch_size
        total_success += batch_success
        total_fail_z_drop += batch_fail_z
        total_fail_tilt += batch_fail_tilt
        
        # 输出批次信息
        batch_success_rate = batch_success / batch_size * 100
        if args.verbose:
            print(f"[Batch {batch_idx+1:3d}/{num_batches}] "
                  f"样本 {start_idx:5d}-{end_idx-1:5d} | "
                  f"成功: {batch_success:4d}/{batch_size} ({batch_success_rate:5.1f}%) | "
                  f"失败: z_drop={batch_fail_z}, tilt={batch_fail_tilt}")
        else:
            # 简洁进度输出
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0 or batch_idx == num_batches - 1:
                current_rate = total_success / total_tested * 100 if total_tested > 0 else 0
                print(f"[Progress] {total_tested:5d}/{total_samples} ({total_tested/total_samples*100:5.1f}%) | "
                      f"累计成功率: {current_rate:5.1f}%")
    
    # ================================================================
    # 输出最终统计
    # ================================================================
    print("=" * 70)
    print(f"\n[main] ========== 检测完成 ==========")
    print(f"[main] 缓存文件: {cache_file}")
    print(f"[main] 检测参数:")
    print(f"       - 稳定步数: {stability_steps}")
    print(f"       - z_drop 阈值: {args.relative_z_drop_threshold:.3f}m")
    print(f"       - tilt 阈值: {args.pencil_tilt_threshold:.3f}m")
    print(f"       - Space E Alpha: {args.alpha:.2f}")
    print()
    
    success_rate = total_success / total_tested * 100 if total_tested > 0 else 0
    print(f"[结果] 总测试数: {total_tested}")
    print(f"[结果] 成功数量: {total_success}")
    print(f"[结果] 失败数量: {total_tested - total_success}")
    print(f"[结果] 成功率:   {success_rate:.2f}%")
    print()
    print(f"[失败原因分析]")
    print(f"       - 高度偏离 (z_drop): {total_fail_z_drop} ({total_fail_z_drop/total_tested*100:.1f}%)")
    print(f"       - 倾倒 (tilt):       {total_fail_tilt} ({total_fail_tilt/total_tested*100:.1f}%)")
    
    # 失败步数分析
    if first_fail_steps:
        fail_steps_arr = np.array(first_fail_steps)
        print()
        print(f"[失败时机分析]")
        print(f"       - 首步失败 (step 1):    {np.sum(fail_steps_arr == 1)} ({np.mean(fail_steps_arr == 1)*100:.1f}%)")
        print(f"       - 前10步失败:           {np.sum(fail_steps_arr <= 10)} ({np.mean(fail_steps_arr <= 10)*100:.1f}%)")
        print(f"       - 平均失败步数:         {np.mean(fail_steps_arr):.1f}")
        print(f"       - 中位数失败步数:       {np.median(fail_steps_arr):.1f}")
    
    # 输出部分失败索引（用于调试）
    if failed_pose_indices and args.verbose:
        print()
        print(f"[失败姿态索引 (前20个)]")
        sample_indices = failed_pose_indices[:20]
        sample_steps = first_fail_steps[:20]
        for idx, step in zip(sample_indices, sample_steps):
            print(f"       - 索引 {idx:5d}, 失败步数: {step}")
    
    print()
    print("[main] 完成!")
    
    # 返回结果供程序化调用
    return {
        'total_tested': total_tested,
        'total_success': total_success,
        'success_rate': success_rate,
        'fail_z_drop': total_fail_z_drop,
        'fail_tilt': total_fail_tilt,
        'failed_indices': failed_pose_indices,
    }


if __name__ == "__main__":
    main()
