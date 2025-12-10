"""
Trajectory Recorder for Curriculum Learning Visualization

在课程学习更新 (curriculum update) 或周期性触发时，录制完整的成功轨迹并导出为视频。

设计要点：
1. **跨 rollout 追踪**: horizon_length 可能小于一个完整 episode，
   需要跨多个 play_steps() 调用追踪同一环境的完整轨迹
2. **选择性录制**: 只录制少数环境以节省显存/计算资源
3. **成功过滤**: 只保存成功的轨迹（rot_angle > threshold）
4. **Top-K 优胜劣汰**: 使用优先级队列只保留旋转角度最高的 K 条轨迹
5. **双触发模式**: 支持 "课程更新触发" 和 "周期性触发" 两种导出方式
6. **相机复用**: 利用 PPO 已有的 GIF 录制帧，而非独立开启相机

使用方法:
    recorder = TrajectoryRecorder(env, config)
    
    # 在 play_steps() 的每一步中（当相机已经启用时）
    recorder.record_step(dones, infos, frame=captured_frame)
    
    # 在 curriculum update 后
    if needs_update:
        recorder.export_on_curriculum_update(writer, agent_steps, alpha)
    
    # 周期性导出 (例如每 100 epoch)
    if epoch % 100 == 0:
        recorder.export(writer, agent_steps, alpha, tag_prefix="periodic")
"""

import os
import numpy as np
import torch
import heapq  # 用于维护 Top-K 优先级队列
from typing import Dict, List, Optional, Tuple


class TrajectoryRecorder:
    """
    轨迹录制器 (Top-K 优胜劣汰模式)
    
    在并行环境中追踪指定环境的完整轨迹，用于课程学习可视化。
    
    关键设计：
    - 不主动开启相机（避免额外开销）
    - 利用 PPO 已捕获的帧进行复用
    - 追踪 rot_angle 累计值来判断成功
    - 使用优先级队列保留最佳的 K 条轨迹（而非最近的 K 条）
    """
    
    def __init__(
        self,
        env,
        num_record_envs: int = 4,
        max_episode_length: int = 1000,
        success_threshold: float = 3.0,  # 成功判定：旋转角度阈值（弧度）
        min_trajectories_to_export: int = 1,  # 只要有一条好轨迹就可以导出
        keep_best_k: int = 5,  # 在缓冲区中只保留最好的 K 条
        device: str = 'cuda:0'
    ):
        """
        初始化轨迹录制器 (Top-K 模式)
        
        Args:
            env: IsaacGym 环境实例
            num_record_envs: 追踪的环境数量
            max_episode_length: 单个 episode 的最大长度
            success_threshold: 成功判定的旋转角度阈值（弧度）
            min_trajectories_to_export: 导出前需要的最小成功轨迹数
            keep_best_k: 缓冲区中保留的最佳轨迹数量
            device: PyTorch 设备
        """
        self.env = env
        self.num_record_envs = num_record_envs
        self.max_episode_length = max_episode_length
        self.success_threshold = success_threshold
        self.min_trajectories_to_export = min_trajectories_to_export
        self.keep_best_k = keep_best_k
        self.device = device
        
        # 选择要录制的环境索引 (选择前 num_record_envs 个环境)
        self.record_env_ids = list(range(min(num_record_envs, env.num_envs)))
        
        # 每个录制环境的帧缓冲 {env_id: List[frame]}
        self.frame_buffers: Dict[int, List[np.ndarray]] = {i: [] for i in self.record_env_ids}
        
        # 每个录制环境的累计旋转角度 {env_id: float}
        self.cumulative_rot_angles: Dict[int, float] = {i: 0.0 for i in self.record_env_ids}
        
        # [改进] Top-K 优先级队列，存储最佳轨迹
        # 元素格式: (rot_angle, env_id, frames)
        # 注意：heapq 是最小堆，rot_angle 放首位方便比较
        self.best_trajectories: List[Tuple[float, int, List[np.ndarray]]] = []
        
        # 录制状态
        self.is_recording = True
        self.export_count = 0  # 统一的导出计数器
        self.curriculum_update_count = 0  # 保持兼容
        
        # 统计信息
        self.total_recorded_episodes = 0
        self.successful_episodes = 0
        
        print(f"\n[TrajectoryRecorder] 初始化完成 (Top-{keep_best_k} 模式)")
        print(f"  录制环境数: {len(self.record_env_ids)}")
        print(f"  最大 episode 长度: {max_episode_length}")
        print(f"  成功阈值: {success_threshold:.2f} rad")
        print(f"  保留最佳轨迹数: {keep_best_k}")
    
    def record_step(
        self,
        dones: torch.Tensor,
        infos: Dict,
        frame: Optional[np.ndarray] = None
    ):
        """
        在每个 step 后调用，记录帧和状态
        
        Args:
            dones: (num_envs,) done 标志
            infos: 环境返回的 info 字典，包含 'rot_angle'
            frame: 可选的已捕获帧（来自 PPO 的 GIF 录制）
        """
        if not self.is_recording:
            return
        
        # 更新累计旋转角度
        if 'rot_angle' in infos:
            rot_angles = infos['rot_angle']
            for env_id in self.record_env_ids:
                self.cumulative_rot_angles[env_id] += rot_angles[env_id].item()
        
        # 如果提供了帧，存储它（只在 episode 未超长时记录，防止内存溢出）
        if frame is not None:
            for env_id in self.record_env_ids:
                if len(self.frame_buffers[env_id]) < self.max_episode_length:
                    self.frame_buffers[env_id].append(frame.copy())
        
        # 检查哪些录制环境的 episode 结束
        for env_id in self.record_env_ids:
            if dones[env_id].item():
                self._handle_episode_done(env_id)
    
    def _handle_episode_done(self, env_id: int):
        """
        处理 episode 结束：使用 Top-K 逻辑保留最好的轨迹
        
        Args:
            env_id: 结束的环境 ID
        """
        self.total_recorded_episodes += 1
        
        frames = self.frame_buffers[env_id]
        rot_angle = self.cumulative_rot_angles[env_id]
        
        # 判断是否成功（基于 rot_angle 阈值）
        # 注意：如果没有帧数据，仍然记录成功但不存储轨迹
        if rot_angle >= self.success_threshold:
            self.successful_episodes += 1
            
            # 只有当有足够的帧时才存储轨迹
            if len(frames) > 10:
                # === Top-K 核心逻辑 ===
                new_record = (rot_angle, env_id, frames.copy())
                
                # 1. 如果缓冲区没满，直接添加
                if len(self.best_trajectories) < self.keep_best_k:
                    heapq.heappush(self.best_trajectories, new_record)
                
                # 2. 如果缓冲区已满，检查是否比最差的好
                else:
                    # 堆顶是最小值 (rot_angle 最小的)
                    min_stored_angle = self.best_trajectories[0][0]
                    if rot_angle > min_stored_angle:
                        # 替换掉最差的
                        heapq.heapreplace(self.best_trajectories, new_record)
        
        # 重置该环境的缓冲
        self.frame_buffers[env_id] = []
        self.cumulative_rot_angles[env_id] = 0.0
    
    def export(
        self,
        writer,
        agent_steps: int,
        current_alpha: float,
        tag_prefix: str = "periodic",
        output_dir: str = None
    ) -> bool:
        """
        通用导出函数 - 支持课程更新和周期性导出两种模式
        
        Args:
            writer: TensorBoard SummaryWriter
            agent_steps: 当前 agent 步数
            current_alpha: 当前课程 alpha 值
            tag_prefix: 标签前缀，如 'curriculum' 或 'periodic'
            output_dir: 可选的输出目录（用于保存 .mp4 文件）
            
        Returns:
            bool: 是否成功导出
        """
        # 打印诊断信息
        print(f"\n[TrajectoryRecorder] Export ({tag_prefix}) 尝试导出 @ step={agent_steps}")
        print(f"  统计: 录制={self.total_recorded_episodes} 成功={self.successful_episodes} 缓冲={len(self.best_trajectories)}")
        
        if len(self.best_trajectories) < self.min_trajectories_to_export:
            print(f"  ❌ 跳过: 缓冲轨迹不足 {len(self.best_trajectories)}/{self.min_trajectories_to_export}")
            if self.total_recorded_episodes > 0 and self.successful_episodes > 0 and len(self.best_trajectories) == 0:
                print(f"  ⚠️ 原因: 有成功 episode 但没有帧数据（相机可能未启用）")
                print(f"     解决方案: 在启动命令中添加 task.env.enableCameraSensors=True")
            return False
        
        self.export_count += 1
        
        # 从堆中取出并排序 (从大到小)
        sorted_trajectories = sorted(
            self.best_trajectories,
            key=lambda x: x[0],  # 按 rot_angle 排序
            reverse=True
        )
        
        print(f"  ✓ 导出 {len(sorted_trajectories)} 条最佳轨迹")
        
        exported_count = 0
        # 导出前 3 名到 TensorBoard
        for i, (rot_angle, env_id, frames) in enumerate(sorted_trajectories[:3]):
            if len(frames) < 10:
                print(f"  >> 轨迹 Rank {i+1}: 跳过（帧数不足 {len(frames)}）")
                continue
            
            try:
                # 转换为 TensorBoard 视频格式: (N, T, H, W, C) -> (N, T, C, H, W)
                frame_array = np.array(frames)  # (T, H, W, C)
                
                # 可能需要调整帧顺序或格式
                if len(frame_array.shape) == 4:
                    # 添加 batch 维度
                    frame_array = frame_array[np.newaxis, ...]  # (1, T, H, W, C)
                    
                    # 标签格式: periodic_trajectory/step_1000_alpha_0.20_rank1
                    tag = f'{tag_prefix}_trajectory/step_{agent_steps}_alpha_{current_alpha:.2f}_rank{i+1}'
                    writer.add_video(
                        tag,
                        frame_array,
                        global_step=agent_steps,
                        dataformats='NTHWC',
                        fps=20
                    )
                    exported_count += 1
                    print(f"  >> 导出 TensorBoard 视频 Rank {i+1}: {rot_angle:.2f} rad, {len(frames)} 帧")
                    print(f"     Tag: {tag}")
            except Exception as e:
                print(f"  >> 导出失败 Rank {i+1}: {e}")
        
        if exported_count > 0:
            writer.flush()
            print(f"  ✓ 成功导出 {exported_count} 个视频到 TensorBoard")
        
        # 可选：保存到文件
        if output_dir is not None:
            self._save_to_file(sorted_trajectories[:3], output_dir, agent_steps, current_alpha, tag_prefix)
        
        # === 关键 ===
        # 导出后清空缓冲区，以便下一个周期收集新的数据
        self.best_trajectories = []
        
        return exported_count > 0

    def export_on_curriculum_update(
        self,
        writer,
        agent_steps: int,
        current_alpha: float,
        output_dir: str = None
    ) -> bool:
        """
        在课程更新时导出轨迹视频 (兼容旧接口)
        
        Args:
            writer: TensorBoard SummaryWriter
            agent_steps: 当前 agent 步数
            current_alpha: 当前课程 alpha 值
            output_dir: 可选的输出目录（用于保存 .mp4 文件）
            
        Returns:
            bool: 是否成功导出
        """
        self.curriculum_update_count += 1
        return self.export(
            writer=writer,
            agent_steps=agent_steps,
            current_alpha=current_alpha,
            tag_prefix="curriculum",
            output_dir=output_dir
        )
    
    def _save_to_file(
        self,
        trajectories: List[Tuple[float, int, List[np.ndarray]]],
        output_dir: str,
        agent_steps: int,
        current_alpha: float,
        prefix: str = "curriculum"
    ):
        """
        将轨迹保存为 MP4 文件
        
        Args:
            trajectories: 轨迹列表 [(rot_angle, env_id, frames), ...]
            output_dir: 输出目录
            agent_steps: agent 步数
            current_alpha: 当前 alpha
            prefix: 文件名前缀
        """
        try:
            import imageio
            
            video_dir = os.path.join(output_dir, 'videos')
            os.makedirs(video_dir, exist_ok=True)
            
            saved_count = 0
            for i, (rot_angle, env_id, frames) in enumerate(trajectories):
                if len(frames) < 10:
                    continue
                
                filename = f'{prefix}_step_{agent_steps:09d}_alpha_{current_alpha:.2f}_rank{i+1}_rot{rot_angle:.2f}.mp4'
                filepath = os.path.join(video_dir, filename)
                
                # 保存为 MP4
                imageio.mimsave(filepath, frames, fps=20)
                saved_count += 1
                print(f"  >> 保存 MP4 文件: {filepath}")
            
            if saved_count > 0:
                print(f"  ✓ 成功保存 {saved_count} 个视频到 {video_dir}")
                
        except ImportError:
            print("  ⚠️ 需要安装 imageio 库来保存 MP4 文件: pip install imageio imageio-ffmpeg")
        except Exception as e:
            print(f"  ❌ 保存文件失败: {e}")
    
    def get_stats(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            Dict: 包含统计信息的字典
        """
        return {
            'total_recorded_episodes': self.total_recorded_episodes,
            'successful_episodes': self.successful_episodes,
            'pending_trajectories': len(self.best_trajectories),
            'best_rot_angle': max([t[0] for t in self.best_trajectories]) if self.best_trajectories else 0.0,
            'export_count': self.export_count,
            'curriculum_updates': self.curriculum_update_count,
            'success_rate': self.successful_episodes / max(self.total_recorded_episodes, 1)
        }
    
    def enable_recording(self):
        """启用录制"""
        self.is_recording = True
    
    def disable_recording(self):
        """禁用录制"""
        self.is_recording = False
    
    def reset(self):
        """重置所有缓冲区和统计"""
        for env_id in self.record_env_ids:
            self.frame_buffers[env_id] = []
            self.cumulative_rot_angles[env_id] = 0.0
        self.best_trajectories = []
        self.total_recorded_episodes = 0
        self.successful_episodes = 0
