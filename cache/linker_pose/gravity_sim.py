import isaacgym
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import to_torch, tensor_clamp, quat_from_euler_xyz

import torch
import numpy as np
import time
import os
import math

# --- 配置区域 ---
# 在这里修改手部和物体的姿态、重力参数和其他属性进行调试
DEBUG_CONFIG = {
    # --- 资源路径 ---
    "hand_urdf_path": "../assets/linker_hand/L25_dof_urdf.urdf",
    "object_urdf_path": "../assets/cylinder/pencil-5-7/0000.urdf",
        
    # --- 手部初始旋转 (让手躺下) ---
    # !! 修改这里的值来设置手的初始基座旋转 (绕 Y 轴旋转的角度，单位：度) !!
    "hand_base_rotate_y_deg": -75.0,
    # --- 手部姿态 (21 个自由度) ---
    "hand_dof_targets_rad": [0.0500, -1.3888, 0.0000, -0.7288, -0.1611, -1.3003, -0.3180, -0.4468, 0.0322, -1.4130, -0.2456, -0.3905, -0.1203, -1.5177, -0.0121, -0.5515, -0.2753, -1.4043, -0.1731, -0.5193, -0.4388],
    # !! 修改这里的值来设置物体的初始位置 [x, y, z] (米) !!
    "object_initial_pos": [-0.10570000112056732, 0.0013000000035390258, 0.1834000051021576],
    # !! 修改这里的值来设置物体的初始旋转 (四元数) !!
    "object_initial_rot": [-0.703499972820282, -0.061799999326467514, 0.0737999975681305, 0.704200029373169],
    
    # 物体缩放比例 (1.0 为原始大小)
    "object_scale": 0.3,
    
    # --- 重力场景模拟参数 --- 
    "gravity_enabled": True,  # 设置为False可以禁用重力
    "gravity_vector": [0.0, 0.0, -9.81],  # 重力向量 [x, y, z] (m/s²)
    "object_density": 200.0,  # 笔的密度 (kg/m³)
    "object_restitution": 0.2,  # 碰撞弹性系数
    "object_static_friction": 0.8,  # 静摩擦系数
    "object_dynamic_friction": 0.6,  # 动摩擦系数
    "object_damping": 0.1,  # 空气阻力系数
    
    # --- 调试参数 ---
    "print_interval_sec": 0.5,  # 每隔多少秒打印一次状态信息
    "num_envs": 1,             # 通常调试时只用一个环境
    "env_spacing": 1.0,        # 环境间距
    "high_stiffness": 1000.0,   # 手部关节的 P Gain (刚度)
    "high_damping": 100.0,    # 手部关节的 D Gain (阻尼)
    "auto_reset_interval_sec": 5.0,   # 自动重置物体的时间间隔（秒）
    "simulation_dt": 1.0/60.0,  # 仿真步长 (秒)
    "simulation_substeps": 3,   # 物理子步骤数量
}
# --- 配置区域结束 ---

def get_absolute_path(path):
    """将相对于当前脚本的路径转换为绝对路径"""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(script_dir, path)

# 辅助函数：将绕轴旋转角度转换为四元数
def quat_from_axis_angle(axis, angle_rad):
    axis = np.array(axis) / np.linalg.norm(axis) # 归一化轴
    angle_half = angle_rad / 2.0
    sin_a2 = math.sin(angle_half)
    cos_a2 = math.cos(angle_half)
    return gymapi.Quat(axis[0] * sin_a2, axis[1] * sin_a2, axis[2] * sin_a2, cos_a2)

# 辅助函数：四元数乘法
def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1.x, q1.y, q1.z, q1.w
    x2, y2, z2, w2 = q2.x, q2.y, q2.z, q2.w
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return gymapi.Quat(x, y, z, w)

def main():
    # --- Isaac Gym 初始化 ---
    gym = gymapi.acquire_gym()

    # 配置仿真参数
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    
    # 设置重力参数
    if DEBUG_CONFIG["gravity_enabled"]:
        gravity = DEBUG_CONFIG["gravity_vector"]
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
    else:
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)  # 禁用重力
    
    # 设置时间步长和物理子步骤
    sim_params.dt = DEBUG_CONFIG["simulation_dt"]
    sim_params.substeps = DEBUG_CONFIG["simulation_substeps"]
    sim_params.use_gpu_pipeline = False  # 根据需要选择 True/False

    # 配置物理引擎参数 - 用于精确的物理模拟
    sim_params.physx.solver_type = 1  # PGS解算器
    sim_params.physx.num_position_iterations = 8  # 位置迭代次数
    sim_params.physx.num_velocity_iterations = 4  # 速度迭代次数
    sim_params.physx.contact_offset = 0.001  # 接触偏移量
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.bounce_threshold_velocity = 0.2  # 反弹阈值
    sim_params.physx.max_depenetration_velocity = 1.0  # 去穿透速度，使运动更平滑
    sim_params.physx.default_buffer_size_multiplier = 5.0
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0025

    # 计算设备
    compute_device_id = 0
    graphics_device_id = 0
    physics_engine = gymapi.SIM_PHYSX

    sim = gym.create_sim(compute_device_id, graphics_device_id, physics_engine, sim_params)
    if sim is None:
        print("*** 创建仿真环境失败")
        quit()

    # 创建地面
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)  # Z轴向上
    plane_params.distance = 0
    plane_params.static_friction = 1.0
    plane_params.dynamic_friction = 0.9
    plane_params.restitution = 0.1  # 地面反弹系数
    gym.add_ground(sim, plane_params)

    # 创建查看器
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** 创建查看器失败")
        quit()

    # --- 加载资源 ---
    # --- Hand Asset ---
    hand_urdf_path_full = get_absolute_path(DEBUG_CONFIG["hand_urdf_path"])
    hand_asset_root = os.path.dirname(hand_urdf_path_full)
    hand_asset_file = os.path.basename(hand_urdf_path_full)

    asset_options_hand = gymapi.AssetOptions()
    asset_options_hand.fix_base_link = True  # 固定手的基础链接
    asset_options_hand.flip_visual_attachments = False
    asset_options_hand.armature = 0.01
    asset_options_hand.disable_gravity = True  # 禁用手部的重力影响
    asset_options_hand.use_mesh_materials = True
    # asset_options_hand.vhacd_enabled = True
    # asset_options_hand.vhacd_params = gymapi.VhacdParams()
    # asset_options_hand.vhacd_params.resolution = 10000
    asset_options_hand.convex_decomposition_from_submeshes = True


    print(f"正在加载手部资源: {hand_asset_file}")
    hand_asset = gym.load_asset(sim, hand_asset_root, hand_asset_file, asset_options_hand)
    if hand_asset is None:
        print(f"*** 加载手部资源失败: {hand_urdf_path_full}")
        quit()

    num_hand_dofs = gym.get_asset_dof_count(hand_asset)
    num_hand_bodies = gym.get_asset_rigid_body_count(hand_asset)
    print(f"手部资源加载成功: {num_hand_dofs} 个自由度, {num_hand_bodies} 个刚体")

    # 检查 DOF 数量是否匹配
    if num_hand_dofs != len(DEBUG_CONFIG["hand_dof_targets_rad"]):
        print(f"*** 错误: URDF中的DOF数量 ({num_hand_dofs}) 与配置中的DOF数量 ({len(DEBUG_CONFIG['hand_dof_targets_rad'])}) 不匹配")
        quit()

    # 获取 DOF 名称 (用于参考)
    dof_names = gym.get_asset_dof_names(hand_asset)
    print("手部自由度名称 (IsaacGym):")
    for i, name in enumerate(dof_names):
        print(f"  {i}: {name}")

    # 配置手部 DOF 属性 (高刚度位置控制)
    hand_dof_props = gym.get_asset_dof_properties(hand_asset)
    hand_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    hand_dof_props["stiffness"].fill(DEBUG_CONFIG["high_stiffness"])
    hand_dof_props["damping"].fill(DEBUG_CONFIG["high_damping"])
    hand_lower_limits = hand_dof_props["lower"]
    hand_upper_limits = hand_dof_props["upper"]
    hand_ranges = hand_upper_limits - hand_lower_limits

    # --- Object Asset (笔) ---
    object_urdf_path_full = get_absolute_path(DEBUG_CONFIG["object_urdf_path"])
    object_asset_root = os.path.dirname(object_urdf_path_full)
    object_asset_file = os.path.basename(object_urdf_path_full)

    # 使用特殊配置加载物体资源，启用重力和物理属性
    asset_options_object = gymapi.AssetOptions()
    asset_options_object.fix_base_link = False  # 不固定物体，使其受重力影响
    asset_options_object.disable_gravity = False  # 启用重力
    asset_options_object.use_mesh_materials = True
    asset_options_object.convex_decomposition_from_submeshes = True
    # asset_options_object.vhacd_enabled = True  # 启用复杂碰撞体计算
    # asset_options_object.vhacd_params = gymapi.VhacdParams()
    # asset_options_object.vhacd_params.resolution = 100000
    
    # 设置物理属性
    asset_options_object.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    asset_options_object.density = DEBUG_CONFIG["object_density"]
    asset_options_object.angular_damping = DEBUG_CONFIG["object_damping"]
    asset_options_object.linear_damping = DEBUG_CONFIG["object_damping"]

    print(f"正在加载物体资源: {object_asset_file}")
    object_asset = gym.load_asset(sim, object_asset_root, object_asset_file, asset_options_object)
    if object_asset is None:
        print(f"*** 加载物体资源失败: {object_urdf_path_full}")
        quit()

    num_object_bodies = gym.get_asset_rigid_body_count(object_asset)
    print(f"物体资源加载成功: {num_object_bodies} 个刚体")
    
    # --- 创建环境和 Actor ---
    num_envs = DEBUG_CONFIG["num_envs"]
    env_spacing = DEBUG_CONFIG["env_spacing"]
    env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    envs = []
    hand_idxs = []
    object_idxs = []
    object_actors = []

    # 设置手和物体的初始姿态
    hand_start_pose = gymapi.Transform()
    hand_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)  # 手基座位置
    rotate_y_rad = math.radians(DEBUG_CONFIG["hand_base_rotate_y_deg"])
    hand_start_pose.r = quat_from_axis_angle([0, 1, 0], rotate_y_rad)

    # 设置物体的初始姿态
    object_start_pose = gymapi.Transform()
    object_start_pose.p = gymapi.Vec3(*DEBUG_CONFIG["object_initial_pos"])
    object_initial_rot = DEBUG_CONFIG["object_initial_rot"]
    object_start_pose.r = gymapi.Quat(
        object_initial_rot[0], 
        object_initial_rot[1], 
        object_initial_rot[2], 
        object_initial_rot[3]
    )

    # 获取物体缩放比例
    object_scale = float(DEBUG_CONFIG["object_scale"])

    print(f"正在创建 {num_envs} 个环境...")
    for i in range(num_envs):
        env_ptr = gym.create_env(sim, env_lower, env_upper, int(math.sqrt(num_envs)))
        envs.append(env_ptr)

        # 创建手 Actor
        hand_actor = gym.create_actor(env_ptr, hand_asset, hand_start_pose, "hand", i, 1, 0)  # 碰撞组 1
        if hand_actor is None:
            print(f"*** 创建手部Actor失败 (环境 {i})")
            quit()
        gym.set_actor_dof_properties(env_ptr, hand_actor, hand_dof_props)
        hand_idx = gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
        hand_idxs.append(hand_idx)

        # 创建物体 Actor (笔)
        object_actor = gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 2, 1)  # 碰撞组 2
        if object_actor is None:
            print(f"*** 创建物体Actor失败 (环境 {i})")
            quit()

        # 设置物体缩放比例
        gym.set_actor_scale(env_ptr, object_actor, object_scale)
        
        object_idxs.append(gym.get_actor_index(env_ptr, object_actor, gymapi.DOMAIN_SIM))
        object_actors.append(object_actor)

    # 转换为 Tensor (用于后续 Tensor API 操作)
    hand_idxs = to_torch(hand_idxs, dtype=torch.long, device='cpu')
    object_idxs = to_torch(object_idxs, dtype=torch.long, device='cpu')

    # 设置查看器位置，更好地观察重力效果
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(0.5, 0.5, 0.5), gymapi.Vec3(0.0, 0.0, 0.0))

    # --- 准备 Tensor API ---
    root_state_tensor = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    dof_state_tensor = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))

    # 确定 Tensor 设备
    tensor_device = root_state_tensor.device
    # ***获取手部和物体的根节点（初始位置）***
    hand_actor_states = root_state_tensor.view(num_envs, -1, 13)[:, 0]
    object_actor_states = root_state_tensor.view(num_envs, -1, 13)[:, 1]
    # ***获取手部 DOF 状态***
    hand_dof_states = dof_state_tensor.view(num_envs, num_hand_dofs, 2)
    hand_dof_pos = hand_dof_states[..., 0]
    hand_dof_vel = hand_dof_states[..., 1]

    # 将配置中的目标 DOF 姿态转换为 Tensor
    dof_targets = torch.tensor(DEBUG_CONFIG["hand_dof_targets_rad"], dtype=torch.float32, device='cpu')
    dof_targets = dof_targets.unsqueeze(0).repeat(num_envs, 1)

    # --- 初始化 DOF 状态 ---
    if tensor_device != 'cpu':
        hand_dof_pos[:] = dof_targets.to(tensor_device)
        hand_dof_vel[:] = 0.0
    else:
        hand_dof_pos[:] = dof_targets
        hand_dof_vel[:] = 0.0

    env_ids_int32 = torch.arange(num_envs, dtype=torch.int32, device='cpu')
    gym.set_dof_state_tensor_indexed(sim,
                                     gymtorch.unwrap_tensor(dof_state_tensor),
                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    # --- 主循环 ---
    last_print_time = time.time()
    last_reset_time = time.time()
    print_interval = DEBUG_CONFIG["print_interval_sec"]
    auto_reset_interval = DEBUG_CONFIG["auto_reset_interval_sec"]
    
    # 跟踪笔的原始位置
    original_object_position = DEBUG_CONFIG["object_initial_pos"]
    original_object_rotation = DEBUG_CONFIG["object_initial_rot"]
    original_object_quat = gymapi.Quat(
        original_object_rotation[0], 
        original_object_rotation[1], 
        original_object_rotation[2], 
        original_object_rotation[3]
    )
    for env_id in range(num_envs):
        # 应用原始位置加上随机偏移
        object_actor_states[env_id, 0] = original_object_position[0]
        object_actor_states[env_id, 1] = original_object_position[1]
        object_actor_states[env_id, 2] = original_object_position[2]
        
        # 设置旋转
        object_actor_states[env_id, 3] = original_object_quat.x
        object_actor_states[env_id, 4] = original_object_quat.y
        object_actor_states[env_id, 5] = original_object_quat.z
        object_actor_states[env_id, 6] = original_object_quat.w
        
        # 重置所有速度为0
        object_actor_states[env_id, 7:13] = 0.0
    # 检查重力设置状态
    is_gravity_on = DEBUG_CONFIG["gravity_enabled"]

    while not gym.query_viewer_has_closed(viewer):
        # 设置手部位置目标
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(dof_targets))

        # 更新对象状态
        object_idxs_int32 = object_idxs.to(dtype=torch.int32)
        gym.set_actor_root_state_tensor_indexed(
            sim,
            gymtorch.unwrap_tensor(root_state_tensor),
            gymtorch.unwrap_tensor(object_idxs_int32),
            len(object_idxs_int32)
        )

        # 自动周期性重置物体 (如果启用)
        current_time = time.time()
        if auto_reset_interval > 0 and current_time - last_reset_time >= auto_reset_interval:
            print("自动重置笔的位置...")
            # 随机略微调整位置，使每次重置稍有不同
            random_offset = [
                np.random.uniform(-0.02, 0.02),
                np.random.uniform(-0.02, 0.02),
                np.random.uniform(0, 0.02)
            ]
            
            for env_id in range(num_envs):
                # 应用原始位置加上随机偏移
                object_actor_states[env_id, 0] = original_object_position[0] + random_offset[0]
                object_actor_states[env_id, 1] = original_object_position[1] + random_offset[1]
                object_actor_states[env_id, 2] = original_object_position[2] + random_offset[2]
                
                # 设置旋转
                object_actor_states[env_id, 3] = original_object_quat.x
                object_actor_states[env_id, 4] = original_object_quat.y
                object_actor_states[env_id, 5] = original_object_quat.z
                object_actor_states[env_id, 6] = original_object_quat.w
                
                # 重置所有速度为0
                object_actor_states[env_id, 7:13] = 0.0
            
            last_reset_time = current_time

        # 步进仿真
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # 刷新 Tensor
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        hand_dof_pos = hand_dof_states[..., 0]

        # 打印状态信息
        if current_time - last_print_time >= print_interval:
            print("-" * 40)
            print(f"时间: {current_time:.2f}")
            
            # 获取笔的当前位置和旋转
            obj_pos = object_actor_states[0, 0:3].cpu().numpy().round(4).tolist()
            obj_rot = object_actor_states[0, 3:7].cpu().numpy().round(4).tolist()
            
            # 获取速度信息
            obj_vel = object_actor_states[0, 7:10].cpu().numpy().round(4).tolist()
            obj_ang_vel = object_actor_states[0, 10:13].cpu().numpy().round(4).tolist()
            
            # 计算高度
            height_from_ground = obj_pos[2]
            
            print(f"笔的位置 = {obj_pos}")
            print(f"离地高度 = {height_from_ground:.4f} m")
            print(f"四元数 = {obj_rot}")
            print(f"角速度 = {obj_ang_vel}")
            
            last_print_time = current_time

        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    # --- 清理 ---
    print("查看器已关闭，清理资源...")
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == '__main__':
    main()