import isaacgym
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import to_torch, tensor_clamp, quat_from_euler_xyz # quat_from_euler_xyz might be useful

import torch
import numpy as np
import time
import os
import math

# --- 配置区域 ---
# 在这里修改手部和物体的姿态、缩放进行调试
DEBUG_CONFIG = {
    # --- 资源路径 ---
    "hand_urdf_path": "../assets/linker_hand/L25_dof_urdf.urdf",
    "object_urdf_path": "../assets/cylinder/pencil-5-7/0000.urdf",

    # --- 手部姿态 (21 个自由度) ---
    # !! 修改这里的 21 个值来设置手的目标关节角度 (弧度) !!
    # "hand_dof_targets_rad": [
    #     0.0, 0.0, 0.0, 0.0,  # index (0-3)
    #     0.0, 0.0, 0.0, 0.0,  # little (4-7)
    #     0.0, 0.0, 0.0, 0.0,  # middle (8-11)
    #     0.0, 0.0, 0.0, 0.0,  # ring (12-15)
    #     0.0, 0.0, 0.0, 0.0, 0.0  # thumb (16-20)
    # ],
    # --- 手部初始旋转 (让手躺下) ---
    # !! 修改这里的值来设置手的初始基座旋转 (绕 Y 轴旋转的角度，单位：度) !!
    # 负值表示向后躺 (假设 Z 轴向上, X 轴向前)
    "hand_base_rotate_y_deg": -75.0,
    # --- 手部姿态 (21 个自由度) ---
    "hand_dof_targets_rad": [0.17949999868869781, -1.18340003490448, -0.396699994802475, -0.5026999711990356, -0.18000000715255737, -1.0195000171661377, -0.8944000005722046, -1.1978000402450562, -0.0, -1.2105000019073486, -0.6015999913215637, -0.8508999943733215, -9.999999747378752e-05, -0.9298999905586243, -1.2790000438690186, -0.9383000135421753, -0.03500000014901161, -1.280400037765503, -0.5479000210762024, -0.27639999985694885, -0.03999999910593033],
    "object_initial_pos": [-0.1124000060558319, 0.03090000070631504, 0.22130000722408295],
    # !! 修改这里的值来设置物体的初始旋转 (四元数) !!
    "object_scale": 0.3,

    # --- 调试参数 ---
    "print_interval_sec": 1.0,  # 每隔多少秒打印一次状态信息
    "num_envs": 1,             # 通常调试时只用一个环境
    "env_spacing": 1.0,        # 环境间距
    "high_stiffness": 1000.0,   # 手部关节的 P Gain (刚度)，设置很高以抵抗接触力
    "high_damping": 100.0,    # 手部关节的 D Gain (阻尼)
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


def main():
    # --- Isaac Gym 初始化 ---
    gym = gymapi.acquire_gym()

    # 配置仿真参数
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)  # 固定使用标准重力
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 4  # 增加子步骤数以提高物理模拟精度
    sim_params.use_gpu_pipeline = False # 根据需要选择 True/False

    # 配置物理引擎参数 - 调整以减少漂移
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 12  # 增加位置迭代次数
    sim_params.physx.num_velocity_iterations = 4   # 增加速度迭代次数
    sim_params.physx.contact_offset = 0.001  # 减小接触偏移量
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.bounce_threshold_velocity = 0.1  # 降低反弹阈值
    sim_params.physx.max_depenetration_velocity = 2.0  # 降低去穿透速度，使运动更平滑
    sim_params.physx.default_buffer_size_multiplier = 5.0  # 增加缓冲区大小
    sim_params.physx.friction_offset_threshold = 0.001  # 减小摩擦力偏移阈值
    sim_params.physx.friction_correlation_distance = 0.001  # 减小摩擦力相关距离

    # 计算设备
    # !! 重要：确保这里的 compute_device_id 与你的系统匹配 !!
    # 如果遇到设备不匹配错误，尝试改为 0 或其他有效的 GPU ID
    # 如果没有 GPU 或想强制使用 CPU，需要修改 sim_params.use_gpu_pipeline = False
    # 并确保所有 tensor 创建和 API 调用都使用 CPU
    compute_device_id = 0
    graphics_device_id = 0
    physics_engine = gymapi.SIM_PHYSX

    sim = gym.create_sim(compute_device_id, graphics_device_id, physics_engine, sim_params)
    if sim is None:
        print("*** Failed to create sim")
        quit()

    # 创建地面
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    plane_params.distance = 0
    plane_params.static_friction = 1.0
    plane_params.dynamic_friction = 1.0
    plane_params.restitution = 0
    gym.add_ground(sim, plane_params)

    # 创建查看器
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")
        quit()

    # --- 加载资源 ---
    # --- Hand Asset ---
    hand_urdf_path_full = DEBUG_CONFIG["hand_urdf_path"] # 使用完整路径配置
    hand_asset_root = os.path.dirname(hand_urdf_path_full)
    hand_asset_file = os.path.basename(hand_urdf_path_full)

    asset_options_hand = gymapi.AssetOptions()
    asset_options_hand.fix_base_link = True # 固定手的基础链接
    asset_options_hand.flip_visual_attachments = False
    asset_options_hand.armature = 0.01
    asset_options_hand.disable_gravity = True # 通常手的重力影响不大，可以禁用
    asset_options_hand.use_mesh_materials = True
    # asset_options_hand.vhacd_enabled = True
    # asset_options_hand.vhacd_params = gymapi.VhacdParams()
    # asset_options_hand.vhacd_params.resolution = 10000
    asset_options_hand.convex_decomposition_from_submeshes = True


    print(f"Loading hand asset from root: {hand_asset_root}, file: {hand_asset_file}")
    hand_asset = gym.load_asset(sim, hand_asset_root, hand_asset_file, asset_options_hand)
    if hand_asset is None:
        print(f"*** Failed to load hand asset: {hand_urdf_path_full}")
        quit()

    num_hand_dofs = gym.get_asset_dof_count(hand_asset)
    num_hand_bodies = gym.get_asset_rigid_body_count(hand_asset)
    print(f"Hand asset loaded: {num_hand_dofs} DOFs, {num_hand_bodies} bodies")

    # 检查 DOF 数量是否匹配
    if num_hand_dofs != len(DEBUG_CONFIG["hand_dof_targets_rad"]):
        print(f"*** ERROR: URDF DOF count ({num_hand_dofs}) does not match config DOF count ({len(DEBUG_CONFIG['hand_dof_targets_rad'])})")
        quit()

    # 获取 DOF 名称 (用于参考)
    dof_names = gym.get_asset_dof_names(hand_asset)
    print("Hand DoF Names (IsaacGym):")
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

    # --- Object Asset ---
    object_urdf_path_full = DEBUG_CONFIG["object_urdf_path"] # 使用完整路径配置
    object_asset_root = os.path.dirname(object_urdf_path_full)
    object_asset_file = os.path.basename(object_urdf_path_full)

    asset_options_object = gymapi.AssetOptions()
    asset_options_object.fix_base_link = False  # 不固定物体
    asset_options_object.disable_gravity = True  # 禁用重力以便更容易控制
    asset_options_object.use_mesh_materials = True
    asset_options_object.vhacd_enabled = True
    asset_options_object.vhacd_params = gymapi.VhacdParams()
    asset_options_object.vhacd_params.resolution = 100000 # 可根据需要调整
    # 增强物理稳定性的参数
    asset_options_object.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    asset_options_object.density = 200.0  # 增加密度
    asset_options_object.angular_damping = 10.0  # 增加角阻尼
    asset_options_object.linear_damping = 10.0   # 增加线性阻尼

    print(f"Loading object asset from root: {object_asset_root}, file: {object_asset_file}")
    object_asset = gym.load_asset(sim, object_asset_root, object_asset_file, asset_options_object)
    if object_asset is None:
        print(f"*** Failed to load object asset: {object_urdf_path_full}")
        quit()

    num_object_bodies = gym.get_asset_rigid_body_count(object_asset)
    print(f"Object asset loaded: {num_object_bodies} bodies")
    
    # --- 创建环境和 Actor ---
    num_envs = DEBUG_CONFIG["num_envs"]
    env_spacing = DEBUG_CONFIG["env_spacing"]
    env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    envs = []
    hand_idxs = []
    object_idxs = []

    # 设置手和物体的初始姿态
    hand_start_pose = gymapi.Transform()
    hand_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0) # 手基座位置，通常在原点
    # 计算手的初始旋转四元数
    rotate_y_rad = math.radians(DEBUG_CONFIG["hand_base_rotate_y_deg"])
    hand_start_pose.r = quat_from_axis_angle([0, 1, 0], rotate_y_rad) # 绕 Y 轴旋转

    object_start_pose = gymapi.Transform()
    object_start_pose.p = gymapi.Vec3(*DEBUG_CONFIG["object_initial_pos"])
    object_start_pose.r = quat_from_axis_angle([1, 0, 0], math.radians(-90))

    # 获取物体缩放比例
    object_scale = float(DEBUG_CONFIG["object_scale"]) # 确保是浮点数

    print(f"Creating {num_envs} environments...")
    for i in range(num_envs):
        env_ptr = gym.create_env(sim, env_lower, env_upper, int(math.sqrt(num_envs)))
        envs.append(env_ptr)

        # 创建手 Actor
        hand_actor = gym.create_actor(env_ptr, hand_asset, hand_start_pose, "hand", i, 0, 0) # 碰撞组 0, 禁用自碰撞
        if hand_actor is None:
            print(f"*** Failed to create hand actor in env {i}")
            quit()
        gym.set_actor_dof_properties(env_ptr, hand_actor, hand_dof_props)
        hand_idx = gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
        hand_idxs.append(hand_idx)

        # 创建物体 Actor
        # !! 修改: 在创建 actor 时加入 scale 参数 !!
        object_actor = gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 1, 0) # 碰撞组 1
        if object_actor is None:
            print(f"*** Failed to create object actor in env {i}")
            quit()

        gym.set_actor_scale(env_ptr, object_actor, object_scale) # 设置缩放比例
        object_idxs.append(gym.get_actor_index(env_ptr, object_actor, gymapi.DOMAIN_SIM))

    # 转换为 Tensor (用于后续 Tensor API 操作)
    hand_idxs = to_torch(hand_idxs, dtype=torch.long, device='cpu')
    object_idxs = to_torch(object_idxs, dtype=torch.long, device='cpu')

    # 设置查看器位置
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(0.5, 0.5, 1.0), gymapi.Vec3(0.0, 0.0, 0.5))

    # --- 准备 Tensor API ---
    root_state_tensor = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    dof_state_tensor = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))

    # 确定 Tensor 设备 (基于 root_state_tensor 推断)
    tensor_device = root_state_tensor.device

    hand_actor_states = root_state_tensor.view(num_envs, -1, 13)[:, 0]
    object_actor_states = root_state_tensor.view(num_envs, -1, 13)[:, 1]

    hand_dof_states = dof_state_tensor.view(num_envs, num_hand_dofs, 2)
    hand_dof_pos = hand_dof_states[..., 0]
    hand_dof_vel = hand_dof_states[..., 1]

    # 将配置中的目标 DOF 姿态转换为 Tensor (放在 CPU 上)
    dof_targets = torch.tensor(DEBUG_CONFIG["hand_dof_targets_rad"], dtype=torch.float32, device='cpu') # 保持在 CPU
    dof_targets = dof_targets.unsqueeze(0).repeat(num_envs, 1) # 扩展到所有环境

    # --- 初始化 DOF 状态 ---
    # 将手的初始位置设置为目标位置
    # 注意: 如果 DOF state tensor 在 GPU，需要先将 dof_targets 复制到 GPU
    if tensor_device != 'cpu':
        hand_dof_pos[:] = dof_targets.to(tensor_device)
        hand_dof_vel[:] = 0.0
    else:
        hand_dof_pos[:] = dof_targets
        hand_dof_vel[:] = 0.0

    # 将初始 DOF 状态写入仿真
    env_ids_int32 = torch.arange(num_envs, dtype=torch.int32, device='cpu') # 保持在 CPU
    gym.set_dof_state_tensor_indexed(sim,
                                     gymtorch.unwrap_tensor(dof_state_tensor),
                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    # --- 主循环 ---
    last_print_time = time.time()
    print_interval = DEBUG_CONFIG["print_interval_sec"]

    # 新增：物体旋转角度
    object_angle = 0.0
    object_rotate_interval = 0.0005  # 每帧旋转的弧度
    object_init_quat = quat_from_axis_angle([1, 0, 0], math.radians(-90))
    while not gym.query_viewer_has_closed(viewer):
        # 设置 DOF 位置目标 (持续设置以保持姿态)
        # !! 目标张量在 CPU 上 !!
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(dof_targets))

        # 让物体绕z轴旋转，基于初始旋转，并重置位置
        object_angle += object_rotate_interval
        z_quat = quat_from_axis_angle([0, -1, 0], object_angle)
        final_quat = object_init_quat * z_quat
        object_init_pos = DEBUG_CONFIG["object_initial_pos"]
        for env_id in range(num_envs):
            object_actor_states[env_id, 0] = object_init_pos[0]
            object_actor_states[env_id, 1] = object_init_pos[1]
            object_actor_states[env_id, 2] = object_init_pos[2]
            object_actor_states[env_id, 3] = final_quat.x
            object_actor_states[env_id, 4] = final_quat.y
            object_actor_states[env_id, 5] = final_quat.z
            object_actor_states[env_id, 6] = final_quat.w
        # 只写object的index，且为int32
        object_idxs_int32 = object_idxs.to(dtype=torch.int32)
        gym.set_actor_root_state_tensor_indexed(
            sim,
            gymtorch.unwrap_tensor(root_state_tensor),
            gymtorch.unwrap_tensor(object_idxs_int32),
            len(object_idxs_int32)
        )

        # 步进仿真
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # 刷新 Tensor (获取最新的状态)
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)

        # --- 定期打印状态 ---
        current_time = time.time()
        if current_time - last_print_time >= print_interval:
            print("-" * 30)
            print(f"Time: {current_time:.2f}")
            current_hand_dof_pos = hand_dof_pos[0].cpu().numpy().round(4).tolist()
            print(f"Hand DOF Positions (Env 0): {current_hand_dof_pos}")
            current_obj_state = object_actor_states[0].cpu().numpy()
            obj_pos = current_obj_state[0:3].round(4).tolist()
            obj_rot = current_obj_state[3:7].round(4).tolist()
            print(f"Object Pose (Env 0): Pos={obj_pos}, RotQuat={obj_rot}")
            last_print_time = current_time

        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    # --- 清理 ---
    print("Viewer closed, cleaning up...")
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == '__main__':
    main()