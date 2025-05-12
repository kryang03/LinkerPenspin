import numpy as np
from isaacgym import gymapi
from isaacgym import gymtorch
# from isaacgym import gymutil # 不再需要 gymutil
import torch
import time
import pprint
import yaml
import os

# --- 常量定义 ---

# 指尖连杆名称 (用于后续查找索引)
FINGERTIP_LINK_NAMES = [
    "little_joint3",
    "ring_joint3",
    "middle_joint3",
    "index_joint3",
    "thumb_joint4"
]
# 手掌连杆名称
PALM_LINK_NAME = "base_link"

# 文件路径 (假设脚本和YAML/CSV在同一目录)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# YAML 文件，包含期望的关节名称列表
JOINT_NAMES_YAML_PATH = os.path.join(CURRENT_DIR, "joint_names_L25-dof-urdf.yaml")
# URDF 信息参考文件 (供用户查看)
URDF_CSV_PATH = os.path.join(CURRENT_DIR, "L25-dof-urdf.csv") #

# !!! 确保此 URDF 路径相对于你的 IsaacGym assets 文件夹是正确的 !!!
# 根据你的项目结构调整 ASSET_ROOT
ASSET_ROOT = os.path.join(CURRENT_DIR, "..", "assets") #
# URDF 文件相对于 ASSET_ROOT 的路径
LINKER_URDF_REL_PATH = "linker_hand/L25_dof_urdf.urdf" #
# URDF 文件的绝对路径
LINKER_URDF_ABS_PATH = os.path.join(ASSET_ROOT, LINKER_URDF_REL_PATH) #

# 控制参数
SWEEP_DURATION = 5.0  # 所有关节从下限扫到上限再扫回下限的总时间（秒） #
TARGET_STIFFNESS = 5000.0 # PD 控制器的 Stiffness (刚度)
TARGET_DAMPING = 2000.0   # PD 控制器的 Damping (阻尼)

# --- 硬编码的默认设置 (替代 args) ---
DEFAULT_PHYSICS_ENGINE = gymapi.SIM_PHYSX
DEFAULT_COMPUTE_DEVICE_ID = 0 # 默认使用 GPU 0
DEFAULT_GRAPHICS_DEVICE_ID = 0 # 默认使用 GPU 0
DEFAULT_HEADLESS = False # 默认显示 Viewer

# --- 函数定义 (与之前相同，保持不变) ---

def load_and_filter_joint_names(yaml_path):
    """从 YAML 文件加载并过滤关节名称。"""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            if 'controller_joint_names' in data:
                filtered_names = [name for name in data['controller_joint_names'] if name and isinstance(name, str)]
                if not filtered_names:
                    print(f"Warning: 'controller_joint_names' in {yaml_path} is empty or contains only invalid entries.")
                    return None
                return filtered_names
            else:
                print(f"Warning: 'controller_joint_names' key not found in {yaml_path}")
                return None
    except FileNotFoundError:
        print(f"Error: YAML file not found at {yaml_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {yaml_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading YAML file {yaml_path}: {e}")
        return None

def print_simulation_parameters(sim_params):
    """打印仿真参数详情。"""
    print("--- Simulation Parameters ---")
    print(f"  dt: {sim_params.dt}")
    print(f"  substeps: {sim_params.substeps}")
    print(f"  gravity: {sim_params.gravity}")
    print(f"  up_axis: {sim_params.up_axis}")
    print("  PhysX Params:")
    physx_params = sim_params.physx
    print(f"    use_gpu: {physx_params.use_gpu}")
    print(f"    solver_type: {physx_params.solver_type}")
    print(f"    num_position_iterations: {physx_params.num_position_iterations}")
    print(f"    num_velocity_iterations: {physx_params.num_velocity_iterations}")
    print(f"    contact_offset: {physx_params.contact_offset}")
    print(f"    rest_offset: {physx_params.rest_offset}")
    print(f"    max_gpu_contact_pairs: {physx_params.max_gpu_contact_pairs}")
    print("-" * 30)

def print_asset_options(asset_options):
    """打印资源加载选项。"""
    print("--- Asset Options ---")
    print(f"  fix_base_link: {asset_options.fix_base_link}")
    print(f"  flip_visual_attachments: {asset_options.flip_visual_attachments}")
    print(f"  use_mesh_materials: {asset_options.use_mesh_materials}")
    print(f"  disable_gravity: {asset_options.disable_gravity}")
    print(f"  thickness: {asset_options.thickness}")
    print(f"  vhacd_enabled: {asset_options.vhacd_enabled}")
    if asset_options.vhacd_enabled:
        print(f"    vhacd_resolution: {asset_options.vhacd_params.resolution}")
    print(f"  mesh_normal_mode: {asset_options.mesh_normal_mode}")
    print(f"  default_dof_drive_mode: {asset_options.default_dof_drive_mode}")
    print("-" * 30)

def verify_dof_consistency(gym, hand_asset, expected_joint_names):
    """验证 IsaacGym 加载的 DoF 与 YAML 文件中的预期是否一致。"""
    num_dofs = gym.get_asset_dof_count(hand_asset)
    isaacgym_dof_names = gym.get_asset_dof_names(hand_asset)

    print("\n--- Verifying DoF Consistency ---")
    print(f"  Number of DoFs (IsaacGym): {num_dofs}")
    print("  DoF Names (IsaacGym):")
    for i, name in enumerate(isaacgym_dof_names):
        print(f"    {i}: {name}")

    if expected_joint_names:
        expected_dof_count = len(expected_joint_names)
        if num_dofs != expected_dof_count:
            print(f"\n*** WARNING: Mismatch in DoF count! IsaacGym found {num_dofs}, but YAML expected {expected_dof_count}. ***")
        elif isaacgym_dof_names != expected_joint_names:
            print("\n*** WARNING: Mismatch in DoF names/order between IsaacGym and YAML file! ***")
            print("  IsaacGym Names:", isaacgym_dof_names)
            print("  YAML Names:    ", expected_joint_names)
            print("  Please carefully check the URDF and the YAML file for consistency.")
        else:
            print("\n--- DoF Count and Names Verified Successfully against YAML ---")
    else:
        print("\n--- Skipping DoF name verification due to YAML loading error or empty list ---")
    print("-" * 30)
    return num_dofs, isaacgym_dof_names

def print_dof_limits_and_defaults(actor_dof_props, default_dof_pos, num_dofs):
    """打印 DoF 的限制和默认位置。"""
    print("\n--- Actor DoF Properties & Defaults ---")
    lower_limits = actor_dof_props["lower"]
    upper_limits = actor_dof_props["upper"]
    effort_limits = actor_dof_props["effort"]
    velocity_limits = actor_dof_props["velocity"]

    print(f"  Lower Limits:\n{lower_limits}")
    print(f"\n  Upper Limits:\n{upper_limits}")
    print(f"\n  Effort Limits:\n{effort_limits}")
    print(f"\n  Velocity Limits:\n{velocity_limits}")
    print(f"\n  Default Positions (Rest Pose):\n{default_dof_pos}")
    print("-" * 30)
    return lower_limits, upper_limits

def find_key_link_indices(gym, env, actor_handle, link_names, link_type_name):
    """查找指定名称列表的连杆 (Rigid Body) 索引。"""
    indices = []
    print(f"\n--- Finding {link_type_name} Link Indices ---")
    print(f"  Attempting to find: {link_names}")
    for name in link_names:
        idx = gym.find_actor_rigid_body_index(env, actor_handle, name, gymapi.DOMAIN_SIM)
        if idx == gymapi.INVALID_HANDLE:
            print(f"  Warning: {link_type_name} link '{name}' not found!")
        else:
            print(f"  Found {link_type_name} link '{name}' at SIM index: {idx}")
            indices.append(idx)
    print("-" * 30)
    return indices

# --- 主函数 ---
def main():
    # --- 初始化 Gym ---
    gym = gymapi.acquire_gym()
    # --- 移除 args 解析 ---
    # args = gymutil.parse_arguments(description="Refactored Linker Hand Visualization - All Fingers Sweep")

    # --- 加载预期的关节名称 ---
    expected_joint_names = load_and_filter_joint_names(JOINT_NAMES_YAML_PATH)

    # --- 创建仿真参数 ---
    sim_params = gymapi.SimParams()
    sim_params.dt = 1 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    # 配置 PhysX 参数
    # 检查 CUDA 是否可用，决定是否使用 GPU
    use_gpu = torch.cuda.is_available()
    sim_params.physx.use_gpu = use_gpu
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.005
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    print_simulation_parameters(sim_params)

    # --- 确定运行设备 (基于硬编码和检测结果) ---
    physics_engine = DEFAULT_PHYSICS_ENGINE
    compute_device_id = DEFAULT_COMPUTE_DEVICE_ID
    graphics_device_id = DEFAULT_GRAPHICS_DEVICE_ID

    if physics_engine == gymapi.SIM_PHYSX and not use_gpu:
        print("Warning: CUDA not available, forcing CPU usage for simulation.")
        compute_device_id = -1 # -1 通常表示 CPU
        graphics_device_id = -1 # 对于 CPU 仿真，图形设备通常也设为 -1 或 0
        sim_device = 'cpu'
    elif physics_engine == gymapi.SIM_PHYSX and use_gpu:
        sim_device = f'cuda:{compute_device_id}' # 使用默认的 GPU ID
    else:
        # 其他物理引擎或特殊情况 (这里假设 PhysX)
        sim_device = 'cpu'

    print(f"--- Running on Device: {sim_device} (Compute ID: {compute_device_id}, Graphics ID: {graphics_device_id}) ---")
    print("-" * 30)

    # --- 创建仿真 ---
    sim = gym.create_sim(compute_device_id, graphics_device_id, physics_engine, sim_params)
    if sim is None:
        print("*** Failed to create sim")
        quit()

    # --- 添加地面 ---
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    plane_params.static_friction = 1.0
    plane_params.dynamic_friction = 1.0
    plane_params.restitution = 0.0
    gym.add_ground(sim, plane_params)

    # --- 创建 Viewer (如果不是 headless) ---
    viewer = None
    if not DEFAULT_HEADLESS:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            print("*** Failed to create viewer")
            # 如果创建 viewer 失败，可能需要退出或给出警告
            quit()
        cam_pos = gymapi.Vec3(0.5, 0.5, 0.5)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.1)
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    else:
        print("--- Running in headless mode (No Viewer) ---")


    # --- 加载手部资源 (Asset) ---
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = False
    asset_options.use_mesh_materials = True
    asset_options.disable_gravity = True
    asset_options.vhacd_enabled = True
    asset_options.vhacd_params.resolution = 300000
    asset_options.vhacd_params.max_convex_hulls = 10
    asset_options.vhacd_params.max_num_vertices_per_ch = 64
    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

    print_asset_options(asset_options)

    # 检查 URDF 文件是否存在
    if not os.path.exists(LINKER_URDF_ABS_PATH):
         print(f"Error: URDF file not found at: {LINKER_URDF_ABS_PATH}")
         print("Please check ASSET_ROOT and LINKER_URDF_REL_PATH variables.")
         quit()

    # 加载资源
    print(f"Loading asset from: {LINKER_URDF_REL_PATH} in root: {ASSET_ROOT}")
    hand_asset = gym.load_asset(sim, ASSET_ROOT, LINKER_URDF_REL_PATH, asset_options)
    if hand_asset is None:
        print(f"*** Failed to load asset {LINKER_URDF_REL_PATH}")
        quit()

    # --- 获取并验证 Asset 信息 ---
    num_bodies = gym.get_asset_rigid_body_count(hand_asset)
    num_shapes = gym.get_asset_rigid_shape_count(hand_asset)
    print("\n--- Asset Details ---")
    print(f"  Number of Rigid Bodies: {num_bodies}")
    print(f"  Number of Collision Shapes: {num_shapes}")
    body_names = gym.get_asset_rigid_body_names(hand_asset)
    print("  Rigid Body Names:")
    for i, name in enumerate(body_names): print(f"    {i}: {name}")

    # 验证 DoF 数量和名称
    num_dofs, isaacgym_dof_names = verify_dof_consistency(gym, hand_asset, expected_joint_names)
    if num_dofs <= 0:
        print("*** Error: Asset has 0 DoFs, cannot proceed.")
        quit()

    # 获取 Asset 级别的默认 DoF 属性
    asset_dof_props = gym.get_asset_dof_properties(hand_asset)
    print("\n  Asset Default DoF Properties (from URDF/Asset Options):")
    for i in range(num_dofs):
        print(f"    DoF {i} ({isaacgym_dof_names[i]}):")
        print(f"      hasLimits: {asset_dof_props['hasLimits'][i]}")
        print(f"      lower: {asset_dof_props['lower'][i]:.4f}")
        print(f"      upper: {asset_dof_props['upper'][i]:.4f}")
        print(f"      driveMode: {asset_dof_props['driveMode'][i]}")
        print(f"      velocity: {asset_dof_props['velocity'][i]:.4f}")
        print(f"      effort: {asset_dof_props['effort'][i]:.4f}")
        print(f"      stiffness: {asset_dof_props['stiffness'][i]:.4f}")
        print(f"      damping: {asset_dof_props['damping'][i]:.4f}")
        print(f"      friction: {asset_dof_props['friction'][i]:.4f}")
        print(f"      armature: {asset_dof_props['armature'][i]:.4f}")
    print("-" * 30)

    # --- 创建环境和 Actor ---
    num_envs = 1
    env_spacing = 1.0
    env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    env = gym.create_env(sim, env_lower, env_upper, num_envs)

    hand_pose = gymapi.Transform()
    hand_pose.p = gymapi.Vec3(0.0, 0.0, 0.15)
    hand_pose.r = gymapi.Quat(0, 0, 0, 1)

    hand_handle = gym.create_actor(env, hand_asset, hand_pose, "linker_hand", 0, 0, 0)
    if hand_handle == gymapi.INVALID_HANDLE:
        print("*** Failed to create actor 'linker_hand'")
        quit()

    # --- 设置 Actor 的 DoF 属性 ---
    actor_dof_props = gym.get_actor_dof_properties(env, hand_handle)
    actor_dof_props["driveMode"] = [gymapi.DOF_MODE_POS] * num_dofs
    actor_dof_props["stiffness"] = [TARGET_STIFFNESS] * num_dofs
    actor_dof_props["damping"] = [TARGET_DAMPING] * num_dofs
    set_success = gym.set_actor_dof_properties(env, hand_handle, actor_dof_props)
    if not set_success:
        print("*** Failed to set actor DoF properties")

    actor_dof_props_after = gym.get_actor_dof_properties(env, hand_handle)
    print("\n--- Actor DoF Properties (After Setting) ---")
    print(f"  Drive Mode (First DoF): {actor_dof_props_after['driveMode'][0]}")
    print(f"  Stiffness (First DoF): {actor_dof_props_after['stiffness'][0]}")
    print(f"  Damping (First DoF): {actor_dof_props_after['damping'][0]}")
    print("-" * 30)

    # --- 获取 DoF 限制和默认状态 ---
    lower_limits, upper_limits = print_dof_limits_and_defaults(
        actor_dof_props_after,
        gym.get_actor_dof_states(env, hand_handle, gymapi.STATE_POS)['pos'],
        num_dofs
    )

    # 转换为张量
    lower_limits_tensor = torch.tensor(lower_limits, dtype=torch.float32, device=sim_device)
    upper_limits_tensor = torch.tensor(upper_limits, dtype=torch.float32, device=sim_device)
    default_dof_state = gym.get_actor_dof_states(env, hand_handle, gymapi.STATE_POS)
    default_dof_pos_tensor = torch.tensor(default_dof_state['pos'].copy(), dtype=torch.float32, device=sim_device)

    # --- 查找关键 Link 的索引 ---
    fingertip_indices = find_key_link_indices(gym, env, hand_handle, FINGERTIP_LINK_NAMES, "Fingertip")
    palm_indices = find_key_link_indices(gym, env, hand_handle, [PALM_LINK_NAME], "Palm")
    palm_idx = palm_indices[0] if palm_indices else gymapi.INVALID_HANDLE
    fingertip_indices_tensor = torch.tensor(fingertip_indices, dtype=torch.long, device=sim_device)

    # --- 准备仿真和获取张量 ---
    gym.prepare_sim(sim)
    _rb_states = gym.acquire_rigid_body_state_tensor(sim)
    rb_states = gymtorch.wrap_tensor(_rb_states)
    print(f"\nRigid Body State Tensor Shape: {rb_states.shape}"); print("-" * 30)

    # --- 预计算用于运动的参数 ---
    amplitude_tensor = (upper_limits_tensor - lower_limits_tensor) / 2.0
    midpoint_tensor = (lower_limits_tensor + upper_limits_tensor) / 2.0
    amplitude_tensor = torch.max(amplitude_tensor, torch.zeros_like(amplitude_tensor))

    # --- 主循环 ---
    start_time = time.time()
    steps = 0
    print("\n--- Starting Simulation Loop (All fingers sweeping) ---")

    # 修改主循环条件，仅在非 headless 模式下检查 viewer 关闭
    while True:
        if not DEFAULT_HEADLESS and gym.query_viewer_has_closed(viewer):
            break

        # --- 计算当前时间的目标关节位置 ---
        current_time = time.time()
        elapsed_time = current_time - start_time
        angle = (2.0 * np.pi / SWEEP_DURATION) * elapsed_time
        offset = amplitude_tensor * torch.sin(torch.tensor(angle, device=sim_device))
        dof_pos_targets = midpoint_tensor + offset

        # 应用目标位置
        gym.set_actor_dof_position_targets(env, hand_handle, dof_pos_targets.cpu().numpy())

        # --- 步进仿真 ---
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # --- 刷新 Tensor ---
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)


        # --- 更新 Viewer (如果不是 headless) ---
        if not DEFAULT_HEADLESS:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
        # 如果是 headless 模式，这里可以加一个退出条件，例如运行固定步数或时间
        # elif steps > MAX_HEADLESS_STEPS: # 假设定义了 MAX_HEADLESS_STEPS
        #    break

        steps += 1

    # --- 清理 ---
    print("\n--- Simulation loop finished. Cleaning up. ---")
    if not DEFAULT_HEADLESS and viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == "__main__":
    main()