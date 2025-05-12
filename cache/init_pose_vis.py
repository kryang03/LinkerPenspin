import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import os
from scipy.spatial.transform import Rotation as R # Import Rotation directly

# --- 可复用代码模块：四元数旋转向量 ---
# 理论原理：四元数 q 旋转一个纯四元数形式的向量 v (即 [vx, vy, vz, 0]) 的公式是 q * v * q_conjugate
# 其中 * 表示四元数乘法，q_conjugate 是 q 的共轭四元数。
# 如果四元数表示为 q = w + xi + yi + zk，其共轭为 q_conjugate = w - xi - yi - zk。
# 四元数乘法 (a + bi + cj + dk) * (e + fi + gj + hk) 的结果是一个新的四元数，
# 其系数计算比较复杂，但可以通过矩阵乘法或直接展开得到。
# 这里我们实现一个直接的四元数-向量旋转函数。

# --- 文件路径定义 ---
output_image_path = 's03_50k.png'
xy_output_image_path = 's03_50k_.png'
# 更新要加载的文件路径
npy_file_path = '/home/yang/Code/linker_penspin/cache/3pose/pencil/s03_50k.npy'

def rotate_vector_by_quaternion(vector, quaternion):
    """
    使用四元数旋转一个 3D 向量。
    假设四元数格式为 [x, y, z, w]。
    向量格式为 [vx, vy, vz]。

    Args:
        vector (np.ndarray): 形状为 (3,) 或 (N, 3) 的 3D 向量。
        quaternion (np.ndarray): 形状为 (4,) 或 (N, 4) 的四元数 [x, y, z, w]。

    Returns:
        np.ndarray: 旋转后的向量，形状与输入向量相同。
    """
    # scipy 的 from_quat 接受 [x, y, z, w] 格式
    # Ensure quaternion is numpy array for Rotation
    if not isinstance(quaternion, np.ndarray):
        quaternion = np.array(quaternion)
    if quaternion.ndim == 1: # Handle single quaternion case
        quaternion = quaternion[np.newaxis, :]
    if vector.ndim == 1: # Handle single vector case
        vector = vector[np.newaxis, :]
        rotation = R.from_quat(quaternion)
        rotated_vector = rotation.apply(vector)
        return rotated_vector.squeeze() # Return single vector
    else:
        rotation = R.from_quat(quaternion)
        rotated_vector = rotation.apply(vector)
        return rotated_vector


teacher_obj_rot=np.array([
                 [-0.5389097929000854, -0.5710364580154419, 0.5835363864898682, -0.20731335878372192],
                 [0.5535288453102112, -0.5571686029434204, 0.36139827966690063, -0.5025537014007568],
                 [-0.703499972820282, -0.061799999326467514, 0.0737999975681305, 0.704200029373169],
                 [-0.6355999708175659, -0.24799999594688416, 0.4408999979496002, 0.583299994468689],
                 [0.7116000056266785, 0.15160000324249268, -0.0754999965429306, -0.6818000078201294], # Corrected last element based on markdown
                 [-0.26589998602867126, -0.6491000056266785, 0.6674000024795532, 0.25],
                 [-0.6159, -0.2865, 0.2555, 0.6880]])



# 加载 50k 数据
try:
    grasp_data = np.load(npy_file_path) # Renaming to grasp_data for consistency below
    print(f"\n成功加载增强后的数据文件: {npy_file_path}")
    print(f"数据形状: {grasp_data.shape}")
except FileNotFoundError:
    print(f"错误：未找到增强后的文件 {npy_file_path}")
    exit()
except Exception as e:
    print(f"加载增强后的数据文件时发生错误: {e}")
    exit()

# 提取物体位姿 (列 21-27)
object_poses = grasp_data[:, 21:28]

# 提取物体四元数 (列 24-27)
object_orientations = object_poses[:, 3:7] # [x, y, z, w] 格式
object_pos = object_poses[:, :3] # [x, y, z] 格式
# --- 可视化物体朝向分布 (50k 数据 + Teacher) ---

print(f"\n正在生成物体朝向分布可视化 (50k) 并保存到 {output_image_path} ...")
print(f"物体平均位置: {np.mean(object_pos, axis=0).tolist()}")
# 假设笔的主轴在物体局部坐标系下沿着 Z 轴正方向 [0, 0, 1]
local_pen_axis = np.array([0.0, 0.0, 1.0])

# 将局部主轴向量通过物体四元数旋转到世界坐标系 (50k 数据)
rotated_pen_axes = rotate_vector_by_quaternion(local_pen_axis, object_orientations)

# 计算 Teacher 四元数对应的世界坐标系 Z 轴方向
teacher_rotated_axes = rotate_vector_by_quaternion(local_pen_axis, teacher_obj_rot)


# 绘制朝向分布
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制 50k 旋转后的向量的端点 (用较浅颜色和透明度)
ax.scatter(rotated_pen_axes[:, 0], rotated_pen_axes[:, 1], rotated_pen_axes[:, 2], s=1, alpha=0.1, label='Generated Data (50k)', color='blue')

# 绘制原点
ax.scatter(0, 0, 0, color='red', s=50, label='Origin')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Object Orientation Distribution (World Frame Z-axis) - s03') # 更新标题
ax.legend()

# 设置坐标轴比例一致
ax.set_box_aspect([1, 1, 1])

# 设置视角与 Teacher 图一致
ax.view_init(elev=20., azim=30)

# 获取此图的坐标轴范围
x_lim = ax.get_xlim()
y_lim = ax.get_ylim()
z_lim = ax.get_zlim()

# 保存图表到文件
try:
    plt.savefig(output_image_path)
    print(f"朝向分布图 (50k) 已成功保存到 {output_image_path}") # 更新打印信息
except Exception as e:
    print(f"保存图片时发生错误: {e}")

# 关闭图表，释放内存
plt.close(fig)


# --- 单独可视化 Teacher 物体朝向分布 ---

print(f"\n正在单独生成 Teacher 物体朝向分布可视化并保存到 teacher_orientation_distribution.png ...")

# 创建新的 3D 图表
fig_teacher = plt.figure(figsize=(8, 8))
ax_teacher = fig_teacher.add_subplot(111, projection='3d')

# 绘制 Teacher 四元数对应的点 (使用之前计算的 teacher_rotated_axes)
ax_teacher.scatter(teacher_rotated_axes[:, 0], teacher_rotated_axes[:, 1], teacher_rotated_axes[:, 2], s=80, alpha=1.0, label='Teacher Orientations', color='red', edgecolor='black')

# 绘制原点
ax_teacher.scatter(0, 0, 0, color='blue', s=50, label='Origin')

# 设置坐标轴标签
ax_teacher.set_xlabel('X')
ax_teacher.set_ylabel('Y')
ax_teacher.set_zlabel('Z')
ax_teacher.set_title('Teacher Object Orientation Distribution (World Frame Z-axis)')
ax_teacher.legend()

# 设置坐标轴比例一致
ax_teacher.set_box_aspect([1, 1, 1])

# 设置视角以便更好地观察
ax_teacher.view_init(elev=20., azim=30)

# !! 强制设置坐标轴范围与 50k 图一致 !!
ax_teacher.set_xlim(x_lim)
ax_teacher.set_ylim(y_lim)
ax_teacher.set_zlim(z_lim)

# 保存单独的 Teacher 图表
teacher_output_image_path = 'teacher_orientation_distribution.png'
try:
    plt.savefig(teacher_output_image_path)
    print(f"单独的 Teacher 朝向分布图已成功保存到 {teacher_output_image_path}")
except Exception as e:
    print(f"保存单独 Teacher 图片时发生错误: {e}")

# 关闭图表
plt.close(fig_teacher)


# --- 可视化物体朝向在 X-Y 平面上的投影分布 (50k 数据) ---

print(f"\n正在生成物体朝向在 X-Y 平面投影的分布图 (50k) 并保存到 {xy_output_image_path}...")

# 提取旋转后向量的 X 和 Y 分量 (50k 数据)
xy_projections = rotated_pen_axes[:, :2]

# 计算每个投影向量与 X 轴正方向的夹角 (弧度)
angles_rad = np.arctan2(xy_projections[:, 1], xy_projections[:, 0])

# 将弧度转换为角度，范围是 [-180, 180]
angles_deg = np.degrees(angles_rad)

# 创建新的图表
fig_xy, ax_xy = plt.subplots(figsize=(8, 8))

# 绘制角度分布的直方图
n_bins = 72 # Increase bins for potentially denser distribution
ax_xy.hist(angles_deg, bins=n_bins, range=(-180, 180), density=True, alpha=0.7, color='skyblue', edgecolor='black')

# 设置图表标题和标签
ax_xy.set_title('Object Orientation Distribution (Projected onto XY Plane) - s03') # 更新标题
ax_xy.set_xlabel('Angle (degrees) relative to positive X-axis')
ax_xy.set_ylabel('Density')
ax_xy.set_xticks(np.arange(-180, 181, 45)) # 设置 X 轴刻度
ax_xy.grid(axis='y', linestyle='--')

# 保存新的图表
try:
    plt.savefig(xy_output_image_path)
    print(f"X-Y 平面朝向分布图 (50k) 已成功保存到 {xy_output_image_path}")
except Exception as e:
    print(f"保存 X-Y 平面分布图时发生错误: {e}")

# 关闭图表
plt.close(fig_xy)


# --- 实现随机选取并呈现姿态的功能 ---

def select_and_present_random_pose(data):
    """
    从数据中随机选取一个姿态并呈现其详细信息。

    Args:
        data (np.ndarray): 形状为 (N, 28) 的抓取姿态数据。
    """
    num_poses = data.shape[0]
    if num_poses == 0:
        print("数据中没有姿态可供选取。")
        return

    # 随机选取一个索引
    random_index = random.randint(0, num_poses - 1)
    print(f"\n随机选取的姿态索引: {random_index}")

    # 获取选取的姿态数据
    selected_pose = data[random_index, :]

    # 提取手部关节角度和物体位姿
    hand_dof_pos = selected_pose[:21]
    object_pose = selected_pose[21:]
    object_position = object_pose[:3]
    object_quaternion = object_pose[3:7] # [x, y, z, w]

    # 呈现详细信息
    print("\n--- 选取的姿态详细信息 ---")
    print("手部关节角度 (21 DOF):")
    print(hand_dof_pos)
    print("\n物体位姿:")
    print(f"  位置 (x, y, z): {object_position}")
    print(f"  朝向 (四元数 x, y, z, w): {object_quaternion}")
    print("---------------------------")

# 调用函数进行随机选取和呈现 (现在使用 50k grasp_data)
# print("\n--- 随机选取并呈现一个姿态 (来自 s03) ---")
# select_and_present_random_pose(grasp_data)

print("\n--- 脚本执行完毕 ---")