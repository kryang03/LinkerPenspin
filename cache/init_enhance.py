import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import os
from scipy.spatial.transform import Rotation as R # Import Rotation directly

# --- 文件路径定义 ---
# Assuming the script is run from /home/yang/Code/linker_penspin/cache/
# Adjust relative paths accordingly
#base_dir = "./allegro_round_tip_thin/pencil/"
base_dir = "./3pose/"
input_origin_file_path = os.path.join(base_dir, 'pencil/s03_50k.npy')
output_enhanced_file_path = os.path.join(base_dir, 'pencil/s03_100k.npy')
NUMDOF = 21

# --- 1. 加载 origin 数据并生成对称数据 ---
print(f"--- 开始加载和处理数据 ---")
try:
    print(f"正在加载文件: {input_origin_file_path}")
    grasp_data_origin = np.load(input_origin_file_path)
    print(f"成功加载 {input_origin_file_path}, 形状: {grasp_data_origin.shape}")

    # 提取原始物体四元数
    object_orientations_origin = grasp_data_origin[:, NUMDOF+3:NUMDOF+7] # [x, y, z, w]

    # 定义绕局部 X 轴旋转 180 度的四元数 [x, y, z, w]
    # Rotation = 180 deg = pi rad. Angle for quat = pi/2.
    # Axis = [1, 0, 0]. Quat = [sin(pi/2)*1, sin(pi/2)*0, sin(pi/2)*0, cos(pi/2)] = [1, 0, 0, 0]
    q_rot_x_180 = np.array([1.0, 0.0, 0.0, 0.0])

    # 计算对称四元数: q_sym = q_orig * q_rot_x
    print("正在计算对称四元数...")
    R_orig = R.from_quat(object_orientations_origin)
    R_rot_x = R.from_quat(q_rot_x_180)
    R_sym = R_orig * R_rot_x # Apply R_rot_x first, then R_orig
    symmetric_orientations = R_sym.as_quat()
    print(f"对称四元数计算完成, 形状: {symmetric_orientations.shape}")

    # 创建对称数据副本并替换四元数
    symmetric_data = grasp_data_origin.copy()
    symmetric_data[:, NUMDOF+3:NUMDOF+7] = symmetric_orientations

    # 拼接原始数据和对称数据
    print("正在拼接原始数据和对称数据...")
    grasp_data_enhanced = np.concatenate((grasp_data_origin, symmetric_data), axis=0)
    print(f"拼接完成，最终数据形状: {grasp_data_enhanced.shape}")

    # 保存 enhanced 数据
    print(f"正在保存合并后的 enhanced 数据到: {output_enhanced_file_path}")
    np.save(output_enhanced_file_path, grasp_data_enhanced)
    print(f"数据成功保存到 {output_enhanced_file_path}")

except FileNotFoundError as e:
    print(f"错误：未找到文件 {e.filename}")
    exit()
except Exception as e:
    print(f"处理数据时发生错误: {e}")
    exit()

print("--- 数据处理和保存完成 ---")
# --- 可复用代码模块：四元数旋转向量 ---
# 理论原理：四元数 q 旋转一个纯四元数形式的向量 v (即 [vx, vy, vz, 0]) 的公式是 q * v * q_conjugate
# 其中 * 表示四元数乘法，q_conjugate 是 q 的共轭四元数。
# 如果四元数表示为 q = w + xi + yi + zk，其共轭为 q_conjugate = w - xi - yi - zk。
# 四元数乘法 (a + bi + cj + dk) * (e + fi + gj + hk) 的结果是一个新的四元数，
# 其系数计算比较复杂，但可以通过矩阵乘法或直接展开得到。
# 这里我们实现一个直接的四元数-向量旋转函数。

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


