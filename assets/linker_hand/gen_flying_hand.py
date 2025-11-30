#!/usr/bin/env python3
"""
生成带六自由度浮空底座的灵巧手 URDF (Flying Hand for Pen Spinning)

针对转笔任务优化的参数设计：
- 极小工作空间：限制手腕在 10cm 立方体内微调，防止利用手臂惯性作弊
- 严格速度限制：极慢的线速度消除甩手腕产生离心力的可能性
- 适度旋转自由度：允许手腕配合做轻微的挑动动作

关节链结构：
world_virtual -> v_link_x (prismatic X)
             -> v_link_y (prismatic Y)  
             -> v_link_z (prismatic Z)
             -> v_link_roll (revolute X)
             -> v_link_pitch (revolute Y)
             -> base_link (revolute Z) -> [原始手部结构]
"""

import xml.etree.ElementTree as ET
import os

# ================= 配置区域 (转笔任务专用) =================
INPUT_FILE = "L25_dof_urdf.urdf"
OUTPUT_FILE = "L25_dof_urdf_flying.urdf"

# 1. 空间限位 (单位: 米)
# 解释：转笔时，手腕应该几乎静止，仅做微小的重心调整。
# 我们将活动范围限制在 10cm 的立方体内，且强制悬空。
LIMITS = {
    "x_lower": -0.05, "x_upper": 0.05,  # 前后仅允许移动 5cm
    "y_lower": -0.05, "y_upper": 0.05,  # 左右仅允许移动 5cm
    "z_lower":  0.30, "z_upper": 0.40,  # 高度强制固定在 30cm~40cm 之间 (桌面以上)
}

# 2. 速度限制 (防作弊配置)
VELOCITY = {
    "linear": "0.1",   # m/s (极慢，防止利用手臂惯性甩笔)
    "angular": "2.0"   # rad/s (约115度/秒，允许适度的腕部翻转以改变重力方向)
}

# 3. 驱动力 (N 或 Nm)
EFFORT = "100"  # 稍微降低扭矩上限，避免物理引擎出现过大的瞬时冲量
# ==========================================================


def create_virtual_link(name):
    """创建虚拟链接（几乎无质量的虚拟刚体）"""
    link = ET.Element("link", {"name": name})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass", {"value": "0.0001"})
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "inertia", {
        "ixx": "1e-6", "ixy": "0", "ixz": "0",
        "iyy": "1e-6", "iyz": "0", "izz": "1e-6"
    })
    return link


def create_virtual_joint(name, joint_type, parent, child, axis, limit_lower, limit_upper, velocity):
    """创建虚拟关节（平移或旋转）"""
    joint = ET.Element("joint", {"name": name, "type": joint_type})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(joint, "axis", {"xyz": axis})
    ET.SubElement(joint, "limit", {
        "lower": str(limit_lower),
        "upper": str(limit_upper),
        "effort": EFFORT,
        "velocity": velocity
    })
    return joint


def main():
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, INPUT_FILE)
    output_path = os.path.join(script_dir, OUTPUT_FILE)
    
    try:
        tree = ET.parse(input_path)
        robot = tree.getroot()
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_path}，请确保该文件存在。")
        return

    original_base_name = "base_link"
    
    # 定义虚拟链接链
    v_links = ["world_virtual", "v_link_x", "v_link_y", "v_link_z", "v_link_roll", "v_link_pitch"]
    
    # 定义关节链：
    # - 前三个是平移关节 (prismatic)，控制位置
    # - 后三个是旋转关节 (revolute)，控制姿态
    # 注意：Z 轴的 limit 已经硬编码了，确保手永远不会掉到地上
    v_joints = [
        # (名称, 类型, 父链接, 子链接, 轴向, 下限, 上限, 速度)
        ("virtual_px", "prismatic", v_links[0], v_links[1], "1 0 0", LIMITS["x_lower"], LIMITS["x_upper"], VELOCITY["linear"]),
        ("virtual_py", "prismatic", v_links[1], v_links[2], "0 1 0", LIMITS["y_lower"], LIMITS["y_upper"], VELOCITY["linear"]),
        ("virtual_pz", "prismatic", v_links[2], v_links[3], "0 0 1", LIMITS["z_lower"], LIMITS["z_upper"], VELOCITY["linear"]),
        ("virtual_rx", "revolute",  v_links[3], v_links[4], "1 0 0", "-1.57", "1.57", VELOCITY["angular"]),  # 限制翻转角度 +/- 90度
        ("virtual_ry", "revolute",  v_links[4], v_links[5], "0 1 0", "-1.57", "1.57", VELOCITY["angular"]),  # 限制俯仰角度
        ("virtual_rz", "revolute",  v_links[5], original_base_name, "0 0 1", "-3.14", "3.14", VELOCITY["angular"])  # 允许偏航全向旋转
    ]

    # 创建新的元素列表
    new_elements = []
    for link_name in v_links:
        new_elements.append(create_virtual_link(link_name))
    for j_params in v_joints:
        new_elements.append(create_virtual_joint(*j_params))

    # 将新元素插入到 URDF 的开头
    for elem in reversed(new_elements):
        robot.insert(0, elem)

    # 写入输出文件
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    
    print(f"\n{'='*60}")
    print(f"成功生成转笔专用 Flying Hand URDF: {OUTPUT_FILE}")
    print(f"{'='*60}")
    print(f"配置参数:")
    print(f"  - 移动范围 (XY平面): ±{abs(LIMITS['x_lower'])*100:.0f}cm")
    print(f"  - 悬浮高度: {LIMITS['z_lower']*100:.0f}cm ~ {LIMITS['z_upper']*100:.0f}cm")
    print(f"  - 线速度限制: {VELOCITY['linear']} m/s (已抑制甩动惯性)")
    print(f"  - 角速度限制: {VELOCITY['angular']} rad/s (约 {float(VELOCITY['angular'])*57.3:.0f}°/s)")
    print(f"\n新增自由度 (6 DoF):")
    print(f"  - virtual_px: 平移 X 轴")
    print(f"  - virtual_py: 平移 Y 轴")
    print(f"  - virtual_pz: 平移 Z 轴")
    print(f"  - virtual_rx: 旋转 Roll")
    print(f"  - virtual_ry: 旋转 Pitch")
    print(f"  - virtual_rz: 旋转 Yaw")
    print(f"\n总自由度: 6 (Flying Base) + 21 (Hand) = 27 DoF")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
