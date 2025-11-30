import numpy as np
import os

# --- 物理常数与设计参数 ---
# 长度单位: 米 (m), 质量单位: 千克 (kg)

# 1. 尼龙主体 (Nylon Body)
L_BODY = 0.15           # 15 cm
R_BODY_OUT = 0.009      # 18 mm OD
R_BODY_IN = 0.006       # 12 mm ID
DENSITY_NYLON = 1150.0

# 2. 铝制配重 (Alu Tips)
L_TIP = 0.015           # 1.5 cm
R_TIP_OUT = 0.009       # 18 mm OD
# 为了模拟钻孔减重，我们定义减重孔
R_TIP_HOLE = 0.004      # 8 mm 孔径 (半径 4mm)
L_TIP_HOLE = 0.015      # 钻孔深度 (贯穿或部分)
DENSITY_ALU = 2700.0

# 3. 连接参数
# 假设螺柱增加的质量大约被钻孔抵消一部分，为了模拟高精度，
# 我们通常稍微增加一点Tip的密度或质量来模拟螺丝，这里我们使用纯几何计算
# 你可以通过这个因子微调总质量
MASS_CORRECTION_FACTOR = 1.0 

def cylinder_inertia(m, r_out, r_in, h):
    """
    计算空心圆柱(或实心 r_in=0)的惯性张量 (对质心)
    Returns: (ixx, iyy, izz) note: ixx=iyy for cylinder
    """
    if m <= 1e-6: return 0.0, 0.0, 0.0
    
    ixx = (1/12) * m * (3 * (r_out**2 + r_in**2) + h**2)
    iyy = ixx
    izz = 0.5 * m * (r_out**2 + r_in**2)
    return ixx, iyy, izz

def generate_urdf(filename="spinning_pen.urdf"):
    # --- A. 计算尼龙主体 ---
    vol_body_solid = np.pi * R_BODY_OUT**2 * L_BODY
    vol_body_air = np.pi * R_BODY_IN**2 * L_BODY
    mass_body = (vol_body_solid - vol_body_air) * DENSITY_NYLON
    
    ib_xx, ib_yy, ib_zz = cylinder_inertia(mass_body, R_BODY_OUT, R_BODY_IN, L_BODY)

    # --- B. 计算铝头 (使用叠加原理: 实心 - 孔洞) ---
    # 实心部分
    vol_tip_solid = np.pi * R_TIP_OUT**2 * L_TIP
    m_tip_solid = vol_tip_solid * DENSITY_ALU
    i_solid_xx, _, i_solid_zz = cylinder_inertia(m_tip_solid, R_TIP_OUT, 0, L_TIP)
    
    # 孔洞部分 (负质量)
    vol_tip_hole = np.pi * R_TIP_HOLE**2 * L_TIP_HOLE
    m_tip_hole = vol_tip_hole * DENSITY_ALU
    i_hole_xx, _, i_hole_zz = cylinder_inertia(m_tip_hole, R_TIP_HOLE, 0, L_TIP_HOLE)
    
    # 合成 Tip 属性
    mass_tip = (m_tip_solid - m_tip_hole) * MASS_CORRECTION_FACTOR
    # 惯量线性叠加 (假设孔洞和实心同轴且同心，这是一个合理的简化)
    it_xx = (i_solid_xx - i_hole_xx) * MASS_CORRECTION_FACTOR
    it_yy = it_xx
    it_zz = (i_solid_zz - i_hole_zz) * MASS_CORRECTION_FACTOR

    # --- C. 汇总数据 ---
    total_mass = mass_body + 2 * mass_tip
    print(f"=== Physics Summary ===")
    print(f"Body Mass: {mass_body*1000:.2f} g")
    print(f"Tip Mass : {mass_tip*1000:.2f} g (each)")
    print(f"Total Mass: {total_mass*1000:.2f} g")
    print(f"Total Length: {(L_BODY + 2*L_TIP)*100:.2f} cm")
    print(f"I_transverse (Body): {ib_xx:.2e}")
    print(f"I_transverse (Tip) : {it_xx:.2e}")
    print(f"=======================")

    # --- D. 生成 URDF 字符串 ---
    # Isaac Gym 注意事项:
    # 1. 尽量使用基本的 <cylinder> collision，比 mesh 稳定且快。
    # 2. 视觉可以和碰撞体分开，但为了简单这里保持一致。
    # 3. 摩擦系数 (mu) 在 URDF 中通常不直接被 Isaac Gym 解析，
    #    需要在 Python 代码的 asset_options 中设置，但这里保留标准格式。

    urdf_content = f"""<?xml version="1.0"?>
<robot name="spinning_pen">

  <!-- Materials -->
  <material name="nylon_white">
    <color rgba="0.95 0.95 0.92 1.0"/>
  </material>
  <material name="aluminum_grey">
    <color rgba="0.75 0.76 0.78 1.0"/>
  </material>

  <!-- 1. BODY LINK -->
  <link name="body_link">
    <inertial>
      <mass value="{mass_body:.5f}"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="{ib_xx:.8f}" ixy="0" ixz="0" iyy="{ib_yy:.8f}" iyz="0" izz="{ib_zz:.8f}"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder radius="{R_BODY_OUT}" length="{L_BODY}"/>
      </geometry>
      <material name="nylon_white"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="{R_BODY_OUT}" length="{L_BODY}"/>
      </geometry>
    </collision>
  </link>

  <!-- 2. LEFT TIP -->
  <link name="left_tip_link">
    <inertial>
      <mass value="{mass_tip:.5f}"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="{it_xx:.8f}" ixy="0" ixz="0" iyy="{it_yy:.8f}" iyz="0" izz="{it_zz:.8f}"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder radius="{R_TIP_OUT}" length="{L_TIP}"/>
      </geometry>
      <material name="aluminum_grey"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="{R_TIP_OUT}" length="{L_TIP}"/>
      </geometry>
    </collision>
  </link>

  <!-- 3. RIGHT TIP -->
  <link name="right_tip_link">
    <inertial>
      <mass value="{mass_tip:.5f}"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="{it_xx:.8f}" ixy="0" ixz="0" iyy="{it_yy:.8f}" iyz="0" izz="{it_zz:.8f}"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder radius="{R_TIP_OUT}" length="{L_TIP}"/>
      </geometry>
      <material name="aluminum_grey"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="{R_TIP_OUT}" length="{L_TIP}"/>
      </geometry>
    </collision>
  </link>

  <!-- JOINTS (Fixed) -->
  <!-- Joint offset calculation: Body Length/2 + Tip Length/2 -->
  <joint name="body_to_left" type="fixed">
    <parent link="body_link"/>
    <child link="left_tip_link"/>
    <origin xyz="0 0 {-(L_BODY/2 + L_TIP/2)}" rpy="0 0 0"/>
  </joint>

  <joint name="body_to_right" type="fixed">
    <parent link="body_link"/>
    <child link="right_tip_link"/>
    <origin xyz="0 0 {(L_BODY/2 + L_TIP/2)}" rpy="0 0 0"/>
  </joint>

</robot>
"""
    
    with open(filename, 'w') as f:
        f.write(urdf_content)
    print(f"Successfully generated: {os.path.abspath(filename)}")

if __name__ == "__main__":
    generate_urdf()