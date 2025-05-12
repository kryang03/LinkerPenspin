import numpy as np
import os

# --- 文件路径定义 ---
# Assuming the script is run from /home/yang/Code/linker_penspin/cache/
# Adjust relative paths accordingly
base_dir = "./tmp/"
input_a_file_path = os.path.join(base_dir, 's03_18540.npy')
input_b_file_path = os.path.join(base_dir, 's03_b.npy') # Added path for b file
output_file_path = os.path.join(base_dir, 's03_a.npy')
SHAPE = 25000 #最终数据条数
# --- 加载、拼接和保存数据 ---
print(f"--- 开始加载和拼接数据 ---")
try:
    print(f"正在加载文件: {input_a_file_path}")
    grasp_data_a = np.load(input_a_file_path)
    print(f"成功加载 {input_a_file_path}, 形状: {grasp_data_a.shape}")

    print(f"正在加载文件: {input_b_file_path}")
    grasp_data_b = np.load(input_b_file_path)
    print(f"成功加载 {input_b_file_path}, 形状: {grasp_data_b.shape}")

    # 拼接数据
    print("正在拼接  数据...")
    grasp_data_  = np.concatenate((grasp_data_a, grasp_data_b), axis=0)
    print(f"拼接完成，最终数据形状: {grasp_data_ [:SHAPE,:].shape}")

    # 保存   数据
    print(f"正在保存合并后的   数据到: {output_file_path}")
    np.save(output_file_path, grasp_data_ )
    print(f"数据成功保存到 {output_file_path}")

except FileNotFoundError as e:
    print(f"错误：未找到文件 {e.filename}")
    exit()
except Exception as e:
    print(f"处理数据时发生错误: {e}")
    exit()

print("--- 数据拼接和保存完成 ---")