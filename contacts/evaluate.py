import re
import os

def calculate_average_rotation(file_path):
    """
    从单个txt文件中提取包含旋转圈数的行，并计算平均旋转圈数。

    参数:
        file_path (str): txt文件的路径。

    返回:
        float: 平均旋转圈数，如果未找到相关行或文件不存在则返回0.0。
               返回一个元组 (average_rotation, status_message) 方便调试
    """
    rotation_counts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if "本轮累计转笔圈数" in line:
                    # 使用正则表达式提取冒号后的数字（可能包含小数）
                    match = re.search(r"本轮累计转笔圈数:\s*(-?\d+\.?\d*)", line)
                    if match:
                        try:
                            if (float(match.group(1)) > 0.05):
                                rotation_counts.append(float(match.group(1)))
                        except ValueError:
                            return 0.0, f"警告：在文件 {os.path.basename(file_path)} 的行 '{line.strip()}' 中找到非数字的旋转圈数值。"
        
        if not rotation_counts:
            return 0.0, f"信息：在文件 {os.path.basename(file_path)} 中未找到包含“本轮累计转笔圈数”的有效数据行。"

        return sum(rotation_counts) / len(rotation_counts), f"成功处理文件 {os.path.basename(file_path)}，找到 {len(rotation_counts)} 条记录。"

    except FileNotFoundError:
        return 0.0, f"错误：文件 {os.path.basename(file_path)} 未找到。"
    except Exception as e:
        return 0.0, f"处理文件 {os.path.basename(file_path)} 时发生错误: {e}"


def process_all_files(base_file_names):
    """
    统一处理所有指定的txt文件，计算每个文件对应的平均旋转圈数。

    参数:
        base_file_names (list): 包含txt文件基本名（不含.txt后缀）的列表。

    返回:
        dict: 一个字典，键是文件的基本名，值是对应的平均旋转圈数。
    """
    results = {}
    print("--- 开始处理文件 ---")
    for base_name in base_file_names:
        file_path = base_name + ".txt"  # 假设所有文件都以 .txt 结尾
        
        average_rotation, message = calculate_average_rotation(file_path)
        results[base_name] = average_rotation
        print(message)
        if average_rotation > 0 : # 只打印有实际结果的平均值，或者根据需要调整
             print(f"文件 '{base_name}.txt' 的计算平均旋转圈数: {average_rotation:.2f}")


    print("--- 文件处理结束 ---")
    return results

# --- 主程序 ---
if __name__ == "__main__":
    # 统一的文件基本名列表（不含.txt后缀）
    # 这包括了您上传的文件名（不带后缀）和图片中看到的文件名
    all_base_file_names = [
        "pose3_50k_cfg2",    # 这是您上传的文件
        "pose3_50k_cfg0",
        "pose3_100k_cfg0",
        "pose3_100k_cfg1",
        "pose3_100k_cfg2",   # 注意：如果列表中有重复，process_all_files会处理它，但结果字典中键是唯一的
        "pose4_50k_cfg0",
        "pose4_50k_cfg2",
        "pose4_100k_cfg0",
        "pose4_100k_cfg2",
        "pose6_50k_cfg1"
    ]
    
    # 使用 set 去除可能重复的条目，然后再转回 list (保持顺序可选，但对于这里不重要)
    unique_base_file_names = sorted(list(set(all_base_file_names)))


    results_dict = process_all_files(unique_base_file_names)

    print("\n--- 所有文件的平均旋转圈数汇总 ---")
    # 为了更美观的输出，可以遍历字典打印
    if results_dict:
        for file_base_name, avg_rot in results_dict.items():
            print(f"文件名: {file_base_name}.txt, 平均旋转圈数: {avg_rot:.2f}")
    else:
        print("没有处理任何文件或所有文件均未找到数据。")

    # 示例输出 (假设 pose3_50k_cfg2.txt 存在且包含数据, 其他文件可能不存在或无数据):
    # --- 开始处理文件 ---
    # 成功处理文件 pose3_50k_cfg2.txt，找到 X 条记录。
    # 文件 'pose3_50k_cfg2.txt' 的计算平均旋转圈数: Y.YY
    # 错误：文件 pose3_50k_cfg0.txt 未找到。
    # ... (其他文件的处理信息)
    # --- 文件处理结束 ---
    #
    # --- 所有文件的平均旋转圈数汇总 ---
    # 文件名: pose3_50k_cfg2.txt, 平均旋转圈数: Y.YY
    # 文件名: pose3_50k_cfg0.txt, 平均旋转圈数: 0.00
    # ...