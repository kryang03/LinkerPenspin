import matplotlib.pyplot as plt
import io
import math # 用于 abs()，虽然Python内置有abs()，但导入math有时更明确意图或用于更多数学函数

# 定义原始数据文本块
allegro_text = """object_linvel_penalty: 0.16312168538570404
rotation_reward: -0.01824352703988552
pose_diff_penalty: 0.06915023922920227
torques: 0.03375497832894325
work_done: 0.023291941732168198
z_dist_penalty: 0.029642149806022644
penalty/position: 0.0004338900907896459
"""

linker_text = """object_linvel_penalty: 0.06777302175760269
rotation_reward: 0.15107642114162445
pose_diff_penalty: 0.2786386013031006
torques: 0.4679492712020874
work_done: 0.22148874402046204
z_dist_penalty: 0.025017142295837402
penalty/position: 0.00040985961095429957
"""
linker_text_bottleneck = """episode rewards: 26.76 | episode lengths: 84.55
step_all_reward: 0.3162657618522644
object_linvel_penalty: 0.02600140869617462
rotation_reward: 0.40371471643447876
pose_diff_penalty: 1.2841839790344238
torques: 0.46644091606140137
work_done: 0.03892243653535843
z_dist_penalty: 0.0061823115684092045
penalty/position: 0.0009341701515950263
penalty/finger_obj: 0.024396169930696487
roll: 0.3695509433746338
pitch: 0.5189700722694397
yaw: 0.5402278900146484
"""
# --- 可复用模块1: 解析文本数据为字典 ---
def parse_reward_text_to_dict(text):
    """Parses multiline reward text into a Python dictionary."""
    data = {}
    # 使用io.StringIO将字符串模拟成文件，或者直接按行分割
    # text_io = io.StringIO(text)
    # for line in text_io:
    # Or simply split by lines:
    for line in text.strip().split('\n'):
        line = line.strip()
        if line:
            # 按第一个冒号分割键和值
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value_str = parts[1].strip()
                try:
                    # 尝试将值转换为浮点数
                    data[key] = float(value_str)
                except ValueError:
                    print(f"Warning: Could not convert value for key '{key}' to float: '{value_str}'")
                    # 跳过无效行
            else:
                print(f"Warning: Skipping ill-formatted line: '{line}'")
    return data

# 解析数据
allegro_dict = parse_reward_text_to_dict(allegro_text)
linker_dict = parse_reward_text_to_dict(linker_text)

# 确保处理的 keys 是两组数据共有的，避免 key error
common_keys = list(allegro_dict.keys()) # 获取 allegro 的所有 key
# 过滤出同时存在于 linker 中的 key
common_keys = [key for key in common_keys if key in linker_dict]

# --- 计算缩放因子 ---
print("--- Linker 到 Allegro 的缩放因子 (Rough Scale Factors) ---")
scale_factors = {}
for key in common_keys:
    allegro_value = allegro_dict[key]
    linker_value = linker_dict[key]

    # 避免除以零
    if linker_value == 0:
        # 如果 linker 值为 0，无法直接计算缩放因子。
        # 根据 allegro 值，可以判断：
        # 如果 allegro 也为 0，因子可以是任何值，或者定义为 1 (1*0=0)
        # 如果 allegro 不为 0，则无穷大，意味着 0 无法缩放到非零值。
        # 这里我们输出提示信息并跳过或给一个特殊值。
        if allegro_value == 0:
             scale_factors[key] = 1.0 # 定义 0 -> 0 的缩放为 1
             print(f"{key}: Linker value is 0, Allegro is also 0. Scale factor set to 1.0")
        else:
             # 表示需要一个无穷大的缩放才能从 0 得到非零值
             scale_factors[key] = float('inf')
             print(f"{key}: Linker value is 0, Allegro is non-zero. Scale factor is infinite.")
    else:
        scale_factor = allegro_value / linker_value
        scale_factors[key] = scale_factor
        print(f"{key}: {scale_factor:.6f}") # 输出粗略值

print("-" * 40)

# --- 计算比例 ---
# 可复用模块2: 计算字典中各数值按绝对值计算占总和的比例
def calculate_proportions_dict(data_dict):
    """Calculates the proportion of each value in a dictionary relative to the sum of absolute values."""
    proportions = {}
    # 计算绝对值总和
    abs_sum = sum(abs(value) for value in data_dict.values())

    if abs_sum == 0:
        # 如果总和为零，所有比例都是零 (避免除以零)
        for key in data_dict.keys():
            proportions[key] = 0.0
    else:
        # 计算每个值的绝对值与总和的比例
        for key, value in data_dict.items():
            proportions[key] = abs(value) / abs_sum

    return proportions

# 计算比例
# 注意：这里我们基于 common_keys 来处理，以确保比例计算是基于同一组 reward
allegro_common_dict = {key: allegro_dict[key] for key in common_keys}
linker_common_dict = {key: linker_dict[key] for key in common_keys}

allegro_proportions = calculate_proportions_dict(allegro_common_dict)
linker_original_proportions = calculate_proportions_dict(linker_common_dict)
# 缩放后的 Linker 就是 Allegro 的值，所以缩放后 Linker 的比例就是 Allegro 的比例
linker_scaled_proportions = allegro_proportions

# --- 可视化比例 ---
# 可复用模块3: 绘制饼图 (使用从字典提取的数据)
def plot_pie_chart_from_dict(proportions_dict, title):
    """Plots a pie chart for the given proportions dictionary."""
    # 从字典中提取 values (比例数值) 和 keys (标签)
    labels = list(proportions_dict.keys())
    sizes = list(proportions_dict.values())

    # 过滤掉比例为0的项目，避免饼图过于拥挤
    non_zero_sizes = [size for size in sizes if size > 0]
    non_zero_labels = [label for label, size in zip(labels, sizes) if size > 0]

    plt.figure(figsize=(10, 8)) # 设置图表大小
    # autopct='%1.1f%%' 显示百分比
    # startangle=140 使饼图起始位置更美观
    # labels=non_zero_labels 使用过滤后的标签
    # pctdistance=0.85 使百分比显示靠近饼图中心一些 (可选)
    # wedgeprops=dict(width=0.3) 可以做成圆环图 (可选)
    plt.pie(non_zero_sizes, labels=non_zero_labels, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 9})
    plt.title(title, fontsize=14)
    plt.axis('equal')  # 确保饼图是圆的
    plt.tight_layout() # 自动调整布局，避免标签重叠

# 绘制三个饼图
plot_pie_chart_from_dict(allegro_proportions, 'Allegro Reward Proportions (Based on Absolute Value)')
plot_pie_chart_from_dict(linker_original_proportions, 'Original Linker Reward Proportions (Based on Absolute Value)')
plot_pie_chart_from_dict(linker_scaled_proportions, 'Scaled Linker Reward Proportions (i.e., Allegro) (Based on Absolute Value)')

# 显示图表
plt.show()