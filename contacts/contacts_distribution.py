import re
import matplotlib.pyplot as plt
from collections import defaultdict
# python contacts/contacts_distribution.py

filename = '/home/yang/Code/linker_penspin/contacts/pose4_50k.txt'  # 替换为你的文件名
CONTACT_THRESH = 0.01

def analyze_force_distribution(filename):
    """
    统计文件中 Fx, Fy, Fz 值的分布，并生成直方图。

    Args:
        filename (str): 包含力数据的文本文件路径。

    Returns:
        dict: 包含 Fx, Fy, Fz 值列表的字典。
    """

    force_values = defaultdict(list)
    try:
        with open(filename, 'r') as file:
            for line in file:
                if 'Fx:' in line and 'Fy:' in line and 'Fz:' in line:
                    # 使用正则表达式提取 Fx, Fy, Fz 的数值
                    # 组一 ([Ff][xyz]) 匹配 "Fx"、"Fy" 或 "Fz"
                    # \s* 匹配零个或多个空白字符
                    # 组二 (-?\d+\.\d+|\-?\d+) 匹配浮点数或整数，包括正负号
                    matches = re.findall(r'([Ff][xyz]):\s*(-?\d+\.\d+|\-?\d+)', line)
                    if matches:
                        for force, value in matches:
                            if force in ['Fx', 'Fy', 'Fz']:
                                # 仅当值大于 CONTACT_THRESH 时才添加到列表中
                                if float(value) > CONTACT_THRESH:
                                    # 将值转换为浮点数并添加到对应的列表中
                                    force_values[force].append(float(value))
    except FileNotFoundError:
        return "Error: File not found."

    return force_values

def plot_force_distribution(force_data):
    """
    生成 Fx, Fy, Fz 值的直方图。

    Args:
        force_data (dict): 包含 Fx, Fy, Fz 值列表的字典。
    """

    plt.figure(figsize=(15, 5))  # 创建一个大的图像，容纳三个子图

    for i, (force, values) in enumerate(force_data.items()):
        plt.subplot(1, 3, i + 1)  # 创建子图：1行3列，当前第 i+1 个
        plt.hist(values, bins=50, color='skyblue', edgecolor='black')  # 绘制直方图
        plt.title(f'Distribution of {force}')
        plt.xlabel('Force Value')
        plt.ylabel('Frequency')

    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.show()  # 显示图像

result = analyze_force_distribution(filename)

if isinstance(result, dict):
    plot_force_distribution(result)
else:
    print(result)