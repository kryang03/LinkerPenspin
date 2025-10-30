import numpy as np
import matplotlib.pyplot as plt

def calculate_hand_similarity(hand_pos_diff, scale_factor):
    """
    根据给定的手部位置差距和缩放因子，计算相似度奖励。
    
    参数:
    hand_pos_diff (np.array): 手部位置与目标的欧氏距离数组。
    scale_factor (float): 奖励函数中的缩放因子 (HAND_SIMILARITY_SCALE_FACTOR)。
    
    返回:
    np.array: 计算出的相似度分数。
    """
    return np.exp(-hand_pos_diff / scale_factor)

# --- 1. 设置绘图数据 ---

# 定义要可视化的手部位置差距范围（X轴），从0到15，共400个点以保证曲线平滑
hand_pos_diff_range = np.linspace(0, 15, 400)

# 定义不同的缩放因子进行对比，重点突出你正在使用的 2.0
scale_factors = {
    "your_setting": 2.0,
    "medium": 1.0,
    "0.8":0.8,
    "stricter": 0.5,
    "very_strict": 0.1
}

# --- 2. 创建并美化图表 ---

# 创建一个图形实例，设置尺寸
plt.figure(figsize=(12, 7))

# 循环遍历每一种缩放因子，计算并绘制对应的奖励曲线
for name, factor in scale_factors.items():
    # 使用上面定义的函数计算相似度分数 (Y轴)
    similarity_scores = calculate_hand_similarity(hand_pos_diff_range, factor)
    
    # 绘制曲线，并根据是否为你当前的设置选择不同的线型和标签
    if name == "your_setting":
        plt.plot(hand_pos_diff_range, similarity_scores, 
                 label=f'Scale Factor = {factor} (current setting)', 
                 linewidth=2.5, zorder=3)
    else:
        plt.plot(hand_pos_diff_range, similarity_scores, 
                 label=f'Scale Factor = {factor} (comparisons)', 
                 linestyle='--')

# # --- 3. 添加辅助线和注释来定位你的问题 ---

# # 在我们上次讨论中，当奖励约为0.002时，对应的相似度约为0.0021
# problematic_similarity = 0.0021
# # 这对应的手部位置差距约为12.34
# problematic_diff = 12.34

# # 添加水平虚线，标记出导致问题的相似度值
# plt.axhline(y=problematic_similarity, color='r', linestyle=':', linewidth=1.5, 
#             label=f'Similarity ≈ {problematic_similarity:.4f}')
# # 添加垂直虚线，标记出对应的巨大手部差距
# plt.axvline(x=problematic_diff, color='r', linestyle=':', linewidth=1.5, 
#             label=f'Difference ≈ {problematic_diff:.2f}')

# # 在交点处添加一个点和注释
# plt.scatter([problematic_diff], [problematic_similarity], color='red', zorder=5)
# plt.annotate(
#     '在你奖励为~0.002时\n手部差距在此处',
#     xy=(problematic_diff, problematic_similarity),
#     xytext=(problematic_diff - 6, problematic_similarity + 0.2),
#     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
#     fontsize=12,
#     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="k", lw=1, alpha=0.7)
# )

# --- 4. 完善图表信息 ---

plt.title('Hand Similarity vs. Hand Position Difference', fontsize=16)
plt.xlabel('Hand Position Difference (hand_pos_diff)', fontsize=12)
plt.ylabel('Hand Similarity Score', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=12)
plt.xlim(0, 15)
plt.ylim(0, 1.05)

# 显示图表
plt.show()