import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
读取一个csv文件并绘制这个图的线性图
"""

def visualize_training_results(csv_file):
    """
    读取训练结果 CSV 文件并可视化多个评估指标的变化趋势。

    参数：
    csv_file (str): 包含训练结果的 CSV 文件路径
    """

    # 读取 CSV 文件并存入 DataFrame
    df = pd.read_csv(csv_file)

    # 确保 'model_file' 列是字符串类型，以便正确显示在 X 轴上
    df['model_file'] = df['model_file'].astype(str)

    # 创建一个新的绘图窗口，设置图像大小
    plt.figure(figsize=(10, 6))

    # 选择间隔：每 20 个数据点显示一个 X 轴标签
    step = 20
    x_labels = df['model_file'][::step]  # 每 20 个点取一个

    # 绘制不同指标的折线图，并用虚线连接中间省略的点
    plt.plot(df['model_file'], df['avg_loss'], label='Avg Loss', marker='o', linestyle='dashed')
    plt.plot(df['model_file'], df['avg_dice'], label='Avg Dice', marker='s', linestyle='dashed')
    plt.plot(df['model_file'], df['avg_iou'], label='Avg IoU', marker='^', linestyle='dashed')
    plt.plot(df['model_file'], df['avg_pixel_acc'], label='Avg Pixel Acc', marker='x', linestyle='dashed')

    # 设置标题和坐标轴标签
    plt.title('Model Training Results')  # 图表标题
    plt.xlabel('Model File')  # X 轴标签
    plt.ylabel('Metric Values')  # Y 轴标签

    # 旋转 X 轴刻度，使其更易读，仅保留部分刻度
    plt.xticks(x_labels, rotation=45, ha='right')

    # 获取当前 Y 轴的刻度范围，并减少网格线的数量
    ax = plt.gca()  # 获取当前坐标轴
    y_min, y_max = ax.get_ylim()  # 获取 Y 轴范围
    y_ticks = np.linspace(y_min, y_max, num=10)  # 生成 10 个均匀间隔的刻度
    ax.set_yticks(y_ticks)  # 设置 Y 轴刻度
    ax.grid(True, linestyle='--', alpha=0.6, which='major')  # 仅对主刻度线显示网格

    # 显示图例，标明不同曲线对应的指标
    plt.legend()

    # 调整布局，以避免标签重叠
    plt.tight_layout()

    # 显示绘制的图表
    plt.show()


# 使用示例（请替换成你的 CSV 文件路径）
csv_file_path = 'answ/unet_isic2018/evaluation_results.csv'
visualize_training_results(csv_file_path)
