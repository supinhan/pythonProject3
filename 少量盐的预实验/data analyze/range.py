import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.font_manager as fm
import warnings
import os
import matplotlib

# 设置环境
matplotlib.use('Agg')
warnings.filterwarnings("ignore")


def configure_fonts():
    """配置中文字体支持"""
    font_families = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'Arial']
    try:
        mpl.font_manager._rebuild()
    except AttributeError:
        try:
            fm._rebuild()
        except AttributeError:
            font_cache_path = mpl.get_cachedir()
            for f in os.listdir(font_cache_path):
                if f.startswith('fontlist'):
                    os.remove(os.path.join(font_cache_path, f))
            fm._load_fontmanager(try_read_cache=False)
    plt.rcParams['font.family'] = font_families
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    print(f"已设置字体栈: {font_families}")


# 配置字体
configure_fonts()

# 加载数据
desired_columns = pd.read_csv('广精样本.csv', nrows=0).columns.tolist()
df = pd.read_csv('广精样本.csv', usecols=desired_columns)
y = pd.to_numeric(df.iloc[:, 0], errors='coerce')


def plot_y_distribution(y, title='因变量Y分布分析', figsize=(12, 8), save_path='y_distribution.png'):
    """
    绘制因变量y的数量分布图（无小点版本）

    参数:
    y -- 因变量数组/Series
    title -- 图表标题
    figsize -- 图表尺寸
    save_path -- 图片保存路径
    """
    plt.figure(figsize=figsize)

    # 计算统计量
    y_clean = y.dropna()
    mean_val = np.mean(y_clean)
    median_val = np.median(y_clean)
    min_val = np.min(y_clean)
    max_val = np.max(y_clean)
    std_val = np.std(y_clean)
    skew_val = y_clean.skew()
    kurt_val = y_clean.kurtosis()

    # 创建分布图 - 只保留直方图+KDE曲线，移除箱线图
    ax = sns.histplot(y_clean, kde=True, color='royalblue', bins=30,
                      edgecolor='white', linewidth=1.2, alpha=0.8)

    stats_text = (f"样本数: {len(y_clean)}\n"
                  f"最小值: {min_val:.2f}\n"
                  f"最大值: {max_val:.2f}\n"
                  f"标准差: {std_val:.2f}\n"
                  f"偏度: {skew_val:.2f}\n"
                  f"峰度: {kurt_val:.2f}\n"
                  f"缺失值: {len(y) - len(y_clean)}")

    plt.text(0.95, 0.95, stats_text, transform=ax.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=12)

    # 设置标题和标签 - 根据图片调整
    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel('k', fontsize=14)  # 根据图片改为'k'
    plt.ylabel('Count', fontsize=14)  # 根据图片改为'Count'
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.3)

    # 设置坐标轴范围，匹配图片
    plt.xlim(0, 20)  # 根据图片x轴范围

    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"已保存分布图至: {save_path}")

    return {
        'mean': mean_val,
        'median': median_val,
        'min': min_val,
        'max': max_val,
        'std': std_val,
        'skew': skew_val,
        'kurtosis': kurt_val,
        'missing': len(y) - len(y_clean)
    }


# 绘制并保存分布图
stats = plot_y_distribution(y, title='因变量Y分布分析', save_path='y_count.png')

# 打印统计信息
print("\n因变量Y统计信息:")
print(f"样本数: {len(y.dropna())}")
print(f"均值: {stats['mean']:.4f}")
print(f"中位数: {stats['median']:.4f}")
print(f"最小值: {stats['min']:.4f}")
print(f"最大值: {stats['max']:.4f}")
print(f"标准差: {stats['std']:.4f}")
print(f"偏度: {stats['skew']:.4f}")
print(f"峰度: {stats['kurtosis']:.4f}")
print(f"缺失值数量: {stats['missing']}")