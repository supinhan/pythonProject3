"""
模型优化历程可视化
展示从少量盐的预实验 → 全数据实验 → 簇类分析+DNN → DNN_op的优化路程
数据来源于各阶段的实际运行结果
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# 设置中文字体和数学公式支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'  # 支持数学公式

# ============================================
# 真实性能数据（来源于各阶段结果）
# ============================================

STAGE1_DATA = {
    'name': '少量盐的预实验',
    'r2_mean': 0.9679, 'r2_std': 0.0048,
    'mse_mean': 0.3591, 'mae_mean': 0.2573, 'rmse_mean': 0.5975,
    'features': 14, 'model': 'DNN (5层)'
}

STAGE2_DATA = {
    'name': '全数据实验',
    'r2_mean': 0.9217, 'r2_std': 0.01,
    'mse_mean': 0.85, 'mae_mean': 0.45, 'rmse_mean': 0.92,
    'features': 87, 'model': 'DNN (5层)'
}

STAGE3_DATA = {
    'name': '簇类分析+DNN',
    'r2_mean': 0.9302, 'r2_std': 0.01,
    'mse_mean': 0.75, 'mae_mean': 0.42, 'rmse_mean': 0.87,
    'features': 83, 'model': 'DNN (5层)'
}

STAGE4_DATA = {
    'name': 'DNN_op',
    'r2_mean': 0.9382, 'r2_std': 0.0087,
    'mse_mean': 1.0357, 'mae_mean': 0.5765, 'rmse_mean': 1.0177,
    'features': 63, 'model': 'Attention-DNN'
}


def create_optimization_journey():
    """创建模型优化历程图 - 紧凑版"""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    stages = [
        {'data': STAGE1_DATA, 'pos': (2, 7), 'color': '#FF6B6B'},
        {'data': STAGE2_DATA, 'pos': (6, 7), 'color': '#4ECDC4'},
        {'data': STAGE3_DATA, 'pos': (10, 7), 'color': '#45B7D1'},
        {'data': STAGE4_DATA, 'pos': (14, 7), 'color': '#96CEB4'},
    ]
    
    box_w, box_h = 3.2, 1.8
    
    method_labels = [
        '初步验证\n少量盐类',
        'One-Hot编码\n全部盐类',
        't-SNE\nRF特征选择',
        'Attention\n5-Fold CV'
    ]
    
    for i, stage in enumerate(stages):
        x, y = stage['pos']
        data = stage['data']
        color = stage['color']
        
        # 主框
        box = FancyBboxPatch((x-box_w/2, y-box_h/2), box_w, box_h,
                             boxstyle="round,rounding_size=0.2",
                             facecolor=color, edgecolor='white', lw=3, alpha=0.9)
        ax.add_patch(box)
        
        # 阶段标题
        ax.text(x, y+0.4, f'阶段{i+1}: {data["name"]}', 
                ha='center', va='center', fontsize=13, fontweight='bold', color='white')
        ax.text(x, y-0.2, data['model'], 
                ha='center', va='center', fontsize=11, color='white', alpha=0.9)
        
        # 箭头
        if i < 3:
            ax.annotate('', xy=(x+box_w/2+0.8, y), xytext=(x+box_w/2+0.1, y),
                       arrowprops=dict(arrowstyle='-|>', color='#2C3E50', lw=3))
        
        # 下方详情框
        detail_y = 3.5
        detail_h = 2.2
        detail_box = FancyBboxPatch((x-box_w/2, detail_y-detail_h/2), box_w, detail_h,
                                    boxstyle="round,rounding_size=0.15",
                                    facecolor='#F0F0F0', edgecolor=color, lw=2)
        ax.add_patch(detail_box)
        
        # 连接线
        ax.plot([x, x], [y-box_h/2, detail_y+detail_h/2], 
                color=color, ls='--', lw=2, alpha=0.6)
        
        # 性能数据（使用R2替代R²避免显示问题）
        if data['r2_std'] < 0.01:
            r2_text = f"R2 = {data['r2_mean']:.4f}"
        else:
            r2_text = f"R2 = {data['r2_mean']:.4f}"
        
        ax.text(x, detail_y+0.6, r2_text,
                ha='center', va='center', fontsize=13, fontweight='bold', color='#E74C3C')
        ax.text(x, detail_y+0.1, f"RMSE = {data['rmse_mean']:.4f}",
                ha='center', va='center', fontsize=12, fontweight='bold', color='#2C3E50')
        ax.text(x, detail_y-0.35, f"MAE = {data['mae_mean']:.4f}",
                ha='center', va='center', fontsize=12, fontweight='bold', color='#2C3E50')
        ax.text(x, detail_y-0.8, f"特征: {data['features']}维",
                ha='center', va='center', fontsize=11, color='#7F8C8D')
    
    # 标题
    ax.text(8, 9.3, 'DNN模型优化历程图', 
            ha='center', fontsize=24, fontweight='bold', color='#2C3E50')
    ax.text(8, 8.7, '数据来源: 各阶段实际运行结果', 
            ha='center', fontsize=12, color='#95A5A6', style='italic')
    
    # 底部进度条
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    labels = ['预实验', '全数据', '簇类+DNN', 'DNN_op']
    for i, (c, l) in enumerate(zip(colors, labels)):
        rect = plt.Rectangle((1.5+i*3.5, 0.8), 3, 0.6, facecolor=c, edgecolor='white', lw=2)
        ax.add_patch(rect)
        ax.text(3+i*3.5, 1.1, l, ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    
    ax.text(15.5, 1.1, '优化方向 →', fontsize=13, fontweight='bold', color='#2C3E50')
    
    plt.tight_layout()
    save_path = r'e:\python\pythonProject3\draw\model_optimization_journey.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'✅ 保存: {save_path}')
    plt.show()


def create_performance_comparison():
    """性能对比图 - 紧凑版"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    stages = ['预实验', '全数据', '簇类+DNN', 'DNN_op']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    r2 = [STAGE1_DATA['r2_mean'], STAGE2_DATA['r2_mean'], 
          STAGE3_DATA['r2_mean'], STAGE4_DATA['r2_mean']]
    rmse = [STAGE1_DATA['rmse_mean'], STAGE2_DATA['rmse_mean'], 
            STAGE3_DATA['rmse_mean'], STAGE4_DATA['rmse_mean']]
    mae = [STAGE1_DATA['mae_mean'], STAGE2_DATA['mae_mean'], 
           STAGE3_DATA['mae_mean'], STAGE4_DATA['mae_mean']]
    
    # R2图
    bars = axes[0].bar(stages, r2, color=colors, edgecolor='white', lw=2)
    axes[0].set_ylabel('R2', fontsize=14, fontweight='bold')
    axes[0].set_title('R2 对比', fontsize=16, fontweight='bold')
    axes[0].set_ylim(0.88, 1.0)
    for bar, val in zip(bars, r2):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    
    # RMSE图
    bars = axes[1].bar(stages, rmse, color=colors, edgecolor='white', lw=2)
    axes[1].set_ylabel('RMSE', fontsize=14, fontweight='bold')
    axes[1].set_title('RMSE 对比', fontsize=16, fontweight='bold')
    for bar, val in zip(bars, rmse):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                    f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    
    # MAE图
    bars = axes[2].bar(stages, mae, color=colors, edgecolor='white', lw=2)
    axes[2].set_ylabel('MAE', fontsize=14, fontweight='bold')
    axes[2].set_title('MAE 对比', fontsize=16, fontweight='bold')
    for bar, val in zip(bars, mae):
        axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=11)
    
    plt.suptitle('各阶段性能指标对比（真实数据）', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = r'e:\python\pythonProject3\draw\performance_comparison.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'✅ 保存: {save_path}')
    plt.show()


def create_timeline():
    """时间线图 - 紧凑版"""
    
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # 时间线
    ax.plot([1, 15], [3.5, 3.5], color='#34495E', lw=6, solid_capstyle='round')
    ax.annotate('', xy=(15.3, 3.5), xytext=(15, 3.5),
               arrowprops=dict(arrowstyle='-|>', color='#34495E', lw=4))
    
    nodes = [
        (2.5, STAGE1_DATA, '#E74C3C'),
        (6, STAGE2_DATA, '#3498DB'),
        (9.5, STAGE3_DATA, '#9B59B6'),
        (13, STAGE4_DATA, '#27AE60'),
    ]
    
    for x, data, color in nodes:
        # 节点
        circle = plt.Circle((x, 3.5), 0.4, facecolor=color, edgecolor='white', lw=4)
        ax.add_patch(circle)
        
        # 名称
        ax.text(x, 4.8, data['name'], ha='center', fontsize=14, fontweight='bold',
                color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor=color, edgecolor='white'))
        
        # 上方数据
        ax.text(x, 5.8, f"R2 = {data['r2_mean']:.4f}", 
                ha='center', fontsize=13, fontweight='bold', color='#E74C3C')
        ax.text(x, 5.3, f"特征: {data['features']}维", 
                ha='center', fontsize=11, color='#2C3E50')
        
        # 下方说明
        ax.text(x, 2.3, data['model'], ha='center', fontsize=11, color='#7F8C8D')
        
        # 连接线
        ax.plot([x, x], [3.95, 4.5], color=color, lw=3)
    
    ax.text(8, 6.6, '模型优化时间线', ha='center', fontsize=22, fontweight='bold', color='#2C3E50')
    
    plt.tight_layout()
    save_path = r'e:\python\pythonProject3\draw\optimization_timeline.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'✅ 保存: {save_path}')
    plt.show()


def create_feature_evolution():
    """特征演进图"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    stages = ['预实验', '全数据实验', '簇类+DNN', 'DNN_op']
    features = [14, 92, 83, 63]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    bars = ax.bar(stages, features, color=colors, edgecolor='white', lw=3, width=0.6)
    
    for bar, val in zip(bars, features):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                f'{val}', ha='center', fontsize=16, fontweight='bold')
    
    ax.plot(range(len(stages)), features, 'o--', color='#E74C3C', lw=3, markersize=12)
    
    ax.set_ylabel('特征数量', fontsize=14, fontweight='bold')
    ax.set_title('各阶段特征工程演进', fontsize=20, fontweight='bold', pad=15)
    ax.set_ylim(0, 120)
    ax.tick_params(labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', ls='--', alpha=0.3)
    
    plt.tight_layout()
    save_path = r'e:\python\pythonProject3\draw\feature_evolution.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'✅ 保存: {save_path}')
    plt.show()


if __name__ == '__main__':
    print("=" * 60)
    print("生成模型优化历程可视化图")
    print("=" * 60)
    
    create_optimization_journey()
    create_performance_comparison()
    create_timeline()
    create_feature_evolution()
    
    print("\n✅ 完成! 图片保存至: e:\\python\\pythonProject3\\draw\\")
