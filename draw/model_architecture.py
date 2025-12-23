"""
模型架构示意图
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib
matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def draw_model_architecture():
    """绘制架构图"""
    
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(-4, 14)
    ax.axis('off')
    
    # 颜色
    c_input = '#3498DB'
    c_attn = '#E74C3C'
    c_dense = '#2ECC71'
    c_bn = '#9B59B6'
    c_drop = '#F39C12'
    c_output = '#1ABC9C'
    
    # 层定义 (名称, 颜色, y位置, 高度, 附加信息)
    layers = [
        ('Input Layer (63维特征)', c_input, 12.5, 0.8, ''),
        ('Attention-1 (自适应权重)', c_attn, 11.3, 0.7, ''),
        ('Dense-1 (128) + BN + Dropout(0.3)', c_dense, 10.0, 0.7, 'ReLU'),
        ('Dense-2 (256) + BN + Dropout(0.4)', c_dense, 8.7, 0.7, 'ReLU'),
        ('Attention-2 (隐层权重)', c_attn, 7.4, 0.7, ''),
        ('Dense-3 (512) + BN + Dropout(0.5)', c_dense, 6.1, 0.7, 'ReLU'),
        ('Dense-4 (256) + BN + Dropout(0.4)', c_dense, 4.8, 0.7, 'ReLU'),
        ('Dense-5 (128) + BN + Dropout(0.3)', c_dense, 3.5, 0.7, 'ReLU'),
        ('Output (1, 电导率)', c_output, 2.2, 0.8, 'Linear'),
    ]
    
    x_center = 6
    box_w = 7
    
    # 标题
    ax.text(6, 13.5, 'Attention-DNN 模型架构', ha='center', fontsize=20, fontweight='bold', color='#2C3E50')
    
    # 绘制各层
    for i, (name, color, y, h, act) in enumerate(layers):
        box = FancyBboxPatch((x_center - box_w/2, y - h/2), box_w, h,
                            boxstyle="round,rounding_size=0.15",
                            facecolor=color, edgecolor='white', lw=2, alpha=0.9)
        ax.add_patch(box)
        ax.text(x_center, y, name, ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        
        if act:
            ax.text(x_center + box_w/2 + 0.3, y, act, fontsize=10, color=color, fontweight='bold')
        
        # 箭头
        if i < len(layers) - 1:
            next_y = layers[i + 1][2]
            ax.annotate('', xy=(x_center, next_y + layers[i+1][3]/2 + 0.1), 
                       xytext=(x_center, y - h/2 - 0.1),
                       arrowprops=dict(arrowstyle='-|>', color='#34495E', lw=2))
    
    # 右侧性能框
    perf_box = FancyBboxPatch((9.5, 5), 2.3, 3.5,
                              boxstyle="round,rounding_size=0.15",
                              facecolor='#E8F6F3', edgecolor='#1ABC9C', lw=2)
    ax.add_patch(perf_box)
    ax.text(10.65, 8.2, '最优性能', ha='center', fontsize=12, fontweight='bold', color='#1ABC9C')
    ax.text(10.65, 7.5, 'seed=50', ha='center', fontsize=11, color='#27AE60')
    ax.text(10.65, 6.8, 'R2=0.9683', ha='center', fontsize=12, fontweight='bold', color='#E74C3C')
    ax.text(10.65, 6.1, 'MAE=0.3832', ha='center', fontsize=11, color='#2C3E50')
    ax.text(10.65, 5.4, 'RMSE=0.7040', ha='center', fontsize=11, color='#2C3E50')
    
    # 左侧图例
    legend_items = [('Input/Output', c_input), ('Attention', c_attn), 
                    ('Dense+BN+Drop', c_dense)]
    for i, (label, color) in enumerate(legend_items):
        rect = plt.Rectangle((0.3, 8 - i*0.7), 0.5, 0.45, facecolor=color, edgecolor='white', lw=2)
        ax.add_patch(rect)
        ax.text(1.0, 8.2 - i*0.7, label, ha='left', va='center', fontsize=10, color='#34495E')
    
    # 底部说明
    ax.text(6, 0.8, '模型特点: Attention机制 + BatchNorm + Dropout正则化', 
            ha='center', fontsize=11, color='#7F8C8D')
    ax.text(6, 0.2, '验证方式: 5-Fold交叉验证 + 100随机种子评估', 
            ha='center', fontsize=11, color='#7F8C8D')
    
    plt.tight_layout()
    
    save_path = r'e:\python\pythonProject3\draw\attention_dnn_architecture.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'已保存: {save_path}')


def draw_simple_flow():
    """绘制简洁流程图"""
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    ax.text(7, 4.5, 'Attention-DNN 架构流程', ha='center', fontsize=18, fontweight='bold', color='#2C3E50')
    
    blocks = [
        (1, 'Input\n63维', '#3498DB'),
        (3, 'Attn', '#E74C3C'),
        (5, 'Dense\n128-256', '#2ECC71'),
        (7, 'Attn', '#E74C3C'),
        (9, 'Dense\n512-256-128', '#2ECC71'),
        (11.5, 'Output\n(k)', '#1ABC9C'),
    ]
    
    for x, label, color in blocks:
        w = 1.5 if 'Dense' in label else 1.0
        box = FancyBboxPatch((x - w/2, 2 - 0.7), w, 1.4,
                            boxstyle="round,rounding_size=0.15",
                            facecolor=color, edgecolor='white', lw=2)
        ax.add_patch(box)
        ax.text(x, 2, label, ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    # 箭头
    arrow_xs = [(1.5, 2.5), (3.5, 4.25), (5.75, 6.5), (7.5, 8.25), (9.75, 11)]
    for x1, x2 in arrow_xs:
        ax.annotate('', xy=(x2, 2), xytext=(x1, 2),
                   arrowprops=dict(arrowstyle='-|>', color='#34495E', lw=2))
    
    ax.text(7, 0.6, 'Dense块 = Dense + BatchNorm + Dropout | 激活: ReLU | 输出: Linear',
            ha='center', fontsize=10, color='#95A5A6')
    
    plt.tight_layout()
    
    save_path = r'e:\python\pythonProject3\draw\attention_dnn_flow.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'已保存: {save_path}')


if __name__ == '__main__':
    print("生成Attention-DNN模型架构图...")
    draw_model_architecture()
    draw_simple_flow()
    print("完成!")
