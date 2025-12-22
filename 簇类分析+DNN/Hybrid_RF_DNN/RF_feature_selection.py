# RF_feature_selection.py
# 专门用于通过随机森林评估特征重要性，以提供不依赖于DNN的“第三方”客观评价。

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
import matplotlib

matplotlib.use('Agg')
warnings.filterwarnings("ignore")

# 字体配置
def configure_fonts():
    font_families = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'Arial']
    plt.rcParams['font.family'] = font_families
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 12})

configure_fonts()

# 1. 数据加载
print("加载数据...")
df = pd.read_csv(r'..\数据预处理\簇类加权特征数据集.csv')
y = pd.to_numeric(df.iloc[:, 1], errors='coerce')
X_raw = df.drop(df.columns[1], axis=1)

# 2. 特征工程 (保持与优化DNN一致)
print("特征工程与编码...")
# 多项式温度
if 'T' in X_raw.columns:
    poly = PolynomialFeatures(degree=3, include_bias=False)
    T_poly = poly.fit_transform(X_raw[['T']])
    X_raw['T_squared'] = T_poly[:, 1]
    X_raw['T_cubed'] = T_poly[:, 2]

# One-Hot 编码盐类型
X_encoded = pd.get_dummies(X_raw, columns=['salt'], prefix='salt')

# 移除 NaN
valid_idx = ~(X_encoded.isna().any(axis=1) | y.isna())
X_encoded = X_encoded[valid_idx]
y = y[valid_idx]
feature_names = X_encoded.columns.tolist()

# ============================================
# 3. 数据分布可视化分析
# ============================================
def plot_data_distributions(X, y, output_dir='.'):
    print("生成数据分布可视化图表...")
    
    # 3.1 目标变量分布 (电导率)
    plt.figure(figsize=(10, 6))
    sns.histplot(y, kde=True, color='steelblue', bins=30)
    plt.title('Conductivity k(ms*cm-1) Distribution', fontsize=16)
    plt.xlabel('k(ms*cm-1)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'target_distribution.png'), dpi=300)
    plt.close()
    
    # 3.2 温度分布
    if 'T' in X.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(X['T'], kde=True, color='coral', bins=20)
        plt.title('Temperature T Distribution', fontsize=16)
        plt.xlabel('Temperature (K)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'temperature_distribution.png'), dpi=300)
        plt.close()

    # 3.3 盐类型分布 (从 One-Hot 还原)
    salt_cols = [c for c in X.columns if c.startswith('salt_')]
    if salt_cols:
        salt_counts = X[salt_cols].sum().sort_values(ascending=False)
        # 简化名称：去掉 salt_ 前缀
        salt_counts.index = [i.replace('salt_', '') for i in salt_counts.index]
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=salt_counts.values, y=salt_counts.index, palette='magma')
        plt.title('Salt Type Distribution', fontsize=16)
        plt.xlabel('Sample Count', fontsize=14)
        plt.ylabel('Salt Type', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'salt_type_distribution.png'), dpi=300)
        plt.close()

    # 3.4 相关性热度图 (选取部分连续型特征)
    # 排除 One-Hot 和 多项式扩展特征
    continuous_cols = [c for c in X.columns if not (c.startswith('salt_') or 'squared' in c or 'cubed' in c)]
    # 只取前 20 个相关性较强的列
    top_corr_cols = X[continuous_cols].columns[:20] 
    
    plt.figure(figsize=(16, 12))
    corr_matrix = pd.concat([X[top_corr_cols], y], axis=1).corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Feature Correlation Heatmap (Top 20 Continuous Features + Target)', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()
    
    print("  ✓ 可视化图表已生成。")

plot_data_distributions(X_encoded, y)

# ============================================
# 4. 随机森林训练 (K-Fold)
# ============================================
print("执行随机森林 K-Fold 评估...")
N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

importances_list = []
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_encoded)):
    X_train, X_val = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # RF 不需要严格标准化，但为了后续对比，我们直接用
    rf = RandomForestRegressor(n_estimators=200, max_features='sqrt', n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_val)
    fold_r2 = r2_score(y_val, y_pred)
    fold_mae = mean_absolute_error(y_val, y_pred)
    
    importances_list.append(rf.feature_importances_)
    fold_metrics.append({'fold': fold+1, 'r2': fold_r2, 'mae': fold_mae})
    print(f"Fold {fold+1}: R² = {fold_r2:.4f}, MAE = {fold_mae:.4f}")

# 4. 汇总特征重要性
avg_importance = np.mean(importances_list, axis=0)

# 分组逻辑 (处理 One-Hot 和多项式)
def group_importances(importances, names):
    grouped = {}
    for imp, name in zip(importances, names):
        base_name = name
        if name.startswith('salt_'): base_name = 'salt'
        elif name in ['T', 'T_squared', 'T_cubed']: base_name = 'T (Polynomial)'
        elif name in ['tsne_x', 'tsne_y']: base_name = 'Cluster projection'
        elif name in ['c_val', 'c_units']: base_name = 'c'
        elif name in ['Linear Carbonyl', 'Cyclic Carbonyl', 'Linear Ether', 'Cyclic Ether']: base_name = 'Solvent Types'
        
        grouped[base_name] = grouped.get(base_name, 0) + imp
    return grouped

grouped_imp = group_importances(avg_importance, feature_names)
imp_df = pd.DataFrame({'Feature': list(grouped_imp.keys()), 'Importance': list(grouped_imp.values())})
imp_df = imp_df.sort_values('Importance', ascending=False)

# 保存
imp_df.to_csv('rf_feature_importance.csv', index=False)
print("特征重要性已保存至 rf_feature_importance.csv")

# 5. 可视化
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=imp_df.head(20), palette='viridis')
plt.title('Random Forest Feature Importance (Top 20)', fontsize=16)
plt.xlabel('Average Gini Importance')
plt.tight_layout()
plt.savefig('rf_importance_plot.png', dpi=300)
plt.close()

print("可视化已保存至 rf_importance_plot.png")
print("RF 评估完成。")
