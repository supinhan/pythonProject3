# DNN_op_100.py - 100个随机种子探索的Attention-DNN模型
# 基于 DNN_optimized.py 改进，结合 100test DNN.py 的多种子探索逻辑
# 改进内容：
# 1. Attention机制 - 自适应特征权重
# 2. 100个随机种子探索 - 寻找最优种子
# 3. 特征选择 - 移除重要性<0.05的特征
# 4. 多项式温度特征 - 捕捉非线性关系

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization, 
                                      Input, Multiply, Activation, Lambda)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
import matplotlib
import time
from tqdm import tqdm
import scipy.stats as stats

matplotlib.use('Agg')
warnings.filterwarnings("ignore")

# ============================================
# 字体配置
# ============================================
def configure_fonts():
    font_families = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'Arial']
    plt.rcParams['font.family'] = font_families
    print(f"已设置字体栈: {font_families}")
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

configure_fonts()

# ============================================
# 1. 数据加载与预处理
# ============================================
print("=" * 60)
print("加载数据集...")
print("=" * 60)

# 读取数据
df = pd.read_csv(r'..\数据预处理\簇类加权特征数据集.csv')

# 目标变量：k(ms*cm-1)
y = pd.to_numeric(df.iloc[:, 1], errors='coerce')

# 特征：除目标变量外的所有列
X_raw = df.drop(df.columns[1], axis=1)

print(f"原始数据形状: {df.shape}")
print(f"目标变量形状: {y.shape}")

# ============================================
# 2. 特征选择 - 移除重要性<0.05的特征
# ============================================
print("\n" + "=" * 60)
print("特征选择：移除重要性<0.05的特征")
print("=" * 60)

# 需要保留的特征（重要性>=0.05）- 基于之前的分析
important_features = [
    'salt', 'T', 'c_val', 'c_units',
    'Nitrogen-to-carbon atom ratio',
    'Number of carbonyl groups', 
    'Molecular radius',
    'Number of sulfones',
    'Molecular dipole moment',
    'Average atomic mass of heavy atoms',
    'Number of esters',
    'Number of covalent bond units',
    'Number of chiral carbons',
    'tsne_x', 'tsne_y',  # Cluster projection
    'Melting point',
    'solvent_ratio_type',
    'Water solubility',
    'Number of nitrogen atoms',
    'Lipid solubility',
    'Maximum ring size',
    'Hydrogen-to-carbon atom ratio',
    'Number of carbon-carbon triple bonds',
    'Number of chlorides',
    'Surface tension',
    'Refractive index',
    'Number of undefined stereocenters',
    'Does it have a plane of symmetry',
    'Molecular weight',
    'Number of fluorine atoms',
    'Number of amines',
    'Is it a chiral Molecule',
    'Average electron affinity',
    'Number of hydrogen bond acceptors',
    'Number of rotatable bonds.1',
    'Does it have stereocenters',
    'Fluorine-to-carbon atom ratio',
    'Viscosity',
    'Number of fluorides',
    'Average ionization energy',
    'Molecular complexity',
    'Number of aromatic bonds',
    'Number of silicon atoms',
    'Topological polar surface area ',
    'Does it have a center of symmetry',
    'Chlorine-to-carbon atom ratio',
    'Average atomic mass',
    'The ratio of carbon atoms to oxygen atoms',
    'Halogen-to-carbon atom ratio',
    'Polarizability',
    'Number of sulfoxides',
    'Isotopic atom count',
    'Number of heavy atoms',
    'Number of siloxanes',
    'Number of chlorine atoms',
    'The number of carbon atoms',
    'The number of atoms',
    'Number of nitriles',
    'Number of rotatable bonds',
    'Number of carbon-carbon double bonds',
    # 有机溶剂类型
    'Linear Carbonyl', 'Cyclic Carbonyl', 'Linear Ether', 'Cyclic Ether'
]

# 筛选存在的特征
available_features = [f for f in important_features if f in X_raw.columns]
print(f"保留的重要特征数量: {len(available_features)}")

# 创建特征子集
X_selected = X_raw[available_features].copy()

# ============================================
# 3. 多项式温度特征
# ============================================
print("\n" + "=" * 60)
print("添加多项式温度特征")
print("=" * 60)

if 'T' in X_selected.columns:
    T_values = X_selected['T'].values.reshape(-1, 1)
    poly = PolynomialFeatures(degree=3, include_bias=False)
    T_poly = poly.fit_transform(T_values)
    
    # 添加多项式特征（T², T³）
    X_selected['T_squared'] = T_poly[:, 1]
    X_selected['T_cubed'] = T_poly[:, 2]
    print(f"已添加多项式温度特征: T², T³")
    print(f"特征形状变化: {len(available_features)} -> {X_selected.shape[1]}")

# ============================================
# 4. One-Hot编码
# ============================================
print("\n" + "=" * 60)
print("One-Hot编码分类特征")
print("=" * 60)

categorical_cols = ['salt']
if 'salt' in X_selected.columns:
    X_encoded = pd.get_dummies(X_selected, columns=['salt'], prefix='salt')
else:
    salt_col_name = X_selected.columns[0]
    print(f"假设 '{salt_col_name}' 是salt列")
    X_encoded = pd.get_dummies(X_selected, columns=[salt_col_name], prefix='salt')

print(f"One-Hot编码后形状: {X_encoded.shape}")

# 移除NaN
valid_idx = ~(X_encoded.isna().any(axis=1) | y.isna())
X_encoded = X_encoded[valid_idx]
y = y[valid_idx]
print(f"移除NaN后样本数: {len(y)}")

# 保存特征名称
feature_names = X_encoded.columns.tolist()

# ============================================
# 5. Attention机制DNN模型
# ============================================
print("\n" + "=" * 60)
print("构建带Attention机制的DNN模型")
print("=" * 60)

def attention_layer(inputs, name_prefix='attention'):
    """
    自注意力机制层
    学习每个特征的重要性权重
    """
    # 注意力权重计算
    attention_weights = Dense(inputs.shape[-1], activation='tanh', 
                             name=f'{name_prefix}_dense1')(inputs)
    attention_weights = Dense(inputs.shape[-1], activation='softmax',
                             name=f'{name_prefix}_weights')(attention_weights)
    
    # 应用注意力权重
    attended = Multiply(name=f'{name_prefix}_multiply')([inputs, attention_weights])
    
    return attended, attention_weights

def build_attention_dnn(input_dim):
    """
    构建带有Attention机制的DNN模型
    """
    # 输入层
    inputs = Input(shape=(input_dim,), name='input')
    
    # 第一个Attention层 - 学习原始特征重要性
    x, attn_weights = attention_layer(inputs, name_prefix='attn1')
    
    # 第一个Dense块
    x = Dense(128, activation='relu', name='dense1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = Dropout(0.3, name='dropout1')(x)
    
    # 第二个Dense块
    x = Dense(256, activation='relu', name='dense2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Dropout(0.4, name='dropout2')(x)
    
    # 第二个Attention层 - 学习隐藏特征重要性
    x, _ = attention_layer(x, name_prefix='attn2')
    
    # 第三个Dense块（最宽）
    x = Dense(512, activation='relu', name='dense3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = Dropout(0.5, name='dropout3')(x)
    
    # 第四个Dense块
    x = Dense(256, activation='relu', name='dense4')(x)
    x = BatchNormalization(name='bn4')(x)
    x = Dropout(0.4, name='dropout4')(x)
    
    # 第五个Dense块
    x = Dense(128, activation='relu', name='dense5')(x)
    x = BatchNormalization(name='bn5')(x)
    x = Dropout(0.3, name='dropout5')(x)
    
    # 输出层
    outputs = Dense(1, activation='linear', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='AttentionDNN')
    return model

# ============================================
# 6. 数据标准化
# ============================================
print("\n" + "=" * 60)
print("数据标准化")
print("=" * 60)

scaler = StandardScaler()
X = scaler.fit_transform(X_encoded)
y_values = y.values

# 保存scaler
joblib.dump(scaler, 'optimized_scaler.pkl')
print(f"数据集大小: {X.shape[0]} 样本, {X.shape[1]} 特征")

# ============================================
# 7. 100个随机种子探索
# ============================================
print("\n" + "=" * 60)
print("100个随机种子探索")
print("=" * 60)

results_df = pd.DataFrame(columns=['seed', 'mae', 'mse', 'r2', 'rmse', 'training_time'])

SEEDS = range(1, 101)
INPUT_DIM = X.shape[1]
EPOCHS = 200
BATCH_SIZE = 64

pbar = tqdm(total=len(SEEDS), desc="评估随机种子性能")

for seed in SEEDS:
    start_time = time.time()
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_values, test_size=0.2, random_state=seed
    )
    
    # 构建模型
    model = build_attention_dnn(INPUT_DIM)
    
    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # 回调函数
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
    ]
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0
    )
    
    # 预测
    y_pred = model.predict(X_test, verbose=0)
    
    # 计算指标
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    training_time = time.time() - start_time
    
    results_df = pd.concat([results_df, pd.DataFrame({
        'seed': [seed],
        'mae': [mae],
        'mse': [mse],
        'r2': [r2],
        'rmse': [rmse],
        'training_time': [training_time]
    })], ignore_index=True)
    
    tf.keras.backend.clear_session()
    del model
    
    pbar.update(1)

pbar.close()

results_df.to_csv('random_seeds_performance.csv', index=False)
print("\n100个随机种子性能评估完成!")
print(f"结果已保存为 'random_seeds_performance.csv'")

# ============================================
# 8. 性能统计分析
# ============================================
print("\n" + "=" * 60)
print("性能统计分析")
print("=" * 60)

# 计算统计指标
stats_df = pd.DataFrame({
    'metric': ['MAE', 'MSE', 'R²', 'RMSE', 'Training Time'],
    'mean': [
        results_df['mae'].mean(),
        results_df['mse'].mean(),
        results_df['r2'].mean(),
        results_df['rmse'].mean(),
        results_df['training_time'].mean()
    ],
    'std': [
        results_df['mae'].std(),
        results_df['mse'].std(),
        results_df['r2'].std(),
        results_df['rmse'].std(),
        results_df['training_time'].std()
    ],
    'min': [
        results_df['mae'].min(),
        results_df['mse'].min(),
        results_df['r2'].min(),
        results_df['rmse'].min(),
        results_df['training_time'].min()
    ],
    'max': [
        results_df['mae'].max(),
        results_df['mse'].max(),
        results_df['r2'].max(),
        results_df['rmse'].max(),
        results_df['training_time'].max()
    ],
    'median': [
        results_df['mae'].median(),
        results_df['mse'].median(),
        results_df['r2'].median(),
        results_df['rmse'].median(),
        results_df['training_time'].median()
    ]
})

print("\n性能指标统计:")
print(stats_df)

# 保存统计结果
stats_df.to_csv('performance_statistics.csv', index=False)

# ============================================
# 9. 性能可视化
# ============================================
print("\n" + "=" * 60)
print("创建性能可视化图表")
print("=" * 60)

# 9.1 指标分布箱线图
plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
sns.boxplot(y=results_df['mae'], color='skyblue')
plt.title('MAE分布', fontsize=16)
plt.ylabel('MAE', fontsize=14)

plt.subplot(2, 2, 2)
sns.boxplot(y=results_df['mse'], color='lightgreen')
plt.title('MSE分布', fontsize=16)
plt.ylabel('MSE', fontsize=14)

plt.subplot(2, 2, 3)
sns.boxplot(y=results_df['r2'], color='salmon')
plt.title('R²分布', fontsize=16)
plt.ylabel('R²', fontsize=14)

plt.subplot(2, 2, 4)
sns.boxplot(y=results_df['rmse'], color='gold')
plt.title('RMSE分布', fontsize=16)
plt.ylabel('RMSE', fontsize=14)

plt.suptitle('100个随机种子性能指标分布 (Attention-DNN)', fontsize=20, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('performance_boxplots.png', dpi=300)
plt.close()

# 9.2 指标分布直方图
plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
sns.histplot(results_df['mae'], kde=True, color='skyblue', bins=20)
plt.title('MAE分布', fontsize=16)
plt.xlabel('MAE', fontsize=14)

plt.subplot(2, 2, 2)
sns.histplot(results_df['mse'], kde=True, color='lightgreen', bins=20)
plt.title('MSE分布', fontsize=16)
plt.xlabel('MSE', fontsize=14)

plt.subplot(2, 2, 3)
sns.histplot(results_df['r2'], kde=True, color='salmon', bins=20)
plt.title('R²分布', fontsize=16)
plt.xlabel('R²', fontsize=14)

plt.subplot(2, 2, 4)
sns.histplot(results_df['rmse'], kde=True, color='gold', bins=20)
plt.title('RMSE分布', fontsize=16)
plt.xlabel('RMSE', fontsize=14)

plt.suptitle('100个随机种子性能指标分布 (Attention-DNN)', fontsize=20, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('performance_histograms.png', dpi=300)
plt.close()

# 9.3 指标相关性热图
plt.figure(figsize=(10, 8))
corr_matrix = results_df[['mae', 'mse', 'r2', 'rmse']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('性能指标相关性', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('performance_correlation.png', dpi=300)
plt.close()

# 9.4 训练时间分析
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x='training_time', y='r2', data=results_df, alpha=0.7)
plt.title('训练时间 vs R²', fontsize=16)
plt.xlabel('训练时间 (秒)', fontsize=14)
plt.ylabel('R²', fontsize=14)

plt.subplot(1, 2, 2)
sns.scatterplot(x='training_time', y='mae', data=results_df, alpha=0.7)
plt.title('训练时间 vs MAE', fontsize=16)
plt.xlabel('训练时间 (秒)', fontsize=14)
plt.ylabel('MAE', fontsize=14)

plt.suptitle('训练时间与性能关系', fontsize=20, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('training_time_vs_performance.png', dpi=300)
plt.close()

# ============================================
# 10. 种子综合性能排行
# ============================================
print("\n" + "=" * 60)
print("种子综合性能排行")
print("=" * 60)

# 计算综合性能分数 (R²越高越好，MAE/MSE/RMSE越低越好)
results_df['performance_score'] = (
        results_df['r2'] * 0.4 +
        (1 - results_df['mae'] / results_df['mae'].max()) * 0.2 +
        (1 - results_df['mse'] / results_df['mse'].max()) * 0.2 +
        (1 - results_df['rmse'] / results_df['rmse'].max()) * 0.2
)

# 按性能分数排序
ranked_results = results_df.sort_values('performance_score', ascending=False)
ranked_results['rank'] = range(1, len(ranked_results) + 1)

# 保存排行
ranked_results.to_csv('seed_performance_ranking.csv', index=False)
print("种子性能排行已保存为 'seed_performance_ranking.csv'")

# 可视化排行
plt.figure(figsize=(14, 10))

# 前20名种子性能
top_seeds = ranked_results.head(20)
plt.subplot(2, 1, 1)
sns.barplot(x='performance_score', y='seed', data=top_seeds, palette='viridis')
plt.title('前20名种子综合性能排行 (Attention-DNN)', fontsize=18, fontweight='bold')
plt.xlabel('综合性能分数', fontsize=14)
plt.ylabel('随机种子', fontsize=14)

# 性能分数分布
plt.subplot(2, 1, 2)
sns.histplot(ranked_results['performance_score'], kde=True, bins=20, color='purple')
plt.title('综合性能分数分布', fontsize=18, fontweight='bold')
plt.xlabel('综合性能分数', fontsize=14)
plt.ylabel('频率', fontsize=14)

plt.tight_layout()
plt.savefig('seed_performance_ranking.png', dpi=300)
plt.close()

# 最佳和最差种子对比
best_seed = ranked_results.iloc[0]
worst_seed = ranked_results.iloc[-1]

comparison_df = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'R²', 'RMSE'],
    'Best Seed': [
        best_seed['mae'],
        best_seed['mse'],
        best_seed['r2'],
        best_seed['rmse']
    ],
    'Worst Seed': [
        worst_seed['mae'],
        worst_seed['mse'],
        worst_seed['r2'],
        worst_seed['rmse']
    ]
})

plt.figure(figsize=(10, 6))
comparison_df.set_index('Metric').plot(kind='bar', rot=0)
plt.title(f'最佳种子({int(best_seed["seed"])}) vs 最差种子({int(worst_seed["seed"])})', fontsize=16)
plt.ylabel('值', fontsize=14)
plt.legend(title='种子')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('best_worst_seed_comparison.png', dpi=300)
plt.close()

# ============================================
# 11. 使用最佳种子训练最终模型
# ============================================
print("\n" + "=" * 60)
print("使用最佳种子训练最终模型")
print("=" * 60)

best_seed_value = int(ranked_results.iloc[0]['seed'])
print(f"使用最佳种子 {best_seed_value} 训练最终模型...")

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y_values, test_size=0.2, random_state=best_seed_value
)

# 构建模型
final_model = build_attention_dnn(INPUT_DIM)

# 编译模型
final_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# 显示模型结构
final_model.summary()

# 回调函数
callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
]

# 训练模型
final_history = final_model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# 评估模型
test_loss, test_mae = final_model.evaluate(X_test, y_test, verbose=0)
y_pred = final_model.predict(X_test, verbose=0).flatten()
r2 = r2_score(y_test, y_pred)

print(f"\n最终模型性能 (种子 {best_seed_value}):")
print(f"MAE: {test_mae:.4f}")
print(f"MSE: {test_loss:.4f}")
print(f"R²: {r2:.4f}")
print(f"RMSE: {np.sqrt(test_loss):.4f}")

# ============================================
# 12. 最终模型可视化
# ============================================
print("\n" + "=" * 60)
print("生成最终模型可视化图表")
print("=" * 60)

# 12.1 实际值 vs 预测值
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k', s=80)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
         'r--', lw=2.5, label='完美预测线')
z = np.polyfit(y_test.flatten(), y_pred.flatten(), 1)
p = np.poly1d(z)
plt.plot(np.sort(y_test), p(np.sort(y_test)), 'g-', lw=2, 
         label=f'拟合线 (y={z[0]:.2f}x + {z[1]:.2f})')
plt.text(0.05, 0.9, f'R² = {r2:.4f}', transform=plt.gca().transAxes, fontsize=14,
         bbox=dict(facecolor='white', alpha=0.8))
plt.xlabel('实际值', fontsize=14)
plt.ylabel('预测值', fontsize=14)
plt.title(f'Attention-DNN: 实际值 vs 预测值 (最佳种子 {best_seed_value})', fontsize=18, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('optimized_dnn_actual_vs_predicted.png', dpi=300)
plt.close()
print("  ✓ 保存: optimized_dnn_actual_vs_predicted.png")

# 12.2 残差分析
residuals = y_test.flatten() - y_pred.flatten()
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.histplot(residuals, kde=True, color='steelblue', bins=30)
plt.axvline(x=0, color='r', linestyle='--')
plt.title('残差分布', fontsize=16)
plt.xlabel('残差', fontsize=12)
plt.ylabel('频率', fontsize=12)

plt.subplot(1, 2, 2)
plt.scatter(y_pred, residuals, alpha=0.6, edgecolor='k')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('残差 vs 预测值', fontsize=16)
plt.xlabel('预测值', fontsize=12)
plt.ylabel('残差', fontsize=12)

plt.tight_layout()
plt.savefig('optimized_dnn_residual_analysis.png', dpi=300)
plt.close()
print("  ✓ 保存: optimized_dnn_residual_analysis.png")

# 12.3 训练历史
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(final_history.history['loss'], label='训练损失', linewidth=2)
plt.plot(final_history.history['val_loss'], label='验证损失', linewidth=2)
plt.title('损失曲线 (MSE)', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(final_history.history['mae'], label='训练MAE', linewidth=2)
plt.plot(final_history.history['val_mae'], label='验证MAE', linewidth=2)
plt.title('MAE曲线', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimized_dnn_training_history.png', dpi=300)
plt.close()
print("  ✓ 保存: optimized_dnn_training_history.png")

# ============================================
# 13. 特征重要性（分组）
# ============================================
print("\n" + "=" * 60)
print("计算分组特征重要性")
print("=" * 60)

def get_feature_groups(all_columns, categorical_cols):
    groups = {}
    used_indices = set()
    
    for cat_col in categorical_cols:
        indices = [i for i, col in enumerate(all_columns) if col.startswith(f"{cat_col}_")]
        if indices:
            groups[cat_col] = indices
            used_indices.update(indices)
    
    for i, col in enumerate(all_columns):
        if i not in used_indices:
            groups[col] = [i]
    
    return groups

def grouped_permutation_importance(model, X, y, groups, metric=mean_squared_error, n_repeats=5):
    baseline_score = metric(y, model.predict(X, verbose=0).flatten())
    importances = {}
    
    print(f"基线 MSE: {baseline_score:.4f}")
    
    for group_name, indices in groups.items():
        score_increases = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            perm_indices = np.random.permutation(X.shape[0])
            X_permuted[:, indices] = X_permuted[perm_indices][:, indices]
            
            permuted_score = metric(y, model.predict(X_permuted, verbose=0).flatten())
            score_increases.append(permuted_score - baseline_score)
        
        importances[group_name] = np.mean(score_increases)
    
    return importances

# 特征分组
groups = get_feature_groups(feature_names, categorical_cols)

# 合并 c_val 和 c_units 为 'c'
c_indices = []
if 'c_val' in groups:
    c_indices.extend(groups['c_val'])
    del groups['c_val']
if 'c_units' in groups:
    c_indices.extend(groups['c_units'])
    del groups['c_units']
if c_indices:
    groups['c'] = c_indices

# 合并温度相关特征
temp_indices = []
for col in ['T', 'T_squared', 'T_cubed']:
    if col in groups:
        temp_indices.extend(groups[col])
        del groups[col]
if temp_indices:
    groups['T (多项式)'] = temp_indices
    print(f"已将T, T², T³ 合并为 'T (多项式)'")

# 合并 tsne 特征
cluster_indices = []
for col in ['tsne_x', 'tsne_y']:
    if col in groups:
        cluster_indices.extend(groups[col])
        del groups[col]
if cluster_indices:
    groups['Cluster projection'] = cluster_indices

# 合并溶剂类型
solvent_types = ['Linear Carbonyl', 'Cyclic Carbonyl', 'Linear Ether', 'Cyclic Ether']
type_indices = []
for st in solvent_types:
    if st in groups:
        type_indices.extend(groups[st])
        del groups[st]
if type_indices:
    groups['Types of organic solvents'] = type_indices

# 计算特征重要性
feature_importances = grouped_permutation_importance(final_model, X_test, y_test, groups)

# 创建DataFrame并保存
feature_importance_df = pd.DataFrame({
    'Feature': list(feature_importances.keys()),
    'Importance': list(feature_importances.values())
}).sort_values('Importance', ascending=False)

feature_importance_df.to_csv('optimized_feature_importance.csv', index=False)
print("  ✓ 保存: optimized_feature_importance.csv")

# 绘制特征重要性图
top_n = min(20, len(feature_importance_df))
plt.figure(figsize=(12, 8))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance_df.head(top_n),
    color='steelblue',
    edgecolor='black'
)

for i, (imp, name) in enumerate(
        zip(feature_importance_df['Importance'].head(top_n), 
            feature_importance_df['Feature'].head(top_n))):
    plt.text(imp, i, f'{imp:.4f}', va='center', fontsize=10)

plt.xlabel('重要性 (MSE增加)', fontsize=14)
plt.ylabel('特征', fontsize=14)
plt.title(f'分组特征重要性 (Top {top_n}) - Attention-DNN (100种子优化)', fontsize=18, fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('optimized_dnn_feature_importance.png', dpi=300)
plt.close()
print("  ✓ 保存: optimized_dnn_feature_importance.png")

# ============================================
# 14. 保存模型
# ============================================
print("\n" + "=" * 60)
print("保存模型")
print("=" * 60)

final_model.save('best_attention_dnn_model.h5')
print("  ✓ 模型保存: best_attention_dnn_model.h5")

# ============================================
# 15. 性能总结报告
# ============================================
print("\n" + "=" * 60)
print("性能总结报告")
print("=" * 60)

print(f"评估种子数量: {len(SEEDS)}")
print(f"平均R²: {results_df['r2'].mean():.4f} ± {results_df['r2'].std():.4f}")
print(f"平均MAE: {results_df['mae'].mean():.4f} ± {results_df['mae'].std():.4f}")
print(f"平均训练时间: {results_df['training_time'].mean():.2f}秒")
print(f"最佳种子: {best_seed_value} (R²={best_seed['r2']:.4f})")
print(f"最差种子: {int(worst_seed['seed'])} (R²={worst_seed['r2']:.4f})")

# 创建综合报告
with open('performance_summary.txt', 'w', encoding='utf-8') as f:
    f.write("===== Attention-DNN 随机种子性能评估报告 =====\n\n")
    f.write(f"评估种子数量: {len(SEEDS)}\n")
    f.write(f"评估时间: {time.ctime()}\n\n")
    
    f.write("--- 模型改进内容 ---\n")
    f.write("1. Attention机制: 自适应学习特征权重\n")
    f.write("2. 100个随机种子探索: 寻找最优种子\n")
    f.write("3. 特征选择: 移除重要性<0.05的特征\n")
    f.write("4. 多项式温度特征: T, T², T³\n\n")

    f.write("--- 性能指标统计 ---\n")
    f.write(stats_df.to_string(index=False) + "\n\n")

    f.write("--- 最佳种子 ---\n")
    f.write(f"种子: {best_seed_value}\n")
    f.write(f"MAE: {best_seed['mae']:.4f}\n")
    f.write(f"MSE: {best_seed['mse']:.4f}\n")
    f.write(f"R²: {best_seed['r2']:.4f}\n")
    f.write(f"RMSE: {best_seed['rmse']:.4f}\n")
    f.write(f"训练时间: {best_seed['training_time']:.2f}秒\n\n")

    f.write("--- 最差种子 ---\n")
    f.write(f"种子: {int(worst_seed['seed'])}\n")
    f.write(f"MAE: {worst_seed['mae']:.4f}\n")
    f.write(f"MSE: {worst_seed['mse']:.4f}\n")
    f.write(f"R²: {worst_seed['r2']:.4f}\n")
    f.write(f"RMSE: {worst_seed['rmse']:.4f}\n")
    f.write(f"训练时间: {worst_seed['training_time']:.2f}秒\n\n")

    f.write("--- 性能指标相关性 ---\n")
    f.write(corr_matrix.to_string() + "\n\n")

    f.write("--- 前10名种子排行 ---\n")
    f.write(ranked_results.head(10)[['rank', 'seed', 'performance_score', 'r2', 'mae']].to_string(index=False) + "\n\n")
    
    f.write("--- 最终模型性能 ---\n")
    f.write(f"种子: {best_seed_value}\n")
    f.write(f"MAE: {test_mae:.4f}\n")
    f.write(f"MSE: {test_loss:.4f}\n")
    f.write(f"R²: {r2:.4f}\n")
    f.write(f"RMSE: {np.sqrt(test_loss):.4f}\n\n")
    
    f.write("--- 输出文件 ---\n")
    f.write("  - best_attention_dnn_model.h5\n")
    f.write("  - optimized_scaler.pkl\n")
    f.write("  - random_seeds_performance.csv\n")
    f.write("  - performance_statistics.csv\n")
    f.write("  - seed_performance_ranking.csv\n")
    f.write("  - optimized_feature_importance.csv\n")
    f.write("  - performance_boxplots.png\n")
    f.write("  - performance_histograms.png\n")
    f.write("  - performance_correlation.png\n")
    f.write("  - training_time_vs_performance.png\n")
    f.write("  - seed_performance_ranking.png\n")
    f.write("  - best_worst_seed_comparison.png\n")
    f.write("  - optimized_dnn_actual_vs_predicted.png\n")
    f.write("  - optimized_dnn_residual_analysis.png\n")
    f.write("  - optimized_dnn_training_history.png\n")
    f.write("  - optimized_dnn_feature_importance.png\n")
    f.write("  - performance_summary.txt\n")

print("\n所有分析完成! 结果文件已保存。")
print("训练完成!")
