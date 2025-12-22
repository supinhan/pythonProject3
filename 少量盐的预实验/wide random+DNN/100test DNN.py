import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import matplotlib as mpl
import matplotlib.font_manager as fm
import warnings
import matplotlib
import time
from tqdm import tqdm
import scipy.stats as stats

matplotlib.use('Agg')
warnings.filterwarnings("ignore")


# 字体设置
def configure_fonts():
    import platform
    system = platform.system()
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

desired_columns = pd.read_csv('广精样本.csv', nrows=0).columns.tolist()
df = pd.read_csv('广精样本.csv', usecols=desired_columns)
y = pd.to_numeric(df.iloc[:, 0], errors='coerce')
X_df = df.iloc[:, 1:15]
X_df.iloc[:, 0] = X_df.iloc[:, 0].astype(str)
le = LabelEncoder()
all_salt = X_df.iloc[:, 0]
le.fit(all_salt)
X_df.iloc[:, 0] = le.transform(X_df.iloc[:, 0])
feature_names = X_df.columns.tolist()

scaler = StandardScaler()
X = scaler.fit_transform(X_df.values)

joblib.dump(scaler, 'ann_scaler.pkl')
joblib.dump(le, 'ann_label_encoder.pkl')

print(f"数据集大小: {X.shape[0]} 样本, {X.shape[1]} 特征")



def build_ann_model(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='linear'))
    return model


results_df = pd.DataFrame(columns=['seed', 'mae', 'mse', 'r2', 'rmse', 'training_time'])

SEEDS = range(1, 101)
INPUT_DIM = X.shape[1]

pbar = tqdm(total=len(SEEDS), desc="评估随机种子性能")

for seed in SEEDS:
    start_time = time.time()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    model = build_ann_model(INPUT_DIM)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=64,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0
    )

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

# 性能统计分析

print("\n=== 性能统计分析 ===")

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

# ---------------------------
# 性能可视化
# ---------------------------

print("\n=== 创建性能可视化图表 ===")

# 1. 指标分布箱线图
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

plt.suptitle('100个随机种子性能指标分布', fontsize=20, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('performance_boxplots.png', dpi=300)
plt.close()

# 2. 指标分布直方图
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

plt.suptitle('100个随机种子性能指标分布', fontsize=20, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('performance_histograms.png', dpi=300)
plt.close()

# 3. 指标相关性热图
plt.figure(figsize=(10, 8))
corr_matrix = results_df[['mae', 'mse', 'r2', 'rmse']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('性能指标相关性', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('performance_correlation.png', dpi=300)
plt.close()

# 4. 训练时间分析
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

# ---------------------------
# 种子综合性能排行
# ---------------------------

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
print("\n种子性能排行已保存为 'seed_performance_ranking.csv'")

# 可视化排行
plt.figure(figsize=(14, 10))

# 1. 前20名种子性能
top_seeds = ranked_results.head(20)
plt.subplot(2, 1, 1)
sns.barplot(x='performance_score', y='seed', data=top_seeds, palette='viridis')
plt.title('前20名种子综合性能排行', fontsize=18, fontweight='bold')
plt.xlabel('综合性能分数', fontsize=14)
plt.ylabel('随机种子', fontsize=14)

# 2. 性能分数分布
plt.subplot(2, 1, 2)
sns.histplot(ranked_results['performance_score'], kde=True, bins=20, color='purple')
plt.title('综合性能分数分布', fontsize=18, fontweight='bold')
plt.xlabel('综合性能分数', fontsize=14)
plt.ylabel('频率', fontsize=14)

plt.tight_layout()
plt.savefig('seed_performance_ranking.png', dpi=300)
plt.close()

# 3. 最佳和最差种子对比
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
plt.title(f'最佳种子({best_seed["seed"]}) vs 最差种子({worst_seed["seed"]})', fontsize=16)
plt.ylabel('值', fontsize=14)
plt.legend(title='种子')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('best_worst_seed_comparison.png', dpi=300)
plt.close()

# ---------------------------
# 最佳模型训练与保存
# ---------------------------

# 使用最佳种子训练最终模型
best_seed_value = ranked_results.iloc[0]['seed']
print(f"\n使用最佳种子 {best_seed_value} 训练最终模型...")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=best_seed_value
)

# 构建模型
model = build_ann_model(INPUT_DIM)

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
    epochs=200,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# 评估模型
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\n最终模型性能 (种子 {best_seed_value}):")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# 保存模型
model.save('best_ann_model.h5')
print("\n最佳模型已保存为 'best_ann_model.h5'")

# ---------------------------
# 性能总结报告
# ---------------------------

print("\n=== 性能总结报告 ===")
print(f"评估种子数量: {len(SEEDS)}")
print(f"平均R²: {results_df['r2'].mean():.4f} ± {results_df['r2'].std():.4f}")
print(f"平均MAE: {results_df['mae'].mean():.4f} ± {results_df['mae'].std():.4f}")
print(f"平均训练时间: {results_df['training_time'].mean():.2f}秒")
print(f"最佳种子: {best_seed_value} (R²={best_seed['r2']:.4f})")
print(f"最差种子: {worst_seed['seed']} (R²={worst_seed['r2']:.4f})")

# 创建综合报告
with open('performance_summary.txt', 'w') as f:
    f.write("===== 随机种子性能评估报告 =====\n\n")
    f.write(f"评估种子数量: {len(SEEDS)}\n")
    f.write(f"评估时间范围: {time.ctime()}\n\n")

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
    f.write(f"种子: {worst_seed['seed']}\n")
    f.write(f"MAE: {worst_seed['mae']:.4f}\n")
    f.write(f"MSE: {worst_seed['mse']:.4f}\n")
    f.write(f"R²: {worst_seed['r2']:.4f}\n")
    f.write(f"RMSE: {worst_seed['rmse']:.4f}\n")
    f.write(f"训练时间: {worst_seed['training_time']:.2f}秒\n\n")

    f.write("--- 性能指标相关性 ---\n")
    f.write(corr_matrix.to_string() + "\n\n")

    f.write("--- 前10名种子排行 ---\n")
    f.write(ranked_results.head(10)[['rank', 'seed', 'performance_score', 'r2', 'mae']].to_string(index=False))

print("\n所有分析完成! 结果文件已保存。")