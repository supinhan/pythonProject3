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

matplotlib.use('Agg')
warnings.filterwarnings("ignore")


def configure_fonts():
    import platform
    system = platform.system()
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
X_train_df, X_test_df, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, random_state=100
)

print(f"数据集大小: {X_df.shape[0]} 样本, {X_df.shape[1]} 特征")
print(f"训练集: {X_train_df.shape[0]} 样本")
print(f"测试集: {X_test_df.shape[0]} 样本")

le = LabelEncoder()
all_salt = pd.concat([X_train_df.iloc[:, 0], X_test_df.iloc[:, 0]])
le.fit(all_salt)
X_train_df = X_train_df.copy()
X_test_df = X_test_df.copy()
X_train_df.iloc[:, 0] = le.transform(X_train_df.iloc[:, 0])
X_test_df.iloc[:, 0] = le.transform(X_test_df.iloc[:, 0])
X_train_raw = X_train_df.values
X_test_raw = X_test_df.values
feature_names = X_df.columns.tolist()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

joblib.dump(scaler, 'ann_scaler.pkl')
joblib.dump(le, 'ann_label_encoder.pkl')  # 保存标签编码器

print(f"标准化后的训练集形状: {X_train.shape}")
print(f"标准化后的测试集形状: {X_test.shape}")



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


INPUT_DIM = X_train.shape[1]

model = build_ann_model(INPUT_DIM)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',  # 均方误差
    metrics=['mae']  # 平均绝对误差
)

model.summary()


callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
]

EPOCHS = 200
BATCH_SIZE = 64

print("\n开始训练ANN模型...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)
print("ANN模型训练完成!")

test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\n测试集评估:")
print(f"均方误差 (MSE): {test_loss:.4f}")
print(f"平均绝对误差 (MAE): {test_mae:.4f}")

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(f"决定系数 (R²): {r2:.4f}")


plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k', s=80)

# 添加完美预测线
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
         'r--', lw=2.5, label='完美预测线')

# 添加拟合线
z = np.polyfit(y_test.to_numpy().flatten(), y_pred.flatten(), 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), 'g-', lw=2, label=f'拟合线 (y={z[0]:.2f}x + {z[1]:.2f})')

# 添加R²值
plt.text(0.05, 0.9, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
         fontsize=14, bbox=dict(facecolor='white', alpha=0.8))

plt.xlabel('实际值', fontsize=14)
plt.ylabel('预测值', fontsize=14)
plt.title('ANN: 实际值 vs 预测值', fontsize=18, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ann_actual_vs_predicted.png', dpi=300)
plt.close()  # 使用close()避免内存泄漏

# 残差分析
residuals = y_test.to_numpy().flatten() - y_pred.flatten()

plt.figure(figsize=(12, 6))
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
plt.savefig('ann_residual_analysis.png', dpi=300)
plt.close()

# 训练历史可视化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('损失函数变化', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='训练MAE')
plt.plot(history.history['val_mae'], label='验证MAE')
plt.title('MAE变化', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.savefig('ann_training_history.png', dpi=300)
plt.close()


# ---------------------------
# 特征重要性分析（排列法）
# ---------------------------

def permutation_feature_importance(model, X, y, metric=mean_squared_error, n_repeats=5):
    """计算排列特征重要性"""
    baseline_score = metric(y, model.predict(X))
    importances = np.zeros(X.shape[1])

    for i in range(X.shape[1]):
        for _ in range(n_repeats):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            permuted_score = metric(y, model.predict(X_permuted))
            importances[i] += (baseline_score - permuted_score) / n_repeats

    return importances


# 计算特征重要性
print("\n计算特征重要性...")
feature_importances = permutation_feature_importance(model, X_test, y_test)

# 创建特征重要性DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

# 特征重要性条形图
plt.figure(figsize=(12, 8))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance_df,
    color='steelblue',
    edgecolor='black'
)

# 添加数值标签
for i, imp in enumerate(feature_importance_df['Importance']):
    plt.text(imp + 0.001, i, f'{imp:.4f}', va='center', fontsize=10)

plt.xlabel('特征重要性', fontsize=14)
plt.ylabel('特征', fontsize=14)
plt.title('ANN特征重要性 (排列法)', fontsize=18, fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('ann_feature_importance.png', dpi=300)
plt.close()

# ---------------------------
# 模型保存
# ---------------------------

# 保存完整模型
model.save('ann_regression_model.h5')

# 保存为TensorFlow Lite格式（移动端部署）
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('ann_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("\n模型已保存为 ann_regression_model.h5 和 ann_model.tflite")

print("\nANN模型训练与分析完成!")
