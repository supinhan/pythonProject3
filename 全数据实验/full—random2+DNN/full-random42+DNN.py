import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import matplotlib as mpl
import warnings
import matplotlib

matplotlib.use('Agg')
warnings.filterwarnings("ignore")


def configure_fonts():
    import platform
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

print("Loading dataset...")
df = pd.read_csv('加权特征数据集.csv')

y = pd.to_numeric(df.iloc[:, 1], errors='coerce')
X_raw = df.drop(df.columns[1], axis=1)

print(f"Original shape: {df.shape}")
print(f"Target shape: {y.shape}")

# One-Hot编码
print("One-Hot Encoding 'salt' column...")
if 'salt' in X_raw.columns:
    X_encoded = pd.get_dummies(X_raw, columns=['salt'], prefix='salt')
else:
    salt_col_name = X_raw.columns[0]
    print(f"Assuming '{salt_col_name}' is salt.")
    X_encoded = pd.get_dummies(X_raw, columns=[salt_col_name], prefix='salt')

print(f"Shape after One-Hot Encoding: {X_encoded.shape}")

X_train_df, X_test_df, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train_df.shape[0]} samples")
print(f"Test set: {X_test_df.shape[0]} samples")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_df)
X_test = scaler.transform(X_test_df)

joblib.dump(scaler, 'full_random100_scaler.pkl')
feature_names = X_encoded.columns.tolist()

print(f"Standardized Training set shape: {X_train.shape}")


def build_dnn_model(input_dim):
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
model = build_dnn_model(INPUT_DIM)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
]

EPOCHS = 200
BATCH_SIZE = 64

print("\nStarting DNN training...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)
print("DNN training completed!")

test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Set Evaluation:")
print(f"MSE: {test_loss:.4f}")
print(f"MAE: {test_mae:.4f}")

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R²: {r2:.4f}")

plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k', s=80)

# 添加完美预测线
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2.5, label='Perfect Prediction')

# 添加拟合线
z = np.polyfit(y_test.to_numpy().flatten(), y_pred.flatten(), 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), 'g-', lw=2, label=f'Fit Line (y={z[0]:.2f}x + {z[1]:.2f})')

# 添加R²值
plt.text(0.05, 0.9, f'R² = {r2:.4f}', transform=plt.gca().transAxes, fontsize=14, bbox=dict(facecolor='white', alpha=0.8))

plt.xlabel('Actual', fontsize=14)
plt.ylabel('Predicted', fontsize=14)
plt.title('DNN: Actual vs Predicted (Full Random 100)', fontsize=18, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('full_random100_dnn_actual_vs_predicted.png', dpi=300)
plt.close()

# 残差分析
residuals = y_test.to_numpy().flatten() - y_pred.flatten()
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(residuals, kde=True, color='steelblue', bins=30)
plt.axvline(x=0, color='r', linestyle='--')
plt.title('Residual Distribution', fontsize=16)
plt.xlabel('Residual', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

plt.subplot(1, 2, 2)
plt.scatter(y_pred, residuals, alpha=0.6, edgecolor='k')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Predicted', fontsize=16)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Residual', fontsize=12)
plt.tight_layout()
plt.savefig('full_random100_dnn_residual_analysis.png', dpi=300)
plt.close()

# 训练历史可视化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss (MSE)', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('MAE', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.savefig('full_random100_dnn_training_history.png', dpi=300)
plt.close()


# ---------------------------
# 特征重要性分析（分组）
# ---------------------------

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
    """计算分组排列特征重要性"""
    baseline_score = metric(y, model.predict(X, verbose=0))
    importances = {}

    print(f"Baseline MSE: {baseline_score:.4f}")

    for group_name, indices in groups.items():
        score_increases = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            perm_indices = np.random.permutation(X.shape[0])
            X_permuted[:, indices] = X_permuted[perm_indices][:, indices]

            permuted_score = metric(y, model.predict(X_permuted, verbose=0))
            score_increases.append(permuted_score - baseline_score)

        importances[group_name] = np.mean(score_increases)

    return importances


print("\nCalculating Grouped Feature Importance...")

if 'categorical_cols' not in locals():
    categorical_cols = ['salt']
    print(f"Using defined categorical columns for grouping: {categorical_cols}")

groups = get_feature_groups(feature_names, categorical_cols)

# 合并 c_val 和 c_units
c_indices = []
if 'c_val' in groups:
    c_indices.extend(groups['c_val'])
    del groups['c_val']
if 'c_units' in groups:
    c_indices.extend(groups['c_units'])
    del groups['c_units']

keys_to_remove = []
for key in groups:
    if key.startswith('c_units'):
        c_indices.extend(groups[key])
        keys_to_remove.append(key)

for key in list(set(keys_to_remove)):
    if key in groups: del groups[key]

if c_indices:
    groups['c'] = c_indices
    print(f"Grouped 'c_val' and 'c_units' into 'c' with {len(c_indices)} columns.")

feature_importances = grouped_permutation_importance(model, X_test, y_test, groups)

feature_importance_df = pd.DataFrame({
    'Feature': list(feature_importances.keys()),
    'Importance': list(feature_importances.values())
}).sort_values('Importance', ascending=False)

feature_importance_df.to_csv('full_random100_feature_importance.csv', index=False)
print("Feature importance saved to full_random100_feature_importance.csv")

top_n = 20
plt.figure(figsize=(12, 8))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance_df.head(top_n),
    color='steelblue',
    edgecolor='black'
)

# 添加数值标签
for i, (imp, name) in enumerate(zip(feature_importance_df['Importance'].head(top_n), feature_importance_df['Feature'].head(top_n))):
    plt.text(imp, i, f'{imp:.4f}', va='center', fontsize=10)

plt.xlabel('Importance (Increase in MSE)', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title(f'Grouped Feature Importance (Top {top_n})', fontsize=18, fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('full_random100_dnn_feature_importance.png', dpi=300)
plt.close()

# ---------------------------
# 模型保存
# ---------------------------

model.save('full_random100_dnn_model.h5')
print("\nModel saved as full_random100_dnn_model.h5")
