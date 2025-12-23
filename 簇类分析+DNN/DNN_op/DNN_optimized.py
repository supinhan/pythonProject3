import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization, 
                                      Input, Multiply, Activation, Lambda)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
import matplotlib

matplotlib.use('Agg')
warnings.filterwarnings("ignore")


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

print("加载数据集...")
df = pd.read_csv(r'..\数据预处理\簇类加权特征数据集.csv')

y = pd.to_numeric(df.iloc[:, 1], errors='coerce')
X_raw = df.drop(df.columns[1], axis=1)

print(f"原始数据形状: {df.shape}")
print(f"目标变量形状: {y.shape}")

# 特征选择
print("\n特征选择：移除重要性<0.05的特征")

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
    'tsne_x', 'tsne_y',
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
    'Linear Carbonyl', 'Cyclic Carbonyl', 'Linear Ether', 'Cyclic Ether'
]

available_features = [f for f in important_features if f in X_raw.columns]
print(f"保留的重要特征数量: {len(available_features)}")

X_selected = X_raw[available_features].copy()

# 多项式温度特征
print("\n添加多项式温度特征")

if 'T' in X_selected.columns:
    T_values = X_selected['T'].values.reshape(-1, 1)
    poly = PolynomialFeatures(degree=3, include_bias=False)
    T_poly = poly.fit_transform(T_values)
    
    X_selected['T_squared'] = T_poly[:, 1]
    X_selected['T_cubed'] = T_poly[:, 2]
    print(f"已添加多项式温度特征: T², T³")
    print(f"特征形状变化: {len(available_features)} -> {X_selected.shape[1]}")

# One-Hot编码
print("\nOne-Hot编码分类特征")

categorical_cols = ['salt']
if 'salt' in X_selected.columns:
    X_encoded = pd.get_dummies(X_selected, columns=['salt'], prefix='salt')
else:
    salt_col_name = X_selected.columns[0]
    print(f"假设 '{salt_col_name}' 是salt列")
    X_encoded = pd.get_dummies(X_selected, columns=[salt_col_name], prefix='salt')

print(f"One-Hot编码后形状: {X_encoded.shape}")

valid_idx = ~(X_encoded.isna().any(axis=1) | y.isna())
X_encoded = X_encoded[valid_idx]
y = y[valid_idx]
print(f"移除NaN后样本数: {len(y)}")

feature_names = X_encoded.columns.tolist()


# ---------------------------
# Attention机制DNN模型
# ---------------------------

def attention_layer(inputs, name_prefix='attention'):
    """自注意力机制层"""
    attention_weights = Dense(inputs.shape[-1], activation='tanh', 
                             name=f'{name_prefix}_dense1')(inputs)
    attention_weights = Dense(inputs.shape[-1], activation='softmax',
                             name=f'{name_prefix}_weights')(attention_weights)
    attended = Multiply(name=f'{name_prefix}_multiply')([inputs, attention_weights])
    return attended, attention_weights

def build_attention_dnn(input_dim):
    """构建带有Attention机制的DNN模型"""
    inputs = Input(shape=(input_dim,), name='input')
    
    x, attn_weights = attention_layer(inputs, name_prefix='attn1')
    
    x = Dense(128, activation='relu', name='dense1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = Dropout(0.3, name='dropout1')(x)
    
    x = Dense(256, activation='relu', name='dense2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Dropout(0.4, name='dropout2')(x)
    
    x, _ = attention_layer(x, name_prefix='attn2')
    
    x = Dense(512, activation='relu', name='dense3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = Dropout(0.5, name='dropout3')(x)
    
    x = Dense(256, activation='relu', name='dense4')(x)
    x = BatchNormalization(name='bn4')(x)
    x = Dropout(0.4, name='dropout4')(x)
    
    x = Dense(128, activation='relu', name='dense5')(x)
    x = BatchNormalization(name='bn5')(x)
    x = Dropout(0.3, name='dropout5')(x)
    
    outputs = Dense(1, activation='linear', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='AttentionDNN')
    return model


# ---------------------------
# K-Fold交叉验证训练
# ---------------------------

print("\nK-Fold交叉验证训练 (5-Fold)")

N_FOLDS = 5
EPOCHS = 200
BATCH_SIZE = 64

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
y_values = y.values

joblib.dump(scaler, 'optimized_scaler.pkl')

kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_results = []
all_predictions = np.zeros(len(y_values))
all_histories = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
    print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
    
    X_train_fold = X_scaled[train_idx]
    X_val_fold = X_scaled[val_idx]
    y_train_fold = y_values[train_idx]
    y_val_fold = y_values[val_idx]
    
    model = build_attention_dnn(X_scaled.shape[1])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    if fold == 0:
        model.summary()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
    ]
    
    history = model.fit(
        X_train_fold, y_train_fold,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val_fold, y_val_fold),
        callbacks=callbacks,
        verbose=0
    )
    
    all_histories.append(history)
    
    y_pred_fold = model.predict(X_val_fold, verbose=0).flatten()
    all_predictions[val_idx] = y_pred_fold
    
    fold_mse = mean_squared_error(y_val_fold, y_pred_fold)
    fold_mae = mean_absolute_error(y_val_fold, y_pred_fold)
    fold_r2 = r2_score(y_val_fold, y_pred_fold)
    
    fold_results.append({
        'fold': fold + 1,
        'mse': fold_mse,
        'mae': fold_mae,
        'r2': fold_r2
    })
    
    print(f"  MSE: {fold_mse:.4f}, MAE: {fold_mae:.4f}, R²: {fold_r2:.4f}")


# ---------------------------
# K-Fold结果汇总
# ---------------------------

print("\nK-Fold交叉验证结果汇总")

results_df = pd.DataFrame(fold_results)
print(results_df.to_string(index=False))

print(f"\n平均指标:")
print(f"  MSE: {results_df['mse'].mean():.4f} ± {results_df['mse'].std():.4f}")
print(f"  MAE: {results_df['mae'].mean():.4f} ± {results_df['mae'].std():.4f}")
print(f"  R²:  {results_df['r2'].mean():.4f} ± {results_df['r2'].std():.4f}")

results_df.to_csv('kfold_results.csv', index=False)


# ---------------------------
# 训练最终模型
# ---------------------------

print("\n训练最终模型（全数据集）")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_values, test_size=0.2, random_state=42
)

final_model = build_attention_dnn(X_scaled.shape[1])
final_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
]

final_history = final_model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

test_loss, test_mae = final_model.evaluate(X_test, y_test, verbose=0)
y_pred = final_model.predict(X_test, verbose=0).flatten()
r2 = r2_score(y_test, y_pred)

print(f"\n最终模型测试集评估:")
print(f"  MSE: {test_loss:.4f}")
print(f"  MAE: {test_mae:.4f}")
print(f"  R²:  {r2:.4f}")


# ---------------------------
# 可视化
# ---------------------------

print("\n生成可视化图表")

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
plt.title('Attention-DNN: 实际值 vs 预测值 (K-Fold优化)', fontsize=18, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('optimized_dnn_actual_vs_predicted.png', dpi=300)
plt.close()

# 残差分析
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

# 训练历史可视化
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

# K-Fold结果对比
plt.figure(figsize=(10, 6))
x = np.arange(N_FOLDS)
width = 0.25

plt.bar(x - width, results_df['mse'], width, label='MSE', color='steelblue')
plt.bar(x, results_df['mae'], width, label='MAE', color='coral')
plt.bar(x + width, results_df['r2'], width, label='R²', color='seagreen')

plt.xlabel('Fold', fontsize=12)
plt.ylabel('指标值', fontsize=12)
plt.title('K-Fold交叉验证各折结果对比', fontsize=16, fontweight='bold')
plt.xticks(x, [f'Fold {i+1}' for i in range(N_FOLDS)])
plt.legend()
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('optimized_dnn_kfold_comparison.png', dpi=300)
plt.close()


# ---------------------------
# 特征重要性（分组）
# ---------------------------

print("\n计算分组特征重要性")

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

groups = get_feature_groups(feature_names, categorical_cols)

# 合并 c_val 和 c_units
c_indices = []
if 'c_val' in groups:
    c_indices.extend(groups['c_val'])
    del groups['c_val']
if 'c_units' in groups:
    c_indices.extend(groups['c_units'])
    del groups['c_units']
if c_indices:
    groups['c'] = c_indices

# 合并温度特征
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

feature_importances = grouped_permutation_importance(final_model, X_test, y_test, groups)

feature_importance_df = pd.DataFrame({
    'Feature': list(feature_importances.keys()),
    'Importance': list(feature_importances.values())
}).sort_values('Importance', ascending=False)

feature_importance_df.to_csv('optimized_feature_importance.csv', index=False)

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
plt.title(f'分组特征重要性 (Top {top_n}) - Attention-DNN', fontsize=18, fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('optimized_dnn_feature_importance.png', dpi=300)
plt.close()


# ---------------------------
# 模型保存
# ---------------------------

final_model.save('optimized_attention_dnn_model.h5')
print("\n模型保存: optimized_attention_dnn_model.h5")

print(f"""
模型改进内容:
  1. Attention机制: 自适应学习特征权重
  2. K-Fold交叉验证: {N_FOLDS}折交叉验证
  3. 特征选择: 移除重要性<0.05的特征
  4. 多项式温度特征: T, T², T³

K-Fold交叉验证结果:
  平均 R²:  {results_df['r2'].mean():.4f} ± {results_df['r2'].std():.4f}
  平均 MSE: {results_df['mse'].mean():.4f} ± {results_df['mse'].std():.4f}
  平均 MAE: {results_df['mae'].mean():.4f} ± {results_df['mae'].std():.4f}

最终模型测试集结果:
  R²:  {r2:.4f}
  MSE: {test_loss:.4f}
  MAE: {test_mae:.4f}
""")

print("训练完成!")
