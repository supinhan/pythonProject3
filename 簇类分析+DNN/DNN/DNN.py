# full-random42+DNN.py
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


# 字体设置
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

# 1. Data Loading
print("Loading dataset...")
# Reading the specific file requested
# Reading the specific file requested
df = pd.read_csv(r'..\数据预处理\簇类加权特征数据集.csv')

# Target is the second column (index 1) 'k(ms*cm-1)', assumes first column is 'salt'
# Based on file inspection: 1: salt, k(ms*cm-1), T, ...
# So salt is column 0, k is column 1.
y = pd.to_numeric(df.iloc[:, 1], errors='coerce')

# Features: 'salt' needs One-Hot, others are numerical features.
# Taking all columns except the target (index 1) as initial features to process
X_raw = df.drop(df.columns[1], axis=1)

print(f"Original shape: {df.shape}")
print(f"Target shape: {y.shape}")

# 2. Preprocessing
# One-Hot Encoding for 'salt'
print("One-Hot Encoding 'salt' column...")
if 'salt' in X_raw.columns:
    X_encoded = pd.get_dummies(X_raw, columns=['salt'], prefix='salt')
else:
    # Fallback if 'salt' is not named 'salt' but is the first column
    salt_col_name = X_raw.columns[0]
    print(f"Assuming '{salt_col_name}' is salt.")
    X_encoded = pd.get_dummies(X_raw, columns=[salt_col_name], prefix='salt')

print(f"Shape after One-Hot Encoding: {X_encoded.shape}")

# Split Data
X_train_df, X_test_df, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train_df.shape[0]} samples")
print(f"Test set: {X_test_df.shape[0]} samples")

# Standardize features
# Note: One-Hot encoded columns are usually left as 0/1, but sometimes standardized.
# The reference script standardized ALL inputs after label encoding.
# To be consistent with "features need scaling" for DNNs, we scale everything.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_df)
X_test = scaler.transform(X_test_df)

joblib.dump(scaler, 'full_random100_scaler.pkl')
feature_names = X_encoded.columns.tolist()

print(f"Standardized Training set shape: {X_train.shape}")


# 3. Model Architecture
def build_dnn_model(input_dim):
    model = Sequential()

    # Replicating structure from random100+DNN.py
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

# 4. Training
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

# 5. Evaluation
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Set Evaluation:")
print(f"MSE: {test_loss:.4f}")
print(f"MAE: {test_mae:.4f}")

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R²: {r2:.4f}")

# 6. Visualization & Saving
# Actual vs Predicted
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k', s=80)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2.5, label='Perfect Prediction')
z = np.polyfit(y_test.to_numpy().flatten(), y_pred.flatten(), 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), 'g-', lw=2, label=f'Fit Line (y={z[0]:.2f}x + {z[1]:.2f})')
plt.text(0.05, 0.9, f'R² = {r2:.4f}', transform=plt.gca().transAxes, fontsize=14,
         bbox=dict(facecolor='white', alpha=0.8))
plt.xlabel('Actual', fontsize=14)
plt.ylabel('Predicted', fontsize=14)
plt.title('DNN: Actual vs Predicted (Full Random 100)', fontsize=18, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('full_random100_dnn_actual_vs_predicted.png', dpi=300)
plt.close()

# Residuals
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

# Training History
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


# 6. Feature Importance (Grouped)
def get_feature_groups(all_columns, categorical_cols):
    groups = {}
    used_indices = set()

    # Handle One-Hot Encoded Categories
    for cat_col in categorical_cols:
        # One-Hot columns default format: "colname_value"
        # We find columns starting with "colname_"
        indices = [i for i, col in enumerate(all_columns) if col.startswith(f"{cat_col}_")]
        if indices:
            groups[cat_col] = indices
            used_indices.update(indices)

    # Handle remaining (numeric) features
    for i, col in enumerate(all_columns):
        if i not in used_indices:
            groups[col] = [i]

    return groups


def grouped_permutation_importance(model, X, y, groups, metric=mean_squared_error, n_repeats=5):
    # Calculate baseline score (MSE: lower is better)
    baseline_score = metric(y, model.predict(X, verbose=0))
    importances = {}

    print(f"Baseline MSE: {baseline_score:.4f}")

    for group_name, indices in groups.items():
        score_increases = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            # Permute rows for this group of columns together
            perm_indices = np.random.permutation(X.shape[0])
            X_permuted[:, indices] = X_permuted[perm_indices][:, indices]

            permuted_score = metric(y, model.predict(X_permuted, verbose=0))

            # Importance = Increase in Error (Permuted MSE - Baseline MSE)
            # Higher increase means feature was important
            score_increases.append(permuted_score - baseline_score)

        importances[group_name] = np.mean(score_increases)

    return importances


print("\nCalculating Grouped Feature Importance...")
# Identify categorical columns again (from column names in dataframe)
# Note: In X_encoded, one-hot cols are flattened.
# We need the original categorical_cols list. It was defined earlier.
# If not available in this scope, we redefine logic or use global
if 'categorical_cols' not in locals():
    # User reverted auto-detection, but we need this for grouping.
    # explicit definition for salt as requested.
    categorical_cols = ['salt']
    # Check if solvent_ratio_type is also encoded?
    # In user's version (Step 98), only 'salt' was passed to get_dummies columns=['salt']
    # So we should only group 'salt'.
    print(f"Using defined categorical columns for grouping: {categorical_cols}")

groups = get_feature_groups(feature_names, categorical_cols)

# --- MERGE RULES: Combine c_val and c_units into 'c' ---
# c_val is numeric, c_units is likely categorical (one-hot groups).
c_indices = []

# 1. Add c_val
if 'c_val' in groups:
    c_indices.extend(groups['c_val'])
    del groups['c_val']

# 2. Add c_units
if 'c_units' in groups:
    c_indices.extend(groups['c_units'])
    del groups['c_units']

# 3. Handle case where c_units might not be in categorical_cols grouping but exists individually
# (e.g. if it was passed partially). We scan keys just in case.
keys_to_remove = []
for key in groups:
    if key.startswith('c_units'):
        c_indices.extend(groups[key])
        keys_to_remove.append(key)

for key in list(set(keys_to_remove)):  # set to deduplicate if c_units was caught twice
    if key in groups: del groups[key]

# Create unified 'c' group
if c_indices:
    groups['c'] = c_indices
    groups['c'] = c_indices
    print(f"Grouped 'c_val' and 'c_units' into 'c' with {len(c_indices)} columns.")

# 4. Group 'Cluster projection' (tsne_x, tsne_y)
cluster_proj_indices = []
for col in ['tsne_x', 'tsne_y']:
    if col in groups:
        cluster_proj_indices.extend(groups[col])
        del groups[col]
if cluster_proj_indices:
    groups['Cluster projection'] = cluster_proj_indices
    print(f"Grouped tsne features into 'Cluster projection'.")

# 5. Group 'Types of organic solvents' (4 one-hot cols)
solvent_types = ['Linear Carbonyl', 'Cyclic Carbonyl', 'Linear Ether', 'Cyclic Ether']
type_indices = []
for st in solvent_types:
    if st in groups:
        type_indices.extend(groups[st])
        del groups[st]
if type_indices:
    groups['Types of organic solvents'] = type_indices
    print(f"Grouped solvent types into 'Types of organic solvents'.")

feature_importances = grouped_permutation_importance(model, X_test, y_test, groups)

# Create DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': list(feature_importances.keys()),
    'Importance': list(feature_importances.values())
}).sort_values('Importance', ascending=False)

# Save to CSV
feature_importance_df.to_csv('full_random100_feature_importance.csv', index=False)
print("Feature importance saved to full_random100_feature_importance.csv")

# Plot
top_n = 20
plt.figure(figsize=(12, 8))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance_df.head(top_n),
    color='steelblue',
    edgecolor='black'
)
# Add labels
for i, (imp, name) in enumerate(
        zip(feature_importance_df['Importance'].head(top_n), feature_importance_df['Feature'].head(top_n))):
    plt.text(imp, i, f'{imp:.4f}', va='center', fontsize=10)

plt.xlabel('Importance (Increase in MSE)', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title(f'Grouped Feature Importance (Top {top_n})', fontsize=18, fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('full_random100_dnn_feature_importance.png', dpi=300)
plt.close()

# Saving
model.save('full_random100_dnn_model.h5')
print("\nModel saved as full_random100_dnn_model.h5")
