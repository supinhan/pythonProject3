import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import joblib

plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path):
    desired_columns = pd.read_csv(file_path, nrows=0).columns.tolist()
    df = pd.read_csv(file_path, usecols=desired_columns)
    y = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    X_df = df.iloc[:, 1:16]  # 第二列到第16列（共15列）
    X_df.iloc[:, 0] = X_df.iloc[:, 0].astype(str)
    return X_df, y, X_df.columns.tolist()

def preprocess_features(X_df):
    le = LabelEncoder()
    all_salt = pd.concat([X_df.iloc[:, 0]])
    le.fit(all_salt)
    X_df = X_df.copy()
    X_df.iloc[:, 0] = le.transform(X_df.iloc[:, 0])
    return X_df.values, le


def random_forest_feature_analysis(X, y, feature_names):
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        oob_score=True
    )
    rf.fit(X, y)
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
        'Std': std
    }).sort_values('Importance', ascending=False)
    print("\n随机森林特征重要性排序:")
    print(feature_importance_df[['Feature', 'Importance']])
    return feature_importance_df, rf


# 可视化特征重要性
def visualize_feature_importance(feature_importance_df):
    """可视化随机森林特征重要性"""
    plt.figure(figsize=(12, 8))

    # 排序特征
    sorted_df = feature_importance_df.sort_values('Importance', ascending=True)

    # 创建条形图
    sns.barplot(
        x='Importance',
        y='Feature',
        data=sorted_df,
        color='steelblue',
        edgecolor='black',
        xerr=sorted_df['Std']
    )

    # 添加数值标签
    for i, (imp, std_val) in enumerate(zip(sorted_df['Importance'], sorted_df['Std'])):
        plt.text(imp + 0.005, i, f'{imp:.4f} ± {std_val:.4f}',
                 va='center', fontsize=10, color='black')

    plt.xlabel('特征重要性', fontsize=14)
    plt.ylabel('特征', fontsize=14)
    plt.title('随机森林特征重要性', fontsize=18, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png', dpi=300)
    plt.close()


if __name__ == "__main__":
    data_path = '广精样本.csv'
    X_df, y, feature_names = load_data(data_path)
    X, le = preprocess_features(X_df)
    print("=" * 50)
    print("开始随机森林特征重要性分析")
    print("=" * 50)
    rf_importance_df, rf_model = random_forest_feature_analysis(X, y, feature_names)

    # 可视化结果
    visualize_feature_importance(rf_importance_df)

    # 保存结果
    joblib.dump(rf_model, 'rf_feature_importance_model.pkl')
    joblib.dump(le, 'rf_label_encoder.pkl')
    rf_importance_df.to_csv('rf_feature_importance.csv', index=False)
    print("\n随机森林特征分析完成! 结果已保存")