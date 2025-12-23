# -*- coding: utf-8 -*-
"""
根据各溶剂组分的比例，对单一溶剂的特征进行加权求和，得到混合物的综合特征
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def load_data(data_path: str, solvent_feature_path: str):

    df_data = pd.read_csv(data_path)
    df_solvent_features = pd.read_csv(solvent_feature_path, index_col=0)
    
    # 转置溶剂特征表，使行为溶剂名，列为特征名
    df_solvent_features = df_solvent_features.T
    
    print(f"主数据形状: {df_data.shape}")
    print(f"溶剂特征形状 (溶剂数 x 特征数): {df_solvent_features.shape}")
    
    return df_data, df_solvent_features


def get_solvent_columns(df_data: pd.DataFrame) -> list:

    # 固定列（非溶剂比例列）
    fixed_cols = ['salt', 'k(ms*cm-1)', 'T', 'c_val', 'c_units', 'solvent_ratio_type']
    

    solvent_cols = [col for col in df_data.columns if col not in fixed_cols]
    
    print(f"检测到 {len(solvent_cols)} 种溶剂组分")
    
    return solvent_cols


def calculate_weighted_features(df_data: pd.DataFrame, 
                                 df_solvent_features: pd.DataFrame,
                                 solvent_cols: list) -> pd.DataFrame:
    # 获取所有特征名
    feature_names = df_solvent_features.columns.tolist()
    
    # 初始化结果矩阵
    n_samples = len(df_data)
    n_features = len(feature_names)
    weighted_features = np.zeros((n_samples, n_features))
    
    print(f"正在计算 {n_samples} 个样本的加权特征...")
    
    # 找出共有的溶剂
    common_solvents = [s for s in solvent_cols if s in df_solvent_features.index]
    print(f"共有 {len(common_solvents)} 种溶剂在特征表中有对应特征")
    
    # 对于每种溶剂，计算其比例加权贡献
    for solvent in common_solvents:
        # 获取该溶剂的比例（列向量）
        ratios = df_data[solvent].values.reshape(-1, 1)  # shape: (n_samples, 1)
        
        # 获取该溶剂的所有特征（行向量）
        features = df_solvent_features.loc[solvent].values.reshape(1, -1)  # shape: (1, n_features)
        
        # 加权贡献 = 比例 × 特征
        weighted_features += ratios * features
    
    # 创建DataFrame
    df_weighted = pd.DataFrame(
        weighted_features,
        columns=[f'weighted_{feat}' for feat in feature_names]
    )
    
    return df_weighted


def create_final_dataset(df_data: pd.DataFrame,
                         df_weighted_features: pd.DataFrame,
                         include_original_ratios: bool = False) -> pd.DataFrame:
    # 提取非溶剂列
    base_cols = ['salt', 'k(ms*cm-1)', 'T', 'c_val', 'c_units', 'solvent_ratio_type']
    df_base = df_data[base_cols].copy()

    df_final = pd.concat([df_base, df_weighted_features], axis=1)
    
    if include_original_ratios:
        solvent_cols = get_solvent_columns(df_data)
        df_ratios = df_data[solvent_cols].copy()
        df_final = pd.concat([df_final, df_ratios], axis=1)
    
    return df_final


def main():

    data_path = '../../数据.csv'
    solvent_feature_path = '../../溶剂标识符wide.csv'
    output_path = '加权特征数据集.csv'
    
    print("=" * 60)
    print("溶剂混合物加权特征提取")
    print("=" * 60)

    print("\n[1/4] 加载数据...")
    df_data, df_solvent_features = load_data(data_path, solvent_feature_path)

    print("\n[2/4] 识别溶剂组分...")
    solvent_cols = get_solvent_columns(df_data)

    print("\n[3/4] 计算加权特征...")
    df_weighted = calculate_weighted_features(df_data, df_solvent_features, solvent_cols)

    print("\n[4/4] 构建最终数据集...")
    df_final = create_final_dataset(df_data, df_weighted, include_original_ratios=False)

    df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"\n输出文件: {output_path}")
    print(f"数据集形状: {df_final.shape}")
    print(f"  - 样本数: {df_final.shape[0]}")
    print(f"  - 特征数: {df_final.shape[1]}")

    print("\n数据集列名:")
    print("-" * 40)
    for i, col in enumerate(df_final.columns):
        print(f"  {i+1:3d}. {col}")
    
    print("\n前5行预览:")
    print(df_final.head())
    
    return df_final


if __name__ == "__main__":
    df_result = main()
