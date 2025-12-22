import pandas as pd
import numpy as np
import os

def load_data(data_path, wide_features_path, matched_features_path):
    # 1. Load Main Data
    df_data = pd.read_csv(data_path)
    print(f"Loaded Data: {df_data.shape}")
    
    # 2. Load Solvent Identification Wide (Physical Properties)
    # Index is Feature Name, Columns are Solvents. We transpose it.
    df_wide = pd.read_csv(wide_features_path, index_col=0).T
    print(f"Loaded Wide Features: {df_wide.shape}")
    
    # 3. Load Matched t-SNE Features
    # Columns: Abbreviation, Linear Carbonyl, ... tsne_y (SMILES was removed? User said "don't record SMILES")
    # We need to set Abbreviation as index to match with df_wide
    try:
        df_matched = pd.read_csv(matched_features_path, encoding='utf-8')
    except:
        df_matched = pd.read_csv(matched_features_path, encoding='gbk')
        
    print(f"Loaded Matched Features: {df_matched.shape}")
    
    # Validating Abbreviation column
    if 'Abbreviation' not in df_matched.columns:
        # Check if index relates to abbreviation, or if it's the first column
        # Previous script saved it as first column 'Abbreviation'
        print("Warning: 'Abbreviation' column not found. Using first column as key.")
        df_matched.rename(columns={df_matched.columns[0]: 'Abbreviation'}, inplace=True)
        
    # Drop SMILES if present (as requested "don't record SMILES number")
    if 'SMILES' in df_matched.columns:
        df_matched.drop(columns=['SMILES'], inplace=True)
        
    # Set index for joining
    df_matched.set_index('Abbreviation', inplace=True)
    
    # 4. Merge Wide Features & Matched Features for each solvent
    # We combine physical properties + t-SNE features
    # Inner join or Left join? Standard solvents in wide file should be covered.
    df_solvent_all = pd.merge(df_wide, df_matched, left_index=True, right_index=True, how='left')
    
    # Fill NaN if any matched features missing? Or zero fill? 
    # t-SNE coordinates shouldn't be zero-filled arbitrarily, but if mostly matching it's fine.
    # We'll valid rows.
    print(f"Combined Solvent Features shape: {df_solvent_all.shape}")
    
    return df_data, df_solvent_all

def get_solvent_columns(df_data):
    # Fixed columns to exclude
    fixed_cols = ['salt', 'k(ms*cm-1)', 'T', 'c_val', 'c_units', 'solvent_ratio_type']
    solvent_cols = [c for c in df_data.columns if c not in fixed_cols]
    return solvent_cols

def calculate_weighted_features(df_data, df_solvent_features, solvent_cols):
    # Filter common solvents
    common_solvents = [s for s in solvent_cols if s in df_solvent_features.index]
    print(f"Using {len(common_solvents)} solvents present in feature set.")
    
    missing = set(solvent_cols) - set(common_solvents)
    if missing:
        print(f"Warning: Missing features for solvents: {missing}")

    feature_names = df_solvent_features.columns.tolist()
    n_samples = len(df_data)
    n_features = len(feature_names)
    
    weighted_matrix = np.zeros((n_samples, n_features))
    
    for solvent in common_solvents:
        # Ratio (N, 1)
        ratios = df_data[solvent].values.reshape(-1, 1)
        # Features (1, F)
        vals = df_solvent_features.loc[solvent].values.reshape(1, -1)
        # Weighted accumulation
        weighted_matrix += ratios * vals
        
    df_weighted = pd.DataFrame(weighted_matrix, columns=feature_names)
    return df_weighted

def main():
    base_dir = r'e:\python\pythonProject3\簇类分析+DNN\数据预处理'
    data_path = os.path.join(base_dir, '数据.csv')
    wide_path = os.path.join(base_dir, '溶剂标识符wide.csv')
    matched_path = os.path.join(base_dir, 'matched_solvent_features.csv')
    output_path = os.path.join(base_dir, '簇类加权特征数据集.csv')
    
    print("Starting cluster weighted feature generation...")
    
    df_data, df_solvent_all = load_data(data_path, wide_path, matched_path)
    
    solvent_cols = get_solvent_columns(df_data)
    
    df_weighted = calculate_weighted_features(df_data, df_solvent_all, solvent_cols)
    
    # Combine with original fixed columns
    fixed_cols = ['salt', 'k(ms*cm-1)', 'T', 'c_val', 'c_units', 'solvent_ratio_type']
    df_final = pd.concat([df_data[fixed_cols], df_weighted], axis=1)
    
    # Ensure target k(ms*cm-1) is present
    
    df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Saved {output_path}")
    print(f"Final Shape: {df_final.shape}")

if __name__ == "__main__":
    main()
