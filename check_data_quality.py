import pandas as pd
import numpy as np


def check_data():
    file_path = '加权特征数据集.csv'
    print(f"Checking {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print(f"Shape: {df.shape}")
        
        # Check for empty/NaN cells
        nan_counts = df.isna().sum()
        cols_with_nans = nan_counts[nan_counts > 0]
        
        if not cols_with_nans.empty:
            print("\nColumns with NaNs:")
            print(cols_with_nans)
        else:
            print("\nNo direct NaNs found in DataFrame.")

        # Check for non-numeric in target column (index 1) which might become NaN
        target_col = df.columns[1] # k(ms*cm-1)
        print(f"\nChecking target column '{target_col}' for non-numeric values...")
        target_numeric = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        nan_targets = target_numeric.isna().sum()
        if nan_targets > 0:
            print(f"Found {nan_targets} rows where target is non-numeric (became NaN).")
            # Show examples
            invalid_rows = df[target_numeric.isna()]
            print("Examples of invalid target values:")
            print(invalid_rows.iloc[:, :2].head())
        else:
            print("Target column is clean.")

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    check_data()
