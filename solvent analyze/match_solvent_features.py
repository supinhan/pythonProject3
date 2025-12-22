import pandas as pd
from rdkit import Chem

def canonicalize_smiles(smiles):
    """Canonicalize SMILES string for consistent matching."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except:
        pass
    return None

def main():
    # File paths
    chem_smiles_path = r'e:\python\pythonProject3\chemical_smiles.csv'
    features_path = r'e:\python\pythonProject3\solvent analyze\1423_tsne_features.csv'
    output_path = r'e:\python\pythonProject3\solvent analyze\matched_solvent_features.csv'

    print("Loading files...")
    # Load chemical_smiles.csv
    # Index 0: Abbreviation, Index 2: SMILES
    try:
        df_chem = pd.read_csv(chem_smiles_path)
    except:
        df_chem = pd.read_csv(chem_smiles_path, encoding='gbk')
        
    # Load features file
    df_features = pd.read_csv(features_path)

    print("Canonicalizing SMILES for matching...")
    # Create canonical SMILES column for matching in both dataframes
    
    # 1. Process chemical_smiles.csv
    # We need Abbreviation (col 0) and SMILES (col 2)
    # Let's create a clean copy
    df_chem_clean = df_chem.iloc[:, [0, 2]].copy()
    df_chem_clean.columns = ['Abbreviation', 'SMILES_orig']
    df_chem_clean['can_SMILES'] = df_chem_clean['SMILES_orig'].apply(canonicalize_smiles)
    
    # 2. Process features file
    # It already has a SMILES column. Let's canonicalize that too just to be safe, 
    # although reproduce_tsne_optimized.py likely already did it.
    df_features['can_SMILES'] = df_features['SMILES'].apply(canonicalize_smiles)

    print("Merging data...")
    # Merge on canonical SMILES
    # We want to keep Abbreviation from df_chem and all columns from df_features
    # Inner join to find matches
    merged_df = pd.merge(
        df_chem_clean[['Abbreviation', 'can_SMILES']],
        df_features,
        on='can_SMILES',
        how='inner'
    )

    # Reorder columns: Abbreviation first, then the rest
    # Drop the temp 'can_SMILES' and keep original 'SMILES' from features file
    cols = ['Abbreviation'] + [c for c in df_features.columns if c != 'can_SMILES']
    final_df = merged_df[cols]

    print(f"Matched {len(final_df)} solvents.")
    
    final_df.to_csv(output_path, index=False)
    print(f"Saved matched features to {output_path}")

if __name__ == "__main__":
    main()
