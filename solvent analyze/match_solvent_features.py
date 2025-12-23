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
    chem_smiles_path = r'e:\python\pythonProject3\chemical_smiles.csv'
    features_path = r'e:\python\pythonProject3\solvent analyze\1423_tsne_features.csv'
    output_path = r'e:\python\pythonProject3\solvent analyze\matched_solvent_features.csv'

    print("Loading files...")
    # Load chemical_smiles.csv
    try:
        df_chem = pd.read_csv(chem_smiles_path)
    except:
        df_chem = pd.read_csv(chem_smiles_path, encoding='gbk')

    df_features = pd.read_csv(features_path)

    print("Canonicalizing SMILES for matching...")
    

    df_chem_clean = df_chem.iloc[:, [0, 2]].copy()
    df_chem_clean.columns = ['Abbreviation', 'SMILES_orig']
    df_chem_clean['can_SMILES'] = df_chem_clean['SMILES_orig'].apply(canonicalize_smiles)
    

    df_features['can_SMILES'] = df_features['SMILES'].apply(canonicalize_smiles)

    print("Merging data...")

    merged_df = pd.merge(
        df_chem_clean[['Abbreviation', 'can_SMILES']],
        df_features,
        on='can_SMILES',
        how='inner'
    )


    cols = ['Abbreviation'] + [c for c in df_features.columns if c != 'can_SMILES']
    final_df = merged_df[cols]

    print(f"Matched {len(final_df)} solvents.")
    
    final_df.to_csv(output_path, index=False)
    print(f"Saved matched features to {output_path}")

if __name__ == "__main__":
    main()
