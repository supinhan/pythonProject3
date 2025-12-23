import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from rdkit import Chem
from rdkit.Chem import AllChem

try:
    import deepchem as dc
    HAS_DEEPCHEM = True
except ImportError:
    HAS_DEEPCHEM = False
    print("DeepChem not found. Using RDKit for fingerprints.")

def load_highlight_smiles(filepath):
    """Loads"""
    try:
        df = pd.read_csv(filepath)
    except:
        df = pd.read_csv(filepath, encoding='gbk')

    smiles_col = df.iloc[:, 2]
    abbr_col = df.iloc[:, 0]
    
    highlights = []
    for s, abbr in zip(smiles_col, abbr_col):
        if isinstance(s, str) and s.strip():
            mol = Chem.MolFromSmiles(s.strip())
            if mol:
                can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
                highlights.append((can, abbr))
    return highlights

def load_and_canonicalize_data(filepath):
    """Loads"""
    with open(filepath, 'r', encoding='utf-16') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    
    valid_smiles = []
    mols = []
    
    print(f"Original SMILES count: {len(smiles_list)}")
    
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            can_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
            mol_can = Chem.MolFromSmiles(can_smi)
            mols.append(mol_can)
            valid_smiles.append(can_smi)
            
    print(f"Valid canonical SMILES count: {len(mols)}")
    return mols, valid_smiles

def categorize_molecule(mol):
    """
    Classifies molecule into:
    - Linear Carbonyl (Yellow)
    - Cyclic Carbonyl (Red)
    - Linear Ether (Green)
    - Cyclic Ether (Blue)
    - Other (Gray - fallback)
    """
    if mol is None:
        return None, None

    is_cyclic = mol.GetRingInfo().NumRings() > 0
    
    # Patterns
    carbonyl_pattern = Chem.MolFromSmarts('[CX3]=[OX1]')
    ether_pattern = Chem.MolFromSmarts('[OD2]([#6])[#6]')
    
    has_carbonyl = mol.HasSubstructMatch(carbonyl_pattern)
    has_ether = mol.HasSubstructMatch(ether_pattern)
    
    category = "Other"
    color = "gray"

    if has_carbonyl:
        if is_cyclic:
            category = "Cyclic Carbonyl"
            color = "red"
        else:
            category = "Linear Carbonyl"
            color = "yellow"
    elif has_ether:
        if is_cyclic:
            category = "Cyclic Ether"
            color = "blue"
        else:
            category = "Linear Ether"
            color = "green"
            
    return category, color

def get_fingerprints(mols):
    """ECFP4 fingerprints"""
    fps = []
    
    if HAS_DEEPCHEM:
        try:
            featurizer = dc.feat.CircularFingerprint(size=2048, radius=2)
            features = featurizer.featurize(mols)
            return features
        except Exception as e:
            print(f"DeepChem featurization failed: {e}. Fallback to RDKit.")

    try:
        from rdkit.Chem import rdFingerprintGenerator
        mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        for mol in mols:
            if mol:
                fp = mfgen.GetFingerprintAsNumPy(mol)
                fps.append(fp)
            else:
                fps.append(np.zeros(2048))
    except ImportError:
        for mol in mols:
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fps.append(np.array(fp))
            else:
                fps.append(np.zeros(2048))
    return np.array(fps)

def main():
    input_file = r'e:\python\pythonProject3\solvent analyze\1423.txt'
    output_image = r'e:\python\pythonProject3\solvent analyze\tsne_result_optimized.png'
    
    print(f"Loading data from {input_file}...")
    mols, valid_smiles = load_and_canonicalize_data(input_file)
    
    if not mols:
        print("No valid molecules found.")
        return

    categories = []
    colors = []
    
    print("Classifying molecules...")
    for mol in mols:
        cat, col = categorize_molecule(mol)
        categories.append(cat)
        colors.append(col)

    cat_counts = pd.Series(categories).value_counts()
    print("\nCategory Distribution:")
    print(cat_counts)
    print("-" * 30)

    print("Generating fingerprints...")
    X = get_fingerprints(mols)
    
    print("Running t-SNE with PCA initialization and Jaccard metric...")
    
    try:
        tsne = TSNE(n_components=2, verbose=1, perplexity=8, init='pca', metric='jaccard', random_state=0, learning_rate='auto')
        X_embedded = tsne.fit_transform(X)
    except Exception as e:
        print(f"t-SNE with Jaccard failed ({e}). Falling back to euclidean.")
        tsne = TSNE(n_components=2, verbose=1, perplexity=8, init='pca', metric='euclidean', random_state=0, learning_rate='auto')
        X_embedded = tsne.fit_transform(X)

    X_embedded[:, 0] = -1 * X_embedded[:, 0]
    
    print("Plotting results...")
    plt.figure(figsize=(12, 10))
    
    df = pd.DataFrame({
        'x': X_embedded[:, 0],
        'y': X_embedded[:, 1],
        'Category': categories,
        'Color': colors,
        'SMILES': valid_smiles
    })

    cat_map = {
        "Linear Carbonyl": "yellow",
        "Cyclic Carbonyl": "red",
        "Linear Ether": "green",
        "Cyclic Ether": "blue",
        "Other": "gray"
    }
    
    for cat, color in cat_map.items():
        subset = df[df['Category'] == cat]
        if not subset.empty:
            plt.scatter(subset['x'], subset['y'], c=color, label=f"{cat} (n={len(subset)})", alpha=0.7, edgecolors='k', linewidth=0.3, s=40)
    
    plt.title('t-SNE Visualization of 1423 Molecules (Optimized)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_image}")
    plt.close()

    highlight_csv = r'e:\python\pythonProject3\chemical_smiles.csv'
    highlight_output_image = r'e:\python\pythonProject3\solvent analyze\tsne_result_highlighted.png'
    
    if os.path.exists(highlight_csv):
        print(f"Generating highlight plot using {highlight_csv}...")
        highlights = load_highlight_smiles(highlight_csv)
        highlight_dict = {h[0]: h[1] for h in highlights}
        
        plt.figure(figsize=(12, 10))
        
        # Plot all points first as background (grey or faded)
        plt.scatter(df['x'], df['y'], c='lightgrey', alpha=0.5, s=20, label='Background')

        
        highlight_mask = df['SMILES'].isin(highlight_dict.keys())
        highlight_df = df[highlight_mask]
        
        if not highlight_df.empty:
            plt.scatter(highlight_df['x'], highlight_df['y'], c='red', s=60, edgecolors='k', label='Target Solvents', zorder=5)
            
            # Add labels
            for idx, row in highlight_df.iterrows():
                label = highlight_dict.get(row['SMILES'], '')
                plt.text(row['x'], row['y'], label, fontsize=8, ha='right', va='bottom')
                
            print(f"Highlighted {len(highlight_df)} molecules.")
        else:
            print("No matching molecules found to highlight.")

        plt.title('t-SNE Visualization (Highlighted Solvents)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.savefig(highlight_output_image, dpi=300, bbox_inches='tight')
        print(f"Highlight plot saved to {highlight_output_image}")
        plt.close()
    else:
        print(f"Highlight file not found: {highlight_csv}")


    output_csv = r'e:\python\pythonProject3\solvent analyze\1423_tsne_features.csv'

    export_df = pd.DataFrame({
        'SMILES': valid_smiles,
        'tsne_x': X_embedded[:, 0],
        'tsne_y': X_embedded[:, 1],
        'Category': categories
    })
    

    target_cats = ['Linear Carbonyl', 'Cyclic Carbonyl', 'Linear Ether', 'Cyclic Ether']
    for cat in target_cats:
        export_df[cat] = (export_df['Category'] == cat).astype(int)

    final_cols = ['SMILES'] + target_cats + ['tsne_x', 'tsne_y']
    export_df = export_df[final_cols]
    
    export_df.to_csv(output_csv, index=False)
    print(f"Features exported to {output_csv}")

if __name__ == "__main__":
    main()
