import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem import AllChem


try:
    import deepchem as dc
    HAS_DEEPCHEM = True
except ImportError:
    HAS_DEEPCHEM = False
    print("DeepChem not found. Using RDKit for fingerprints.")

def load_data(filepath):
    """Loads SMILES from file."""
    with open(filepath, 'r', encoding='utf-16') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    return smiles_list

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
        featurizer = dc.feat.CircularFingerprint(size=2048, radius=2)
        features = featurizer.featurize(mols)
        return features
    else:
        # RDKit Fallback
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
            # Fallback for older RDKit if generator not found
            for mol in mols:
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    fps.append(np.array(fp))
                else:
                    fps.append(np.zeros(2048))
        return np.array(fps)

def main():
    input_file = r'e:\python\pythonProject3\solvent analyze\1399.txt'
    output_image = r'e:\python\pythonProject3\solvent analyze\tsne_result.png'
    
    print(f"Loading data from {input_file}...")
    raw_smiles = load_data(input_file)
    
    mols = []
    valid_smiles = []
    categories = []
    colors = []
    
    print("Classifying molecules...")
    for s in raw_smiles:
        s = s.strip()
        mol = Chem.MolFromSmiles(s)
        if mol:
            cat, col = categorize_molecule(mol)
            mols.append(mol)
            valid_smiles.append(s)
            categories.append(cat)
            colors.append(col)
        else:
            pass

    print(f"Valid molecules: {len(mols)} / {len(raw_smiles)}")
    
    if not mols:
        print("No valid molecules found. Check input file format.")
        return

    print("Generating fingerprints...")
    X = get_fingerprints(mols)
    
    print("Running t-SNE with PCA initialization...")
    tsne = TSNE(n_components=2, verbose=1, perplexity=8, init='pca', random_state=0, learning_rate='auto')
    X_embedded = tsne.fit_transform(X)

    X_embedded[:, 0] = -1 * X_embedded[:, 0]
    
    print("Plotting results...")
    plt.figure(figsize=(10, 8))

    df = pd.DataFrame({
        'x': X_embedded[:, 0],
        'y': X_embedded[:, 1],
        'Category': categories,
        'Color': colors
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
            plt.scatter(subset['x'], subset['y'], c=color, label=cat, alpha=0.6, edgecolors='w', s=30)
    
    plt.title('t-SNE Visualization of 1399 Molecules')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.savefig(output_image, dpi=300)
    print(f"Plot saved to {output_image}")
    plt.close()

if __name__ == "__main__":
    main()
