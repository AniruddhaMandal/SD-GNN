"""
MolHIV Dataset Loader

Loads ogbg-molhiv dataset without depending on the `ogb` Python module.
Uses RDKit for SMILES-to-graph conversion with OGB-style featurization.
Provides scaffold splitting (80/10/10).

Dataset source: https://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/hiv.zip
"""

import os
import csv
import zipfile
import urllib.request
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    raise ImportError(
        "RDKit is required for MolHIV dataset. Install with: pip install rdkit"
    )


# Mapping dictionaries for categorical features
CHIRALITY_MAP = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    Chem.rdchem.ChiralType.CHI_OTHER: 3,
}

HYBRIDIZATION_MAP = {
    Chem.rdchem.HybridizationType.SP: 0,
    Chem.rdchem.HybridizationType.SP2: 1,
    Chem.rdchem.HybridizationType.SP3: 2,
    Chem.rdchem.HybridizationType.SP3D: 3,
    Chem.rdchem.HybridizationType.SP3D2: 4,
}

BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}

STEREO_MAP = {
    Chem.rdchem.BondStereo.STEREONONE: 0,
    Chem.rdchem.BondStereo.STEREOZ: 1,
    Chem.rdchem.BondStereo.STEREOE: 2,
    Chem.rdchem.BondStereo.STEREOCIS: 3,
    Chem.rdchem.BondStereo.STEREOTRANS: 4,
    Chem.rdchem.BondStereo.STEREOANY: 5,
}


def smiles_to_graph(smiles: str) -> Optional[Data]:
    """
    Convert SMILES string to PyG Data object with OGB-style features.

    Atom features (9 categorical):
        0: atomic_num (119 values: 1-118 + unknown)
        1: chirality (4 values)
        2: degree (11 values: 0-10)
        3: formal_charge (11 values: -5 to +5)
        4: num_hs (9 values: 0-8)
        5: num_radical_e (5 values: 0-4)
        6: hybridization (5 values)
        7: is_aromatic (2 values)
        8: is_in_ring (2 values)

    Bond features (3 categorical):
        0: bond_type (5 values: SINGLE, DOUBLE, TRIPLE, AROMATIC, misc)
        1: stereo (6 values)
        2: is_conjugated (2 values)

    Returns:
        Data object or None if SMILES is invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Extract atom features
    atom_features = []
    for atom in mol.GetAtoms():
        # Atomic number: 1-118, with 0 for unknown
        atomic_num = atom.GetAtomicNum()
        if atomic_num > 118:
            atomic_num = 0  # unknown

        # Chirality
        chirality = CHIRALITY_MAP.get(atom.GetChiralTag(), 3)  # 3 = CHI_OTHER

        # Degree (capped at 10)
        degree = min(atom.GetTotalDegree(), 10)

        # Formal charge (-5 to +5, mapped to 0-10)
        charge = atom.GetFormalCharge()
        charge_idx = max(-5, min(5, charge)) + 5  # Map to 0-10

        # Number of hydrogens (capped at 8)
        num_hs = min(atom.GetTotalNumHs(), 8)

        # Number of radical electrons (capped at 4)
        num_radical = min(atom.GetNumRadicalElectrons(), 4)

        # Hybridization
        hybridization = HYBRIDIZATION_MAP.get(atom.GetHybridization(), 4)  # 4 = SP3D2 (misc)

        # Is aromatic
        is_aromatic = int(atom.GetIsAromatic())

        # Is in ring
        is_in_ring = int(atom.IsInRing())

        atom_features.append([
            atomic_num,
            chirality,
            degree,
            charge_idx,
            num_hs,
            num_radical,
            hybridization,
            is_aromatic,
            is_in_ring,
        ])

    # Extract bond features and edge indices
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Bond type
        bond_type = BOND_TYPE_MAP.get(bond.GetBondType(), 4)  # 4 = misc

        # Stereo
        stereo = STEREO_MAP.get(bond.GetStereo(), 0)

        # Is conjugated
        is_conjugated = int(bond.GetIsConjugated())

        bond_feat = [bond_type, stereo, is_conjugated]

        # Add both directions (undirected graph)
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bond_feat)
        edge_attr.append(bond_feat)

    # Convert to tensors
    x = torch.tensor(atom_features, dtype=torch.long)

    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def scaffold_split(
    smiles_list: List[str],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """
    Murcko scaffold split matching OGB implementation.

    Groups molecules by their Murcko scaffold, then assigns scaffold groups
    to train/valid/test to ensure molecules with similar scaffolds are in
    the same split.

    Args:
        smiles_list: List of SMILES strings
        frac_train: Fraction for training set (default 0.8)
        frac_valid: Fraction for validation set (default 0.1)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with 'train', 'valid', 'test' keys containing index lists
    """
    n = len(smiles_list)

    # Group molecules by scaffold
    scaffold_to_indices = defaultdict(list)

    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=False
                )
            else:
                scaffold = ""
        except Exception:
            scaffold = ""

        scaffold_to_indices[scaffold].append(i)

    # Sort scaffolds by size (largest first), then by first index for determinism
    scaffolds = list(scaffold_to_indices.values())
    scaffolds.sort(key=lambda x: (len(x), x[0]), reverse=True)

    # Assign scaffolds to splits
    train_idx = []
    valid_idx = []
    test_idx = []

    for group in scaffolds:
        if len(train_idx) / n < frac_train:
            train_idx.extend(group)
        elif len(valid_idx) / n < frac_valid:
            valid_idx.extend(group)
        else:
            test_idx.extend(group)

    # Sort indices within each split for reproducibility
    train_idx.sort()
    valid_idx.sort()
    test_idx.sort()

    return {
        'train': train_idx,
        'valid': valid_idx,
        'test': test_idx,
    }


class MolHIVDataset(InMemoryDataset):
    """
    OGB-MolHIV dataset without ogb dependency.

    - Downloads hiv.zip from SNAP
    - Converts SMILES to graphs using RDKit
    - Extracts 9 atom + 3 bond categorical features
    - Provides scaffold split (80/10/10)

    Task: Binary classification (HIV activity prediction)
    Metric: ROC-AUC
    """

    url = "https://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/hiv.zip"

    def __init__(
        self,
        root: str,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self._smiles_list = None
        self._split = None
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        # Load splits
        self._split = torch.load(self.processed_paths[1], weights_only=False)

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        return ['hiv/mapping/mol.csv.gz']

    @property
    def processed_file_names(self):
        return ['data.pt', 'split.pt']

    def download(self):
        """Download and extract hiv.zip from SNAP."""
        os.makedirs(self.raw_dir, exist_ok=True)
        zip_path = os.path.join(self.raw_dir, 'hiv.zip')

        print(f"Downloading HIV dataset from {self.url}...")
        urllib.request.urlretrieve(self.url, zip_path)

        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(self.raw_dir)

        # Clean up zip file
        os.remove(zip_path)
        print("Download complete.")

    def process(self):
        """Process SMILES to graphs with OGB-style features."""
        import gzip

        # CSV file path (matches raw_file_names)
        csv_path = self.raw_paths[0]
        print(f"Processing SMILES from {csv_path}...")

        smiles_list = []
        labels = []

        with gzip.open(csv_path, 'rt', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                smiles_list.append(row['smiles'])
                labels.append(int(row['HIV_active']))

        print(f"Found {len(smiles_list)} molecules.")

        # Convert SMILES to graphs
        data_list = []
        valid_indices = []

        for i, (smi, label) in enumerate(zip(smiles_list, labels)):
            if i % 5000 == 0:
                print(f"Processing molecule {i}/{len(smiles_list)}...")

            graph = smiles_to_graph(smi)
            if graph is not None:
                graph.y = torch.tensor([label], dtype=torch.long)
                graph.smiles = smi

                if self.pre_filter is not None and not self.pre_filter(graph):
                    continue

                if self.pre_transform is not None:
                    graph = self.pre_transform(graph)

                data_list.append(graph)
                valid_indices.append(i)

        print(f"Successfully processed {len(data_list)} molecules.")

        # Compute scaffold split using valid SMILES only
        valid_smiles = [smiles_list[i] for i in valid_indices]
        split = scaffold_split(valid_smiles)

        print(f"Split sizes - Train: {len(split['train'])}, "
              f"Valid: {len(split['valid'])}, Test: {len(split['test'])}")

        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        torch.save(split, self.processed_paths[1])

        self._smiles_list = valid_smiles
        self._split = split

    def get_idx_split(self) -> Dict[str, np.ndarray]:
        """
        Get train/valid/test split indices (scaffold-based).

        Returns:
            Dictionary with keys 'train', 'valid', 'test' containing numpy arrays of indices
        """
        if self._split is None:
            self._split = torch.load(self.processed_paths[1], weights_only=False)

        return {
            'train': np.array(self._split['train']),
            'valid': np.array(self._split['valid']),
            'test': np.array(self._split['test']),
        }

    @property
    def num_classes(self):
        """Number of classes for binary classification."""
        return 2

    @property
    def num_node_features(self):
        """Number of categorical node features."""
        return 9

    @property
    def num_edge_features(self):
        """Number of categorical edge features."""
        return 3
