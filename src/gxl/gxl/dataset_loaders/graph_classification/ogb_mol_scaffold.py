"""
OGB Molecular Datasets with Scaffold Splitting (molbbbp, molbace, moltox21)

Loads OGB-style molecular datasets without depending on the `ogb` Python module.
Uses RDKit for SMILES-to-graph conversion with OGB-style featurization.
Provides scaffold splitting (80/10/10) matching the OGB implementation.

Dataset sources (SNAP):
  https://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/bbbp.zip
  https://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/bace.zip
  https://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/tox21.zip
"""

import os
import csv
import gzip
import zipfile
import urllib.request
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from .molhiv import smiles_to_graph, scaffold_split


# ---- Generic OGBMolScaffoldDataset ----

class OGBMolScaffoldDataset(InMemoryDataset):
    """
    Generic OGB-style molecular dataset with Murcko scaffold split.

    Handles single-task binary classification (bbbp, bace) and
    multi-task binary classification with NaN labels (tox21).

    Args:
        root: Root directory for storing raw/processed data
        url: Download URL for the zip file
        zip_name: Name of the zip file (e.g. 'bbbp.zip')
        csv_subpath: Path inside the zip to the CSV (may be gzipped)
        smiles_col: Name of the SMILES column in the CSV
        label_cols: List of label column names
        multi_label: If True, stack all label columns; labels may contain NaN
        transform: PyG transform
        pre_transform: PyG pre-transform
        pre_filter: PyG pre-filter
    """

    def __init__(
        self,
        root: str,
        url: str,
        zip_name: str,
        csv_subpath: str,
        smiles_col: str,
        label_cols: List[str],
        multi_label: bool = False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self._url = url
        self._zip_name = zip_name
        self._csv_subpath = csv_subpath
        self._smiles_col = smiles_col
        self._label_cols = label_cols
        self._multi_label = multi_label
        self._split = None

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self._split = torch.load(self.processed_paths[1], weights_only=False)

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        return [self._csv_subpath]

    @property
    def processed_file_names(self):
        return ['data.pt', 'split.pt']

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        zip_path = os.path.join(self.raw_dir, self._zip_name)

        print(f"Downloading {self._zip_name} from {self._url}...")
        urllib.request.urlretrieve(self._url, zip_path)

        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(self.raw_dir)

        os.remove(zip_path)
        print("Download complete.")

    def _open_csv(self, path: str):
        """Open a CSV file, handling optional gzip compression."""
        if path.endswith('.gz'):
            return gzip.open(path, 'rt', encoding='utf-8')
        return open(path, 'r', encoding='utf-8')

    def process(self):
        csv_path = self.raw_paths[0]
        print(f"Processing SMILES from {csv_path}...")

        smiles_list = []
        raw_labels = []

        with self._open_csv(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                smiles_list.append(row[self._smiles_col])
                if self._multi_label:
                    row_labels = []
                    for col in self._label_cols:
                        val = row[col].strip()
                        row_labels.append(float('nan') if val == '' else float(val))
                    raw_labels.append(row_labels)
                else:
                    val = row[self._label_cols[0]].strip()
                    raw_labels.append(int(float(val)))

        print(f"Found {len(smiles_list)} molecules.")

        data_list = []
        valid_indices = []

        for i, (smi, label) in enumerate(zip(smiles_list, raw_labels)):
            if i % 2000 == 0:
                print(f"Processing molecule {i}/{len(smiles_list)}...")

            graph = smiles_to_graph(smi)
            if graph is None:
                continue

            if self._multi_label:
                # [num_tasks] float tensor; NaN for missing assays
                y = torch.tensor(label, dtype=torch.float).unsqueeze(0)  # [1, T]
            else:
                y = torch.tensor([label], dtype=torch.long)

            graph.y = y

            if self.pre_filter is not None and not self.pre_filter(graph):
                continue
            if self.pre_transform is not None:
                graph = self.pre_transform(graph)

            data_list.append(graph)
            valid_indices.append(i)

        print(f"Successfully processed {len(data_list)} molecules.")

        valid_smiles = [smiles_list[i] for i in valid_indices]
        split = scaffold_split(valid_smiles)

        print(f"Split sizes - Train: {len(split['train'])}, "
              f"Valid: {len(split['valid'])}, Test: {len(split['test'])}")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        torch.save(split, self.processed_paths[1])

        self._split = split

    def get_idx_split(self) -> Dict[str, np.ndarray]:
        if self._split is None:
            self._split = torch.load(self.processed_paths[1], weights_only=False)
        return {
            'train': np.array(self._split['train']),
            'valid': np.array(self._split['valid']),
            'test':  np.array(self._split['test']),
        }

    @property
    def num_node_features(self):
        return 9

    @property
    def num_edge_features(self):
        return 3


# ---- Dataset registration ----

from ...registry import register_dataset
from ... import ExperimentConfig
from ...utils.split_and_loader import build_dataloaders_from_dataset


def _build_ogb_mol(cfg: ExperimentConfig, dataset: OGBMolScaffoldDataset):
    """Shared builder: apply OGB encoders and build dataloaders."""
    from ...encoder import OGBAtomEncoder, OGBBondEncoder
    from torch_geometric.transforms import Compose

    emb_dim = cfg.model_config.node_feature_dim
    edge_emb_dim = getattr(cfg.model_config, 'edge_feature_dim', None) or emb_dim

    # Re-instantiate with transforms applied
    dataset.transform = Compose([
        OGBAtomEncoder(emb_dim=emb_dim),
        OGBBondEncoder(emb_dim=edge_emb_dim),
    ])

    return build_dataloaders_from_dataset(dataset, cfg)


@register_dataset('ogbg-molbbbp')
@register_dataset('molbbbp')
def build_molbbbp(cfg: ExperimentConfig):
    """
    Build OGB-MolBBBP dataset (Blood-Brain Barrier Penetration).

    Task: Binary classification
    Size: ~2039 molecules
    Metric: ROC-AUC
    Split: Scaffold (80/10/10)
    """
    from ...encoder import OGBAtomEncoder, OGBBondEncoder
    from torch_geometric.transforms import Compose

    emb_dim = cfg.model_config.node_feature_dim
    edge_emb_dim = getattr(cfg.model_config, 'edge_feature_dim', None) or emb_dim

    dataset = OGBMolScaffoldDataset(
        root='./data/OGB/molbbbp',
        url='https://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/bbbp.zip',
        zip_name='bbbp.zip',
        csv_subpath='bbbp/mapping/mol.csv.gz',
        smiles_col='smiles',
        label_cols=['p_np'],
        multi_label=False,
        transform=Compose([
            OGBAtomEncoder(emb_dim=emb_dim),
            OGBBondEncoder(emb_dim=edge_emb_dim),
        ]),
    )

    return build_dataloaders_from_dataset(dataset, cfg)


@register_dataset('ogbg-molbace')
@register_dataset('molbace')
def build_molbace(cfg: ExperimentConfig):
    """
    Build OGB-MolBACE dataset (Beta-Secretase Inhibition).

    Task: Binary classification
    Size: ~1513 molecules
    Metric: ROC-AUC
    Split: Scaffold (80/10/10)
    """
    from ...encoder import OGBAtomEncoder, OGBBondEncoder
    from torch_geometric.transforms import Compose

    emb_dim = cfg.model_config.node_feature_dim
    edge_emb_dim = getattr(cfg.model_config, 'edge_feature_dim', None) or emb_dim

    dataset = OGBMolScaffoldDataset(
        root='./data/OGB/molbace',
        url='https://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/bace.zip',
        zip_name='bace.zip',
        csv_subpath='bace/mapping/mol.csv.gz',
        smiles_col='smiles',
        label_cols=['Class'],
        multi_label=False,
        transform=Compose([
            OGBAtomEncoder(emb_dim=emb_dim),
            OGBBondEncoder(emb_dim=edge_emb_dim),
        ]),
    )

    return build_dataloaders_from_dataset(dataset, cfg)


@register_dataset('ogbg-moltox21')
@register_dataset('moltox21')
def build_moltox21(cfg: ExperimentConfig):
    """
    Build OGB-MolTox21 dataset (Toxicology in the 21st Century).

    Task: Multi-label binary classification (12 toxicity assays)
    Size: ~7831 molecules
    Metric: ROC-AUC (mean over tasks, skipping tasks with single-class labels)
    Split: Scaffold (80/10/10)
    Labels may contain NaN for missing assay measurements.
    """
    from ...encoder import OGBAtomEncoder, OGBBondEncoder
    from torch_geometric.transforms import Compose

    emb_dim = cfg.model_config.node_feature_dim
    edge_emb_dim = getattr(cfg.model_config, 'edge_feature_dim', None) or emb_dim

    tox21_tasks = [
        'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
        'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
        'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53',
    ]

    dataset = OGBMolScaffoldDataset(
        root='./data/OGB/moltox21',
        url='https://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/tox21.zip',
        zip_name='tox21.zip',
        csv_subpath='tox21/mapping/mol.csv.gz',
        smiles_col='smiles',
        label_cols=tox21_tasks,
        multi_label=True,
        transform=Compose([
            OGBAtomEncoder(emb_dim=emb_dim),
            OGBBondEncoder(emb_dim=edge_emb_dim),
        ]),
    )

    return build_dataloaders_from_dataset(dataset, cfg)
