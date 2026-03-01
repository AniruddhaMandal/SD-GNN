"""
SubGNN Dataset Loader

Loads datasets from the SubGNN paper (Alsentzer et al., NeurIPS 2020):
- PPI-BP: Protein-protein interaction - Biological process classification
- HPO-METAB: Human Phenotype Ontology - Metabolic disorder classification
- HPO-NEURO: Human Phenotype Ontology - Neurological disorder classification
- EM-USER: Endomondo user - Gender classification

Dataset format:
- edge_list.txt: Base graph edges (node1 node2 {})
- subgraphs.pth: Subgraph definitions (node1-node2-...\tlabel\tsplit)
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from typing import Dict, List, Tuple
import networkx as nx


class SubGNNDataset(InMemoryDataset):
    """
    Dataset loader for SubGNN format datasets.

    Each sample is a subgraph from a larger base graph. The task is to classify
    the subgraph based on its structure and context.
    """

    def __init__(self, root, name, transform=None, pre_transform=None):
        """
        Args:
            root: Root directory where datasets are stored
            name: Dataset name ('ppi_bp', 'hpo_metab', 'hpo_neuro', 'em_user')
            transform: Optional transform to apply to each graph
            pre_transform: Optional transform to apply before saving
        """
        self.name = name.lower().replace('-', '_')
        self.label_names = {}
        self._num_classes = 0
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name)

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['edge_list.txt', 'subgraphs.pth']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Datasets should already be downloaded from Dropbox
        pass

    def process(self):
        """
        Process SubGNN dataset format into PyTorch Geometric Data objects.
        """
        # Load base graph
        base_graph = self._load_base_graph()

        # Load subgraphs
        subgraphs_data = self._load_subgraphs()

        # Create PyG Data objects
        data_list = []

        for subgraph_nodes, label, split in subgraphs_data:
            # Create subgraph
            subgraph = base_graph.subgraph(subgraph_nodes).copy()

            # Create node mapping (global to local)
            node_mapping = {node: i for i, node in enumerate(subgraph_nodes)}

            # Extract edges
            edge_index = []
            for u, v in subgraph.edges():
                edge_index.append([node_mapping[u], node_mapping[v]])
                edge_index.append([node_mapping[v], node_mapping[u]])  # Undirected

            if len(edge_index) == 0:
                # No edges - isolated nodes
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            # Node features (use one-hot degree encoding or all-ones)
            num_nodes = len(subgraph_nodes)
            degrees = [subgraph.degree(node) for node in subgraph_nodes]
            max_degree = max(degrees) if degrees else 0

            # Use all-ones features (to avoid information leakage from degree)
            x = torch.ones((num_nodes, 1), dtype=torch.float)

            # Create PyG Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                y=label,
                num_nodes=num_nodes,
                split=split  # Store split information
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _load_base_graph(self) -> nx.Graph:
        """Load the base graph from edge_list.txt"""
        G = nx.Graph()
        edge_file = os.path.join(self.raw_dir, 'edge_list.txt')

        with open(edge_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    u, v = int(parts[0]), int(parts[1])
                    G.add_edge(u, v)

        return G

    def _load_subgraphs(self) -> List[Tuple[List[int], int, str]]:
        """
        Load subgraph definitions from subgraphs.pth

        Returns:
            List of (nodes, label_idx, split) tuples
        """
        subgraph_file = os.path.join(self.raw_dir, 'subgraphs.pth')

        subgraphs_data = []
        label_to_idx = {}
        label_counter = 0

        with open(subgraph_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue

                nodes_str, label_str, split = parts

                # Parse nodes
                nodes = [int(n) for n in nodes_str.split('-')]

                # Convert label to index
                if label_str not in label_to_idx:
                    label_to_idx[label_str] = label_counter
                    label_counter += 1

                label_idx = label_to_idx[label_str]

                subgraphs_data.append((nodes, label_idx, split))

        # Store label mapping as instance attribute
        self.label_names = {idx: name for name, idx in label_to_idx.items()}
        self._num_classes = len(label_to_idx)

        print(f"Loaded {len(subgraphs_data)} subgraphs with {self._num_classes} classes")
        print(f"Label mapping: {self.label_names}")

        return subgraphs_data

    @property
    def num_classes(self):
        """Return the number of classes"""
        return self._num_classes

    def get_idx_split(self) -> Dict[str, np.ndarray]:
        """
        Get train/val/test split indices.

        Returns:
            Dictionary with keys 'train', 'valid', 'test' containing indices
        """
        train_idx = []
        val_idx = []
        test_idx = []

        for i, data in enumerate(self):
            if data.split == 'train':
                train_idx.append(i)
            elif data.split == 'val':
                val_idx.append(i)
            elif data.split == 'test':
                test_idx.append(i)

        return {
            'train': np.array(train_idx),
            'valid': np.array(val_idx),
            'test': np.array(test_idx)
        }


# Dataset metadata
SUBGNN_DATASETS = {
    'ppi_bp': {
        'name': 'PPI-BP',
        'task': 'Multi-Class-Classification',
        'num_classes': 6,
        'description': 'Protein-protein interaction - Biological process classification',
        'classes': ['metabolism', 'development', 'signal_transduction', 'stress/death', 'cell_organization', 'transport']
    },
    'hpo_metab': {
        'name': 'HPO-METAB',
        'task': 'Multi-Class-Classification',
        'num_classes': 6,
        'description': 'Human Phenotype Ontology - Metabolic disorder classification',
        'classes': ['Lysosomal', 'Energy', 'Amino_Acid', 'Carbohydrate', 'Lipid', 'Glycosylation']
    },
    'hpo_neuro': {
        'name': 'HPO-NEURO',
        'task': 'Multi-Label-Classification',
        'num_classes': 10,
        'description': 'Human Phenotype Ontology - Neurological disorder classification',
        'classes': ['Neurodegenerative', 'Epilepsy', 'Ataxia', 'Genetic_Dementia', 'Intellectual', 'Movement', 'Myopathy', 'Neuropathy', 'Vision', 'Other']
    },
    'em_user': {
        'name': 'EM-USER',
        'task': 'Binary-Classification',
        'num_classes': 2,
        'description': 'Endomondo user - Gender classification',
        'classes': ['male', 'female']
    }
}
