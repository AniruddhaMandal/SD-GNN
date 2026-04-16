"""
gxl Dataset loaders

This package contains all dataset implementations, organized by category.
Importing this package triggers all @register_dataset decorators.
"""

from . import synthetic           # registers all synthetic datasets
from . import graph_classification # registers all graph classification datasets
from . import node_classification  # registers all node classification datasets

# Keep MolHIVDataset accessible at the top level for backward compatibility
from .graph_classification.molhiv import MolHIVDataset

__all__ = ['MolHIVDataset']
