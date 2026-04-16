"""
Synthetic graph dataset loaders.
Importing this package triggers all @register_dataset decorators for synthetic datasets.
"""
from . import csl        # registers 'CSL'
from . import sr_exp     # registers 'SR', 'EXP'
from . import variants   # registers 'K4', 'Triangle-Parity', 'Clique-Detection',
                         #            'Multi-Clique-Detection', 'Clique-Detection-Controlled',
                         #            'Sparse-Clique-Detection'
