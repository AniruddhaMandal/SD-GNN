"""
Graph classification dataset loaders.
Importing this package triggers all @register_dataset decorators for graph classification datasets.
"""
from . import lrgb        # registers 'PascalVOC-SP', 'COCO-SP', 'PCQM-Contact', 'Peptides-func', 'Peptides-struct'
from . import tu          # registers 'MUTAG', 'ENZYMES', 'PROTEINS', 'COLLAB', 'IMDB-BINARY',
                          #            'REDDIT-BINARY', 'PTC_MR', 'AIDS'
from . import zinc_qm9    # registers 'ZINC', 'QM9'
from . import subgnn      # registers 'PPI-BP', 'HPO-METAB', 'HPO-NEURO', 'EM-USER'
from . import molhiv      # registers 'ogbg-molhiv', 'molhiv'
from . import molecules   # registers 'ogbg-ppa', 'BBBP', 'Tox21'
