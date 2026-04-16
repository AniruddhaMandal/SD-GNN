"""
Node classification dataset loaders.
Importing this package triggers all @register_dataset decorators for node classification datasets.
"""
from . import planetoid        # registers 'Cora', 'CiteSeer', 'PubMed'
from . import amazon_coauthor  # registers 'AmazonPhoto', 'AmazonComputers', 'CoauthorCS', 'CoauthorPhysics'
from . import heterophilic     # registers 'Chameleon', 'Squirrel', 'Actor', 'Cornell', 'Texas', 'Wisconsin',
                               #            'RomanEmpire', 'AmazonRatings', 'Minesweeper', 'Tolokers', 'Questions'
from . import linkx            # registers 'ArxivYear', 'SnapPatents', 'Penn94', 'Pokec', 'TwitchGamers', 'Genius'
from . import ogb_node         # registers 'ogbn-arxiv', 'ogbn-proteins'
