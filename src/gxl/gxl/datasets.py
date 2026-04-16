"""
Dataset registry — importing this module triggers all @register_dataset decorators.
All implementations live in dataset_loaders/.
"""
from .dataset_loaders import *  # noqa: F401, F403
