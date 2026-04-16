"""
Base dataset utilities shared across dataset loaders.
"""


class _ListDataset:
    """Minimal dataset wrapper around a list of PyG Data objects."""
    def __init__(self, data_list, transform=None):
        self._data      = data_list
        self._transform = transform

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        g = self._data[idx]
        if self._transform is not None:
            g = self._transform(g)
        return g
