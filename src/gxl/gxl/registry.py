# registry.py
"""Simple, thread-safe registry for models, datasets, transforms, metrics, losses, etc.

Usage:
    - decorate functions/classes with @register_model("name")
    - retrieve with get_model("name"), or list registered names with list_models()

This registry stores callables/constructors. It does not instantiate for you.
"""
from typing import Callable, Dict, Iterable, Optional
import threading

_LOCK = threading.RLock()

class RegistryError(Exception):
    pass

class _Registry:
    def __init__(self):
        self._items: Dict[str, Callable] = {}

    def register(self, name: str, obj: Callable, overwrite: bool = False):
        name = str(name)
        with _LOCK:
            if name in self._items and not overwrite:
                raise RegistryError(f"'{name}' already registered")
            self._items[name] = obj

    def get(self, name: str) -> Callable:
        try:
            return self._items[name]
        except KeyError:
            raise RegistryError(f"'{name}' not found in registry. Available: {list(self._items.keys())}")

    def contains(self, name: str) -> bool:
        return name in self._items

    def list(self) -> Iterable[str]:
        return list(self._items.keys())

    def unregister(self, name: str):
        with _LOCK:
            if name in self._items:
                del self._items[name]
            else:
                raise RegistryError(f"'{name}' not found, cannot unregister")

# Top-level registries
MODEL_REG = _Registry()
DATASET_REG = _Registry()
TRANSFORM_REG = _Registry()
METRIC_REG = _Registry()
LOSS_REG = _Registry()
AGGREGATOR_REG = _Registry()

# Decorators for each registry
def register_model(name: str, overwrite: bool = False):
    def dec(obj: Callable):
        MODEL_REG.register(name, obj, overwrite=overwrite)
        return obj
    return dec

def register_dataset(name: str, overwrite: bool = False):
    def dec(obj: Callable):
        DATASET_REG.register(name, obj, overwrite=overwrite)
        return obj
    return dec

def register_transform(name: str, overwrite: bool = False):
    def dec(obj: Callable):
        TRANSFORM_REG.register(name, obj, overwrite=overwrite)
        return obj
    return dec

def register_metric(name: str, overwrite: bool = False):
    def dec(obj: Callable):
        METRIC_REG.register(name, obj, overwrite=overwrite)
        return obj
    return dec

def register_loss(name: str, overwrite: bool = False):
    def dec(obj: Callable):
        LOSS_REG.register(name, obj, overwrite=overwrite)
        return obj
    return dec

def register_aggregator(name: str, overwrite: bool = False):
    def dec(obj: Callable):
        AGGREGATOR_REG.register(name, obj, overwrite=overwrite)
        return obj
    return dec

# Getter helpers
def get_model(name: str) -> Callable:
    return MODEL_REG.get(name)

def get_dataset(name: str) -> Callable:
    return DATASET_REG.get(name)

def get_transform(name: str) -> Callable:
    return TRANSFORM_REG.get(name)

def get_metric(name: str) -> Callable:
    return METRIC_REG.get(name)

def get_loss(name: str) -> Callable:
    return LOSS_REG.get(name)

def get_aggregator(name: str) -> Callable:
    return AGGREGATOR_REG.get(name)

# Listing helpers
def list_models() -> Iterable[str]:
    return MODEL_REG.list()

def list_datasets() -> Iterable[str]:
    return DATASET_REG.list()

def list_transforms() -> Iterable[str]:
    return TRANSFORM_REG.list()

def list_metrics() -> Iterable[str]:
    return METRIC_REG.list()

def list_losses() -> Iterable[str]:
    return LOSS_REG.list()
