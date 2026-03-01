# utils/split_and_loader.py
from typing import Dict
import random
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from collections.abc import Mapping, Sequence
import logging

from .. import ExperimentConfig

logger = logging.getLogger(__name__)


# ---------- split logic ----------
def build_or_load_splits(dataset, cfg: ExperimentConfig) -> Dict[str, list]:
    """
    Returns dict {'train','val','test'} of index lists.
    Order:
      1) OGB-style get_idx_split()
      2) deterministic stratified split
    """
    seed = int(getattr(cfg, "seed", 0))

    # 1) OGB-style
    if hasattr(dataset, "get_idx_split"):
        ogb = dataset.get_idx_split()
        # normalize to {'train','val','test'}
        val_key = "valid" if "valid" in ogb else "val"
        return {
            "train": [int(i) for i in ogb["train"]],
            "val":   [int(i) for i in ogb[val_key]],
            "test":  [int(i) for i in ogb["test"]],
        }

    # 2) deterministic stratified split
    n = len(dataset)
    idx = list(range(n))

    # labels for optional stratification
    strat = None
    try:
        labels = []
        for d in dataset:
            y = getattr(d, "y", None)
            if y is None:
                labels.append(None)
            else:
                a = np.asarray(y)
                labels.append(int(a.argmax()) if a.ndim > 0 else int(a))
        if any(l is not None for l in labels):
            strat = np.array([(-1 if l is None else l) for l in labels])
            if np.all(strat == -1):
                strat = None
    except Exception:
        strat = None

    # ratios
    train_ratio = float(cfg.train.train_ratio)
    val_ratio   = float(cfg.train.val_ratio)
    test_ratio  = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    # split
    if strat is not None:
        train_idx, rest = train_test_split(
            idx, train_size=train_ratio, stratify=strat, random_state=seed
        )
        if rest:
            val_frac = val_ratio / (1.0 - train_ratio)
            rest_strat = strat[rest]
            val_idx, test_idx = train_test_split(
                rest, train_size=val_frac, stratify=rest_strat, random_state=seed
            )
        else:
            val_idx, test_idx = [], []
    else:
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        train_idx = idx[:n_train]
        val_idx   = idx[n_train:n_train + n_val]
        test_idx  = idx[n_train + n_val:]

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    return splits


# ---------- loader construction ----------
def _is_split_container(obj) -> bool:
    # dict-like with required keys
    if isinstance(obj, Mapping):
        return all(k in obj for k in ("train", "val", "test"))
    # tuple/list of length 3
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)) and len(obj) == 3:
        return True
    # object with attributes
    return all(hasattr(obj, a) for a in ("train", "val", "test"))

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_dataloaders_from_dataset(dataset_or_splits, cfg: ExperimentConfig, collate_fn=None, pin_memory=False):
    """
    Accepts:
      - a single Dataset (will create/cache splits), OR
      - a split-container: dict('train','val','test') / (train_ds, val_ds, test_ds) / attrs .train/.val/.test
      - an object implementing get_idx_split() (OGB-style)
    """
    # 1) user-provided splits/datasets
    if _is_split_container(dataset_or_splits):
        if isinstance(dataset_or_splits, Mapping):
            train_ds = dataset_or_splits["train"]
            val_ds   = dataset_or_splits["val"]
            test_ds  = dataset_or_splits["test"]
        elif isinstance(dataset_or_splits, Sequence):
            train_ds, val_ds, test_ds = dataset_or_splits
        else:
            train_ds, val_ds, test_ds = dataset_or_splits.train, dataset_or_splits.val, dataset_or_splits.test

    else:
        # 2) single dataset → indices (OGB or deterministic) → Subset
        dataset = dataset_or_splits
        if hasattr(dataset, "get_idx_split"):
            spl = dataset.get_idx_split()
            val_key = "valid" if "valid" in spl else "val"
            train_ds = torch.utils.data.Subset(dataset, list(spl["train"]))
            val_ds   = torch.utils.data.Subset(dataset, list(spl[val_key]))
            test_ds  = torch.utils.data.Subset(dataset, list(spl["test"]))
        else:
            spl = build_or_load_splits(dataset, cfg)
            train_ds = torch.utils.data.Subset(dataset, spl["train"])
            val_ds   = torch.utils.data.Subset(dataset, spl["val"])
            test_ds  = torch.utils.data.Subset(dataset, spl["test"])

    # normalize target if necessary
    try:
        normalize_targets = cfg.train.dataloader_kwargs.get("normalize_target", False)
    except (AttributeError, TypeError):
        normalize_targets = False

    if normalize_targets:
        logger.info("  → Normalizing targets...")
        from gxl.encoder import NormaliseTarget
        # Compute stats from training data
        normalizer_train = NormaliseTarget(train_ds)
        mean, std = normalizer_train.get_stats()

        # Apply same stats to all splits
        train_ds = normalizer_train  # Already computed
        val_ds = NormaliseTarget(val_ds, mean=mean, std=std)
        test_ds = NormaliseTarget(test_ds, mean=mean, std=std)

    # 3) DataLoaders
    nw = int(getattr(cfg, "num_workers", 0))
    g = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=int(cfg.train.train_batch_size), shuffle=True,
                              num_workers=nw, worker_init_fn=seed_worker, generator=g, collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=int(cfg.train.val_batch_size),   shuffle=True,
                              num_workers=max(nw // 2, 0), worker_init_fn=seed_worker, generator=g,collate_fn=collate_fn, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=int(cfg.train.val_batch_size),   shuffle=True,
                              num_workers=max(nw // 2, 0), worker_init_fn=seed_worker, generator=g, collate_fn=collate_fn, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader
