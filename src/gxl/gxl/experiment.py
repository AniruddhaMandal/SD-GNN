"""
Experiment runner template
==========================

A flexible Experiment class and ExperimentConfig dataclass you can drop into your project.

Purpose
-------
- Provide structure for running training/evaluation experiments for PyTorch models (including
  GNNs with PyTorch Geometric) with minimal changes.
- The class is intentionally generic: you supply small callables or override hooks to instantiate
  datasets / models / metrics. This file provides the scaffolding and logging/checkpointing.

How to use
----------
- Fill `ExperimentConfig` fields or construct one and pass callables in `model_fn`, `dataloader_fn`.
- `model_fn(cfg)` should return an nn.Module (uninitialized weights OK).
- `dataloader_fn(cfg)` should return a tuple (train_loader, val_loader, test_loader) or
  `(train_loader, val_loader)` (test optional).
- Optionally provide `criterion_fn`, `metric_fn` or override `evaluate`.

Notes
-----
- This template supports AMP (torch.cuda.amp), gradient clipping, LR scheduler hooks, checkpointing,
  TensorBoard logging, deterministic seeding, and simple CLI friendly prints.
- You can subclass `Experiment` and override hooks for complex dataset/model construction.

"""

from __future__ import annotations
import os, sys, warnings
from pathlib import Path
import time
import json
import logging
from typing import Any, Dict, Optional, Literal 
import random 
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from . import ExperimentConfig
from . import SubgraphFeaturesBatch
from dataclasses import asdict

# ------- VSCode Debug ------
def running_in_vscode_debug():
    trace = sys.gettrace()
    return (
        (trace is not None and "debugpy" in str(trace))
        or "debugpy" in sys.modules
    )
if running_in_vscode_debug():
    warnings.simplefilter("error", UserWarning)


# ------- Indexed Dataset Wrapper for Presampling -------
class IndexedDatasetWrapper:
    """Wrapper that adds graph_idx and split_id to each Data object for presample cache lookup.

    This wrapper adds the dataset index and split identifier to each Data object so that
    during batching, we know which original graphs are in the batch and can
    look up their presampled subgraphs from the correct split's cache.
    """

    def __init__(self, base_dataset, split_id: int):
        """
        Args:
            base_dataset: The underlying PyG dataset
            split_id: Integer identifier for the split (0=train, 1=val, 2=test)
        """
        self.base = base_dataset
        self.split_id = split_id

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]
        # Add graph index and split_id for cache lookup during training
        data.graph_idx = torch.tensor([idx], dtype=torch.long)
        data.split_id = torch.tensor([self.split_id], dtype=torch.long)
        return data


# ------- Experiment class -------

class Experiment:
    """**Experiment scaffolding. Override or pass callables via ExperimentConfig.**

    Key hooks to override or supply in config:
      - `model_fn(cfg)` -> `nn.Module`
      - `dataloader_fn(cfg)` -> (train_loader, val_loader, test_loader)
      - `criterion_fn(cfg)` -> loss module
      - `metric_fn(preds, targets)` -> `dict` of metrics

    The default training/eval code is generic for classification/regression tasks. For
    more custom tasks (structured outputs, GNNs with special batching), override
    `train_one_epoch` and `evaluate`.

    ## Example usage snippet 
    (fill these callables)

    ```python
    def my_model_fn(cfg):
        # return an nn.Module instance
        return MyModel(**cfg.model_kwargs)

    def my_dataloader_fn(cfg):
        # return train_loader, val_loader, test_loader (test optional)
        # e.g. using torch_geometric's DataLoader or standard DataLoader
        return train_loader, val_loader, test_loader

    def my_metric_fn(preds, targets):
        # preds / targets are cpu tensors. Return dict of metrics, e.g. {'accuracy':acc}
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(targets.numpy(), preds.numpy())
        return {'accuracy': acc, 'metric': acc}

    cfg = ExperimentConfig(name='myexp', model_fn=my_model_fn, dataloader_fn=my_dataloader_fn,
                           criterion_fn=lambda c: nn.CrossEntropyLoss(), metric_fn=my_metric_fn)
    exp = Experiment(cfg)
    exp.train()
    ```

    Customize by subclassing Experiment and overriding `train_one_epoch` and `evaluate` for advanced flows.
    """

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

        # logging
        self.logger = self._setup_logger()

        # deterministic
        self._set_seed(cfg.seed)

        # placeholders (filled in build)
        self.model: Optional[nn.Module] = None
        self.criterion: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None

        # data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # bookkeeping
        self.down_metrics = ['MAE']
        if self.cfg.train.metric in self.down_metrics:
            self.best_metric = float('inf')
        else:
            self.best_metric = -float('inf')
        self.history = {"train_loss": [], "val_loss": []}

        # Set subgraph sampler based on config (default: graphlet)
        sampler_name = getattr(self.cfg, 'sampler', 'graphlet')
        if sampler_name == 'rwr':
            import rwr_sampler
            self.sampler = rwr_sampler.sample_batch
            self.logger.info(f"Using RWR sampler")
        elif sampler_name == 'ugs':
            import ugs_sampler
            self.sampler = ugs_sampler.sample_batch
            self.logger.info(f"Using UGS sampler")
        elif sampler_name == 'uniform':
            import uniform_sampler
            self.sampler = uniform_sampler.sample_batch
            self.logger.info(f"Using Uniform sampler")
        elif sampler_name == 'graphlet':
            import graphlet_sampler
            self.sampler = graphlet_sampler.sample_batch
            self.logger.info(f"Using Graphlet sampler")
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")

        # Build components
        self.build()
        
        # Log experiment setup
        self.log_config()

    # ---------- Setup helpers ----------
    def _setup_logger(self):
        # Color codes for terminal output
        class ColoredFormatter(logging.Formatter):
            COLORS = {
                'DEBUG': '\033[36m',    # Cyan
                'INFO': '\033[32m',     # Green
                'WARNING': '\033[33m',  # Yellow
                'ERROR': '\033[31m',    # Red
                'CRITICAL': '\033[35m', # Magenta
            }
            RESET = '\033[0m'
            BOLD = '\033[1m'
            DIM = '\033[2m'

            def format(self, record):
                # Color the level name
                levelname_color = self.COLORS.get(record.levelname, self.RESET)
                record.levelname = f"{levelname_color}{record.levelname}{self.RESET}"

                # Format timestamp in dim color
                log_time = self.formatTime(record, '%H:%M:%S')
                formatted_time = f"{self.DIM}{log_time}{self.RESET}"

                # Format the message
                message = record.getMessage()

                return f"{formatted_time} | {record.levelname} | {message}"

        logger = logging.getLogger(self.cfg.name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            # Console handler with colors
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(ColoredFormatter())
            logger.addHandler(ch)

            # File handler (plain text, no ANSI colors)
            fh = logging.FileHandler(os.path.join(self.cfg.log_dir, "train.log"))
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                              datefmt="%Y-%m-%d %H:%M:%S"))
            logger.addHandler(fh)

        if self.cfg.tracker in ['Off', 'OFF', 'off', 'false', 'False']:
            from gxl.wandb_writer import DummyWriter
            self.writer = DummyWriter()
            return logger
        # if tracker is on initiate W&B or Tensorboard
        try:
            from gxl.wandb_writer import WandBWriter
            self.writer = WandBWriter(self.cfg)
        except ImportError:
            self.logger.info('WandB: supports wandb, `pip install wandb` and login to use.')
            try: 
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.cfg.log_dir)
                self.writer.add_hparams(self.cfg.parameter_dict(),{})
            except ImportError:
                self.logger.error('Install Tensorboard or WandB.')
                os._exit(-1)
        return logger

    def log_config(self):
        # Only log selected config sections, not everything
        key_params = {
            "experiment": self.cfg.name,
            "dataset": self.cfg.dataset_name,
            "task": self.cfg.task,
            "device": self.cfg.device,
            "seed": self.cfg.seed,
            "training": {
                "epochs": self.cfg.train.epochs,
                "batch_size": f"train={self.cfg.train.train_batch_size}, val={self.cfg.train.val_batch_size}",
                "lr": self.cfg.train.lr,
                "optimizer": self.cfg.train.optimizer,
                "weight_decay": self.cfg.train.weight_decay,
            }
        }

        # Add scheduler info if available
        if hasattr(self.cfg.train, 'scheduler') and self.cfg.train.scheduler:
            key_params["training"]["scheduler"] = self.cfg.train.scheduler.type

        pretty = json.dumps(key_params, indent=2)
        self.logger.info("=== Experiment Configuration ===")
        self.logger.info(pretty)

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ---------- Build components (override or supply fns) ----------
    def build(self):
        self.logger.info("=" * 60)
        self.logger.info("Building experiment components...")
        self.logger.info("=" * 60)

        # model
        self.logger.info("→ Building model...")
        if callable(self.cfg.model_fn):
            self.model = self.cfg.model_fn(self.cfg)
        else:
            raise NotImplementedError("Provide cfg.model_fn(cfg) returning an nn.Module")
        self.model.to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"  ✓ Model built: {total_params:,} params ({trainable_params:,} trainable)")

        # criterion
        self.logger.info("→ Building loss criterion...")
        if callable(self.cfg.criterion_fn):
            self.criterion = self.cfg.criterion_fn()
            self.logger.info(f"  ✓ Criterion: {self.criterion.__class__.__name__}")

        # dataloaders
        self.logger.info("→ Building dataloaders (this may take a while)...")
        if callable(self.cfg.dataloader_fn):
            loaders = self.cfg.dataloader_fn(self.cfg)
            if isinstance(loaders, tuple) and (2 <= len(loaders) <= 3):
                self.train_loader, self.val_loader = loaders[0], loaders[1]
                self.test_loader = loaders[2] if len(loaders) == 3 else None
                self.logger.info(f"  ✓ Train: {len(self.train_loader)} batches | Val: {len(self.val_loader)} batches" +
                               (f" | Test: {len(self.test_loader)} batches" if self.test_loader else ""))
            else:
                raise ValueError("dataloader_fn must return (train_loader, val_loader, [test_loader])")
        else:
            raise NotImplementedError("Provide cfg.dataloader_fn(cfg) returning dataloaders")

        # Presampling setup (if enabled)
        self.presample_cache = None
        if (getattr(self.cfg, 'presample', False)
                and self.cfg.model_config.subgraph_sampling
                and self.cfg.model_name != 'SD-GNN'):
            self._setup_presampling()

        # optimizer
        self.logger.info("→ Building optimizer...")
        self.optimizer = self._build_optimizer()
        self.logger.info(f"  ✓ Optimizer: {self.optimizer.__class__.__name__} (lr={self.cfg.train.lr})")

        # scheduler (optional)
        self.logger.info("→ Building scheduler...")
        self.scheduler = self._build_scheduler()
        if self.scheduler:
            self.logger.info(f"  ✓ Scheduler: {self.scheduler.__class__.__name__}")
        else:
            self.logger.info("  ✓ No scheduler")

        # AMP scaler
        if self.cfg.train.use_amp:
            self.logger.info("→ Enabling mixed precision training...")
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
            self.logger.info("  ✓ AMP enabled")

        # optionally resume
        if self.cfg.resume_from:
            self.logger.info(f"→ Resuming from checkpoint: {self.cfg.resume_from}")
            self.load_checkpoint(self.cfg.resume_from)

        self.logger.info("=" * 60)
        self.logger.info("✓ Build complete - Ready to train!")
        self.logger.info("=" * 60)

    def _build_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.cfg.train.optimizer.lower() in ('adam', 'adamw'):
            opt_cls = optim.AdamW if self.cfg.train.optimizer.lower() == 'adamw' else optim.Adam
            return opt_cls(params, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)
        elif self.cfg.train.optimizer.lower() == 'sgd':
            return optim.SGD(params, lr=self.cfg.train.lr, momentum=0.9, weight_decay=self.cfg.train.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.train.optimizer}")

    def _build_scheduler(self):
        sch = None
        s = self.cfg.train.scheduler
        if not s:
            return None
        if s.type == 'step':
            sch = optim.lr_scheduler.StepLR(self.optimizer, step_size=s.step_size, gamma=s.gamma)
        elif s.type == 'cosine':
            sch = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg.train.epochs)
        elif s.type == 'reduce_on_plateau':
            sch = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=s.patience)
        else:
            self.logger.warning("Unknown scheduler type: %s", s.type)
        return sch

    def _setup_presampling(self):
        """Set up presampling: wrap datasets with IndexedDatasetWrapper and build cache."""
        from torch_geometric.loader import DataLoader as PyGDataLoader

        k = self.cfg.model_config.subgraph_param.k
        m = self.cfg.model_config.subgraph_param.m

        self.logger.info(f"→ Setting up presampling (k={k}, m={m})...")

        # Build presample cache for each split
        self.presample_cache = {'train': {}, 'val': {}, 'test': {}}

        datasets_info = [
            ("train", self.train_loader.dataset),
            ("val", self.val_loader.dataset),
        ]
        if self.test_loader is not None:
            datasets_info.append(("test", self.test_loader.dataset))

        total_presampled = 0
        total_failed = 0

        for split_name, dataset in datasets_info:
            self.logger.info(f"  Presampling {split_name} set ({len(dataset)} graphs)...")

            for i in tqdm(range(len(dataset)), desc=f"  {split_name}", ncols=80, leave=False):
                data = dataset[i]

                # Create ptr tensor for single graph
                num_nodes = data.num_nodes
                ptr = torch.tensor([0, num_nodes], dtype=torch.long)

                # Use varying seed per graph for diversity
                graph_seed = self.cfg.seed + total_presampled

                try:
                    result = self.sampler(data.edge_index, ptr, m, k,
                                         mode="sample", seed=graph_seed)

                    if len(result) == 6:
                        nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_t, log_probs_t = result
                    else:
                        nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_t = result
                        log_probs_t = None

                    # Store in cache - these are already in the correct format for a single graph
                    cache_entry = {
                        'nodes': nodes_t.clone(),           # [m, k] node indices
                        'edge_index': edge_index_t.clone(), # [2, E] already adjusted for samples
                        'edge_ptr': edge_ptr_t.clone(),     # [m+1] edge pointers
                        'edge_src': edge_src_t.clone(),     # [E] original edge indices
                    }
                    if log_probs_t is not None:
                        cache_entry['log_probs'] = log_probs_t.clone()
                    self.presample_cache[split_name][i] = cache_entry
                    total_presampled += 1
                except Exception as e:
                    self.presample_cache[split_name][i] = None
                    total_failed += 1
                    total_presampled += 1

        if total_failed > 0:
            self.logger.warning(f"  ⚠ {total_failed} graphs failed presampling")
        self.logger.info(f"  ✓ Presampled {total_presampled - total_failed}/{total_presampled} graphs")

        # Wrap dataloaders to add graph_idx and split_id
        self.train_loader = self._wrap_loader_with_index(self.train_loader, 0)  # 0 = train
        self.val_loader = self._wrap_loader_with_index(self.val_loader, 1)      # 1 = val
        if self.test_loader is not None:
            self.test_loader = self._wrap_loader_with_index(self.test_loader, 2)  # 2 = test

    def _wrap_loader_with_index(self, loader, split_id: int):
        """Wrap dataloader with IndexedDatasetWrapper to add graph_idx and split_id."""
        from torch_geometric.loader import DataLoader as PyGDataLoader

        wrapped_dataset = IndexedDatasetWrapper(loader.dataset, split_id)

        return PyGDataLoader(
            wrapped_dataset,
            batch_size=loader.batch_size,
            shuffle=(split_id == 0),  # Only shuffle train (split_id=0)
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            drop_last=loader.drop_last
        )

    # ---------- Training / evaluation ----------
    def train(self):
        """ Returns: `dict` with keys ['train_metric', 'test_metric', 'val_metric']
        """
        self.logger.info("Starting training for %d epochs", self.cfg.train.epochs)
        val_ema_alpha = self.cfg.model_config.kwargs.get('val_ema_alpha', 0.0)
        self._ema_val = None  # reset at the start of each training run
        for epoch in range(1, self.cfg.train.epochs + 1):
            t0 = time.time()
            train_stats = self.train_one_epoch(epoch)
            val_stats = self.evaluate(epoch)

            # logging
            self.history['train_loss'].append(train_stats.get('loss', None))
            self.history['val_loss'].append(val_stats.get('loss', None))

            # scheduler step (handle reduce_on_plateau separately)
            if self.scheduler is not None and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_stats.get('loss'))

            # save checkpoint
            if epoch % self.cfg.save_every == 0:
                raw_metric = val_stats.get("metric")
                if val_ema_alpha > 0.0 and raw_metric is not None:
                    if self._ema_val is None:
                        self._ema_val = raw_metric
                    else:
                        self._ema_val = val_ema_alpha * self._ema_val + (1 - val_ema_alpha) * raw_metric
                    metric_for_save = self._ema_val
                else:
                    metric_for_save = raw_metric
                self.save_checkpoint(epoch, metric_for_save)

            dt = time.time() - t0
            ema_str = f" | val_ema {self._ema_val:.5f}" if self._ema_val is not None else ""
            self.logger.info("Epoch %03d | time %.1fs | train_loss %.4f | val_loss %.4f | val_metric %.5f%s\n",
                              epoch, dt, train_stats.get('loss', float('nan')), val_stats.get('loss', float('nan')),
                              float(val_stats.get('metric', None)), ema_str)

        fname_best_model = f"best_model.pth"
        path_best_model = os.path.join(self.cfg.checkpoint_dir, fname_best_model)
        self.load_checkpoint(path_best_model)
        test_best_stats  = self.evaluate(split='test')
        train_best_stats = self.evaluate(split='train')
        val_best_stats   = self.evaluate(split='val')

        test_metric  = test_best_stats.get('metric', None)
        train_metric = train_best_stats.get('metric', None)
        val_metric   = val_best_stats.get('metric', None)

        result_out = {
            "train_metric" : train_metric,
            "test_metric" : test_metric,
            "val_metric" : val_metric
        }
        return result_out

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Default training loop. Override for custom tasks.

        Returns a dict of statistics (must contain 'loss').
        """
        self.model.train()
        running_loss = 0.0
        n_examples = 0

        pbar = tqdm(self.train_loader, desc=f"Train {epoch}",ncols=100,dynamic_ncols=False)
        for batch in pbar:
            # Expectation: user dataloader yields a tuple (inputs..., labels)
            # You can adapt these lines to your use-case. For GNNs using torch_geometric,
            # you might receive a Batch object where `batch.x, batch.edge_index, batch.batch`
            # If subgraph sampling is required then return: 
            # `x_global`, `nodes_t`, `edge_index_t`, `edge_ptr_t`, `graph_id_t`, `k`

            batch, labels = self._unpack_batch(batch)

            batch = batch.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=self.cfg.train.use_amp):
                output = self.model(batch)
                #user-provided criterion must accept (outputs, labels)
                if self.cfg.task == "Binary-Classification":
                    # CrossEntropyLoss needs long targets, BCEWithLogitsLoss needs float
                    if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
                        loss = self.criterion(output, labels.long())
                    else:
                        # BCEWithLogitsLoss needs float targets and matching shapes
                        if output.dim() > 1 and output.shape[-1] == 1:
                            output = output.squeeze(-1)  # [B, 1] -> [B]
                        loss = self.criterion(output, labels.float())
                elif self.cfg.task == "Multi-Lable-Binary-Classification":
                    loss = self.criterion(output, labels.float())
                elif self.cfg.task == "Multi-Target-Regression":
                    loss = self.criterion(output, labels.float())
                elif self.cfg.task == "Multi-Class-Classification":
                    loss = self.criterion(output, labels.long())
                elif self.cfg.task == "Node-Classification":
                    # Node-level classification: output is [num_nodes, num_classes], labels is [num_nodes]
                    # Apply train_mask if available
                    if batch.train_mask is not None:
                        train_mask = batch.train_mask
                        loss = self.criterion(output[train_mask], labels[train_mask].long())
                    else:
                        loss = self.criterion(output, labels.long())
                elif self.cfg.task == "Node-Multilabel-Classification":
                    # Multi-label node classification (e.g. ogbn-proteins): labels are [N, C] float
                    if batch.train_mask is not None:
                        train_mask = batch.train_mask
                        loss = self.criterion(output[train_mask], labels[train_mask].float())
                    else:
                        loss = self.criterion(output, labels.float())
                elif self.cfg.task == "Link-Prediction":
                    loss = self.criterion(output, labels)
                elif self.cfg.task == "Regression":
                    loss = self.criterion(output, labels.unsqueeze(-1))
                elif self.cfg.task == "Single-Target-Regression":
                    loss = self.criterion(output, labels.float())

                else:
                    raise ValueError(f"unknown task: {self.cfg.task}")


            # backward
            if self.cfg.train.use_amp:
                assert self.scaler is not None
                self.scaler.scale(loss).backward()
                if self.cfg.train.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.cfg.train.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
                self.optimizer.step()

            # bookkeeping
            batch_size = self._get_batch_size(batch)
            running_loss += loss.item() * batch_size
            n_examples += batch_size
            avg_loss = running_loss / max(1, n_examples)
            pbar.set_postfix(loss=avg_loss)

        return {"loss": avg_loss}

    def evaluate(self, epoch: Optional[int] = None, split: Optional[Literal['test', 'train', 'val']] = 'val') -> Dict[str, Any]:
        """Evaluate on validation set. Returns dict with 'loss' and optionally 'metric'."""
        self.model.eval()
        running_loss = 0.0
        n_examples = 0
        all_logits = []
        all_targets = []
        all_edge_label_index = []
        loaders = {
            "test": self.test_loader,
            "train": self.train_loader,
            "val": self.val_loader,
        }

        if split not in loaders:
            raise ValueError(f"Unknown split type ({split}) in evaluate.")

        loader = loaders[split]

        pbar = tqdm(loader, desc=f"{split} {epoch}" if epoch else f"{split}",ncols=86,dynamic_ncols=False)
        with torch.no_grad():
            for batch in pbar:
                batch, labels = self._unpack_batch(batch)
                batch = batch.to(self.device)
                labels = labels.to(self.device)

                with torch.amp.autocast('cuda', enabled=self.cfg.train.use_amp):
                    output = self.model(batch)
                    #user-provided criterion must accept (outputs, labels)
                    if self.cfg.task == "Binary-Classification":
                        # CrossEntropyLoss needs long targets, BCEWithLogitsLoss needs float
                        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
                            loss = self.criterion(output, labels.long())
                        else:
                            # BCEWithLogitsLoss needs float targets and matching shapes
                            if output.dim() > 1 and output.shape[-1] == 1:
                                output = output.squeeze(-1)  # [B, 1] -> [B]
                            loss = self.criterion(output, labels.float())
                    if self.cfg.task == "Multi-Lable-Binary-Classification":
                        loss = self.criterion(output, labels.float())
                    if self.cfg.task == "Multi-Target-Regression":
                        loss = self.criterion(output, labels.float())
                    if self.cfg.task == "Multi-Class-Classification":
                        loss = self.criterion(output, labels.long())
                    if self.cfg.task == "Node-Classification":
                        # Node-level classification: output is [num_nodes, num_classes], labels is [num_nodes]
                        # Apply the appropriate mask based on split
                        if split == 'val' and batch.val_mask is not None:
                            mask = batch.val_mask
                        elif split == 'test' and batch.test_mask is not None:
                            mask = batch.test_mask
                        elif split == 'train' and batch.train_mask is not None:
                            mask = batch.train_mask
                        else:
                            mask = None

                        if mask is not None:
                            loss = self.criterion(output[mask], labels[mask].long())
                        else:
                            loss = self.criterion(output, labels.long())
                    if self.cfg.task == "Node-Multilabel-Classification":
                        # Multi-label node classification: labels are [N, C] float
                        if split == 'val' and batch.val_mask is not None:
                            mask = batch.val_mask
                        elif split == 'test' and batch.test_mask is not None:
                            mask = batch.test_mask
                        elif split == 'train' and batch.train_mask is not None:
                            mask = batch.train_mask
                        else:
                            mask = None

                        if mask is not None:
                            loss = self.criterion(output[mask], labels[mask].float())
                        else:
                            loss = self.criterion(output, labels.float())
                    if self.cfg.task == "Link-Prediction":
                        loss = self.criterion(output, labels)
                    if self.cfg.task == "Regression":
                        loss = self.criterion(output, labels.unsqueeze(-1))
                    if self.cfg.task == "Single-Target-Regression":
                        loss = self.criterion(output, labels.float())


                # collect logits and targets
                if self.cfg.task == "Link-Prediction":
                    all_logits.append(output.detach().cpu().numpy())
                    all_targets.append(labels.detach().cpu().numpy())
                    all_edge_label_index.append(batch.edge_label_index.detach().cpu().numpy())
                elif self.cfg.task == "Node-Classification":
                    # Apply the appropriate mask for node classification
                    if split == 'val' and batch.val_mask is not None:
                        mask = batch.val_mask
                    elif split == 'test' and batch.test_mask is not None:
                        mask = batch.test_mask
                    elif split == 'train' and batch.train_mask is not None:
                        mask = batch.train_mask
                    else:
                        mask = None

                    if mask is not None:
                        all_logits.append(output[mask].detach().cpu())
                        all_targets.append(labels[mask].detach().cpu())
                    else:
                        all_logits.append(output.detach().cpu())
                        all_targets.append(labels.detach().cpu())
                else:
                    all_logits.append(output.detach().cpu())
                    all_targets.append(labels.detach().cpu())

                batch_size = self._get_batch_size(batch)
                running_loss += loss.item() * batch_size
                n_examples += batch_size

        avg_loss = running_loss / max(1, n_examples)
        if self.cfg.task != 'Link-Prediction':
            all_logits = torch.cat(all_logits, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

        metrics = {}
        if callable(self.cfg.metric_fn):
            try:
                # IMPORTANT: average_precision_score expects (y_true, y_score)
                # y_true shape: (N,) for binary or (N, n_classes) for multilabel
                # y_score shape: same as y_true for multilabel
                if self.cfg.task == "Multi-Lable-Binary-Classification":
                    all_probs = torch.sigmoid(all_logits)
                    metrics = self.cfg.metric_fn(all_targets.numpy(), all_probs.numpy())
                if self.cfg.task == "Binary-Classification":
                    # For ROCAUC, we need scores not predictions
                    if self.cfg.train.metric == "ROCAUC":
                        # Handle both out_dim=1 (single logit) and out_dim=2 (two logits)
                        if all_logits.dim() == 1 or all_logits.shape[-1] == 1:
                            all_scores = all_logits.squeeze()  # [B]
                        else:
                            all_scores = all_logits[:, 1]  # Take positive class score
                        metrics = self.cfg.metric_fn(all_targets.numpy(), all_scores.numpy())
                    else:
                        all_preds = torch.argmax(all_logits, dim=1)   # [B]
                        metrics = self.cfg.metric_fn(all_preds.numpy(), all_targets.numpy())
                if self.cfg.task == "Multi-Target-Regression":
                    metrics = self.cfg.metric_fn(all_logits.numpy(), all_targets.numpy())
                if self.cfg.task == "Multi-Class-Classification":
                    all_preds  = all_logits.argmax(dim=-1)
                    metrics = self.cfg.metric_fn(all_preds.numpy(), all_targets.numpy())
                if self.cfg.task == "Node-Classification":
                    # Node-level classification: argmax over classes, then compute metrics
                    all_preds = all_logits.argmax(dim=-1)  # [num_nodes]
                    metrics = self.cfg.metric_fn(all_preds.numpy(), all_targets.numpy())
                if self.cfg.task == "Node-Multilabel-Classification":
                    # Multi-label: sigmoid → per-task ROCAUC
                    all_probs = torch.sigmoid(all_logits)
                    metrics = self.cfg.metric_fn(all_targets.numpy(), all_probs.numpy())
                if self.cfg.task == "Regression":
                    metrics = self.cfg.metric_fn(all_logits.numpy(), all_targets.squeeze().numpy())
                if self.cfg.task == "Link-Prediction":
                    metrics = self.cfg.metric_fn(all_logits,
                                                 all_targets,
                                                 all_edge_label_index)
                if self.cfg.task == "Single-Target-Regression":
                    metrics = self.cfg.metric_fn(all_logits.numpy(), all_targets.numpy())
                
            except Exception as e:
                self.logger.warning("metric_fn failed: %s", e)
                exit()
        metric_val = metrics.get(self.cfg.train.metric)

        # tracker logging
        if epoch is not None:
            self.writer.add_scalar('val/loss', avg_loss, epoch)
            if metric_val is not None:
                self.writer.add_scalar('val/metric', metric_val, epoch)

        return {"loss": avg_loss, "metric": metric_val, **metrics}

    # ---------- Utilities / I/O ----------
    def _unpack_batch(self, batch: SubgraphFeaturesBatch) -> tuple[SubgraphFeaturesBatch, torch.Tensor]:
        """
        Unpack batch data for GNN processing.
    
        Args:
            batch: Input batch (tuple, list, or torch_geometric.data.Batch)
            subgraph_sampling: Whether to perform subgraph sampling
        
        Returns:
            Tuple of (features, labels)
        """
        is_link_prediction = (self.cfg.task == 'Link-Prediction')
        edge_attr_required = (self.cfg.model_config.mpnn_type == 'gine')
        sampling_required  = self.cfg.model_config.subgraph_sampling
    
        # Validate batch format
        required_attrs = ['x', 'edge_index', 'batch']
        if edge_attr_required:
            required_attrs.append('edge_attr')
        if is_link_prediction:
            required_attrs.extend(['edge_label_index', 'edge_label'])
        else:
            required_attrs.append('y')
        if sampling_required:
            required_attrs.append('ptr')
    
        missing_attrs = [attr for attr in required_attrs if not hasattr(batch, attr)]
        if missing_attrs:
            raise ValueError(
                f"Unknown batch format. Expected attributes: {required_attrs}. "
                f"Missing: {missing_attrs}. "
                f"Override _unpack_batch to handle your dataloader output. "
                f"Received: {batch}"
            )
        # Minimal batch 
        sf_batch = SubgraphFeaturesBatch(x=batch.x.float(), 
                                         edge_index=batch.edge_index, 
                                         batch=batch.batch)
    
        # Handle optional edge attributes
        has_edge_attr = getattr(batch, 'edge_attr', None) is not None
        if has_edge_attr:
            sf_batch.edge_attr = batch.edge_attr.float()
        
        # For link prediction: labels are edge_label
        if self.cfg.task == 'Link-Prediction':
            sf_batch.edge_label_index = batch.edge_label_index
            labels = batch.edge_label.float()
        else:
            labels = batch.y.float()

        # Handle node classification masks
        if self.cfg.task in ('Node-Classification', 'Node-Multilabel-Classification'):
            if hasattr(batch, 'train_mask'):
                sf_batch.train_mask = batch.train_mask
            if hasattr(batch, 'val_mask'):
                sf_batch.val_mask = batch.val_mask
            if hasattr(batch, 'test_mask'):
                sf_batch.test_mask = batch.test_mask

        # load sample features
        if self.cfg.model_config.subgraph_sampling:
            sf_batch.ptr = batch.ptr
            # Pass graph_idx and split_id for presample cache lookup (if available)
            if hasattr(batch, 'graph_idx'):
                sf_batch.graph_idx = batch.graph_idx
            if hasattr(batch, 'split_id'):
                sf_batch.split_id = getattr(batch, 'split_id', None)

            # Build target_nodes / target_ptr for node/edge-level tasks with graphlet sampler.
            sampler_name = getattr(self.cfg, 'sampler', 'graphlet')
            if sampler_name == 'graphlet':
                if self.cfg.task == 'Node-Classification':
                    sf_batch = self._build_node_targets(sf_batch, batch)
                elif self.cfg.task == 'Link-Prediction':
                    sf_batch = self._build_link_targets(sf_batch)

            # SD-GNN needs per-node subgraphs regardless of sampler
            if self.cfg.model_name in ('SD-GNN', 'ARCH-2'):
                sf_batch = self._build_all_node_targets(sf_batch)

            sf_batch = self._sample_and_load_subgraphs(sf_batch)

        return (sf_batch, labels)


    def _build_node_targets(self, sf_batch: SubgraphFeaturesBatch, raw_batch) -> SubgraphFeaturesBatch:
        """Build target_nodes and target_ptr from node classification masks.

        For each graph g in the batch, the target nodes are those where the
        union of all split masks is True (train | val | test).  This ensures
        every node that may be evaluated gets subgraph embeddings.
        The result is stored as:
          target_nodes [T]   — global node IDs of all targets across the batch
          target_ptr   [G+1] — CSR boundaries so targets for graph g are
                               target_nodes[target_ptr[g] : target_ptr[g+1]]
        """
        ptr = sf_batch.ptr  # [G+1]
        G = ptr.size(0) - 1

        # Use union of all available masks so every evaluatable node
        # gets subgraph embeddings (train, val, AND test nodes).
        mask = None
        for m_attr in ('train_mask', 'val_mask', 'test_mask'):
            m_val = getattr(sf_batch, m_attr, None)
            if m_val is not None:
                mask = m_val if mask is None else (mask | m_val)

        if mask is None:
            return sf_batch  # no mask → no targets, sampler uses random seeds

        target_list = []
        target_ptr_list = [0]
        for g in range(G):
            start = ptr[g].item()
            end = ptr[g + 1].item()
            graph_mask = mask[start:end]                    # [n_g]
            local_targets = graph_mask.nonzero(as_tuple=False).view(-1)  # graph-local
            global_targets = local_targets + start          # global IDs
            target_list.append(global_targets)
            target_ptr_list.append(target_ptr_list[-1] + global_targets.size(0))

        if target_ptr_list[-1] == 0:
            return sf_batch  # no targets found in any graph

        sf_batch.target_nodes = torch.cat(target_list, dim=0).long()
        sf_batch.target_ptr = torch.tensor(target_ptr_list, dtype=torch.long)
        return sf_batch

    def _build_all_node_targets(self, sf_batch: SubgraphFeaturesBatch) -> SubgraphFeaturesBatch:
        """Set target_nodes = all nodes, target_ptr = batch.ptr for SD-GNN.

        The sampler produces S = N_total * m samples, m per node, ordered
        by graph then by node within each graph (matching arange(N_total)).
        """
        ptr = sf_batch.ptr                    # [G+1]
        N_total = int(ptr[-1].item())
        sf_batch.target_nodes = torch.arange(N_total, dtype=torch.long, device=ptr.device)
        sf_batch.target_ptr   = ptr.clone().long()
        return sf_batch

    def _build_link_targets(self, sf_batch: SubgraphFeaturesBatch) -> SubgraphFeaturesBatch:
        """Build target_nodes and target_ptr from link prediction edges.

        For each graph g, the target nodes are the unique endpoint nodes from
        edge_label_index (both source and destination of every target edge).
        Each such node gets m subgraphs seeded on it, so the model can later
        look up embeddings for both endpoints of each predicted edge.
        """
        eli = sf_batch.edge_label_index  # [2, num_pred_edges]
        if eli is None or eli.numel() == 0:
            return sf_batch

        ptr = sf_batch.ptr  # [G+1]
        G = ptr.size(0) - 1

        # All endpoint nodes (may have duplicates across edges)
        all_endpoints = torch.cat([eli[0], eli[1]], dim=0)  # [2 * num_pred_edges]

        target_list = []
        target_ptr_list = [0]
        for g in range(G):
            start = ptr[g].item()
            end = ptr[g + 1].item()
            # Endpoints belonging to graph g
            mask = (all_endpoints >= start) & (all_endpoints < end)
            graph_endpoints = all_endpoints[mask]
            # Deduplicate within this graph
            unique_targets = graph_endpoints.unique()
            target_list.append(unique_targets)
            target_ptr_list.append(target_ptr_list[-1] + unique_targets.size(0))

        if target_ptr_list[-1] == 0:
            return sf_batch

        sf_batch.target_nodes = torch.cat(target_list, dim=0).long()
        sf_batch.target_ptr = torch.tensor(target_ptr_list, dtype=torch.long)
        return sf_batch

    def _sample_and_load_subgraphs(self, batch: SubgraphFeaturesBatch)->SubgraphFeaturesBatch:
        """Handle batch unpacking with subgraph sampling."""
        # Validate subgraph parameters
        if not getattr(self.cfg.model_config, 'subgraph_param', None):
            raise ValueError(
                "Subgraph parameters required in config. "
                "Please set cfg.model_config.subgraph_param with 'k' and 'm' values."
            )

        k = self.cfg.model_config.subgraph_param.k
        m = self.cfg.model_config.subgraph_param.m

        # Check if we can use presample cache
        if self.presample_cache is not None and batch.graph_idx is not None:
            return self._load_from_presample_cache(batch, k, m)

        # Move to CPU for sampling (if needed by sampler)
        batch.edge_index = batch.edge_index.cpu()
        batch.ptr = batch.ptr.cpu()

        # Determine target info for graphlet sampler
        target_nodes = getattr(batch, 'target_nodes', None)
        target_ptr_t = getattr(batch, 'target_ptr', None)

        # Perform subgraph sampling on-the-fly
        try:
            sampler_kwargs = dict(mode="sample", seed=self.cfg.seed)
            if target_nodes is not None and target_ptr_t is not None:
                sampler_kwargs['target_nodes'] = target_nodes.cpu()
                sampler_kwargs['target_ptr'] = target_ptr_t.cpu()

            result = self.sampler(batch.edge_index, batch.ptr, m, k, **sampler_kwargs)

            if len(result) == 6:
                batch.nodes_sampled, batch.edge_index_sampled, batch.edge_ptr, \
                    batch.sample_ptr, batch.edge_src_global, batch.log_probs = result
            else:
                batch.nodes_sampled, batch.edge_index_sampled, batch.edge_ptr, \
                    batch.sample_ptr, batch.edge_src_global = result

            # Store target info on batch for downstream model use
            if target_nodes is not None:
                batch.target_nodes = target_nodes
                batch.target_ptr = target_ptr_t

        except Exception as e:
            # Fallback: use placeholders if sampling fails
            self.logger.warning(f"\nSubgraph sampler failed, using placeholders: {e}")

            num_graphs = int(batch.batch.max().item()) + 1
            batch.nodes_sampled, batch.edge_index_sampled, batch.edge_ptr, batch.sample_ptr, batch.edge_src_global = \
                self._make_placeholders(num_graphs, m, k)

        return batch

    def _load_from_presample_cache(self, batch: SubgraphFeaturesBatch, k: int, m: int) -> SubgraphFeaturesBatch:
        """Load presampled subgraphs from cache and combine them for the batch.

        This reconstructs the same output format as the on-the-fly sampler:
        - nodes_sampled: [total_samples, k] with GLOBAL node indices
        - edge_index_sampled: [2, total_edges] with adjusted indices for batched samples
        - edge_ptr: [total_samples + 1] cumulative edge counts
        - sample_ptr: [num_graphs + 1] cumulative sample counts
        - edge_src_global: [total_edges] original edge indices in batched graph
        """
        graph_indices = batch.graph_idx.cpu().flatten().tolist()
        num_graphs = len(graph_indices)
        ptr = batch.ptr.cpu()

        # Get split from split_id (0=train, 1=val, 2=test)
        split_id = batch.split_id[0].item() if batch.split_id is not None else None
        split_map = {0: 'train', 1: 'val', 2: 'test'}
        split_name = split_map.get(split_id)

        if split_name is None or split_name not in self.presample_cache:
            # Cache miss - fall back to on-the-fly sampling
            batch.edge_index = batch.edge_index.cpu()
            batch.ptr = batch.ptr.cpu()
            result = self.sampler(batch.edge_index, batch.ptr, m, k, mode="sample", seed=self.cfg.seed)
            if len(result) == 6:
                batch.nodes_sampled, batch.edge_index_sampled, batch.edge_ptr, \
                    batch.sample_ptr, batch.edge_src_global, batch.log_probs = result
            else:
                batch.nodes_sampled, batch.edge_index_sampled, batch.edge_ptr, \
                    batch.sample_ptr, batch.edge_src_global = result
            return batch

        cache = self.presample_cache[split_name]

        # Collect presampled data for each graph
        all_nodes = []
        all_edges = []
        all_edge_src = []
        all_log_probs = []
        has_log_probs = False
        edge_ptr_list = [0]
        sample_ptr_list = [0]

        cumulative_node_offset = 0
        cumulative_edge_offset = 0
        cumulative_sample_offset = 0

        # Compute original graph edge offsets for edge_src adjustment
        orig_edge_offsets = [0]
        for g in range(num_graphs - 1):
            node_start = ptr[g].item()
            node_end = ptr[g + 1].item()
            edge_mask = (batch.edge_index[0] >= node_start) & (batch.edge_index[0] < node_end)
            orig_edge_offsets.append(orig_edge_offsets[-1] + edge_mask.sum().item())

        for g_idx, orig_idx in enumerate(graph_indices):
            ps_data = cache.get(orig_idx)

            if ps_data is None:
                # This graph failed presampling - use placeholders for it
                nodes_g = torch.full((m, k), -1, dtype=torch.long)
                edges_g = torch.empty((2, 0), dtype=torch.long)
                edge_ptr_g = torch.zeros(m + 1, dtype=torch.long)
                edge_src_g = torch.empty(0, dtype=torch.long)
                log_probs_g = None
            else:
                nodes_g = ps_data['nodes']        # [m, k]
                edges_g = ps_data['edge_index']   # [2, E_g]
                edge_ptr_g = ps_data['edge_ptr']  # [m+1]
                edge_src_g = ps_data['edge_src']  # [E_g]
                log_probs_g = ps_data.get('log_probs')  # [m] or None
                if log_probs_g is not None:
                    has_log_probs = True

            n_samples = nodes_g.shape[0]
            n_edges = edges_g.shape[1]

            # Adjust node indices: add cumulative node offset from batched graph
            node_offset = ptr[g_idx].item()
            nodes_adjusted = nodes_g + node_offset

            # DO NOT adjust edge indices - the model does this itself!
            # The sampler returns LOCAL indices [0, k-1] per sample
            # The model adds sample offsets in encode_subgraphs (lines 436-437)
            edges_adjusted = edges_g  # Keep as-is

            # Adjust edge_src: add original graph's edge offset
            edge_src_adjusted = edge_src_g + orig_edge_offsets[g_idx]

            # Collect
            all_nodes.append(nodes_adjusted)
            all_edges.append(edges_adjusted)
            all_edge_src.append(edge_src_adjusted)
            if log_probs_g is not None:
                all_log_probs.append(log_probs_g)
            elif has_log_probs:
                # Placeholder -inf for graphs without log_probs
                all_log_probs.append(torch.full((nodes_g.shape[0],), float('-inf'), dtype=torch.float32))

            # Build edge_ptr entries (offset by cumulative edge count)
            for i in range(n_samples):
                edge_ptr_list.append(cumulative_edge_offset + edge_ptr_g[i + 1].item())

            # Update cumulative counters
            cumulative_node_offset += ptr[g_idx + 1].item() - ptr[g_idx].item()
            cumulative_edge_offset += n_edges
            cumulative_sample_offset += n_samples
            sample_ptr_list.append(cumulative_sample_offset)

        # Concatenate all
        batch.nodes_sampled = torch.cat(all_nodes, dim=0)  # [total_samples, k]
        batch.edge_index_sampled = torch.cat(all_edges, dim=1) if all_edges else torch.empty((2, 0), dtype=torch.long)
        batch.edge_ptr = torch.tensor(edge_ptr_list, dtype=torch.long)
        batch.sample_ptr = torch.tensor(sample_ptr_list, dtype=torch.long)
        batch.edge_src_global = torch.cat(all_edge_src, dim=0) if all_edge_src else torch.empty(0, dtype=torch.long)
        if has_log_probs and all_log_probs:
            batch.log_probs = torch.cat(all_log_probs, dim=0)

        return batch

    def _get_batch_size(self, batch):
        # infer a batch-size for loss averaging. Override if needed.
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x = batch[0]
            if hasattr(x, 'size'):
                return x.size(0)
            # for GNN Batch, batch.batch is a vector of node->graph ids
        try:
            if hasattr(batch, 'batch'):
                # number of graphs = max(batch.batch) + 1
                return int(batch.batch.max().item()) + 1
        except Exception:
            pass
        return 1



    @staticmethod
    def _safe_torch_load(path: Path, map_location):
        """Safely load a PyTorch checkpoint handling different PyTorch versions.
    
        Args:
            path: Path to checkpoint file
            map_location: Device to map tensors to
        
        Returns:
            Loaded state dictionary
        """
        # Try with weights_only=False first (for optimizer states, configs, etc.)
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            # PyTorch < 2.6 doesn't have weights_only parameter
            return torch.load(path, map_location=map_location)
        except Exception as e:
            # If we get the numpy._core error, add safe globals and retry
            if "numpy._core.multiarray.scalar" in str(e):
                try:
                    # Add numpy types to safe globals
                    import numpy as np
                    torch.serialization.add_safe_globals([
                        np.core.multiarray.scalar,
                        np.ndarray,
                        np.dtype,
                    ])
                    return torch.load(path, map_location=map_location, weights_only=False)
                except:
                    pass
            raise


    def save_checkpoint(self, epoch: int, metric: Optional[float] = None):
        """Save model checkpoint and optionally update best model.
    
        Args:
            epoch: Current training epoch
            metric: Validation metric value for this epoch
        """
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = Path(self.cfg.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
        # Determine if this is the best model
        is_best = self._is_best_metric(metric)
    
        # Update best metric if needed
        if is_best:
            self.best_metric = metric
            self.logger.info(f"New best model! Metric: {metric:.4f}")
    
        # Prepare checkpoint state
        state = self._create_checkpoint_state(epoch, metric)
    
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f"{self.cfg.name}_epoch{epoch:03d}.pth"
        self._save_state(state, checkpoint_path)
    
        # Save best model if applicable
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            self._save_state(state, best_path)
    
        # Clean up old checkpoints
        self._cleanup_old_checkpoints(checkpoint_dir)


    def _is_best_metric(self, metric: Optional[float]) -> bool:
        """Check if the current metric is the best so far.
    
        Args:
            metric: Current metric value
        
        Returns:
            True if this is the best metric, False otherwise
        """
        if metric is None:
            return False
    
        is_down_metric = self.cfg.train.metric in self.down_metrics
    
        if is_down_metric:
            return metric < self.best_metric
        else:
            return metric > self.best_metric


    def _create_checkpoint_state(self, epoch: int, metric: Optional[float]) -> Dict[str, Any]:
        """Create checkpoint state dictionary.
    
        Args:
            epoch: Current epoch
            metric: Current metric value
        
        Returns:
            Dictionary containing all checkpoint information
        """
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optim_state': self.optimizer.state_dict(),
            'cfg': self._serialize_cfg(),
            'best_metric': self.best_metric if self._is_best_metric(metric) else self.best_metric,
            'current_metric': metric,
        }
    
        # Add gradient scaler state if using mixed precision
        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()
    
        # Optionally add scheduler state if it exists
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            state['scheduler_state'] = self.scheduler.state_dict()
    
        return state


    def _save_state(self, state: Dict[str, Any], path: Path):
        """Safely save checkpoint state to disk.
    
        Args:
            state: Checkpoint state dictionary
            path: Path to save checkpoint
        """
        try:
            # Save to temporary file first to avoid corruption
            temp_path = path.with_suffix('.pth.tmp')
            torch.save(state, temp_path)
        
            # Atomic rename
            temp_path.replace(path)
            self.logger.info(f"Checkpoint saved: {path}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint {path}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise


    def _cleanup_old_checkpoints(self, checkpoint_dir: Path):
        """Remove old checkpoints, keeping only the last K.
    
        Args:
            checkpoint_dir: Directory containing checkpoints
        """
        try:
            # Find all epoch checkpoints (exclude best_model.pth)
            pattern = f"{self.cfg.name}_epoch*.pth"
            files = sorted(
                checkpoint_dir.glob(pattern),
                key=lambda p: p.stat().st_mtime
            )
        
            # Remove old checkpoints
            if len(files) > self.cfg.keep_last_k:
                for old_file in files[:-self.cfg.keep_last_k]:
                    try:
                        old_file.unlink()
                        self.logger.info(f"Removed old checkpoint: {old_file.name}")
                    except Exception as e:
                        self.logger.warning(f"Could not remove {old_file}: {e}")
        except Exception as e:
            self.logger.warning(f"Checkpoint cleanup failed: {e}")


    def _serialize_cfg(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of the ExperimentConfig.
    
        Returns:
            Serialized configuration dictionary
        """
        def _serialize_value(v: Any) -> Any:
            """Recursively serialize a value."""
            # Handle callables
            if callable(v):
                return f"<callable {getattr(v, '__name__', repr(v))}>"
        
            # Try JSON serialization
            try:
                json.dumps(v)
                return v
            except (TypeError, ValueError):
                pass
        
            # Handle objects with __dict__
            if hasattr(v, "__dict__"):
                return {k: _serialize_value(vv) for k, vv in v.__dict__.items()}
        
            # Handle common iterables
            if isinstance(v, (list, tuple)):
                try:
                    return [_serialize_value(item) for item in v]
                except Exception:
                    return repr(v)
        
            if isinstance(v, dict):
                try:
                    return {k: _serialize_value(vv) for k, vv in v.items()}
                except Exception:
                    return repr(v)
        
            # Fallback to repr
            return repr(v)
    
        serial = {}
        excluded_keys = {'writer', 'logger', 'device'}
    
        for k, v in self.cfg.__dict__.items():
            if k in excluded_keys:
                continue
            serial[k] = _serialize_value(v)
    
        return serial


    def load_checkpoint(self, path: str, strict: bool = True, load_optimizer: bool = True):
        """Load checkpoint from disk.
    
        Args:
            path: Path to checkpoint file
            strict: Whether to strictly enforce state dict loading
            load_optimizer: Whether to load optimizer state
        
        Returns:
            Loaded checkpoint state dictionary
        """
        path = Path(path)
    
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
    
        self.logger.info(f"Loading checkpoint: {path}")
    
        try:
            state = self._safe_torch_load(path, self.device)
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise
    
        # Load model state
        try:
            self.model.load_state_dict(state['model_state'], strict=strict)
        except Exception as e:
            self.logger.error(f"Failed to load model state: {e}")
            if strict:
                raise
            self.logger.warning("Continuing with partial model loading...")
    
        # Load optimizer state
        if load_optimizer and 'optim_state' in state:
            try:
                self.optimizer.load_state_dict(state['optim_state'])
            except Exception as e:
                self.logger.warning(f"Failed to load optimizer state: {e}")
    
        # Load gradient scaler state
        if 'scaler' in state and self.scaler is not None:
            try:
                self.scaler.load_state_dict(state['scaler'])
            except Exception as e:
                self.logger.warning(f"Failed to load scaler state: {e}")
    
        # Load scheduler state if available
        if 'scheduler_state' in state and hasattr(self, 'scheduler') and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(state['scheduler_state'])
            except Exception as e:
                self.logger.warning(f"Failed to load scheduler state: {e}")
    
        # Update best metric
        self.best_metric = state.get('best_metric', self.best_metric)
    
        epoch = state.get('epoch', 0)
        self.logger.info(f"Checkpoint loaded successfully (epoch {epoch})")
    
        return state

    def _make_placeholders(self, G, m_per_graph, k):
        """Return placeholder sampler outputs for G graphs (all -1s / empty edges)."""
        B_total = G * m_per_graph
        nodes_t = torch.full((B_total, k), -1, dtype=torch.long)        # CPU or device
        edge_index_t = torch.empty((2, 0), dtype=torch.long)            # no edges
        edge_ptr_t = torch.zeros((B_total + 1,), dtype=torch.long)     # all zeros -> empty blocks
        sample_ptr_t = torch.zeros((G+1), dtype=torch.long)
        edge_src_global_t = torch.empty((0), dtype=torch.long)
        # Return 6-tuple when graphlet sampler is active (includes log_probs)
        sampler_name = getattr(self.cfg, 'sampler', 'graphlet')
        if sampler_name == 'graphlet':
            log_probs_t = torch.full((B_total,), float('-inf'), dtype=torch.float32)
            return nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_global_t, log_probs_t
        return nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_global_t