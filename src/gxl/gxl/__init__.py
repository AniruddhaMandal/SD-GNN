from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, asdict
from typing import Any, Callable, Dict, Optional, Tuple, Literal, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

# ----- Type aliases -----
PoolingType = Literal["mean", "max", "add", "sum", "off"]
OptimizerType = Literal["adam", "adamw", "sgd"]
MpnnType = Literal["gcn", "gin", "graphsage", "gine", "sage", "gat", "gatv2", "sgc", "gcnii", "pna", "jknet"]

ModelFactory = Callable[["ExperimentConfig"], nn.Module]
DataloaderFactory = Callable[["ExperimentConfig"], Tuple[DataLoader, DataLoader, Optional[DataLoader]]]
CriterionFactory = Callable[["ExperimentConfig"], nn.Module]
MetricFn = Callable[[np.ndarray, np.ndarray], Dict[str, float]]


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ----- Nested/section dataclasses -----
@dataclass
class SubgraphParam:
    k: int = None
    m: int = None
    pooling: Optional[PoolingType] = None

@dataclass
class SchedulerCfg:
    type: Optional[str] = None
    setp_size: Optional[int] = None
    gamma: Optional[float] = None
    patience: Optional[int] = None

@dataclass
class ModelConfig:
    mpnn_type: MpnnType = "gcn"
    mpnn_layers: Optional[int] = None
    node_feature_dim: Optional[int] = None
    edge_feature_dim: Optional[int] = None
    hidden_dim: Optional[int] = None
    out_dim: Optional[int] = None
    dropout: Optional[float] = None
    pooling: Optional[PoolingType] = None
    temperature: Optional[float] = None
    subgraph_sampling: bool = False
    subgraph_param: SubgraphParam = field(default_factory=SubgraphParam)
    # Extra user-defined knobs
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    # Training hyperparameters
    epochs: int = 100
    train_batch_size: int = 32
    val_batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: OptimizerType = "adam"  # "adam" | "adamw" | "sgd"
    scheduler: Optional[SchedulerCfg] = field(default_factory=SchedulerCfg)  
    train_ratio: float = 0.8
    val_ratio: float = 0.1

    # Objective / metrics
    metric: Optional[str] = None
    loss_fn: Optional[str] = None

    # Numerical / stability
    use_amp: bool = False
    grad_clip: Optional[float] = None  # clip norm value

    # Extra kwargs per block
    dataloader_kwargs: Dict[str, Any] = field(default_factory=dict)
    criterion_kwargs: Dict[str, Any] = field(default_factory=dict)


# ----- Top-level experiment config -----
@dataclass
class ExperimentConfig:
    # Identifiers
    name: str = "exp"
    task: Optional[str] = None
    dataset_name: Optional[str] = None
    model_name: Optional[str] = None

    # Structured sections
    model_config: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # Misc/runtime
    device: str = field(default_factory=_default_device)
    seed: int = 42
    num_workers: int = 4
    output_dir: str = "experiments"  # Base directory for all experiment outputs
    experiment_dir: Optional[str] = None  # Will be set to {output_dir}/{name} in config.py
    log_dir: str = "logs"  # Will be set to {experiment_dir}/logs in config.py
    checkpoint_dir: str = "checkpoints"  # Will be set to {experiment_dir}/checkpoints in config.py
    save_every: int = 1
    keep_last_k: int = 3
    resume_from: Optional[str] = None
    tracker: Optional[str] = "True" # set "False" to turn W&B or Tensorboard off

    # Derived/callables 
    model_fn: Optional[ModelFactory] = None
    dataloader_fn: Optional[DataloaderFactory] = None
    criterion_fn: Optional[CriterionFactory] = None
    metric_fn: Optional[MetricFn] = None

    # Assign sampler
    sampler: str = "graphlet"  # Which sampler to use: "graphlet", "rwr", "ugs", etc.

    # Presampling option (set via CLI --presample flag)
    presample: bool = False  # If True, presample subgraphs once before training for speed

    def parameter_dict(self) -> dict:
        """Add params for TensorBoard hparams."""
        items =[ 
            ('dataset', self.dataset_name),
            ('model_name', self.model_name),
            ('model_config.hidden_dim', self.model_config.hidden_dim),
            ('model_config.layers', self.model_config.mpnn_layers),
            ('model_config.pooling', self.model_config.pooling),
            ('model_config.subgraph_param.k', self.model_config.subgraph_param.k),
            ('model_config.subgraph_param.m', self.model_config.subgraph_param.m),
            ('model_config.temperature', self.model_config.temperature),
            ('model_config.dropout', self.model_config.dropout)
        ]
        return dict(items)

    @classmethod
    def from_sweep(cls, base_config: Optional['ExperimentConfig'] = None):
        """Handle flat dot-notation parameters"""
        try: 
            import wandb
        except ImportError:
            raise ImportError('install `wandb` to use `from_sweep`.')

        config = base_config if base_config is not None else cls()
        sweep_params = dict(wandb.config)
    
        for key, value in sweep_params.items():
            if '.' in key:
                # Handle nested parameter like 'model_config.hidden_dim'
                parts = key.split('.')
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                # Handle top-level parameter
                if hasattr(config, key):
                    setattr(config, key, value)
    
        return config

# ----- Subgraph Batch Features ------

@dataclass
class SubgraphFeaturesBatch:
    """
    Standardized container for subgraph batch features.
    
    This container handles all combinations of:
    - Standard vs subgraph-sampled graphs
    - Node/graph classification vs link prediction
    - With or without edge attributes
    """
    # ============ Core features (always present) ============
    x: torch.Tensor                    # [num_nodes, node_dim] Node features for batch
    edge_index: torch.Tensor           # [2, num_edges] Edge connectivity for batch
    batch: torch.Tensor                # [num_nodes] Graph assignment for each nodes of batch
    
    # ============ Optional standard features ============
    edge_attr: Optional[torch.Tensor] = None  # [num_edges, edge_dim] Edge attributes
    ptr: Optional[torch.Tensor] = None        # [num_graphs+1] Node pointer for batched graphs
    
    # ============ Subgraph sampling features ============
    # Present only when subgraph_sampling=True
    nodes_sampled: Optional[torch.Tensor] = None      # [num_samples, k] Sampled node indices
    edge_index_sampled: Optional[torch.Tensor] = None # [2, num_subg_edges] Sampled edge connectivity
    edge_ptr: Optional[torch.Tensor] = None           # [num_samples,] Sample pointer for edges
    sample_ptr: Optional[torch.Tensor] = None         # [num_graphs+1,] Graph pointer for samples
    edge_src_global: Optional[torch.Tensor] = None    # [num_subg_edges,]Global edge source idx in edge_index
    log_probs: Optional[torch.Tensor] = None          # [num_samples] float32 log sampling probability
    target_nodes: Optional[torch.Tensor] = None        # [T] int64 global target node IDs
    target_ptr: Optional[torch.Tensor] = None          # [num_graphs+1] int64 CSR targets per graph

    # ============ Link prediction features ============
    # Present only for link prediction tasks
    edge_label_index: Optional[torch.Tensor] = None   # [2, num_pred_edges] Edges to predict

    # ============ Graph index for presampling ============
    # Used to track which original graphs are in a batch (for presample cache lookup)
    graph_idx: Optional[torch.Tensor] = None  # [num_graphs] Original dataset indices
    split_id: Optional[torch.Tensor] = None   # [num_graphs] Split identifier (0=train, 1=val, 2=test)

    # ============ Node classification masks ============
    # Used for node classification tasks with train/val/test splits
    train_mask: Optional[torch.Tensor] = None  # [num_nodes] Boolean mask for training nodes
    val_mask: Optional[torch.Tensor] = None    # [num_nodes] Boolean mask for validation nodes
    test_mask: Optional[torch.Tensor] = None   # [num_nodes] Boolean mask for test nodes
    
    def to(self, device: torch.device):
        """Move all tensors to the specified device."""
        for field_name in self.__dataclass_fields__:
            tensor = getattr(self, field_name)
            if tensor is not None and isinstance(tensor, torch.Tensor):
                setattr(self, field_name, tensor.to(device))
        return self

    def __repr__(self):
        """String representation showing variable names and tensor shapes."""
        parts = ["SubgraphFeaturesBatch("]
        for field_name in self.__dataclass_fields__:
            tensor = getattr(self, field_name)
            if tensor is not None:
                if isinstance(tensor, torch.Tensor):
                    parts.append(f"  {field_name}=Tensor{tuple(tensor.shape)}")
                else:
                    parts.append(f"  {field_name}={tensor}")
        parts.append(")")
        return "\n".join(parts)

# ----- Minimal recursive merge helper -----
def merge_into_dataclass(dc, src: Mapping[str, Any] | None):
    """
    Recursively copy values from dict-like `src` into dataclass `dc`,
    preserving defaults for any keys not present in `src`.
    """
    if src is None:
        return dc
    if not isinstance(src, Mapping):
        raise TypeError("merge_into_dataclass expects a Mapping as `src`")

    for f in fields(dc):
        if f.name not in src or src[f.name] is None:
            continue

        incoming = src[f.name]
        current = getattr(dc, f.name)

        if is_dataclass(current) and isinstance(incoming, Mapping):
            merge_into_dataclass(current, incoming)  # mutate nested dc
        else:
            setattr(dc, f.name, incoming)

    return dc
