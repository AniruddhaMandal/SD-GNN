import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional

class ClassifierHead(nn.Module):
    def __init__(self,
                 in_dim: int,
                 num_classes: int,
                 hidden_dim: int = 64,
                 dropout: float =0.1):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.classifier(x)

    
class LinkPredictorHead(nn.Module):
    """
    Link prediction head for GNN node representations.

    Inputs:
        - h:          (N, d) node embeddings/logits from model
        - edge_index: (2, E) tensor of [src; dst] node indices (0-based)

    Returns:
        - logits: (E,) raw scores for each edge (use with BCEWithLogitsLoss)

    score_fn:
        - "dot":      <u_i, v_i>
        - "bilinear": u_i^T W v_i
        - "mlp":      MLP([u_i || v_i || |u_i - v_i| || u_i * v_i]) -> 1
        - "cos":      scaled cosine similarity

    Notes:
        - Works for directed or undirected graphs (pass the target edges).
    """
    def __init__(
        self,
        in_dim: int,
        score_fn: Literal["dot", "bilinear", "mlp", "cos"] = "dot",
        mlp_hidden: int = 128,
        mlp_layers: int = 2,
        cos_scale: float = 10.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.score_fn = score_fn
        self.dropout = nn.Dropout(dropout)

        if score_fn == "bilinear":
            self.W = nn.Parameter(torch.empty(in_dim, in_dim))
            nn.init.xavier_uniform_(self.W)
        elif score_fn == "mlp":
            feat_dim = in_dim * 4  # [u, v, |u-v|, u*v]
            layers = []
            dim = feat_dim
            for _ in range(mlp_layers - 1):
                layers += [nn.Linear(dim, mlp_hidden), nn.ReLU(), nn.Dropout(dropout)]
                dim = mlp_hidden
            layers += [nn.Linear(dim, 1)]
            self.mlp = nn.Sequential(*layers)
        elif score_fn == "dot":
            pass
        elif score_fn == "cos":
            self.scale = cos_scale
        else:
            raise ValueError(f"Unknown score_fn: {score_fn}")

    @staticmethod
    def _gather_pairs(h: torch.Tensor, edge_index: torch.Tensor):
        src, dst = edge_index[0], edge_index[1]
        return h[src], h[dst]

    def _score(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.score_fn == "dot":
            return (u * v).sum(dim=-1)
        elif self.score_fn == "bilinear":
            return (u @ self.W * v).sum(dim=-1)
        elif self.score_fn == "mlp":
            x = torch.cat([u, v, torch.abs(u - v), u * v], dim=-1)
            return self.mlp(x).squeeze(-1)
        elif self.score_fn == "cos":
            u_n = F.normalize(u, dim=-1)
            v_n = F.normalize(v, dim=-1)
            return self.scale * (u_n * v_n).sum(dim=-1)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor):
        h = self.dropout(h)
        u, v = self._gather_pairs(h, edge_index)
        logits = self._score(u, v)
        return logits
