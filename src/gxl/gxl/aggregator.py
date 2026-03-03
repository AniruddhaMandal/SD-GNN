import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, to_dense_batch
from .registry import register_aggregator
import warnings

# Suppress torch-scatter installation warning
warnings.filterwarnings('ignore', message='.*torch.?scatter.*')

@register_aggregator('attention')
class AttentionAggregator(nn.Module):
    def __init__(self, hidden_dim, temperature=0.2):
        super().__init__()
        self.temperature = temperature
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, subgraph_embeddings, batch):
        """
        Args:
            subgraph_embeddings: [total_subgraphs, hidden_dim]
            batch: [total_subgraphs] - which graph each subgraph belongs to
        Returns:
            graph_embeddings: [num_graphs, hidden_dim]
        """
        num_graphs = batch.max().item() + 1
        scores = self.attention_mlp(subgraph_embeddings)  # [total_subgraphs, 1]
        scores = scores / self.temperature

        max_scores = scatter(scores, batch, dim=0, dim_size=num_graphs, reduce='max')  # [num_graphs, 1]
        max_scores = max_scores[batch]  # Broadcast back
        scores_exp = torch.exp(scores - max_scores)
        scores_sum = scatter(scores_exp, batch, dim=0, dim_size=num_graphs, reduce='sum')  # [num_graphs, 1]
        scores_sum = scores_sum[batch] 
        attention_weights = scores_exp / (scores_sum + 1e-8)  # [total_subgraphs, 1]
        
        weighted_embeddings = attention_weights * subgraph_embeddings  # [total_subgraphs, hidden_dim]
        graph_embeddings = scatter(weighted_embeddings, batch, dim=0, dim_size=num_graphs, reduce='sum')

        return graph_embeddings


@register_aggregator('weighted_mean')
class WeightedMeanAggregator(nn.Module):
    """Inverse-probability weighted mean aggregator.

    Uses log-probabilities from the sampler to compute normalized weights:
        w_i = exp(-log_p_i) / sum_j exp(-log_p_j)     (per graph)
        result = sum(h_i * w_i)

    Weights are normalized before multiplying with embeddings to avoid
    overflow when probabilities are very small (exp(-log_p) → large).
    The per-graph normalization is computed as a softmax over -log_p
    with max-subtraction for numerical stability.
    """
    needs_log_probs = True

    def __init__(self, hidden_dim, temperature=None):
        super().__init__()

    def forward(self, subgraph_embeddings, batch, log_probs=None):
        """
        Args:
            subgraph_embeddings: [total_subgraphs, hidden_dim]
            batch: [total_subgraphs] - which graph each subgraph belongs to
            log_probs: [total_subgraphs] - log sampling probabilities
        Returns:
            graph_embeddings: [num_graphs, hidden_dim]
        """
        from torch_geometric.nn import global_mean_pool

        if log_probs is None:
            return global_mean_pool(subgraph_embeddings, batch)

        num_graphs = batch.max().item() + 1

        # Mask degenerate samples (log_p = -inf)
        valid = torch.isfinite(log_probs)
        neg_log_p = -log_probs                                    # [B]
        neg_log_p = neg_log_p.masked_fill(~valid, float('-inf'))  # degenerate → -inf after negation

        # Numerically stable per-graph softmax: subtract max per graph first.
        # If ALL samples in a group are degenerate (log_p=-inf → neg_log_p=-inf),
        # max_per_graph = -inf and -inf - (-inf) = NaN. Clamp to 0 so those
        # groups get shifted=-inf → exp=0 → zero embedding instead of NaN.
        max_per_graph = scatter(neg_log_p, batch, dim=0,
                                dim_size=num_graphs, reduce='max')  # [G]
        max_per_graph = torch.where(torch.isfinite(max_per_graph),
                                    max_per_graph,
                                    torch.zeros_like(max_per_graph))
        shifted = neg_log_p - max_per_graph[batch]                  # [B]
        exp_shifted = torch.exp(shifted) * valid.float()            # [B]

        sum_exp = scatter(exp_shifted, batch, dim=0,
                          dim_size=num_graphs, reduce='sum')        # [G]
        normalized_w = exp_shifted / sum_exp[batch].clamp(min=1e-8) # [B]

        weighted_x = subgraph_embeddings * normalized_w.unsqueeze(-1)  # [B, H]
        return scatter(weighted_x, batch, dim=0,
                       dim_size=num_graphs, reduce='sum')              # [G, H]


@register_aggregator('transformer')
class TransformerAggregator(nn.Module):
    """Self-attention aggregator with CLS token and optional log-prob bias.

    Pads variable-length subgraph sets into dense batches, prepends a learnable
    CLS token, runs a TransformerEncoder, and returns the CLS output as the
    aggregated group embedding.

    An optional log-probability attention bias adds ``alpha * (-log_p_j)`` to
    the CLS row of the attention mask so that rarer subgraphs receive more
    attention.  ``alpha`` is a learnable scalar initialised to 0 so the model
    starts with pure content-based attention and learns whether to incorporate
    sampling probabilities during training.
    """
    needs_log_probs = True

    def __init__(self, hidden_dim, temperature=None,
                 num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Learnable CLS token [1, 1, H]
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Learnable scalar for log-prob bias strength (init=0 -> no bias at start)
        self.log_prob_alpha = nn.Parameter(torch.tensor(0.0))

        # TransformerEncoder with Pre-LN for stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    # ------------------------------------------------------------------
    def _build_attn_mask(self, valid_mask, log_probs, dense_lp):
        """Build a combined float attention mask [G*num_heads, S, S].

        Returns ``None`` when no masking is needed (no padding, no log_probs)
        for the most efficient path.
        """
        G, max_len = valid_mask.shape
        S = 1 + max_len  # CLS + subgraphs
        has_padding = not valid_mask.all()
        has_lp = log_probs is not None

        if not has_padding and not has_lp:
            return None

        # Start from zeros [G, S, S]
        mask = valid_mask.new_zeros(G, S, S, dtype=torch.float32)

        # -- Padding mask: -inf for padded positions -----------------------
        if has_padding:
            # cls_and_valid: True for CLS (col 0) and valid subgraph cols
            # [G, S] — column validity
            col_valid = torch.cat(
                [valid_mask.new_ones(G, 1), valid_mask], dim=1,
            )  # [G, S]
            row_valid = col_valid  # same: CLS row always valid

            # Invalid columns get -inf in every row
            mask = mask.masked_fill(~col_valid.unsqueeze(1), float('-inf'))
            # Invalid rows get -inf in every column
            mask = mask.masked_fill(~row_valid.unsqueeze(2), float('-inf'))

        # -- Log-prob bias on CLS row (columns 1..S) -----------------------
        if has_lp:
            # dense_lp: [G, max_len], may contain -inf for degenerate/padded
            neg_lp = -dense_lp  # [G, max_len]
            # Clamp degenerate (+inf after negation) to 0 -> no bias
            neg_lp = torch.where(torch.isfinite(neg_lp), neg_lp, torch.zeros_like(neg_lp))
            alpha = self.log_prob_alpha
            bias = alpha * neg_lp  # [G, max_len]
            # Add bias to CLS row, subgraph columns (1..S)
            mask[:, 0, 1:] = mask[:, 0, 1:] + bias

        # Expand to [G*num_heads, S, S] for nn.TransformerEncoder
        mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        mask = mask.reshape(G * self.num_heads, S, S)
        return mask

    # ------------------------------------------------------------------
    def forward(self, subgraph_embeddings, batch, log_probs=None):
        """
        Args:
            subgraph_embeddings: [B_total, H] flat subgraph embeddings
            batch: [B_total] group index for each subgraph
            log_probs: [B_total] optional log sampling probabilities
        Returns:
            group_embeddings: [G, H]
        """
        # 1. Dense batch: [G, max_len, H] + valid_mask [G, max_len]
        dense_x, valid_mask = to_dense_batch(subgraph_embeddings, batch)
        G, max_len, H = dense_x.shape

        # Densify log_probs if present: [G, max_len]
        dense_lp = None
        if log_probs is not None:
            dense_lp, _ = to_dense_batch(
                log_probs.unsqueeze(-1), batch, fill_value=float('-inf'),
            )
            dense_lp = dense_lp.squeeze(-1)  # [G, max_len]

        # 2. Prepend CLS token: [G, 1+max_len, H]
        cls_tokens = self.cls_token.expand(G, -1, -1)  # [G, 1, H]
        seq = torch.cat([cls_tokens, dense_x], dim=1)   # [G, S, H]

        # 3. Build combined attention mask
        attn_mask = self._build_attn_mask(valid_mask, log_probs, dense_lp)

        # 4. Transformer
        out = self.transformer(seq, mask=attn_mask)  # [G, S, H]

        # 5. Extract CLS output and normalise
        cls_out = out[:, 0, :]        # [G, H]
        cls_out = self.out_norm(cls_out)
        return cls_out