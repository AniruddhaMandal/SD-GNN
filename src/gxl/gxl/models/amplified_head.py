"""
Classification heads that AMPLIFY small differences in embeddings.

Problem: Graph embeddings have small differences → classifier struggles
Solution: Normalize + scale to amplify separation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizedClassifierHead(nn.Module):
    """
    Normalize embeddings to unit sphere, then classify.
    This amplifies relative differences by removing magnitude variation.
    """
    def __init__(self, in_dim: int, num_classes: int, scale: float = 10.0,
                 hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.scale = scale

        # MLP that operates on normalized embeddings
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # Normalize to unit sphere
        x_norm = F.normalize(x, p=2, dim=-1)

        # Classify
        logits = self.classifier(x_norm)

        # Scale logits to amplify differences
        logits = logits * self.scale

        return logits


class CosineClassifierHead(nn.Module):
    """
    Cosine similarity classifier with learnable class prototypes.
    Forces embeddings to separate on unit sphere.
    """
    def __init__(self, in_dim: int, num_classes: int, scale: float = 20.0):
        super().__init__()
        self.scale = scale
        self.num_classes = num_classes

        # Learnable class prototypes (normalized)
        self.prototypes = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, x):
        # Normalize embeddings
        x_norm = F.normalize(x, p=2, dim=-1)  # [B, D]

        # Normalize prototypes
        proto_norm = F.normalize(self.prototypes, p=2, dim=-1)  # [C, D]

        # Cosine similarity
        logits = torch.matmul(x_norm, proto_norm.t())  # [B, C]

        # Scale to amplify differences
        logits = logits * self.scale

        return logits


class ContrastiveProjectionHead(nn.Module):
    """
    Project to a space optimized for contrastive separation.
    Uses a bottleneck to force the model to learn discriminative features.
    """
    def __init__(self, in_dim: int, num_classes: int, projection_dim: int = 128,
                 hidden_dim: int = 64, dropout: float = 0.1, scale: float = 10.0):
        super().__init__()
        self.scale = scale

        # Projection to bottleneck (forces compression)
        self.projection = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU()
        )

        # Classification from projected space
        self.classifier = nn.Linear(projection_dim, num_classes)

    def forward(self, x):
        # Project to compressed space
        z = self.projection(x)

        # Normalize projection
        z_norm = F.normalize(z, p=2, dim=-1)

        # Classify
        logits = self.classifier(z_norm)

        # Scale
        logits = logits * self.scale

        return logits


class AdaptiveScaleClassifierHead(nn.Module):
    """
    Learns the optimal scale parameter to amplify differences.
    The scale is learned during training based on gradient feedback.
    """
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 64,
                 dropout: float = 0.1, init_scale: float = 10.0):
        super().__init__()

        # Learnable scale (log-scale for stability)
        self.log_scale = nn.Parameter(torch.tensor([init_scale]).log())

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # Normalize embeddings
        x_norm = F.normalize(x, p=2, dim=-1)

        # Classify
        logits = self.classifier(x_norm)

        # Apply learned scale
        scale = self.log_scale.exp()
        logits = logits * scale

        return logits


class CenterLossClassifierHead(nn.Module):
    """
    Classification head with center loss to pull embeddings toward class centers.
    This explicitly encourages inter-class separation.

    Note: Requires special training loop to compute center loss.
    """
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 64,
                 dropout: float = 0.1, center_loss_weight: float = 0.01):
        super().__init__()
        self.center_loss_weight = center_loss_weight
        self.num_classes = num_classes

        # Class centers (learned)
        self.centers = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.xavier_uniform_(self.centers)

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, labels=None, return_center_loss=False):
        logits = self.classifier(x)

        if return_center_loss and labels is not None:
            # Compute center loss: pull embeddings toward their class center
            batch_size = x.size(0)
            centers_batch = self.centers[labels]  # [B, D]
            center_loss = F.mse_loss(x, centers_batch)

            return logits, center_loss * self.center_loss_weight

        return logits


def build_amplified_head(head_type: str, in_dim: int, num_classes: int,
                        hidden_dim: int = 64, dropout: float = 0.1,
                        scale: float = 10.0):
    """
    Factory function to build different amplified classification heads.

    Args:
        head_type: One of ['normalized', 'cosine', 'contrastive', 'adaptive', 'center']
        in_dim: Input embedding dimension
        num_classes: Number of output classes
        hidden_dim: Hidden dimension for MLP
        dropout: Dropout probability
        scale: Temperature scale for amplification

    Returns:
        Classification head module
    """
    if head_type == 'normalized':
        return NormalizedClassifierHead(in_dim, num_classes, scale, hidden_dim, dropout)
    elif head_type == 'cosine':
        return CosineClassifierHead(in_dim, num_classes, scale)
    elif head_type == 'contrastive':
        return ContrastiveProjectionHead(in_dim, num_classes,
                                        projection_dim=128, hidden_dim=hidden_dim,
                                        dropout=dropout, scale=scale)
    elif head_type == 'adaptive':
        return AdaptiveScaleClassifierHead(in_dim, num_classes, hidden_dim, dropout, scale)
    elif head_type == 'center':
        return CenterLossClassifierHead(in_dim, num_classes, hidden_dim, dropout)
    else:
        raise ValueError(f"Unknown head_type: {head_type}")


if __name__ == "__main__":
    # Test the heads
    batch_size = 4
    in_dim = 64
    num_classes = 10

    x = torch.randn(batch_size, in_dim)

    print("Testing amplified classification heads:\n")

    for head_type in ['normalized', 'cosine', 'contrastive', 'adaptive']:
        head = build_amplified_head(head_type, in_dim, num_classes, scale=10.0)
        logits = head(x)
        print(f"{head_type:15s}: logits shape {logits.shape}, "
              f"mean {logits.mean().item():.4f}, std {logits.std().item():.4f}")
