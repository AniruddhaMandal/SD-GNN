"""**Loss/Criterion Builder**:
Rerturns a callable `loss_fn`. 
```python
def loss_fn(predict: Tesnsor, target: Tensor)-> Tensor
```
"""
from .registry import register_loss

@register_loss('BCEWithLogitsLoss')
def build_bcelogit():
    from torch.nn import BCEWithLogitsLoss
    return BCEWithLogitsLoss()

@register_loss("CrossEntropyLoss")
def build_crs_entpy():
    from torch.nn import CrossEntropyLoss
    return CrossEntropyLoss()

@register_loss("L1Loss")
def build_l1loss():
    from torch.nn import L1Loss 
    return L1Loss()

@register_loss("MSELoss")
def build_mseloss():
    from torch.nn import MSELoss
    return MSELoss()


@register_loss("BCEWithLogitsLoss-NaN")
def build_bcelogit_nan():
    """
    BCE-with-logits loss that ignores NaN labels.
    Used for multi-task datasets like Tox21 where some assay labels are missing.
    Computes mean over valid (non-NaN) entries only.
    """
    import torch
    import torch.nn.functional as F

    def nan_masked_bce(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(target)
        if not mask.any():
            return output.sum() * 0.0  # zero loss, differentiable
        return F.binary_cross_entropy_with_logits(
            output[mask], target[mask].float()
        )

    return nan_masked_bce
