"""**Metric Builder**
Returns metric function `metric_fn`.
```python
def metric_fn(predict: Tensor(CPU), target: Tensor(CPU)) -> Dict
```
`metric_fn` returns a dict, e.g.`{'AP': 0.345}`
"""
import numpy as np
from .registry import register_metric

@register_metric('AP')
def build_ap():
    from sklearn.metrics import average_precision_score
    metric_fn = average_precision_score
    return _metric_decoretor(metric_fn, 'AP')

@register_metric('ACC')
def build_acc():
    from sklearn.metrics import accuracy_score
    metric_fn = accuracy_score
    return _metric_decoretor(metric_fn, 'ACC')

@register_metric('MAE')
def build_mae():
    from sklearn.metrics import mean_absolute_error
    metric_fn = mean_absolute_error
    return _metric_decoretor(metric_fn, 'MAE')

@register_metric('F1')
def build_f1():
    from sklearn.metrics import f1_score
    # Use macro averaging for multi-class F1 (as used in LRGB COCO-SP)
    def f1_macro(y_true, y_pred):
        return f1_score(y_true, y_pred, average='macro')
    return _metric_decoretor(f1_macro, 'F1')

@register_metric('ROCAUC')
def build_rocauc():
    from sklearn.metrics import roc_auc_score
    def rocauc(y_true, y_pred):
        """
        ROC-AUC for binary classification.
        y_pred can be logits or probabilities - roc_auc_score handles both.
        """
        # Flatten if needed
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        return roc_auc_score(y_true, y_pred)
    return _metric_decoretor(rocauc, 'ROCAUC')

@register_metric('ROCAUC-multilabel')
def build_rocauc_multilabel():
    from sklearn.metrics import roc_auc_score
    def rocauc_multilabel(y_true, y_pred):
        """
        Per-task mean ROCAUC for multi-label node classification (e.g. ogbn-proteins).
        Tasks where y_true has only one class are skipped.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.ndim == 1:
            return roc_auc_score(y_true, y_pred)
        scores = []
        for i in range(y_true.shape[1]):
            col_true = y_true[:, i]
            col_pred = y_pred[:, i]
            if len(np.unique(col_true)) < 2:
                continue
            scores.append(roc_auc_score(col_true, col_pred))
        return float(np.mean(scores)) if scores else 0.0
    return _metric_decoretor(rocauc_multilabel, 'ROCAUC-multilabel')

@register_metric('MRR')
def build_mrr():
    return _metric_decoretor(mean_reciprocal_rank, 'MRR')

def _metric_decoretor(func,name):
    def wrapper(*args, **kwargs):
        score = func(*args, **kwargs)
        return {name: score}
    return wrapper

def mean_reciprocal_rank(all_logits_batches, all_label_batches, target_index_batches):
    '''
    Calculates filtered MRR for the entire dataset 
    Args:
        all_logits: `list(numpy.array)` list of batches of logits for all the target edges.
        all_edge_labels: `list(numpy.array)` list of batches of  true labels for target edes. 
        target_index: `list(numpy.array)` list of batches of target edge index(2d tensor).
        graph_id_batches: `list(numpy.array)` list of graph id for each node of the batch(batch.batch.to_numpy()) 
    Returns:
        mean_reciprocal_rank: `float`
    '''
    print("\ncalculating MRR metric...")
    num_batches = len(all_logits_batches)
    all_mrr = []
    for b in range(num_batches):
        logits = all_logits_batches[b]
        t_edge_labels = all_label_batches[b]
        t_edge_index = target_index_batches[b]

        for head in np.unique(t_edge_index[0]):
            head_mask = t_edge_index[0] == head
            logits_for_head = logits[head_mask]
            labels_for_head = t_edge_labels[head_mask]
            all_mrr += filtered_mrr_single_head(logits_for_head, labels_for_head)
    all_mrr= np.array(all_mrr)
    return np.mean(all_mrr)

def filtered_mrr_single_head(logits, edge_labels)->list:
    """
    Calculate filtered MRR for a single head.
    Args:
        logits: (num_candidates,) - scores for all candidate tails
        edge_labels: (num_candidates,) - labels (0 or 1) for each candidate
    Returns:
        reciprocal_ranks: Reciprocal Ranks for all true edges for this head
    """
    # Find all true tails (where label == 1)
    true_tail_indices = np.where(edge_labels == 1)[0]
    
    reciprocal_ranks = []
    
    if len(true_tail_indices) == 0:
        return reciprocal_ranks  # No true edges

    # Evaluate each true tail separately (filtered setting)
    for target_idx in true_tail_indices:
        # Create filter: exclude OTHER true tails (not current target)
        filter_mask = np.ones(len(logits), dtype=bool)
        
        for other_true_idx in true_tail_indices:
            if other_true_idx != target_idx:
                filter_mask[other_true_idx] = False
        
        # Get filtered candidates
        filtered_logits = logits[filter_mask]
        filtered_indices = np.arange(len(logits))[filter_mask]
        
        # Sort by logits (descending)
        sorted_order = np.argsort(-filtered_logits)
        sorted_indices = filtered_indices[sorted_order]
        
        # Find rank of target
        rank = np.where(sorted_indices == target_idx)[0][0] + 1  # 1-indexed
        reciprocal_ranks.append(1.0 / rank)
    
    # for all true tails for this head
    return reciprocal_ranks

