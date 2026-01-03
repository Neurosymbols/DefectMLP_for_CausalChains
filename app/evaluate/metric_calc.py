from typing import Dict
import numpy as np

# ============================================================================
# PER-CLASS METRICS
# ============================================================================

def calculate_per_class_metrics(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                num_classes: int = 3) -> Dict[str, Dict[str, float]]:
    """
    Calculate precision, recall, F1 for each class
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
    
    Returns:
        Dictionary with metrics per class
    """
    metrics = {}
    labels = list(range(0, num_classes))

    from sklearn.metrics import (
        precision_recall_fscore_support, 
        confusion_matrix
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    for i, c in enumerate(labels):
        metrics[f'class_{c}'] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i]),
            'tp': int(tp[i]),
            'fp': int(fp[i]),
            'fn': int(fn[i]),
            'tn': int(tn[i])
        }
    weighted_precision, weighted_recall, weighted_f1, _= \
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average='weighted',
        zero_division=0
    )
    metrics['weighted_avg'] = {
        'precision': float(weighted_precision),
        'recall': float(weighted_recall),
        'f1_score': float(weighted_f1),
        'support': len(y_true)
    }
    
    # Calculate macro averages
    macro_precision = np.mean([metrics[f'class_{c}']['precision'] for c in range(num_classes)])
    macro_recall = np.mean([metrics[f'class_{c}']['recall'] for c in range(num_classes)])
    macro_f1 = np.mean([metrics[f'class_{c}']['f1_score'] for c in range(num_classes)])
    
    metrics['macro_avg'] = {
        'precision': macro_precision,
        'recall': macro_recall,
        'f1_score': macro_f1,
        'support': len(y_true)
    }
    
    # Overall accuracy
    accuracy = (y_pred == y_true).mean()
    metrics['accuracy'] = accuracy
    
    return metrics

def calculate_mechanism_metrics(y_true: np.ndarray,
                                y_pred: np.ndarray,) -> Dict[str, float]:
    """
    Calculate precision, recall, F1 for mechanism (binary)
    
    Args:
        predictions: Dictionary from get_multitask_predictions()
    
    Returns:
        Dictionary with metrics
    """
    from sklearn.metrics import (
        precision_score, 
        recall_score, 
        f1_score, 
        confusion_matrix,
        accuracy_score)
    
    # All metrics in one go with sklearn
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Get confusion matrix for additional details
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Specificity (not in sklearn, but simple)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Specificity': specificity,
        'True Positives': int(tp),
        'False Positives': int(fp),
        'False Negatives': int(fn),
        'True Negatives': int(tn)
    }
