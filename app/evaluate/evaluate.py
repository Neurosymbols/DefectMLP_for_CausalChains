"""
Evaluation and Metrics for PCB Defect Prediction MLP

Implements:
- Confusion matrix visualization
- Per-class metrics (precision, recall, F1-score)
- ROC curves (one-vs-rest)
- Calibration plots
- Performance reports
- Prediction confidence analysis
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import json

from app.evaluate.metric_calc import calculate_per_class_metrics

# ============================================================================
# PREDICTIONS GENERATION
# ============================================================================

def get_predictions(model: nn.Module,
                   dataloader: DataLoader,
                   device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get model predictions, probabilities, and true labels
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader with test/val data
        device: Device to run on
    
    Returns:
        Tuple of (y_true, y_pred, y_proba)
        - y_true: True labels (n_samples,)
        - y_pred: Predicted labels (n_samples,)
        - y_proba: Class probabilities (n_samples, n_classes)
    """
    model.eval()
    model = model.to(device)
    
    all_targets = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            
            # Get predictions
            logits = model(features)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_proba = np.concatenate(all_probs)
    
    return y_true, y_pred, y_proba


# ============================================================================
# CONFUSION MATRIX
# ============================================================================

def compute_confusion_matrix(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            num_classes: int = 3) -> np.ndarray:
    """
    Compute confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
    
    Returns:
        Confusion matrix (num_classes, num_classes)
        Rows = true class, Cols = predicted class
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    
    return cm


def print_confusion_matrix(cm: np.ndarray,
                          class_names: Optional[List[str]] = None):
    """
    Pretty print confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: Optional class names
    """
    num_classes = cm.shape[0]
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    # Print header
    print("\n" + "="*70)
    print("CONFUSION MATRIX")
    print("="*70)
    print("Rows = True Class, Columns = Predicted Class")
    print()
    
    # Column headers
    header = "True \\ Pred".ljust(20)
    for name in class_names:
        header += name[:15].rjust(15)
    header += "Total".rjust(10)
    print(header)
    print("-" * 70)
    
    # Rows
    for i, true_class in enumerate(class_names):
        row = true_class[:20].ljust(20)
        for j in range(num_classes):
            row += f"{cm[i, j]}".rjust(15)
        row += f"{cm[i].sum()}".rjust(10)
        print(row)
    
    # Total row
    total_row = "Total".ljust(20)
    for j in range(num_classes):
        total_row += f"{cm[:, j].sum()}".rjust(15)
    total_row += f"{cm.sum()}".rjust(10)
    print("-" * 70)
    print(total_row)
    print("="*70)
    
    # Normalized version (percentages)
    print("\nNormalized (by true class):")
    print("-" * 70)
    
    header = "True \\ Pred".ljust(20)
    for name in class_names:
        header += name[:15].rjust(15)
    print(header)
    print("-" * 70)
    
    for i, true_class in enumerate(class_names):
        row = true_class[:20].ljust(20)
        total = cm[i].sum()
        for j in range(num_classes):
            pct = (cm[i, j] / total * 100) if total > 0 else 0
            row += f"{pct:.1f}%".rjust(15)
        print(row)
    print("="*70)


def print_classification_report(metrics: Dict,
                               class_names: Optional[List[str]] = None):
    """
    Print classification report (sklearn-style)
    
    Args:
        metrics: Dictionary from calculate_per_class_metrics
        class_names: Optional class names
    """
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    
    # Header
    print(f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    
    # Per-class metrics
    num_classes = len([k for k in metrics.keys() if k.startswith('class_')])
    
    for c in range(num_classes):
        key = f'class_{c}'
        if key in metrics:
            name = class_names[c] if class_names else f"Class {c}"
            m = metrics[key]
            print(f"{name:<25} {m['precision']:>11.4f} {m['recall']:>11.4f} "
                  f"{m['f1_score']:>11.4f} {m['support']:>9d}")
    
    print("-" * 70)
    
    # Macro average
    if 'macro_avg' in metrics:
        m = metrics['macro_avg']
        print(f"{'Macro avg':<25} {m['precision']:>11.4f} {m['recall']:>11.4f} "
              f"{m['f1_score']:>11.4f} {m['support']:>9d}")
    
    # Weighted average
    if 'weighted_avg' in metrics:
        m = metrics['weighted_avg']
        print(f"{'Weighted avg':<25} {m['precision']:>11.4f} {m['recall']:>11.4f} "
              f"{m['f1_score']:>11.4f} {m['support']:>9d}")
    
    print("-" * 70)
    
    # Overall accuracy
    if 'accuracy' in metrics:
        print(f"{'Accuracy':<25} {metrics['accuracy']:>11.4f}")
    
    print("="*70)


# ============================================================================
# ROC CURVE ANALYSIS
# ============================================================================

def calculate_roc_curve(y_true: np.ndarray,
                       y_proba: np.ndarray,
                       class_idx: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate ROC curve for one class (one-vs-rest)
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities (n_samples, n_classes)
        class_idx: Which class to compute ROC for
    
    Returns:
        Tuple of (fpr, tpr, auc)
        - fpr: False positive rates
        - tpr: True positive rates
        - auc: Area under curve
    """
    # Binary labels (class_idx vs rest)
    y_binary = (y_true == class_idx).astype(int)
    y_scores = y_proba[:, class_idx]
    
    # Sort by score
    sorted_indices = np.argsort(y_scores)[::-1]
    y_binary_sorted = y_binary[sorted_indices]
    
    # Calculate TPR and FPR at each threshold
    total_positives = y_binary.sum()
    total_negatives = len(y_binary) - total_positives
    
    tpr_list = [0.0]
    fpr_list = [0.0]
    
    tp = 0
    fp = 0
    
    for i in range(len(y_binary_sorted)):
        if y_binary_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        
        tpr = tp / total_positives if total_positives > 0 else 0
        fpr = fp / total_negatives if total_negatives > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    # Add final point
    tpr_list.append(1.0)
    fpr_list.append(1.0)
    
    tpr = np.array(tpr_list)
    fpr = np.array(fpr_list)
    
    # Calculate AUC using trapezoidal rule
    auc = np.trapz(tpr, fpr)
    
    return fpr, tpr, auc


def print_roc_analysis(y_true: np.ndarray,
                      y_proba: np.ndarray,
                      class_names: Optional[List[str]] = None):
    """
    Print ROC curve analysis for all classes
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        class_names: Optional class names
    """
    num_classes = y_proba.shape[1]
    
    print("\n" + "="*70)
    print("ROC CURVE ANALYSIS (One-vs-Rest)")
    print("="*70)
    
    for c in range(num_classes):
        fpr, tpr, auc = calculate_roc_curve(y_true, y_proba, c)
        
        name = class_names[c] if class_names else f"Class {c}"
        print(f"\n{name}:")
        print(f"  AUC: {auc:.4f}")
        print(f"  ROC points: {len(fpr)}")
        
        # Show some key points
        # At 10% FPR
        idx_10 = np.argmin(np.abs(fpr - 0.1))
        if idx_10 < len(tpr):
            print(f"  TPR at 10% FPR: {tpr[idx_10]:.4f}")
        
        # At 5% FPR
        idx_05 = np.argmin(np.abs(fpr - 0.05))
        if idx_05 < len(tpr):
            print(f"  TPR at 5% FPR: {tpr[idx_05]:.4f}")
    
    print("="*70)


# ============================================================================
# CALIBRATION ANALYSIS
# ============================================================================

def calculate_calibration(y_true: np.ndarray,
                         y_proba: np.ndarray,
                         n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate calibration curve (reliability diagram)
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bins: Number of bins
    
    Returns:
        Tuple of (bin_edges, predicted_probs, true_probs)
    """
    # Get predicted class and its probability
    pred_class = np.argmax(y_proba, axis=1)
    pred_conf = np.max(y_proba, axis=1)
    
    # Check if prediction was correct
    correct = (pred_class == y_true).astype(int)
    
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    predicted_probs = []
    true_probs = []
    
    for i in range(n_bins):
        # Samples in this bin
        in_bin = (pred_conf >= bin_edges[i]) & (pred_conf < bin_edges[i + 1])
        
        if in_bin.sum() > 0:
            # Average predicted probability in bin
            avg_pred = pred_conf[in_bin].mean()
            # Actual accuracy in bin
            avg_true = correct[in_bin].mean()
            
            predicted_probs.append(avg_pred)
            true_probs.append(avg_true)
        else:
            predicted_probs.append(bin_centers[i])
            true_probs.append(0.0)
    
    return bin_centers, np.array(predicted_probs), np.array(true_probs)


def calculate_ece(y_true: np.ndarray,
                 y_proba: np.ndarray,
                 n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE)
    
    Lower is better. ECE = 0 means perfect calibration.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bins: Number of bins
    
    Returns:
        ECE value
    """
    pred_class = np.argmax(y_proba, axis=1)
    pred_conf = np.max(y_proba, axis=1)
    correct = (pred_class == y_true).astype(int)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        in_bin = (pred_conf >= bin_edges[i]) & (pred_conf < bin_edges[i + 1])
        
        if in_bin.sum() > 0:
            bin_acc = correct[in_bin].mean()
            bin_conf = pred_conf[in_bin].mean()
            bin_weight = in_bin.sum() / total_samples
            # “On average, how far is confidence from reality?”
            ece += bin_weight * np.abs(bin_acc - bin_conf)
    
    return ece


def print_calibration_analysis(y_true: np.ndarray,
                              y_proba: np.ndarray,
                              n_bins: int = 10):
    """
    Print calibration analysis
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bins: Number of bins
    """
    print("\n" + "="*70)
    print("CALIBRATION ANALYSIS")
    print("="*70)
    
    # Calculate ECE
    ece = calculate_ece(y_true, y_proba, n_bins)
    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
    print("(Lower is better, 0 = perfect calibration)")
    
    # Calibration curve
    bin_centers, pred_probs, true_probs = calculate_calibration(y_true, y_proba, n_bins)
    
    print(f"\nCalibration Curve ({n_bins} bins):")
    print(f"{'Predicted Prob':<20} {'Actual Accuracy':<20} {'Gap':<15}")
    print("-" * 70)
    
    for pred, true in zip(pred_probs, true_probs):
        gap = true - pred
        print(f"{pred:>19.3f} {true:>19.3f} {gap:>14.3f}")
    
    print("="*70)
    print("\nInterpretation:")
    print("  - Well calibrated: Predicted prob ≈ Actual accuracy")
    print("  - Over-confident: Predicted prob > Actual accuracy (negative gap)")
    print("  - Under-confident: Predicted prob < Actual accuracy (positive gap)")


# ============================================================================
# PREDICTION CONFIDENCE ANALYSIS
# ============================================================================

def analyze_prediction_confidence(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 y_proba: np.ndarray,
                                 class_names: Optional[List[str]] = None):
    """
    Analyze prediction confidence distribution
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        class_names: Optional class names
    """
    print("\n" + "="*70)
    print("PREDICTION CONFIDENCE ANALYSIS")
    print("="*70)
    
    # Get max probability for each prediction
    max_probs = np.max(y_proba, axis=1)
    
    # Overall confidence stats
    print(f"\nOverall Confidence Statistics:")
    print(f"  Mean:   {max_probs.mean():.4f}")
    print(f"  Median: {np.median(max_probs):.4f}")
    print(f"  Min:    {max_probs.min():.4f}")
    print(f"  Max:    {max_probs.max():.4f}")
    print(f"  Std:    {max_probs.std():.4f}")
    
    # Confidence for correct vs incorrect
    correct_mask = (y_pred == y_true)
    incorrect_mask = ~correct_mask
    
    print(f"\nConfidence for Correct Predictions:")
    print(f"  Mean: {max_probs[correct_mask].mean():.4f}")
    print(f"  Min:  {max_probs[correct_mask].min():.4f}")
    
    if incorrect_mask.sum() > 0:
        print(f"\nConfidence for Incorrect Predictions:")
        print(f"  Mean: {max_probs[incorrect_mask].mean():.4f}")
        print(f"  Max:  {max_probs[incorrect_mask].max():.4f}")
    
    # Per-class confidence
    num_classes = y_proba.shape[1]
    print(f"\nPer-Class Confidence (when predicted):")
    print(f"{'Class':<25} {'Mean Conf':<15} {'Count':<10}")
    print("-" * 70)
    
    for c in range(num_classes):
        mask = (y_pred == c)
        if mask.sum() > 0:
            name = class_names[c] if class_names else f"Class {c}"
            mean_conf = max_probs[mask].mean()
            count = mask.sum()
            print(f"{name:<25} {mean_conf:>14.4f} {count:>9d}")
    
    # Confidence bins
    print(f"\nConfidence Distribution:")
    bins = [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
    bin_labels = ["<0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-0.95", "0.95-1.0"]
    
    print(f"{'Confidence Range':<20} {'Count':<10} {'Accuracy':<15}")
    print("-" * 70)
    
    for i in range(len(bins)):
        if i == 0:
            mask = max_probs < bins[i]
        elif i == len(bins) - 1:
            mask = max_probs >= bins[i-1]
        else:
            mask = (max_probs >= bins[i-1]) & (max_probs < bins[i])
        
        if mask.sum() > 0:
            count = mask.sum()
            acc = (y_pred[mask] == y_true[mask]).mean()
            print(f"{bin_labels[i]:<20} {count:>9d} {acc:>14.4f}")
    
    print("="*70)


# ============================================================================
# COMPLETE EVALUATION
# ============================================================================

def evaluate_model(model: nn.Module,
                  dataloader: DataLoader,
                  class_names: Optional[List[str]] = None,
                  device: str = 'cpu',
                  save_path: Optional[str] = None) -> Dict:
    """
    Complete model evaluation
    
    Args:
        model: Trained model
        dataloader: Test DataLoader
        class_names: Class names for reporting
        device: Device
        save_path: Optional path to save results
    
    Returns:
        Dictionary with all evaluation metrics
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    # Get predictions
    print("\nGenerating predictions...")
    y_true, y_pred, y_proba = get_predictions(model, dataloader, device)
    
    num_classes = y_proba.shape[1]
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    # Confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    print_confusion_matrix(cm, class_names)
    
    # Per-class metrics
    metrics = calculate_per_class_metrics(y_true, y_pred, num_classes)
    print_classification_report(metrics, class_names)
    
    # ROC analysis
    print_roc_analysis(y_true, y_proba, class_names)
    
    # Calibration
    print_calibration_analysis(y_true, y_proba)
    
    # Confidence analysis
    analyze_prediction_confidence(y_true, y_pred, y_proba, class_names)
    
    # Compile results
    results = {
        'confusion_matrix': cm.tolist(),
        'metrics': metrics,
        'ece': calculate_ece(y_true, y_proba),
        'class_names': class_names,
        'n_samples': len(y_true)
    }
    
    # Save if requested
    if save_path:
        with open(save_path, 'w') as f:
            # Convert numpy types for JSON
            json_results = {
                'confusion_matrix': cm.tolist(),
                'metrics': {
                    k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                        for kk, vv in v.items()} if isinstance(v, dict) else float(v)
                    for k, v in metrics.items()
                },
                'ece': float(results['ece']),
                'class_names': class_names,
                'n_samples': int(len(y_true))
            }
            json.dump(json_results, f, indent=2)
        print(f"\n✓ Saved evaluation results to {save_path}")
    
    return results
