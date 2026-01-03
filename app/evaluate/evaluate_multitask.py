"""
Multi-Task Evaluation for PCB Defect + Mechanism Prediction

Step-by-step build:
1. Get predictions (defect + mechanism) ✓
2. Overall accuracy & F1 metrics ✓
3. Confusion matrices (next)
4. Per-class metrics (next)
5. ECE & calibration (next)
6. ROC curves (next)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import json

from app.evaluate.metric_calc import calculate_per_class_metrics, calculate_mechanism_metrics
from app.evaluate.confusion_matrix import (
    print_defect_confusion_matrix, 
    print_mechanism_confusion_matrix
)
from app.evaluate.ece_calc import (
    calculate_calibration_metrics,
    print_calibration_analysis
)


# ============================================================================
# STEP 1: GET PREDICTIONS (MULTI-TASK)
# ============================================================================

def get_multitask_predictions(model: nn.Module,
                              dataloader: DataLoader,
                              device: str = 'cpu') -> Dict[str, np.ndarray]:
    """
    Get model predictions for both tasks
    
    Args:
        model: Trained multi-task model
        dataloader: DataLoader with test data
        device: Device to run on
    
    Returns:
        Dictionary containing:
        - defect_true: True defect labels (n_samples,)
        - defect_pred: Predicted defect labels (n_samples,)
        - defect_proba: Defect probabilities (n_samples, 3)
        - mechanism_true: True mechanism labels (n_samples,)
        - mechanism_pred: Predicted mechanism labels (n_samples,)
        - mechanism_proba: Mechanism probabilities (n_samples,)
    """
    model.eval()
    model = model.to(device)

    # Storage for predictions
    all_defect_true = []
    all_defect_pred = []
    all_defect_proba = []
    all_mechanism_true = []
    all_mechanism_pred = []
    all_mechanism_proba = []

    print("Generating predictions...")
    
    with torch.no_grad():
        for features, defect_targets, mechanism_targets in dataloader:
            features = features.to(device)
            
            # Forward pass (multi-task)
            defect_logits, mechanism_logits = model(features)
            
            # Defect predictions
            defect_probs = torch.softmax(defect_logits, dim=1)
            defect_preds = torch.argmax(defect_probs, dim=1)
            
            # Mechanism predictions
            mechanism_probs = torch.sigmoid(mechanism_logits).squeeze()
            mechanism_preds = (mechanism_probs > 0.5).float()
            
            # Store
            all_defect_true.append(defect_targets.cpu().numpy())
            all_defect_pred.append(defect_preds.cpu().numpy())
            all_defect_proba.append(defect_probs.cpu().numpy())
            all_mechanism_true.append(mechanism_targets.cpu().numpy())
            all_mechanism_pred.append(mechanism_preds.cpu().numpy())
            all_mechanism_proba.append(mechanism_probs.cpu().numpy())
    
    # Concatenate all batches
    predictions = {
        'defect_true': np.concatenate(all_defect_true),
        'defect_pred': np.concatenate(all_defect_pred),
        'defect_proba': np.concatenate(all_defect_proba),
        'mechanism_true': np.concatenate(all_mechanism_true),
        'mechanism_pred': np.concatenate(all_mechanism_pred),
        'mechanism_proba': np.concatenate(all_mechanism_proba)
    }
    
    print(f"✓ Generated predictions for {len(predictions['defect_true'])} samples")
    
    return predictions

def calculate_overall_metrics(predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Calculate overall accuracy and F1 for both tasks
    
    Args:
        predictions: Dictionary from get_multitask_predictions()
    
    Returns:
        Dictionary with overall metrics
    """
    # Defect metrics
    defect_metrics = calculate_per_class_metrics(
        predictions['defect_true'],
        predictions['defect_pred'],
        num_classes=3
    )
    defect_accuracy = defect_metrics['accuracy']
    
    defect_f1 = defect_metrics['weighted_avg']['f1_score']
    
    # Mechanism metrics
    mech_metrics = calculate_mechanism_metrics(
        predictions['mechanism_true'],
        predictions['mechanism_pred']
    )
    
    metrics = {
        'defect_accuracy': float(defect_accuracy),
        'defect_f1': float(defect_f1),
        'mechanism_accuracy': float(mech_metrics['Accuracy']),
        'mechanism_f1': float(mech_metrics['F1-Score'])
    }
    
    return metrics

def print_overall_metrics(metrics: Dict[str, float]):
    """
    Pretty print overall metrics
    
    Args:
        metrics: Dictionary from calculate_overall_metrics()
    """
    print("\n" + "="*70)
    print("OVERALL METRICS")
    print("="*70)
    
    print("\nDEFECT CLASSIFICATION:")
    print(f"  Accuracy: {metrics['defect_accuracy']:.4f} ({metrics['defect_accuracy']*100:.2f}%)")
    print(f"  F1-Score: {metrics['defect_f1']:.4f}")
    
    print("\nMECHANISM PREDICTION:")
    print(f"  Accuracy: {metrics['mechanism_accuracy']:.4f} ({metrics['mechanism_accuracy']*100:.2f}%)")
    print(f"  F1-Score: {metrics['mechanism_f1']:.4f}")
    
    print("="*70)

# ============================================================================
# SIMPLE EVALUATION (Steps 1-2 only)
# ============================================================================

def evaluate_multitask_simple(model: nn.Module,
                              dataloader: DataLoader,
                              device: str = 'cpu') -> Dict:
    """
    Simple evaluation: predictions + overall metrics
    
    This is Step 6.1 - just the basics
    
    Args:
        model: Trained multi-task model
        dataloader: Test DataLoader
        device: Device
    
    Returns:
        Dictionary with predictions and metrics
    """
    print("\n" + "="*70)
    print("MULTI-TASK MODEL EVALUATION (Simple)")
    print("="*70)
    
    # Step 1: Get predictions
    predictions = get_multitask_predictions(model, dataloader, device)
    
    # Step 2: Calculate overall metrics
    metrics = calculate_overall_metrics(predictions)

    # Step 3: Compute defect confusion matrix
    defect_cm = print_defect_confusion_matrix(
        predictions['defect_true'],
        predictions['defect_pred'],
        3
    )

    #Step 4: Compute mechanism confusion matrix
    mechanism_cm = print_mechanism_confusion_matrix(
        predictions['mechanism_true'],
        predictions['mechanism_pred'],
        2
    )

    #Step 5: Calculate calibration metrics
    cb_metrics = calculate_calibration_metrics(predictions)
    
    # Print results
    print_overall_metrics(metrics)
    print_calibration_analysis(cb_metrics)
    
    return {
        'predictions': predictions,
        'overall_metrics': metrics
    }

# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_evaluation_results(results: Dict, filepath: str = 'evaluation_results.json'):
    """
    Save evaluation results to JSON file
    
    Args:
        results: Dictionary from evaluate_multitask_simple()
        filepath: Path to save
    """
    # Prepare data for JSON (convert numpy/pandas to standard types)
    json_results = {
        'overall_metrics': results['overall_metrics'],
        'confusion_matrices': {
            'defect': results['confusion_matrices']['defect_cm'].tolist(),
            'mechanism': results['confusion_matrices']['mechanism_cm'].tolist()
        },
        'defect_per_class': results['defect_per_class'].to_dict('records'),
        'mechanism_metrics': results['mechanism_metrics'],
        'calibration': {
            'defect_ece': float(results['calibration_metrics']['defect_ece']),
            'mechanism_ece': float(results['calibration_metrics']['mechanism_ece']),
            'defect_calibration': results['calibration_metrics']['defect_calibration'].to_dict('records'),
            'mechanism_calibration': results['calibration_metrics']['mechanism_calibration'].to_dict('records')
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✓ Saved evaluation results to {filepath}")