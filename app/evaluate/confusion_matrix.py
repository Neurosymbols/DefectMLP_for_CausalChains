from typing import Dict
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np

def print_defect_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int
):
    """
    Print defect confusion matrix using pandas
    
    Args:
        cm: Confusion matrix (3, 3)
    """

    labels = list(range(0, num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print("\n" + "="*70)
    print("DEFECT CONFUSION MATRIX")
    print("="*70)
    print("Rows = True Class, Columns = Predicted Class\n")
    
    # Create DataFrame
    labels = ['No Defect', 'Open Circuit', 'Solder Bridge']
    df = pd.DataFrame(cm, index=labels, columns=labels)
    
    # Add totals
    df['Total'] = df.sum(axis=1)
    df.loc['Total'] = df.sum(axis=0)
    
    print(df)

    # Normalized (percentages)
    print("\n" + "Normalized by Row (%):")
    print("-" * 70)
    cm_normalized = cm.astype(float)
    for i in range(3):
        row_sum = cm[i].sum()
        if row_sum > 0:
            cm_normalized[i] = cm[i] / row_sum * 100
    
    df_norm = pd.DataFrame(cm_normalized, index=labels, columns=labels)
    print(df_norm.round(1))
    
    print("="*70)
    return cm

def print_mechanism_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int
):
    """
    Print mechanism confusion matrix using pandas
    
    Args:
        cm: Confusion matrix (2, 2)
    """
    labels = list(range(0, num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print("\n" + "="*70)
    print("MECHANISM CONFUSION MATRIX (poorpastetransfer)")
    print("="*70)
    print("Rows = True, Columns = Predicted\n")
    
    # Create DataFrame
    labels = ['Absent (0)', 'Present (1)']
    df = pd.DataFrame(cm, index=labels, columns=labels)
    
    # Add totals
    df['Total'] = df.sum(axis=1)
    df.loc['Total'] = df.sum(axis=0)
    
    print(df)

    # Normalized
    print("\n" + "Normalized by Row (%):")
    print("-" * 70)
    cm_normalized = cm.astype(float)
    for i in range(2):
        row_sum = cm[i].sum()
        if row_sum > 0:
            cm_normalized[i] = cm[i] / row_sum * 100
    
    df_norm = pd.DataFrame(cm_normalized, index=labels, columns=labels)
    print(df_norm.round(1))
    
    print("="*70)
    return cm

def calculate_confusion_matrices(predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Calculate confusion matrices for both tasks
    
    Args:
        predictions: Dictionary from get_multitask_predictions()
    
    Returns:
        Dictionary with confusion matrices
    """
    defect_cm = print_defect_confusion_matrix(
        predictions['defect_true'],
        predictions['defect_pred'],
        num_classes=3
    )
    
    mechanism_cm = print_mechanism_confusion_matrix(
        predictions['mechanism_true'],
        predictions['mechanism_pred'],
        num_cases=2
    )
    
    return {
        'defect_cm': defect_cm,
        'mechanism_cm': mechanism_cm
    }