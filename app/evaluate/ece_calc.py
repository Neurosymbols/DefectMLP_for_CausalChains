import numpy as np
import pandas as pd
from typing import Dict
from sklearn.calibration import calibration_curve

def calculate_ece(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Expected Calibration Error (ECE)
    Lower is better. 0 = perfect calibration.
    """
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    #shape of bin_ids is (N,) wher N is no of rows in y_proba. 
    #np.digitize returns the index of the bin the confidence belongs to.
    #-1 makes the bin_ids zero-index based
    bin_ids = np.digitize(confidences, bins) - 1

    ece = 0.0
    for b in range(n_bins):
        #create a mask to index samples belonging to particular bin
        # accuracies[mask].mean() --> bin accuracy calculated from all samples belonging to a bin
        # confidences[mask].mean() --> mean confidence calculated from all samples belonging to a bin
        # mask.mean() --> no of samples belonging to a bin / total samples (fractional sample share for the bin)
        mask = bin_ids == b
        if mask.any():
            ece += (
                np.abs(accuracies[mask].mean() - confidences[mask].mean())
                * mask.mean()
            )

    return ece

def get_calibration_curve(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """
    Get calibration curve data as DataFrame
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bins: Number of bins
    
    Returns:
        DataFrame with calibration curve data
    """
    # Get predicted class and confidence
    pred_class = np.argmax(y_proba, axis=1)
    confidences = np.max(y_proba, axis=1)
    accuracies = (pred_class == y_true).astype(float)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    calibration_data = []
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        #create a mask to index samples belonging to particular bin
        mask = (confidences > bin_lower) & (confidences <= bin_upper)
        
        if mask.sum() > 0:
            avg_confidence = confidences[mask].mean()
            avg_accuracy = accuracies[mask].mean()
            count = mask.sum()
        else:
            avg_confidence = (bin_lower + bin_upper) / 2
            avg_accuracy = 0.0
            count = 0
        
        calibration_data.append({
            'Bin': f'{bin_lower:.1f}-{bin_upper:.1f}',
            'Avg_Confidence': avg_confidence,
            'Avg_Accuracy': avg_accuracy,
            'Gap': avg_accuracy - avg_confidence,
            'Count': count
        })
    
    return pd.DataFrame(calibration_data)

def calculate_calibration_metrics(
    predictions: Dict[str, np.ndarray],
    n_bins: int = 10
) -> Dict:
    """
    Calculate calibration metrics for multitask outputs
    """

    # ---------- Defect (multiclass) ----------
    defect_ece = calculate_ece(
        predictions["defect_true"],
        predictions["defect_proba"],
        n_bins
    )

    defect_curve = get_calibration_curve(
        predictions["defect_true"],
        predictions["defect_proba"],
        n_bins
    )

    # ---------- Mechanism (binary → 2D probs) ----------
    # to match multiclass interface
    # total probs of the classes should be 1
    mechanism_proba_2d = np.stack(
        [
            1 - predictions["mechanism_proba"],
            predictions["mechanism_proba"]
        ],
        axis=1
    )

    mechanism_ece = calculate_ece(
        predictions["mechanism_true"].astype(int),
        mechanism_proba_2d,
        n_bins
    )

    mechanism_curve = get_calibration_curve(
        predictions["mechanism_true"].astype(int),
        mechanism_proba_2d,
        n_bins
    )

    return {
        "defect_ece": defect_ece,
        "defect_calibration": defect_curve,
        "mechanism_ece": mechanism_ece,
        "mechanism_calibration": mechanism_curve
    }

def print_calibration_analysis(calibration_metrics: Dict):
    print("\n" + "=" * 70)
    print("CALIBRATION ANALYSIS")
    print("=" * 70)

    # Defect
    print("\nDEFECT CALIBRATION")
    print("-" * 70)
    print(f"ECE: {calibration_metrics['defect_ece']:.4f}\n")
    print(calibration_metrics["defect_calibration"]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Mechanism
    print("\n\nMECHANISM CALIBRATION")
    print("-" * 70)
    print(f"ECE: {calibration_metrics['mechanism_ece']:.4f}\n")
    print(calibration_metrics["mechanism_calibration"]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nInterpretation:")
    print("- Gap > 0 → Under-confident")
    print("- Gap < 0 → Over-confident")
    print("- Gap ≈ 0 → Well-calibrated")
    print("=" * 70)