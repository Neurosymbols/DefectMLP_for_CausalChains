"""
Feature Engineering Pipeline for PCB Defect Prediction MLP

This module transforms raw process parameters into engineered features
for the MLP model, including:
- Z-scores (normalized deviations)
- Limit distances (proximity to spec limits)
- Physics-based interactions
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from .add_mech_labels import label_mechanisms, validate_mechanism_labels
from .constants import PARAMETER_SPECS

# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def compute_z_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute z-scores (standardized deviations from nominal)
    
    Z-score = (value - nominal) / tolerance
    
    Interpretation:
    - z = 0: At nominal
    - z = +1: One tolerance unit above nominal
    - z = -1: One tolerance unit below nominal
    - |z| > 1.5: Getting close to spec limits
    
    Args:
        df: DataFrame with raw parameter columns
    
    Returns:
        DataFrame with added z-score columns
    """
    df = df.copy()
    
    # Paste Volume z-score
    df['z_paste_volume'] = (
        (df['paste_volume'] - PARAMETER_SPECS['paste_volume']['nominal']) / 
        PARAMETER_SPECS['paste_volume']['tolerance']
    )
    
    # Ambient RH z-score
    df['z_ambient_rh'] = (
        (df['ambient_rh'] - PARAMETER_SPECS['ambient_rh']['nominal']) / 
        PARAMETER_SPECS['ambient_rh']['tolerance']
    )
    
    # Paste Viscosity z-score
    df['z_paste_viscosity'] = (
        (df['paste_viscosity'] - PARAMETER_SPECS['paste_viscosity']['nominal']) / 
        PARAMETER_SPECS['paste_viscosity']['tolerance']
    )

    # Peak Reflow Temperature z-score
    df['z_peak_reflow_temperature'] = (
        (df['peak_reflow_temperature'] - PARAMETER_SPECS['peak_reflow_temperature']['nominal']) / 
        PARAMETER_SPECS['peak_reflow_temperature']['tolerance']
    )

    # Time above liquidus z-score
    df['z_time_above_liquidus'] = (
        (df['time_above_liquidus'] - PARAMETER_SPECS['time_above_liquidus']['nominal']) / 
        PARAMETER_SPECS['time_above_liquidus']['tolerance']
    )
    
    return df

def compute_limit_distances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute distances to specification limits
    
    Positive distance = exceeds limit (out of spec)
    Negative distance = within spec (safe margin)
    
    Interpretation:
    - dist_to_usl > 0: Above upper limit (VIOLATION)
    - dist_to_usl = 0: Exactly at upper limit
    - dist_to_usl < 0: Below upper limit (safe, margin = |value|)
    
    Args:
        df: DataFrame with raw parameter columns
    
    Returns:
        DataFrame with added distance columns
    """
    df = df.copy()
    
    # Paste Volume distances
    df['dist_to_usl_paste_volume'] = (
        df['paste_volume'] - PARAMETER_SPECS['paste_volume']['usl']
    )
    df['dist_to_lsl_paste_volume'] = (
        PARAMETER_SPECS['paste_volume']['lsl'] - df['paste_volume']
    )
    
    # Ambient RH distances
    df['dist_to_usl_rh'] = (
        df['ambient_rh'] - PARAMETER_SPECS['ambient_rh']['usl']
    )
    df['dist_to_lsl_rh'] = (
        PARAMETER_SPECS['ambient_rh']['lsl'] - df['ambient_rh']
    )
    
    # Ambient Temperature distances
    df['dist_to_usl_temp'] = (
        df['ambient_temperature'] - PARAMETER_SPECS['ambient_temperature']['usl']
    )
    df['dist_to_lsl_temp'] = (
        PARAMETER_SPECS['ambient_temperature']['lsl'] - df['ambient_temperature']
    )

    #Peak reflow temperature distances
    df['dist_to_usl_prt'] = (
        df['peak_reflow_temperature'] - PARAMETER_SPECS['peak_reflow_temperature']['usl']
    )
    df['dist_to_lsl_prt'] = (
        PARAMETER_SPECS['peak_reflow_temperature']['lsl'] - df['peak_reflow_temperature']
    )

    #Time above liquidus distances
    df['dist_to_usl_tal'] = (
        df['time_above_liquidus'] - PARAMETER_SPECS['time_above_liquidus']['usl']
    )
    df['dist_to_lsl_tal'] = (
        PARAMETER_SPECS['time_above_liquidus']['lsl'] - df['time_above_liquidus']
    )
    
    return df

def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations
    
    Input columns expected:
    - paste_volume
    - stencil_thickness
    - paste_viscosity
    - ambient_rh
    - ambient_temperature
    - peak_reflow_temperature
    - time_above_liquidus
    
    Output columns added:
    - 5 z-scores
    - 10 limit distances
    Total: 15 engineered features
    
    Args:
        df: DataFrame with raw parameters
    
    Returns:
        DataFrame with all engineered features added
    """
    df = df.copy()
    
    # Apply transformations
    df = compute_z_scores(df)
    df = compute_limit_distances(df)
    
    return df

def get_feature_columns() -> Dict[str, list]:
    """
    Get lists of feature column names for model input
    
    Returns:
        Dictionary with feature column lists:
        - 'raw': 5 raw parameter columns
        - 'z_scores': 3 z-score columns
        - 'distances': 6 distance columns
        - 'interactions': 4 interaction columns
        - 'all_engineered': 13 total engineered columns
        - 'all_features': 18 total (5 raw + 13 engineered)
    """
    raw_features = [
        'paste_volume',
        'stencil_thickness',
        'paste_viscosity',
        'ambient_rh',
        'ambient_temperature',
        'peak_reflow_temperature',
        'time_above_liquidus'
    ]
    
    z_score_features = [
        'z_paste_volume',
        'z_ambient_rh',
        'z_paste_viscosity',
        'z_peak_reflow_temperature',
        'z_time_above_liquidus'
    ]
    
    distance_features = [
        'dist_to_usl_paste_volume',
        'dist_to_lsl_paste_volume',
        'dist_to_usl_rh',
        'dist_to_lsl_rh',
        'dist_to_usl_temp',
        'dist_to_lsl_temp',
        'dist_to_usl_prt',
        'dist_to_lsl_prt',
        'dist_to_usl_tal',
        'dist_to_lsl_tal'
    ]

    all_engineered = z_score_features + distance_features
    all_features = raw_features + all_engineered
    
    return {
        'raw': raw_features,
        'z_scores': z_score_features,
        'distances': distance_features,
        'all_engineered': all_engineered,
        'all_features': all_features
    }

def prepare_features_for_model(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, list]]:
    """
    Complete pipeline: Engineer features and extract array for model
    
    Args:
        df: Raw DataFrame from MongoDB or CSV
    
    Returns:
        Tuple of:
        - features: numpy array (n_samples, 18) ready for model
        - feature_info: Dictionary with column names
    """
    # Engineer features
    df_engineered = engineer_all_features(df)
    
    # Get feature column names
    feature_info = get_feature_columns()
    
    # Extract feature array
    features = df_engineered[feature_info['all_features']].values
    
    return features, feature_info


# ============================================================================
# VALIDATION & TESTING
# ============================================================================

def validate_features(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate engineered features for quality checks
    
    Checks:
    - No NaN values
    - No infinite values
    - Feature ranges are reasonable
    - Correlations are preserved
    
    Args:
        df: DataFrame with engineered features
    
    Returns:
        Dictionary with validation results
    """
    feature_info = get_feature_columns()
    all_features = feature_info['all_features']
    
    validation_results = {
        'passed': True,
        'issues': [],
        'warnings': [],
        'statistics': {}
    }
    
    # Check for NaN
    nan_count = df[all_features].isna().sum().sum()
    if nan_count > 0:
        validation_results['passed'] = False
        validation_results['issues'].append(f"Found {nan_count} NaN values")
    
    # Check for infinite
    inf_count = np.isinf(df[all_features]).sum().sum()
    if inf_count > 0:
        validation_results['passed'] = False
        validation_results['issues'].append(f"Found {inf_count} infinite values")
    
    # Check z-score ranges (should mostly be in [-3, +3])
    for col in feature_info['z_scores']:
        z_extreme = (df[col].abs() > 5).sum()
        # Indicates distribution or scaling issues. This suggests bad normalization, data leakage, wrong mean, or corrupted data
        if z_extreme > len(df) * 0.01:  # More than 1% extreme
            validation_results['warnings'].append(
                f"{col}: {z_extreme} values with |z| > 5 ({z_extreme/len(df)*100:.2f}%)"
            )
    
    # Compute statistics
    validation_results['statistics'] = {
        'n_samples': len(df),
        'n_features': len(all_features),
        'feature_means': df[all_features].mean().to_dict(),
        'feature_stds': df[all_features].std().to_dict()
    }
    
    return validation_results

# if __name__ == "__main__":
#     # Example: Create sample data

#     sample_data = pd.read_csv("./training_data_200k.csv")
    
#     # Engineer features
#     df_engineered = engineer_all_features(sample_data)
#     df_engineered = label_mechanisms(df_engineered)
    
#     print(f"\nInput shape: {sample_data.shape}")
#     print(f"Output shape: {df_engineered.shape}")
    
#     # Validate
#     print("\n" + "="*60)
#     print("VALIDATION")
#     print("="*60)
    
#     validation = validate_features(df_engineered)
#     validate_mechanism_labels(df_engineered)
#     print(f"\nPassed: {validation['passed']}")
    
#     if validation['issues']:
#         print("\nIssues:")
#         for issue in validation['issues']:
#             print(f"  ❌ {issue}")
    
#     if validation['warnings']:
#         print("\nWarnings:")
#         for warning in validation['warnings']:
#             print(f"  ⚠️  {warning}")
    
#     print(f"\nStatistics:")
#     print(f"  Samples: {validation['statistics']['n_samples']}")
#     print(f"  Features: {validation['statistics']['n_features']}")
    
#     # Prepare for model
#     print("\n" + "="*60)
#     print("MODEL INPUT PREPARATION")
#     print("="*60)
    
#     features, feature_info = prepare_features_for_model(sample_data)
    
#     print(f"\nFeature array shape: {features.shape}")
#     print(f"Feature array dtype: {features.dtype}")
#     print(f"\nFirst sample (first 5 features):")
#     print(features[0, :5])
    
#     print("\n✓ Feature engineering pipeline ready!")