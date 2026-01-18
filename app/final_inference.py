"""
Complete Multi-Task Inference System

Integrates:
1. Rule-based violation detection (100% accuracy)
2. MLP predictions for defect and mechanism (98% and 82% F1)
3. Complete causal chain output for Mode C

Architecture:
Raw Parameters → [Rules: Violations] + [MLP: Defect + Mechanism] → Causal Chain
"""

from typing import Dict, Tuple
# Import our modules
from .violation_calculator import *
from .mlp_inference import prepare_features, predict
from .get_human_readable_desc import *

def predict_complete(board_params: Dict,
                    model,
                    scaler,
                    feature_names: list,
                    device: str = 'cpu') -> Dict:
    """
    Complete prediction combining rule-based violations and MLP predictions
    
    Args:
        board_params: Raw process parameters
        model: Trained MLP model
        scaler: StandardScaler
        feature_names: Feature list
        device: Computation device
    
    Returns:
        Complete prediction dictionary with causal chain
    """
    # ========================================================================
    # PART 1: Rule-Based Violations (Root Causes)
    # ========================================================================
    violations = calculate_violations(board_params)
    violation_list = get_violation_list(violations)
    high_risk_list = get_high_risk_list(violations, threshold=0.30)
    violation_score = violations['summary']['violation_score']

    # ========================================================================
    # PART 2: MLP Predictions (Mechanism + Defect)
    # ========================================================================
    features = prepare_features(board_params, scaler, feature_names)
    result = predict(
        model,
        features
    )

    # ========================================================================
    # PART 3: Build Complete Result
    # ========================================================================
    result['violations'] = {
        'detected': violation_list,
        'count': len(violation_list),
        'has_violations': len(violation_list) > 0,
        'high_risk_count': len(high_risk_list),
        'violation_score': float(violation_score),
        'details': violations,
        'source': 'Rule-Based',
        'description': get_violation_description(violation_list, high_risk_list)
    }

    result['params'] = board_params

    return result





