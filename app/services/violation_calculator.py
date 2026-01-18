"""
Rule-Based Violation Detection with Risk-Based Probability

Calculates probability of mechanism/defect occurrence based on:
1. How close parameter is to specification limit (proximity risk)
2. How far it exceeds the limit (severity risk)

This models the actual physics: parameters near/beyond limits → higher defect probability
"""

from typing import Dict, List
import math
import pandas as pd

# ============================================================================
# RISK-BASED PROBABILITY CALCULATION
# ============================================================================

def _calculate_risk_probability(value: float,
                               nominal: float,
                               limit: float,
                               tolerance: float,
                               direction: str = 'high') -> float:
    """
    Calculate probability of mechanism/defect based on proximity to and 
    exceeding specification limits.

    Key principle: Risk increases as value approaches the RELEVANT limit
    
    Args:
        value: Actual parameter value
        nominal: Nominal/target value (NV)
        limit: Specification limit (USL or LSL)
        tolerance: Tolerance range (typically USL - nominal or nominal - LSL)
        direction: 'high' or 'low'
    
    Returns:
        Probability (0.0 to 1.0) that this parameter state will cause 
        a mechanism or defect
    
    Examples (HIGH direction, USL=0.044, nominal=0.040, tolerance=0.004):
        value = 0.038 → probability ≈ 0.01 (moving away from USL, very low risk)
        value = 0.040 → probability ≈ 0.05 (at nominal, minimal risk)
        value = 0.042 → probability ≈ 0.30 (halfway to USL, moderate risk)
        value = 0.044 → probability ≈ 0.70 (at USL, high risk)
        value = 0.046 → probability ≈ 0.90 (beyond USL, very high risk)
    """

    if direction == 'high':
        # For high violations, risk increases as value exceeds nominal toward USL
        if value <= nominal:
            # Below nominal - MOVING AWAY from the HIGH limit
            # Very minimal risk (just base process variation)
            # The further below nominal, the LOWER the risk (not higher!)
            
            # Cap at minimal base risk regardless of how far below
            probability = 0.01  # Constant minimal risk
        elif value <= limit:
            # Between nominal and USL - APPROACHING the limit
            # This is the "warning zone" - risk increases
            distance_from_nominal = value - nominal
            distance_to_limit = limit - nominal
            
            # Normalized position: 0 at nominal, 1 at limit
            position = distance_from_nominal / distance_to_limit
            # Sigmoid-like curve: slow increase, then rapid near limit
            # At nominal (0): ~5% probability
            # At midpoint (0.5): ~30% probability  
            # At limit (1.0): ~70% probability
            probability = 0.05 + 0.65 * (position ** 2)
        else:
            # Beyond USL - EXCEEDED the limit
            # High risk! Increases rapidly
            
            distance_beyond_limit = value - limit
            
            # Normalized by tolerance
            exceedance = distance_beyond_limit / tolerance
            
            # Start at 0.70 (at limit) and approach 1.0 asymptotically
            # At limit: 0.70
            # At limit + tolerance: ~0.90
            # At limit + 2*tolerance: ~0.97
            probability = 0.70 + 0.30 * (1 - math.exp(-exceedance * 2))
    else:  # direction == 'low'
        # For LOW violations, risk increases as value goes below LSL
        
        if value >= nominal:
            # Above nominal - MOVING AWAY from the LOW limit
            # Very minimal risk (just base process variation)
            
            # Cap at minimal base risk regardless of how far above
            probability = 0.01  # Constant minimal risk
        elif value >= limit:
            # Between LSL and nominal - APPROACHING the limit
            # Risk increases as we approach LSL
            
            distance_from_nominal = nominal - value
            distance_to_limit = nominal - limit
            
            # Normalized position: 0 at nominal, 1 at limit
            position = distance_from_nominal / distance_to_limit
            
            # Sigmoid-like curve
            probability = 0.05 + 0.65 * (position ** 2)
        else:
            # Below LSL - EXCEEDED the limit
            # High risk!
            
            distance_beyond_limit = limit - value
            
            # Normalized by tolerance
            exceedance = distance_beyond_limit / tolerance
            
            # Asymptotic approach to 1.0
            probability = 0.70 + 0.30 * (1 - math.exp(-exceedance * 2))
    # Cap at 0.99 (never 100% certain in real processes)
    if probability > 0.99:
        probability = 0.99
    
    # Floor at 0.01 (always some minimal risk)
    if probability < 0.01:
        probability = 0.01
    
    return probability

def calculate_all_parameter_risks(row: pd.Series, 
                                  process_parameters: Dict) -> Dict[str, float]:
    """
    Calculate risk probabilities for all parameters
    
    Args:
        row: DataFrame row with parameter values
        process_parameters: Specification limits
    
    Returns:
        Dictionary of parameter risks
    """
    risks = {}
    
    # Loop through all parameters
    for param_name in process_parameters.keys():
        param_info = process_parameters[param_name]
        value = row[param_name]
        nominal = param_info['nominal']
        tolerance = param_info['tolerance']
        
        # Determine direction based on value
        if value > nominal:
            # High direction
            risks[f'high {param_name.lower()}'] = _calculate_risk_probability(
                value, nominal, param_info['usl'], tolerance, 'high'
            )
            risks[f'low {param_name.lower()}'] = 0.01
        else:
            # Low direction
            risks[f'low {param_name.lower()}'] = _calculate_risk_probability(
                value, nominal, param_info['lsl'], tolerance, 'low'
            )
            risks[f'high {param_name.lower()}'] = 0.01
    
    return risks
