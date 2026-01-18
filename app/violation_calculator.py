"""
Rule-Based Violation Detection with Risk-Based Probability

Calculates probability of mechanism/defect occurrence based on:
1. How close parameter is to specification limit (proximity risk)
2. How far it exceeds the limit (severity risk)

This models the actual physics: parameters near/beyond limits → higher defect probability
"""

from typing import Dict, List
import math
from .constants import PARAMETER_SPECS

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

def _check_single_violation(param_name: str, 
                          value: float, 
                          direction: str) -> Dict:
    """
    Check a single violation with risk-based probability
    
    Args:
        param_name: Parameter name
        value: Actual value
        direction: 'high' or 'low'
    
    Returns:
        Dictionary with violation info including risk probability
    """
    specs = PARAMETER_SPECS[param_name]
    
    # Determine limit
    if direction == 'high':
        limit = specs['usl']
        violated = value > limit
    else:
        limit = specs['lsl']
        violated = value < limit
    
    # Calculate risk-based probability
    probability = _calculate_risk_probability(
        value,
        specs['nominal'],
        limit,
        specs['tolerance'],
        direction
    )
    
    return {
        'violated': violated,
        'probability': probability,
        'value': float(value),
        'limit': limit,
        'nominal': specs['nominal'],
        'unit': specs['unit'],
        'risk_level': _get_risk_level(probability)
    }

def _get_risk_level(probability: float) -> str:
    """
    Convert probability to risk level label
    
    Args:
        probability: Risk probability (0-1)
    
    Returns:
        Risk level: 'minimal', 'low', 'moderate', 'high', 'critical'
    """
    if probability < 0.10:
        return 'minimal'
    elif probability < 0.30:
        return 'low'
    elif probability < 0.50:
        return 'moderate'
    elif probability < 0.70:
        return 'high'
    else:
        return 'critical'

def calculate_violations(board_params: Dict) -> Dict:
    """
    Calculate all parameter violations with risk-based probabilities
    
    Returns probabilities that represent:
    "How likely is this parameter state to cause a mechanism or defect?"
    
    Args:
        board_params: Raw process parameters
    
    Returns:
        Violation results with risk probabilities
    """
    violations = {}
    
    # Loop through all parameters
    for param_name in PARAMETER_SPECS.keys():
        value = board_params[param_name]
        
        # Check both directions
        violations[param_name] = {}
        
        for direction in ['high', 'low']:
            violations[param_name][direction] = _check_single_violation(
                param_name, value, direction
            )
    # Build summary
    total_violations = 0
    violated_parameters = []
    violation_details = []
    high_risk_parameters = []

    # Track all high-risk probabilities for overall score
    high_risk_probs = []

    for param_name, directions in violations.items():
        param_violated = False
        param_high_risk = False
        
        for direction, info in directions.items():
            # Count actual violations
            if info['violated']:
                total_violations += 1
                param_violated = True
                
                violation_details.append({
                    'parameter': param_name,
                    'direction': direction,
                    'probability': info['probability'],
                    'risk_level': info['risk_level'],
                    'value': info['value'],
                    'limit': info['limit']
                })
                
                # Add to high-risk probabilities
                high_risk_probs.append(info['probability'])
            
            # Track high risk even if not violated (>= 30% threshold)
            elif info['probability'] >= 0.30:
                param_high_risk = True
                
                violation_details.append({
                    'parameter': param_name,
                    'direction': direction,
                    'probability': info['probability'],
                    'risk_level': info['risk_level'],
                    'value': info['value'],
                    'limit': info['limit'],
                    'warning': 'Approaching limit - not yet violated'
                })
                
                # Add to high-risk probabilities
                high_risk_probs.append(info['probability'])
        
        if param_violated:
            violated_parameters.append(param_name)
        
        if param_high_risk:
            high_risk_parameters.append(param_name)
    # Calculate overall violation score (average of high-risk probabilities)
    if high_risk_probs:
        violation_score = sum(high_risk_probs) / len(high_risk_probs)
    else:
        violation_score = 0.01  # Minimal baseline
    violations['summary'] = {
        'total_violations': total_violations,
        'violated_parameters': violated_parameters,
        'high_risk_parameters': high_risk_parameters,
        'has_violations': total_violations > 0,
        'has_high_risk': len(high_risk_parameters) > 0,
        'violation_details': violation_details,
        'violation_score': violation_score
    }
    
    return violations

def get_violation_list(violations: Dict) -> List[str]:
    """
    Get list of violated parameters (actual violations only)
    """
    violation_list = []
    
    parameter_names = list(PARAMETER_SPECS.keys())
    
    for param_name in parameter_names:
        if param_name in violations:
            for direction in ['high', 'low']:
                if violations[param_name][direction]['violated']:
                    violation_list.append(f"{param_name}_{direction}")
    
    return violation_list

def get_high_risk_list(violations: Dict, threshold: float = 0.50) -> List[Dict]:
    """
    Get list of parameters with high risk probability (even if not violated)
    
    This is useful for early warning system!
    
    Args:
        violations: Output from calculate_violations()
        threshold: Probability threshold (default 0.50 = 50%)
    
    Returns:
        List of high-risk parameters with details
    """
    high_risk = []
    
    parameter_names = list(PARAMETER_SPECS.keys())

    for param_name in parameter_names:
        if param_name in violations:
            for direction in ['high', 'low']:
                info = violations[param_name][direction]
                
                if info['probability'] >= threshold:
                    high_risk.append({
                        'parameter': param_name,
                        'direction': direction,
                        'probability': info['probability'],
                        'risk_level': info['risk_level'],
                        'violated': info['violated'],
                        'value': info['value'],
                        'limit': info['limit']
                    })
    
    # Sort by probability (highest first)
    high_risk.sort(key=lambda x: x['probability'], reverse=True)
    
    return high_risk
