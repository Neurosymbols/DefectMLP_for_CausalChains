"""
Comprehensive Test Suite for violation_calculator.py

Tests the corrected three-region risk probability model:
- Region 1: Below nominal (constant low risk)
- Region 2: Nominal to limit (quadratic increase)
- Region 3: Beyond limit (exponential asymptote)
"""
from app.violation_calculator import (
    calculate_violations, 
    get_violation_list, 
    get_high_risk_list )

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("RISK-BASED VIOLATION DETECTION TEST")
    print("="*70)
    
    # Test case: Board with violations
    board = {
        'paste_volume': 0.046,        # HIGH violation (USL=0.044)
        'stencil_thickness': 102,      # OK
        'paste_viscosity': 165,        # LOW (approaching LSL=150)
        'ambient_rh': 55,              # HIGH violation (USL=50)
        'ambient_temperature': 25      # OK
    }
    
    print("\nTest board parameters:")
    for param, value in board.items():
        print(f"  {param}: {value}")
    
    violations = calculate_violations(board)
    
    print(f"\nSummary:")
    print(f"  Total violations: {violations['summary']['total_violations']}")
    print(f"  Violated parameters: {violations['summary']['violated_parameters']}")
    print(f"  High-risk parameters: {violations['summary']['high_risk_parameters']}")
    print(f"  Overall violation score: {violations['summary']['violation_score']:.2f}")
    
    print(f"\nDetailed Violations:")
    for detail in violations['summary']['violation_details']:
        warning = f" ({detail.get('warning', '')})" if 'warning' in detail else ""
        print(f"  {detail['parameter']}_{detail['direction']}: "
              f"probability={detail['probability']:.3f} ({detail['risk_level']}){warning}")
    
    print(f"\nViolation list: {get_violation_list(violations)}")
    
    print(f"\nHigh-risk list (threshold=0.50):")
    for risk in get_high_risk_list(violations, threshold=0.50):
        print(f"  {risk['parameter']}_{risk['direction']}: "
              f"prob={risk['probability']:.2f}, violated={risk['violated']}")
