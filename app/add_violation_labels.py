from typing import Dict
import pandas as pd
from app.services.violation_calculator import calculate_all_parameter_risks
from app.constants import parameter_column_map, PARAMETER_SPECS

def create_parameter_risk_labels(df: pd.DataFrame, 
                                 process_parameters: Dict, param_column_map: Dict) -> pd.DataFrame:
    """
    Create risk score ground truth for each parameter
    
    Returns continuous risk scores [0.0 - 1.0] for each parameter.
    Uses max(high_risk, low_risk) since only one direction can be risky.
    
    Args:
        df: DataFrame with parameter values
        process_parameters: Specification limits
    
    Returns:
        DataFrame with added risk score columns (e.g., 'stencil_thickness_risk')
    """
    df = df.copy()
    
    print("\nCalculating parameter risk scores...")
    print(f"Processing {len(df):,} boards...")
    
    # Initialize risk score columns dynamically
    risk_columns = []
    for col_name in process_parameters.keys():
        risk_col = f"{col_name}_risk"
        df[risk_col] = 0.0
        risk_columns.append(risk_col)
    
    # Calculate risk for each row
    for idx, row in df.iterrows():
        # Get risks using risk calculation function
        risks = calculate_all_parameter_risks(row, process_parameters)
        
        # For each parameter, take MAX of high_risk and low_risk
        for col_name in process_parameters.keys():
            risk_score = max(
                risks[f'high {col_name}'],
                risks[f'low {col_name}']
            )
            
            df.loc[idx, f"{col_name}_risk"] = risk_score
    
    print("✓ Parameter risk scores calculated")
    
    # Print statistics
    print("\n" + "="*60)
    print("PARAMETER RISK SCORE STATISTICS")
    print("="*60)
    
    for risk_col in risk_columns:
        mean_risk = df[risk_col].mean()
        median_risk = df[risk_col].median()
        max_risk = df[risk_col].max()
        high_risk_count = (df[risk_col] > 0.70).sum()
        high_risk_pct = (df[risk_col] > 0.70).mean() * 100
        
        print(f"\n{risk_col}:")
        print(f"  Mean:              {mean_risk:.4f}")
        print(f"  Median:            {median_risk:.4f}")
        print(f"  Max:               {max_risk:.4f}")
        print(f"  High risk (>0.70): {high_risk_count:>6,} ({high_risk_pct:5.2f}%)")
    
    print("="*60)
    
    return df

# df = pd.read_csv("./training_data_200k_v3.csv")
# df = create_parameter_risk_labels(df, PARAMETER_SPECS, parameter_column_map)
# df.to_csv("./training_data_200k_v3_param_viols.csv")