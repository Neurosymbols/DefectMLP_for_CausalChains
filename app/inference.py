"""
Simple Multi-Task Inference for PCB Defect + Mechanism Prediction
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict


# ============================================================================
# STEP 1: PREPARE FEATURES
# ============================================================================

def prepare_features(board_params: Dict, scaler, feature_names) -> np.ndarray:
    """
    Engineer features from raw board parameters
    
    Args:
        board_params: Dict with raw parameters:
            {
                'paste_volume': 0.041,
                'stencil_thickness': 0.100,
                'paste_viscosity': 200.0,
                'ambient_rh': 40.0,
                'ambient_temperature': 23.0
            }
        scaler: Fitted StandardScaler
        feature_names: List of feature names
    
    Returns:
        Feature array (1, n_features)
    """
    from app.feature_engg import engineer_all_features
    
    # Create DataFrame
    df = pd.DataFrame([board_params])
    
    # Engineer features
    df_engineered = engineer_all_features(df)
    
    # Extract features
    features = df_engineered[feature_names].values
    
    # Standardize
    if scaler is not None:
        features = scaler.transform(features)
    
    return features


# ============================================================================
# STEP 2: PREDICT
# ============================================================================

def predict(model, features: np.ndarray, device: str = 'cpu') -> Dict:
    """
    Predict defect and mechanism for a board
    
    Args:
        model: Trained multi-task model
        features: Engineered features (1, n_features)
        device: Device
    
    Returns:
        Dictionary with predictions:
        {
            'defect': {
                'class': 'No Defect',
                'confidence': 0.95,
                'probabilities': {'No Defect': 0.95, 'Open': 0.03, 'Bridge': 0.02}
            },
            'mechanism': {
                'present': False,
                'probability': 0.12
            }
        }
    """
    # Convert to tensor
    features_tensor = torch.from_numpy(features).float().to(device)
    
    with torch.no_grad():
        # Forward pass
        defect_logits, mechanism_logits = model(features_tensor)
        
        # Defect predictions
        defect_probs = torch.softmax(defect_logits, dim=1)[0]
        defect_pred = torch.argmax(defect_probs).item()
        defect_conf = defect_probs[defect_pred].item()
        
        # Mechanism predictions
        mechanism_prob = torch.sigmoid(mechanism_logits)[0].item()
        mechanism_present = mechanism_prob > 0.5
    
    # Format results
    defect_classes = ['No Defect', 'Open Circuit', 'Solder Bridging']
    
    result = {
        'defect': {
            'class': defect_classes[defect_pred],
            'label': defect_pred,
            'confidence': float(defect_conf),
            'probabilities': {
                defect_classes[i]: float(defect_probs[i])
                for i in range(3)
            }
        },
        'mechanism': {
            'present': bool(mechanism_present),
            'probability': float(mechanism_prob),
            'name': 'poorpastetransfer'
        }
    }
    
    return result


# ============================================================================
# STEP 3: SIMPLE API
# ============================================================================

def predict_board(board_params: Dict,
                  model,
                  scaler,
                  features,
                  device: str = 'cpu') -> Dict:
    """
    One-function prediction (loads model each time - use for single predictions)
    
    Args:
        board_params: Raw board parameters dict
        model_path: Path to model
        preprocessing_path: Path to preprocessing
        device: Device
    
    Returns:
        Prediction results
    """    
    # Prepare
    features = prepare_features(board_params, scaler, features)
    
    # Predict
    result = predict(model, features, device)
    
    return result
