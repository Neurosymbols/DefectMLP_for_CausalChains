"""
Simple Multi-Task Inference for PCB Defect + Mechanism Prediction
"""

import torch
import pickle
import numpy as np
import pandas as pd
from typing import Dict

from .get_human_readable_desc import *
from .constants import *

# ============================================================================
# STEP 1: LOAD MODEL
# ============================================================================

def load_model(model_path: str, preprocessing_path: str, device: str = 'cpu'):
    """
    Load trained multi-task model and preprocessing artifacts
    
    Args:
        model_path: Path to saved model (.pth)
        preprocessing_path: Path to preprocessing artifacts (.pkl)
        device: 'cpu' or 'cuda'
    
    Returns:
        Tuple of (model, scaler, feature_names)
    """
    print(f"Loading model from {model_path}...")
    
    # Load preprocessing
    with open(preprocessing_path, 'rb') as f:
        artifacts = pickle.load(f)
    
    scaler = artifacts.get('scaler')
    feature_names = artifacts['feature_info']['all_features']
    
    # Load model
    from app.model import create_model
    
    model = create_model('full_multitask', input_dim=len(feature_names))
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded ({len(feature_names)} features)")
    
    return model, scaler, feature_names

# ============================================================================
# STEP 2: PREPARE FEATURES
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
# STEP 3: EXTRACT PARAMETER DIRECTIONS
# ============================================================================

def extract_parameter_directions(param_risk_scores: np.ndarray,
                                 board_params: Dict,
                                 process_parameters: Dict,
                                 risk_threshold: float = 0.70) -> Dict:
    """
    Extract parameter violation directions from risk scores
    
    Args:
        param_risk_scores: (5,) array of risk scores [0-1]
        board_params: Dict with actual parameter values
        process_parameters: Specification limits
        risk_threshold: Threshold for high-risk (default: 0.70)
    
    Returns:
        Dict with parameter risk details
    """
    param_names = list(PARAMETER_SPECS.keys())    
    param_results = {}
    
    for i, param_name in enumerate(param_names):
        risk_score = float(param_risk_scores[i])
        actual_value = board_params[param_name]
        
        # Get spec limits
        param_info = process_parameters[param_name]
        nominal = param_info['nominal']
        usl = param_info['usl']
        lsl = param_info['lsl']
        
        # Determine direction and status
        if risk_score > risk_threshold:
            # High risk - determine direction from actual value
            if actual_value > nominal:
                direction = 'High'
                status = f'High risk (approaching/exceeding USL={usl})'
            else:
                direction = 'Low'
                status = f'High risk (approaching/below LSL={lsl})'
        else:
            direction = 'Safe'
            status = 'Within safe limits'
        
        param_results[param_name] = {
            'risk_score': risk_score,
            'actual_value': actual_value,
            'nominal': nominal,
            'usl': usl,
            'lsl': lsl,
            'direction': direction,
            'status': status,
            'high_risk': risk_score > risk_threshold
        }
    
    return param_results

# ============================================================================
# STEP 4: PREDICT (UPDATED)
# ============================================================================

def predict(model, features: np.ndarray, board_params: Dict, 
            process_parameters: Dict, device: str = 'cpu') -> Dict:
    """
    Predict defect, mechanism, and parameter risks for a board
    
    Args:
        model: Trained multi-task model
        features: Engineered features (1, n_features)
        board_params: Dict with actual parameter values (for direction extraction)
        process_parameters: Specification limits (for direction extraction)
        device: Device
    
    Returns:
        Dictionary with predictions:
        {
            'defect': {...},
            'mechanism': {...},
            'parameters': {
                'paste_volume': {
                    'risk_score': 0.65,
                    'actual_value': 0.043,
                    'direction': 'High',
                    'status': '...',
                    ...
                },
                ...
            }
        }
    """
    # Convert to tensor
    features_tensor = torch.from_numpy(features).float().to(device)
    
    with torch.no_grad():
        # Forward pass
        defect_logits, mechanism_logits, param_risk_scores = model(features_tensor)
        
        # Defect predictions
        defect_probs = torch.softmax(defect_logits, dim=1)[0]
        defect_pred = torch.argmax(defect_probs).item()
        defect_conf = defect_probs[defect_pred].item()
        
        # Mechanism predictions
        mech_probs = torch.softmax(mechanism_logits, dim=1)[0]
        mech_pred = torch.argmax(mech_probs).item()
        mech_conf = mech_probs[mech_pred].item()

        # Parameter risk scores (already sigmoid in model)
        param_risks = param_risk_scores[0].cpu().numpy()  # (5,)
    
    # Format results
    defect_classes = ['No Defect', 'Open Circuit', 'Solder Bridging']
    mechanism_classes = ['Aperture Overfill', 'No Mechanism', 'Poor paste transfer']

    # Extract parameter directions
    param_details = extract_parameter_directions(
        param_risks,
        board_params,
        process_parameters,
        risk_threshold=0.60
    )
    
    result = {
        'defect': {
            'class': defect_classes[defect_pred],
            'label': defect_pred,
            'confidence': float(defect_conf),
            'probabilities': {
                defect_classes[i]: float(defect_probs[i])
                for i in range(3)
            },
            'source': 'MLP',
            'description': get_defect_description(defect_pred)
        },
        'mechanism': {
            'class': mechanism_classes[mech_pred],
            'label': mech_pred,
            'confidence': float(mech_conf),
            'probabilities': {
                mechanism_classes[i]: float(mech_probs[i])
                for i in range(3)
            },
            'source': 'MLP',
            'description': get_mechanism_description(mech_pred)
        },
        'parameters': param_details
    }
    return result