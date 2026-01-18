"""
Simple Multi-Task Inference for PCB Defect + Mechanism Prediction
"""

import torch
import pickle
import numpy as np
import pandas as pd
from typing import Dict

from .get_human_readable_desc import *

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
    
    model = create_model('multitask', input_dim=len(feature_names))
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
# STEP 3: PREDICT
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
                'class': 'Poor paste transfer',
                'confidence': 0.95,
                'probabilities': {'No Mechanism': 0.30, 'Poor paste transfer': 0.65, 'Aperture Overfill': 0.05}
            },
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
        mech_probs = torch.softmax(mechanism_logits, dim=1)[0]
        mech_pred = torch.argmax(mech_probs).item()
        mech_conf = mech_probs[mech_pred].item()
    
    # Format results
    defect_classes = ['No Defect', 'Open Circuit', 'Solder Bridging']
    mechanism_classes = ['Aperture Overfill', 'No Mechanism', 'Poor paste transfer']
    
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
        }
    }
    print(get_mechanism_description(mech_pred))
    print(mech_pred)
    return result