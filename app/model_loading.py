import pickle
import torch

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
