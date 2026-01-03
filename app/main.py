# 1. Prepare data
from .data_prep import prepare_data_for_training, save_preprocessing_artifacts
data = prepare_data_for_training(
    data_source='mongodb',
    use_standardization=True
)
# Save artifacts
save_preprocessing_artifacts(
    data['label_encoder'],
    data['scaler'],
    data['feature_info'],
    filepath='./preprocessing_artifacts.pkl'
)
#2. create loss weights and data loaders
from .dataset_loader import compute_class_weights, create_dataloaders
# Compute class weights
class_weights = compute_class_weights(data['y_train'], method='balanced')

# Create dataloaders
dataloaders = create_dataloaders(
    data['X_train'], data['y_train'],
    data['X_val'], data['y_val'],
    data['X_test'], data['y_test'],
    batch_size=512,
    num_workers=0
)
# # 3. Train model
# from .train import train_model
# from .model import create_model
# model = create_model('engineered', input_dim=14)
# trained_model, history = train_model(model, dataloaders, class_weights)

# 3. Evaluate & Infere
from .inference import predict_single_board, load_model_simple
from .evaluate.evaluate import evaluate_model
# Load model
predictor = load_model_simple(
    model_path='best_model.pth',
    preprocessing_path='preprocessing_artifacts.pkl',
    device='cpu'
)
# results = evaluate_model(
#     predictor.model, 
#     dataloaders['test'],
#     class_names=['No Defect', 'Open Circuit', 'Solder Bridging'],
#     device='cpu',
#     save_path='evaluation_results.json'
# )
# # Access results
# print(f"Overall accuracy: {results['metrics']['accuracy']:.4f}")
# print(f"Weighted F1: {results['metrics']['weighted_avg']['f1_score']:.4f}")
# print(f"ECE: {results['ece']:.4f}")
# Predict for a board
result = predict_single_board(
        predictor,
        paste_volume=0.034,        # High - may cause bridging
        stencil_thickness=0.105,
        paste_viscosity=240,      # Low - may cause spreading
        ambient_rh=25,             # High
        ambient_temperature=25     # High
)

print(result)