from app.data_prep import prepare_data_for_training, save_preprocessing_artifacts
from app.dataset_loader import create_dataloaders, compute_class_weights

data = prepare_data_for_training(data_source='mongodb')
save_preprocessing_artifacts(
    data['defect_encoder'],
    data['mechanism_encoder'],
    data['scaler'],
    data['feature_info'],
    "./app/multitask_model/preprocessing_artifacts.pkl"
)

# Compute weights
defect_weights = compute_class_weights(data['y_train'])
mechanism_weights = compute_class_weights(data['y_mechanism_train'])

# Create dataloaders
dataloaders = create_dataloaders(
    data['X_train'], data['y_train'], data['y_mechanism_train'], data['y_param_risk_train'],
    data['X_val'], data['y_val'], data['y_mechanism_val'], data['y_param_risk_val'],
    data['X_test'], data['y_test'], data['y_mechanism_test'], data['y_param_risk_test'],
    batch_size=256,
    num_workers=0
)

# Test one batch
train_loader = dataloaders['train']
features, defect_labels, mechanism_labels, violation_labels = next(iter(train_loader))

print("\n" + "="*60)
print("BATCH TEST")
print("="*60)
print(f"Features shape:        {features.shape}")        # Should be (256, 18)
print(f"Defect labels shape:   {defect_labels.shape}")   # Should be (256,)
print(f"Mechanism labels shape: {mechanism_labels.shape}") # Should be (256,)
print(f"Violation labels shape: {violation_labels.shape}") # Should be (256, 5)
print(f"\nDefect label range:    {defect_labels.min()}-{defect_labels.max()}")  # 0-2
print(f"\nMechanism label range: {mechanism_labels.min()}-{mechanism_labels.max()}")  # 0-2
print(f"\nViolation label range: {violation_labels.min()}-{violation_labels.max()}")  # [0-1]
print(f"Mechanism label type:  {mechanism_labels.dtype}")  #torch.int64

print("\n✓ Multi-task dataset working!")


