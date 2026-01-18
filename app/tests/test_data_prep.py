from app.data_prep import prepare_data_for_training, save_preprocessing_artifacts
from app.dataset_loader import create_dataloaders, compute_class_weights, compute_mechanism_weights

data = prepare_data_for_training(data_source='mongodb')
save_preprocessing_artifacts(
    data['label_encoder'],
    data['scaler'],
    data['feature_info'],
    "./app/multitask_model/preprocessing_artifacts.pkl"
)

# Check output
print("\nOutput keys:", data.keys())
print(f"y_mechanism_train shape: {data['y_mechanism_train'].shape}")
print(f"Mechanism positive rate: {data['split_info']['train_mechanism_rate']:.2%}")

# Compute weights
defect_weights = compute_class_weights(data['y_train'])
mechanism_weights = compute_mechanism_weights(data['y_mechanism_train'])

# Create dataloaders
dataloaders = create_dataloaders(
    data['X_train'], data['y_train'], data['y_mechanism_train'],
    data['X_val'], data['y_val'], data['y_mechanism_val'],
    data['X_test'], data['y_test'], data['y_mechanism_test'],
    batch_size=256,
    num_workers=0
)

# Test one batch
train_loader = dataloaders['train']
features, defect_labels, mechanism_labels = next(iter(train_loader))

print("\n" + "="*60)
print("BATCH TEST")
print("="*60)
print(f"Features shape:        {features.shape}")        # Should be (256, 18)
print(f"Defect labels shape:   {defect_labels.shape}")   # Should be (256,)
print(f"Mechanism labels shape: {mechanism_labels.shape}") # Should be (256,)
print(f"\nDefect label range:    {defect_labels.min()}-{defect_labels.max()}")  # 0-2
print(f"Mechanism label range: {mechanism_labels.min()}-{mechanism_labels.max()}")  # 0-1
print(f"Mechanism label type:  {mechanism_labels.dtype}")  # float32

print("\n✓ Multi-task dataset working!")


