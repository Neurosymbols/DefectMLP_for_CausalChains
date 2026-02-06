from app.data_prep import prepare_data_for_training, save_preprocessing_artifacts
from app.dataset_loader import create_dataloaders, compute_class_weights

data = prepare_data_for_training(data_source='mongodb')
save_preprocessing_artifacts(
    data['defect_encoder'],
    data['print_encoder'],
    data['reflow_encoder'],
    data['scaler'],
    data['feature_info'],
    "./app/multitask_model/preprocessing_artifacts.pkl"
)

# Compute weights
defect_weights = compute_class_weights(data['y_train'])
print_mechanism_weights = compute_class_weights(data['y_print_mech_train'])
reflow_mechanism_weights = compute_class_weights(data['y_reflow_mech_train'])

# Create dataloaders
dataloaders = create_dataloaders(
    data,
    batch_size=256,
    num_workers=0
)

# Test one batch
train_loader = dataloaders['train']
features, defect_labels, print_mechanism_labels, reflow_mechanism_labels, violation_labels = next(iter(train_loader))

print("\n" + "="*60)
print("BATCH TEST")
print("="*60)
print(f"Features shape:        {features.shape}")        # Should be (256, 18)
print(f"Defect labels shape:   {defect_labels.shape}")   # Should be (256,)
print(f"Print Stage Mechanism labels shape: {print_mechanism_labels.shape}") # Should be (256,)
print(f"Reflow Stage Mechanism labels shape: {reflow_mechanism_labels.shape}") # Should be (256,)
print(f"Violation labels shape: {violation_labels.shape}") # Should be (256, 7)
print(f"\nDefect label range:    {defect_labels.min()}-{defect_labels.max()}")  # 0-2
print(f"\nPrint Mechanism label range: {print_mechanism_labels.min()}-{print_mechanism_labels.max()}")  # 0-2
print(f"\nReflow Mechanism label range: {reflow_mechanism_labels.min()}-{reflow_mechanism_labels.max()}")  # 0-2
print(f"\nViolation label range: {violation_labels.min()}-{violation_labels.max()}")  # [0-1]

print("\n✓ Multi-task dataset working!")


