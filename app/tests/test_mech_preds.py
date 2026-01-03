"""
Simple script to export test predictions to CSV
"""

import torch
import pandas as pd
import numpy as np
from app.data_prep import prepare_data_for_training, load_from_mongodb
from app.dataset_loader import create_dataloaders, compute_class_weights, compute_mechanism_weights
from app.model import create_model

# Load data
print("Loading data...")
data = prepare_data_for_training(data_source='mongodb')

# Create test dataloader
defect_weights = compute_class_weights(data['y_train'])
mechanism_weights = compute_mechanism_weights(data['y_mechanism_train'])

dataloaders = create_dataloaders(
    data['X_train'], data['y_train'], data['y_mechanism_train'],
    data['X_val'], data['y_val'], data['y_mechanism_val'],
    data['X_test'], data['y_test'], data['y_mechanism_test'],
    batch_size=256, num_workers=0
)

# Load model
print("Loading model...")
model = create_model('multitask', input_dim=14)
checkpoint = torch.load('./app/multitask_model/best_multitask_model.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get predictions
print("Getting predictions...")
all_defect_true = []
all_defect_pred = []
all_defect_proba = []
all_mechanism_true = []
all_mechanism_pred = []
all_mechanism_proba = []

with torch.no_grad():
    for features, defect_targets, mechanism_targets in dataloaders['test']:
        defect_logits, mechanism_logits = model(features)
        
        defect_probs = torch.softmax(defect_logits, dim=1)
        defect_preds = torch.argmax(defect_probs, dim=1)
        
        mechanism_probs = torch.sigmoid(mechanism_logits).squeeze()
        mechanism_preds = (mechanism_probs > 0.5).float()
        
        all_defect_true.append(defect_targets.numpy())
        all_defect_pred.append(defect_preds.numpy())
        all_defect_proba.append(defect_probs.numpy())
        all_mechanism_true.append(mechanism_targets.numpy())
        all_mechanism_pred.append(mechanism_preds.numpy())
        all_mechanism_proba.append(mechanism_probs.numpy())

# Concatenate
defect_true = np.concatenate(all_defect_true)
defect_pred = np.concatenate(all_defect_pred)
defect_proba = np.concatenate(all_defect_proba)
mechanism_true = np.concatenate(all_mechanism_true)
mechanism_pred = np.concatenate(all_mechanism_pred)
mechanism_proba = np.concatenate(all_mechanism_proba)

# Get test data
print("Loading test data...")
df_full = load_from_mongodb()
test_batches = data['split_info']['test_batches']
test_df = df_full[df_full['batch_id'].isin(test_batches)].reset_index(drop=True)

# Create results dataframe
defect_classes = ['No Defect', 'Open Circuit', 'Solder Bridging']

results_df = pd.DataFrame({
    # Original data
    'board_id': test_df['board_id'],
    'batch_id': test_df['batch_id'],
    'paste_volume': test_df['paste_volume'],
    'stencil_thickness': test_df['stencil_thickness'],
    'paste_viscosity': test_df['paste_viscosity'],
    'ambient_rh': test_df['ambient_rh'],
    'ambient_temperature': test_df['ambient_temperature'],
    
    # True labels
    'true_defect': [defect_classes[int(x)] for x in defect_true],
    'true_mechanism': mechanism_true,
    
    # Predictions
    'pred_defect': [defect_classes[int(x)] for x in defect_pred],
    'pred_mechanism': mechanism_pred,
    
    # Probabilities
    'prob_no_defect': defect_proba[:, 0],
    'prob_open_circuit': defect_proba[:, 1],
    'prob_solder_bridging': defect_proba[:, 2],
    'prob_poorpastetransfer': mechanism_proba,
    
    # Correct/wrong
    'defect_correct': (defect_pred == defect_true).astype(int),
    'mechanism_correct': (mechanism_pred == mechanism_true).astype(int)
})

# Save
results_df.to_csv('test_predictions.csv', index=False)

print(f"\n✓ Saved {len(results_df)} predictions to test_predictions.csv")
print(f"  Defect accuracy: {results_df['defect_correct'].mean()*100:.2f}%")
print(f"  Mechanism accuracy: {results_df['mechanism_correct'].mean()*100:.2f}%")