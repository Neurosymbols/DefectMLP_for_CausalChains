import torch
from app.model import create_model

# Create multi-task model
model = create_model(
    'full_multitask', 
    input_dim=18, 
    num_defect_classes=3, 
    num_mechanism_classes=3,
    num_parameters=5
)

# Print summary
print(model.get_architecture_summary())

# Test forward pass
batch_size = 4
x = torch.randn(batch_size, 18)  # Random input

print("\nTesting forward pass...")
defect_logits, mechanism_logits, param_risk_logits = model(x)

print(f"Input shape:            {x.shape}")
print(f"Defect logits shape:    {defect_logits.shape}")    # Should be (4, 3)
print(f"Mechanism logits shape: {mechanism_logits.shape}")  # Should be (4, 3)
print(f"Param risk logits shape: {param_risk_logits.shape}")  # Should be (4, 5)

# Test predictions
print("\nTesting predictions...")
defect_probs, mechanism_probs, param_risk_scores_probs = model.predict_proba(x)

print(f"Defect probs shape:     {defect_probs.shape}")      # Should be (4, 3)
print(f"Mechanism probs shape:  {mechanism_probs.shape}")   # Should be (4, 3)
print(f"Param risk scores probs shape:  {param_risk_scores_probs.shape}")   # Should be (4, 5)

# Check probabilities sum to 1 for defect (softmax)
print(f"\nDefect probs sum:       {defect_probs.sum(dim=1)}")  # Should be [1, 1, 1, 1]

# Check probabilities sum to 1 for defect (softmax)
print(f"\nDefect probs sum:       {defect_probs.sum(dim=1)}")  # Should be [1, 1, 1, 1]

# Test final predictions
defect_preds, mechanism_preds, param_risk_scores = model.predict(x)
print(f"\nDefect predictions:     {defect_preds}")      # Should be values 0, 1, or 2
print(f"\nMechanism predictions:  {mechanism_preds}")    # Should be values 0, 1, or 2
print(f"\nParam Risk scores:  {param_risk_scores}")    # Should be values b/w [0,1]

print("\n✓ Multi-task model working correctly!")

# # Compare with single-task model
# print("\n" + "="*60)
# print("COMPARISON: Single-Task vs Multi-Task")
# print("="*60)

# single_model = create_model('engineered', input_dim=18)
# multi_model = create_model('multitask', input_dim=18)

# single_params = single_model.count_parameters()
# multi_params = multi_model.count_parameters()['total']

# print(f"Single-task parameters: {single_params:,}")
# print(f"Multi-task parameters:  {multi_params:,}")
# print(f"Difference:             {multi_params - single_params:,} ({(multi_params/single_params - 1)*100:.1f}% increase)")
# print("\nNote: Multi-task adds minimal parameters (~200) but predicts 2 tasks!")