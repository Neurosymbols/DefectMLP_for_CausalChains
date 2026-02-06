import torch
from app.model import create_model

# Create multi-task model
model = create_model(
    'full_multitask_multistage', 
    input_dim=14, 
    num_defect_classes=3, 
    num_mechanism_stages_classes={'print': 3, 'reflow': 3},
    num_parameters=7
)

# Print summary
print(model.get_architecture_summary())

# Test forward pass
batch_size = 4
x = torch.randn(batch_size, 14)  # Random input

print("\nTesting forward pass...")
defect_logits, print_mechanism_logits, reflow_mechanism_logits, param_risk_logits = model(x)

print(f"Input shape:            {x.shape}")
print(f"Defect logits shape:    {defect_logits.shape}")    # Should be (4, 3)
print(f"Print Mechanism logits shape: {print_mechanism_logits.shape}")  # Should be (4, 3)
print(f"Reflow Mechanism logits shape: {reflow_mechanism_logits.shape}")  # Should be (4, 3)
print(f"Param risk logits shape: {param_risk_logits.shape}")  # Should be (4, 7)

# Test predictions
print("\nTesting predictions...")
defect_probs, print_mechanism_probs, reflow_mechanism_probs, param_risk_scores_probs = model.predict_proba(x)

print(f"Defect probs shape:     {defect_probs.shape}")      # Should be (4, 3)
print(f"Printing Mechanism probs shape:  {print_mechanism_probs.shape}")   # Should be (4, 3)
print(f"Reflow Mechanism probs shape:  {reflow_mechanism_probs.shape}")   # Should be (4, 3)
print(f"Param risk scores probs shape:  {param_risk_scores_probs.shape}")   # Should be (4, 7)

# Check probabilities sum to 1 for defect (softmax)
print(f"\nDefect probs sum:       {defect_probs.sum(dim=1)}")  # Should be [1, 1, 1, 1]

# Check probabilities sum to 1 for print stage mechanism (softmax)
print(f"\nPrint Stage mechanism probs sum:       {print_mechanism_probs.sum(dim=1)}")  # Should be [1, 1, 1, 1]

# Check probabilities sum to 1 for reflow stage mechanism (softmax)
print(f"\nReflow Stage mechanism probs sum:       {reflow_mechanism_probs.sum(dim=1)}")  # Should be [1, 1, 1, 1]

# Test final predictions
defect_preds, print_mechanism_preds, reflow_mechanism_preds, param_risk_scores = model.predict(x)
print(f"\nDefect predictions:     {defect_preds}")      # Should be values 0, 1, or 2
print(f"\nPrint Stage Mechanism predictions:  {print_mechanism_preds}")    # Should be values 0, 1, or 2
print(f"\nReflow Stage Mechanism predictions:  {reflow_mechanism_preds}")    # Should be values 0, 1, or 2
print(f"\nParam Risk scores:  {param_risk_scores}")    # Should be values b/w [0,1]

print("\n✓ Multi-task model working correctly!")
