"""
Training Loop for PCB Defect Prediction MLP

Implements:
- Training and validation loops
- CrossEntropyLoss with class weights
- Adam optimizer with weight decay
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping
- Metrics tracking (loss, accuracy, F1-score)
- Model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Tuple, Optional, List
import time
from collections import defaultdict
import json


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Training hyperparameters and settings"""
    
    def __init__(self):
        # Training
        self.epochs = 50
        self.batch_size = 256
        
        # Optimizer
        self.learning_rate = 0.001
        # Weight decay is a tiny force that keeps the model from becoming too confident.
        self.weight_decay = 1e-5  # L2 regularization
        self.optimizer_type = 'adam'
        
        # Learning rate scheduler
        self.scheduler_patience = 5  # Reduce LR after 5 epochs no improvement
        self.scheduler_factor = 0.5  # Reduce LR by 50%
        self.scheduler_min_lr = 1e-6  # Minimum learning rate
        
        # Early stopping
        self.early_stopping_patience = 10  # Stop after 10 epochs no improvement
        self.early_stopping_min_delta = 1e-4  # Minimum improvement to count
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Logging
        self.print_every = 10  # Print every N batches
        self.save_best_model = True
        
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'optimizer_type': self.optimizer_type,
            'scheduler_patience': self.scheduler_patience,
            'scheduler_factor': self.scheduler_factor,
            'early_stopping_patience': self.early_stopping_patience,
            'device': self.device
        }


# ============================================================================
# METRICS CALCULATION
# ============================================================================

# def calculate_metrics(outputs: torch.Tensor, 
#                      targets: torch.Tensor,
#                      num_classes: int = 3) -> Dict[str, float]:
#     """
#     Calculate accuracy and per-class metrics
    
#     Args:
#         outputs: Model outputs (logits or probabilities)
#         targets: True labels
#         num_classes: Number of classes
    
#     Returns:
#         Dictionary with metrics
#     """
#     # Get predictions
#     if outputs.dim() > 1:
#         preds = torch.argmax(outputs, dim=1)
#     else:
#         preds = outputs
    
#     # Overall accuracy
#     accuracy = (preds == targets).float().mean().item()
    
#     # Per-class accuracy
#     class_accuracies = {}
#     for c in range(num_classes):
#         mask = targets == c
#         if mask.sum() > 0:
#             class_acc = (preds[mask] == targets[mask]).float().mean().item()
#             class_accuracies[f'class_{c}_acc'] = class_acc
#         else:
#             class_accuracies[f'class_{c}_acc'] = 0.0
    
#     metrics = {
#         'accuracy': accuracy,
#         **class_accuracies
#     }
    
#     return metrics


def calculate_f1_score(outputs: torch.Tensor,
                      targets: torch.Tensor,
                      num_classes: int = 3,
                      average: str = 'weighted') -> float:
    """
    Calculate F1-score
    
    Args:
        outputs: Model outputs (logits or probabilities)
        targets: True labels
        num_classes: Number of classes
        average: 'weighted', 'macro', or 'micro'
    
    Returns:
        F1-score
    """
    # Get predictions
    if outputs.dim() > 1:
        #reducing dimension here
        #argmax on logits gives the predicted class directly.
        preds = torch.argmax(outputs, dim=1)
    else:
        preds = outputs
    
    # Convert to numpy for sklearn
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Calculate per-class precision and recall
    f1_scores = []
    weights = []
    
    for c in range(num_classes):
        # True positives, false positives, false negatives
        tp = ((preds_np == c) & (targets_np == c)).sum()
        fp = ((preds_np == c) & (targets_np != c)).sum()
        fn = ((preds_np != c) & (targets_np == c)).sum()
        
        # Precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
        
        # Weight by support
        weights.append((targets_np == c).sum())
    
    # Average
    if average == 'weighted':
        total_weight = sum(weights)
        if total_weight > 0:
            f1 = sum(f * w for f, w in zip(f1_scores, weights)) / total_weight
        else:
            f1 = 0.0
    elif average == 'macro':
        f1 = np.mean(f1_scores)
    elif average == 'micro':
        # For micro, calculate global TP, FP, FN
        tp_total = (preds_np == targets_np).sum()
        f1 = tp_total / len(targets_np)
    else:
        raise ValueError(f"Unknown average type: {average}")
    
    return f1

# ============================================================================
# MULTI-TASK LOSS FUNCTION
# ============================================================================

def multitask_loss(defect_logits: torch.Tensor,
                   mechanism_logits: torch.Tensor,
                   defect_targets: torch.Tensor,
                   mechanism_targets: torch.Tensor,
                   defect_weights: torch.Tensor,
                   mechanism_pos_weight: torch.Tensor,
                   task_weights: dict = {'defect': 1.0, 'mechanism': 0.5},
                   device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined loss for defect + mechanism prediction
    
    Args:
        defect_logits: (batch, 3) - raw scores for defect classes
        mechanism_logits: (batch, 1) - raw scores for mechanism
        defect_targets: (batch,) - defect class indices 0/1/2
        mechanism_targets: (batch,) - mechanism binary labels 0/1
        defect_weights: (3,) - class weights for defect loss
        mechanism_pos_weight: (1,) - pos_weight for mechanism loss
        task_weights: dict - relative importance of each task
        device: device to run on
    
    Returns:
        tuple: (total_loss, defect_loss, mechanism_loss)
    """
    # Defect loss (multi-class classification)
    defect_loss = nn.functional.cross_entropy(
        defect_logits,
        defect_targets,
        weight=defect_weights.to(device)
    )

    # Mechanism loss (binary classification)
    mechanism_targets_expanded = mechanism_targets.unsqueeze(1)  # (batch,) -> (batch, 1)
    
    mechanism_loss = nn.functional.binary_cross_entropy_with_logits(
        mechanism_logits,
        mechanism_targets_expanded,
        pos_weight=mechanism_pos_weight.to(device)
    )
    
    # Combined loss (weighted sum)
    total_loss = (task_weights['defect'] * defect_loss + 
                  task_weights['mechanism'] * mechanism_loss)
    
    return total_loss, defect_loss, mechanism_loss

# ============================================================================
# TRAINING EPOCH
# ============================================================================

def train_one_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        defect_weights: torch.Tensor,
        mechanism_pos_weight: torch.Tensor,
        task_weights: dict,
        device: str,
        epoch: int,
        print_every: int = 10
    ) -> Dict[str, float]:
    """
    Train for one epoch
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        optimizer: Optimizer
        defect_weights: Class weights for defect loss
        mechanism_pos_weight: Positive class weight for mechanism
        task_weights: Task importance weights
        device: Device to train on
        epoch: Current epoch number
        print_every: Print stats every N batches
    
    Returns:
        Dictionary with average metrics for the epoch
    """
    model.train()
    
    # Metrics tracking
    running_total_loss = 0.0
    running_defect_loss = 0.0
    running_mechanism_loss = 0.0
    total_samples = 0
    
    all_defect_outputs = []
    all_defect_targets = []
    all_mechanism_preds = []
    all_mechanism_targets = []
    
    start_time = time.time()
    
    # Training loop
    for batch_idx, (features, defect_targets, mechanism_targets) in enumerate(train_loader):
        # Move to device
        features = features.to(device)
        defect_targets = defect_targets.to(device)
        mechanism_targets = mechanism_targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass (returns two outputs)
        defect_logits, mechanism_logits = model(features)

        # Compute multi-task loss
        total_loss, defect_loss, mechanism_loss = multitask_loss(
            defect_logits, mechanism_logits,
            defect_targets, mechanism_targets,
            defect_weights, mechanism_pos_weight,
            task_weights, device
        )
        
        # Backward pass
        total_loss.backward()
        
        # Optimizer step
        # update weights and biases
        optimizer.step()
        
        # Track metrics
        # defect_loss.item() = mean loss per sample in batch
        # mechanism_loss.item() = mean loss per sample in batch
        batch_size = features.size(0)
        running_total_loss += total_loss.item() * batch_size
        running_defect_loss += defect_loss.item() * batch_size
        running_mechanism_loss += mechanism_loss.item() * batch_size
        total_samples += batch_size

        # Get predictions
        mechanism_preds = (torch.sigmoid(mechanism_logits) > 0.5).float().squeeze()

        # Store for metrics calculation
        all_defect_outputs.append(defect_logits.detach())
        all_defect_targets.append(defect_targets.detach())
        all_mechanism_preds.append(mechanism_preds.detach().cpu().numpy())
        all_mechanism_targets.append(mechanism_targets.detach().cpu().numpy())
        
        # Print progress
        if (batch_idx + 1) % print_every == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)}: "
                  f"Total={total_loss.item():.4f}, "
                  f"Defect={defect_loss.item():.4f}, "
                  f"Mech={mechanism_loss.item():.4f}")
    
    # Calculate epoch metrics
    # weighted epoch average
    epoch_total_loss = running_total_loss / total_samples
    epoch_defect_loss = running_defect_loss / total_samples
    epoch_mechanism_loss = running_mechanism_loss / total_samples
    
    # Defect metrics
    all_defect_outputs = torch.cat(all_defect_outputs)
    all_defect_targets = torch.cat(all_defect_targets)
    defect_f1 = calculate_f1_score(all_defect_outputs, all_defect_targets)

    defect_preds = torch.argmax(all_defect_outputs, dim=1)
    defect_acc = (defect_preds == all_defect_targets).float().mean().item()

    # Mechanism metrics
    all_mechanism_preds = np.concatenate(all_mechanism_preds)
    all_mechanism_targets = np.concatenate(all_mechanism_targets)
    
    mechanism_acc = (all_mechanism_preds == all_mechanism_targets).mean()

    # Mechanism F1
    from sklearn.metrics import f1_score
    mechanism_f1 = f1_score(all_mechanism_targets, all_mechanism_preds, zero_division=0)
    
    elapsed_time = time.time() - start_time
    
    metrics = {
        'total_loss': epoch_total_loss,
        'defect_loss': epoch_defect_loss,
        'mechanism_loss': epoch_mechanism_loss,
        'defect_accuracy': defect_acc,
        'defect_f1': defect_f1,
        'mechanism_accuracy': mechanism_acc,
        'mechanism_f1': mechanism_f1,
        'time': elapsed_time
    }
    
    return metrics


# ============================================================================
# VALIDATION EPOCH
# ============================================================================

def validate(model: nn.Module,
            val_loader: DataLoader,
            defect_weights: torch.Tensor,
            mechanism_pos_weight: torch.Tensor,
            task_weights: dict,
            device: str) -> Dict[str, float]:
    """
    Validate model (multi-task version)
    
    Args:
        model: Multi-task model
        val_loader: Validation DataLoader
        defect_weights: Class weights for defect loss
        mechanism_pos_weight: Positive class weight for mechanism
        task_weights: Task importance weights
        device: Device
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    
    running_total_loss = 0.0
    running_defect_loss = 0.0
    running_mechanism_loss = 0.0
    total_samples = 0
    
    all_defect_outputs = []
    all_defect_targets = []
    all_mechanism_preds = []
    all_mechanism_targets = []
    
    with torch.no_grad():
        for features, defect_targets, mechanism_targets in val_loader:
            # Move to device
            # Move to device
            features = features.to(device)
            defect_targets = defect_targets.to(device)
            mechanism_targets = mechanism_targets.to(device)
            
            # Forward pass
            defect_logits, mechanism_logits = model(features)
            # Compute loss
            total_loss, defect_loss, mechanism_loss = multitask_loss(
                defect_logits, mechanism_logits,
                defect_targets, mechanism_targets,
                defect_weights, mechanism_pos_weight,
                task_weights, device
            )
            
            # Track metrics
            batch_size = features.size(0)
            running_total_loss += total_loss.item() * batch_size
            running_defect_loss += defect_loss.item() * batch_size
            running_mechanism_loss += mechanism_loss.item() * batch_size
            total_samples += batch_size

            # Get predictions
            mechanism_preds = (torch.sigmoid(mechanism_logits) > 0.5).float().squeeze()
            
            # Store for metrics
            all_defect_outputs.append(defect_logits)
            all_defect_targets.append(defect_targets)
            all_mechanism_preds.append(mechanism_preds.cpu().numpy())
            all_mechanism_targets.append(mechanism_targets.cpu().numpy())
    
    # Calculate metrics
    val_total_loss = running_total_loss / total_samples
    val_defect_loss = running_defect_loss / total_samples
    val_mechanism_loss = running_mechanism_loss / total_samples

    # Defect metrics
    all_defect_outputs = torch.cat(all_defect_outputs)
    all_defect_targets = torch.cat(all_defect_targets)
    defect_f1 = calculate_f1_score(all_defect_outputs, all_defect_targets)
    
    defect_preds = torch.argmax(all_defect_outputs, dim=1)
    defect_acc = (defect_preds == all_defect_targets).float().mean().item()
    
    # Mechanism metrics
    all_mechanism_preds = np.concatenate(all_mechanism_preds)
    all_mechanism_targets = np.concatenate(all_mechanism_targets)
    
    mechanism_acc = (all_mechanism_preds == all_mechanism_targets).mean()
    
    from sklearn.metrics import f1_score
    mechanism_f1 = f1_score(all_mechanism_targets, all_mechanism_preds, zero_division=0)
    
    metrics = {
        'total_loss': val_total_loss,
        'defect_loss': val_defect_loss,
        'mechanism_loss': val_mechanism_loss,
        'defect_accuracy': defect_acc,
        'defect_f1': defect_f1,
        'mechanism_accuracy': mechanism_acc,
        'mechanism_f1': mechanism_f1
    }
    
    return metrics


# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving
    """
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 1e-4,
                 mode: str = 'min'):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to count as improvement
            mode: 'min' (for loss) or 'max' (for accuracy)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if should stop
        
        Args:
            score: Current metric value
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        # Check improvement
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_model(model: nn.Module,
               dataloaders: Dict[str, DataLoader],
               defect_weights: Optional[torch.Tensor] = None,
               mechanism_pos_weight: Optional[torch.Tensor] = None,
               config: Optional[TrainingConfig] = None,
               task_weights: dict = {'defect': 1.0, 'mechanism': 0.5},
               save_path: str = 'best_model.pth') -> Tuple[nn.Module, Dict]:
    """
    Complete training pipeline
    
    Args:
        model: Multi-task model to train
        dataloaders: Dict with 'train' and 'val' DataLoaders
        defect_weights: Class weights for defect loss
        mechanism_pos_weight: Positive class weight for mechanism
        config: Training configuration
        task_weights: Relative importance of each task
        save_path: Path to save best model
    
    Returns:
        Tuple of (trained_model, training_history)
    """
    # Default config
    if config is None:
        config = TrainingConfig()
    
    # Setup device
    device = config.device
    model = model.to(device)
    
    print("="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Device: {device}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Weight decay: {config.weight_decay}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup loss weights
    if defect_weights is not None:
        defect_weights = defect_weights.to(device)
        print(f"\nDefect class weights: {defect_weights.cpu().numpy()}")
    else:
        defect_weights = torch.ones(3).to(device)
        print(f"\nNo defect weights (uniform)")
    
    if mechanism_pos_weight is not None:
        mechanism_pos_weight = mechanism_pos_weight.to(device)
        print(f"Mechanism pos_weight: {mechanism_pos_weight.cpu().numpy()}")
    else:
        mechanism_pos_weight = torch.ones(1).to(device)
        print(f"No mechanism weight (uniform)")
    
    print(f"\nTask weights: defect={task_weights['defect']}, mechanism={task_weights['mechanism']}")
    
    # Setup optimizer
    if config.optimizer_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer_type}")
    
    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr
    )
    
    # Setup early stopping
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        mode='min'
    )
    
    # Training history
    history = {
        'train_total_loss': [],
        'train_defect_loss': [],
        'train_mechanism_loss': [],
        'train_defect_acc': [],
        'train_defect_f1': [],
        'train_mechanism_acc': [],
        'train_mechanism_f1': [],
        'val_total_loss': [],
        'val_defect_loss': [],
        'val_mechanism_loss': [],
        'val_defect_acc': [],
        'val_defect_f1': [],
        'val_mechanism_acc': [],
        'val_mechanism_f1': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    # Training loop
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        print("-" * 60)
        
        # Train
        train_metrics = train_one_epoch(
            model,
            dataloaders['train'],
            optimizer,
            defect_weights,
            mechanism_pos_weight,
            task_weights,
            device,
            epoch,
            config.print_every
        )
        
        print(f"\nTrain: Total={train_metrics['total_loss']:.4f}, "
              f"Defect={train_metrics['defect_loss']:.4f}, "
              f"Mech={train_metrics['mechanism_loss']:.4f}, "
              f"Time={train_metrics['time']:.1f}s")
        print(f"       Defect F1={train_metrics['defect_f1']:.4f}, "
              f"Mech F1={train_metrics['mechanism_f1']:.4f}")
        
        # Validate
        val_metrics = validate(
            model,
            dataloaders['val'],
            defect_weights,
            mechanism_pos_weight,
            task_weights,
            device
        )
        
        print(f"Val:   Total={val_metrics['total_loss']:.4f}, "
              f"Defect={val_metrics['defect_loss']:.4f}, "
              f"Mech={val_metrics['mechanism_loss']:.4f}")
        print(f"       Defect F1={val_metrics['defect_f1']:.4f}, "
              f"Mech F1={val_metrics['mechanism_f1']:.4f}")
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['total_loss'])
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != current_lr:
            print(f"Learning rate reduced: {current_lr:.6f} → {new_lr:.6f}")
        
        # Save history
        history['train_total_loss'].append(train_metrics['total_loss'])
        history['train_defect_loss'].append(train_metrics['defect_loss'])
        history['train_mechanism_loss'].append(train_metrics['mechanism_loss'])
        history['train_defect_acc'].append(train_metrics['defect_accuracy'])
        history['train_defect_f1'].append(train_metrics['defect_f1'])
        history['train_mechanism_acc'].append(train_metrics['mechanism_accuracy'])
        history['train_mechanism_f1'].append(train_metrics['mechanism_f1'])
        
        history['val_total_loss'].append(val_metrics['total_loss'])
        history['val_defect_loss'].append(val_metrics['defect_loss'])
        history['val_mechanism_loss'].append(val_metrics['mechanism_loss'])
        history['val_defect_acc'].append(val_metrics['defect_accuracy'])
        history['val_defect_f1'].append(val_metrics['defect_f1'])
        history['val_mechanism_acc'].append(val_metrics['mechanism_accuracy'])
        history['val_mechanism_f1'].append(val_metrics['mechanism_f1'])
        
        history['lr'].append(current_lr)
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            if config.save_best_model:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_total_loss': val_metrics['total_loss'],
                    'val_defect_f1': val_metrics['defect_f1'],
                    'val_mechanism_f1': val_metrics['mechanism_f1'],
                    'config': config.to_dict()
                }, save_path)
                print(f"✓ Saved best model (val_total_loss={val_metrics['total_loss']:.4f})")
        
        # Early stopping (also tracks validation loss score)
        if early_stopping(val_metrics['total_loss']):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best val_loss: {best_val_loss:.4f}")
            break
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Load best model
    if config.save_best_model:
        checkpoint = torch.load(save_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded best model from epoch {checkpoint['epoch']+1}")
    
    return model, history


# ============================================================================
# HISTORY SAVING
# ============================================================================

def save_training_history(history: Dict, filepath: str = 'training_history.json'):
    """
    Save training history to JSON file
    
    Args:
        history: Training history dictionary
        filepath: Path to save
    """
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Saved training history to {filepath}")
