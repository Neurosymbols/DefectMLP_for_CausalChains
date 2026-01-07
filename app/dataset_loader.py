"""
PyTorch Dataset and DataLoader for PCB Defect Prediction

Handles:
- Custom Dataset class for features and labels
- DataLoader creation with batching
- Class weights calculation for imbalanced data
- Data augmentation (optional)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict
from collections import Counter


# ============================================================================
# CUSTOM DATASET CLASS
# ============================================================================

class PCBDefectDataset(Dataset):
    """
    PyTorch Dataset for PCB defect prediction
    
    Wraps feature arrays and labels for efficient batching
    """
    
    def __init__(self, 
                 X: np.ndarray, 
                 y_defect: np.ndarray,
                 y_mechanism: np.ndarray,
                 transform=None):
        """
        Args:
            X: Feature array (n_samples, n_features)
            y: Label array (n_samples,)
            y_mechanism: Mechanism label array (n_samples,) - binary 0/1
            transform: Optional transform to apply to features
        """
        # Convert to float32 (PyTorch default)
        self.X = torch.from_numpy(X).float()
        self.y_defect = torch.from_numpy(y_defect).long()  # long for classification
        self.y_mechanism = torch.from_numpy(y_mechanism).float()  # float for binary classification
        # self.transform = transform
        
        assert len(self.X) == len(self.y_defect) == len(self.y_mechanism), \
            "X, y_defect, and y_mechanism must have same length"
    
    def __len__(self) -> int:
        """Return number of samples"""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get one sample
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (features, defect_label, mechanism_label)
        """
        # we can treat torch.tensor and np.ndarray as mentally equivalent because of shared properties but technically they are not equivalent
        features = self.X[idx]
        defect_label = self.y_defect[idx]
        mechanism_label = self.y_mechanism[idx]
        
        # # Apply transform if provided
        # if self.transform:
        #     features = self.transform(features)
        
        return features, defect_label, mechanism_label
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get count of samples per class (defect)"""
        return dict(Counter(self.y_defect.numpy()))
    
    def get_mechanism_distribution(self) -> Dict[str, int]:
        """Get count of mechanism labels"""
        y_mech_np = self.y_mechanism.numpy()
        return {
            'negative': int((y_mech_np == 0).sum()),
            'positive': int((y_mech_np == 1).sum()),
            'rate': float(y_mech_np.mean())
        }

# ============================================================================
# CLASS WEIGHTS CALCULATION
# ============================================================================

def compute_class_weights(y_train: np.ndarray, 
                         method: str = 'balanced') -> torch.Tensor:
    """
    Compute class weights for imbalanced dataset
    
    For imbalanced classification, we weight loss by inverse frequency:
    - Rare classes get higher weight (model pays more attention)
    - Common classes get lower weight
    
    Args:
        y_train: Training labels
        method: 'balanced' or 'sqrt_balanced'
            - 'balanced': weight = n_samples / (n_classes * n_samples_per_class)
            - 'sqrt_balanced': weight = sqrt(n_samples / (n_classes * n_samples_per_class))
    
    Returns:
        Tensor of class weights (n_classes,)
    
    Example:
        y_train = [0, 0, 0, 0, 1, 2]  # 4 class-0, 1 class-1, 1 class-2
        weights = compute_class_weights(y_train)
        # weights ≈ [0.5, 2.0, 2.0]  (class 0 down-weighted, others up-weighted)
    """
    # Count samples per class
    class_counts = np.bincount(y_train) # each item in array is number of instances per class
    n_samples = len(y_train)
    n_classes = len(class_counts)
    
    if method == 'balanced':
        # Standard balanced weighting
        weights = n_samples / (n_classes * class_counts)
    
    elif method == 'sqrt_balanced':
        # Softer weighting (sqrt of balanced)
        balanced_weights = n_samples / (n_classes * class_counts)
        weights = np.sqrt(balanced_weights)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Convert to tensor
    weights_tensor = torch.from_numpy(weights).float()
    
    # Print info
    print("\n" + "="*60)
    print("CLASS WEIGHTS")
    print("="*60)
    print(f"Method: {method}")
    print("\nClass distribution:")
    for class_idx, count in enumerate(class_counts):
        weight = weights[class_idx]
        print(f"  Class {class_idx}: {count:6d} samples, weight: {weight:6.2f}")
    
    total_weight = (weights * class_counts).sum()
    print(f"\nTotal weighted samples: {total_weight:.2f} (should ≈ {n_samples})")
    
    return weights_tensor

def compute_mechanism_weights(y_mechanism_train: np.ndarray) -> torch.Tensor:
    """
    Compute pos_weight for binary mechanism classification
    
    We use pos_weight
    to handle class imbalance:
    
    pos_weight = (# negative samples) / (# positive samples)
    
    This tells the loss to weight positive samples more heavily.
    
    Args:
        y_mechanism_train: Binary mechanism labels (0 or 1)
    
    Returns:
        Tensor with pos_weight value
    
    Example:
        y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # 90% negative, 10% positive
        pos_weight = 9.0  (weight positive class 9x more)
    """
    # Count positive and negative samples
    n_positive = (y_mechanism_train == 1).sum()
    n_negative = (y_mechanism_train == 0).sum()
    n_total = len(y_mechanism_train)
    
    # Calculate pos_weight
    if n_positive > 0:
        pos_weight = n_negative / n_positive
    else:
        pos_weight = 1.0  # No positive samples, use neutral weight
    
    # Convert to tensor
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)
    
    # Print info
    print("\n" + "="*60)
    print("MECHANISM CLASS WEIGHTS")
    print("="*60)
    print(f"Total samples:    {n_total}")
    print(f"Negative (0):     {n_negative:6d} ({n_negative/n_total*100:5.2f}%)")
    print(f"Positive (1):     {n_positive:6d} ({n_positive/n_total*100:5.2f}%)")
    print(f"pos_weight:       {pos_weight:6.2f}")
    print("\nInterpretation:")
    print(f"  Loss will weight positive samples {pos_weight:.1f}x more than negative")
    
    return pos_weight_tensor

# ============================================================================
# DATALOADER CREATION
# ============================================================================

def create_dataloaders(X_train: np.ndarray,
                      y_train: np.ndarray,
                      y_mechanism_train: np.ndarray,
                      X_val: np.ndarray,
                      y_val: np.ndarray,
                      y_mechanism_val: np.ndarray,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      y_mechanism_test: np.ndarray,
                      batch_size: int = 256,
                      num_workers: int = 0,
                      shuffle_train: bool = True) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train/val/test sets with mechanism labels
    
    Args:
        X_train, y_train, y_mechanism_train: Training data
        X_val, y_val, y_mechanism_val: Validation data
        X_test, y_test, y_mechanism_test: Test data
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        shuffle_train: Whether to shuffle training data
    
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    # Create datasets
    train_dataset = PCBDefectDataset(X_train, y_train, y_mechanism_train)
    val_dataset = PCBDefectDataset(X_val, y_val, y_mechanism_val)
    test_dataset = PCBDefectDataset(X_test, y_test, y_mechanism_test)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train, #indices shuffle
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Never shuffle val/test
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Print info
    print("\n" + "="*60)
    print("DATALOADERS CREATED")
    print("="*60)
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    print(f"\nTrain loader:")
    print(f"  Dataset size: {len(train_dataset)}")
    print(f"  Num batches:  {len(train_loader)}")
    print(f"  Shuffle:      {shuffle_train}")
    
    print(f"\nVal loader:")
    print(f"  Dataset size: {len(val_dataset)}")
    print(f"  Num batches:  {len(val_loader)}")
    
    print(f"\nTest loader:")
    print(f"  Dataset size: {len(test_dataset)}")
    print(f"  Num batches:  {len(test_loader)}")
    
    # Print class distribution
    print(f"\nClass distribution:")
    print(f"  Train: {train_dataset.get_class_distribution()}")
    print(f"  Val:   {val_dataset.get_class_distribution()}")
    print(f"  Test:  {test_dataset.get_class_distribution()}")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }