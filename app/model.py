"""
MLP Model Architecture for PCB Defect Prediction

Implements the Engineered Features MLP (Option 2):
- Input: 18 features (5 raw + 13 engineered)
- Hidden: 256 → 128 → 64
- Output: 3 classes (No Defect, Open Circuit, Solder Bridging)
- Components: BatchNorm, ReLU, Dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from .constants import *

# ============================================================================
# MAIN MODEL: ENGINEERED FEATURES MLP
# ============================================================================

class EngineeredFeaturesMLP(nn.Module):
    """
    Multi-Layer Perceptron for defect prediction
    
    Architecture:
        Input (18) 
        → Dense(256) + BatchNorm + ReLU + Dropout(0.3)
        → Dense(128) + BatchNorm + ReLU + Dropout(0.3)
        → Dense(64) + BatchNorm + ReLU + Dropout(0.2)
        → Dense(3) + Softmax
        → Output: [P(No Defect), P(Solder Bridging), P(Open Circuit)]
    
    Parameters: ~47,000
    """
    
    def __init__(self,
        input_dim: int,
        num_classes: int = 3
    ):
        """
        Initialize MLP
        """
        super(EngineeredFeaturesMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = HIDDEN_DIMS
        self.num_classes = num_classes
        self.dropout_rate = DROPOUT_RATE
        self.use_batchnorm = USE_BATCHNORM
        
        # Build layers
        self._build_layers()
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_layers(self):
        """Build network layers"""
        
        # Layer 1: Input → Hidden1
        # This is creation of weight matrix ( i x 256 )
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dims[0])
        if self.use_batchnorm:
            self.bn1 = nn.BatchNorm1d(self.hidden_dims[0])
        self.dropout1 = nn.Dropout(self.dropout_rate)
        
        # Layer 2: Hidden1 → Hidden2
        # This is creation of weight matrix ( 256 x 128 )
        self.fc2 = nn.Linear(self.hidden_dims[0], self.hidden_dims[1])
        if self.use_batchnorm:
            self.bn2 = nn.BatchNorm1d(self.hidden_dims[1])
        self.dropout2 = nn.Dropout(self.dropout_rate)
        
        # Layer 3: Hidden2 → Hidden3
        # This is creation of weight matrix ( 128 x 64 )
        self.fc3 = nn.Linear(self.hidden_dims[1], self.hidden_dims[2])
        if self.use_batchnorm:
            self.bn3 = nn.BatchNorm1d(self.hidden_dims[2])
        # Lighter dropout for last hidden layer
        self.dropout3 = nn.Dropout(self.dropout_rate * 0.67)  # 0.2 if dropout=0.3
        
        # Output Layer: Hidden3 → Classes
        # This is creation of weight matrix ( 64 x 3 )
        self.fc4 = nn.Linear(self.hidden_dims[2], self.num_classes)
    
    def _initialize_weights(self):
        """
        Initialize weights using Xavier/Kaiming initialization
        
        Xavier (Glorot) initialization: Good for tanh/sigmoid
        Kaiming (He) initialization: Better for ReLU
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming initialization for ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            Output tensor (batch_size, num_classes) with probabilities
        """
        # 1 x i (number of input features)
        # Layer 1
        x = self.fc1(x) # 1 x i . [i x 256] = 1 x 256
        if self.use_batchnorm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x) # 1 x 256 . [256 x 128] = 1 x 128
        if self.use_batchnorm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x) # 1 x 128 . [128 x 64] = 1 x 64
        if self.use_batchnorm:
            x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Output layer
        x = self.fc4(x) # 1 x 64 . [64 x 3] = 1 x 3
        
        # Softmax for probabilities
        # Note: During training, we use CrossEntropyLoss which applies softmax internally
        # So we return logits here. For inference, apply softmax.
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities (for inference)
        
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            Probabilities (batch_size, num_classes)
        """
        logits = self.forward(x) # 1 x 3. Model is outputting logits(raw scores) instead of probabilities
        return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions (for inference)
        
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            Class indices (batch_size,)
        """
        probs = self.predict_proba(x)
        #dim=1 means look across columns
        return torch.argmax(probs, dim=1)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_architecture_summary(self) -> str:
        """Get human-readable architecture summary"""
        summary = []
        summary.append("="*60)
        summary.append("MODEL ARCHITECTURE SUMMARY")
        summary.append("="*60)
        summary.append(f"Model: EngineeredFeaturesMLP")
        summary.append(f"\nInput:")
        summary.append(f"  Features: {self.input_dim}")
        summary.append(f"\nHidden Layers:")
        for i, dim in enumerate(self.hidden_dims, 1):
            summary.append(f"  Layer {i}: {dim} units")
        summary.append(f"\nOutput:")
        summary.append(f"  Classes: {self.num_classes}")
        summary.append(f"\nConfiguration:")
        summary.append(f"  Dropout rate: {self.dropout_rate}")
        summary.append(f"  BatchNorm: {self.use_batchnorm}")
        summary.append(f"\nParameters:")
        summary.append(f"  Total: {self.count_parameters():,}")
        summary.append("="*60)
        return "\n".join(summary)

# ============================================================================
# MULTI-TASK MODEL: DEFECT + MECHANISM PREDICTION
# ============================================================================

class DefectAndMechanismMLP(nn.Module):
    """
    Multi-Task MLP for defect + mechanism prediction
    
    Architecture:
        Input (18)
        ↓ [SHARED ENCODER]
        → Dense(256) + BatchNorm + ReLU + Dropout(0.3)
        → Dense(128) + BatchNorm + ReLU + Dropout(0.3)
        → Dense(64) + BatchNorm + ReLU + Dropout(0.2)
        ↓ [TASK-SPECIFIC HEADS]
        ├→ Defect Head: Dense(3) → [No Defect, Open, Bridge]
        └→ Mechanism Head: Dense(1) → [poorpastetransfer probability]
    
    Parameters: ~47,000 (shared) + ~200 (heads) = ~47,200
    """
    
    def __init__(self,
                 input_dim: int = 18,
                 num_defect_classes: int = 3,
                 num_mechanism_classes: int = 3):
        """
        Initialize Multi-Task MLP
        
        Args:
            input_dim: Number of input features (default: 18)
            num_defect_classes: Number of defect classes (default: 3)
            num_mechanism_classes: Number of mechanisms to predict (default: 3)
        """
        super(DefectAndMechanismMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = HIDDEN_DIMS
        self.num_defect_classes = num_defect_classes
        self.num_mechanism_classes = num_mechanism_classes
        self.dropout_rate = DROPOUT_RATE
        self.use_batchnorm = USE_BATCHNORM
        
        # Build layers
        self._build_shared_encoder()
        self._build_task_heads()
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_shared_encoder(self):
        """Build shared encoder layers (same as single-task model)"""
        
        # Layer 1: Input → Hidden1
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dims[0])
        if self.use_batchnorm:
            self.bn1 = nn.BatchNorm1d(self.hidden_dims[0])
        self.dropout1 = nn.Dropout(self.dropout_rate)
        
        # Layer 2: Hidden1 → Hidden2
        self.fc2 = nn.Linear(self.hidden_dims[0], self.hidden_dims[1])
        if self.use_batchnorm:
            self.bn2 = nn.BatchNorm1d(self.hidden_dims[1])
        self.dropout2 = nn.Dropout(self.dropout_rate)
        
        # Layer 3: Hidden2 → Hidden3
        self.fc3 = nn.Linear(self.hidden_dims[1], self.hidden_dims[2])
        if self.use_batchnorm:
            self.bn3 = nn.BatchNorm1d(self.hidden_dims[2])
        self.dropout3 = nn.Dropout(self.dropout_rate * 0.67)
    
    def _build_task_heads(self):
        """Build task-specific prediction heads"""
        
        # Defect prediction head (multi-class)
        self.defect_head = nn.Linear(self.hidden_dims[2], self.num_defect_classes)
        
        # Mechanism prediction head (binary)
        self.mechanism_head = nn.Linear(self.hidden_dims[2], self.num_mechanism_classes)
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass - returns logits for both tasks
        
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            Tuple of (defect_logits, mechanism_logits)
            - defect_logits: (batch_size, num_defect_classes)
            - mechanism_logits: (batch_size, num_mechanisms)
        """
        # Shared encoder
        # Layer 1
        x = self.fc1(x)
        if self.use_batchnorm:
            # batchnorm adds n weight params + n bias params
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        if self.use_batchnorm:
            x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Task-specific heads
        defect_logits = self.defect_head(x)       # (batch, 3)
        mechanism_logits = self.mechanism_head(x)  # (batch, 3)
        
        return defect_logits, mechanism_logits
    
    def predict_proba(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get probabilities for both tasks (for inference)
        
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            Tuple of (defect_probs, mechanism_probs)
            - defect_probs: (batch_size, num_defect_classes) - softmax
            - mechanism_probs: (batch_size, num_mechanisms) - sigmoid
        """
        defect_logits, mechanism_logits = self.forward(x)
        
        # Defect: multi-class (use softmax)
        defect_probs = F.softmax(defect_logits, dim=1)
        
        # Mechanism: binary (use sigmoid)
        mechanism_probs = F.softmax(mechanism_logits, dim=1)
        
        return defect_probs, mechanism_probs
    
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions for both tasks (for inference)
        
        Args:
            x: Input tensor (batch_size, input_dim)
            mechanism_threshold: Threshold for binary mechanism prediction
        
        Returns:
            Tuple of (defect_preds, mechanism_preds)
            - defect_preds: (batch_size,) - class indices 0/1/2
            - mechanism_preds: (batch_size,) - binary 0/1
        """
        defect_probs, mechanism_probs = self.predict_proba(x)
        
        # Defect: argmax for class
        defect_preds = torch.argmax(defect_probs, dim=1)
        
        # Mechanism: threshold for binary
        mechanism_preds = torch.argmax(mechanism_probs, dim=1)
        
        return defect_preds, mechanism_preds
    
    def count_parameters(self) -> dict:
        """Count parameters by component"""
        shared_params = sum(p.numel() for name, p in self.named_parameters() 
                           if p.requires_grad and 'head' not in name) # (18 x 256 + 256 + 512) + (256 x 128 + 128 + 256) + (128 x 64 + 64 + 128)
        defect_head_params = sum(p.numel() for name, p in self.named_parameters() 
                                if p.requires_grad and 'defect_head' in name)
        mechanism_head_params = sum(p.numel() for name, p in self.named_parameters() 
                                   if p.requires_grad and 'mechanism_head' in name)
        
        total = shared_params + defect_head_params + mechanism_head_params
        
        return {
            'shared': shared_params,
            'defect_head': defect_head_params,
            'mechanism_head': mechanism_head_params,
            'total': total
        }
    
    def get_architecture_summary(self) -> str:
        """Get human-readable architecture summary"""
        params = self.count_parameters()
        
        summary = []
        summary.append("="*60)
        summary.append("MULTI-TASK MODEL ARCHITECTURE")
        summary.append("="*60)
        summary.append(f"Model: DefectAndMechanismMLP")
        summary.append(f"\nInput:")
        summary.append(f"  Features: {self.input_dim}")
        summary.append(f"\nShared Encoder:")
        for i, dim in enumerate(self.hidden_dims, 1):
            summary.append(f"  Layer {i}: {dim} units")
        summary.append(f"\nTask-Specific Heads:")
        summary.append(f"  Defect:    {self.num_defect_classes} classes (multi-class)")
        summary.append(f"  Mechanism: {self.num_mechanism_classes} mechanism (binary)")
        summary.append(f"\nConfiguration:")
        summary.append(f"  Dropout rate: {self.dropout_rate}")
        summary.append(f"  BatchNorm: {self.use_batchnorm}")
        summary.append(f"\nParameters:")
        summary.append(f"  Shared encoder:  {params['shared']:>6,}")
        summary.append(f"  Defect head:     {params['defect_head']:>6,}")
        summary.append(f"  Mechanism head:  {params['mechanism_head']:>6,}")
        summary.append(f"  Total:           {params['total']:>6,}")
        summary.append("="*60)
        return "\n".join(summary)

# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(model_type: str = 'engineered',
                 input_dim: Optional[int] = None,
                 num_defect_classes: int = 3,
                 num_mechanism_classes: int = 3,
                 **kwargs) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_type: 'engineered', 'multitask', or 'simple'
        input_dim: Number of input features (auto-set based on type if None)
        num_classes: Number of output classes
        **kwargs: Additional model arguments
    
    Returns:
        Initialized model
    """
    if model_type == 'engineered':
        input_dim = input_dim or 18  # 5 raw + 13 engineered
        model = EngineeredFeaturesMLP(
            input_dim=input_dim,
            num_classes=num_defect_classes,
            **kwargs
        )
    elif model_type == 'multitask':
        # Multi-task model (defect + mechanism)
        input_dim = input_dim or 18  # 5 raw + 13 engineered
        model = DefectAndMechanismMLP(
            input_dim=input_dim,
            num_defect_classes=num_defect_classes,
            num_mechanism_classes=num_mechanism_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return model