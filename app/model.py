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
# MULTI-TASK MODEL: DEFECT + MECHANISM + PARAMETER RISK PREDICTION
# ============================================================================

class DefectMechanismParameterMLP(nn.Module):
    """
    Multi-Task MLP for complete causal chain prediction
    
    Architecture:
        Input (18)
        ↓ [SHARED ENCODER]
        → Dense(256) + BatchNorm + ReLU + Dropout(0.3)
        → Dense(128) + BatchNorm + ReLU + Dropout(0.3)
        → Dense(64) + BatchNorm + ReLU + Dropout(0.2)
        ↓ [TASK-SPECIFIC HEADS]
        ├→ Defect Head: Dense(3) → [No Defect, Open Circuit, Solder Bridging]
        ├→ Mechanism Head: Dense(3) → [No mechanism, Poor paste transfer, Aperture overfill]
        └→ Parameter Risk Head: Dense(5) → [paste_vol_risk, stencil_risk, viscosity_risk, rh_risk, temp_risk]
    
    Three tasks: 2 classification + 1 regression
    
    Parameters: ~47,000 (shared) + ~600 (heads) = ~47,600
    """
    def __init__(self,
                input_dim: int = 18,
                num_defect_classes: int = 3,
                num_mechanism_classes: int = 3,
                num_parameters: int = 5):
        """
        Initialize Multi-Task MLP
        
        Args:
            input_dim: Number of input features (default: 18)
            num_defect_classes: Number of defect classes (default: 3)
            num_mechanism_classes: Number of mechanism classes (default: 3)
            num_parameters: Number of parameters to predict risk for (default: 5)
        """
        super(DefectMechanismParameterMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = HIDDEN_DIMS
        self.num_defect_classes = num_defect_classes
        self.num_mechanism_classes = num_mechanism_classes
        self.num_parameters = num_parameters  # NEW
        self.dropout_rate = DROPOUT_RATE
        self.use_batchnorm = USE_BATCHNORM
        
        # Build layers
        self._build_shared_encoder()
        self._build_task_heads()
        
        # Initialize weights
        self._initialize_weights()

    def _build_shared_encoder(self):
        """Build shared encoder layers"""
        
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
        
        # Mechanism prediction head (multi-class)
        self.mechanism_head = nn.Linear(self.hidden_dims[2], self.num_mechanism_classes)
        
        # Parameter risk prediction head (regression)
        self.param_risk_head = nn.Linear(self.hidden_dims[2], self.num_parameters)
    
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
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass - returns outputs for all three tasks
        
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            Tuple of (defect_logits, mechanism_logits, param_risk_scores)
            - defect_logits: (batch_size, num_defect_classes)
            - mechanism_logits: (batch_size, num_mechanism_classes)
            - param_risk_scores: (batch_size, num_parameters)
        """
        # Shared encoder
        # Layer 1
        x = self.fc1(x)
        if self.use_batchnorm:
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
        defect_logits = self.defect_head(x)           # (batch, 3)
        mechanism_logits = self.mechanism_head(x)     # (batch, 3)
        param_risk_scores = self.param_risk_head(x)   # (batch, 5)
        
        # Apply sigmoid to param_risk_scores to constrain to [0, 1]
        param_risk_scores = torch.sigmoid(param_risk_scores)
        
        return defect_logits, mechanism_logits, param_risk_scores
    
    def predict_proba(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get probabilities/scores for all tasks (for inference)
        
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            Tuple of (defect_probs, mechanism_probs, param_risk_scores)
            - defect_probs: (batch_size, num_defect_classes) - softmax
            - mechanism_probs: (batch_size, num_mechanism_classes) - softmax
            - param_risk_scores: (batch_size, num_parameters) - sigmoid [0-1]
        """
        defect_logits, mechanism_logits, param_risk_scores = self.forward(x)
        
        # Defect: multi-class (use softmax)
        defect_probs = F.softmax(defect_logits, dim=1)
        
        # Mechanism: multi-class (use softmax)
        mechanism_probs = F.softmax(mechanism_logits, dim=1)
        
        # Parameter risk: already sigmoid applied in forward()
        
        return defect_probs, mechanism_probs, param_risk_scores
    
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get predictions for all tasks (for inference)
        
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            Tuple of (defect_preds, mechanism_preds, param_risk_scores)
            - defect_preds: (batch_size,) - class indices 0/1/2
            - mechanism_preds: (batch_size,) - class indices 0/1/2
            - param_risk_scores: (batch_size, num_parameters) - continuous [0-1]
        """
        defect_probs, mechanism_probs, param_risk_scores = self.predict_proba(x)
        
        # Defect: argmax for class
        defect_preds = torch.argmax(defect_probs, dim=1)
        
        # Mechanism: argmax for class
        mechanism_preds = torch.argmax(mechanism_probs, dim=1)
        
        # Parameter risk: return as-is (continuous scores)
        
        return defect_preds, mechanism_preds, param_risk_scores
    
    def count_parameters(self) -> dict:
        """Count parameters by component"""
        shared_params = sum(p.numel() for name, p in self.named_parameters() 
                           if p.requires_grad and 'head' not in name)
        defect_head_params = sum(p.numel() for name, p in self.named_parameters() 
                                if p.requires_grad and 'defect_head' in name)
        mechanism_head_params = sum(p.numel() for name, p in self.named_parameters() 
                                   if p.requires_grad and 'mechanism_head' in name)
        param_risk_head_params = sum(p.numel() for name, p in self.named_parameters() 
                                     if p.requires_grad and 'param_risk_head' in name)
        
        total = shared_params + defect_head_params + mechanism_head_params + param_risk_head_params
        
        return {
            'shared': shared_params,
            'defect_head': defect_head_params,
            'mechanism_head': mechanism_head_params,
            'param_risk_head': param_risk_head_params,  # NEW
            'total': total
        }
    
    def get_architecture_summary(self) -> str:
        """Get human-readable architecture summary"""
        params = self.count_parameters()
        
        summary = []
        summary.append("="*60)
        summary.append("MULTI-TASK MODEL ARCHITECTURE")
        summary.append("="*60)
        summary.append(f"Model: DefectMechanismParameterMLP")
        summary.append(f"\nInput:")
        summary.append(f"  Features: {self.input_dim}")
        summary.append(f"\nShared Encoder:")
        for i, dim in enumerate(self.hidden_dims, 1):
            summary.append(f"  Layer {i}: {dim} units")
        summary.append(f"\nTask-Specific Heads:")
        summary.append(f"  Defect:         {self.num_defect_classes} classes (No Defect, Open Circuit, Solder Bridging)")
        summary.append(f"  Mechanism:      {self.num_mechanism_classes} classes (No mechanism, Poor paste transfer, Aperture overfill)")
        summary.append(f"  Parameter Risk: {self.num_parameters} continuous scores (paste_vol, stencil, viscosity, rh, temp)")
        summary.append(f"\nConfiguration:")
        summary.append(f"  Dropout rate: {self.dropout_rate}")
        summary.append(f"  BatchNorm: {self.use_batchnorm}")
        summary.append(f"\nParameters:")
        summary.append(f"  Shared encoder:     {params['shared']:>6,}")
        summary.append(f"  Defect head:        {params['defect_head']:>6,}")
        summary.append(f"  Mechanism head:     {params['mechanism_head']:>6,}")
        summary.append(f"  Param risk head:    {params['param_risk_head']:>6,}")
        summary.append(f"  Total:              {params['total']:>6,}")
        summary.append("="*60)
        return "\n".join(summary)

# ============================================================================
# MULTI-TASK MODEL: DEFECT + MECHANISM STAGES + PARAMETER RISK PREDICTION
# ============================================================================

class DefectMechanismStagesParameterMLP(nn.Module):
    """
    Multi-Task MLP for complete causal chain prediction
    
    Architecture:
        Input (18)
        ↓ [SHARED ENCODER]
        → Dense(256) + BatchNorm + ReLU + Dropout(0.3)
        → Dense(128) + BatchNorm + ReLU + Dropout(0.3)
        → Dense(64) + BatchNorm + ReLU + Dropout(0.2)
        ↓ [TASK-SPECIFIC HEADS]
        ├→ Defect Head: Dense(3) → [No Defect, Open Circuit, Solder Bridging]
        ├→ Mechanism Head: Dense(3) → [No mechanism, Poor paste transfer, Aperture overfill]
        └→ Parameter Risk Head: Dense(5) → [paste_vol_risk, stencil_risk, viscosity_risk, rh_risk, temp_risk]
    
    Three tasks: 2 classification + 1 regression
    
    Parameters: ~47,000 (shared) + ~600 (heads) = ~47,600
    """
    def __init__(self,
                input_dim: int = 18,
                num_defect_classes: int = 3,
                num_print_mechanism_classes: int = 3,
                num_reflow_mechanism_classes: int = 3,
                num_parameters: int = 5):
        """
        Initialize Multi-Task MLP
        
        Args:
            input_dim: Number of input features (default: 18)
            num_defect_classes: Number of defect classes (default: 3)
            num_mechanism_classes: Number of mechanism classes (default: 3)
            num_parameters: Number of parameters to predict risk for (default: 5)
        """
        super(DefectMechanismStagesParameterMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = HIDDEN_DIMS
        self.num_defect_classes = num_defect_classes
        self.num_print_mechanism_classes = num_print_mechanism_classes
        self.num_reflow_mechanism_classes = num_reflow_mechanism_classes
        self.num_parameters = num_parameters
        self.dropout_rate = DROPOUT_RATE
        self.use_batchnorm = USE_BATCHNORM
        
        # Build layers
        self._build_shared_encoder()
        self._build_task_heads()
        
        # Initialize weights
        self._initialize_weights()

    def _build_shared_encoder(self):
        """Build shared encoder layers"""
        
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
        
        # Print Stage Mechanism prediction head (multi-class)
        self.print_mechanism_head = nn.Linear(self.hidden_dims[2], self.num_print_mechanism_classes)

        # Reflow Stage Mechanism prediction head (multi-class)
        self.reflow_mechanism_head = nn.Linear(self.hidden_dims[2], self.num_reflow_mechanism_classes)
        
        # Parameter risk prediction head (regression)
        self.param_risk_head = nn.Linear(self.hidden_dims[2], self.num_parameters)
    
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
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass - returns outputs for all three tasks
        
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            Tuple of (defect_logits, mechanism_logits, param_risk_scores)
            - defect_logits: (batch_size, num_defect_classes)
            - mechanism_logits: (batch_size, num_mechanism_classes)
            - param_risk_scores: (batch_size, num_parameters)
        """
        # Shared encoder
        # Layer 1
        x = self.fc1(x)
        if self.use_batchnorm:
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
        defect_logits = self.defect_head(x)   # (batch, 3)
        print_mechanism_logits = self.print_mechanism_head(x)  # (batch, 3)
        reflow_mechanism_logits = self.reflow_mechanism_head(x)  # (batch, 3)
        param_risk_scores = self.param_risk_head(x)   # (batch, 7)
        
        # Apply sigmoid to param_risk_scores to constrain to [0, 1]
        param_risk_scores = torch.sigmoid(param_risk_scores)
        
        return defect_logits, print_mechanism_logits, reflow_mechanism_logits, param_risk_scores
    
    def predict_proba(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get probabilities/scores for all tasks (for inference)
        
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            Tuple of (defect_probs, mechanism_probs, param_risk_scores)
            - defect_probs: (batch_size, num_defect_classes) - softmax
            - mechanism_probs: (batch_size, num_mechanism_classes) - softmax
            - param_risk_scores: (batch_size, num_parameters) - sigmoid [0-1]
        """
        defect_logits, print_mechanism_logits, reflow_mechanism_logits, param_risk_scores = self.forward(x)
        
        # Defect: multi-class (use softmax)
        defect_probs = F.softmax(defect_logits, dim=1)
        
        # Printing Stage Mechanism: multi-class (use softmax)
        print_mechanism_probs = F.softmax(print_mechanism_logits, dim=1)

        # Reflow Stage Mechanism: multi-class (use softmax)
        reflow_mechanism_probs = F.softmax(reflow_mechanism_logits, dim=1)
        
        # Parameter risk: already sigmoid applied in forward()
        
        return defect_probs, print_mechanism_probs, reflow_mechanism_probs, param_risk_scores
    
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get predictions for all tasks (for inference)
        
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            Tuple of (defect_preds, mechanism_preds, param_risk_scores)
            - defect_preds: (batch_size,) - class indices 0/1/2
            - mechanism_preds: (batch_size,) - class indices 0/1/2
            - param_risk_scores: (batch_size, num_parameters) - continuous [0-1]
        """
        defect_probs, print_mechanism_probs, reflow_mechanism_probs, param_risk_scores = self.predict_proba(x)
        
        # Defect: argmax for class
        defect_preds = torch.argmax(defect_probs, dim=1)
        
        # Printing Stage Mechanism: argmax for class
        print_mechanism_preds = torch.argmax(print_mechanism_probs, dim=1)

        # Reflow Stage Mechanism: argmax for class
        reflow_mechanism_preds = torch.argmax(reflow_mechanism_probs, dim=1)
        
        # Parameter risk: return as-is (continuous scores)
        
        return defect_preds, print_mechanism_preds, reflow_mechanism_preds, param_risk_scores
    
    def count_parameters(self) -> dict:
        """Count parameters by component"""
        shared_params = sum(p.numel() for name, p in self.named_parameters() 
                           if p.requires_grad and 'head' not in name)
        defect_head_params = sum(p.numel() for name, p in self.named_parameters() 
                                if p.requires_grad and 'defect_head' in name)
        print_mechanism_head_params = sum(p.numel() for name, p in self.named_parameters() 
                                   if p.requires_grad and 'print_mechanism_head' in name)
        reflow_mechanism_head_params = sum(p.numel() for name, p in self.named_parameters() 
                                   if p.requires_grad and 'reflow_mechanism_head' in name)
        param_risk_head_params = sum(p.numel() for name, p in self.named_parameters() 
                                     if p.requires_grad and 'param_risk_head' in name)
        
        total = shared_params + defect_head_params + print_mechanism_head_params + \
            reflow_mechanism_head_params + param_risk_head_params
        
        return {
            'shared': shared_params,
            'defect_head': defect_head_params,
            'print_mechanism_head': print_mechanism_head_params,
            'reflow_mechanism_head': reflow_mechanism_head_params,
            'param_risk_head': param_risk_head_params,  # NEW
            'total': total
        }
    
    def get_architecture_summary(self) -> str:
        """Get human-readable architecture summary"""
        params = self.count_parameters()
        
        summary = []
        summary.append("="*60)
        summary.append("MULTI-TASK MODEL ARCHITECTURE")
        summary.append("="*60)
        summary.append(f"Model: DefectMechanismParameterMLP")
        summary.append(f"\nInput:")
        summary.append(f"  Features: {self.input_dim}")
        summary.append(f"\nShared Encoder:")
        for i, dim in enumerate(self.hidden_dims, 1):
            summary.append(f"  Layer {i}: {dim} units")
        summary.append(f"\nTask-Specific Heads:")
        summary.append(f"  Defect: {self.num_defect_classes} classes (No Defect, Open Circuit, Solder Bridging)")
        summary.append(f"  Printing Stage Mechanism: {self.num_print_mechanism_classes} classes (No mechanism, Poor paste transfer, Aperture overfill)")
        summary.append(f"  Reflow Stage Mechanism: {self.num_reflow_mechanism_classes} classes (No mechanism, Non coalescence, Excess reflow spreading)")
        summary.append(f"  Parameter Risk: {self.num_parameters} continuous scores (paste_vol, stencil, viscosity, rh, temp)")
        summary.append(f"\nConfiguration:")
        summary.append(f"  Dropout rate: {self.dropout_rate}")
        summary.append(f"  BatchNorm: {self.use_batchnorm}")
        summary.append(f"\nParameters:")
        summary.append(f"  Shared encoder:     {params['shared']:>6,}")
        summary.append(f"  Defect head:        {params['defect_head']:>6,}")
        summary.append(f"  Printing Stage Mechanism head:     {params['print_mechanism_head']:>6,}")
        summary.append(f"  Reflow Stage Mechanism head:     {params['reflow_mechanism_head']:>6,}")
        summary.append(f"  Param risk head:    {params['param_risk_head']:>6,}")
        summary.append(f"  Total:              {params['total']:>6,}")
        summary.append("="*60)
        return "\n".join(summary)


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(model_type: str = 'engineered',
                 input_dim: Optional[int] = None,
                 num_defect_classes: int = 3,
                 num_mechanism_classes: int = 3,
                 num_mechanism_stages_classes: dict = None,
                 num_parameters: int = 5,
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
    elif model_type == 'full_multitask':
        # Defect + Mechanism + Parameter Risk
        input_dim = input_dim or 18
        model = DefectMechanismParameterMLP(
            input_dim=input_dim,
            num_defect_classes=num_defect_classes,
            num_mechanism_classes=num_mechanism_classes,
            num_parameters=num_parameters,
            **kwargs
        )
    elif model_type == 'full_multitask_multistage':
                # Defect + Mechanism stages + Parameter Risk
        input_dim = input_dim or 18
        model = DefectMechanismStagesParameterMLP(
            input_dim=input_dim,
            num_defect_classes=num_defect_classes,
            num_print_mechanism_classes=num_mechanism_stages_classes['print'],
            num_reflow_mechanism_classes=num_mechanism_stages_classes['reflow'],
            num_parameters=num_parameters,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return model