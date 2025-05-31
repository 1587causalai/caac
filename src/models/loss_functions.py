"""
Loss Functions for Shared Cauchy OvR Classifier

This module implements various loss functions suitable for the One-vs-Rest
classification strategy with uncertainty quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class OvRBinaryCrossEntropyLoss(nn.Module):
    """
    One-vs-Rest Binary Cross-Entropy Loss
    
    This is the standard loss function for OvR classification as described
    in the theory document. Each class is treated as an independent binary
    classification problem.
    """
    
    def __init__(
        self, 
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        """
        Initialize OvR BCE Loss
        
        Args:
            class_weights: Optional weights for each class to handle imbalance
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction
    
    def forward(
        self, 
        probabilities: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute OvR Binary Cross-Entropy Loss
        
        Args:
            probabilities: Class probabilities P_k(z) (batch_size, num_classes)
            targets: True class indices (batch_size,)
            
        Returns:
            Loss value
        """
        batch_size, num_classes = probabilities.shape
        device = probabilities.device
        
        # Convert targets to one-hot binary labels for OvR
        # y_binary[i, k] = 1 if targets[i] == k, else 0
        binary_targets = torch.zeros(
            batch_size, num_classes, device=device
        )
        binary_targets.scatter_(1, targets.unsqueeze(1), 1.0)
        
        # Compute binary cross-entropy for each class
        # BCE = -[y * log(p) + (1-y) * log(1-p)]
        eps = 1e-8  # For numerical stability
        log_probs = torch.log(probabilities + eps)
        log_neg_probs = torch.log(1.0 - probabilities + eps)
        
        bce_per_class = -(
            binary_targets * log_probs + 
            (1.0 - binary_targets) * log_neg_probs
        )  # (batch_size, num_classes)
        
        # Apply class weights if provided
        if self.class_weights is not None:
            bce_per_class = bce_per_class * self.class_weights.unsqueeze(0)
        
        # Reduce according to specified method
        if self.reduction == 'mean':
            return torch.mean(bce_per_class)
        elif self.reduction == 'sum':
            return torch.sum(bce_per_class)
        else:  # 'none'
            return bce_per_class


class WeightedOvRBCELoss(nn.Module):
    """
    Weighted OvR BCE Loss for handling class imbalance
    
    Automatically computes class weights based on inverse frequency
    """
    
    def __init__(
        self, 
        alpha: float = 1.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Weighted OvR BCE Loss
        
        Args:
            alpha: Weighting factor for rare classes
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(
        self, 
        probabilities: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted OvR BCE loss with automatic class weighting
        """
        batch_size, num_classes = probabilities.shape
        device = probabilities.device
        
        # Convert to binary targets
        binary_targets = torch.zeros(batch_size, num_classes, device=device)
        binary_targets.scatter_(1, targets.unsqueeze(1), 1.0)
        
        # Compute class frequencies and weights
        class_counts = torch.sum(binary_targets, dim=0)
        total_samples = batch_size
        
        # Inverse frequency weighting
        class_weights = total_samples / (num_classes * (class_counts + 1e-8))
        class_weights = torch.pow(class_weights, self.alpha)
        
        # Compute weighted BCE
        eps = 1e-8
        log_probs = torch.log(probabilities + eps)
        log_neg_probs = torch.log(1.0 - probabilities + eps)
        
        bce_per_class = -(
            binary_targets * log_probs + 
            (1.0 - binary_targets) * log_neg_probs
        )
        
        # Apply computed weights
        weighted_bce = bce_per_class * class_weights.unsqueeze(0)
        
        if self.reduction == 'mean':
            return torch.mean(weighted_bce)
        elif self.reduction == 'sum':
            return torch.sum(weighted_bce)
        else:
            return weighted_bce


class FocalOvRLoss(nn.Module):
    """
    Focal Loss adapted for OvR classification
    
    Helps with hard examples and class imbalance by down-weighting
    easy examples and focusing on hard ones.
    """
    
    def __init__(
        self, 
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal OvR Loss
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self, 
        probabilities: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal Loss for OvR classification
        """
        batch_size, num_classes = probabilities.shape
        device = probabilities.device
        
        # Convert to binary targets
        binary_targets = torch.zeros(batch_size, num_classes, device=device)
        binary_targets.scatter_(1, targets.unsqueeze(1), 1.0)
        
        # Compute focal loss
        eps = 1e-8
        ce_loss = -(
            binary_targets * torch.log(probabilities + eps) + 
            (1.0 - binary_targets) * torch.log(1.0 - probabilities + eps)
        )
        
        # Compute focal weights
        pt = torch.where(
            binary_targets == 1, 
            probabilities, 
            1.0 - probabilities
        )
        focal_weights = self.alpha * torch.pow(1.0 - pt, self.gamma)
        
        focal_loss = focal_weights * ce_loss
        
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class UncertaintyRegularizedLoss(nn.Module):
    """
    OvR BCE Loss with uncertainty regularization
    
    Adds regularization terms based on the Cauchy scale parameters
    to encourage appropriate uncertainty quantification.
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        scale_regularizer_weight: float = 0.01,
        target_scale: float = 1.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Uncertainty Regularized Loss
        
        Args:
            base_loss: Base loss function (e.g., OvRBinaryCrossEntropyLoss)
            scale_regularizer_weight: Weight for scale regularization
            target_scale: Target scale value for regularization
            reduction: Reduction method
        """
        super().__init__()
        self.base_loss = base_loss
        self.scale_regularizer_weight = scale_regularizer_weight
        self.target_scale = target_scale
        self.reduction = reduction
    
    def forward(
        self, 
        probabilities: torch.Tensor, 
        targets: torch.Tensor,
        class_scales: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute uncertainty regularized loss
        
        Args:
            probabilities: Class probabilities
            targets: True class indices
            class_scales: Cauchy scale parameters for each class
            
        Returns:
            Dictionary with total loss and components
        """
        # Base classification loss
        base_loss_value = self.base_loss(probabilities, targets)
        
        # Scale regularization: encourage scales near target value
        scale_diff = class_scales - self.target_scale
        scale_regularizer = torch.mean(scale_diff ** 2)
        
        # Total loss
        total_loss = base_loss_value + self.scale_regularizer_weight * scale_regularizer
        
        return {
            'total_loss': total_loss,
            'base_loss': base_loss_value,
            'scale_regularizer': scale_regularizer
        }


def create_loss_function(
    loss_type: str = 'ovr_bce',
    num_classes: int = None,
    class_weights: Optional[torch.Tensor] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions
    
    Args:
        loss_type: Type of loss ('ovr_bce', 'weighted_ovr_bce', 'focal_ovr', 'uncertainty_reg')
        num_classes: Number of classes
        class_weights: Optional class weights
        **kwargs: Additional arguments for specific loss functions
        
    Returns:
        Loss function module
    """
    if loss_type == 'ovr_bce':
        return OvRBinaryCrossEntropyLoss(
            class_weights=class_weights,
            reduction=kwargs.get('reduction', 'mean')
        )
    
    elif loss_type == 'weighted_ovr_bce':
        return WeightedOvRBCELoss(
            alpha=kwargs.get('alpha', 1.0),
            reduction=kwargs.get('reduction', 'mean')
        )
    
    elif loss_type == 'focal_ovr':
        return FocalOvRLoss(
            alpha=kwargs.get('alpha', 1.0),
            gamma=kwargs.get('gamma', 2.0),
            reduction=kwargs.get('reduction', 'mean')
        )
    
    elif loss_type == 'uncertainty_reg':
        base_loss = OvRBinaryCrossEntropyLoss(class_weights=class_weights)
        return UncertaintyRegularizedLoss(
            base_loss=base_loss,
            scale_regularizer_weight=kwargs.get('scale_regularizer_weight', 0.01),
            target_scale=kwargs.get('target_scale', 1.0),
            reduction=kwargs.get('reduction', 'mean')
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}") 