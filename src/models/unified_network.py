"""
Unified Network Architecture for CAAC models.

This module implements the UnifiedClassificationNetwork that combines
FeatureNetwork, AbductionNetwork, and ActionNetwork into a single architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .components import FeatureNetwork, AbductionNetwork, ActionNetwork


class UnifiedClassificationNetwork(nn.Module):
    """
    Unified Classification Network
    
    Combines FeatureNetwork -> AbductionNetwork -> ActionNetwork into a single
    end-to-end trainable architecture for CAAC models.
    
    Args:
        input_dim (int): Input feature dimension
        representation_dim (int): Representation dimension
        latent_dim (int): Latent vector dimension
        n_classes (int): Number of output classes
        feature_hidden_dims (List[int]): Hidden dimensions for feature network
        abduction_hidden_dims (List[int]): Hidden dimensions for abduction network
        
    Example:
        >>> network = UnifiedClassificationNetwork(
        ...     input_dim=10, representation_dim=64, latent_dim=32, n_classes=3,
        ...     feature_hidden_dims=[128, 64], abduction_hidden_dims=[64, 32]
        ... )
        >>> x = torch.randn(16, 10)
        >>> logits, location, scale = network(x)
        >>> print(logits.shape, location.shape, scale.shape)  # (16, 3), (16, 32), (16, 32)
    """
    
    def __init__(self, input_dim: int, representation_dim: int, latent_dim: int, 
                 n_classes: int, feature_hidden_dims: List[int], 
                 abduction_hidden_dims: List[int]):
        super(UnifiedClassificationNetwork, self).__init__()
        
        # Initialize component networks
        self.feature_net = FeatureNetwork(
            input_dim=input_dim,
            representation_dim=representation_dim,
            hidden_dims=feature_hidden_dims
        )
        
        self.abduction_net = AbductionNetwork(
            representation_dim=representation_dim,
            latent_dim=latent_dim,
            hidden_dims=abduction_hidden_dims
        )
        
        self.action_net = ActionNetwork(
            latent_dim=latent_dim,
            n_classes=n_classes
        )
        
        # Store configuration
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.feature_hidden_dims = feature_hidden_dims
        self.abduction_hidden_dims = abduction_hidden_dims

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the entire network.
        
        Args:
            x (torch.Tensor): Input features, shape: (batch_size, input_dim)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - logits: Raw output logits, shape: (batch_size, n_classes)
                - location_param: Latent location parameters, shape: (batch_size, latent_dim)
                - scale_param: Latent scale parameters, shape: (batch_size, latent_dim)
        """
        # Feature extraction
        representation = self.feature_net(x)
        
        # Latent parameter inference
        location_param, scale_param = self.abduction_net(representation)
        
        # Action/output computation
        logits = self.action_net(location_param)
        
        return logits, location_param, scale_param

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities using softmax (for compatibility).
        
        Args:
            x (torch.Tensor): Input features, shape: (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Class probabilities, shape: (batch_size, n_classes)
        """
        logits, _, _ = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def sample_and_score(self, x: torch.Tensor, n_samples: int = 10) -> torch.Tensor:
        """
        Sample from latent distribution and compute scores.
        
        This method is useful for models that need to sample from the latent
        distribution to compute final scores.
        
        Args:
            x (torch.Tensor): Input features, shape: (batch_size, input_dim)
            n_samples (int): Number of samples per input
            
        Returns:
            torch.Tensor: Sampled scores, shape: (batch_size, n_samples, n_classes)
        """
        # Get representation
        representation = self.feature_net(x)
        
        # Sample from latent distribution
        latent_samples = self.abduction_net.sample_latent(representation, n_samples)
        
        # Compute scores for each sample
        scores = self.action_net.compute_scores_from_samples(latent_samples)
        
        return scores
    
    def get_latent_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get latent distribution parameters without computing final logits.
        
        Args:
            x (torch.Tensor): Input features, shape: (batch_size, input_dim)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - location_param: shape: (batch_size, latent_dim)
                - scale_param: shape: (batch_size, latent_dim)
        """
        representation = self.feature_net(x)
        return self.abduction_net(representation)
    
    def get_class_distribution_params(self, x: torch.Tensor, 
                                    distribution_type: str = 'cauchy') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get class-wise distribution parameters.
        
        Args:
            x (torch.Tensor): Input features, shape: (batch_size, input_dim)
            distribution_type (str): Type of distribution ('cauchy' or 'gaussian')
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - class_locations: shape: (batch_size, n_classes)
                - class_scales: shape: (batch_size, n_classes)
        """
        _, location_param, scale_param = self.forward(x)
        return self.action_net.compute_class_distribution_params(
            location_param, scale_param, distribution_type
        )
    
    def get_config(self) -> dict:
        """
        Get network configuration.
        
        Returns:
            dict: Configuration dictionary
        """
        return {
            'input_dim': self.input_dim,
            'representation_dim': self.representation_dim,
            'latent_dim': self.latent_dim,
            'n_classes': self.n_classes,
            'feature_hidden_dims': self.feature_hidden_dims,
            'abduction_hidden_dims': self.abduction_hidden_dims,
            'feature_network_config': self.feature_net.get_config(),
            'abduction_network_config': self.abduction_net.get_config(),
            'action_network_config': self.action_net.get_config()
        }
