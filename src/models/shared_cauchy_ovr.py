"""
Shared Latent Cauchy Vector based One-vs-Rest (OvR) Classifier

This module implements the novel multi-classifier architecture described in theory_v2.md:
- Shared latent Cauchy random vector learning
- Linear transformation to class-specific score random variables  
- Uncertainty quantification through Cauchy distribution parameters
- One-vs-Rest classification strategy with enhanced interpretability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any


class SharedCauchyOvRClassifier(nn.Module):
    """
    Shared Latent Cauchy Vector based One-vs-Rest Classifier
    
    This model learns a low-dimensional latent Cauchy random vector and maps it
    to class-specific score random variables through linear transformation.
    Each class score follows a Cauchy distribution, enabling precise uncertainty
    quantification and interpretable decision making.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        latent_dim: int,
        hidden_dims: Optional[list] = None,
        feature_extractor: Optional[nn.Module] = None,
        thresholds: Optional[torch.Tensor] = None,
        eps: float = 1e-6
    ):
        """
        Initialize the Shared Cauchy OvR Classifier
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of classes (N)
            latent_dim: Latent Cauchy vector dimension (M)
            hidden_dims: Hidden layer dimensions for feature extractor
            feature_extractor: Custom feature extractor network
            thresholds: Fixed thresholds C_k for each class (default: zeros)
            eps: Small value for numerical stability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.eps = eps
        
        # Feature extractor network f(x) -> z
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            self.feature_extractor = self._build_default_feature_extractor(
                input_dim, hidden_dims
            )
        
        # Get feature dimension from the last layer of feature extractor
        feature_dim = self._get_feature_dim()
        
        # Latent Cauchy vector parameter networks
        # For each latent component U_j, predict location μ_j(z) and scale σ_j(z)
        self.location_layers = nn.ModuleList([
            nn.Linear(feature_dim, 1) for _ in range(latent_dim)
        ])
        self.scale_layers = nn.ModuleList([
            nn.Linear(feature_dim, 1) for _ in range(latent_dim)
        ])
        
        # Linear transformation from latent space to class scores: S = AU + B
        # A: (N x M) transformation matrix, B: (N,) bias vector
        self.transformation_matrix = nn.Parameter(
            torch.randn(num_classes, latent_dim) * 0.1
        )
        self.transformation_bias = nn.Parameter(
            torch.zeros(num_classes)
        )
        
        # Fixed thresholds C_k for each class (default: all zeros)
        if thresholds is None:
            thresholds = torch.zeros(num_classes)
        self.register_buffer('thresholds', thresholds)
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _build_default_feature_extractor(
        self, 
        input_dim: int, 
        hidden_dims: Optional[list]
    ) -> nn.Module:
        """Build default feature extractor network"""
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _get_feature_dim(self) -> int:
        """Get the output dimension of feature extractor"""
        # Find the last Linear layer in feature extractor
        for module in reversed(list(self.feature_extractor.modules())):
            if isinstance(module, nn.Linear):
                return module.out_features
        raise ValueError("Could not determine feature dimension")
    
    def _initialize_parameters(self):
        """Initialize model parameters"""
        # Initialize transformation matrix with small random values
        nn.init.normal_(self.transformation_matrix, mean=0.0, std=0.1)
        nn.init.zeros_(self.transformation_bias)
        
        # Initialize location and scale layers
        for loc_layer, scale_layer in zip(self.location_layers, self.scale_layers):
            nn.init.normal_(loc_layer.weight, mean=0.0, std=0.1)
            nn.init.zeros_(loc_layer.bias)
            nn.init.normal_(scale_layer.weight, mean=0.0, std=0.1)
            nn.init.constant_(scale_layer.bias, 0.0)  # Will be exponentiated
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features z = f(x)"""
        return self.feature_extractor(x)
    
    def predict_latent_params(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict latent Cauchy vector parameters from features
        
        Args:
            z: Feature tensor (batch_size, feature_dim)
            
        Returns:
            locations: μ_j(z) for each latent component (batch_size, latent_dim)
            scales: σ_j(z) for each latent component (batch_size, latent_dim)
        """
        batch_size = z.size(0)
        
        # Predict location parameters μ_j(z)
        locations = torch.cat([
            layer(z) for layer in self.location_layers
        ], dim=1)  # (batch_size, latent_dim)
        
        # Predict scale parameters σ_j(z) (ensure positive via exp)
        log_scales = torch.cat([
            layer(z) for layer in self.scale_layers
        ], dim=1)  # (batch_size, latent_dim)
        scales = torch.exp(log_scales) + self.eps  # (batch_size, latent_dim)
        
        return locations, scales
    
    def compute_class_score_params(
        self, 
        latent_locations: torch.Tensor, 
        latent_scales: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute class score Cauchy distribution parameters via linear transformation
        
        For each class k: S_k = Σ_j A_kj * U_j + B_k
        - Location: loc(S_k) = Σ_j A_kj * μ_j(z) + B_k  
        - Scale: scale(S_k) = Σ_j |A_kj| * σ_j(z)
        
        Args:
            latent_locations: μ_j(z) (batch_size, latent_dim)
            latent_scales: σ_j(z) (batch_size, latent_dim)
            
        Returns:
            class_locations: loc(S_k) for each class (batch_size, num_classes)
            class_scales: scale(S_k) for each class (batch_size, num_classes)
        """
        # Linear transformation for location parameters
        # class_locations = latent_locations @ A^T + B
        class_locations = torch.matmul(
            latent_locations, self.transformation_matrix.t()
        ) + self.transformation_bias  # (batch_size, num_classes)
        
        # Linear transformation for scale parameters (use absolute values)
        # class_scales = latent_scales @ |A|^T
        abs_transformation_matrix = torch.abs(self.transformation_matrix)
        class_scales = torch.matmul(
            latent_scales, abs_transformation_matrix.t()
        ) + self.eps  # (batch_size, num_classes)
        
        return class_locations, class_scales
    
    def compute_class_probabilities(
        self, 
        class_locations: torch.Tensor, 
        class_scales: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute P_k(z) = P(S_k > C_k) using Cauchy CDF
        
        P_k(z) = 1 - F(C_k; loc(S_k), scale(S_k))
               = 1/2 - (1/π) * arctan((C_k - loc(S_k)) / scale(S_k))
        
        Args:
            class_locations: loc(S_k) (batch_size, num_classes)
            class_scales: scale(S_k) (batch_size, num_classes)
            
        Returns:
            probabilities: P_k(z) for each class (batch_size, num_classes)
        """
        # Compute (C_k - loc(S_k)) / scale(S_k)
        normalized_diff = (
            self.thresholds.unsqueeze(0) - class_locations
        ) / (class_scales + self.eps)
        
        # Compute Cauchy CDF complement: 1 - F(C_k)
        probabilities = 0.5 - (1.0 / np.pi) * torch.atan(normalized_diff)
        
        # Ensure probabilities are in [0, 1]
        probabilities = torch.clamp(probabilities, min=self.eps, max=1.0 - self.eps)
        
        return probabilities
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the Shared Cauchy OvR Classifier
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Dictionary containing:
            - 'probabilities': Class probabilities P_k(z) (batch_size, num_classes)
            - 'class_locations': Class score locations (batch_size, num_classes)
            - 'class_scales': Class score scales (batch_size, num_classes)
            - 'latent_locations': Latent locations (batch_size, latent_dim)
            - 'latent_scales': Latent scales (batch_size, latent_dim)
            - 'features': Extracted features (batch_size, feature_dim)
        """
        # Step 1: Feature extraction z = f(x)
        features = self.extract_features(x)
        
        # Step 2: Predict latent Cauchy vector parameters
        latent_locations, latent_scales = self.predict_latent_params(features)
        
        # Step 3: Linear transformation to class score parameters
        class_locations, class_scales = self.compute_class_score_params(
            latent_locations, latent_scales
        )
        
        # Step 4: Compute class probabilities using Cauchy CDF
        probabilities = self.compute_class_probabilities(
            class_locations, class_scales
        )
        
        return {
            'probabilities': probabilities,
            'class_locations': class_locations,
            'class_scales': class_scales,
            'latent_locations': latent_locations,
            'latent_scales': latent_scales,
            'features': features
        }
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions using argmax over class probabilities
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Predicted class indices (batch_size,)
        """
        with torch.no_grad():
            outputs = self.forward(x)
            predictions = torch.argmax(outputs['probabilities'], dim=1)
            return predictions
    
    def get_uncertainty_metrics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract uncertainty metrics for interpretability
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Dictionary containing uncertainty metrics:
            - 'max_probability': Highest class probability
            - 'entropy': Entropy of probability distribution
            - 'avg_scale': Average scale (uncertainty) across classes
            - 'max_scale': Maximum scale across classes
            - 'predicted_class_scale': Scale of predicted class
        """
        with torch.no_grad():
            outputs = self.forward(x)
            probs = outputs['probabilities']
            scales = outputs['class_scales']
            
            # Maximum probability (confidence)
            max_prob = torch.max(probs, dim=1)[0]
            
            # Entropy of probability distribution
            entropy = -torch.sum(probs * torch.log(probs + self.eps), dim=1)
            
            # Scale-based uncertainty metrics
            avg_scale = torch.mean(scales, dim=1)
            max_scale = torch.max(scales, dim=1)[0]
            
            # Scale of predicted class
            predicted_classes = torch.argmax(probs, dim=1)
            predicted_class_scale = scales[
                torch.arange(scales.size(0)), predicted_classes
            ]
            
            return {
                'max_probability': max_prob,
                'entropy': entropy,
                'avg_scale': avg_scale,
                'max_scale': max_scale,
                'predicted_class_scale': predicted_class_scale
            } 