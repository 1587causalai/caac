"""
CAAC Model Implementations using modular components.

This module implements the CAAC model variants using the new modular architecture
while maintaining backward compatibility with the original interface.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional, Tuple, List
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from .base_model import BaseUnifiedModel
from .unified_network import UnifiedClassificationNetwork


class CAACOvRModel(BaseUnifiedModel):
    """
    CAAC One-vs-Rest Model with Cauchy Distribution
    
    This model uses the Cauchy distribution for modeling latent variables
    and implements One-vs-Rest classification strategy with proper Cauchy loss.
    """
    
    def __init__(self, input_dim: int, n_classes: int = 2,
                 representation_dim: int = 64,
                 latent_dim: Optional[int] = None,
                 feature_hidden_dims: Optional[List[int]] = None,
                 abduction_hidden_dims: Optional[List[int]] = None,
                 lr: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 device: Optional[Union[str, torch.device]] = None,
                 early_stopping_patience: Optional[int] = None,
                 early_stopping_min_delta: float = 0.0001,
                 learnable_threshold: bool = False,
                 **kwargs):
        
        super().__init__(
            input_dim=input_dim, n_classes=n_classes,
            representation_dim=representation_dim, latent_dim=latent_dim,
            feature_hidden_dims=feature_hidden_dims,
            abduction_hidden_dims=abduction_hidden_dims,
            lr=lr, batch_size=batch_size, epochs=epochs, device=device,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            **kwargs
        )
        
        self.learnable_threshold = learnable_threshold
        self.distribution_type = 'cauchy'
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        
        # Network and thresholds will be initialized in fit()
        self.network = None
        self.thresholds = None
        
    def _initialize_network(self):
        """Initialize the unified network and thresholds."""
        self.network = UnifiedClassificationNetwork(
            input_dim=self.input_dim,
            representation_dim=self.representation_dim,
            latent_dim=self.latent_dim,
            n_classes=self.n_classes,
            feature_hidden_dims=self.feature_hidden_dims,
            abduction_hidden_dims=self.abduction_hidden_dims
        ).to(self.device)
        
        # Initialize thresholds
        if self.learnable_threshold:
            self.thresholds = nn.Parameter(torch.zeros(self.n_classes, device=self.device))
        else:
            self.thresholds = torch.zeros(self.n_classes, device=self.device)
    
    def _compute_cauchy_loss(self, y_true: torch.Tensor, logits: torch.Tensor, 
                            location_param: torch.Tensor, scale_param: torch.Tensor) -> torch.Tensor:
        """
        Compute CAAC Cauchy loss function with proper distribution calculations.
        
        This implements the core CAAC algorithm:
        1. Compute class-wise Cauchy distribution parameters
        2. Calculate probabilities using Cauchy CDF
        3. Apply binary cross-entropy loss
        """
        batch_size = y_true.size(0)
        n_classes = self.n_classes
        device = y_true.device
        
        # Compute class-wise Cauchy distribution parameters
        class_locations, class_scales = self.network.action_net.compute_class_distribution_params(
            location_param, scale_param, distribution_type='cauchy'
        )
        
        # Get thresholds (fixed or learnable)
        if isinstance(self.thresholds, nn.Parameter):
            thresholds = self.thresholds
        else:
            thresholds = self.thresholds
        
        # Compute Cauchy CDF probabilities: P_k = P(S_k > C_k) = 1 - F(C_k; loc, scale)
        # For Cauchy: F(x; loc, scale) = 0.5 + (1/π) * atan((x - loc)/scale)
        # So: P_k = 0.5 - (1/π) * atan((C_k - loc)/scale)
        pi = torch.tensor(np.pi, device=device)
        normalized_thresholds = (thresholds.unsqueeze(0) - class_locations) / class_scales
        P_k = 0.5 - (1/pi) * torch.atan(normalized_thresholds)
        P_k = torch.clamp(P_k, min=1e-7, max=1-1e-7)
        
        # Convert labels to binary format
        y_binary = torch.zeros(batch_size, n_classes, device=device)
        y_binary.scatter_(1, y_true.unsqueeze(1), 1)
        
        # Compute binary cross-entropy loss
        bce_loss = -(y_binary * torch.log(P_k) + (1 - y_binary) * torch.log(1 - P_k))
        total_loss = torch.mean(bce_loss)
        
        return total_loss
    
    def fit(self, X_train: Union[np.ndarray, torch.Tensor], 
            y_train: Union[np.ndarray, torch.Tensor],
            X_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
            y_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
            verbose: int = 1) -> 'CAACOvRModel':
        """Train the CAAC OvR model with proper Cauchy loss."""
        # Convert to numpy if needed
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.cpu().numpy()
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.cpu().numpy()
            
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        self.n_classes = len(self.label_encoder.classes_)
        
        # Create validation split if not provided
        if X_val is None or y_val is None:
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
            )
        else:
            X_train_split, y_train_split = X_train, y_train_encoded
            if isinstance(X_val, torch.Tensor):
                X_val = X_val.cpu().numpy()
            if isinstance(y_val, torch.Tensor):
                y_val = y_val.cpu().numpy()
            X_val_split = X_val
            y_val_split = self.label_encoder.transform(y_val)
        
        # Initialize network
        self._initialize_network()
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_split).to(self.device)
        y_train_tensor = torch.LongTensor(y_train_split).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_split).to(self.device)
        y_val_tensor = torch.LongTensor(y_val_split).to(self.device)
        
        # Setup optimizer (include thresholds if learnable)
        params_to_optimize = list(self.network.parameters())
        if isinstance(self.thresholds, nn.Parameter):
            params_to_optimize.append(self.thresholds)
        optimizer = torch.optim.Adam(params_to_optimize, lr=self.lr)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.network.train()
            
            # Batch training
            total_loss = 0
            n_batches = 0
            
            for i in range(0, len(X_train_tensor), self.batch_size):
                batch_X = X_train_tensor[i:i+self.batch_size]
                batch_y = y_train_tensor[i:i+self.batch_size]
                
                optimizer.zero_grad()
                
                # Forward pass
                logits, location, scale = self.network(batch_X)
                
                # Compute CAAC Cauchy loss
                loss = self._compute_cauchy_loss(batch_y, logits, location, scale)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = total_loss / n_batches
            
            # Validation
            self.network.eval()
            with torch.no_grad():
                val_logits, val_location, val_scale = self.network(X_val_tensor)
                val_loss = self._compute_cauchy_loss(y_val_tensor, val_logits, val_location, val_scale).item()
            
            # Store history
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)
            
            if verbose > 0 and epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Early stopping
            if self.early_stopping_patience is not None:
                if val_loss < best_val_loss - self.early_stopping_min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.history['best_epoch'] = epoch
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        if verbose > 0:
                            print(f"Early stopping at epoch {epoch}")
                        break
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict class probabilities using Cauchy distribution."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to tensor
        if isinstance(X, np.ndarray):
            X_tensor = torch.FloatTensor(X).to(self.device)
        else:
            X_tensor = X.to(self.device)
        
        self.network.eval()
        with torch.no_grad():
            # Get network outputs
            logits, location_param, scale_param = self.network(X_tensor)
            
            # Compute class-wise Cauchy distribution parameters
            class_locations, class_scales = self.network.action_net.compute_class_distribution_params(
                location_param, scale_param, distribution_type='cauchy'
            )
            
            # Get thresholds
            if isinstance(self.thresholds, nn.Parameter):
                thresholds = self.thresholds
            else:
                thresholds = self.thresholds
            
            # Compute Cauchy CDF probabilities
            pi = torch.tensor(np.pi, device=self.device)
            normalized_thresholds = (thresholds.unsqueeze(0) - class_locations) / class_scales
            P_k = 0.5 - (1/pi) * torch.atan(normalized_thresholds)
            P_k = torch.clamp(P_k, min=1e-7, max=1-1e-7)
            
            # Normalize to sum to 1 (convert from OvR to multiclass probabilities)
            probabilities = P_k / P_k.sum(dim=1, keepdim=True)
        
        return probabilities.cpu().numpy()
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict class labels."""
        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        return self.label_encoder.inverse_transform(predictions)


class CAACOvRGaussianModel(CAACOvRModel):
    """
    CAAC One-vs-Rest Model with Gaussian Distribution
    
    Similar to CAACOvRModel but uses Gaussian distribution for latent variables.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distribution_type = 'gaussian'
    
    def _compute_gaussian_loss(self, y_true: torch.Tensor, logits: torch.Tensor, 
                              location_param: torch.Tensor, scale_param: torch.Tensor) -> torch.Tensor:
        """
        Compute CAAC Gaussian loss function with proper distribution calculations.
        """
        batch_size = y_true.size(0)
        n_classes = self.n_classes
        device = y_true.device
        
        # Compute class-wise Gaussian distribution parameters
        class_locations, class_stds = self.network.action_net.compute_class_distribution_params(
            location_param, scale_param, distribution_type='gaussian'
        )
        
        # Get thresholds (fixed or learnable)
        if isinstance(self.thresholds, nn.Parameter):
            thresholds = self.thresholds
        else:
            thresholds = self.thresholds
        
        # Compute Gaussian CDF probabilities: P_k = P(S_k > C_k) = 1 - Φ((C_k - μ)/σ)
        normalized_thresholds = (thresholds.unsqueeze(0) - class_locations) / class_stds
        
        from torch.distributions import Normal
        standard_normal = Normal(0, 1)
        P_k = 1 - standard_normal.cdf(normalized_thresholds)
        P_k = torch.clamp(P_k, min=1e-7, max=1-1e-7)
        
        # Convert labels to binary format
        y_binary = torch.zeros(batch_size, n_classes, device=device)
        y_binary.scatter_(1, y_true.unsqueeze(1), 1)
        
        # Compute binary cross-entropy loss
        bce_loss = -(y_binary * torch.log(P_k) + (1 - y_binary) * torch.log(1 - P_k))
        total_loss = torch.mean(bce_loss)
        
        return total_loss
    
    def fit(self, X_train: Union[np.ndarray, torch.Tensor], 
            y_train: Union[np.ndarray, torch.Tensor],
            X_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
            y_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
            verbose: int = 1) -> 'CAACOvRGaussianModel':
        """Train the CAAC OvR Gaussian model."""
        # Same as parent but use Gaussian loss
        # Convert to numpy if needed
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.cpu().numpy()
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.cpu().numpy()
            
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        self.n_classes = len(self.label_encoder.classes_)
        
        # Create validation split if not provided
        if X_val is None or y_val is None:
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
            )
        else:
            X_train_split, y_train_split = X_train, y_train_encoded
            if isinstance(X_val, torch.Tensor):
                X_val = X_val.cpu().numpy()
            if isinstance(y_val, torch.Tensor):
                y_val = y_val.cpu().numpy()
            X_val_split = X_val
            y_val_split = self.label_encoder.transform(y_val)
        
        # Initialize network
        self._initialize_network()
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_split).to(self.device)
        y_train_tensor = torch.LongTensor(y_train_split).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_split).to(self.device)
        y_val_tensor = torch.LongTensor(y_val_split).to(self.device)
        
        # Setup optimizer
        params_to_optimize = list(self.network.parameters())
        if isinstance(self.thresholds, nn.Parameter):
            params_to_optimize.append(self.thresholds)
        optimizer = torch.optim.Adam(params_to_optimize, lr=self.lr)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.network.train()
            
            # Batch training
            total_loss = 0
            n_batches = 0
            
            for i in range(0, len(X_train_tensor), self.batch_size):
                batch_X = X_train_tensor[i:i+self.batch_size]
                batch_y = y_train_tensor[i:i+self.batch_size]
                
                optimizer.zero_grad()
                
                # Forward pass
                logits, location, scale = self.network(batch_X)
                
                # Compute CAAC Gaussian loss
                loss = self._compute_gaussian_loss(batch_y, logits, location, scale)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = total_loss / n_batches
            
            # Validation
            self.network.eval()
            with torch.no_grad():
                val_logits, val_location, val_scale = self.network(X_val_tensor)
                val_loss = self._compute_gaussian_loss(y_val_tensor, val_logits, val_location, val_scale).item()
            
            # Store history
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)
            
            if verbose > 0 and epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Early stopping
            if self.early_stopping_patience is not None:
                if val_loss < best_val_loss - self.early_stopping_min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.history['best_epoch'] = epoch
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        if verbose > 0:
                            print(f"Early stopping at epoch {epoch}")
                        break
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict class probabilities using Gaussian distribution."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to tensor
        if isinstance(X, np.ndarray):
            X_tensor = torch.FloatTensor(X).to(self.device)
        else:
            X_tensor = X.to(self.device)
        
        self.network.eval()
        with torch.no_grad():
            # Get network outputs
            logits, location_param, scale_param = self.network(X_tensor)
            
            # Compute class-wise Gaussian distribution parameters
            class_locations, class_stds = self.network.action_net.compute_class_distribution_params(
                location_param, scale_param, distribution_type='gaussian'
            )
            
            # Get thresholds
            if isinstance(self.thresholds, nn.Parameter):
                thresholds = self.thresholds
            else:
                thresholds = self.thresholds
            
            # Compute Gaussian CDF probabilities
            normalized_thresholds = (thresholds.unsqueeze(0) - class_locations) / class_stds
            
            from torch.distributions import Normal
            standard_normal = Normal(0, 1)
            P_k = 1 - standard_normal.cdf(normalized_thresholds)
            P_k = torch.clamp(P_k, min=1e-7, max=1-1e-7)
            
            # Normalize to sum to 1
            probabilities = P_k / P_k.sum(dim=1, keepdim=True)
        
        return probabilities.cpu().numpy() 