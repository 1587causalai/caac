"""
Base model interface for CAAC models.

This module defines the common interface that all CAAC models should implement,
ensuring consistency across different model variants.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin


class BaseCAACModel(BaseEstimator, ClassifierMixin, ABC):
    """
    Abstract base class for all CAAC model variants.
    
    This class defines the standard interface that all CAAC models must implement,
    ensuring consistency and enabling seamless model swapping in experiments.
    
    Inherits from scikit-learn's BaseEstimator and ClassifierMixin to ensure
    compatibility with scikit-learn's ecosystem.
    """
    
    def __init__(self, input_dim: int, n_classes: int = 2, **kwargs):
        """
        Initialize the base model.
        
        Args:
            input_dim (int): Input feature dimension
            n_classes (int): Number of classes
            **kwargs: Additional model-specific parameters
        """
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X_train: Union[np.ndarray, torch.Tensor], 
            y_train: Union[np.ndarray, torch.Tensor],
            X_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
            y_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
            verbose: int = 1) -> 'BaseCAACModel':
        """
        Train the model on the given data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            verbose: Verbosity level
            
        Returns:
            self: The fitted model
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Predicted class labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Predicted class probabilities
        """
        pass
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.
        
        Args:
            deep: If True, will return the parameters for this estimator
                 and contained subobjects that are estimators.
                 
        Returns:
            Dict[str, Any]: Parameter names mapped to their values
        """
        # Get all attributes that don't start with underscore
        params = {}
        for key in dir(self):
            if not key.startswith('_') and not callable(getattr(self, key)):
                value = getattr(self, key)
                # Skip complex objects that aren't basic types
                if isinstance(value, (int, float, str, bool, list, tuple, type(None))):
                    params[key] = value
        return params
    
    def set_params(self, **params) -> 'BaseCAACModel':
        """
        Set the parameters of this estimator.
        
        Args:
            **params: Parameters to set
            
        Returns:
            self: The estimator instance
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key} for estimator {type(self).__name__}")
        return self
    
    def get_training_history(self) -> Dict[str, Any]:
        """
        Get training history if available.
        
        Returns:
            Dict[str, Any]: Training history containing losses, metrics, etc.
        """
        return getattr(self, 'history', {})
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'BaseCAACModel':
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            BaseCAACModel: The loaded model
        """
        import pickle
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def __repr__(self) -> str:
        """String representation of the model."""
        params = self.get_params()
        param_str = ', '.join([f"{k}={v}" for k, v in params.items()])
        return f"{self.__class__.__name__}({param_str})"


class BaseUnifiedModel(BaseCAACModel):
    """
    Base class for unified network models.
    
    This class provides common functionality for models that use the
    unified FeatureNetwork -> AbductionNetwork -> ActionNetwork architecture.
    """
    
    def __init__(self, input_dim: int, n_classes: int = 2,
                 representation_dim: int = 64,
                 latent_dim: Optional[int] = None,
                 feature_hidden_dims: Optional[list] = None,
                 abduction_hidden_dims: Optional[list] = None,
                 lr: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 device: Optional[Union[str, torch.device]] = None,
                 early_stopping_patience: Optional[int] = None,
                 early_stopping_min_delta: float = 0.0001,
                 **kwargs):
        """
        Initialize the unified model.
        
        Args:
            input_dim: Input feature dimension
            n_classes: Number of classes
            representation_dim: Representation dimension
            latent_dim: Latent dimension (defaults to representation_dim)
            feature_hidden_dims: Hidden dimensions for feature network
            abduction_hidden_dims: Hidden dimensions for abduction network
            lr: Learning rate
            batch_size: Batch size
            epochs: Number of training epochs
            device: Device to use for training
            early_stopping_patience: Early stopping patience
            early_stopping_min_delta: Early stopping minimum delta
            **kwargs: Additional parameters
        """
        super().__init__(input_dim, n_classes, **kwargs)
        
        # Network architecture parameters
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim if latent_dim is not None else representation_dim
        self.feature_hidden_dims = feature_hidden_dims or [64]
        self.abduction_hidden_dims = abduction_hidden_dims or [128, 64]
        
        # Training parameters
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
            
        # Store device as string for serialization
        self.device_str = str(self.device)
        
        # Initialize training history
        self.history = {
            'train_loss': [], 
            'val_loss': [], 
            'train_time': 0, 
            'best_epoch': 0
        } 