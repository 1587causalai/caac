"""
Core implementation of the CAAC (Shared Latent Cauchy Vector OvR Classifier).

This module maintains backward compatibility while using the new modular architecture.
All classes are now implemented using the new components but preserve the original API.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Tuple, Dict, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Import new modular implementations
from .caac_models import CAACOvRModel as NewCAACOvRModel, CAACOvRGaussianModel as NewCAACOvRGaussianModel
from .mlp_models import SoftmaxMLPModel as NewSoftmaxMLPModel, OvRCrossEntropyMLPModel as NewOvRCrossEntropyMLPModel, CrammerSingerMLPModel as NewCrammerSingerMLPModel

# Re-export components for backward compatibility
from .components import FeatureNetwork, AbductionNetwork, ActionNetwork
from .unified_network import UnifiedClassificationNetwork


class CAACOvRModel(BaseEstimator, ClassifierMixin):
    """
    CAAC OvR Classification Model - Backward Compatible Wrapper
    
    This class maintains the original interface while using the new modular implementation.
    """
    
    def __init__(self, input_dim, 
                 representation_dim=64, 
                 latent_dim=None,
                 n_classes=2,
                 feature_hidden_dims=[64], 
                 abduction_hidden_dims=[128, 64], 
                 lr=0.001, batch_size=32, epochs=100, device=None,
                 early_stopping_patience=None, early_stopping_min_delta=0.0001,
                 learnable_thresholds=False,
                 uniqueness_constraint=False,
                 uniqueness_samples=10,
                 uniqueness_weight=0.1):
        
        # Store parameters for compatibility
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim if latent_dim is not None else representation_dim
        self.n_classes = n_classes
        self.feature_hidden_dims = feature_hidden_dims
        self.abduction_hidden_dims = abduction_hidden_dims
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.learnable_thresholds = learnable_thresholds
        self.uniqueness_constraint = uniqueness_constraint
        self.uniqueness_samples = uniqueness_samples
        self.uniqueness_weight = uniqueness_weight
        self.device_str = str(device)
        
        # Initialize the new model
        self._model = NewCAACOvRModel(
            input_dim=input_dim,
            n_classes=n_classes,
            representation_dim=representation_dim,
            latent_dim=latent_dim,
            feature_hidden_dims=feature_hidden_dims,
            abduction_hidden_dims=abduction_hidden_dims,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            learnable_threshold=learnable_thresholds
        )
        
        # Compatibility attributes
        self.label_encoder = self._model.label_encoder
        self.history = {'train_loss': [], 'val_loss': [], 'train_time': 0, 'best_epoch': 0}
    
    def _setup_model_optimizer(self):
        """Backward compatibility method - no longer needed."""
        pass
    
    def compute_loss(self, y_true, logits, location_param, scale_param):
        """Backward compatibility method for loss computation."""
        # Convert to tensors if needed
        if isinstance(y_true, np.ndarray):
            y_true = torch.LongTensor(y_true).to(self._model.device)
        return F.cross_entropy(logits, y_true)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        """Fit the model with backward compatibility."""
        result = self._model.fit(X_train, y_train, X_val, y_val, verbose)
        
        # Update compatibility attributes
        self.label_encoder = self._model.label_encoder
        self.history = self._model.history
        self.n_classes = self._model.n_classes
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self._model.predict_proba(X)
    
    def predict(self, X):
        """Predict class labels."""
        return self._model.predict(X)
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            'input_dim': self.input_dim,
            'representation_dim': self.representation_dim,
            'latent_dim': self.latent_dim,
            'n_classes': self.n_classes,
            'feature_hidden_dims': self.feature_hidden_dims,
            'abduction_hidden_dims': self.abduction_hidden_dims,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta,
            'learnable_thresholds': self.learnable_thresholds,
            'uniqueness_constraint': self.uniqueness_constraint,
            'uniqueness_samples': self.uniqueness_samples,
            'uniqueness_weight': self.uniqueness_weight,
            'device_str': self.device_str
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class SoftmaxMLPModel(BaseEstimator, ClassifierMixin):
    """
    Softmax MLP Model - Backward Compatible Wrapper
    """
    
    def __init__(self, input_dim, 
                 representation_dim=64, 
                 latent_dim=None,
                 n_classes=2,
                 feature_hidden_dims=[64], 
                 abduction_hidden_dims=[128, 64], 
                 lr=0.001, batch_size=32, epochs=100, device=None,
                 early_stopping_patience=None, early_stopping_min_delta=0.0001):
        
        # Store parameters for compatibility
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim if latent_dim is not None else representation_dim
        self.n_classes = n_classes
        self.feature_hidden_dims = feature_hidden_dims
        self.abduction_hidden_dims = abduction_hidden_dims
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device_str = str(device)
        
        # Initialize the new model
        self._model = NewSoftmaxMLPModel(
            input_dim=input_dim,
            n_classes=n_classes,
            representation_dim=representation_dim,
            latent_dim=latent_dim,
            feature_hidden_dims=feature_hidden_dims,
            abduction_hidden_dims=abduction_hidden_dims,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta
        )
        
        # Compatibility attributes
        self.label_encoder = self._model.label_encoder
        self.history = {'train_loss': [], 'val_loss': [], 'train_time': 0, 'best_epoch': 0}
    
    def _setup_model_optimizer(self):
        """Backward compatibility method."""
        pass
    
    def compute_loss(self, y_true, logits, location_param, scale_param):
        """Backward compatibility method for loss computation."""
        if isinstance(y_true, np.ndarray):
            y_true = torch.LongTensor(y_true).to(self._model.device)
        return F.cross_entropy(logits, y_true)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        """Fit the model with backward compatibility."""
        result = self._model.fit(X_train, y_train, X_val, y_val, verbose)
        
        # Update compatibility attributes
        self.label_encoder = self._model.label_encoder
        self.history = self._model.history
        self.n_classes = self._model.n_classes
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self._model.predict_proba(X)
    
    def predict(self, X):
        """Predict class labels."""
        return self._model.predict(X)
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            'input_dim': self.input_dim,
            'representation_dim': self.representation_dim,
            'latent_dim': self.latent_dim,
            'n_classes': self.n_classes,
            'feature_hidden_dims': self.feature_hidden_dims,
            'abduction_hidden_dims': self.abduction_hidden_dims,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta,
            'device_str': self.device_str
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class OvRCrossEntropyMLPModel(BaseEstimator, ClassifierMixin):
    """
    OvR Cross Entropy MLP Model - Backward Compatible Wrapper
    """
    
    def __init__(self, input_dim, 
                 representation_dim=64, 
                 latent_dim=None,
                 n_classes=2,
                 feature_hidden_dims=[64], 
                 abduction_hidden_dims=[128, 64], 
                 lr=0.001, batch_size=32, epochs=100, device=None,
                 early_stopping_patience=None, early_stopping_min_delta=0.0001):
        
        # Store parameters for compatibility
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim if latent_dim is not None else representation_dim
        self.n_classes = n_classes
        self.feature_hidden_dims = feature_hidden_dims
        self.abduction_hidden_dims = abduction_hidden_dims
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device_str = str(device)
        
        # Initialize the new model
        self._model = NewOvRCrossEntropyMLPModel(
            input_dim=input_dim,
            n_classes=n_classes,
            representation_dim=representation_dim,
            latent_dim=latent_dim,
            feature_hidden_dims=feature_hidden_dims,
            abduction_hidden_dims=abduction_hidden_dims,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta
        )
        
        # Compatibility attributes
        self.label_encoder = self._model.label_encoder
        self.history = {'train_loss': [], 'val_loss': [], 'train_time': 0, 'best_epoch': 0}
    
    def _setup_model_optimizer(self):
        """Backward compatibility method."""
        pass
    
    def compute_loss(self, y_true, logits, location_param, scale_param):
        """Backward compatibility method for loss computation."""
        if isinstance(y_true, np.ndarray):
            y_true = torch.LongTensor(y_true).to(self._model.device)
        return F.cross_entropy(logits, y_true)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        """Fit the model with backward compatibility."""
        result = self._model.fit(X_train, y_train, X_val, y_val, verbose)
        
        # Update compatibility attributes
        self.label_encoder = self._model.label_encoder
        self.history = self._model.history
        self.n_classes = self._model.n_classes
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self._model.predict_proba(X)
    
    def predict(self, X):
        """Predict class labels."""
        return self._model.predict(X)
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            'input_dim': self.input_dim,
            'representation_dim': self.representation_dim,
            'latent_dim': self.latent_dim,
            'n_classes': self.n_classes,
            'feature_hidden_dims': self.feature_hidden_dims,
            'abduction_hidden_dims': self.abduction_hidden_dims,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta,
            'device_str': self.device_str
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class CAACOvRGaussianModel(BaseEstimator, ClassifierMixin):
    """
    CAAC OvR Gaussian Model - Backward Compatible Wrapper
    """
    
    def __init__(self, input_dim, 
                 representation_dim=64, 
                 latent_dim=None,
                 n_classes=2,
                 feature_hidden_dims=[64], 
                 abduction_hidden_dims=[128, 64], 
                 lr=0.001, batch_size=32, epochs=100, device=None,
                 early_stopping_patience=None, early_stopping_min_delta=0.0001,
                 learnable_thresholds=False,
                 uniqueness_constraint=False,
                 uniqueness_samples=10,
                 uniqueness_weight=0.1):
        
        # Store parameters for compatibility
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim if latent_dim is not None else representation_dim
        self.n_classes = n_classes
        self.feature_hidden_dims = feature_hidden_dims
        self.abduction_hidden_dims = abduction_hidden_dims
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.learnable_thresholds = learnable_thresholds
        self.uniqueness_constraint = uniqueness_constraint
        self.uniqueness_samples = uniqueness_samples
        self.uniqueness_weight = uniqueness_weight
        self.device_str = str(device)
        
        # Initialize the new model
        self._model = NewCAACOvRGaussianModel(
            input_dim=input_dim,
            n_classes=n_classes,
            representation_dim=representation_dim,
            latent_dim=latent_dim,
            feature_hidden_dims=feature_hidden_dims,
            abduction_hidden_dims=abduction_hidden_dims,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta
        )
        
        # Compatibility attributes
        self.label_encoder = self._model.label_encoder
        self.history = {'train_loss': [], 'val_loss': [], 'train_time': 0, 'best_epoch': 0}
    
    def _setup_model_optimizer(self):
        """Backward compatibility method."""
        pass
    
    def compute_loss(self, y_true, logits, location_param, scale_param):
        """Backward compatibility method for loss computation."""
        if isinstance(y_true, np.ndarray):
            y_true = torch.LongTensor(y_true).to(self._model.device)
        return F.cross_entropy(logits, y_true)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        """Fit the model with backward compatibility."""
        result = self._model.fit(X_train, y_train, X_val, y_val, verbose)
        
        # Update compatibility attributes
        self.label_encoder = self._model.label_encoder
        self.history = self._model.history
        self.n_classes = self._model.n_classes
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self._model.predict_proba(X)
    
    def predict(self, X):
        """Predict class labels."""
        return self._model.predict(X)
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            'input_dim': self.input_dim,
            'representation_dim': self.representation_dim,
            'latent_dim': self.latent_dim,
            'n_classes': self.n_classes,
            'feature_hidden_dims': self.feature_hidden_dims,
            'abduction_hidden_dims': self.abduction_hidden_dims,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta,
            'learnable_thresholds': self.learnable_thresholds,
            'uniqueness_constraint': self.uniqueness_constraint,
            'uniqueness_samples': self.uniqueness_samples,
            'uniqueness_weight': self.uniqueness_weight,
            'device_str': self.device_str
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class CrammerSingerMLPModel(BaseEstimator, ClassifierMixin):
    """
    Crammer Singer MLP Model - Backward Compatible Wrapper
    """
    
    def __init__(self, input_dim, 
                 representation_dim=64, 
                 latent_dim=None,
                 n_classes=2,
                 feature_hidden_dims=[64], 
                 abduction_hidden_dims=[128, 64], 
                 lr=0.001, batch_size=32, epochs=100, device=None,
                 early_stopping_patience=None, early_stopping_min_delta=0.0001):
        
        # Store parameters for compatibility
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim if latent_dim is not None else representation_dim
        self.n_classes = n_classes
        self.feature_hidden_dims = feature_hidden_dims
        self.abduction_hidden_dims = abduction_hidden_dims
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device_str = str(device)
        
        # Initialize the new model
        self._model = NewCrammerSingerMLPModel(
            input_dim=input_dim,
            n_classes=n_classes,
            representation_dim=representation_dim,
            latent_dim=latent_dim,
            feature_hidden_dims=feature_hidden_dims,
            abduction_hidden_dims=abduction_hidden_dims,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta
        )
        
        # Compatibility attributes
        self.label_encoder = self._model.label_encoder
        self.history = {'train_loss': [], 'val_loss': [], 'train_time': 0, 'best_epoch': 0}
    
    def _setup_model_optimizer(self):
        """Backward compatibility method."""
        pass
    
    def compute_loss(self, y_true, logits, location_param, scale_param):
        """Backward compatibility method for loss computation."""
        if isinstance(y_true, np.ndarray):
            y_true = torch.LongTensor(y_true).to(self._model.device)
        return F.cross_entropy(logits, y_true)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        """Fit the model with backward compatibility."""
        result = self._model.fit(X_train, y_train, X_val, y_val, verbose)
        
        # Update compatibility attributes
        self.label_encoder = self._model.label_encoder
        self.history = self._model.history
        self.n_classes = self._model.n_classes
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self._model.predict_proba(X)
    
    def predict(self, X):
        """Predict class labels."""
        return self._model.predict(X)
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            'input_dim': self.input_dim,
            'representation_dim': self.representation_dim,
            'latent_dim': self.latent_dim,
            'n_classes': self.n_classes,
            'feature_hidden_dims': self.feature_hidden_dims,
            'abduction_hidden_dims': self.abduction_hidden_dims,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta,
            'device_str': self.device_str
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

