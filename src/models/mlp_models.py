"""
MLP Model Implementations using modular components.

This module implements various MLP-based classification models using the unified
architecture while maintaining backward compatibility with the original interface.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional, List
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from .base_model import BaseUnifiedModel
from .unified_network import UnifiedClassificationNetwork


class SoftmaxMLPModel(BaseUnifiedModel):
    """
    Multi-Layer Perceptron with Softmax Classification
    
    This model uses the unified architecture but applies softmax classification
    at the output layer, treating it as a standard neural network classifier.
    
    Args:
        input_dim (int): Input feature dimension
        n_classes (int): Number of classes
        representation_dim (int): Representation dimension
        latent_dim (int, optional): Latent dimension (defaults to representation_dim)
        feature_hidden_dims (List[int], optional): Hidden dimensions for feature network
        abduction_hidden_dims (List[int], optional): Hidden dimensions for abduction network
        lr (float): Learning rate
        batch_size (int): Batch size
        epochs (int): Number of training epochs
        device (Union[str, torch.device], optional): Device to use
        early_stopping_patience (int, optional): Early stopping patience
        early_stopping_min_delta (float): Early stopping minimum delta
        **kwargs: Additional parameters
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
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        
        # Network will be initialized in fit()
        self.network = None
        
    def _initialize_network(self):
        """Initialize the unified network."""
        self.network = UnifiedClassificationNetwork(
            input_dim=self.input_dim,
            representation_dim=self.representation_dim,
            latent_dim=self.latent_dim,
            n_classes=self.n_classes,
            feature_hidden_dims=self.feature_hidden_dims,
            abduction_hidden_dims=self.abduction_hidden_dims
        ).to(self.device)
        
    def fit(self, X_train: Union[np.ndarray, torch.Tensor], 
            y_train: Union[np.ndarray, torch.Tensor],
            X_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
            y_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
            verbose: int = 1) -> 'SoftmaxMLPModel':
        """
        Train the Softmax MLP model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            verbose: Verbosity level
            
        Returns:
            self: The fitted model
        """
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
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        
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
                
                # Forward pass - only use logits for standard softmax classification
                logits, _, _ = self.network(batch_X)
                
                # Compute loss using cross-entropy
                loss = F.cross_entropy(logits, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = total_loss / n_batches
            
            # Validation
            self.network.eval()
            with torch.no_grad():
                val_logits, _, _ = self.network(X_val_tensor)
                val_loss = F.cross_entropy(val_logits, y_val_tensor).item()
            
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
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to tensor
        if isinstance(X, np.ndarray):
            X_tensor = torch.FloatTensor(X).to(self.device)
        else:
            X_tensor = X.to(self.device)
        
        self.network.eval()
        with torch.no_grad():
            logits, _, _ = self.network(X_tensor)
            predictions = torch.argmax(logits, dim=1)
        
        # Convert back to original labels
        predictions_np = predictions.cpu().numpy()
        return self.label_encoder.inverse_transform(predictions_np)
    
    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Predicted class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to tensor
        if isinstance(X, np.ndarray):
            X_tensor = torch.FloatTensor(X).to(self.device)
        else:
            X_tensor = X.to(self.device)
        
        self.network.eval()
        with torch.no_grad():
            logits, _, _ = self.network(X_tensor)
            probabilities = F.softmax(logits, dim=1)
        
        return probabilities.cpu().numpy()


class OvRCrossEntropyMLPModel(SoftmaxMLPModel):
    """
    MLP with One-vs-Rest Cross Entropy Classification
    
    This model implements One-vs-Rest strategy using cross-entropy loss
    for each binary classifier.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ovr_classifiers = None
        
    def _train_ovr_classifier(self, X_tensor: torch.Tensor, y_binary: torch.Tensor, 
                             class_idx: int) -> nn.Module:
        """Train a single binary classifier for one class vs rest."""
        # Create a simplified network for binary classification
        classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.feature_hidden_dims[0] if self.feature_hidden_dims else 64),
            nn.ReLU(),
            nn.Linear(self.feature_hidden_dims[0] if self.feature_hidden_dims else 64, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        optimizer = torch.optim.Adam(classifier.parameters(), lr=self.lr)
        
        # Training loop for this binary classifier
        for epoch in range(min(self.epochs, 50)):  # Reduced epochs for OvR
            classifier.train()
            
            for i in range(0, len(X_tensor), self.batch_size):
                batch_X = X_tensor[i:i+self.batch_size]
                batch_y = y_binary[i:i+self.batch_size]
                
                optimizer.zero_grad()
                
                outputs = classifier(batch_X).squeeze()
                loss = F.binary_cross_entropy(outputs, batch_y.float())
                
                loss.backward()
                optimizer.step()
        
        return classifier
    
    def fit(self, X_train: Union[np.ndarray, torch.Tensor], 
            y_train: Union[np.ndarray, torch.Tensor],
            X_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
            y_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
            verbose: int = 1) -> 'OvRCrossEntropyMLPModel':
        """
        Train the OvR Cross Entropy MLP model.
        """
        # Convert to numpy if needed
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.cpu().numpy()
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.cpu().numpy()
            
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        self.n_classes = len(self.label_encoder.classes_)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        
        # Train one binary classifier for each class
        self.ovr_classifiers = []
        
        for class_idx in range(self.n_classes):
            if verbose > 0:
                print(f"Training classifier for class {class_idx}")
            
            # Create binary labels (1 for current class, 0 for others)
            y_binary = (y_train_encoded == class_idx).astype(int)
            y_binary_tensor = torch.LongTensor(y_binary).to(self.device)
            
            # Train binary classifier
            classifier = self._train_ovr_classifier(X_train_tensor, y_binary_tensor, class_idx)
            self.ovr_classifiers.append(classifier)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict using OvR strategy."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to tensor
        if isinstance(X, np.ndarray):
            X_tensor = torch.FloatTensor(X).to(self.device)
        else:
            X_tensor = X.to(self.device)
        
        # Get predictions from all binary classifiers
        scores = torch.zeros(X.shape[0], self.n_classes, device=self.device)
        
        for class_idx, classifier in enumerate(self.ovr_classifiers):
            classifier.eval()
            with torch.no_grad():
                class_scores = classifier(X_tensor).squeeze()
                scores[:, class_idx] = class_scores
        
        # Choose class with highest score
        predictions = torch.argmax(scores, dim=1)
        
        # Convert back to original labels
        predictions_np = predictions.cpu().numpy()
        return self.label_encoder.inverse_transform(predictions_np)
    
    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict probabilities using OvR strategy."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to tensor
        if isinstance(X, np.ndarray):
            X_tensor = torch.FloatTensor(X).to(self.device)
        else:
            X_tensor = X.to(self.device)
        
        # Get scores from all binary classifiers
        scores = torch.zeros(X.shape[0], self.n_classes, device=self.device)
        
        for class_idx, classifier in enumerate(self.ovr_classifiers):
            classifier.eval()
            with torch.no_grad():
                class_scores = classifier(X_tensor).squeeze()
                scores[:, class_idx] = class_scores
        
        # Convert to probabilities using softmax
        probabilities = F.softmax(scores, dim=1)
        
        return probabilities.cpu().numpy()


class CrammerSingerMLPModel(SoftmaxMLPModel):
    """
    MLP with Crammer & Singer Multi-class Hinge Loss
    
    This model uses the Crammer & Singer formulation of multi-class hinge loss
    for training.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _crammer_singer_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Crammer & Singer multi-class hinge loss.
        
        Loss = sum_y≠t max(0, 1 + f_y - f_t) where t is the true class
        """
        batch_size, n_classes = logits.shape
        
        # Get the logit for the true class
        true_class_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute margin loss for each class
        margins = 1 + logits - true_class_logits.unsqueeze(1)
        
        # Set margin for true class to 0 (we don't want to penalize correct predictions)
        margins.scatter_(1, targets.unsqueeze(1), 0)
        
        # Apply hinge loss (max(0, margin))
        hinge_loss = torch.clamp(margins, min=0)
        
        # Sum over all classes except the true class, then average over batch
        loss = hinge_loss.sum(dim=1).mean()
        
        return loss
    
    def fit(self, X_train: Union[np.ndarray, torch.Tensor], 
            y_train: Union[np.ndarray, torch.Tensor],
            X_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
            y_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
            verbose: int = 1) -> 'CrammerSingerMLPModel':
        """
        Train the Crammer & Singer MLP model.
        """
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
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        
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
                logits, _, _ = self.network(batch_X)
                
                # Compute Crammer & Singer loss
                loss = self._crammer_singer_loss(logits, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = total_loss / n_batches
            
            # Validation using cross-entropy for comparison
            self.network.eval()
            with torch.no_grad():
                val_logits, _, _ = self.network(X_val_tensor)
                val_loss = F.cross_entropy(val_logits, y_val_tensor).item()
            
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