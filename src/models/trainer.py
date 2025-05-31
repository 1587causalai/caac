"""
Trainer for Shared Cauchy OvR Classifier

This module provides training, validation, and evaluation functionality
for the shared latent Cauchy vector based OvR classifier.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from .shared_cauchy_ovr import SharedCauchyOvRClassifier
from .loss_functions import create_loss_function, UncertaintyRegularizedLoss


class SharedCauchyOvRTrainer:
    """
    Trainer class for Shared Cauchy OvR Classifier
    
    Handles training, validation, evaluation, and uncertainty analysis
    """
    
    def __init__(
        self,
        model: SharedCauchyOvRClassifier,
        loss_function: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'auto',
        log_level: str = 'INFO'
    ):
        """
        Initialize trainer
        
        Args:
            model: SharedCauchyOvRClassifier instance
            loss_function: Loss function for training
            optimizer: Optimizer (default: Adam)
            scheduler: Learning rate scheduler
            device: Device to use ('auto', 'cpu', 'cuda')
            log_level: Logging level
        """
        self.model = model
        self.loss_function = loss_function
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Set optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=1e-3, 
                weight_decay=1e-4
            )
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        
        # Set up logging
        logging.basicConfig(level=getattr(logging, log_level))
        self.logger = logging.getLogger(__name__)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_accuracy = 0.0
        self.best_model_state = None
    
    def train_epoch(
        self, 
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Progress bar
        pbar = tqdm(
            train_loader, 
            desc=f'Epoch {epoch+1} Training',
            leave=False
        )
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            probabilities = outputs['probabilities']
            
            # Compute loss
            if isinstance(self.loss_function, UncertaintyRegularizedLoss):
                # Uncertainty regularized loss
                loss_dict = self.loss_function(
                    probabilities, targets, outputs['class_scales']
                )
                loss = loss_dict['total_loss']
            else:
                # Standard loss
                loss = self.loss_function(probabilities, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(probabilities, dim=1)
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)
            
            # Update progress bar
            current_accuracy = correct_predictions / total_samples
            pbar.set_postfix({
                'Loss': f'{total_loss/(batch_idx+1):.4f}',
                'Acc': f'{current_accuracy:.4f}'
            })
        
        # Compute epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_samples
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy
        }
    
    def validate_epoch(
        self, 
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Validate for one epoch
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation', leave=False)
            
            for inputs, targets in pbar:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                probabilities = outputs['probabilities']
                
                # Compute loss
                if isinstance(self.loss_function, UncertaintyRegularizedLoss):
                    loss_dict = self.loss_function(
                        probabilities, targets, outputs['class_scales']
                    )
                    loss = loss_dict['total_loss']
                else:
                    loss = self.loss_function(probabilities, targets)
                
                # Statistics
                total_loss += loss.item()
                predictions = torch.argmax(probabilities, dim=1)
                correct_predictions += (predictions == targets).sum().item()
                total_samples += targets.size(0)
                
                # Update progress bar
                current_accuracy = correct_predictions / total_samples
                pbar.set_postfix({
                    'Loss': f'{total_loss/(len(pbar)):.4f}',
                    'Acc': f'{current_accuracy:.4f}'
                })
        
        # Compute epoch metrics
        epoch_loss = total_loss / len(val_loader)
        epoch_accuracy = correct_predictions / total_samples
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        early_stopping_patience: int = 10,
        save_best_model: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            early_stopping_patience: Early stopping patience
            save_best_model: Whether to save best model state
            
        Returns:
            Training history dictionary
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader)
            else:
                val_metrics = {'loss': 0.0, 'accuracy': 0.0}
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Logging
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # Save best model
            if save_best_model and val_loader is not None:
                if val_metrics['accuracy'] > self.best_val_accuracy:
                    self.best_val_accuracy = val_metrics['accuracy']
                    self.best_model_state = self.model.state_dict().copy()
                    self.logger.info(f"New best validation accuracy: {self.best_val_accuracy:.4f}")
            
            # Early stopping
            if val_loader is not None and early_stopping_patience > 0:
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Load best model if available
        if save_best_model and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.logger.info("Loaded best model state")
        
        return self.history
    
    def evaluate(
        self, 
        test_loader: DataLoader,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the model on test data
        
        Args:
            test_loader: Test data loader
            class_names: Optional class names for reporting
            
        Returns:
            Dictionary with evaluation results
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_uncertainties = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Evaluation', leave=False)
            
            for inputs, targets in pbar:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                uncertainties = self.model.get_uncertainty_metrics(inputs)
                
                # Collect results
                predictions = torch.argmax(outputs['probabilities'], dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(outputs['probabilities'].cpu().numpy())
                all_uncertainties.extend(uncertainties['predicted_class_scale'].cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        all_uncertainties = np.array(all_uncertainties)
        
        # Compute metrics
        accuracy = (all_predictions == all_targets).mean()
        
        # Classification report
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(self.model.num_classes)]
        
        classification_rep = classification_report(
            all_targets, all_predictions, 
            target_names=class_names, 
            output_dict=True
        )
        
        return {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities,
            'uncertainties': all_uncertainties,
            'classification_report': classification_rep
        }
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history
        
        Args:
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.history['train_accuracy'], label='Train Acc', color='blue')
        axes[0, 1].plot(self.history['val_accuracy'], label='Val Acc', color='red')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(self.history['learning_rates'], color='green')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Remove empty subplot
        axes[1, 1].remove()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def analyze_uncertainty(
        self, 
        test_loader: DataLoader,
        save_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Analyze uncertainty predictions
        
        Args:
            test_loader: Test data loader
            save_path: Optional path to save plots
            
        Returns:
            Dictionary with uncertainty analysis results
        """
        self.model.eval()
        
        all_uncertainties = []
        all_confidences = []
        all_correct = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc='Uncertainty Analysis'):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                uncertainties = self.model.get_uncertainty_metrics(inputs)
                
                predictions = torch.argmax(outputs['probabilities'], dim=1)
                max_probs = torch.max(outputs['probabilities'], dim=1)[0]
                
                all_uncertainties.extend(uncertainties['predicted_class_scale'].cpu().numpy())
                all_confidences.extend(max_probs.cpu().numpy())
                all_correct.extend((predictions == targets).cpu().numpy())
        
        all_uncertainties = np.array(all_uncertainties)
        all_confidences = np.array(all_confidences)
        all_correct = np.array(all_correct)
        
        # Plot uncertainty analysis
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Uncertainty vs Confidence
        colors = ['red' if not correct else 'blue' for correct in all_correct]
        axes[0].scatter(all_confidences, all_uncertainties, c=colors, alpha=0.6)
        axes[0].set_xlabel('Confidence (Max Probability)')
        axes[0].set_ylabel('Uncertainty (Predicted Class Scale)')
        axes[0].set_title('Confidence vs Uncertainty')
        axes[0].grid(True)
        
        # Uncertainty distribution for correct vs incorrect
        correct_uncertainties = all_uncertainties[all_correct]
        incorrect_uncertainties = all_uncertainties[~all_correct]
        
        axes[1].hist(correct_uncertainties, bins=30, alpha=0.7, label='Correct', color='blue')
        axes[1].hist(incorrect_uncertainties, bins=30, alpha=0.7, label='Incorrect', color='red')
        axes[1].set_xlabel('Uncertainty (Predicted Class Scale)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Uncertainty Distribution')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
        return {
            'uncertainties': all_uncertainties,
            'confidences': all_confidences,
            'correct': all_correct
        } 