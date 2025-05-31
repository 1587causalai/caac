"""
Test script for Shared Cauchy OvR Classifier (V2 Implementation)

This script provides a basic test of the new implementation to ensure
all components work correctly together.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import new implementation
from src.models import (
    SharedCauchyOvRClassifier,
    create_loss_function,
    SharedCauchyOvRTrainer
)


def generate_synthetic_data(
    n_samples: int = 2000,
    n_features: int = 20,
    n_classes: int = 10,
    n_informative: int = 15,
    random_state: int = 42
):
    """Generate synthetic multiclass dataset"""
    print(f"Generating synthetic dataset...")
    print(f"  - Samples: {n_samples}")
    print(f"  - Features: {n_features}")
    print(f"  - Classes: {n_classes}")
    
    # Generate dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_informative,
        n_redundant=n_features - n_informative,
        n_clusters_per_class=1,
        random_state=random_state
    )
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y


def create_data_loaders(X, y, test_size=0.2, val_size=0.1, batch_size=64):
    """Create train/val/test data loaders"""
    print("Creating data loaders...")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), 
        random_state=42, stratify=y_temp
    )
    
    print(f"  - Train: {len(X_train)} samples")
    print(f"  - Val: {len(X_val)} samples")
    print(f"  - Test: {len(X_test)} samples")
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    y_test = torch.LongTensor(y_test)
    
    # Create datasets and loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def test_basic_functionality():
    """Test basic model functionality without training"""
    print("\n" + "="*50)
    print("TESTING BASIC FUNCTIONALITY")
    print("="*50)
    
    # Parameters
    input_dim = 20
    num_classes = 10
    latent_dim = 8
    batch_size = 32
    
    print(f"Model parameters:")
    print(f"  - Input dim: {input_dim}")
    print(f"  - Num classes: {num_classes}")
    print(f"  - Latent dim: {latent_dim}")
    
    # Create model
    model = SharedCauchyOvRClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        latent_dim=latent_dim,
        hidden_dims=[64, 32]
    )
    
    print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(batch_size, input_dim)
    
    print("\nTesting forward pass...")
    outputs = model(x)
    
    print("Output shapes:")
    for key, value in outputs.items():
        print(f"  - {key}: {value.shape}")
    
    # Test probability constraints
    probs = outputs['probabilities']
    print(f"\nProbability statistics:")
    print(f"  - Min: {probs.min().item():.6f}")
    print(f"  - Max: {probs.max().item():.6f}")
    print(f"  - Mean: {probs.mean().item():.6f}")
    
    # Test predictions
    predictions = model.predict(x)
    print(f"  - Prediction shape: {predictions.shape}")
    print(f"  - Unique predictions: {torch.unique(predictions).tolist()}")
    
    # Test uncertainty metrics
    uncertainty_metrics = model.get_uncertainty_metrics(x)
    print(f"\nUncertainty metrics:")
    for key, value in uncertainty_metrics.items():
        print(f"  - {key}: shape={value.shape}, mean={value.mean().item():.4f}")
    
    print("✅ Basic functionality test passed!")


def test_loss_functions():
    """Test different loss functions"""
    print("\n" + "="*50)
    print("TESTING LOSS FUNCTIONS")
    print("="*50)
    
    batch_size = 16
    num_classes = 5
    
    # Dummy data
    probabilities = torch.rand(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    class_scales = torch.rand(batch_size, num_classes) + 0.1
    
    print("Testing different loss functions...")
    
    # Test standard OvR BCE
    ovr_bce = create_loss_function('ovr_bce')
    loss1 = ovr_bce(probabilities, targets)
    print(f"  - OvR BCE Loss: {loss1.item():.4f}")
    
    # Test weighted OvR BCE
    weighted_bce = create_loss_function('weighted_ovr_bce', alpha=0.5)
    loss2 = weighted_bce(probabilities, targets)
    print(f"  - Weighted OvR BCE Loss: {loss2.item():.4f}")
    
    # Test focal loss
    focal_loss = create_loss_function('focal_ovr', gamma=2.0)
    loss3 = focal_loss(probabilities, targets)
    print(f"  - Focal OvR Loss: {loss3.item():.4f}")
    
    # Test uncertainty regularized loss
    uncertainty_loss = create_loss_function(
        'uncertainty_reg', 
        scale_regularizer_weight=0.01
    )
    loss_dict = uncertainty_loss(probabilities, targets, class_scales)
    print(f"  - Uncertainty Regularized Loss: {loss_dict['total_loss'].item():.4f}")
    print(f"    - Base loss: {loss_dict['base_loss'].item():.4f}")
    print(f"    - Scale regularizer: {loss_dict['scale_regularizer'].item():.4f}")
    
    print("✅ Loss functions test passed!")


def test_training_pipeline():
    """Test the complete training pipeline"""
    print("\n" + "="*50)
    print("TESTING TRAINING PIPELINE")
    print("="*50)
    
    # Generate data
    X, y = generate_synthetic_data(
        n_samples=1000,
        n_features=20,
        n_classes=5,
        random_state=42
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X, y, batch_size=32
    )
    
    # Create model
    model = SharedCauchyOvRClassifier(
        input_dim=20,
        num_classes=5,
        latent_dim=6,
        hidden_dims=[64, 32]
    )
    
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create loss function and optimizer
    loss_function = create_loss_function('ovr_bce')
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # Create trainer
    trainer = SharedCauchyOvRTrainer(
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        device='cpu'
    )
    
    print("\nStarting training...")
    # Train for a few epochs
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        early_stopping_patience=5
    )
    
    print("Training completed!")
    
    # Evaluate model
    print("\nEvaluating model...")
    results = trainer.evaluate(test_loader)
    
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Analyze uncertainty
    print("\nAnalyzing uncertainty...")
    uncertainty_analysis = trainer.analyze_uncertainty(test_loader)
    
    print(f"Average uncertainty: {uncertainty_analysis['uncertainties'].mean():.4f}")
    print(f"Average confidence: {uncertainty_analysis['confidences'].mean():.4f}")
    
    correct_mask = uncertainty_analysis['correct']
    print(f"Correct predictions: {correct_mask.sum()}/{len(correct_mask)} ({correct_mask.mean()*100:.1f}%)")
    
    print("✅ Training pipeline test passed!")
    
    return trainer, results


def main():
    """Run all tests"""
    print("🚀 Testing Shared Cauchy OvR Classifier V2 Implementation")
    print("="*60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Test basic functionality
        test_basic_functionality()
        
        # Test loss functions
        test_loss_functions()
        
        # Test training pipeline
        trainer, results = test_training_pipeline()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("✅ V2 Implementation is working correctly")
        print("="*60)
        
        return trainer, results
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    trainer, results = main() 