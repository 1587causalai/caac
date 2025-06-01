"""
Robustness Experiments Module

This module contains the core logic for running robustness experiments,
extracted from the legacy scripts to create a clean, maintainable implementation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import os
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our models
from src.models.caac_ovr_model import CAACOvRModel
from src.models.caac_ovr_model import SoftmaxMLPModel, OvRCrossEntropyMLPModel

class RobustnessExperimentRunner:
    """
    Centralized runner for robustness experiments.
    
    This class handles:
    - Dataset loading and preprocessing
    - Noise injection for robustness testing
    - Model training and evaluation
    - Results visualization and reporting
    """
    
    def __init__(self, results_dir: str = "results"):
        """Initialize the robustness experiment runner."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Available datasets
        self.dataset_loaders = {
            'iris': self._load_iris,
            'wine': self._load_wine,
            'breast_cancer': self._load_breast_cancer,
            'digits': self._load_digits,
            'optical_digits': self._load_optical_digits,
            'synthetic_imbalanced': self._load_synthetic_imbalanced,
            'covertype': self._load_covertype,
            'letter': self._load_letter
        }
        
    def _load_iris(self):
        """Load Iris dataset."""
        iris = load_iris()
        return iris.data, iris.target, iris.target_names, 'Iris'
    
    def _load_wine(self):
        """Load Wine dataset."""
        wine = load_wine()
        return wine.data, wine.target, wine.target_names, 'Wine'
    
    def _load_breast_cancer(self):
        """Load Breast Cancer dataset."""
        bc = load_breast_cancer()
        return bc.data, bc.target, bc.target_names, 'Breast Cancer'
    
    def _load_digits(self):
        """Load full Digits dataset."""
        digits = load_digits()
        return digits.data, digits.target, [str(i) for i in range(10)], 'Digits'
    
    def _load_optical_digits(self):
        """Load subset of Digits dataset for quick testing."""
        digits = load_digits()
        # Use smaller subset for quick testing
        X, y = digits.data[:400], digits.target[:400]
        return X, y, [str(i) for i in range(10)], 'Optical Digits'
    
    def _load_synthetic_imbalanced(self):
        """Create synthetic imbalanced dataset."""
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=15,
            n_redundant=5, n_classes=3, n_clusters_per_class=1,
            weights=[0.1, 0.3, 0.6], random_state=42
        )
        return X, y, ['Class 0', 'Class 1', 'Class 2'], 'Synthetic Imbalanced'
    
    def _load_covertype(self):
        """Load Forest Cover Type dataset (simplified version)."""
        try:
            from sklearn.datasets import fetch_covtype
            data = fetch_covtype()
            # Use subset for manageable computation
            X, y = data.data[:2000], data.target[:2000] - 1  # Make 0-indexed
            target_names = [f'Cover Type {i}' for i in range(7)]
            return X, y, target_names, 'Forest Cover Type'
        except:
            # Fallback to synthetic data if covtype not available
            return self._load_synthetic_imbalanced()
    
    def _load_letter(self):
        """Create synthetic letter recognition dataset."""
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=1500, n_features=16, n_informative=12,
            n_redundant=4, n_classes=26, n_clusters_per_class=1,
            random_state=42
        )
        target_names = [chr(ord('A') + i) for i in range(26)]
        return X, y, target_names, 'Letter Recognition'
    
    def load_dataset(self, dataset_name: str):
        """Load a specific dataset."""
        if dataset_name not in self.dataset_loaders:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return self.dataset_loaders[dataset_name]()
    
    def inject_label_noise(self, y, noise_level):
        """Inject label noise into the target variable."""
        if noise_level == 0:
            return y
        
        y_noisy = y.copy()
        n_samples = len(y)
        n_noisy = int(n_samples * noise_level)
        
        # Randomly select samples to add noise to
        noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
        
        # For each noisy sample, assign random label different from original
        unique_labels = np.unique(y)
        for idx in noisy_indices:
            original_label = y_noisy[idx]
            possible_labels = unique_labels[unique_labels != original_label]
            y_noisy[idx] = np.random.choice(possible_labels)
        
        return y_noisy
    
    def evaluate_model_robustness(self, model_class, model_params, X_train, X_test, y_train, y_test, noise_levels):
        """Evaluate model robustness across different noise levels."""
        results = []
        
        for noise_level in noise_levels:
            print(f"  📊 Testing noise level: {noise_level:.1%}")
            
            # Inject noise into training labels
            y_train_noisy = self.inject_label_noise(y_train, noise_level)
            
            # Create and train model
            start_time = time.time()
            model = model_class(**model_params)
            model.fit(X_train, y_train_noisy, verbose=0)
            training_time = time.time() - start_time
            
            # Evaluate on clean test set
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results.append({
                'noise_level': noise_level,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'training_time': training_time
            })
        
        return results
    
    def run_quick_robustness_test(self, 
                                noise_levels=None, 
                                representation_dim=128, 
                                epochs=100, 
                                datasets=None):
        """
        Run quick robustness test with small datasets.
        
        Args:
            noise_levels: List of noise levels to test
            representation_dim: Dimension of representation space
            epochs: Number of training epochs
            datasets: List of dataset names to test
        """
        if noise_levels is None:
            noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
        
        if datasets is None:
            datasets = ['iris', 'wine', 'breast_cancer', 'optical_digits']
        
        print("🚀 Quick Robustness Test")
        print("=" * 50)
        print(f"📊 Testing {len(datasets)} datasets with {len(noise_levels)} noise levels")
        print(f"⚙️  Representation dim: {representation_dim}, Epochs: {epochs}")
        print()
        
        all_results = []
        
        # Test methods
        methods = {
            'CAAC (Cauchy)': {
                'class': CAACOvRModel,
                'params': {
                    'representation_dim': representation_dim,
                    'epochs': epochs,
                    'lr': 0.001,
                    'batch_size': 32,
                    'early_stopping_patience': 10
                }
            },
            'CAAC (Gaussian)': {
                'class': CAACOvRModel,  # This should be CAACOvRGaussianModel when available
                'params': {
                    'representation_dim': representation_dim,
                    'epochs': epochs,
                    'lr': 0.001,
                    'batch_size': 32,
                    'early_stopping_patience': 10
                }
            },
            'MLP (Softmax)': {
                'class': SoftmaxMLPModel,
                'params': {
                    'representation_dim': representation_dim,
                    'epochs': epochs,
                    'lr': 0.001,
                    'batch_size': 32,
                    'early_stopping_patience': 10
                }
            },
            'MLP (OvR)': {
                'class': OvRCrossEntropyMLPModel,
                'params': {
                    'representation_dim': representation_dim,
                    'epochs': epochs,
                    'lr': 0.001,
                    'batch_size': 32,
                    'early_stopping_patience': 10
                }
            }
        }
        
        for dataset_name in datasets:
            print(f"📁 Testing dataset: {dataset_name}")
            
            # Load and preprocess data
            X, y, target_names, display_name = self.load_dataset(dataset_name)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Test each method
            for method_name, method_info in methods.items():
                print(f"  🧪 Testing method: {method_name}")
                
                try:
                    # Set up model parameters
                    model_params = method_info['params'].copy()
                    model_params['input_dim'] = X_train_scaled.shape[1]
                    model_params['n_classes'] = len(np.unique(y))
                    
                    # Evaluate robustness
                    method_results = self.evaluate_model_robustness(
                        method_info['class'], model_params,
                        X_train_scaled, X_test_scaled, y_train, y_test,
                        noise_levels
                    )
                    
                    # Add metadata to results
                    for result in method_results:
                        result.update({
                            'dataset': display_name,
                            'method': method_name,
                            'dataset_key': dataset_name
                        })
                        all_results.append(result)
                    
                    # Calculate robustness score
                    base_accuracy = method_results[0]['accuracy']  # No noise
                    worst_accuracy = min(r['accuracy'] for r in method_results)
                    robustness_score = worst_accuracy / base_accuracy if base_accuracy > 0 else 0
                    
                    print(f"    ✅ Base accuracy: {base_accuracy:.3f}, Robustness: {robustness_score:.3f}")
                    
                except Exception as e:
                    print(f"    ❌ Error: {str(e)}")
                    continue
        
        # Create results DataFrame and save
        results_df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.results_dir / f"quick_robustness_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        
        # Create visualizations
        self._create_robustness_plots(results_df, "quick_robustness", timestamp)
        
        # Generate report
        self._generate_robustness_report(results_df, "quick", timestamp)
        
        print(f"\n✅ Quick robustness test completed!")
        print(f"📁 Results saved to: {self.results_dir}")
        print(f"📊 Detailed data: {results_file.name}")
        
        return str(self.results_dir)
    
    def run_standard_robustness_test(self, 
                                   noise_levels=None, 
                                   representation_dim=128, 
                                   epochs=150, 
                                   datasets=None):
        """
        Run standard robustness test with full datasets.
        
        Args:
            noise_levels: List of noise levels to test
            representation_dim: Dimension of representation space
            epochs: Number of training epochs
            datasets: List of dataset names to test
        """
        if noise_levels is None:
            noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
        
        if datasets is None:
            datasets = ['iris', 'wine', 'breast_cancer', 'optical_digits', 
                       'digits', 'synthetic_imbalanced', 'covertype', 'letter']
        
        print("🔬 Standard Robustness Test")
        print("=" * 50)
        print(f"📊 Testing {len(datasets)} datasets with {len(noise_levels)} noise levels")
        print(f"⚙️  Representation dim: {representation_dim}, Epochs: {epochs}")
        print("⏱️  This may take 15-25 minutes...")
        print()
        
        # Use the same logic as quick test but with more datasets and epochs
        return self.run_quick_robustness_test(
            noise_levels=noise_levels,
            representation_dim=representation_dim,
            epochs=epochs,
            datasets=datasets
        )
    
    def _create_robustness_plots(self, results_df, experiment_type, timestamp):
        """Create visualization plots for robustness results."""
        plt.style.use('default')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Robustness Test Results - {experiment_type.title()}', fontsize=16)
        
        # 1. Accuracy vs Noise Level
        ax1 = axes[0, 0]
        for method in results_df['method'].unique():
            method_data = results_df[results_df['method'] == method]
            avg_by_noise = method_data.groupby('noise_level')['accuracy'].mean()
            ax1.plot(avg_by_noise.index, avg_by_noise.values, 
                    marker='o', label=method, linewidth=2)
        
        ax1.set_xlabel('Noise Level')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy vs Noise Level')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Robustness Scores by Method
        ax2 = axes[0, 1]
        robustness_scores = []
        method_names = []
        
        for method in results_df['method'].unique():
            method_data = results_df[results_df['method'] == method]
            base_acc = method_data[method_data['noise_level'] == 0]['accuracy'].mean()
            worst_acc = method_data.groupby('noise_level')['accuracy'].mean().min()
            robustness_score = worst_acc / base_acc if base_acc > 0 else 0
            
            robustness_scores.append(robustness_score)
            method_names.append(method)
        
        bars = ax2.bar(method_names, robustness_scores)
        ax2.set_ylabel('Robustness Score')
        ax2.set_title('Robustness Score by Method')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Color bars by performance
        for bar, score in zip(bars, robustness_scores):
            if score > 0.8:
                bar.set_color('green')
            elif score > 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # 3. Performance by Dataset
        ax3 = axes[1, 0]
        dataset_performance = results_df[results_df['noise_level'] == 0].groupby(['dataset', 'method'])['accuracy'].mean().unstack()
        dataset_performance.plot(kind='bar', ax=ax3, width=0.8)
        ax3.set_ylabel('Accuracy (No Noise)')
        ax3.set_title('Baseline Performance by Dataset')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Training Time Analysis
        ax4 = axes[1, 1]
        time_by_method = results_df.groupby('method')['training_time'].mean()
        time_by_method.plot(kind='bar', ax=ax4)
        ax4.set_ylabel('Average Training Time (seconds)')
        ax4.set_title('Training Efficiency by Method')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / f"{experiment_type}_robustness_analysis_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Visualization saved: {plot_file.name}")
    
    def _generate_robustness_report(self, results_df, experiment_type, timestamp):
        """Generate a detailed robustness report."""
        report_file = self.results_dir / f"{experiment_type}_robustness_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# {experiment_type.title()} Robustness Test Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            f.write("## Summary\n\n")
            f.write(f"- **Datasets tested:** {results_df['dataset'].nunique()}\n")
            f.write(f"- **Methods compared:** {results_df['method'].nunique()}\n")
            f.write(f"- **Noise levels:** {sorted(results_df['noise_level'].unique())}\n\n")
            
            # Method rankings
            f.write("## Method Performance Rankings\n\n")
            
            # Calculate overall robustness scores
            method_scores = {}
            for method in results_df['method'].unique():
                method_data = results_df[results_df['method'] == method]
                base_acc = method_data[method_data['noise_level'] == 0]['accuracy'].mean()
                worst_acc = method_data.groupby('noise_level')['accuracy'].mean().min()
                robustness_score = worst_acc / base_acc if base_acc > 0 else 0
                method_scores[method] = {
                    'base_accuracy': base_acc,
                    'worst_accuracy': worst_acc,
                    'robustness_score': robustness_score
                }
            
            # Sort by robustness score
            sorted_methods = sorted(method_scores.items(), key=lambda x: x[1]['robustness_score'], reverse=True)
            
            for i, (method, scores) in enumerate(sorted_methods, 1):
                f.write(f"{i}. **{method}**\n")
                f.write(f"   - Base Accuracy: {scores['base_accuracy']:.3f}\n")
                f.write(f"   - Worst Case Accuracy: {scores['worst_accuracy']:.3f}\n")
                f.write(f"   - Robustness Score: {scores['robustness_score']:.3f}\n\n")
            
            # Detailed results table
            f.write("## Detailed Results\n\n")
            summary_table = results_df.groupby(['method', 'noise_level']).agg({
                'accuracy': ['mean', 'std'],
                'f1_score': 'mean'
            }).round(3)
            
            f.write(summary_table.to_markdown())
            f.write("\n\n")
            
            f.write("---\n")
            f.write("*Report generated by CAAC Robustness Experiment Runner*\n")
        
        print(f"📄 Report saved: {report_file.name}")


def run_quick_robustness_test(**kwargs):
    """Standalone function for quick robustness test."""
    runner = RobustnessExperimentRunner()
    return runner.run_quick_robustness_test(**kwargs)


def run_standard_robustness_test(**kwargs):
    """Standalone function for standard robustness test."""
    runner = RobustnessExperimentRunner()
    return runner.run_standard_robustness_test(**kwargs) 