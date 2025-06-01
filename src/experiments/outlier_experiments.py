"""
Outlier Robustness Experiments Module

This module contains the core logic for running outlier robustness experiments,
extracted from the legacy scripts to create a clean, maintainable implementation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from src.models.caac_ovr_model import CAACOvRModel, SoftmaxMLPModel

class OutlierRobustnessRunner:
    """
    Centralized runner for outlier robustness experiments.
    
    This class handles:
    - Dataset loading and preprocessing
    - Outlier injection strategies
    - Model training and evaluation under outlier conditions
    - Results visualization and reporting
    """
    
    def __init__(self, results_dir: str = "results"):
        """Initialize the outlier robustness runner."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Store relative path for display purposes
        self.results_dir_display = results_dir if "/" in results_dir else f"./{results_dir}"
        
        # Set random seed for reproducibility
        np.random.seed(42)
    
    def inject_outliers(self, X, y, outlier_fraction=0.1, outlier_type='gaussian'):
        """
        Inject outliers into the dataset.
        
        Args:
            X: Feature matrix
            y: Labels
            outlier_fraction: Fraction of samples to make outliers
            outlier_type: Type of outliers ('gaussian', 'uniform', 'extreme')
        """
        X_corrupted = X.copy()
        n_samples = len(X)
        n_outliers = int(n_samples * outlier_fraction)
        
        # Randomly select samples to corrupt
        outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
        
        if outlier_type == 'gaussian':
            # Add Gaussian noise with large variance
            noise = np.random.normal(0, 3 * np.std(X, axis=0), (n_outliers, X.shape[1]))
            X_corrupted[outlier_indices] += noise
        elif outlier_type == 'uniform':
            # Replace with uniform random values in extended range
            X_min, X_max = np.min(X, axis=0), np.max(X, axis=0)
            X_range = X_max - X_min
            outlier_values = np.random.uniform(
                X_min - 2 * X_range, X_max + 2 * X_range, 
                (n_outliers, X.shape[1])
            )
            X_corrupted[outlier_indices] = outlier_values
        elif outlier_type == 'extreme':
            # Replace with extreme values
            X_corrupted[outlier_indices] = np.random.choice(
                [-100, 100], size=(n_outliers, X.shape[1])
            )
        
        return X_corrupted, outlier_indices
    
    def load_dataset(self, dataset_name):
        """Load a specific dataset."""
        if dataset_name == 'iris':
            data = load_iris()
            return data.data, data.target, data.target_names, 'Iris'
        elif dataset_name == 'wine':
            data = load_wine()
            return data.data, data.target, data.target_names, 'Wine'
        elif dataset_name == 'breast_cancer':
            data = load_breast_cancer()
            return data.data, data.target, data.target_names, 'Breast Cancer'
        elif dataset_name == 'digits':
            data = load_digits()
            return data.data, data.target, [str(i) for i in range(10)], 'Digits'
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def evaluate_outlier_robustness(self, model_class, model_params, 
                                  X_train, X_test, y_train, y_test, 
                                  outlier_fractions, outlier_types):
        """Evaluate model robustness against outliers."""
        results = []
        
        for outlier_type in outlier_types:
            for outlier_fraction in outlier_fractions:
                print(f"    📊 Testing {outlier_type} outliers at {outlier_fraction:.1%}")
                
                # Inject outliers into training data
                X_train_corrupted, outlier_indices = self.inject_outliers(
                    X_train, y_train, outlier_fraction, outlier_type
                )
                
                # Train model on corrupted data
                start_time = time.time()
                model = model_class(**model_params)
                model.fit(X_train_corrupted, y_train, verbose=0)
                training_time = time.time() - start_time
                
                # Evaluate on clean test set
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                results.append({
                    'outlier_type': outlier_type,
                    'outlier_fraction': outlier_fraction,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'training_time': training_time
                })
        
        return results
    
    def run_outlier_robustness_experiments(self, 
                                         datasets=None, 
                                         outlier_fractions=None, 
                                         outlier_types=None,
                                         representation_dim=128,
                                         epochs=100):
        """
        Run comprehensive outlier robustness experiments.
        
        Args:
            datasets: List of dataset names to test
            outlier_fractions: List of outlier fractions to test
            outlier_types: List of outlier types to test
            representation_dim: Model representation dimension
            epochs: Number of training epochs
        """
        if datasets is None:
            datasets = ['iris', 'wine', 'breast_cancer', 'digits']
        
        if outlier_fractions is None:
            outlier_fractions = [0.0, 0.10, 0.20]
        
        if outlier_types is None:
            outlier_types = ['gaussian', 'uniform', 'extreme']
        
        print("🎯 Starting Outlier Robustness Experiments")
        print("=" * 60)
        print(f"📊 Testing {len(datasets)} datasets")
        print(f"📊 Outlier fractions: {outlier_fractions}")
        print(f"📊 Outlier types: {outlier_types}")
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
            'MLP (Softmax)': {
                'class': SoftmaxMLPModel,
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
                    
                    # Evaluate outlier robustness
                    method_results = self.evaluate_outlier_robustness(
                        method_info['class'], model_params,
                        X_train_scaled, X_test_scaled, y_train, y_test,
                        outlier_fractions, outlier_types
                    )
                    
                    # Add metadata to results
                    for result in method_results:
                        result.update({
                            'dataset': display_name,
                            'method': method_name,
                            'dataset_key': dataset_name
                        })
                        all_results.append(result)
                    
                    # Calculate robustness score for clean vs worst case
                    clean_accuracy = next(r['accuracy'] for r in method_results 
                                        if r['outlier_fraction'] == 0.0)
                    worst_accuracy = min(r['accuracy'] for r in method_results)
                    robustness_score = worst_accuracy / clean_accuracy if clean_accuracy > 0 else 0
                    
                    print(f"    ✅ Clean accuracy: {clean_accuracy:.3f}, Robustness: {robustness_score:.3f}")
                    
                except Exception as e:
                    print(f"    ❌ Error: {str(e)}")
                    continue
        
        # Create results DataFrame and save
        results_df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.results_dir / f"outlier_robustness_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        
        # Create visualizations
        self._create_outlier_plots(results_df, timestamp)
        
        # Generate report
        self._generate_outlier_report(results_df, timestamp)
        
        print(f"\n✅ Outlier robustness experiments completed!")
        print(f"📁 Results saved to: {self.results_dir_display}")
        print(f"📊 Detailed data: {self.results_dir_display}/{results_file.name}")
        
        return str(self.results_dir)
    
    def _create_outlier_plots(self, results_df, timestamp):
        """Create visualization plots for outlier robustness results."""
        plt.style.use('default')
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Outlier Robustness Test Results', fontsize=16)
        
        # 1. Accuracy vs Outlier Fraction by Type
        ax1 = axes[0, 0]
        for outlier_type in results_df['outlier_type'].unique():
            for method in results_df['method'].unique():
                data = results_df[(results_df['outlier_type'] == outlier_type) & 
                                (results_df['method'] == method)]
                avg_by_fraction = data.groupby('outlier_fraction')['accuracy'].mean()
                ax1.plot(avg_by_fraction.index, avg_by_fraction.values, 
                        marker='o', label=f"{method} ({outlier_type})", linewidth=2)
        
        ax1.set_xlabel('Outlier Fraction')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy vs Outlier Fraction')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Robustness by Outlier Type
        ax2 = axes[0, 1]
        robustness_data = []
        for method in results_df['method'].unique():
            for outlier_type in results_df['outlier_type'].unique():
                data = results_df[(results_df['method'] == method) & 
                                (results_df['outlier_type'] == outlier_type)]
                clean_acc = data[data['outlier_fraction'] == 0]['accuracy'].mean()
                worst_acc = data['accuracy'].min()
                robustness = worst_acc / clean_acc if clean_acc > 0 else 0
                robustness_data.append({
                    'method': method,
                    'outlier_type': outlier_type,
                    'robustness': robustness
                })
        
        rob_df = pd.DataFrame(robustness_data)
        rob_pivot = rob_df.pivot(index='method', columns='outlier_type', values='robustness')
        rob_pivot.plot(kind='bar', ax=ax2)
        ax2.set_ylabel('Robustness Score')
        ax2.set_title('Robustness by Outlier Type')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Training Time vs Outlier Fraction
        ax3 = axes[1, 0]
        for method in results_df['method'].unique():
            data = results_df[results_df['method'] == method]
            avg_time = data.groupby('outlier_fraction')['training_time'].mean()
            ax3.plot(avg_time.index, avg_time.values, 
                    marker='o', label=method, linewidth=2)
        
        ax3.set_xlabel('Outlier Fraction')
        ax3.set_ylabel('Training Time (seconds)')
        ax3.set_title('Training Time vs Outlier Fraction')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Dataset Performance Comparison
        ax4 = axes[1, 1]
        clean_data = results_df[results_df['outlier_fraction'] == 0.0]
        dataset_perf = clean_data.groupby(['dataset', 'method'])['accuracy'].mean().unstack()
        dataset_perf.plot(kind='bar', ax=ax4)
        ax4.set_ylabel('Accuracy (Clean Data)')
        ax4.set_title('Baseline Performance by Dataset')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / f"outlier_robustness_analysis_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Visualization saved: {self.results_dir_display}/{plot_file.name}")
    
    def _generate_outlier_report(self, results_df, timestamp):
        """Generate detailed outlier robustness report."""
        report_file = self.results_dir / f"outlier_robustness_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write(f"""# Outlier Robustness Test Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Experiment Overview

This report analyzes the robustness of CAAC and baseline methods against different types of outliers in training data.

### Test Configuration
- **Datasets:** {list(results_df['dataset'].unique())}
- **Methods:** {list(results_df['method'].unique())}
- **Outlier Types:** {list(results_df['outlier_type'].unique())}
- **Outlier Fractions:** {sorted(results_df['outlier_fraction'].unique())}

## Results Summary

### Robustness Rankings by Outlier Type

""")
            
            # Calculate and show robustness rankings
            for outlier_type in results_df['outlier_type'].unique():
                f.write(f"#### {outlier_type.title()} Outliers\n\n")
                
                type_data = results_df[results_df['outlier_type'] == outlier_type]
                robustness_scores = {}
                
                for method in type_data['method'].unique():
                    method_data = type_data[type_data['method'] == method]
                    clean_acc = method_data[method_data['outlier_fraction'] == 0]['accuracy'].mean()
                    worst_acc = method_data['accuracy'].min()
                    robustness = worst_acc / clean_acc if clean_acc > 0 else 0
                    robustness_scores[method] = robustness
                
                # Sort by robustness
                sorted_methods = sorted(robustness_scores.items(), key=lambda x: x[1], reverse=True)
                
                for i, (method, score) in enumerate(sorted_methods, 1):
                    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                    f.write(f"{emoji} **{method}**: {score:.3f}\n")
                f.write("\n")
            
            # Detailed results table
            f.write("## Detailed Results\n\n")
            summary_table = results_df.groupby(['method', 'outlier_type', 'outlier_fraction']).agg({
                'accuracy': ['mean', 'std'],
                'f1_score': 'mean'
            }).round(3)
            
            f.write(summary_table.to_markdown())
            f.write("\n\n")
            
            f.write("---\n")
            f.write("*Report generated by CAAC Outlier Robustness Experiment Runner*\n")
        
        print(f"📄 Report saved: {self.results_dir_display}/{report_file.name}")


def run_outlier_robustness_experiments(**kwargs):
    """Standalone function for outlier robustness experiments."""
    runner = OutlierRobustnessRunner()
    return runner.run_outlier_robustness_experiments(**kwargs) 