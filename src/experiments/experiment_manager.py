"""
Unified Experiment Manager for CAAC Project

This module provides a centralized interface for running different types of experiments,
consolidating the functionality from scattered scripts while maintaining backward compatibility.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class ExperimentManager:
    """
    Centralized experiment management for CAAC project.
    
    This class provides a unified interface for:
    - Basic model comparison experiments
    - Robustness testing (quick/standard)
    - Outlier robustness experiments
    - Custom parameter experiments
    """
    
    def __init__(self, 
                 base_results_dir: str = "results",
                 config_file: Optional[str] = None):
        """
        Initialize the experiment manager.
        
        Args:
            base_results_dir: Base directory for storing experiment results
            config_file: Path to configuration file (optional)
        """
        self.base_results_dir = Path(base_results_dir)
        self.base_results_dir.mkdir(exist_ok=True)
        
        # Load configuration if provided
        self.config = self._load_config(config_file) if config_file else {}
        
        # Default experiment configurations
        self.default_configs = {
            'quick_robustness': {
                'noise_levels': [0.0, 0.05, 0.10, 0.15, 0.20],
                'representation_dim': 128,
                'epochs': 100,
                'datasets': ['iris', 'wine', 'breast_cancer', 'optical_digits']
            },
            'standard_robustness': {
                'noise_levels': [0.0, 0.05, 0.10, 0.15, 0.20],
                'representation_dim': 128,
                'epochs': 150,
                'datasets': ['iris', 'wine', 'breast_cancer', 'optical_digits', 
                           'digits', 'synthetic_imbalanced', 'covertype', 'letter']
            },
            'basic_comparison': {
                'datasets': ['iris', 'wine', 'breast_cancer', 'digits'],
                'representation_dim': 64,
                'epochs': 100
            }
        }
    
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from file."""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
            return {}
    
    def run_quick_robustness_test(self, **kwargs) -> str:
        """
        Run quick robustness test (3-5 minutes).
        
        Args:
            **kwargs: Override default parameters
            
        Returns:
            str: Path to experiment results directory
        """
        print("🚀 Starting Quick Robustness Test...")
        config = {**self.default_configs['quick_robustness'], **kwargs}
        
        # Use the new robustness experiments module
        try:
            from src.experiments.robustness_experiments import run_quick_robustness_test
            return run_quick_robustness_test(**config)
        except ImportError as e:
            print(f"Error importing robustness experiments: {e}")
            print("Make sure src/experiments/robustness_experiments.py exists")
            return None
    
    def run_standard_robustness_test(self, **kwargs) -> str:
        """
        Run standard robustness test (15-25 minutes).
        
        Args:
            **kwargs: Override default parameters
            
        Returns:
            str: Path to experiment results directory
        """
        print("🔬 Starting Standard Robustness Test...")
        config = {**self.default_configs['standard_robustness'], **kwargs}
        
        try:
            from src.experiments.robustness_experiments import run_standard_robustness_test
            return run_standard_robustness_test(**config)
        except ImportError as e:
            print(f"Error importing robustness experiments: {e}")
            print("Make sure src/experiments/robustness_experiments.py exists")
            return None
    
    def run_basic_comparison(self, **kwargs) -> str:
        """
        Run basic method comparison experiments.
        
        Args:
            **kwargs: Override default parameters
            
        Returns:
            str: Path to experiment results directory
        """
        print("📊 Starting Basic Method Comparison...")
        config = {**self.default_configs['basic_comparison'], **kwargs}
        
        try:
            from src.experiments.comparison_experiments import run_comparison_experiments
            return run_comparison_experiments(**config)
        except ImportError as e:
            print(f"Error importing comparison experiments: {e}")
            print("Make sure src/experiments/comparison_experiments.py exists")
            return None
    
    def run_outlier_robustness_test(self, **kwargs) -> str:
        """
        Run outlier robustness experiments.
        
        Args:
            **kwargs: Override default parameters
            
        Returns:
            str: Path to experiment results directory
        """
        print("🎯 Starting Outlier Robustness Test...")
        
        try:
            from src.experiments.outlier_experiments import run_outlier_robustness_experiments
            return run_outlier_robustness_experiments(**kwargs)
        except ImportError as e:
            print(f"Error importing outlier experiments: {e}")
            print("Make sure src/experiments/outlier_experiments.py exists")
            return None
    
    def run_custom_experiment(self, 
                            experiment_type: str,
                            config: Dict,
                            save_name: Optional[str] = None) -> str:
        """
        Run a custom experiment with specified configuration.
        
        Args:
            experiment_type: Type of experiment ('robustness', 'comparison', etc.)
            config: Experiment configuration dictionary
            save_name: Custom name for saving results
            
        Returns:
            str: Path to experiment results directory
        """
        print(f"⚙️ Starting Custom Experiment: {experiment_type}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = save_name or f"custom_{experiment_type}_{timestamp}"
        
        if experiment_type == 'robustness':
            return self.run_standard_robustness_test(**config)
        elif experiment_type == 'comparison':
            return self.run_basic_comparison(**config)
        elif experiment_type == 'outlier_robustness':
            return self.run_outlier_robustness_test(**config)
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    def list_available_experiments(self) -> List[str]:
        """List all available experiment types."""
        return [
            'quick_robustness',
            'standard_robustness', 
            'basic_comparison',
            'outlier_robustness',
            'custom'
        ]
    
    def get_experiment_config(self, experiment_type: str) -> Dict:
        """Get default configuration for an experiment type."""
        return self.default_configs.get(experiment_type, {})
    
    def create_experiment_summary(self, results_dir: str) -> Dict:
        """
        Create a summary of experiment results.
        
        Args:
            results_dir: Directory containing experiment results
            
        Returns:
            Dict: Summary of results
        """
        results_path = Path(results_dir)
        if not results_path.exists():
            return {"error": f"Results directory {results_dir} not found"}
        
        summary = {
            "experiment_dir": str(results_path),
            "timestamp": datetime.now().isoformat(),
            "files": []
        }
        
        # List all result files
        for file_path in results_path.glob("*"):
            if file_path.is_file():
                summary["files"].append({
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
        
        return summary


def main():
    """Command-line interface for the experiment manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CAAC Project Experiment Manager")
    parser.add_argument('--experiment', '-e', 
                       choices=['quick', 'standard', 'comparison', 'outlier', 'list'],
                       default='list',
                       help='Type of experiment to run')
    parser.add_argument('--config', '-c',
                       help='Path to configuration file')
    parser.add_argument('--results-dir', '-r',
                       default='results',
                       help='Results directory')
    
    args = parser.parse_args()
    
    manager = ExperimentManager(base_results_dir=args.results_dir, 
                              config_file=args.config)
    
    if args.experiment == 'list':
        print("Available experiments:")
        for exp in manager.list_available_experiments():
            print(f"  - {exp}")
        return
    
    # Run the specified experiment
    experiment_map = {
        'quick': manager.run_quick_robustness_test,
        'standard': manager.run_standard_robustness_test,
        'comparison': manager.run_basic_comparison,
        'outlier': manager.run_outlier_robustness_test
    }
    
    if args.experiment in experiment_map:
        result_dir = experiment_map[args.experiment]()
        print(f"\n✅ Experiment completed! Results saved to: {result_dir}")
        
        # Create summary
        summary = manager.create_experiment_summary(result_dir)
        print(f"📋 Generated {len(summary.get('files', []))} result files")
    else:
        print(f"Unknown experiment type: {args.experiment}")


if __name__ == "__main__":
    main() 