#!/usr/bin/env python3
"""
CAAC Project - Quick Experiment Runner

This is the main entry point for running experiments in the CAAC project.
It provides a user-friendly interface to access all experiment functionalities.

Usage:
    python run_experiments.py --help                    # Show all options
    python run_experiments.py --quick                   # Quick robustness test (3-5 min)
    python run_experiments.py --standard                # Standard robustness test (15-25 min)
    python run_experiments.py --comparison              # Basic method comparison
    python run_experiments.py --outlier                 # Outlier robustness test
    python run_experiments.py --interactive             # Interactive mode
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.experiments.experiment_manager import ExperimentManager
except ImportError:
    # Fallback for when src structure is not yet organized
    from experiments.experiment_manager import ExperimentManager


def print_welcome():
    """Print welcome message with project info."""
    print("=" * 60)
    print("🧠 CAAC Project - Shared Latent Cauchy Vector OvR Classifier")
    print("=" * 60)
    print()
    print("📖 Based on the theoretical framework in docs/theory/motivation.md")
    print("🔬 Core implementation in src/models/caac_ovr_model.py")
    print("📊 Comprehensive experiment suite available")
    print()


def print_experiment_options():
    """Print available experiment options."""
    print("🔬 Available Experiments:")
    print()
    print("  🚀 --quick        Quick robustness test (3-5 minutes)")
    print("                    Tests on 4 small datasets with label noise")
    print()
    print("  🔬 --standard     Standard robustness test (15-25 minutes)")
    print("                    Tests on 8 datasets with comprehensive analysis")
    print()
    print("  📊 --comparison   Basic method comparison")
    print("                    Compare CAAC with traditional methods")
    print()
    print("  🎯 --outlier      Outlier robustness test") 
    print("                    Test robustness against outliers and noise")
    print()
    print("  🎮 --interactive  Interactive experiment designer")
    print("                    Custom configuration with guided setup")
    print()


def interactive_experiment_designer():
    """Interactive mode for designing custom experiments."""
    print("🎮 Interactive Experiment Designer")
    print("=" * 40)
    
    # Choose experiment type
    exp_types = {
        '1': 'quick_robustness',
        '2': 'standard_robustness', 
        '3': 'basic_comparison',
        '4': 'outlier_robustness'
    }
    
    print("\nExperiment Types:")
    print("1. Quick Robustness Test")
    print("2. Standard Robustness Test")
    print("3. Basic Method Comparison")
    print("4. Outlier Robustness Test")
    
    choice = input("\nSelect experiment type (1-4): ").strip()
    if choice not in exp_types:
        print("❌ Invalid choice!")
        return None
    
    exp_type = exp_types[choice]
    manager = ExperimentManager()
    config = manager.get_experiment_config(exp_type)
    
    # Show current config
    print(f"\n📋 Default configuration for {exp_type}:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Ask if user wants to modify
    modify = input("\nModify configuration? (y/n): ").strip().lower()
    if modify == 'y':
        print("\n⚙️ Configuration Modification:")
        print("(Press Enter to keep default value)")
        
        new_config = {}
        for key, default_value in config.items():
            if key == 'datasets' and isinstance(default_value, list):
                print(f"\nAvailable datasets: {default_value}")
                new_value = input(f"{key} (comma-separated): ").strip()
                if new_value:
                    new_config[key] = [x.strip() for x in new_value.split(',')]
            elif key == 'noise_levels' and isinstance(default_value, list):
                new_value = input(f"{key} (comma-separated, e.g., 0.0,0.05,0.10): ").strip()
                if new_value:
                    new_config[key] = [float(x.strip()) for x in new_value.split(',')]
            else:
                new_value = input(f"{key} [{default_value}]: ").strip()
                if new_value:
                    # Try to convert to appropriate type
                    if isinstance(default_value, int):
                        new_config[key] = int(new_value)
                    elif isinstance(default_value, float):
                        new_config[key] = float(new_value)
                    else:
                        new_config[key] = new_value
        
        config.update(new_config)
    
    print(f"\n🚀 Running {exp_type} with configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nStarting experiment...")
    return exp_type, config


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="CAAC Project Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py --quick                # Quick test
  python run_experiments.py --standard             # Standard test  
  python run_experiments.py --comparison           # Method comparison
  python run_experiments.py --interactive          # Interactive mode
        """
    )
    
    # Experiment type arguments (mutually exclusive)
    exp_group = parser.add_mutually_exclusive_group()
    exp_group.add_argument('--quick', action='store_true',
                          help='Run quick robustness test (3-5 minutes)')
    exp_group.add_argument('--standard', action='store_true', 
                          help='Run standard robustness test (15-25 minutes)')
    exp_group.add_argument('--comparison', action='store_true',
                          help='Run basic method comparison')
    exp_group.add_argument('--outlier', action='store_true',
                          help='Run outlier robustness test')
    exp_group.add_argument('--interactive', action='store_true',
                          help='Interactive experiment designer')
    
    # Configuration arguments
    parser.add_argument('--config', '-c', 
                       help='Path to configuration file')
    parser.add_argument('--results-dir', '-r', default='results',
                       help='Results directory (default: results)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress welcome message')
    
    args = parser.parse_args()
    
    # Show welcome message
    if not args.quiet:
        print_welcome()
    
    # If no specific experiment is chosen, show options
    if not any([args.quick, args.standard, args.comparison, args.outlier, args.interactive]):
        print_experiment_options()
        print("💡 Use --help to see all command-line options")
        print("💡 Use --interactive for guided experiment setup")
        return
    
    # Initialize experiment manager
    manager = ExperimentManager(base_results_dir=args.results_dir, 
                              config_file=args.config)
    
    result_dir = None
    
    try:
        if args.quick:
            result_dir = manager.run_quick_robustness_test()
        elif args.standard:
            result_dir = manager.run_standard_robustness_test()
        elif args.comparison:
            result_dir = manager.run_basic_comparison()
        elif args.outlier:
            result_dir = manager.run_outlier_robustness_test()
        elif args.interactive:
            exp_result = interactive_experiment_designer()
            if exp_result:
                exp_type, config = exp_result
                if exp_type == 'quick_robustness':
                    result_dir = manager.run_quick_robustness_test(**config)
                elif exp_type == 'standard_robustness':
                    result_dir = manager.run_standard_robustness_test(**config)
                elif exp_type == 'basic_comparison':
                    result_dir = manager.run_basic_comparison(**config)
                elif exp_type == 'outlier_robustness':
                    result_dir = manager.run_outlier_robustness_test(**config)
    
    except KeyboardInterrupt:
        print("\n\n❌ Experiment interrupted by user")
        return
    except Exception as e:
        print(f"\n❌ Error running experiment: {e}")
        print("💡 Make sure you have activated the 'base' conda environment")
        print("💡 Check that all dependencies are installed")
        return
    
    # Show results
    if result_dir is not None:
        print(f"\n✅ Experiment completed successfully!")
        print(f"📁 Results saved to: {result_dir}")
        
        # Create and show summary only if result_dir is a valid path string
        if isinstance(result_dir, str):
            try:
                summary = manager.create_experiment_summary(result_dir)
                if 'files' in summary:
                    print(f"📋 Generated {len(summary['files'])} result files:")
                    for file_info in summary['files'][:5]:  # Show first 5 files
                        print(f"   - {file_info['name']}")
                    if len(summary['files']) > 5:
                        print(f"   ... and {len(summary['files']) - 5} more files")
            except Exception as e:
                print(f"📋 Results saved, but could not create summary: {e}")
            
            print(f"\n🔍 View results:")
            print(f"   📊 Open visualization files (.png) in {result_dir}")
            print(f"   📈 Check experiment_results.json for detailed metrics")
            print(f"   📝 Review training_history.json for training progress")
        else:
            print(f"📋 Experiment completed with return value: {type(result_dir)}")
    else:
        print("\n❌ Experiment did not complete successfully")


if __name__ == "__main__":
    main() 