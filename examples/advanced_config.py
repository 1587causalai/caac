"""
⚙️ CAAC项目高级配置示例

本文件展示CAAC项目的高级配置选项，包括：
- 配置文件管理
- 动态参数调整
- 性能优化配置
- 实验配置模板

请在项目根目录运行：python examples/advanced_config.py
"""

import sys
import os
import json
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.caac_ovr_model import CAACOvRModel
from src.experiments.experiment_manager import ExperimentManager
from sklearn.datasets import load_digits, load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt


class ConfigurationManager:
    """
    高级配置管理器
    """
    
    def __init__(self, config_dir="examples/configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
    def create_default_configs(self):
        """
        创建默认配置文件
        """
        print("=" * 60)
        print("⚙️ 创建默认配置文件")
        print("=" * 60)
        
        configs = {
            # 模型配置
            'model_configs': {
                'small_dataset': {
                    'representation_dim': 32,
                    'latent_dim': 16,
                    'feature_hidden_dims': [32],
                    'abduction_hidden_dims': [32, 16],
                    'lr': 0.01,
                    'batch_size': 16,
                    'epochs': 100,
                    'early_stopping_patience': 10,
                    'learnable_thresholds': False,
                    'uniqueness_constraint': False
                },
                'medium_dataset': {
                    'representation_dim': 64,
                    'latent_dim': 32,
                    'feature_hidden_dims': [128, 64],
                    'abduction_hidden_dims': [64, 32],
                    'lr': 0.001,
                    'batch_size': 32,
                    'epochs': 150,
                    'early_stopping_patience': 15,
                    'learnable_thresholds': True,
                    'uniqueness_constraint': False
                },
                'large_dataset': {
                    'representation_dim': 128,
                    'latent_dim': 64,
                    'feature_hidden_dims': [256, 128],
                    'abduction_hidden_dims': [128, 64],
                    'lr': 0.001,
                    'batch_size': 64,
                    'epochs': 200,
                    'early_stopping_patience': 20,
                    'learnable_thresholds': True,
                    'uniqueness_constraint': True,
                    'uniqueness_weight': 0.1
                },
                'robust_model': {
                    'representation_dim': 64,
                    'latent_dim': 32,
                    'feature_hidden_dims': [128, 64],
                    'abduction_hidden_dims': [64, 32],
                    'lr': 0.001,
                    'batch_size': 32,
                    'epochs': 200,
                    'early_stopping_patience': 25,
                    'learnable_thresholds': True,
                    'uniqueness_constraint': True,
                    'uniqueness_weight': 0.15,
                    'uniqueness_samples': 15
                }
            },
            
            # 实验配置
            'experiment_configs': {
                'quick_test': {
                    'datasets': ['iris', 'wine'],
                    'noise_levels': [0.0, 0.1, 0.2],
                    'test_size': 0.3,
                    'n_runs': 3,
                    'save_plots': True,
                    'save_detailed_results': False
                },
                'standard_robustness': {
                    'datasets': ['iris', 'wine', 'breast_cancer', 'digits'],
                    'noise_levels': [0.0, 0.05, 0.1, 0.15, 0.2],
                    'test_size': 0.3,
                    'n_runs': 5,
                    'save_plots': True,
                    'save_detailed_results': True
                },
                'comprehensive_evaluation': {
                    'datasets': ['iris', 'wine', 'breast_cancer', 'optical_digits', 
                               'digits', 'synthetic_imbalanced'],
                    'noise_levels': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                    'test_size': 0.2,
                    'n_runs': 10,
                    'save_plots': True,
                    'save_detailed_results': True,
                    'cross_validation': True,
                    'cv_folds': 5
                }
            },
            
            # 性能配置
            'performance_configs': {
                'debug_mode': {
                    'verbose': 2,
                    'save_intermediate_results': True,
                    'plot_training_curves': True,
                    'validate_inputs': True
                },
                'production_mode': {
                    'verbose': 1,
                    'save_intermediate_results': False,
                    'plot_training_curves': False,
                    'validate_inputs': False,
                    'use_gpu': True,
                    'parallel_experiments': True
                },
                'memory_optimized': {
                    'batch_size_multiplier': 0.5,
                    'representation_dim_multiplier': 0.75,
                    'save_models': False,
                    'early_stopping_patience_multiplier': 0.8
                }
            }
        }
        
        # 保存配置文件
        for config_name, config_data in configs.items():
            # JSON格式
            json_file = self.config_dir / f"{config_name}.json"
            with open(json_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            print(f"📝 创建 {json_file}")
            
            # YAML格式
            yaml_file = self.config_dir / f"{config_name}.yaml"
            with open(yaml_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            print(f"📝 创建 {yaml_file}")
        
        return configs
    
    def load_config(self, config_name: str, config_type: str = 'json') -> Dict[str, Any]:
        """
        加载配置文件
        """
        if config_type == 'json':
            config_file = self.config_dir / f"{config_name}.json"
        else:
            config_file = self.config_dir / f"{config_name}.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        
        if config_type == 'json':
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
    
    def create_custom_config(self, base_config: str, modifications: Dict[str, Any], 
                           save_name: str) -> Dict[str, Any]:
        """
        基于现有配置创建自定义配置
        """
        base = self.load_config(base_config)
        
        # 深度合并配置
        def deep_merge(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_merge(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_merge(base, modifications)
        
        # 保存新配置
        save_file = self.config_dir / f"{save_name}.json"
        with open(save_file, 'w') as f:
            json.dump(base, f, indent=2)
        
        print(f"💾 自定义配置已保存: {save_file}")
        return base


class AdvancedExperimentRunner:
    """
    高级实验运行器
    """
    
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.experiment_manager = ExperimentManager(base_results_dir="examples/advanced_results")
    
    def adaptive_hyperparameter_tuning(self):
        """
        示例1: 自适应超参数调优
        """
        print("\n" + "=" * 60)
        print("🎯 示例1: 自适应超参数调优")
        print("=" * 60)
        
        # 1. 加载数据
        data = load_digits()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 2. 定义参数空间
        param_space = {
            'representation_dim': [32, 64, 128],
            'lr': [0.001, 0.01, 0.1],
            'batch_size': [16, 32, 64],
            'learnable_thresholds': [False, True],
            'uniqueness_constraint': [False, True]
        }
        
        # 3. 自适应网格搜索
        print("🔍 开始自适应参数搜索...")
        best_config = None
        best_score = 0
        search_results = []
        
        # 粗搜索
        print("  第一阶段: 粗搜索...")
        coarse_grid = {
            'representation_dim': [32, 128],
            'lr': [0.001, 0.1],
            'learnable_thresholds': [False, True]
        }
        
        for repr_dim in coarse_grid['representation_dim']:
            for lr in coarse_grid['lr']:
                for learnable_thresh in coarse_grid['learnable_thresholds']:
                    config = {
                        'input_dim': X.shape[1],
                        'n_classes': len(np.unique(y)),
                        'representation_dim': repr_dim,
                        'lr': lr,
                        'learnable_thresholds': learnable_thresh,
                        'epochs': 50,
                        'verbose': 0
                    }
                    
                    model = CAACOvRModel(**config)
                    model.fit(X_train_scaled, y_train)
                    score = accuracy_score(y_test, model.predict(X_test_scaled))
                    
                    search_results.append((config.copy(), score))
                    if score > best_score:
                        best_score = score
                        best_config = config.copy()
                    
                    print(f"    repr_dim={repr_dim}, lr={lr}, thresh={learnable_thresh}: {score:.4f}")
        
        # 细搜索
        print("  第二阶段: 细搜索...")
        fine_search_space = {
            'representation_dim': [max(16, best_config['representation_dim']//2), 
                                 best_config['representation_dim'], 
                                 min(256, best_config['representation_dim']*2)],
            'lr': [best_config['lr']/3, best_config['lr'], best_config['lr']*3],
            'batch_size': [16, 32, 64]
        }
        
        for repr_dim in fine_search_space['representation_dim']:
            for lr in fine_search_space['lr']:
                for batch_size in fine_search_space['batch_size']:
                    config = best_config.copy()
                    config.update({
                        'representation_dim': repr_dim,
                        'lr': lr,
                        'batch_size': batch_size,
                        'epochs': 100
                    })
                    
                    model = CAACOvRModel(**config)
                    model.fit(X_train_scaled, y_train)
                    score = accuracy_score(y_test, model.predict(X_test_scaled))
                    
                    search_results.append((config.copy(), score))
                    if score > best_score:
                        best_score = score
                        best_config = config.copy()
                    
                    print(f"    repr_dim={repr_dim}, lr={lr:.4f}, batch={batch_size}: {score:.4f}")
        
        print(f"\n✅ 最佳配置 (准确率: {best_score:.4f}):")
        for key, value in best_config.items():
            if key not in ['input_dim', 'n_classes', 'verbose']:
                print(f"  {key}: {value}")
        
        # 保存最佳配置
        self.config_manager.create_custom_config(
            'model_configs', 
            {'adaptive_best': best_config}, 
            'adaptive_best_config'
        )
        
        return best_config, search_results
    
    def configuration_template_experiment(self):
        """
        示例2: 配置模板实验
        """
        print("\n" + "=" * 60)
        print("📋 示例2: 配置模板实验")
        print("=" * 60)
        
        # 1. 创建默认配置
        self.config_manager.create_default_configs()
        
        # 2. 加载不同配置模板
        configs_to_test = ['small_dataset', 'medium_dataset', 'large_dataset', 'robust_model']
        
        # 3. 准备测试数据
        data = load_wine()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 4. 测试每个配置模板
        results = {}
        for config_name in configs_to_test:
            print(f"\n🔧 测试配置模板: {config_name}")
            
            # 加载配置
            model_configs = self.config_manager.load_config('model_configs')
            config = model_configs[config_name].copy()
            config.update({
                'input_dim': X.shape[1],
                'n_classes': len(np.unique(y)),
                'verbose': 0
            })
            
            # 创建和训练模型
            model = CAACOvRModel(**config)
            history = model.fit(X_train_scaled, y_train)
            
            # 评估
            predictions = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='macro')
            
            results[config_name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'training_time': history.get('total_time', 0),
                'final_epoch': history.get('best_epoch', len(history.get('train_losses', [])))
            }
            
            print(f"  ✅ 准确率: {accuracy:.4f}, F1: {f1:.4f}")
            print(f"  ⏱️ 训练时间: {results[config_name]['training_time']:.2f}s")
        
        # 5. 结果比较
        print("\n📊 配置模板对比结果:")
        print("-" * 80)
        print(f"{'配置名称':<20} {'准确率':<10} {'F1分数':<10} {'训练时间':<12} {'收敛轮次':<10}")
        print("-" * 80)
        for config_name, result in results.items():
            print(f"{config_name:<20} {result['accuracy']:<10.4f} {result['f1_score']:<10.4f} "
                  f"{result['training_time']:<12.2f} {result['final_epoch']:<10}")
        
        return results
    
    def environment_adaptive_config(self):
        """
        示例3: 环境自适应配置
        """
        print("\n" + "=" * 60)
        print("🌐 示例3: 环境自适应配置")
        print("=" * 60)
        
        # 1. 检测运行环境
        import psutil
        import platform
        
        # 系统信息
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        print(f"🖥️ 系统信息:")
        print(f"  CPU核心数: {cpu_count}")
        print(f"  内存容量: {memory_gb:.1f}GB")
        print(f"  操作系统: {platform.system()}")
        
        # 2. 根据环境调整配置
        base_config = {
            'input_dim': 64,  # 示例
            'n_classes': 10,  # 示例
            'representation_dim': 64,
            'epochs': 100,
            'verbose': 1
        }
        
        # 内存优化
        if memory_gb < 8:
            print("📉 检测到低内存环境，应用内存优化配置...")
            memory_config = self.config_manager.load_config('performance_configs')['memory_optimized']
            base_config['batch_size'] = 16
            base_config['representation_dim'] = int(base_config['representation_dim'] * 
                                                  memory_config['representation_dim_multiplier'])
            base_config['early_stopping_patience'] = int(20 * 
                                                        memory_config['early_stopping_patience_multiplier'])
        else:
            base_config['batch_size'] = 64
        
        # CPU优化
        if cpu_count >= 8:
            print("🚀 检测到多核环境，启用并行处理...")
            base_config['parallel_processing'] = True
        else:
            base_config['parallel_processing'] = False
        
        # GPU检测
        try:
            import torch
            if torch.cuda.is_available():
                print("🎮 检测到GPU，启用GPU加速...")
                base_config['device'] = 'cuda'
            else:
                base_config['device'] = 'cpu'
        except ImportError:
            print("⚠️ 未安装PyTorch，使用CPU模式...")
            base_config['device'] = 'cpu'
        
        print(f"\n⚙️ 自适应配置:")
        for key, value in base_config.items():
            if key not in ['input_dim', 'n_classes']:
                print(f"  {key}: {value}")
        
        # 3. 保存环境自适应配置
        env_config = {
            'system_info': {
                'cpu_count': cpu_count,
                'memory_gb': memory_gb,
                'platform': platform.system()
            },
            'adaptive_config': base_config
        }
        
        save_file = self.config_manager.config_dir / "environment_adaptive.json"
        with open(save_file, 'w') as f:
            json.dump(env_config, f, indent=2)
        print(f"💾 环境自适应配置已保存: {save_file}")
        
        return env_config
    
    def config_validation_and_testing(self):
        """
        示例4: 配置验证和测试
        """
        print("\n" + "=" * 60)
        print("✅ 示例4: 配置验证和测试")
        print("=" * 60)
        
        # 1. 配置验证规则
        validation_rules = {
            'representation_dim': {'min': 8, 'max': 512, 'type': int},
            'lr': {'min': 1e-5, 'max': 1.0, 'type': float},
            'epochs': {'min': 10, 'max': 1000, 'type': int},
            'batch_size': {'min': 4, 'max': 256, 'type': int},
            'early_stopping_patience': {'min': 5, 'max': 100, 'type': int}
        }
        
        def validate_config(config: Dict[str, Any]) -> tuple[bool, List[str]]:
            """验证配置有效性"""
            errors = []
            
            for param, rules in validation_rules.items():
                if param in config:
                    value = config[param]
                    
                    # 类型检查
                    if not isinstance(value, rules['type']):
                        errors.append(f"{param}: 期望类型 {rules['type'].__name__}, 得到 {type(value).__name__}")
                    
                    # 范围检查
                    if isinstance(value, (int, float)):
                        if 'min' in rules and value < rules['min']:
                            errors.append(f"{param}: 值 {value} 小于最小值 {rules['min']}")
                        if 'max' in rules and value > rules['max']:
                            errors.append(f"{param}: 值 {value} 大于最大值 {rules['max']}")
            
            return len(errors) == 0, errors
        
        # 2. 测试有效配置
        valid_config = {
            'input_dim': 20,
            'n_classes': 3,
            'representation_dim': 64,
            'lr': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'early_stopping_patience': 15,
            'verbose': 0
        }
        
        is_valid, errors = validate_config(valid_config)
        print(f"✅ 有效配置验证: {'通过' if is_valid else '失败'}")
        if errors:
            for error in errors:
                print(f"  ❌ {error}")
        
        # 3. 测试无效配置
        invalid_configs = [
            {'representation_dim': 1000, 'lr': 10.0},  # 超出范围
            {'epochs': 'not_a_number'},  # 类型错误
            {'batch_size': 1000, 'early_stopping_patience': 1}  # 多个错误
        ]
        
        print("\n🚫 无效配置测试:")
        for i, config in enumerate(invalid_configs, 1):
            test_config = valid_config.copy()
            test_config.update(config)
            
            is_valid, errors = validate_config(test_config)
            print(f"  测试 {i}: {'通过' if is_valid else '失败'}")
            for error in errors:
                print(f"    ❌ {error}")
        
        # 4. 配置兼容性测试
        print("\n🔗 配置兼容性测试:")
        
        # 测试极小配置
        minimal_config = {
            'input_dim': 4,
            'n_classes': 2,
            'representation_dim': 8,
            'epochs': 10,
            'verbose': 0
        }
        
        try:
            model = CAACOvRModel(**minimal_config)
            print("  ✅ 极小配置: 可创建模型")
        except Exception as e:
            print(f"  ❌ 极小配置: {e}")
        
        # 测试大型配置
        large_config = {
            'input_dim': 100,
            'n_classes': 10,
            'representation_dim': 256,
            'feature_hidden_dims': [512, 256],
            'abduction_hidden_dims': [256, 128],
            'epochs': 200,
            'verbose': 0
        }
        
        try:
            model = CAACOvRModel(**large_config)
            print("  ✅ 大型配置: 可创建模型")
        except Exception as e:
            print(f"  ❌ 大型配置: {e}")
        
        return validation_rules
    
    def performance_profiling_experiment(self):
        """
        示例5: 性能分析实验
        """
        print("\n" + "=" * 60)
        print("📈 示例5: 性能分析实验")
        print("=" * 60)
        
        import time
        import tracemalloc
        
        # 1. 准备测试数据
        data = load_digits()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 2. 测试不同配置的性能
        test_configs = {
            'lightweight': {
                'representation_dim': 32,
                'feature_hidden_dims': [32],
                'abduction_hidden_dims': [32, 16],
                'epochs': 50
            },
            'standard': {
                'representation_dim': 64,
                'feature_hidden_dims': [128, 64],
                'abduction_hidden_dims': [64, 32],
                'epochs': 100
            },
            'heavy': {
                'representation_dim': 128,
                'feature_hidden_dims': [256, 128],
                'abduction_hidden_dims': [128, 64],
                'epochs': 150
            }
        }
        
        performance_results = {}
        
        for config_name, config in test_configs.items():
            print(f"\n🔧 测试配置: {config_name}")
            
            # 基础配置
            full_config = {
                'input_dim': X.shape[1],
                'n_classes': len(np.unique(y)),
                'verbose': 0,
                **config
            }
            
            # 开始内存追踪
            tracemalloc.start()
            
            # 创建和训练模型
            start_time = time.time()
            model = CAACOvRModel(**full_config)
            
            # 训练时间
            train_start = time.time()
            history = model.fit(X_train_scaled, y_train)
            train_time = time.time() - train_start
            
            # 预测时间
            predict_start = time.time()
            predictions = model.predict(X_test_scaled)
            predict_time = time.time() - predict_start
            
            total_time = time.time() - start_time
            
            # 内存使用
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # 准确率
            accuracy = accuracy_score(y_test, predictions)
            
            performance_results[config_name] = {
                'accuracy': accuracy,
                'train_time': train_time,
                'predict_time': predict_time,
                'total_time': total_time,
                'memory_current_mb': current / (1024 * 1024),
                'memory_peak_mb': peak / (1024 * 1024),
                'parameters': sum(p.numel() for p in model.model.parameters()),
                'epochs_used': history.get('best_epoch', len(history.get('train_losses', [])))
            }
            
            print(f"  ✅ 准确率: {accuracy:.4f}")
            print(f"  ⏱️ 训练时间: {train_time:.2f}s")
            print(f"  🎯 预测时间: {predict_time:.4f}s")
            print(f"  💾 峰值内存: {peak / (1024 * 1024):.1f}MB")
        
        # 3. 性能报告
        print("\n" + "=" * 80)
        print("📊 性能分析报告")
        print("=" * 80)
        print(f"{'配置':<12} {'准确率':<8} {'训练时间':<10} {'预测时间':<10} {'峰值内存':<10} {'参数数量':<10}")
        print("-" * 80)
        
        for config_name, result in performance_results.items():
            print(f"{config_name:<12} {result['accuracy']:<8.4f} {result['train_time']:<10.2f} "
                  f"{result['predict_time']:<10.4f} {result['memory_peak_mb']:<10.1f} {result['parameters']:<10}")
        
        # 4. 效率分析
        print("\n📈 效率分析:")
        for config_name, result in performance_results.items():
            efficiency = result['accuracy'] / result['train_time']
            memory_efficiency = result['accuracy'] / result['memory_peak_mb']
            print(f"  {config_name}: 时间效率={efficiency:.6f}, 内存效率={memory_efficiency:.6f}")
        
        # 5. 保存性能报告
        report_file = Path("examples/advanced_results/performance_report.json")
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'system_info': {
                    'python_version': platform.python_version(),
                    'platform': platform.platform()
                },
                'performance_results': performance_results
            }, f, indent=2)
        
        print(f"💾 性能报告已保存: {report_file}")
        
        return performance_results


def main():
    """
    运行所有高级配置示例
    """
    print("⚙️ CAAC项目高级配置示例")
    print("=" * 60)
    print("本示例展示高级配置管理和性能优化技巧")
    print()
    
    runner = AdvancedExperimentRunner()
    
    try:
        # 1. 自适应超参数调优
        best_config, search_results = runner.adaptive_hyperparameter_tuning()
        
        # 2. 配置模板实验
        template_results = runner.configuration_template_experiment()
        
        # 3. 环境自适应配置
        env_config = runner.environment_adaptive_config()
        
        # 4. 配置验证和测试
        validation_rules = runner.config_validation_and_testing()
        
        # 5. 性能分析实验
        performance_results = runner.performance_profiling_experiment()
        
        # 总结
        print("\n" + "=" * 60)
        print("🎊 所有高级配置示例完成!")
        print("=" * 60)
        print("📊 主要成果:")
        print(f"  - 最佳超参数准确率: {max(r[1] for r in search_results):.4f}")
        print(f"  - 配置模板测试: {len(template_results)} 个模板")
        print(f"  - 环境自适应: 已保存环境特定配置")
        print(f"  - 性能分析: {len(performance_results)} 个配置对比")
        print()
        print("💡 生成的文件:")
        print("  - examples/configs/*.json|*.yaml - 配置模板")
        print("  - examples/advanced_results/ - 实验结果")
        print()
        print("🎯 最佳实践总结:")
        print("  1. 根据数据集大小选择合适的配置模板")
        print("  2. 使用自适应超参数搜索优化性能")
        print("  3. 考虑运行环境调整配置参数")
        print("  4. 定期验证和测试配置兼容性")
        print("  5. 监控性能指标优化资源使用")
        
    except Exception as e:
        print(f"❌ 运行高级配置示例时出错: {e}")
        print("请检查环境配置和依赖安装")


if __name__ == "__main__":
    main() 