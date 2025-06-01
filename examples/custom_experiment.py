"""
🔬 CAAC项目自定义实验示例

本文件展示如何创建自定义实验配置，包括：
- 自定义实验参数
- 交互式实验设计
- 高级实验控制

请在项目根目录运行：python examples/custom_experiment.py
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.experiments.experiment_manager import ExperimentManager
from src.models.caac_ovr_model import CAACOvRModel, CAACOvRGaussianModel
from sklearn.datasets import make_classification, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt


class CustomExperimentDesigner:
    """
    自定义实验设计器
    """
    
    def __init__(self, results_base_dir="examples/custom_results"):
        self.manager = ExperimentManager(base_results_dir=results_base_dir)
        self.results_base_dir = results_base_dir
        
    def create_synthetic_dataset_experiment(self):
        """
        示例1: 合成数据集实验
        """
        print("=" * 60)
        print("🔬 示例1: 合成数据集实验")
        print("=" * 60)
        
        # 1. 创建合成数据集
        print("📊 创建合成数据集...")
        datasets = {}
        
        # 简单分离数据
        X_easy, y_easy = make_classification(
            n_samples=1000, n_features=20, n_classes=3, n_informative=15,
            n_redundant=5, random_state=42, class_sep=2.0
        )
        datasets['easy'] = (X_easy, y_easy)
        
        # 困难分离数据
        X_hard, y_hard = make_classification(
            n_samples=1000, n_features=20, n_classes=3, n_informative=10,
            n_redundant=5, random_state=42, class_sep=0.5
        )
        datasets['hard'] = (X_hard, y_hard)
        
        # 高维数据
        X_highdim, y_highdim = make_classification(
            n_samples=1000, n_features=100, n_classes=5, n_informative=50,
            n_redundant=25, random_state=42, class_sep=1.0
        )
        datasets['high_dimensional'] = (X_highdim, y_highdim)
        
        # 2. 对每个数据集运行实验
        results = {}
        for name, (X, y) in datasets.items():
            print(f"\n🔥 测试数据集: {name}")
            
            # 数据预处理
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 创建和训练模型
            model = CAACOvRModel(
                input_dim=X.shape[1],
                n_classes=len(np.unique(y)),
                representation_dim=min(64, X.shape[1]),
                epochs=50,
                verbose=0
            )
            
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
            
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='macro')
            
            results[name] = {'accuracy': accuracy, 'f1': f1}
            print(f"  ✅ 准确率: {accuracy:.4f}, F1: {f1:.4f}")
        
        # 3. 保存结果
        self._save_experiment_results("synthetic_dataset_experiment", results)
        return results
    
    def parameter_sensitivity_experiment(self):
        """
        示例2: 参数敏感性实验
        """
        print("\n" + "=" * 60)
        print("⚙️ 示例2: 参数敏感性实验")
        print("=" * 60)
        
        # 1. 准备数据
        print("📊 加载Digits数据集...")
        data = load_digits()
        X, y = data.data, data.target
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 2. 定义参数网格
        param_grids = {
            'representation_dim': [32, 64, 128],
            'learning_rate': [0.001, 0.01, 0.1],
            'uniqueness_constraint': [False, True],
            'learnable_thresholds': [False, True]
        }
        
        # 3. 运行参数敏感性分析
        results = {}
        base_config = {
            'input_dim': X.shape[1],
            'n_classes': len(np.unique(y)),
            'epochs': 30,
            'verbose': 0
        }
        
        for param_name, param_values in param_grids.items():
            print(f"\n🔧 测试参数: {param_name}")
            param_results = []
            
            for value in param_values:
                config = base_config.copy()
                if param_name == 'learning_rate':
                    config['lr'] = value
                else:
                    config[param_name] = value
                
                # 创建和训练模型
                model = CAACOvRModel(**config)
                model.fit(X_train_scaled, y_train)
                predictions = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, predictions)
                
                param_results.append(accuracy)
                print(f"  {param_name}={value}: {accuracy:.4f}")
            
            results[param_name] = dict(zip(param_values, param_results))
        
        # 4. 可视化结果
        self._plot_parameter_sensitivity(results)
        self._save_experiment_results("parameter_sensitivity_experiment", results)
        return results
    
    def distribution_comparison_experiment(self):
        """
        示例3: 分布对比实验 (Cauchy vs Gaussian)
        """
        print("\n" + "=" * 60)
        print("📊 示例3: 分布对比实验")
        print("=" * 60)
        
        # 1. 准备多个数据集
        datasets = {
            'digits': load_digits(),
        }
        
        # 添加合成数据集
        X_synthetic, y_synthetic = make_classification(
            n_samples=1000, n_features=20, n_classes=4,
            n_informative=15, random_state=42
        )
        datasets['synthetic'] = type('obj', (object,), {
            'data': X_synthetic, 'target': y_synthetic
        })()
        
        # 2. 对比Cauchy和Gaussian分布
        all_results = {}
        
        for dataset_name, data in datasets.items():
            print(f"\n📊 测试数据集: {dataset_name}")
            X, y = data.data, data.target
            
            # 数据预处理
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 测试不同分布
            models = {
                'CAAC_Cauchy': CAACOvRModel(
                    input_dim=X.shape[1],
                    n_classes=len(np.unique(y)),
                    representation_dim=64,
                    epochs=50,
                    verbose=0
                ),
                'CAAC_Gaussian': CAACOvRGaussianModel(
                    input_dim=X.shape[1],
                    n_classes=len(np.unique(y)),
                    representation_dim=64,
                    epochs=50,
                    verbose=0
                )
            }
            
            dataset_results = {}
            for model_name, model in models.items():
                print(f"  🔥 训练 {model_name}...")
                model.fit(X_train_scaled, y_train)
                predictions = model.predict(X_test_scaled)
                
                accuracy = accuracy_score(y_test, predictions)
                f1 = f1_score(y_test, predictions, average='macro')
                
                dataset_results[model_name] = {
                    'accuracy': accuracy,
                    'f1_score': f1
                }
                print(f"    ✅ 准确率: {accuracy:.4f}, F1: {f1:.4f}")
            
            all_results[dataset_name] = dataset_results
        
        # 3. 保存和可视化结果
        self._save_experiment_results("distribution_comparison_experiment", all_results)
        self._plot_distribution_comparison(all_results)
        return all_results
    
    def noise_robustness_custom_experiment(self):
        """
        示例4: 自定义噪声鲁棒性实验
        """
        print("\n" + "=" * 60)
        print("🛡️ 示例4: 自定义噪声鲁棒性实验")
        print("=" * 60)
        
        # 1. 准备数据
        X, y = make_classification(
            n_samples=2000, n_features=20, n_classes=3,
            n_informative=15, random_state=42, class_sep=1.5
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 2. 自定义噪声水平和类型
        noise_configs = [
            {'type': 'label_noise', 'levels': [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]},
            {'type': 'feature_noise', 'levels': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]}
        ]
        
        all_results = {}
        
        for noise_config in noise_configs:
            noise_type = noise_config['type']
            noise_levels = noise_config['levels']
            
            print(f"\n🔧 测试噪声类型: {noise_type}")
            
            type_results = {}
            for noise_level in noise_levels:
                print(f"  噪声水平: {noise_level}")
                
                # 应用噪声
                if noise_type == 'label_noise':
                    X_train_noisy = X_train_scaled.copy()
                    y_train_noisy = self._add_label_noise(y_train, noise_level)
                else:  # feature_noise
                    X_train_noisy = self._add_feature_noise(X_train_scaled, noise_level)
                    y_train_noisy = y_train.copy()
                
                # 训练模型
                model = CAACOvRModel(
                    input_dim=X.shape[1],
                    n_classes=len(np.unique(y)),
                    representation_dim=64,
                    epochs=50,
                    learnable_thresholds=True,
                    uniqueness_constraint=True,
                    verbose=0
                )
                
                model.fit(X_train_noisy, y_train_noisy)
                predictions = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, predictions)
                
                type_results[noise_level] = accuracy
                print(f"    准确率: {accuracy:.4f}")
            
            all_results[noise_type] = type_results
        
        # 3. 可视化和保存结果
        self._plot_noise_robustness(all_results)
        self._save_experiment_results("noise_robustness_custom_experiment", all_results)
        return all_results
    
    def interactive_experiment_design(self):
        """
        示例5: 交互式实验设计
        """
        print("\n" + "=" * 60)
        print("🎮 示例5: 交互式实验设计 (自动化版本)")
        print("=" * 60)
        
        # 模拟用户选择
        print("📝 自动配置实验参数...")
        
        experiment_config = {
            'dataset_type': 'synthetic',  # 模拟选择
            'dataset_params': {
                'n_samples': 1500,
                'n_features': 30,
                'n_classes': 4,
                'class_sep': 1.0
            },
            'model_config': {
                'representation_dim': 64,
                'learnable_thresholds': True,
                'uniqueness_constraint': True,
                'epochs': 100
            },
            'experiment_params': {
                'test_size': 0.3,
                'cross_validation': True,
                'n_folds': 5
            }
        }
        
        print("⚙️ 选择的配置:")
        print(json.dumps(experiment_config, indent=2))
        
        # 执行自定义实验
        result_dir = self.manager.run_custom_experiment(
            experiment_type='robustness',
            config={
                'datasets': ['synthetic_imbalanced'],
                'noise_levels': [0.0, 0.1, 0.2],
                'epochs': experiment_config['model_config']['epochs'],
                'representation_dim': experiment_config['model_config']['representation_dim']
            },
            save_name='interactive_designed_experiment'
        )
        
        print(f"✅ 交互式实验完成! 结果保存在: {result_dir}")
        return result_dir
    
    def _add_label_noise(self, y, noise_level):
        """添加标签噪声"""
        if noise_level == 0:
            return y.copy()
        
        n_samples = len(y)
        n_noisy = int(n_samples * noise_level)
        noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
        
        y_noisy = y.copy()
        unique_labels = np.unique(y)
        
        for idx in noisy_indices:
            current_label = y_noisy[idx]
            possible_labels = unique_labels[unique_labels != current_label]
            y_noisy[idx] = np.random.choice(possible_labels)
        
        return y_noisy
    
    def _add_feature_noise(self, X, noise_level):
        """添加特征噪声"""
        if noise_level == 0:
            return X.copy()
        
        noise = np.random.normal(0, noise_level, X.shape)
        return X + noise
    
    def _plot_parameter_sensitivity(self, results):
        """绘制参数敏感性图"""
        try:
            n_params = len(results)
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for i, (param_name, param_results) in enumerate(results.items()):
                if i < len(axes):
                    ax = axes[i]
                    values = list(param_results.keys())
                    accuracies = list(param_results.values())
                    
                    ax.plot(values, accuracies, 'o-', linewidth=2, markersize=8)
                    ax.set_title(f'Sensitivity to {param_name}', fontsize=12)
                    ax.set_xlabel(param_name)
                    ax.set_ylabel('Accuracy')
                    ax.grid(True, alpha=0.3)
            
            # 隐藏多余的子图
            for i in range(len(results), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f'{self.results_base_dir}/parameter_sensitivity.png', dpi=300, bbox_inches='tight')
            print("📈 参数敏感性图已保存")
            plt.close()
        except Exception as e:
            print(f"⚠️ 绘图失败: {e}")
    
    def _plot_distribution_comparison(self, results):
        """绘制分布对比图"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            datasets = list(results.keys())
            models = list(results[datasets[0]].keys())
            
            metrics = ['accuracy', 'f1_score']
            metric_names = ['Accuracy', 'F1 Score']
            
            for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
                ax = axes[i]
                
                x = np.arange(len(datasets))
                width = 0.35
                
                for j, model in enumerate(models):
                    values = [results[dataset][model][metric] for dataset in datasets]
                    ax.bar(x + j*width, values, width, label=model)
                
                ax.set_xlabel('Dataset')
                ax.set_ylabel(metric_name)
                ax.set_title(f'{metric_name} Comparison')
                ax.set_xticks(x + width/2)
                ax.set_xticklabels(datasets)
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{self.results_base_dir}/distribution_comparison.png', dpi=300, bbox_inches='tight')
            print("📊 分布对比图已保存")
            plt.close()
        except Exception as e:
            print(f"⚠️ 绘图失败: {e}")
    
    def _plot_noise_robustness(self, results):
        """绘制噪声鲁棒性图"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            for i, (noise_type, type_results) in enumerate(results.items()):
                ax = axes[i]
                
                noise_levels = list(type_results.keys())
                accuracies = list(type_results.values())
                
                ax.plot(noise_levels, accuracies, 'o-', linewidth=2, markersize=8)
                ax.set_title(f'Robustness to {noise_type.replace("_", " ").title()}')
                ax.set_xlabel('Noise Level')
                ax.set_ylabel('Accuracy')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{self.results_base_dir}/noise_robustness.png', dpi=300, bbox_inches='tight')
            print("🛡️ 噪声鲁棒性图已保存")
            plt.close()
        except Exception as e:
            print(f"⚠️ 绘图失败: {e}")
    
    def _save_experiment_results(self, experiment_name, results):
        """保存实验结果"""
        try:
            os.makedirs(self.results_base_dir, exist_ok=True)
            
            result_file = f"{self.results_base_dir}/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # 处理numpy类型以便JSON序列化
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                return obj
            
            results_serializable = convert_numpy(results)
            
            with open(result_file, 'w') as f:
                json.dump({
                    'experiment_name': experiment_name,
                    'timestamp': datetime.now().isoformat(),
                    'results': results_serializable
                }, f, indent=2)
            
            print(f"💾 结果已保存到: {result_file}")
        except Exception as e:
            print(f"⚠️ 保存结果失败: {e}")


def main():
    """
    运行所有自定义实验示例
    """
    print("🔬 CAAC项目自定义实验示例")
    print("=" * 60)
    print("本示例展示如何设计和运行自定义实验")
    print()
    
    # 创建实验设计器
    designer = CustomExperimentDesigner()
    
    try:
        # 运行各种自定义实验
        print("🚀 开始运行自定义实验...")
        
        # 1. 合成数据集实验
        synthetic_results = designer.create_synthetic_dataset_experiment()
        
        # 2. 参数敏感性实验
        sensitivity_results = designer.parameter_sensitivity_experiment()
        
        # 3. 分布对比实验
        distribution_results = designer.distribution_comparison_experiment()
        
        # 4. 噪声鲁棒性实验
        noise_results = designer.noise_robustness_custom_experiment()
        
        # 5. 交互式实验设计
        interactive_result = designer.interactive_experiment_design()
        
        # 总结
        print("\n" + "=" * 60)
        print("🎊 所有自定义实验完成!")
        print("=" * 60)
        print("📊 实验结果总结:")
        print(f"  - 合成数据集实验: {len(synthetic_results)} 个数据集测试")
        print(f"  - 参数敏感性实验: {len(sensitivity_results)} 个参数测试")
        print(f"  - 分布对比实验: {len(distribution_results)} 个数据集对比")
        print(f"  - 噪声鲁棒性实验: {len(noise_results)} 种噪声类型测试")
        print(f"  - 交互式实验: 结果保存在 {interactive_result}")
        print()
        print("💡 生成的文件:")
        print("  - examples/custom_results/*.json - 详细结果数据")
        print("  - examples/custom_results/*.png - 可视化图表")
        print()
        print("🎯 下一步建议:")
        print("  1. 查看生成的可视化图表分析结果")
        print("  2. 基于参数敏感性结果优化模型配置")
        print("  3. 参考这些示例设计你自己的实验")
        
    except Exception as e:
        print(f"❌ 运行自定义实验时出错: {e}")
        print("请检查环境配置和依赖安装")


if __name__ == "__main__":
    main() 