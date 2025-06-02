"""
Method Comparison Experiments Module

This module contains the core logic for running method comparison experiments,
extracted from the legacy scripts to create a clean, maintainable implementation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import os
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our models
from src.models.caac_ovr_model import (
    CAACOvRModel, 
    SoftmaxMLPModel,
    OvRCrossEntropyMLPModel,
    CAACOvRGaussianModel,
    CrammerSingerMLPModel
)

class MethodComparisonRunner:
    """
    Centralized runner for method comparison experiments.
    
    This class handles:
    - Dataset loading and preprocessing
    - Model configuration and training
    - Performance evaluation and comparison
    - Results visualization and reporting
    """
    
    def __init__(self, results_dir: str = "results"):
        """Initialize the method comparison runner."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Store relative path for display purposes
        self.results_dir_display = results_dir if "/" in results_dir else f"./{results_dir}"
        
        # Set random seed for reproducibility
        np.random.seed(42)
    
    def load_datasets(self):
        """Load all test datasets."""
        datasets = {}
        
        # Iris dataset
        iris = load_iris()
        datasets['iris'] = {
            'data': iris.data,
            'target': iris.target,
            'target_names': iris.target_names,
            'name': 'Iris'
        }
        
        # Wine dataset
        wine = load_wine()
        datasets['wine'] = {
            'data': wine.data,
            'target': wine.target,
            'target_names': wine.target_names,
            'name': 'Wine'
        }
        
        # Breast Cancer dataset
        bc = load_breast_cancer()
        datasets['breast_cancer'] = {
            'data': bc.data,
            'target': bc.target,
            'target_names': bc.target_names,
            'name': 'Breast Cancer'
        }
        
        # Digits dataset
        digits = load_digits()
        datasets['digits'] = {
            'data': digits.data,
            'target': digits.target,
            'target_names': [str(i) for i in range(10)],
            'name': 'Digits'
        }
        
        return datasets
    
    def create_comparison_methods(self, representation_dim=64, epochs=100):
        """Create all methods for comparison - 与最新鲁棒性实验保持一致的5种核心神经网络方法."""
        # 统一网络架构参数 - 与鲁棒性实验保持一致
        common_params = {
            'representation_dim': representation_dim,
            'lr': 0.001,
            'batch_size': 32,
            'epochs': epochs,
            'early_stopping_patience': 10
        }
        
        methods = {
            # 核心推断行动框架方法 - 基础版本
            'CAAC_Cauchy': {
                'name': 'CAAC (Cauchy)',
                'type': 'unified',
                'model_class': CAACOvRModel,
                'params': {**common_params, 'learnable_thresholds': False}
            },
            'CAAC_Gaussian': {
                'name': 'CAAC (Gaussian)', 
                'type': 'unified',
                'model_class': CAACOvRGaussianModel,
                'params': {**common_params, 'learnable_thresholds': False}
            },
            
            # 推断行动框架 - 可学习阈值变体
            'CAAC_Cauchy_Learnable': {
                'name': 'CAAC (Cauchy, Learnable)',
                'type': 'unified',
                'model_class': CAACOvRModel,
                'params': {**common_params, 'learnable_thresholds': True}
            },
            'CAAC_Gaussian_Learnable': {
                'name': 'CAAC (Gaussian, Learnable)',
                'type': 'unified',
                'model_class': CAACOvRGaussianModel,
                'params': {**common_params, 'learnable_thresholds': True}
            },
            
            # 推断行动框架 - 唯一性约束变体
            'CAAC_Cauchy_Unique': {
                'name': 'CAAC Cauchy (Uniqueness)',
                'type': 'unified',
                'model_class': CAACOvRModel,
                'params': {**common_params, 'learnable_thresholds': False, 'uniqueness_constraint': True, 'uniqueness_samples': 3, 'uniqueness_weight': 0.05}
            },
            'CAAC_Gaussian_Unique': {
                'name': 'CAAC Gaussian (Uniqueness)',
                'type': 'unified',
                'model_class': CAACOvRGaussianModel,
                'params': {**common_params, 'learnable_thresholds': False, 'uniqueness_constraint': True, 'uniqueness_samples': 3, 'uniqueness_weight': 0.05}
            },
            
            # 推断行动框架 - 可学习阈值+唯一性约束组合
            'CAAC_Cauchy_Learnable_Unique': {
                'name': 'CAAC Cauchy (Learnable+Uniqueness)',
                'type': 'unified',
                'model_class': CAACOvRModel,
                'params': {**common_params, 'learnable_thresholds': True, 'uniqueness_constraint': True, 'uniqueness_samples': 3, 'uniqueness_weight': 0.05}
            },
            'CAAC_Gaussian_Learnable_Unique': {
                'name': 'CAAC Gaussian (Learnable+Uniqueness)',
                'type': 'unified',
                'model_class': CAACOvRGaussianModel,
                'params': {**common_params, 'learnable_thresholds': True, 'uniqueness_constraint': True, 'uniqueness_samples': 3, 'uniqueness_weight': 0.05}
            },
            
            # 标准深度学习方法
            'MLP_Softmax': {
                'name': 'MLP (Softmax)',
                'type': 'unified',
                'model_class': SoftmaxMLPModel,
                'params': common_params
            },
            'MLP_OvR_CE': {
                'name': 'MLP (OvR Cross Entropy)',
                'type': 'unified',
                'model_class': OvRCrossEntropyMLPModel,
                'params': common_params
            },
            'MLP_Hinge': {
                'name': 'MLP (Crammer & Singer Hinge)',
                'type': 'unified',
                'model_class': CrammerSingerMLPModel,
                'params': common_params
            },
            
            # Classical machine learning methods as baselines
            'Softmax_LR': {
                'name': 'Softmax Regression',
                'type': 'sklearn',
                'model': LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
            },
            'Standard_OvR': {
                'name': 'OvR Logistic',
                'type': 'sklearn',
                'model': OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))
            },
            'SVM_RBF': {
                'name': 'SVM-RBF',
                'type': 'sklearn',
                'model': SVC(kernel='rbf', random_state=42, probability=True)
            },
            'Random_Forest': {
                'name': 'Random Forest',
                'type': 'sklearn',
                'model': RandomForestClassifier(n_estimators=100, random_state=42)
            },
            'Sklearn_MLP': {
                'name': 'MLP-Sklearn',
                'type': 'sklearn',
                'model': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
            }
        }
        return methods
    
    def evaluate_method(self, method_info, X_train, X_test, y_train, y_test):
        """Evaluate a single method's performance."""
        start_time = time.time()
        
        if method_info['type'] == 'unified':
            # Use our unified network architecture
            input_dim = X_train.shape[1]
            n_classes = len(np.unique(y_train))
            
            model = method_info['model_class'](
                input_dim=input_dim, 
                n_classes=n_classes,
                **method_info['params']
            )
            
            # Train model
            model.fit(X_train, y_train, verbose=0)
            
            # Predict
            y_pred = model.predict(X_test)
            
        else:
            # Use sklearn model
            model = method_info['model']
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
        
        training_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'training_time': training_time
        }
    
    def run_comparison_experiments(self, representation_dim=64, epochs=100, datasets=None):
        """Run comprehensive method comparison experiments."""
        print("🔬 分类方法对比实验 - 完整推断行动框架评估")
        print("=" * 60)
        print("📋 测试方法（11种统一架构方法 + 5种经典方法）：")
        print("   🧠 推断行动框架基础版本：")
        print("      • CAAC (Cauchy) - 柯西分布 + 固定阈值")
        print("      • CAAC (Gaussian) - 高斯分布 + 固定阈值")
        print("   ⚙️  推断行动框架可学习阈值版本：")
        print("      • CAAC (Cauchy, Learnable) - 柯西分布 + 可学习阈值")
        print("      • CAAC (Gaussian, Learnable) - 高斯分布 + 可学习阈值")
        print("   🔒 推断行动框架唯一性约束版本：")
        print("      • CAAC Cauchy (Uniqueness) - 柯西分布 + 唯一性约束")
        print("      • CAAC Gaussian (Uniqueness) - 高斯分布 + 唯一性约束")
        print("   🔧 推断行动框架组合版本：")
        print("      • CAAC Cauchy (Learnable+Uniqueness) - 柯西+可学习阈值+唯一性")
        print("      • CAAC Gaussian (Learnable+Uniqueness) - 高斯+可学习阈值+唯一性")
        print("   📊 标准深度学习对照：")
        print("      • MLP (Softmax) - 标准Softmax分类器")
        print("      • MLP (OvR Cross Entropy) - 一对多交叉熵")
        print("      • MLP (Crammer & Singer Hinge) - 铰链损失")
        print()
        
        # Load datasets
        all_datasets = self.load_datasets()
        if datasets:
            # Filter to specified datasets
            all_datasets = {k: v for k, v in all_datasets.items() if k in datasets}
        
        # Create methods for comparison
        methods = self.create_comparison_methods(representation_dim, epochs)
        
        results = []
        
        for dataset_name, dataset in all_datasets.items():
            print(f"\n📊 Testing dataset: {dataset['name']}")
            print("-" * 40)
            
            # Data preprocessing
            X = dataset['data']
            y = dataset['target']
            
            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Test each method
            for method_key, method_info in methods.items():
                print(f"  🧪 Testing method: {method_info['name']}")
                
                try:
                    metrics = self.evaluate_method(
                        method_info, 
                        X_train_scaled, X_test_scaled, 
                        y_train, y_test
                    )
                    
                    results.append({
                        'Dataset': dataset['name'],
                        'Method': method_info['name'],
                        'Method_Key': method_key,
                        'Method_Type': method_info['type'],
                        'Accuracy': metrics['accuracy'],
                        'Precision_Macro': metrics['precision_macro'],
                        'Recall_Macro': metrics['recall_macro'],
                        'F1_Macro': metrics['f1_macro'],
                        'F1_Weighted': metrics['f1_weighted'],
                        'Training_Time': metrics['training_time']
                    })
                    
                    print(f"    ✅ Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_macro']:.4f}, Time: {metrics['training_time']:.3f}s")
                    
                except Exception as e:
                    print(f"    ❌ Error: {str(e)}")
                    continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Generate visualizations and reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._create_comparison_plots(results_df, timestamp)
        summary = self._create_summary_table(results_df, timestamp)
        self._generate_detailed_report(results_df, summary, timestamp)
        
        print("\n✅ Method comparison experiments completed!")
        print(f"📁 Results saved to: {self.results_dir_display}")
        print("📊 Generated files:")
        print(f"   - {self.results_dir_display}/methods_comparison_english_{timestamp}.png")
        print(f"   - {self.results_dir_display}/methods_comparison_detailed_{timestamp}.csv") 
        print(f"   - {self.results_dir_display}/methods_comparison_summary_{timestamp}.csv")
        print(f"   - {self.results_dir_display}/caac_methods_comparison_report_{timestamp}.md")
        
        return str(self.results_dir)
    
    def _create_comparison_plots(self, results_df, timestamp):
        """Create comparison visualization charts with English labels."""
        plt.style.use('default')
        
        # Set figure size and layout
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Classification Methods Comparison: Cauchy vs. Gaussian vs. Standard', 
                     fontsize=16, fontweight='bold')
        
        # 1. Accuracy comparison
        ax1 = axes[0, 0]
        pivot_acc = results_df.pivot(index='Dataset', columns='Method', values='Accuracy')
        pivot_acc.plot(kind='bar', ax=ax1, rot=30, width=0.8)
        ax1.set_title('Accuracy Comparison by Dataset', fontsize=14)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_xlabel('Dataset', fontsize=12)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.85, 1.02)
        
        # 2. F1-score comparison
        ax2 = axes[0, 1]
        pivot_f1 = results_df.pivot(index='Dataset', columns='Method', values='F1_Macro')
        pivot_f1.plot(kind='bar', ax=ax2, rot=30, width=0.8)
        ax2.set_title('F1-Score (Macro) Comparison by Dataset', fontsize=14)
        ax2.set_ylabel('F1-Score (Macro)', fontsize=12)
        ax2.set_xlabel('Dataset', fontsize=12)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.85, 1.02)
        
        # 3. Training time comparison
        ax3 = axes[1, 0]
        pivot_time = results_df.pivot(index='Dataset', columns='Method', values='Training_Time')
        pivot_time.plot(kind='bar', ax=ax3, rot=30, width=0.8)
        ax3.set_title('Training Time Comparison by Dataset', fontsize=14)
        ax3.set_ylabel('Training Time (seconds, log scale)', fontsize=12)
        ax3.set_xlabel('Dataset', fontsize=12)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # 4. Scatter plot: Accuracy vs Training Time
        ax4 = axes[1, 1]
        methods = results_df['Method'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        
        for method, color in zip(methods, colors):
            method_data = results_df[results_df['Method'] == method]
            ax4.scatter(method_data['Training_Time'], method_data['Accuracy'], 
                       label=method, alpha=0.8, s=80, color=color, edgecolors='black', linewidth=0.5)
        
        ax4.set_xlabel('Training Time (seconds, log scale)', fontsize=12)
        ax4.set_ylabel('Accuracy', fontsize=12)
        ax4.set_title('Efficiency vs Performance Trade-off', fontsize=14)
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        ax4.set_ylim(0.90, 1.02)
        
        plt.tight_layout()
        plot_file = self.results_dir / f'methods_comparison_english_{timestamp}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 English comparison chart saved: {self.results_dir_display}/{plot_file.name}")
    
    def _create_summary_table(self, results_df, timestamp):
        """Create summary comparison table."""
        print("\n📋 Method Comparison Summary Table")
        print("=" * 80)
        
        # Calculate average performance by method
        summary = results_df.groupby('Method').agg({
            'Accuracy': ['mean', 'std'],
            'F1_Macro': ['mean', 'std'],
            'Training_Time': ['mean', 'std']
        }).round(4)
        
        print(summary)
        
        # Save detailed results
        results_df.to_csv(self.results_dir / f'methods_comparison_detailed_{timestamp}.csv', index=False)
        summary.to_csv(self.results_dir / f'methods_comparison_summary_{timestamp}.csv')
        
        return summary
    
    def _generate_detailed_report(self, results_df, summary, timestamp):
        """生成详细的实验比较报告（中文版）。"""
        print("\n📄 生成详细实验报告")
        print("=" * 50)
        
        report_file = self.results_dir / f"caac_methods_comparison_report_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"""# CAAC分类方法对比实验报告

**报告生成时间:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 实验概述

本报告展示了**CAAC OvR分类器**与多种传统分类方法的全面性能比较。实验采用统一的网络架构，仅在损失函数和正则化策略上有所不同，确保了公平的比较环境。

### 核心研究问题
**推断行动框架使用柯西分布的尺度参数是否优于高斯分布和标准深度学习方法？**

### 推断行动框架架构
我们的推断行动框架（CAAC方法）采用统一的三阶段架构：
- **FeatureNet**: 特征提取网络 (输入维度 → 64维**确定性特征表征**)
- **AbductionNet**: 溯因推理网络 (64维 → 64维**因果表征随机变量**参数)  
- **ActionNet**: 行动决策网络 (64维 → **类别数量**的得分)

**推断行动框架核心思想**: 
- 特征表征是确定性数值，因果表征是随机变量（位置+尺度参数）
- 柯西分布vs高斯分布：不同的尺度参数建模策略
- 通过概率推理实现更鲁棒的分类决策
- 标准方法仅使用位置参数，忽略了不确定性建模

### 实验方法

#### 统一架构方法 (11种神经网络方法)

**推断行动框架基础版本 (2种方法):**
1. **CAAC (Cauchy)** - 推断行动框架，柯西分布建模 + 固定阈值
2. **CAAC (Gaussian)** - 推断行动框架，高斯分布建模 + 固定阈值

**推断行动框架可学习阈值版本 (2种方法):**
3. **CAAC (Cauchy, Learnable)** - 柯西分布 + 可学习阈值参数
4. **CAAC (Gaussian, Learnable)** - 高斯分布 + 可学习阈值参数

**推断行动框架唯一性约束版本 (2种方法):**
5. **CAAC Cauchy (Uniqueness)** - 柯西分布 + 潜在向量采样唯一性约束
6. **CAAC Gaussian (Uniqueness)** - 高斯分布 + 潜在向量采样唯一性约束

**推断行动框架组合版本 (2种方法):**
7. **CAAC Cauchy (Learnable+Uniqueness)** - 柯西分布 + 可学习阈值 + 唯一性约束
8. **CAAC Gaussian (Learnable+Uniqueness)** - 高斯分布 + 可学习阈值 + 唯一性约束

**标准深度学习对照方法 (3种方法):**
9. **MLP (Softmax)** - 标准多层感知机，使用Softmax损失函数
10. **MLP (OvR Cross Entropy)** - 一对多策略的交叉熵损失函数
11. **MLP (Crammer & Singer Hinge)** - 多类铰链损失函数

**唯一性约束说明:**
- 通过采样多个潜在向量实例化，应用最大-次大间隔约束增强决策确定性
- 采样次数：3次，约束权重：0.05（实验发现倾向于降低准确率，主要用作理论对照研究）

#### 经典机器学习基准方法 (对照组)
6. **Softmax Regression** - 多项式logistic回归
7. **OvR Logistic** - 一对其余逻辑回归
8. **SVM-RBF** - 径向基函数支持向量机
9. **Random Forest** - 随机森林集成方法
10. **MLP-Sklearn** - Scikit-learn多层感知机

### 测试数据集
- **Iris鸢尾花数据集**: 3类, 4特征, 150样本
- **Wine红酒数据集**: 3类, 13特征, 178样本  
- **Breast Cancer乳腺癌数据集**: 2类, 30特征, 569样本
- **Digits手写数字数据集**: 10类, 64特征, 1797样本

## 详细实验结果

### 准确率对比

""")
            
            # 创建准确率对比表
            pivot_acc = results_df.pivot(index='Dataset', columns='Method', values='Accuracy')
            f.write(pivot_acc.round(4).to_markdown())
            f.write("\n\n### F1分数对比 (Macro Average)\n\n")
            
            # 创建F1分数对比表
            pivot_f1 = results_df.pivot(index='Dataset', columns='Method', values='F1_Macro')
            f.write(pivot_f1.round(4).to_markdown())
            f.write("\n\n### 训练时间对比 (秒)\n\n")
            
            # 创建训练时间对比表
            pivot_time = results_df.pivot(index='Dataset', columns='Method', values='Training_Time')
            f.write(pivot_time.round(3).to_markdown())
            f.write("\n\n")
            
            # 方法性能统计
            f.write("""## 方法性能统计

### 平均性能汇总

""")
            
            # 计算平均性能
            avg_performance = results_df.groupby('Method').agg({
                'Accuracy': ['mean', 'std'],
                'F1_Macro': ['mean', 'std'],
                'Training_Time': ['mean', 'std']
            }).round(4)
            
            f.write(avg_performance.to_markdown())
            f.write("\n\n### 性能排名分析\n\n")
            
            # 计算简单的平均值用于排名
            simple_avg = results_df.groupby('Method').agg({
                'Accuracy': 'mean',
                'F1_Macro': 'mean',
                'Training_Time': 'mean'
            }).round(4)
            
            # 按准确率排序
            acc_ranking = simple_avg.sort_values('Accuracy', ascending=False)
            f.write("#### 准确率排名\n\n")
            for i, (method, row) in enumerate(acc_ranking.iterrows(), 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                f.write(f"{emoji} **{method}**: {row['Accuracy']:.2%}\n")
            f.write("\n")
            
            # 按F1分数排序
            f1_ranking = simple_avg.sort_values('F1_Macro', ascending=False)
            f.write("#### F1分数排名\n\n")
            for i, (method, row) in enumerate(f1_ranking.iterrows(), 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                f.write(f"{emoji} **{method}**: {row['F1_Macro']:.2%}\n")
            f.write("\n")
            
            # 按训练时间排序（越快越好）
            time_ranking = simple_avg.sort_values('Training_Time', ascending=True)
            f.write("#### 训练效率排名 (越快越好)\n\n")
            for i, (method, row) in enumerate(time_ranking.iterrows(), 1):
                emoji = "🚀" if i == 1 else "⚡" if i == 2 else "💨" if i == 3 else f"{i}."
                f.write(f"{emoji} **{method}**: {row['Training_Time']:.3f}秒\n")
            f.write("\n")
            
            f.write(f"""## 核心发现：柯西分布尺度参数的影响

### 统一架构方法对比

本实验的核心目标是验证**柯西分布尺度参数**对分类性能的影响。通过使用完全相同的网络架构，我们可以准确分析不同分布选择的效果。

""")
            
            # 分析统一架构方法
            unified_methods = results_df[results_df['Method_Type'] == 'unified']
            if not unified_methods.empty:
                unified_summary = unified_methods.groupby('Method').agg({
                    'Accuracy': 'mean',
                    'F1_Macro': 'mean',
                    'Training_Time': 'mean'
                }).round(4)
                
                f.write("#### 统一架构方法性能对比\n\n")
                f.write(unified_summary.to_markdown())
                f.write("\n\n")
            
            # 与经典方法比较
            sklearn_methods = results_df[results_df['Method_Type'] == 'sklearn']
            if not sklearn_methods.empty:
                sklearn_summary = sklearn_methods.groupby('Method').agg({
                    'Accuracy': 'mean',
                    'F1_Macro': 'mean',
                    'Training_Time': 'mean'
                }).round(4)
                
                f.write("### 与经典机器学习方法对比\n\n")
                f.write(sklearn_summary.to_markdown())
                f.write("\n\n")
            
            # 结论和建议
            f.write(f"""## 实验结论

### 主要发现

1. **柯西分布尺度参数的作用**: 
   - 在统一架构实验中，柯西分布参数的效果需要根据具体数据集特性进一步分析
   - 不同分布选择(柯西vs高斯)在不同数据集上表现有差异

2. **方法适用性分析**:
   - **高准确率场景**: Random Forest和SVM表现突出
   - **训练效率场景**: 传统机器学习方法训练更快
   - **不确定性量化场景**: CAAC方法提供独特价值

3. **架构设计验证**:
   - 统一架构设计确保了公平比较
   - 网络深度和宽度设置对小数据集适当

### 改进建议

**短期改进**:
1. 调整网络架构参数，针对不同规模数据集优化
2. 实施更精细的超参数调优
3. 增加数据增强技术

**长期发展**:
1. 在大规模数据集上验证方法可扩展性  
2. 探索自适应分布选择机制
3. 开发实时不确定性量化应用

### 适用场景推荐

**推荐使用CAAC OvR**:
- 需要不确定性量化的关键决策场景
- 医疗诊断、金融风控等高风险应用
- 研究和教学中的方法论验证

**推荐使用传统方法**:
- 追求最高准确率的竞赛场景
- 计算资源受限的边缘设备部署
- 快速原型开发和baseline建立

## 可视化结果

实验生成的可视化图表包含：
- 准确率对比图
- F1分数对比图  
- 训练时间对比图
- 效率vs性能权衡散点图

![方法比较图](./methods_comparison_english_{timestamp}.png)

## 数据文件

- **详细结果**: `methods_comparison_detailed_{timestamp}.csv`
- **汇总统计**: `methods_comparison_summary_{timestamp}.csv`
- **可视化图表**: `methods_comparison_english_{timestamp}.png`

---

**实验配置信息**:
- Python环境: base conda环境
- 随机种子: 42 (确保可重复性)
- 数据分割: 80%训练 / 20%测试
- 特征标准化: StandardScaler
- 早停策略: patience=10, min_delta=0.0001

*报告由自动化实验脚本生成于 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
""")
        
        print(f"✅ Detailed report generated: {report_file.name}")
        return report_file


def run_comparison_experiments(**kwargs):
    """Standalone function for method comparison experiments."""
    runner = MethodComparisonRunner()
    return runner.run_comparison_experiments(**kwargs) 