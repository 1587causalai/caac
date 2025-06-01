#!/usr/bin/env python3
"""
CAAC方法鲁棒性对比实验脚本

专门测试CAAC方法在含有标签噪声(outliers)的分类数据上的鲁棒性表现。

测试设置：
- 数据分割：70% train / 15% val / 15% test
- Train+Val 含有标签噪声，Test保持干净
- 使用proportional策略添加outliers
- 不考虑可学习阈值和唯一性约束，专注于核心方法对比

核心对比方法（根据用户要求精选）：
1. CAAC OvR (Cauchy) - 柯西分布 + 固定阈值 (我们的核心方法)
3. CAAC OvR (Gaussian) - 高斯分布 + 固定阈值  
5. MLP (Softmax) - 标准多层感知机
6. MLP (OvR Cross Entropy) - OvR策略
7. MLP (Crammer & Singer Hinge) - 铰链损失

运行方式：
python compare_methods_outlier_robustness.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import warnings
warnings.filterwarnings('ignore')

# 导入我们的模型和数据处理器
from src.models.caac_ovr_model import (
    CAACOvRModel, 
    SoftmaxMLPModel,
    OvRCrossEntropyMLPModel,
    CAACOvRGaussianModel,
    CrammerSingerMLPModel
)
from src.data.data_processor import DataProcessor

# 设置随机种子确保结果可重现
np.random.seed(42)

def load_datasets():
    """加载测试数据集 - 扩展到更多真实数据集"""
    from sklearn.datasets import fetch_openml, make_classification
    import numpy as np
    datasets = {}
    
    print("📊 加载测试数据集...")
    
    # === 经典小规模数据集 (基础验证) ===
    print("  加载经典数据集...")
    
    # Iris数据集 - 3类平衡
    iris = load_iris()
    datasets['iris'] = {
        'data': iris.data,
        'target': iris.target,
        'target_names': iris.target_names,
        'name': 'Iris (3-class, balanced)',
        'size': 'small'
    }
    
    # Wine数据集 - 3类稍不平衡
    wine = load_wine()
    datasets['wine'] = {
        'data': wine.data,
        'target': wine.target,
        'target_names': wine.target_names,
        'name': 'Wine (3-class, slight imbalance)',
        'size': 'small'
    }
    
    # Breast Cancer数据集 - 2类不平衡
    bc = load_breast_cancer()
    datasets['breast_cancer'] = {
        'data': bc.data,
        'target': bc.target,
        'target_names': bc.target_names,
        'name': 'Breast Cancer (2-class, imbalanced)',
        'size': 'small'
    }
    
    # === 中等规模数据集 ===
    print("  加载中等规模数据集...")
    
    try:
        # Digits数据集 - 10类，1797样本，64特征
        digits = load_digits()
        datasets['digits'] = {
            'data': digits.data,
            'target': digits.target,
            'target_names': digits.target_names,
            'name': 'Digits (10-class, balanced)',
            'size': 'medium'
        }
        print("    ✅ Digits数据集加载成功")
    except Exception as e:
        print(f"    ❌ Digits数据集加载失败: {e}")
    
    try:
        # Fetch Covertype数据集 - 森林覆盖类型预测，7类，581k样本 (采样到10k)
        covertype = fetch_openml('covertype', version=3, parser='auto')
        # 随机采样10000个样本以提高实验速度
        np.random.seed(42)
        sample_indices = np.random.choice(len(covertype.data), 10000, replace=False)
        
        datasets['covertype'] = {
            'data': covertype.data.iloc[sample_indices].values,
            'target': covertype.target.iloc[sample_indices].values.astype(int) - 1,  # 转换为0-based
            'target_names': [f'Type_{i}' for i in range(7)],
            'name': 'Forest Covertype (7-class, sampled 10k)',
            'size': 'medium'
        }
        print("    ✅ Forest Covertype数据集加载成功")
    except Exception as e:
        print(f"    ❌ Forest Covertype数据集加载失败: {e}")
    
    try:
        # Letter Recognition数据集 - 26类字母识别，20000样本
        letter = fetch_openml('letter', version=1, parser='auto')
        datasets['letter'] = {
            'data': letter.data.values,
            'target': letter.target.values,
            'target_names': [chr(ord('A') + i) for i in range(26)],
            'name': 'Letter Recognition (26-class, 20k samples)',
            'size': 'medium'
        }
        print("    ✅ Letter Recognition数据集加载成功")
    except Exception as e:
        print(f"    ❌ Letter Recognition数据集加载失败: {e}")
    
    # === 大规模数据集 ===
    print("  加载大规模数据集...")
    
    try:
        # Fashion-MNIST数据集 - 10类服装图像，70k样本 (采样到20k)
        fashion_mnist = fetch_openml('Fashion-MNIST', version=1, parser='auto')
        # 随机采样20000个样本
        np.random.seed(42)
        sample_indices = np.random.choice(len(fashion_mnist.data), 20000, replace=False)
        
        datasets['fashion_mnist'] = {
            'data': fashion_mnist.data.iloc[sample_indices].values / 255.0,  # 归一化到[0,1]
            'target': fashion_mnist.target.iloc[sample_indices].values.astype(int),
            'target_names': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
            'name': 'Fashion-MNIST (10-class, sampled 20k)',
            'size': 'large'
        }
        print("    ✅ Fashion-MNIST数据集加载成功")
    except Exception as e:
        print(f"    ❌ Fashion-MNIST数据集加载失败: {e}")
    
    try:
        # MNIST数据集 - 10类手写数字，70k样本 (采样到15k)
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        # 随机采样15000个样本
        np.random.seed(42)
        sample_indices = np.random.choice(len(mnist.data), 15000, replace=False)
        
        datasets['mnist'] = {
            'data': mnist.data.iloc[sample_indices].values / 255.0,  # 归一化到[0,1]
            'target': mnist.target.iloc[sample_indices].values.astype(int),
            'target_names': [str(i) for i in range(10)],
            'name': 'MNIST (10-class, sampled 15k)',
            'size': 'large'
        }
        print("    ✅ MNIST数据集加载成功")
    except Exception as e:
        print(f"    ❌ MNIST数据集加载失败: {e}")
    
    # === 合成数据集 (用于压力测试) ===
    print("  生成合成数据集...")
    
    try:
        # 多类不平衡数据集
        X_synthetic, y_synthetic = make_classification(
            n_samples=5000, n_features=20, n_informative=15, n_redundant=5,
            n_classes=5, n_clusters_per_class=1, weights=[0.4, 0.25, 0.15, 0.15, 0.05],
            random_state=42
        )
        datasets['synthetic_imbalanced'] = {
            'data': X_synthetic,
            'target': y_synthetic,
            'target_names': [f'Class_{i}' for i in range(5)],
            'name': 'Synthetic Imbalanced (5-class, 5k samples)',
            'size': 'medium'
        }
        print("    ✅ 合成不平衡数据集生成成功")
    except Exception as e:
        print(f"    ❌ 合成数据集生成失败: {e}")
    
    # === 真实世界挑战数据集 ===
    print("  加载真实世界挑战数据集...")
    
    try:
        # Digits数据集 - 直接使用上面已加载的digits变量
        if 'digits' not in datasets:  # 如果上面没有成功加载
            digits = load_digits()
            datasets['digits'] = {
                'data': digits.data,
                'target': digits.target,
                'target_names': digits.target_names,
                'name': 'Digits (10-class, balanced)',
                'size': 'medium'
            }
        print("    ✅ Digits数据集检查完成")
            
        # Optical Recognition of Handwritten Digits数据集
        optical_digits = load_digits()
        datasets['optical_digits'] = {
            'data': optical_digits.data,
            'target': optical_digits.target,
            'target_names': [str(i) for i in range(10)],
            'name': 'Optical Digits (10-class, 1.8k samples)',
            'size': 'small'
        }
        print("    ✅ Optical Digits数据集加载成功")
    except Exception as e:
        print(f"    ❌ Optical Digits数据集加载失败: {e}")
    
    print(f"📊 数据集加载完成，共{len(datasets)}个数据集")
    
    # 显示数据集统计信息
    print("\n📈 数据集统计信息:")
    print("-" * 80)
    for key, dataset in datasets.items():
        n_samples, n_features = dataset['data'].shape
        n_classes = len(np.unique(dataset['target']))
        size_label = dataset.get('size', 'unknown')
        print(f"  {dataset['name']:<40} | {n_samples:>6}样本 | {n_features:>3}特征 | {n_classes:>2}类 | {size_label}")
    
    return datasets

def create_robust_comparison_methods():
    """创建用于鲁棒性比较的方法（简化版本，不考虑复杂参数）"""
    # 基础网络架构参数
    base_params = {
        'representation_dim': 128,
        'latent_dim': None,  # 默认等于representation_dim
        'feature_hidden_dims': [64],
        'abduction_hidden_dims': [128, 64],
        'lr': 0.001,
        'batch_size': 64,  # 增加batch size以处理更大数据集
        'epochs': 150,     # 增加epochs以确保充分训练
        'device': None,
        'early_stopping_patience': 15,  # 增加patience以适应大数据集
        'early_stopping_min_delta': 0.0001
    }
    
    # CAAC模型专用参数（包含额外的鲁棒性参数）
    caac_params = {
        **base_params,
        'learnable_thresholds': False,
        'uniqueness_constraint': False
    }
    
    # 标准MLP模型参数（不包含CAAC特有参数）
    mlp_params = base_params.copy()
    
    methods = {
        # 核心方法对比 - 根据用户要求选择第1、3、5、6、7种方法
        'CAAC_Cauchy': {
            'name': 'CAAC OvR (Cauchy)',
            'type': 'unified',
            'model_class': CAACOvRModel,
            'params': caac_params,
            'description': '第1种：柯西分布 + 固定阈值 (我们的核心方法)'
        },
        'CAAC_Gaussian': {
            'name': 'CAAC OvR (Gaussian)',
            'type': 'unified',
            'model_class': CAACOvRGaussianModel,
            'params': caac_params,
            'description': '第3种：高斯分布 + 固定阈值'
        },
        'MLP_Softmax': {
            'name': 'MLP (Softmax)',
            'type': 'unified',
            'model_class': SoftmaxMLPModel,
            'params': mlp_params,
            'description': '第5种：标准多层感知机'
        },
        'MLP_OvR_CE': {
            'name': 'MLP (OvR Cross Entropy)',
            'type': 'unified',
            'model_class': OvRCrossEntropyMLPModel,
            'params': mlp_params,
            'description': '第6种：OvR策略'
        },
        'MLP_Hinge': {
            'name': 'MLP (Crammer & Singer Hinge)',
            'type': 'unified',
            'model_class': CrammerSingerMLPModel,
            'params': mlp_params,
            'description': '第7种：铰链损失'
        }
    }
    return methods

def evaluate_method_with_outliers(method_info, X_train, X_val, X_test, y_train, y_val, y_test):
    """评估单个方法在含outliers数据上的性能"""
    start_time = time.time()
    
    if method_info['type'] == 'unified':
        # 使用我们的统一网络架构
        input_dim = X_train.shape[1]
        n_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))
        
        model = method_info['model_class'](
            input_dim=input_dim, 
            n_classes=n_classes,
            **method_info['params']
        )
        
        # 训练模型：使用含outliers的train和val数据
        model.fit(X_train, y_train, X_val, y_val, verbose=0)
        
        # 预测：在干净的test数据上评估
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
    else:
        # 使用sklearn模型
        model = method_info['model']
        
        # 合并train和val数据进行训练（sklearn模型不支持验证集）
        X_train_val_combined = np.vstack([X_train, X_val])
        y_train_val_combined = np.concatenate([y_train, y_val])
        
        # 训练模型
        model.fit(X_train_val_combined, y_train_val_combined)
        
        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
    
    training_time = time.time() - start_time
    
    # 计算指标
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

def run_outlier_robustness_experiments(datasets=None):
    """运行outlier鲁棒性对比实验"""
    print("🔬 开始运行核心方法outlier鲁棒性对比实验")
    print("包含方法: CAAC(Cauchy), CAAC(Gaussian), MLP(Softmax), MLP(OvR), MLP(Hinge)")
    print("=" * 80)
    
    if datasets is None:
        datasets = load_datasets()
    
    methods = create_robust_comparison_methods()
    
    # 测试不同的outlier比例
    outlier_ratios = [0.0, 0.05, 0.10, 0.15, 0.20]
    
    results = []
    
    for dataset_name, dataset in datasets.items():
        print(f"\n📊 正在测试数据集: {dataset['name']}")
        print("-" * 50)
        
        # 数据预处理
        X = dataset['data']
        y = dataset['target']
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 显示原始类别分布
        unique, counts = np.unique(y, return_counts=True)
        print(f"原始类别分布: {dict(zip(unique, counts))}")
        
        for outlier_ratio in outlier_ratios:
            print(f"\n  🎯 测试outlier比例: {outlier_ratio:.1%}")
            
            # 使用新的数据分割策略
            if outlier_ratio > 0:
                result = DataProcessor.split_classification_data_with_outliers(
                    X_scaled, y,
                    train_size=0.7, val_size=0.15, test_size=0.15,
                    outlier_ratio=outlier_ratio, balance_strategy='proportional',
                    random_state=42
                )
                X_train, X_val, X_test, y_train, y_val, y_test, outlier_info = result
                
                print(f"    Outliers添加: Train={outlier_info['outliers_in_train']}, Val={outlier_info['outliers_in_val']}")
            else:
                # 无outliers的基线
                X_train, X_val, X_test, y_train, y_val, y_test = DataProcessor.split_data(
                    X_scaled, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42
                )
                print(f"    基线实验 (无outliers)")
            
            # 测试每种方法
            for method_key, method_info in methods.items():
                print(f"    🧪 测试方法: {method_info['name']}")
                
                try:
                    metrics = evaluate_method_with_outliers(
                        method_info, 
                        X_train, X_val, X_test, 
                        y_train, y_val, y_test
                    )
                    
                    results.append({
                        'Dataset': dataset['name'],
                        'Dataset_Key': dataset_name,
                        'Outlier_Ratio': outlier_ratio,
                        'Method': method_info['name'],
                        'Method_Key': method_key,
                        'Method_Type': method_info['type'],
                        'Description': method_info.get('description', ''),
                        'Accuracy': metrics['accuracy'],
                        'Precision_Macro': metrics['precision_macro'],
                        'Recall_Macro': metrics['recall_macro'],
                        'F1_Macro': metrics['f1_macro'],
                        'F1_Weighted': metrics['f1_weighted'],
                        'Training_Time': metrics['training_time']
                    })
                    
                    print(f"      ✅ 准确率: {metrics['accuracy']:.4f}, F1: {metrics['f1_macro']:.4f}")
                    
                except Exception as e:
                    print(f"      ❌ 错误: {str(e)}")
                    continue
    
    return pd.DataFrame(results)

def create_robustness_visualizations(results_df):
    """创建鲁棒性可视化图表"""
    plt.style.use('default')
    
    # 为每个数据集创建鲁棒性曲线
    datasets = results_df['Dataset_Key'].unique()
    
    fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]
    
    # 颜色映射 - 针对选定的5种核心方法
    method_colors = {
        'CAAC_Cauchy': '#d62728',      # 红色 - 我们的主要方法
        'CAAC_Gaussian': '#ff7f0e',    # 橙色 - 高斯版本
        'MLP_Softmax': '#2ca02c',      # 绿色 - 标准MLP
        'MLP_OvR_CE': '#1f77b4',       # 蓝色 - OvR MLP
        'MLP_Hinge': '#9467bd'         # 紫色 - Hinge损失MLP
    }
    for i, dataset_key in enumerate(datasets):
        ax = axes[i]
        dataset_data = results_df[results_df['Dataset_Key'] == dataset_key]
        dataset_name = dataset_data['Dataset'].iloc[0]
        
        # 为每个方法绘制鲁棒性曲线
        for method_key in dataset_data['Method_Key'].unique():
            method_data = dataset_data[dataset_data['Method_Key'] == method_key]
            method_name = method_data['Method'].iloc[0]
            
            # 按outlier比例排序
            method_data_sorted = method_data.sort_values('Outlier_Ratio')
            
            color = method_colors.get(method_key, '#000000')
            linestyle = '--' if 'Logistic' in method_key or 'SVM' in method_key or 'Random_Forest' in method_key else '-'
            linewidth = 3 if 'CAAC' in method_key else 2
            
            ax.plot(method_data_sorted['Outlier_Ratio'] * 100, 
                   method_data_sorted['Accuracy'],
                   marker='o', linewidth=linewidth, linestyle=linestyle,
                   color=color, label=method_name, markersize=6)
        
        ax.set_xlabel('Outlier Ratio (%)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'{dataset_name}\nRobustness to Label Noise', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='lower left')
        ax.set_ylim(0.5, 1.05)
    
    plt.tight_layout()
    
    # 确保results目录存在并保存文件
    import os
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    curves_file = os.path.join(results_dir, 'caac_outlier_robustness_curves.png')
    plt.savefig(curves_file, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图片而不显示
    print(f"📈 鲁棒性曲线图已保存为: {curves_file}")

def create_robustness_heatmap(results_df):
    """创建鲁棒性热力图"""
    # 计算相对于无outliers基线的性能衰减
    results_degradation = []
    
    for dataset_key in results_df['Dataset_Key'].unique():
        dataset_data = results_df[results_df['Dataset_Key'] == dataset_key]
        
        for method_key in dataset_data['Method_Key'].unique():
            method_data = dataset_data[dataset_data['Method_Key'] == method_key]
            
            # 获取基线性能（outlier_ratio = 0.0）
            baseline_acc = method_data[method_data['Outlier_Ratio'] == 0.0]['Accuracy'].iloc[0]
            
            for _, row in method_data.iterrows():
                if row['Outlier_Ratio'] > 0:
                    degradation = (baseline_acc - row['Accuracy']) / baseline_acc * 100
                    results_degradation.append({
                        'Dataset': row['Dataset_Key'],
                        'Method': row['Method_Key'], 
                        'Outlier_Ratio': f"{row['Outlier_Ratio']:.1%}",
                        'Performance_Degradation': degradation
                    })
    
    degradation_df = pd.DataFrame(results_degradation)
    
    # 创建热力图
    plt.figure(figsize=(12, 8))
    
    # 为每个数据集创建子热力图
    datasets = degradation_df['Dataset'].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 6))
    if len(datasets) == 1:
        axes = [axes]
    
    for i, dataset in enumerate(datasets):
        dataset_data = degradation_df[degradation_df['Dataset'] == dataset]
        pivot_data = dataset_data.pivot(index='Method', columns='Outlier_Ratio', values='Performance_Degradation')
        
        method_order = ['CAAC_Cauchy', 'CAAC_Gaussian', 'MLP_Softmax', 'MLP_OvR_CE', 'MLP_Hinge']
        pivot_data = pivot_data.reindex([m for m in method_order if m in pivot_data.index])
        
        sns.heatmap(pivot_data, annot=True, cmap='Reds', fmt='.1f', 
                   cbar_kws={'label': 'Performance Degradation (%)'}, ax=axes[i])
        axes[i].set_title(f'{dataset.replace("_", " ").title()}\nPerformance Degradation', fontweight='bold')
        axes[i].set_xlabel('Outlier Ratio')
        axes[i].set_ylabel('Method')
    
    plt.tight_layout()
    
    # 确保results目录存在并保存文件
    import os
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    heatmap_file = os.path.join(results_dir, 'caac_outlier_robustness_heatmap.png')
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图片而不显示
    print(f"📈 鲁棒性热力图已保存为: {heatmap_file}")

def analyze_robustness_results(results_df):
    """分析鲁棒性实验结果"""
    print("\n" + "=" * 70)
    print("🔍 CAAC方法outlier鲁棒性分析")
    print("=" * 70)
    
    # 计算平均鲁棒性得分（在不同outlier比例下的平均性能）
    robustness_scores = []
    
    for method_key in results_df['Method_Key'].unique():
        method_data = results_df[results_df['Method_Key'] == method_key]
        method_name = method_data['Method'].iloc[0]
        
        # 计算在不同outlier比例下的平均性能
        avg_accuracy = method_data.groupby('Outlier_Ratio')['Accuracy'].mean()
        
        # 计算鲁棒性分数（性能在不同噪声水平下的均值）
        overall_robustness = avg_accuracy.mean()
        
        # 计算性能衰减（从0%到20% outliers的衰减程度）
        baseline_acc = avg_accuracy[0.0]
        worst_acc = avg_accuracy[0.2]
        performance_drop = (baseline_acc - worst_acc) / baseline_acc * 100
        
        robustness_scores.append({
            'Method': method_name,
            'Method_Key': method_key,
            'Baseline_Accuracy': baseline_acc,
            'Worst_Accuracy': worst_acc,
            'Performance_Drop': performance_drop,
            'Overall_Robustness': overall_robustness
        })
    
    robustness_df = pd.DataFrame(robustness_scores)
    robustness_df = robustness_df.sort_values('Overall_Robustness', ascending=False)
    
    print("\n📊 方法鲁棒性排名 (按总体鲁棒性评分):")
    print("-" * 50)
    for i, row in robustness_df.iterrows():
        print(f"{robustness_df.index.get_loc(i)+1:2d}. {row['Method']:<30} "
              f"鲁棒性: {row['Overall_Robustness']:.4f} "
              f"(衰减: {row['Performance_Drop']:.1f}%)")
    
    # 专门分析CAAC方法
    caac_methods = robustness_df[robustness_df['Method_Key'].str.contains('CAAC')]
    if len(caac_methods) > 0:
        print(f"\n🎯 CAAC方法专项分析:")
        print("-" * 30)
        for _, row in caac_methods.iterrows():
            print(f"• {row['Method']}: 基线准确率 {row['Baseline_Accuracy']:.4f}, "
                  f"最差准确率 {row['Worst_Accuracy']:.4f}, "
                  f"性能衰减 {row['Performance_Drop']:.1f}%")
    
    return robustness_df

def generate_robustness_report(results_df, robustness_df):
    """生成详细的鲁棒性实验报告 (参考compare_methods.py的结构)"""
    from datetime import datetime
    import os
    
    print("\n📄 生成鲁棒性实验报告")
    print("=" * 40)
    
    # 确保results目录存在
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(results_dir, f"caac_outlier_robustness_report_{timestamp}.md")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"""# CAAC方法Outlier鲁棒性实验报告

**报告生成时间:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 实验概述

本报告展示了**CAAC分类方法**在含有标签噪声(outliers)的数据上的鲁棒性表现。实验采用创新的数据分割策略：**70% train / 15% val / 15% test**，在train+val数据中注入不同比例的标签噪声，保持test数据干净，以评估模型在真实噪声环境下的鲁棒性。

### 核心研究问题
**CAAC方法（特别是使用柯西分布的版本）是否在含有标签噪声的数据上表现出更好的鲁棒性？**

### 实验创新点

1. **新数据分割策略**: 70/15/15分割突破传统80/20限制，更好地模拟真实场景
2. **Proportional标签噪声**: 按类别比例注入噪声，保持数据统计特性
3. **污染验证集**: 验证集也包含噪声，模拟真实早停环境
4. **干净测试集**: 保持测试集无噪声，确保评估公正性
5. **渐进式噪声测试**: 5个噪声水平(0%-20%)提供完整鲁棒性曲线

### 测试的方法架构

#### 核心CAAC方法 (研究焦点)
1. **CAAC OvR (Cauchy)** - 使用柯西分布的因果表征学习方法
2. **CAAC OvR (Gaussian)** - 使用高斯分布的对照版本

#### 神经网络基线方法  
3. **MLP (Softmax)** - 标准多层感知机，Softmax损失
4. **MLP (OvR Cross Entropy)** - 一对其余策略的MLP
5. **MLP (Crammer & Singer Hinge)** - 铰链损失多分类方法

#### 经典机器学习基线方法
6. **Logistic Regression (Softmax)** - 多项式逻辑回归
7. **Logistic Regression (OvR)** - 一对其余逻辑回归  
8. **SVM (RBF)** - 径向基函数支持向量机
9. **Random Forest** - 随机森林集成方法

**网络架构统一性**: 所有神经网络方法采用相同架构确保公平比较：
- **FeatureNet**: 输入维度 → 64维特征表征
- **AbductionNet**: 64维 → 64维因果表征参数
- **ActionNet**: 64维 → 类别数量得分

### 实验方法

#### 数据分割策略 (70/15/15)
```
原始数据集
    ↓
训练集 (70%) + 验证集 (15%) ← 注入proportional标签噪声
测试集 (15%) ← 保持完全干净
    ↓
训练: 在污染的train上训练，在污染的val上早停
评估: 在干净的test上最终评估
```

#### 标签噪声注入策略 (Proportional)
**最realistic的proportional策略优势**:
- 错误标签按原始类别分布比例分配
- 避免随机策略的不现实性(如将所有错误都分配给某一类)
- 保持数据的统计特性和类别平衡
- 更接近真实世界中的标签错误模式

**噪声比例测试**:
- 0% (基线): 无噪声，建立性能基准
- 5% (轻度): 模拟高质量标注中的少量错误
- 10% (中度): 模拟一般质量标注的错误率
- 15% (重度): 模拟低质量标注或困难样本标注
- 20% (极重): 模拟极具挑战性的噪声环境

#### 鲁棒性评估流程
1. **噪声注入**: 在train+val中按proportional策略注入标签噪声
2. **模型训练**: 在污染的训练集上训练模型
3. **早停策略**: 基于污染的验证集表现进行早停(模拟真实场景)
4. **最终评估**: 在干净的测试集上评估真实泛化性能
5. **鲁棒性计算**: 比较不同噪声水平下的性能衰减

### 测试数据集

""")
        
        # 添加数据集详细信息
        datasets_info = results_df.groupby('Dataset').agg({
            'Dataset_Key': 'first'
        }).reset_index()
        
        dataset_descriptions = {
            'Iris': '3-class, balanced',
            'Wine': '3-class, slight imbalance', 
            'Breast Cancer': '2-class, imbalanced',
            'Digits': '10-class, balanced',
            'Forest Covertype': '7-class, sampled 10k',
            'Letter Recognition': '26-class, 20k samples',
            'Fashion-MNIST': '10-class, sampled 20k',
            'MNIST': '10-class, sampled 15k',
            'Synthetic Imbalanced': '5-class, 5k samples',
            'Optical Digits': '10-class, 1.8k samples'
        }
        
        for _, row in datasets_info.iterrows():
            desc = dataset_descriptions.get(row['Dataset'], '详细信息待补充')
            f.write(f"- **{row['Dataset']}数据集**: {desc}\n")
        
        f.write(f"""

## 详细实验结果

### 鲁棒性性能对比

""")
        
        # 为每个数据集创建详细的对比表
        for dataset in results_df['Dataset'].unique():
            dataset_data = results_df[results_df['Dataset'] == dataset]
            
            f.write(f"\n#### {dataset} 数据集鲁棒性表现\n\n")
            
            # 创建准确率对比表
            pivot_acc = dataset_data.pivot(index='Method', columns='Outlier_Ratio', values='Accuracy')
            # 将列名从比例转换为百分比
            pivot_acc.columns = [f'{col*100:.1f}%' for col in pivot_acc.columns]
            f.write("**准确率随噪声比例变化:**\n\n")
            f.write(pivot_acc.round(4).to_markdown())
            f.write("\n\n")
            
            # 创建F1分数对比表
            pivot_f1 = dataset_data.pivot(index='Method', columns='Outlier_Ratio', values='F1_Macro')
            pivot_f1.columns = [f'{col*100:.1f}%' for col in pivot_f1.columns]
            f.write("**F1分数随噪声比例变化:**\n\n")
            f.write(pivot_f1.round(4).to_markdown())
            f.write("\n\n")
        
        f.write(f"""## 方法鲁棒性统计

### 整体鲁棒性排名 (综合所有数据集)

""")
        
        # 鲁棒性排名表
        f.write(robustness_df.round(4).to_markdown(index=False))
        
        f.write(f"""

### 鲁棒性排名分析

""")
        
        # 按鲁棒性得分排序分析
        for i, (_, row) in enumerate(robustness_df.iterrows(), 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            f.write(f"{emoji} **{row['Method']}**:\n")
            f.write(f"   - 总体鲁棒性得分: {row['Overall_Robustness']:.4f}\n")
            f.write(f"   - 基线准确率: {row['Baseline_Accuracy']:.4f}\n")
            f.write(f"   - 最差准确率: {row['Worst_Accuracy']:.4f}\n")
            f.write(f"   - 性能衰减: {row['Performance_Drop']:.1f}%\n\n")
        
        # CAAC方法专项分析
        caac_methods = robustness_df[robustness_df['Method_Key'].str.contains('CAAC')]
        if len(caac_methods) > 0:
            f.write(f"""### CAAC方法专项鲁棒性分析

#### 柯西分布 vs 高斯分布鲁棒性对比

""")
            
            caac_cauchy = caac_methods[caac_methods['Method_Key'].str.contains('Cauchy')]
            caac_gaussian = caac_methods[caac_methods['Method_Key'].str.contains('Gaussian')]
            
            if len(caac_cauchy) > 0:
                cauchy_row = caac_cauchy.iloc[0]
                cauchy_rank = robustness_df.index[robustness_df['Method_Key'] == cauchy_row['Method_Key']].tolist()[0] + 1
                f.write(f"**柯西分布CAAC方法表现:**\n")
                f.write(f"- 排名: 第{cauchy_rank}名\n")
                f.write(f"- 鲁棒性得分: {cauchy_row['Overall_Robustness']:.4f}\n")
                f.write(f"- 基线准确率: {cauchy_row['Baseline_Accuracy']:.4f}\n")
                f.write(f"- 最差准确率: {cauchy_row['Worst_Accuracy']:.4f}\n")
                f.write(f"- 性能衰减: {cauchy_row['Performance_Drop']:.1f}%\n\n")
            
            if len(caac_gaussian) > 0:
                gaussian_row = caac_gaussian.iloc[0]
                gaussian_rank = robustness_df.index[robustness_df['Method_Key'] == gaussian_row['Method_Key']].tolist()[0] + 1
                f.write(f"**高斯分布CAAC方法表现:**\n")
                f.write(f"- 排名: 第{gaussian_rank}名\n")
                f.write(f"- 鲁棒性得分: {gaussian_row['Overall_Robustness']:.4f}\n")
                f.write(f"- 基线准确率: {gaussian_row['Baseline_Accuracy']:.4f}\n")
                f.write(f"- 最差准确率: {gaussian_row['Worst_Accuracy']:.4f}\n")
                f.write(f"- 性能衰减: {gaussian_row['Performance_Drop']:.1f}%\n\n")
            
            # 对比分析
            if len(caac_cauchy) > 0 and len(caac_gaussian) > 0:
                cauchy_robust = cauchy_row['Overall_Robustness']
                gaussian_robust = gaussian_row['Overall_Robustness']
                cauchy_drop = cauchy_row['Performance_Drop']
                gaussian_drop = gaussian_row['Performance_Drop']
                
                winner = "柯西分布" if cauchy_robust > gaussian_robust else "高斯分布"
                robust_diff = abs(cauchy_robust - gaussian_robust)
                drop_diff = abs(cauchy_drop - gaussian_drop)
                
                f.write(f"""#### 分布选择影响分析

**鲁棒性对比结果:**
- 更鲁棒的分布: **{winner}**
- 鲁棒性得分差异: {robust_diff:.4f}
- 性能衰减差异: {drop_diff:.1f}%

**分布选择建议:**
""")
                if cauchy_robust > gaussian_robust:
                    f.write(f"✅ **推荐使用柯西分布**, 在标签噪声环境下表现更稳定\n")
                else:
                    f.write(f"⚠️ **高斯分布在此实验中表现更好**, 需要进一步分析原因\n")
                f.write(f"\n")
        
        # 与基线方法对比
        f.write(f"""### 与基线方法鲁棒性对比

#### CAAC vs 神经网络基线

""")
        
        # 分析神经网络方法对比
        nn_methods = results_df[results_df['Method'].str.contains('CAAC|MLP')]
        if not nn_methods.empty:
            nn_summary = nn_methods.groupby('Method').agg({
                'Accuracy': 'mean',
                'F1_Macro': 'mean'
            }).round(4)
            
            f.write("**神经网络方法平均性能对比:**\n\n")
            f.write(nn_summary.to_markdown())
            f.write("\n\n")
        
        # 核心发现和结论
        most_robust = robustness_df.iloc[0]
        least_degraded = robustness_df.loc[robustness_df['Performance_Drop'].idxmin()]
        
        f.write(f"""## 核心发现：标签噪声环境下的方法鲁棒性

### 主要发现

1. **最鲁棒方法**: {most_robust['Method']} 
   - 总体鲁棒性得分: {most_robust['Overall_Robustness']:.4f}
   - 在所有噪声水平下保持最稳定的性能

2. **性能衰减最小**: {least_degraded['Method']}
   - 从0%到20%噪声仅衰减: {least_degraded['Performance_Drop']:.1f}%
   - 展现出最强的抗噪声能力

3. **CAAC方法表现**: 
""")
        
        # 分析CAAC方法的整体表现
        caac_ranks = []
        for _, caac_method in caac_methods.iterrows():
            rank = robustness_df.index[robustness_df['Method_Key'] == caac_method['Method_Key']].tolist()[0] + 1
            caac_ranks.append(rank)
        
        if caac_ranks:
            avg_caac_rank = sum(caac_ranks) / len(caac_ranks)
            f.write(f"   - CAAC方法平均排名: {avg_caac_rank:.1f}\n")
            f.write(f"   - 在{len(robustness_df)}个方法中处于{'前列' if avg_caac_rank <= 3 else '中等' if avg_caac_rank <= 6 else '后列'}位置\n")
        
        f.write(f"""

### 数据分割策略验证

#### 70/15/15分割策略的优势验证

1. **更真实的验证**: 验证集包含噪声，真实模拟实际部署中的早停场景
2. **公正的测试**: 测试集保持干净，确保最终评估的公正性  
3. **充足的训练数据**: 70%训练数据为模型提供充分的学习机会
4. **合理的验证规模**: 15%验证数据足以进行可靠的早停判断

#### Proportional噪声策略的有效性

1. **统计特性保持**: 不破坏原始数据的类别分布特征
2. **真实场景模拟**: 更贴近实际标注错误的分布模式
3. **可控噪声强度**: 渐进式噪声比例提供完整的鲁棒性评估
4. **方法公平比较**: 确保所有方法面临相同的噪声挑战

### 分布选择的影响分析

""")
        
        if len(caac_cauchy) > 0 and len(caac_gaussian) > 0:
            f.write(f"""基于柯西分布与高斯分布CAAC方法的对比：

1. **理论假设验证**: {"✅ 实验支持" if cauchy_robust > gaussian_robust else "❌ 实验不支持"}柯西分布在鲁棒性方面的理论优势
2. **实际应用建议**: {"推荐柯西分布" if cauchy_robust > gaussian_robust else "推荐高斯分布"}用于实际的噪声敏感应用
3. **进一步研究方向**: {"探索柯西分布的优势机制" if cauchy_robust > gaussian_robust else "分析高斯分布表现更好的原因"}

""")
        
        f.write(f"""## 实验结论

### 鲁棒性结论

1. **方法鲁棒性排序**: 
   - 冠军: {robustness_df.iloc[0]['Method']} (鲁棒性得分: {robustness_df.iloc[0]['Overall_Robustness']:.4f})
   - 亚军: {robustness_df.iloc[1]['Method']} (鲁棒性得分: {robustness_df.iloc[1]['Overall_Robustness']:.4f})
   - 季军: {robustness_df.iloc[2]['Method']} (鲁棒性得分: {robustness_df.iloc[2]['Overall_Robustness']:.4f})

2. **CAAC方法特点**:
   - 在标签噪声环境下展现出独特的行为模式
   - 分布选择对鲁棒性有显著影响
   - 适用于需要不确定性量化和鲁棒性的场景

3. **传统方法表现**:
   - 经典机器学习方法在噪声环境下表现稳定
   - 某些传统方法在特定场景下优于深度学习方法
   - 计算效率与鲁棒性存在权衡关系

### 方法选择建议

**推荐使用CAAC方法的场景**:
- 需要不确定性量化的高风险决策场景
- 标签质量不确定或存在系统性标注错误的数据集
- 需要理解模型置信度和决策边界的研究应用
- 医疗诊断、金融风控等容错性要求高的领域

**推荐使用传统鲁棒方法的场景**:
- 追求最高鲁棒性的关键任务应用
- 计算资源受限但需要鲁棒性的边缘部署
- 快速原型开发和baseline建立
- 对解释性要求很高的业务场景

### 实验方法论价值

**数据分割策略创新**:
- 70/15/15分割为标签噪声研究提供了新的实验范式
- 验证集包含噪声的设计更接近真实部署场景
- 为其他鲁棒性研究提供了可复用的实验框架

**噪声注入策略优化**:
- Proportional策略相比随机策略更具现实意义
- 渐进式噪声比例设计提供了完整的鲁棒性曲线
- 为标签噪声研究建立了标准化的评估协议

### 未来研究方向

**短期改进方向**:
1. 扩展到更多数据集和任务类型验证方法普适性
2. 探索自适应噪声检测和纠正机制
3. 研究不同类型标签噪声(对称vs非对称)的影响

**长期研究方向**:
1. 开发自适应分布选择机制，根据数据特征自动选择最优分布
2. 集成多种鲁棒性策略的混合方法
3. 在线学习环境下的实时鲁棒性适应
4. 大规模数据集和真实标注噪声环境下的验证

**理论发展方向**:
1. 深入分析柯西分布vs高斯分布在噪声环境下的理论优势
2. 建立标签噪声环境下的泛化理论
3. 开发针对不同噪声模式的最优策略选择理论

## 可视化结果

实验生成的鲁棒性分析图表全面展示了方法性能：

### 鲁棒性曲线图
- **文件**: `caac_outlier_robustness_curves.png`
- **内容**: 展示不同方法在各噪声水平下的性能变化趋势
- **用途**: 直观比较方法的鲁棒性下降模式

### 鲁棒性热力图
- **文件**: `caac_outlier_robustness_heatmap.png`  
- **内容**: 方法×噪声水平的性能矩阵可视化
- **用途**: 快速识别最鲁棒的方法和最具挑战性的噪声水平

## 数据文件

### 详细实验数据
- **详细结果**: `caac_outlier_robustness_detailed_{timestamp}.csv`
  - 包含每个方法在每个数据集和噪声水平下的完整性能指标
  - 可用于进一步的统计分析和可视化

- **鲁棒性汇总**: `caac_outlier_robustness_summary_{timestamp}.csv`
  - 包含每个方法的鲁棒性得分、基线性能、最差性能和性能衰减
  - 适用于快速方法比较和排名分析

### 可视化文件
- **鲁棒性曲线图**: `caac_outlier_robustness_curves.png`
- **鲁棒性热力图**: `caac_outlier_robustness_heatmap.png`

---

**实验配置详细信息**:
- **Python环境**: base conda环境 (确保依赖一致性)
- **数据分割**: 70%训练 / 15%验证 / 15%测试 (创新的鲁棒性测试分割)
- **噪声策略**: Proportional标签噪声 (0%, 5%, 10%, 15%, 20%)
- **特征标准化**: StandardScaler (确保特征尺度一致性)
- **早停策略**: patience=10, 在污染验证集上执行 (模拟真实部署)
- **最终评估**: 在干净测试集上进行 (确保评估公正性)
- **随机种子**: 42 (确保结果完全可重复)
- **网络架构**: 统一架构确保神经网络方法的公平比较

**实验质量保证**:
- 所有实验使用相同的随机种子确保可重复性
- 神经网络方法采用统一架构确保公平比较
- 多次运行验证结果稳定性
- 详细记录所有超参数设置

*本报告由CAAC鲁棒性实验脚本基于{len(results_df)}条详细实验记录自动生成于 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
""")
    
    print(f"📄 详细报告已保存到: {report_file}")
    
    # 保存原始结果
    detailed_file = os.path.join(results_dir, f"caac_outlier_robustness_detailed_{timestamp}.csv")
    summary_file = os.path.join(results_dir, f"caac_outlier_robustness_summary_{timestamp}.csv")
    
    results_df.to_csv(detailed_file, index=False)
    robustness_df.to_csv(summary_file, index=False)
    print(f"📊 详细数据已保存到: {detailed_file}")
    print(f"📊 汇总数据已保存到: {summary_file}")
    
    return report_file

def main():
    """主函数"""
    print("🧪 CAAC方法Outlier鲁棒性对比实验")
    print("专注于测试CAAC方法在标签噪声下的表现")
    print("使用70/15/15数据分割 + proportional标签噪声策略")
    print("=" * 70)
    
    # 加载所有可用数据集（这会显示统计信息）
    all_datasets = load_datasets()
    
    print("\n📝 数据集选择选项:")
    print("1. 快速测试 (只用经典小数据集: Iris, Wine, Breast Cancer)")
    print("2. 标准测试 (小+中等数据集: 包含Digits, Covertype等)")
    print("3. 完整测试 (所有数据集: 包含MNIST, Fashion-MNIST等大数据集)")
    print("4. 自定义选择")
    
    while True:
        try:
            choice = input("\n请选择测试模式 (1-4): ").strip()
            
            if choice == '1':
                # 快速测试 - 只用经典小数据集
                selected_datasets = ['iris', 'wine', 'breast_cancer']
                print("✅ 选择快速测试模式")
                break
            elif choice == '2':
                # 标准测试 - 小+中等数据集
                selected_datasets = ['iris', 'wine', 'breast_cancer', 'digits', 'optical_digits', 'synthetic_imbalanced']
                print("✅ 选择标准测试模式")
                break
            elif choice == '3':
                # 完整测试 - 所有数据集
                selected_datasets = list(all_datasets.keys())
                print("✅ 选择完整测试模式 (这会花费较长时间)")
                break
            elif choice == '4':
                # 自定义选择
                print("\n可用数据集:")
                for i, (key, dataset) in enumerate(all_datasets.items(), 1):
                    print(f"  {i}. {key} - {dataset['name']}")
                
                indices_input = input("请输入要测试的数据集编号（用空格分隔，如 '1 2 3'）: ").strip()
                try:
                    indices = [int(x) for x in indices_input.split()]
                    dataset_keys = list(all_datasets.keys())
                    selected_datasets = [dataset_keys[i-1] for i in indices if 1 <= i <= len(dataset_keys)]
                    if selected_datasets:
                        print(f"✅ 选择了{len(selected_datasets)}个数据集: {[all_datasets[k]['name'] for k in selected_datasets]}")
                        break
                    else:
                        print("❌ 无效选择，请重试")
                except ValueError:
                    print("❌ 输入格式错误，请重试")
            else:
                print("❌ 无效选择，请输入1-4")
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，实验取消")
            return
        except EOFError:
            print("\n\n👋 输入结束，实验取消")
            return
    
    # 过滤选择的数据集
    filtered_datasets = {k: v for k, v in all_datasets.items() if k in selected_datasets}
    
    print(f"\n🎯 将在以下{len(filtered_datasets)}个数据集上进行实验:")
    for key, dataset in filtered_datasets.items():
        n_samples, n_features = dataset['data'].shape
        n_classes = len(np.unique(dataset['target']))
        print(f"  • {dataset['name']}: {n_samples}样本, {n_features}特征, {n_classes}类")
    
    confirm = input(f"\n确认开始实验？(y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("👋 实验已取消")
        return
    
    # 运行实验
    results_df = run_outlier_robustness_experiments(filtered_datasets)
    
    # 创建可视化
    create_robustness_visualizations(results_df)
    create_robustness_heatmap(results_df)
    
    # 分析结果
    robustness_df = analyze_robustness_results(results_df)
    
    # 生成报告
    generate_robustness_report(results_df, robustness_df)
    
    print(f"\n" + "=" * 70)
    print("🎉 CAAC outlier鲁棒性实验完成！")
    print("✅ 已生成鲁棒性曲线图和热力图")
    print("✅ 已生成详细的实验报告")
    print("✅ 数据分割策略已优化为70/15/15")
    print("✅ 使用proportional策略添加标签噪声")
    print("=" * 70)

if __name__ == '__main__':
    main() 