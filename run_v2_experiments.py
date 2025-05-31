"""
V2 Shared Cauchy OvR Classifier 实验脚本

测试基于共享潜在柯西向量的One-vs-Rest分类器在不同场景下的性能，
包括多分类任务、不确定性量化、大规模类别处理、噪声鲁棒性等。
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
from datetime import datetime
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
import time
import warnings
warnings.filterwarnings('ignore')

# 导入V2模型
from src.models import (
    SharedCauchyOvRClassifier,
    create_loss_function,
    SharedCauchyOvRTrainer
)


def generate_multiclass_dataset(
    n_samples: int = 2000,
    n_features: int = 20,
    n_classes: int = 10,
    n_informative: int = 15,
    noise_level: float = 0.1,
    outlier_ratio: float = 0.0,
    cluster_std: float = 1.0,
    random_state: int = 42
):
    """生成多分类数据集"""
    print(f"生成数据集: {n_samples}样本, {n_features}特征, {n_classes}类别")
    
    # 生成基础数据
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_informative,
        n_redundant=n_features - n_informative,
        n_clusters_per_class=1,
        class_sep=1.0,
        random_state=random_state
    )
    
    # 添加噪声
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, X.shape)
        X = X + noise
    
    # 添加异常值
    if outlier_ratio > 0:
        n_outliers = int(len(X) * outlier_ratio)
        outlier_indices = np.random.choice(len(X), n_outliers, replace=False)
        # 生成远离正常数据的异常值
        outlier_scale = 3.0
        X[outlier_indices] = np.random.normal(
            X.mean(axis=0), 
            X.std(axis=0) * outlier_scale, 
            (n_outliers, n_features)
        )
    
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, scaler


def create_data_loaders(X, y, test_size=0.2, val_size=0.1, batch_size=64):
    """创建数据加载器"""
    # 分割数据
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), 
        random_state=42, stratify=y_temp
    )
    
    # 转换为张量
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    y_test = torch.LongTensor(y_test)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'data': {
            'X_train': X_train.numpy(),
            'X_val': X_val.numpy(),
            'X_test': X_test.numpy(),
            'y_train': y_train.numpy(),
            'y_val': y_val.numpy(),
            'y_test': y_test.numpy()
        }
    }


def run_shared_cauchy_experiment(
    data_loaders: dict,
    n_features: int,
    n_classes: int,
    latent_dim: int = None,
    loss_type: str = 'ovr_bce',
    epochs: int = 50,
    learning_rate: float = 1e-3,
    hidden_dims: list = None,
    device: str = 'auto'
):
    """运行Shared Cauchy OvR实验"""
    
    if latent_dim is None:
        latent_dim = min(n_classes // 2, 20)  # 默认潜在维度
    
    if hidden_dims is None:
        hidden_dims = [128, 64]
    
    print(f"运行Shared Cauchy OvR实验:")
    print(f"  - 潜在维度: {latent_dim}")
    print(f"  - 损失函数: {loss_type}")
    print(f"  - 隐藏层: {hidden_dims}")
    
    # 创建模型
    model = SharedCauchyOvRClassifier(
        input_dim=n_features,
        num_classes=n_classes,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims
    )
    
    # 创建损失函数
    loss_function = create_loss_function(loss_type)
    
    # 创建优化器和调度器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # 创建训练器
    trainer = SharedCauchyOvRTrainer(
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )
    
    # 训练
    start_time = time.time()
    history = trainer.train(
        train_loader=data_loaders['train_loader'],
        val_loader=data_loaders['val_loader'],
        num_epochs=epochs,
        early_stopping_patience=10
    )
    train_time = time.time() - start_time
    
    # 评估
    results = trainer.evaluate(
        data_loaders['test_loader'],
        class_names=[f'Class_{i}' for i in range(n_classes)]
    )
    
    # 不确定性分析
    uncertainty_analysis = trainer.analyze_uncertainty(data_loaders['test_loader'])
    
    return {
        'model': model,
        'trainer': trainer,
        'history': history,
        'results': results,
        'uncertainty_analysis': uncertainty_analysis,
        'train_time': train_time,
        'config': {
            'latent_dim': latent_dim,
            'loss_type': loss_type,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'hidden_dims': hidden_dims
        }
    }


def run_baseline_experiments(data_loaders: dict, n_classes: int):
    """运行基线方法实验"""
    print("运行基线方法实验...")
    
    # 提取数据
    X_train = data_loaders['data']['X_train']
    X_test = data_loaders['data']['X_test']
    y_train = data_loaders['data']['y_train']
    y_test = data_loaders['data']['y_test']
    
    baselines = {
        'Softmax_LR': LogisticRegression(
            multi_class='multinomial', 
            solver='lbfgs', 
            max_iter=1000,
            random_state=42
        ),
        'OvR_LR': OneVsRestClassifier(
            LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, 
            random_state=42
        ),
        'SVM_OvR': OneVsRestClassifier(
            SVC(kernel='rbf', probability=True, random_state=42)
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
    }
    
    baseline_results = {}
    
    for name, model in baselines.items():
        print(f"  训练 {name}...")
        start_time = time.time()
        
        try:
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # 预测
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # 计算指标
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            baseline_results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1': f1,
                'train_time': train_time,
                'predictions': y_pred,
                'probabilities': y_proba
            }
            
            print(f"    {name}: 准确率={accuracy:.4f}, F1={f1:.4f}, 时间={train_time:.2f}s")
            
        except Exception as e:
            print(f"    {name} 训练失败: {str(e)}")
            baseline_results[name] = None
    
    return baseline_results


def analyze_scalability(
    base_config: dict,
    class_counts: list = [5, 10, 20, 50, 100],
    sample_counts: list = None
):
    """分析可扩展性"""
    print("\n" + "="*50)
    print("可扩展性分析")
    print("="*50)
    
    if sample_counts is None:
        sample_counts = [count * 100 for count in class_counts]  # 每类100样本
    
    scalability_results = []
    
    for n_classes, n_samples in zip(class_counts, sample_counts):
        print(f"\n测试 {n_classes} 类别, {n_samples} 样本...")
        
        # 生成数据
        X, y, _ = generate_multiclass_dataset(
            n_samples=n_samples,
            n_features=base_config['n_features'],
            n_classes=n_classes,
            random_state=42
        )
        
        data_loaders = create_data_loaders(X, y, batch_size=64)
        
        # 运行实验
        result = run_shared_cauchy_experiment(
            data_loaders=data_loaders,
            n_features=base_config['n_features'],
            n_classes=n_classes,
            epochs=30,  # 减少epochs以节省时间
            device='cpu'
        )
        
        scalability_results.append({
            'n_classes': n_classes,
            'n_samples': n_samples,
            'accuracy': result['results']['accuracy'],
            'f1': result['results']['classification_report']['weighted avg']['f1-score'],
            'train_time': result['train_time'],
            'model_params': sum(p.numel() for p in result['model'].parameters()),
            'avg_uncertainty': result['uncertainty_analysis']['uncertainties'].mean()
        })
        
        print(f"  准确率: {result['results']['accuracy']:.4f}")
        print(f"  训练时间: {result['train_time']:.2f}s")
        print(f"  模型参数: {sum(p.numel() for p in result['model'].parameters()):,}")
    
    return scalability_results


def analyze_loss_functions(data_loaders: dict, n_features: int, n_classes: int):
    """分析不同损失函数的效果"""
    print("\n" + "="*50)
    print("损失函数比较分析")
    print("="*50)
    
    loss_functions = [
        'ovr_bce',
        'weighted_ovr_bce',
        'focal_ovr',
        'uncertainty_reg'
    ]
    
    loss_results = []
    
    for loss_type in loss_functions:
        print(f"\n测试损失函数: {loss_type}")
        
        result = run_shared_cauchy_experiment(
            data_loaders=data_loaders,
            n_features=n_features,
            n_classes=n_classes,
            loss_type=loss_type,
            epochs=40,
            device='cpu'
        )
        
        loss_results.append({
            'loss_type': loss_type,
            'accuracy': result['results']['accuracy'],
            'f1': result['results']['classification_report']['weighted avg']['f1-score'],
            'train_time': result['train_time'],
            'final_train_loss': result['history']['train_loss'][-1],
            'final_val_loss': result['history']['val_loss'][-1],
            'avg_uncertainty': result['uncertainty_analysis']['uncertainties'].mean(),
            'convergence_epoch': len(result['history']['train_loss'])
        })
        
        print(f"  准确率: {result['results']['accuracy']:.4f}")
        print(f"  平均不确定性: {result['uncertainty_analysis']['uncertainties'].mean():.4f}")
    
    return loss_results


def analyze_noise_robustness(
    base_config: dict,
    noise_levels: list = [0.0, 0.1, 0.2, 0.3],
    outlier_ratios: list = [0.0, 0.05, 0.1, 0.2]
):
    """分析噪声鲁棒性"""
    print("\n" + "="*50)
    print("噪声鲁棒性分析")
    print("="*50)
    
    robustness_results = []
    
    # 测试噪声水平
    for noise_level in noise_levels:
        print(f"\n测试噪声水平: {noise_level}")
        
        X, y, _ = generate_multiclass_dataset(
            n_samples=base_config['n_samples'],
            n_features=base_config['n_features'],
            n_classes=base_config['n_classes'],
            noise_level=noise_level,
            random_state=42
        )
        
        data_loaders = create_data_loaders(X, y)
        
        # Shared Cauchy OvR
        cauchy_result = run_shared_cauchy_experiment(
            data_loaders=data_loaders,
            n_features=base_config['n_features'],
            n_classes=base_config['n_classes'],
            epochs=30,
            device='cpu'
        )
        
        # 基线方法
        baseline_results = run_baseline_experiments(data_loaders, base_config['n_classes'])
        
        result_entry = {
            'condition': 'noise',
            'level': noise_level,
            'cauchy_accuracy': cauchy_result['results']['accuracy'],
            'cauchy_uncertainty': cauchy_result['uncertainty_analysis']['uncertainties'].mean()
        }
        
        for name, baseline in baseline_results.items():
            if baseline is not None:
                result_entry[f'{name.lower()}_accuracy'] = baseline['accuracy']
        
        robustness_results.append(result_entry)
    
    # 测试异常值比例
    for outlier_ratio in outlier_ratios:
        print(f"\n测试异常值比例: {outlier_ratio}")
        
        X, y, _ = generate_multiclass_dataset(
            n_samples=base_config['n_samples'],
            n_features=base_config['n_features'],
            n_classes=base_config['n_classes'],
            outlier_ratio=outlier_ratio,
            random_state=42
        )
        
        data_loaders = create_data_loaders(X, y)
        
        # Shared Cauchy OvR
        cauchy_result = run_shared_cauchy_experiment(
            data_loaders=data_loaders,
            n_features=base_config['n_features'],
            n_classes=base_config['n_classes'],
            epochs=30,
            device='cpu'
        )
        
        # 基线方法
        baseline_results = run_baseline_experiments(data_loaders, base_config['n_classes'])
        
        result_entry = {
            'condition': 'outlier',
            'level': outlier_ratio,
            'cauchy_accuracy': cauchy_result['results']['accuracy'],
            'cauchy_uncertainty': cauchy_result['uncertainty_analysis']['uncertainties'].mean()
        }
        
        for name, baseline in baseline_results.items():
            if baseline is not None:
                result_entry[f'{name.lower()}_accuracy'] = baseline['accuracy']
        
        robustness_results.append(result_entry)
    
    return robustness_results


def save_results(results: dict, results_dir: str):
    """保存实验结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存主要结果
    main_results_file = os.path.join(results_dir, f"v2_experiment_results_{timestamp}.json")
    
    # 转换numpy数组为列表以便JSON序列化
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    # 创建可序列化的结果副本，排除模型对象
    def make_serializable(obj):
        if isinstance(obj, dict):
            serializable_dict = {}
            for key, value in obj.items():
                # 跳过模型对象和训练器对象
                if key in ['model', 'trainer'] or str(type(value).__name__).endswith('Classifier') or str(type(value).__name__).endswith('Trainer'):
                    continue
                serializable_dict[key] = make_serializable(value)
            return serializable_dict
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        else:
            return convert_numpy(obj)
    
    serializable_results = make_serializable(results)
    
    with open(main_results_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"实验结果已保存到: {main_results_file}")
    
    return main_results_file


def plot_comprehensive_results(results: dict, results_dir: str):
    """绘制综合结果图表"""
    plt.style.use('default')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 可扩展性分析图
    if 'scalability' in results:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        scalability_df = pd.DataFrame(results['scalability'])
        
        # 准确率 vs 类别数
        axes[0, 0].plot(scalability_df['n_classes'], scalability_df['accuracy'], 'bo-')
        axes[0, 0].set_xlabel('Number of Classes')
        axes[0, 0].set_ylabel('Test Accuracy')
        axes[0, 0].set_title('Scalability: Accuracy vs Classes')
        axes[0, 0].grid(True)
        
        # 训练时间 vs 类别数
        axes[0, 1].plot(scalability_df['n_classes'], scalability_df['train_time'], 'ro-')
        axes[0, 1].set_xlabel('Number of Classes')
        axes[0, 1].set_ylabel('Training Time (s)')
        axes[0, 1].set_title('Scalability: Training Time vs Classes')
        axes[0, 1].grid(True)
        
        # 模型参数 vs 类别数
        axes[1, 0].plot(scalability_df['n_classes'], scalability_df['model_params'], 'go-')
        axes[1, 0].set_xlabel('Number of Classes')
        axes[1, 0].set_ylabel('Model Parameters')
        axes[1, 0].set_title('Scalability: Parameters vs Classes')
        axes[1, 0].grid(True)
        
        # 不确定性 vs 类别数
        axes[1, 1].plot(scalability_df['n_classes'], scalability_df['avg_uncertainty'], 'mo-')
        axes[1, 1].set_xlabel('Number of Classes')
        axes[1, 1].set_ylabel('Average Uncertainty')
        axes[1, 1].set_title('Scalability: Uncertainty vs Classes')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        scalability_plot_file = os.path.join(results_dir, f"scalability_analysis_{timestamp}.png")
        plt.savefig(scalability_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"可扩展性分析图已保存到: {scalability_plot_file}")
    
    # 2. 损失函数比较图
    if 'loss_comparison' in results:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        loss_df = pd.DataFrame(results['loss_comparison'])
        
        # 准确率比较
        axes[0].bar(loss_df['loss_type'], loss_df['accuracy'])
        axes[0].set_ylabel('Test Accuracy')
        axes[0].set_title('Loss Function Comparison: Accuracy')
        axes[0].tick_params(axis='x', rotation=45)
        
        # 不确定性比较
        axes[1].bar(loss_df['loss_type'], loss_df['avg_uncertainty'])
        axes[1].set_ylabel('Average Uncertainty')
        axes[1].set_title('Loss Function Comparison: Uncertainty')
        axes[1].tick_params(axis='x', rotation=45)
        
        # 收敛速度比较
        axes[2].bar(loss_df['loss_type'], loss_df['convergence_epoch'])
        axes[2].set_ylabel('Convergence Epoch')
        axes[2].set_title('Loss Function Comparison: Convergence')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        loss_plot_file = os.path.join(results_dir, f"loss_comparison_{timestamp}.png")
        plt.savefig(loss_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"损失函数比较图已保存到: {loss_plot_file}")
    
    # 3. 鲁棒性分析图
    if 'robustness' in results:
        robustness_df = pd.DataFrame(results['robustness'])
        
        # 分别分析噪声和异常值
        noise_data = robustness_df[robustness_df['condition'] == 'noise']
        outlier_data = robustness_df[robustness_df['condition'] == 'outlier']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 噪声鲁棒性
        if not noise_data.empty:
            model_cols = [col for col in noise_data.columns if col.endswith('_accuracy')]
            for col in model_cols:
                model_name = col.replace('_accuracy', '').upper()
                axes[0].plot(noise_data['level'], noise_data[col], 'o-', label=model_name)
            axes[0].set_xlabel('Noise Level')
            axes[0].set_ylabel('Test Accuracy')
            axes[0].set_title('Robustness to Noise')
            axes[0].legend()
            axes[0].grid(True)
        
        # 异常值鲁棒性
        if not outlier_data.empty:
            model_cols = [col for col in outlier_data.columns if col.endswith('_accuracy')]
            for col in model_cols:
                model_name = col.replace('_accuracy', '').upper()
                axes[1].plot(outlier_data['level'], outlier_data[col], 'o-', label=model_name)
            axes[1].set_xlabel('Outlier Ratio')
            axes[1].set_ylabel('Test Accuracy')
            axes[1].set_title('Robustness to Outliers')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        robustness_plot_file = os.path.join(results_dir, f"robustness_analysis_{timestamp}.png")
        plt.savefig(robustness_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"鲁棒性分析图已保存到: {robustness_plot_file}")


def main():
    """主实验函数"""
    print("🚀 开始V2 Shared Cauchy OvR Classifier实验")
    print("="*60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建结果目录
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_v2')
    os.makedirs(results_dir, exist_ok=True)
    
    # 基础配置
    base_config = {
        'n_samples': 2000,
        'n_features': 20,
        'n_classes': 10
    }
    
    all_results = {}
    
    try:
        # 1. 基础性能测试
        print("\n" + "="*50)
        print("基础性能测试")
        print("="*50)
        
        X, y, scaler = generate_multiclass_dataset(**base_config)
        data_loaders = create_data_loaders(X, y)
        
        # Shared Cauchy OvR实验
        cauchy_result = run_shared_cauchy_experiment(
            data_loaders=data_loaders,
            n_features=base_config['n_features'],
            n_classes=base_config['n_classes'],
            epochs=50
        )
        
        # 基线方法实验
        baseline_results = run_baseline_experiments(data_loaders, base_config['n_classes'])
        
        all_results['main_experiment'] = {
            'cauchy': cauchy_result,
            'baselines': baseline_results
        }
        
        # 2. 可扩展性分析
        scalability_results = analyze_scalability(base_config)
        all_results['scalability'] = scalability_results
        
        # 3. 损失函数比较
        loss_results = analyze_loss_functions(
            data_loaders, base_config['n_features'], base_config['n_classes']
        )
        all_results['loss_comparison'] = loss_results
        
        # 4. 噪声鲁棒性分析
        robustness_results = analyze_noise_robustness(base_config)
        all_results['robustness'] = robustness_results
        
        # 保存结果
        results_file = save_results(all_results, results_dir)
        
        # 绘制图表
        plot_comprehensive_results(all_results, results_dir)
        
        # 打印摘要
        print("\n" + "="*60)
        print("🎉 实验完成摘要")
        print("="*60)
        
        # 主实验结果
        main_cauchy = all_results['main_experiment']['cauchy']
        print(f"Shared Cauchy OvR性能:")
        print(f"  - 测试准确率: {main_cauchy['results']['accuracy']:.4f}")
        print(f"  - 加权F1分数: {main_cauchy['results']['classification_report']['weighted avg']['f1-score']:.4f}")
        print(f"  - 训练时间: {main_cauchy['train_time']:.2f}s")
        print(f"  - 平均不确定性: {main_cauchy['uncertainty_analysis']['uncertainties'].mean():.4f}")
        
        print(f"\n基线方法对比:")
        for name, result in all_results['main_experiment']['baselines'].items():
            if result is not None:
                print(f"  - {name}: 准确率={result['accuracy']:.4f}, F1={result['f1']:.4f}")
        
        # 可扩展性结果
        if scalability_results:
            max_classes = max(r['n_classes'] for r in scalability_results)
            max_acc = [r['accuracy'] for r in scalability_results if r['n_classes'] == max_classes][0]
            print(f"\n可扩展性:")
            print(f"  - 最大测试类别: {max_classes}")
            print(f"  - 对应准确率: {max_acc:.4f}")
        
        print(f"\n结果已保存到: {results_dir}")
        
        return all_results, results_dir
        
    except Exception as e:
        print(f"\n❌ 实验过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, results_dir


if __name__ == "__main__":
    results, results_dir = main() 