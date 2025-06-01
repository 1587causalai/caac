"""
实验脚本

运行分类实验，评估模型性能。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import json
from sklearn.metrics import classification_report

import sys
sys.path.append('../')

from data.data_processor import DataProcessor
from experiments.model_evaluator import ModelEvaluator
from models.caac_ovr_model import CAACOvRModel

def run_experiment(dataset_name, model_params=None, results_dir='../../results', random_state=42):
    """
    运行实验
    
    Args:
        dataset_name: 数据集名称
        model_params: 模型参数
        results_dir: 结果保存目录
        random_state: 随机种子
        
    Returns:
        experiment_results: 实验结果
    """
    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)
    experiment_name = f"caac_ovr_{dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = os.path.join(results_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 加载数据集
    data_processor = DataProcessor()
    X, y, feature_names, target_names = data_processor.load_dataset(dataset_name, random_state=random_state)
    
    # 数据预处理
    X_processed, scaler = data_processor.preprocess_data(X, standardize=True)
    
    # 数据分割
    X_train, X_val, X_test, y_train, y_val, y_test = data_processor.split_data(
        X_processed, y, test_size=0.2, val_size=0.2, random_state=random_state
    )
    
    # 获取数据集信息
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    class_counts, class_distribution = data_processor.get_class_distribution(y)
    
    print(f"Dataset: {dataset_name}")
    print(f"Number of samples: {n_samples}")
    print(f"Number of features: {n_features}")
    print(f"Number of classes: {n_classes}")
    print(f"Class distribution: {class_distribution}")
    
    # 设置默认模型参数
    default_params = {
        'input_dim': n_features,
        'representation_dim': 64,
        'latent_dim': 32,
        'n_classes': n_classes,
        'feature_hidden_dims': [128, 64],
        'abduction_hidden_dims': [64, 32],
        'threshold': 0.0,
        'lr': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'early_stopping_patience': 10
    }
    
    # 更新模型参数
    if model_params:
        default_params.update(model_params)
    
    # 创建模型
    model = CAACOvRModel(**default_params)
    
    # 训练模型
    print("Training model...")
    model.fit(X_train, y_train, X_val, y_val, verbose=1)
    
    # 评估模型
    print("Evaluating model...")
    evaluator = ModelEvaluator(results_dir=results_dir)
    metrics, y_pred_proba, y_pred, cm = evaluator.evaluate_model(model, X_test, y_test)
    
    # 打印评估指标
    print("Evaluation metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # 打印分类报告
    print("\nClassification Report:")
    # 处理target_names，确保是字符串类型或None
    if target_names is not None:
        # 将numpy数组或任何类型的target_names转换为字符串列表
        target_names_str = [str(name) for name in target_names]
    else:
        target_names_str = None
    
    print(classification_report(y_test, y_pred, target_names=target_names_str))
    
    # 可视化混淆矩阵
    cm_fig = evaluator.visualize_confusion_matrix(
        cm, 
        class_names=target_names_str if target_names_str is not None else [str(i) for i in range(n_classes)],
        title=f'Confusion Matrix - {dataset_name}',
        save_path=os.path.join(experiment_dir, 'confusion_matrix.png')
    )
    
    # 可视化ROC曲线
    roc_fig = evaluator.visualize_roc_curve(
        y_test, 
        y_pred_proba, 
        n_classes,
        save_path=os.path.join(experiment_dir, 'roc_curve.png')
    )
    
    # 可视化不确定性
    uncertainty_fig = evaluator.visualize_uncertainty(
        model, 
        X_test, 
        y_test, 
        n_samples=5,
        save_path=os.path.join(experiment_dir, 'uncertainty.png')
    )
    
    # 保存实验结果
    result_path = evaluator.save_experiment_results(
        experiment_name,
        dataset_name,
        default_params,
        metrics,
        model.history['train_time'],
        n_samples,
        n_features,
        n_classes,
        class_distribution
    )
    
    # 保存训练历史
    history_path = os.path.join(experiment_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        # 将numpy数组转换为列表
        history_dict = {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                       for k, v in model.history.items()}
        json.dump(history_dict, f, indent=4)
    
    # 可视化训练历史
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(model.history['train_loss'], label='Train Loss')
    if 'val_loss' in model.history and len(model.history['val_loss']) > 0:
        plt.plot(model.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(model.history['train_acc'], label='Train Accuracy')
    if 'val_acc' in model.history and len(model.history['val_acc']) > 0:
        plt.plot(model.history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'training_history.png'))
    
    print(f"Experiment results saved to {experiment_dir}")
    
    return {
        'experiment_name': experiment_name,
        'experiment_dir': experiment_dir,
        'metrics': metrics,
        'model': model,
        'result_path': result_path,
        'history_path': history_path
    }

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='Run classification experiments')
    parser.add_argument('--dataset', type=str, default='iris', 
                        choices=['iris', 'wine', 'digits', 'breast_cancer'],
                        help='Dataset name')
    parser.add_argument('--results_dir', type=str, default='../../results',
                        help='Results directory')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    run_experiment(args.dataset, results_dir=args.results_dir, random_state=args.random_state)

if __name__ == '__main__':
    main()
