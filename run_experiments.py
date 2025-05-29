"""
主实验运行脚本

运行CAAC-SPSFT二分类算法的实验，并与基线方法比较
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.binary_classification import run_binary_classification_experiment, compare_with_baselines

def main():
    # 设置随机种子
    random_state = 42
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    
    # 创建结果目录
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 实验配置
    experiment_configs = [
        # 线性数据，无异常值
        {'data_type': 'linear', 'outlier_ratio': 0.0, 'n_samples': 1000, 'n_features': 10},
        # 线性数据，有异常值
        {'data_type': 'linear', 'outlier_ratio': 0.1, 'n_samples': 1000, 'n_features': 10},
        # 非线性数据，无异常值
        {'data_type': 'nonlinear', 'outlier_ratio': 0.0, 'n_samples': 1000, 'n_features': 10},
        # 非线性数据，有异常值
        {'data_type': 'nonlinear', 'outlier_ratio': 0.1, 'n_samples': 1000, 'n_features': 10},
    ]
    
    # 运行实验
    all_results = []
    for i, config in enumerate(experiment_configs):
        print(f"\n=== 运行实验 {i+1}/{len(experiment_configs)} ===")
        print(f"配置: {config}")
        
        # 运行CAAC实验
        caac_results = run_binary_classification_experiment(
            n_samples=config['n_samples'],
            n_features=config['n_features'],
            data_type=config['data_type'],
            outlier_ratio=config['outlier_ratio'],
            random_state=random_state
        )
        
        # 与基线方法比较
        comparison = compare_with_baselines(
            caac_results['data']['X_train'],
            caac_results['data']['y_train'],
            caac_results['data']['X_test'],
            caac_results['data']['y_test'],
            caac_model=caac_results['model'],
            random_state=random_state
        )
        
        # 保存结果
        experiment_name = f"{config['data_type']}_outlier{config['outlier_ratio']}"
        comparison_df = comparison['comparison_df']
        
        # 打印比较结果
        print("\n=== 性能比较 ===")
        print(comparison_df)
        
        # 保存比较结果到CSV
        comparison_file = os.path.join(results_dir, f"comparison_{experiment_name}.csv")
        comparison_df.to_csv(comparison_file)
        print(f"比较结果已保存到: {comparison_file}")
        
        # 绘制训练历史
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(caac_results['history']['train_loss'], label='Train Loss')
        if 'val_loss' in caac_results['history'] and len(caac_results['history']['val_loss']) > 0:
            plt.plot(caac_results['history']['val_loss'], label='Val Loss')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(caac_results['history']['train_acc'], label='Train Acc')
        if 'val_acc' in caac_results['history'] and len(caac_results['history']['val_acc']) > 0:
            plt.plot(caac_results['history']['val_acc'], label='Val Acc')
        plt.title('Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        history_plot_file = os.path.join(results_dir, f"history_{experiment_name}.png")
        plt.savefig(history_plot_file)
        plt.close()
        print(f"训练历史图已保存到: {history_plot_file}")
        
        # 收集结果
        result_entry = {
            'experiment_name': experiment_name,
            'config': config,
            'caac_metrics': caac_results['metrics'],
            'comparison': comparison['results'],
            'train_time': caac_results['history']['train_time']
        }
        all_results.append(result_entry)
    
    # 保存所有实验结果摘要
    summary_df = pd.DataFrame([
        {
            'experiment': r['experiment_name'],
            'data_type': r['config']['data_type'],
            'outlier_ratio': r['config']['outlier_ratio'],
            'caac_accuracy': r['caac_metrics']['accuracy'],
            'caac_f1': r['caac_metrics']['f1'],
            'lr_accuracy': r['comparison']['LogisticRegression']['accuracy'],
            'lr_f1': r['comparison']['LogisticRegression']['f1'],
            'rf_accuracy': r['comparison']['RandomForest']['accuracy'],
            'rf_f1': r['comparison']['RandomForest']['f1'],
            'svm_accuracy': r['comparison']['SVM']['accuracy'],
            'svm_f1': r['comparison']['SVM']['f1'],
            'caac_train_time': r['train_time'],
            'lr_train_time': r['comparison']['LogisticRegression']['train_time'],
            'rf_train_time': r['comparison']['RandomForest']['train_time'],
            'svm_train_time': r['comparison']['SVM']['train_time']
        }
        for r in all_results
    ])
    
    # 保存摘要到CSV
    summary_file = os.path.join(results_dir, "experiment_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"\n实验摘要已保存到: {summary_file}")
    
    # 打印摘要
    print("\n=== 实验摘要 ===")
    print(summary_df)
    
    # 绘制比较图表
    plt.figure(figsize=(12, 10))
    
    # 准确率比较
    plt.subplot(2, 2, 1)
    for model in ['caac', 'lr', 'rf', 'svm']:
        plt.plot(summary_df['experiment'], summary_df[f'{model}_accuracy'], marker='o', label=model.upper())
    plt.title('Accuracy Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy')
    plt.legend()
    
    # F1分数比较
    plt.subplot(2, 2, 2)
    for model in ['caac', 'lr', 'rf', 'svm']:
        plt.plot(summary_df['experiment'], summary_df[f'{model}_f1'], marker='o', label=model.upper())
    plt.title('F1 Score Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('F1 Score')
    plt.legend()
    
    # 训练时间比较
    plt.subplot(2, 2, 3)
    for model in ['caac', 'lr', 'rf', 'svm']:
        plt.bar(np.arange(len(summary_df)) + 0.2 * (['caac', 'lr', 'rf', 'svm'].index(model) - 1.5), 
                summary_df[f'{model}_train_time'], 
                width=0.2, 
                label=model.upper())
    plt.title('Training Time Comparison')
    plt.xticks(np.arange(len(summary_df)), summary_df['experiment'], rotation=45)
    plt.ylabel('Time (seconds)')
    plt.legend()
    
    # 异常值鲁棒性比较（准确率下降百分比）
    plt.subplot(2, 2, 4)
    
    # 计算有无异常值的准确率差异
    robustness_data = []
    for data_type in summary_df['data_type'].unique():
        clean_data = summary_df[(summary_df['data_type'] == data_type) & (summary_df['outlier_ratio'] == 0.0)]
        noisy_data = summary_df[(summary_df['data_type'] == data_type) & (summary_df['outlier_ratio'] > 0.0)]
        
        if len(clean_data) > 0 and len(noisy_data) > 0:
            for model in ['caac', 'lr', 'rf', 'svm']:
                clean_acc = clean_data[f'{model}_accuracy'].values[0]
                noisy_acc = noisy_data[f'{model}_accuracy'].values[0]
                acc_drop_pct = (clean_acc - noisy_acc) / clean_acc * 100 if clean_acc > 0 else 0
                robustness_data.append({
                    'data_type': data_type,
                    'model': model.upper(),
                    'acc_drop_pct': acc_drop_pct
                })
    
    if robustness_data:
        robustness_df = pd.DataFrame(robustness_data)
        for i, data_type in enumerate(robustness_df['data_type'].unique()):
            data = robustness_df[robustness_df['data_type'] == data_type]
            x = np.arange(len(data))
            plt.bar(x + i * 0.25, data['acc_drop_pct'], width=0.25, label=data_type)
            plt.xticks(x + 0.25/2, data['model'])
        plt.title('Robustness to Outliers')
        plt.ylabel('Accuracy Drop (%)')
        plt.legend()
    
    plt.tight_layout()
    comparison_plot_file = os.path.join(results_dir, "model_comparison.png")
    plt.savefig(comparison_plot_file)
    plt.close()
    print(f"比较图表已保存到: {comparison_plot_file}")
    
    return summary_df, all_results, results_dir

if __name__ == "__main__":
    main()
