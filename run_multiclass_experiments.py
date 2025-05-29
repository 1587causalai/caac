"""
多分类实验运行脚本

运行CAAC-SPSFT多分类实验，包括：
1. 不同类别数量的实验（3类、4类、5类）
2. 有无异常值的对比实验
3. 与基线方法的比较
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

from src.experiments.multiclass_classification import (
    run_multiclass_classification_experiment,
    compare_multiclass_with_baselines,
    visualize_multiclass_results
)

def run_all_multiclass_experiments():
    """运行所有多分类实验"""
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 实验配置
    experiment_configs = [
        # 3类分类
        {
            'name': '3-Class Classification',
            'n_classes': 3,
            'n_samples': 1500,
            'n_features': 10,
            'class_sep': 1.0,
            'model_params': {
                'n_paths': 3,
                'representation_dim': 64,
                'latent_dim': 32,
                'epochs': 150,
                'early_stopping_patience': 15
            }
        },
        # 4类分类
        {
            'name': '4-Class Classification',
            'n_classes': 4,
            'n_samples': 2000,
            'n_features': 12,
            'class_sep': 0.8,
            'model_params': {
                'n_paths': 4,
                'representation_dim': 64,
                'latent_dim': 32,
                'epochs': 150,
                'early_stopping_patience': 15
            }
        },
        # 5类分类
        {
            'name': '5-Class Classification',
            'n_classes': 5,
            'n_samples': 2500,
            'n_features': 15,
            'class_sep': 0.6,
            'model_params': {
                'n_paths': 5,
                'representation_dim': 64,
                'latent_dim': 32,
                'epochs': 200,
                'early_stopping_patience': 20
            }
        }
    ]
    
    # 异常值比例配置
    outlier_ratios = [0.0, 0.1]  # 无异常值和10%异常值
    
    # 创建结果目录
    results_dir = 'results/multiclass'
    os.makedirs(results_dir, exist_ok=True)
    
    # 记录所有结果
    all_results = []
    
    for config in experiment_configs:
        print(f"\n{'='*60}")
        print(f"Running {config['name']} Experiments")
        print(f"{'='*60}")
        
        for outlier_ratio in outlier_ratios:
            outlier_str = f"{int(outlier_ratio*100)}%" if outlier_ratio > 0 else "Clean"
            print(f"\n--- {outlier_str} Data ---")
            
            # 运行实验
            results = run_multiclass_classification_experiment(
                n_samples=config['n_samples'],
                n_features=config['n_features'],
                n_classes=config['n_classes'],
                class_sep=config['class_sep'],
                outlier_ratio=outlier_ratio,
                model_params=config['model_params'],
                random_state=42
            )
            
            # 与基线方法比较
            comparison = compare_multiclass_with_baselines(
                results['data']['X_train'],
                results['data']['y_train'],
                results['data']['X_test'],
                results['data']['y_test'],
                caac_model=results['model'],
                n_classes=config['n_classes'],
                random_state=42
            )
            
            # 打印比较结果
            print("\nModel Comparison:")
            print(comparison['comparison_df'][['accuracy', 'f1_macro', 'auc_ovr', 'train_time']])
            
            # 可视化结果
            exp_name = f"{config['n_classes']}_classes_{outlier_str.lower()}"
            fig = visualize_multiclass_results(
                results,
                save_path=os.path.join(results_dir, f'{exp_name}_visualization.png')
            )
            plt.close(fig)
            
            # 保存详细结果
            result_summary = {
                'experiment': config['name'],
                'n_classes': config['n_classes'],
                'outlier_ratio': outlier_ratio,
                'outlier_type': outlier_str,
                'metrics': comparison['results'],
                'training_history': {
                    'final_epoch': len(results['history']['train_loss']),
                    'best_val_acc': max(results['history'].get('val_acc', [0])),
                    'final_train_acc': results['history']['train_acc'][-1] if results['history']['train_acc'] else 0
                },
                'confusion_matrix': results['metrics']['confusion_matrix'].tolist(),
                'experiment_config': results['experiment_config']
            }
            all_results.append(result_summary)
            
            # 保存模型
            if results['model'] is not None:
                model_path = os.path.join(results_dir, f'{exp_name}_model.pth')
                torch.save(results['model'].model.state_dict(), model_path)
                print(f"Model saved to {model_path}")
    
    # 保存所有结果到JSON
    results_json_path = os.path.join(results_dir, 'all_multiclass_results.json')
    
    # 转换numpy数组为列表以便JSON序列化
    def convert_numpy_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_to_list(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        else:
            return obj
    
    # 转换所有结果中的numpy数组
    all_results_json = convert_numpy_to_list(all_results)
    
    with open(results_json_path, 'w') as f:
        json.dump(all_results_json, f, indent=2)
    print(f"\nAll results saved to {results_json_path}")
    
    # 创建汇总表格
    create_summary_table(all_results, results_dir)
    
    # 创建鲁棒性分析图
    create_robustness_analysis(all_results, results_dir)
    
    print("\n" + "="*60)
    print("All multiclass experiments completed!")
    print("="*60)

def create_summary_table(all_results, results_dir):
    """创建实验结果汇总表格"""
    
    summary_data = []
    
    for result in all_results:
        for model_name, metrics in result['metrics'].items():
            summary_data.append({
                'Classes': result['n_classes'],
                'Data Type': result['outlier_type'],
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.3f}",
                'F1 (Macro)': f"{metrics['f1_macro']:.3f}",
                'AUC (OvR)': f"{metrics.get('auc_ovr', 0):.3f}",
                'Train Time': f"{metrics['train_time']:.2f}s"
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # 保存为CSV
    csv_path = os.path.join(results_dir, 'multiclass_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSummary table saved to {csv_path}")
    
    # 创建Markdown表格
    markdown_path = os.path.join(results_dir, 'multiclass_summary.md')
    with open(markdown_path, 'w') as f:
        f.write("# Multiclass Classification Results Summary\n\n")
        f.write(summary_df.to_markdown(index=False))
    print(f"Markdown summary saved to {markdown_path}")

def create_robustness_analysis(all_results, results_dir):
    """创建鲁棒性分析图表"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    class_numbers = [3, 4, 5]
    models = ['CAAC', 'LogisticRegression', 'RandomForest', 'SVM']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, n_classes in enumerate(class_numbers):
        ax = axes[idx]
        
        # 收集该类别数的结果
        clean_results = next(r for r in all_results if r['n_classes'] == n_classes and r['outlier_ratio'] == 0.0)
        outlier_results = next(r for r in all_results if r['n_classes'] == n_classes and r['outlier_ratio'] == 0.1)
        
        # 计算准确率下降
        accuracy_drops = []
        model_names = []
        
        for model in models:
            if model in clean_results['metrics'] and model in outlier_results['metrics']:
                clean_acc = clean_results['metrics'][model]['accuracy']
                outlier_acc = outlier_results['metrics'][model]['accuracy']
                drop = (clean_acc - outlier_acc) * 100  # 转换为百分比
                accuracy_drops.append(drop)
                model_names.append(model)
        
        # 绘制条形图
        x = np.arange(len(model_names))
        bars = ax.bar(x, accuracy_drops, color=colors[:len(model_names)])
        
        # 添加数值标签
        for bar, drop in zip(bars, accuracy_drops):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{drop:.1f}%', ha='center', va='bottom')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy Drop (%)')
        ax.set_title(f'{n_classes}-Class Robustness')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(accuracy_drops) * 1.2 if accuracy_drops else 10)
    
    plt.suptitle('Model Robustness Analysis: Accuracy Drop with 10% Outliers', fontsize=16)
    plt.tight_layout()
    
    robustness_path = os.path.join(results_dir, 'multiclass_robustness_analysis.png')
    plt.savefig(robustness_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Robustness analysis saved to {robustness_path}")

if __name__ == "__main__":
    run_all_multiclass_experiments() 