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

def run_all_multiclass_experiments(
    results_base_dir='results/multiclass',
    experiment_configs_override=None,
    outlier_ratios_override=None,
    global_caac_model_params_override=None,
    baseline_model_params_override=None,
    use_new_implementation=True  # 新增参数
):
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
            'n_samples': 800,  # 减少样本数进行快速测试
            'n_features': 10,
            'class_sep': 1.0,
            'model_params': {
                'n_paths': 3,
                'encoder_hidden_dim': 32,  # 减小网络规模
                'encoder_output_dim': 16,
                'd_c': 8,
                'epochs': 50,  # 减少训练轮数
                'early_stopping_patience': 10
            }
        },
        # 4类分类
        {
            'name': '4-Class Classification',
            'n_classes': 4,
            'n_samples': 1000,
            'n_features': 12,
            'class_sep': 0.8,
            'model_params': {
                'n_paths': 4,
                'encoder_hidden_dim': 32,
                'encoder_output_dim': 16,
                'd_c': 8,
                'epochs': 50,
                'early_stopping_patience': 10
            }
        }
    ]
    
    if experiment_configs_override is not None:
        experiment_configs = experiment_configs_override
        
    # 异常值比例配置
    outlier_ratios = [0.0, 0.1]  # 无异常值和10%异常值
    if outlier_ratios_override is not None:
        outlier_ratios = outlier_ratios_override
    
    # 创建结果目录
    results_dir = results_base_dir
    os.makedirs(results_dir, exist_ok=True)
    
    # 记录所有结果
    all_results = []
    
    for config in experiment_configs:
        print(f"\n{'='*60}")
        print(f"Running {config['name']} Experiments with {'New' if use_new_implementation else 'Old'} Implementation")
        print(f"{'='*60}")
        
        current_caac_params = config['model_params'].copy()
        if global_caac_model_params_override:
            current_caac_params.update(global_caac_model_params_override)

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
                model_params=current_caac_params,
                random_state=42,
                use_new_implementation=use_new_implementation  # 传递新参数
            )
            
            # 与基线方法比较
            comparison = compare_multiclass_with_baselines(
                results['data']['X_train'],
                results['data']['y_train'],
                results['data']['X_test'],
                results['data']['y_test'],
                caac_model=results['model'],
                n_classes=config['n_classes'],
                random_state=42,
                baseline_params_override=baseline_model_params_override
            )
            
            # 打印比较结果
            print("\nModel Comparison:")
            print(comparison['comparison_df'][['accuracy', 'f1_macro', 'auc_ovr', 'train_time']])
            
            # 可视化结果
            exp_name = f"{config['n_classes']}_classes_{outlier_str.lower()}"
            if use_new_implementation:
                exp_name += "_new_impl"
            
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
                'implementation': 'new' if use_new_implementation else 'old',
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
    print(f"All multiclass experiments completed using {'New' if use_new_implementation else 'Old'} implementation!")
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
    
    # Dynamically get the class numbers that were actually run
    executed_class_numbers = sorted(list(set(r['n_classes'] for r in all_results)))
    if not executed_class_numbers:
        print("No results found to create robustness analysis.")
        return
    class_numbers = executed_class_numbers

    # Adjust the number of subplots dynamically
    num_cols = len(class_numbers)
    
    fig, axes = plt.subplots(1, num_cols, figsize=(6 * num_cols, 6), squeeze=False) # Ensure axes is always 2D
    
    models = ['CAAC', 'LogisticRegression', 'RandomForest', 'SVM', 'MLP']
    # Ensure enough colors if more models/class_numbers are ever used, though 5 models is current max
    default_colors = plt.cm.get_cmap('tab10', max(len(models), num_cols)).colors 
    colors = [default_colors[i % len(default_colors)] for i in range(len(models))]

    for idx, n_classes in enumerate(class_numbers):
        ax = axes[0, idx] # Access subplot correctly for 1 row
        
        # 收集该类别数的结果
        clean_results_iter = (r for r in all_results if r['n_classes'] == n_classes and r['outlier_ratio'] == 0.0)
        outlier_results_iter = (r for r in all_results if r['n_classes'] == n_classes and r['outlier_ratio'] == 0.1)

        clean_results = next(clean_results_iter, None)
        outlier_results = next(outlier_results_iter, None)

        if clean_results is None or outlier_results is None:
            print(f"Skipping robustness analysis for {n_classes}-Class: Missing clean or outlier data.")
            ax.text(0.5, 0.5, f'{n_classes}-Class\nRobustness data N/A', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_title(f'{n_classes}-Class Robustness (N/A)')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
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
    # 运行新实现的快速测试
    print("Running multiclass experiments with NEW CAAC-SPSFT implementation...")
    
    # 快速测试配置
    custom_config = [{
        'name': '3-Class Fast Test with New Implementation',
        'n_classes': 3,
        'n_samples': 600,  # 减少样本数进行快速测试
        'n_features': 10,
        'class_sep': 1.0,
        'model_params': {
            'n_paths': 3,  # 路径数等于类别数
            'encoder_hidden_dim': 32,
            'encoder_output_dim': 16,
            'd_c': 8,  # 因果表征维度
            'epochs': 200,  # 减少训练轮数
            'early_stopping_patience': 8,
            'lr': 0.001,
            'batch_size': 32
        }
    }]
    
    run_all_multiclass_experiments(
        results_base_dir='results/multiclass_new_implementation_test',
        experiment_configs_override=custom_config,
        outlier_ratios_override=[0.0, 0.1],  # 运行干净数据和10%异常值数据
        global_caac_model_params_override={'epochs': 40, 'early_stopping_patience': 10},
        baseline_model_params_override={'MLP': {'max_iter': 200, 'hidden_layer_sizes': (50,)}},
        use_new_implementation=True  # 使用新实现
    ) 