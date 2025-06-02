"""
标签噪声注入演示脚本

演示DataProcessor类中新增的标签噪声注入功能，包括各种噪声类型的效果对比。
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris, load_wine
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_processor import DataProcessor

def demo_label_noise_injection():
    """演示标签噪声注入功能"""
    
    print("🔬 Label Noise Injection Demo")
    print("=" * 50)
    
    # 加载数据集
    print("📁 Loading datasets...")
    datasets = {
        'Iris': load_iris(),
        'Wine': load_wine()
    }
    
    # 噪声类型列表
    noise_types = [
        'random_uniform',
        'proportional', 
        'majority_bias',
        'minority_bias',
        'adjacent',
        'flip_pairs'
    ]
    
    # 噪声水平
    noise_levels = [0.1, 0.2, 0.3]
    
    # 为每个数据集测试噪声注入
    for dataset_name, dataset in datasets.items():
        print(f"\n📊 Testing dataset: {dataset_name}")
        print("-" * 30)
        
        X, y = dataset.data, dataset.target
        target_names = dataset.target_names
        
        print(f"Dataset info: {len(X)} samples, {len(np.unique(y))} classes")
        
        # 显示原始类别分布
        original_dist = {f'Class {i} ({target_names[i]})': np.sum(y == i) 
                        for i in range(len(target_names))}
        print("Original distribution:", original_dist)
        
        # 测试不同噪声类型
        for noise_type in noise_types:
            print(f"\n🧪 Testing noise type: {noise_type}")
            
            for noise_level in noise_levels:
                try:
                    # 注入噪声
                    y_noisy, noise_info = DataProcessor.inject_label_noise(
                        y, noise_level, noise_type, random_state=42
                    )
                    
                    # 显示结果
                    print(f"  📈 Noise level {noise_level:.1%}: "
                          f"{noise_info['changes']} samples changed")
                    
                    # 显示标签变化统计
                    if noise_info['label_changes']:
                        for orig_label, changes in noise_info['label_changes'].items():
                            for new_label, count in changes.items():
                                orig_name = target_names[orig_label]
                                new_name = target_names[new_label]
                                print(f"    {orig_name} -> {new_name}: {count} changes")
                
                except Exception as e:
                    print(f"  ❌ Error with {noise_type} at {noise_level:.1%}: {str(e)}")
    
    print("\n✅ Demo completed!")
    
def visualize_noise_effects():
    """可视化噪声效果"""
    print("\n🎨 Creating visualizations...")
    
    # 加载iris数据集
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # 测试参数
    noise_level = 0.2
    noise_types = ['random_uniform', 'proportional', 'majority_bias', 'minority_bias']
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # 绘制原始分布
    ax = axes[0]
    unique_labels, counts = np.unique(y, return_counts=True)
    bars = ax.bar(range(len(unique_labels)), counts, 
                  color=['red', 'green', 'blue'], alpha=0.7)
    ax.set_title('Original Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_xticks(range(len(unique_labels)))
    ax.set_xticklabels([f'Class {i}' for i in unique_labels])
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                str(count), ha='center', va='bottom')
    
    # 为每种噪声类型绘制分布
    for idx, noise_type in enumerate(noise_types, 1):
        ax = axes[idx]
        
        # 注入噪声
        y_noisy, noise_info = DataProcessor.inject_label_noise(
            y, noise_level, noise_type, random_state=42
        )
        
        # 绘制新分布
        unique_labels_new, counts_new = np.unique(y_noisy, return_counts=True)
        bars = ax.bar(range(len(unique_labels_new)), counts_new, 
                      color=['red', 'green', 'blue'], alpha=0.7)
        
        ax.set_title(f'{noise_type.replace("_", " ").title()}\n'
                    f'({noise_info["changes"]} changes)', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_xticks(range(len(unique_labels_new)))
        ax.set_xticklabels([f'Class {i}' for i in unique_labels_new])
        
        # 添加数值标签
        for bar, count in zip(bars, counts_new):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    str(count), ha='center', va='bottom')
    
    # 隐藏多余的子图
    axes[5].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle(f'Label Noise Effects Comparison (Noise Level: {noise_level:.1%})', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # 保存图片
    plt.savefig('label_noise_effects.png', dpi=300, bbox_inches='tight')
    print("📊 Visualization saved as 'label_noise_effects.png'")
    plt.show()

def compare_noise_robustness():
    """比较不同噪声类型对模型性能的影响"""
    print("\n🧪 Comparing noise robustness...")
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    
    # 加载数据
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 测试参数
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
    noise_types = ['random_uniform', 'proportional', 'majority_bias', 'minority_bias']
    
    # 存储结果
    results = []
    
    for noise_type in noise_types:
        for noise_level in noise_levels:
            # 注入噪声
            y_train_noisy, _ = DataProcessor.inject_label_noise(
                y_train, noise_level, noise_type, random_state=42
            )
            
            # 训练模型
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train_scaled, y_train_noisy)
            
            # 评估（在干净的测试集上）
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            results.append({
                'noise_type': noise_type,
                'noise_level': noise_level,
                'accuracy': accuracy
            })
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(results)
    
    # 可视化结果
    plt.figure(figsize=(12, 8))
    
    for noise_type in noise_types:
        data = results_df[results_df['noise_type'] == noise_type]
        plt.plot(data['noise_level'], data['accuracy'], 
                marker='o', linewidth=2, markersize=8,
                label=noise_type.replace('_', ' ').title())
    
    plt.xlabel('Noise Level', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Model Robustness to Different Label Noise Types', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    # 保存图片
    plt.savefig('noise_robustness_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 Robustness comparison saved as 'noise_robustness_comparison.png'")
    plt.show()
    
    return results_df

if __name__ == "__main__":
    # 运行演示
    demo_label_noise_injection()
    
    # 创建可视化
    visualize_noise_effects()
    
    # 比较鲁棒性
    results_df = compare_noise_robustness()
    
    print("\n📋 Final Results Summary:")
    print(results_df.groupby(['noise_type'])['accuracy'].agg(['mean', 'std']).round(3)) 