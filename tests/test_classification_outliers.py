#!/usr/bin/env python3
"""
测试分类数据的outlier添加功能

展示如何使用：
- 70% train / 15% val / 15% test 分割
- 在 train+val 中通过改变标签添加 outliers
- 保持 test 数据干净
- 处理类别不平衡问题
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
import sys
import os

# 添加src路径以导入我们的模块
sys.path.append('../src')
from data.data_processor import DataProcessor

def test_iris_data():
    """测试Iris数据集"""
    print("=" * 60)
    print("测试Iris数据集 (3类，平衡)")
    print("=" * 60)
    
    # 加载Iris数据
    iris = load_iris()
    X, y = iris.data, iris.target
    
    print(f"原始数据: X shape: {X.shape}, y shape: {y.shape}")
    print(f"类别: {iris.target_names}")
    
    # 显示原始类别分布
    unique, counts = np.unique(y, return_counts=True)
    print(f"原始类别分布: {dict(zip(unique, counts))}")
    
    # 测试不同的balance策略
    strategies = ['proportional', 'uniform', 'majority_to_minority', 'random']
    
    results = {}
    for strategy in strategies:
        print(f"\n--- 测试策略: {strategy} ---")
        
        result = DataProcessor.split_classification_data_with_outliers(
            X, y,
            train_size=0.7, val_size=0.15, test_size=0.15,
            outlier_ratio=0.15, balance_strategy=strategy,
            random_state=42
        )
        
        X_train, X_val, X_test, y_train, y_val, y_test, outlier_info = result
        results[strategy] = result
        
        print(f"数据分割: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        print(f"Outliers: Train={outlier_info['outliers_in_train']}, Val={outlier_info['outliers_in_val']}")
        
        # 显示类别变化
        if 'class_changes' in outlier_info:
            print("标签变化:")
            for orig_cls, changes in outlier_info['class_changes'].items():
                for new_cls, count in changes.items():
                    print(f"  {orig_cls} -> {new_cls}: {count} samples")
        
        # 显示新的类别分布
        print("新类别分布:")
        print(f"  Train: {outlier_info['train_class_distribution']}")
        print(f"  Val: {outlier_info['val_class_distribution']}")
        print(f"  Test: {outlier_info['test_class_distribution']}")
    
    return results

def test_imbalanced_data():
    """测试不平衡数据集（乳腺癌数据）"""
    print("\n" + "=" * 60)
    print("测试乳腺癌数据集 (2类，不平衡)")
    print("=" * 60)
    
    # 加载乳腺癌数据
    bc = load_breast_cancer()
    X, y = bc.data, bc.target
    
    print(f"原始数据: X shape: {X.shape}, y shape: {y.shape}")
    print(f"类别: {bc.target_names}")
    
    # 显示原始类别分布
    unique, counts = np.unique(y, return_counts=True)
    print(f"原始类别分布: {dict(zip(unique, counts))}")
    print(f"不平衡比例: {counts[0]/counts[1]:.2f}")
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 测试majority_to_minority策略（适合不平衡数据）
    print(f"\n--- 测试策略: majority_to_minority ---")
    
    result = DataProcessor.split_classification_data_with_outliers(
        X_scaled, y,
        train_size=0.7, val_size=0.15, test_size=0.15,
        outlier_ratio=0.12, balance_strategy='majority_to_minority',
        random_state=42
    )
    
    X_train, X_val, X_test, y_train, y_val, y_test, outlier_info = result
    
    print(f"数据分割: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    print(f"Outliers: Train={outlier_info['outliers_in_train']}, Val={outlier_info['outliers_in_val']}")
    
    print("标签变化:")
    for orig_cls, changes in outlier_info['class_changes'].items():
        for new_cls, count in changes.items():
            print(f"  类别{orig_cls} -> 类别{new_cls}: {count} samples")
    
    print("类别分布变化:")
    print(f"原始分布: {outlier_info['original_class_distribution']}")
    print(f"新分布: {outlier_info['new_class_distribution']}")
    
    return result

def visualize_outlier_effects(results_iris):
    """可视化outlier策略的效果"""
    print("\n" + "=" * 60)
    print("可视化outlier策略效果")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    strategies = ['proportional', 'uniform', 'majority_to_minority', 'random']
    
    for i, strategy in enumerate(strategies):
        ax = axes[i//2, i%2]
        
        _, _, _, y_train, y_val, y_test, outlier_info = results_iris[strategy]
        
        # 合并train和val数据用于显示
        y_train_val = np.concatenate([y_train, y_val])
        
        # 计算类别分布
        train_val_dist = {cls: np.sum(y_train_val == cls) for cls in np.unique(y_train_val)}
        test_dist = {cls: np.sum(y_test == cls) for cls in np.unique(y_test)}
        
        # 绘制条形图
        classes = sorted(train_val_dist.keys())
        train_val_counts = [train_val_dist[cls] for cls in classes]
        test_counts = [test_dist[cls] for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        
        ax.bar(x - width/2, train_val_counts, width, label='Train+Val (with outliers)', alpha=0.8)
        ax.bar(x + width/2, test_counts, width, label='Test (clean)', alpha=0.8)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Number of Samples')
        ax.set_title(f'Strategy: {strategy}\nOutliers: {outlier_info["total_outliers"]}')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Class {cls}' for cls in classes])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('classification_outliers_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_model_training_workflow():
    """演示完整的模型训练工作流程"""
    print("\n" + "=" * 60)
    print("演示模型训练工作流程")
    print("=" * 60)
    
    # 使用Wine数据集
    wine = load_wine()
    X, y = wine.data, wine.target
    
    print(f"使用Wine数据集: {X.shape}, 类别数: {len(np.unique(y))}")
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 分割数据并添加outliers
    result = DataProcessor.split_classification_data_with_outliers(
        X_scaled, y,
        train_size=0.7, val_size=0.15, test_size=0.15,
        outlier_ratio=0.1, balance_strategy='proportional',
        random_state=42
    )
    
    X_train, X_val, X_test, y_train, y_val, y_test, outlier_info = result
    
    print("\n完整训练工作流程:")
    print("1. 数据已分割为 70% train / 15% val / 15% test")
    print("2. Train+Val 数据包含 label outliers，Test 数据保持干净")
    print("3. 模型训练在含噪声标签的数据上进行，测试鲁棒性")
    print("4. 早停基于 validation set (含 label noise)")
    print("5. 最终评估在干净的 test set 上进行")
    
    print(f"\n实际分割结果:")
    print(f"- Train: {len(X_train)} samples, label outliers: {outlier_info['outliers_in_train']}")
    print(f"- Val: {len(X_val)} samples, label outliers: {outlier_info['outliers_in_val']}")
    print(f"- Test: {len(X_test)} samples, label outliers: 0 (clean)")
    
    print(f"\n模拟训练代码:")
    print(f"# 使用含噪声标签的数据训练")
    print(f"model.fit(X_train, y_train, X_val, y_val)")
    print(f"# 在干净测试集上评估真实性能")
    print(f"clean_accuracy = model.evaluate(X_test, y_test)")
    
    print(f"\nOutlier策略效果:")
    print(f"- 策略: {outlier_info['balance_strategy']}")
    print(f"- 总outliers: {outlier_info['total_outliers']}")
    print(f"- 标签变化统计: {outlier_info['class_changes']}")
    
    return result

def main():
    """主函数"""
    print("分类数据outlier添加功能测试")
    print("目标: 通过改变标签创建outliers，测试模型鲁棒性")
    print("设计: train+val含label outliers, test保持干净")
    
    # 测试平衡数据集
    results_iris = test_iris_data()
    
    # 测试不平衡数据集
    result_bc = test_imbalanced_data()
    
    # 可视化效果
    visualize_outlier_effects(results_iris)
    
    # 演示训练工作流程
    test_model_training_workflow()
    
    print(f"\n" + "=" * 60)
    print("测试完成！")
    print("- 分类数据分割策略已更新为 70%/15%/15%")
    print("- 支持4种label outlier添加策略")
    print("- 自动处理类别不平衡问题")
    print("- Train+Val 含label outliers，Test保持干净")
    print("- 可视化结果已保存为 'classification_outliers_comparison.png'")
    print("=" * 60)

if __name__ == "__main__":
    main() 