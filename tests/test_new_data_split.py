#!/usr/bin/env python3
"""
测试新的数据分割策略和outlier添加功能

展示如何使用：
- 70% train / 15% val / 15% test 分割
- 在 train+val 中添加 outliers
- 保持 test 数据干净
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, load_diabetes
from sklearn.preprocessing import StandardScaler
import sys
import os

# 添加src路径以导入我们的模块
sys.path.append('../src')
from data.data_processor import DataProcessor

def test_synthetic_data():
    """测试合成回归数据"""
    print("=" * 60)
    print("测试合成回归数据")
    print("=" * 60)
    
    # 生成合成回归数据
    X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
    
    print(f"原始数据: X shape: {X.shape}, y shape: {y.shape}")
    print(f"y 统计: mean={y.mean():.2f}, std={y.std():.2f}")
    
    # 使用新的数据分割方法
    result = DataProcessor.split_data_with_outliers(
        X, y, 
        train_size=0.7, val_size=0.15, test_size=0.15,
        outlier_ratio=0.1, outlier_strength=3.0, 
        random_state=42
    )
    
    X_train, X_val, X_test, y_train, y_val, y_test, outlier_info = result
    
    # 打印分割结果
    print(f"\n数据分割结果:")
    print(f"Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # 打印outlier信息
    print(f"\nOutlier 信息:")
    for key, value in outlier_info.items():
        print(f"{key}: {value}")
    
    # 比较数据统计
    print(f"\n数据统计对比:")
    print(f"Train y: mean={y_train.mean():.2f}, std={y_train.std():.2f}")
    print(f"Val y:   mean={y_val.mean():.2f}, std={y_val.std():.2f}")
    print(f"Test y:  mean={y_test.mean():.2f}, std={y_test.std():.2f}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, outlier_info

def test_real_data():
    """测试真实数据集（糖尿病数据）"""
    print("\n" + "=" * 60)
    print("测试真实数据集（糖尿病数据）")
    print("=" * 60)
    
    # 加载糖尿病数据集
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"糖尿病数据: X shape: {X_scaled.shape}, y shape: {y.shape}")
    print(f"y 统计: mean={y.mean():.2f}, std={y.std():.2f}")
    
    # 使用新的数据分割方法
    result = DataProcessor.split_data_with_outliers(
        X_scaled, y,
        train_size=0.7, val_size=0.15, test_size=0.15,
        outlier_ratio=0.15, outlier_strength=2.5,
        random_state=42
    )
    
    X_train, X_val, X_test, y_train, y_val, y_test, outlier_info = result
    
    print(f"\n数据分割结果:")
    print(f"Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    print(f"\nOutlier 信息:")
    for key, value in outlier_info.items():
        print(f"{key}: {value}")
    
    print(f"\n数据统计对比:")
    print(f"Train y: mean={y_train.mean():.2f}, std={y_train.std():.2f}")
    print(f"Val y:   mean={y_val.mean():.2f}, std={y_val.std():.2f}")
    print(f"Test y:  mean={y_test.mean():.2f}, std={y_test.std():.2f}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, outlier_info

def visualize_outliers(y_train, y_val, y_test, outlier_info):
    """可视化outlier的影响"""
    print("\n" + "=" * 60)
    print("可视化outlier影响")
    print("=" * 60)
    
    plt.figure(figsize=(15, 5))
    
    # 子图1: 目标变量分布对比
    plt.subplot(1, 3, 1)
    plt.hist(y_train, bins=30, alpha=0.7, label=f'Train (n={len(y_train)})', color='blue')
    plt.hist(y_val, bins=20, alpha=0.7, label=f'Val (n={len(y_val)})', color='orange')
    plt.hist(y_test, bins=20, alpha=0.7, label=f'Test (n={len(y_test)})', color='green')
    plt.xlabel('Target Value')
    plt.ylabel('Frequency')
    plt.title('Target Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 箱线图对比
    plt.subplot(1, 3, 2)
    data_for_boxplot = [y_train, y_val, y_test]
    labels = ['Train\n(with outliers)', 'Val\n(with outliers)', 'Test\n(clean)']
    bp = plt.boxplot(data_for_boxplot, labels=labels, patch_artist=True)
    
    # 设置颜色
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.ylabel('Target Value')
    plt.title('Distribution Comparison (Box Plot)')
    plt.grid(True, alpha=0.3)
    
    # 子图3: 统计信息条形图
    plt.subplot(1, 3, 3)
    datasets = ['Train', 'Val', 'Test']
    means = [y_train.mean(), y_val.mean(), y_test.mean()]
    stds = [y_train.std(), y_val.std(), y_test.std()]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    plt.bar(x - width/2, means, width, label='Mean', alpha=0.8)
    plt.bar(x + width/2, stds, width, label='Std Dev', alpha=0.8)
    
    plt.xlabel('Dataset')
    plt.ylabel('Value')
    plt.title('Statistical Summary')
    plt.xticks(x, datasets)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_split_with_outliers_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印详细统计
    print(f"统计对比:")
    print(f"Train: mean={y_train.mean():.3f}, std={y_train.std():.3f}, min={y_train.min():.3f}, max={y_train.max():.3f}")
    print(f"Val:   mean={y_val.mean():.3f}, std={y_val.std():.3f}, min={y_val.min():.3f}, max={y_val.max():.3f}")
    print(f"Test:  mean={y_test.mean():.3f}, std={y_test.std():.3f}, min={y_test.min():.3f}, max={y_test.max():.3f}")

def test_model_training_example():
    """演示如何在模型训练中使用新的数据分割"""
    print("\n" + "=" * 60)
    print("演示模型训练流程")
    print("=" * 60)
    
    # 生成数据
    X, y = make_regression(n_samples=500, n_features=5, noise=5, random_state=42)
    
    # 使用新的数据分割
    result = DataProcessor.split_data_with_outliers(
        X, y,
        train_size=0.7, val_size=0.15, test_size=0.15,
        outlier_ratio=0.12, outlier_strength=2.0,
        random_state=42
    )
    
    X_train, X_val, X_test, y_train, y_val, y_test, outlier_info = result
    
    print("模型训练示例流程:")
    print("1. 数据已分割为 70% train / 15% val / 15% test")
    print("2. Train+Val 数据包含 outliers，Test 数据保持干净")
    print("3. 模型训练将在含有 outliers 的数据上进行")
    print("4. 早停将基于 validation set (含 outliers)")
    print("5. 最终评估将在干净的 test set 上进行")
    
    print(f"\n实际分割结果:")
    print(f"- Train: {len(X_train)} samples, outliers: {outlier_info['outliers_in_train']}")
    print(f"- Val: {len(X_val)} samples, outliers: {outlier_info['outliers_in_val']}")
    print(f"- Test: {len(X_test)} samples, outliers: 0 (clean)")
    
    # 模拟训练流程
    print(f"\n模拟训练流程:")
    print(f"model.fit(X_train, y_train, X_val, y_val)  # 使用含outliers的数据训练和验证")
    print(f"final_score = model.evaluate(X_test, y_test)  # 在干净数据上最终评估")
    
    return result

def main():
    """主函数"""
    print("新数据分割策略和outlier添加功能测试")
    print("目标: 70% train / 15% val / 15% test")
    print("设计: train+val含outliers, test保持干净")
    
    # 测试合成数据
    X_train1, X_val1, X_test1, y_train1, y_val1, y_test1, info1 = test_synthetic_data()
    
    # 测试真实数据
    X_train2, X_val2, X_test2, y_train2, y_val2, y_test2, info2 = test_real_data()
    
    # 可视化 (使用真实数据的结果)
    visualize_outliers(y_train2, y_val2, y_test2, info2)
    
    # 演示模型训练流程
    test_model_training_example()
    
    print(f"\n" + "=" * 60)
    print("测试完成！")
    print("- 数据分割策略已更新为 70%/15%/15%")
    print("- Outlier添加功能已实现")
    print("- Train+Val 含outliers，Test保持干净")
    print("- 可视化结果已保存为 'data_split_with_outliers_analysis.png'")
    print("=" * 60)

if __name__ == "__main__":
    main() 