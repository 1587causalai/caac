#!/usr/bin/env python3
"""
测试新扩展的数据集加载功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from compare_methods_outlier_robustness import load_datasets

def test_dataset_loading():
    """测试数据集加载功能"""
    print("🧪 测试数据集加载功能")
    print("=" * 50)
    
    try:
        # 加载所有数据集
        datasets = load_datasets()
        
        print(f"\n✅ 成功加载 {len(datasets)} 个数据集")
        
        # 显示详细统计
        print("\n📊 数据集详细信息:")
        print("-" * 80)
        
        total_samples = 0
        for key, dataset in datasets.items():
            n_samples, n_features = dataset['data'].shape
            n_classes = len(set(dataset['target']))
            size_label = dataset.get('size', 'unknown')
            total_samples += n_samples
            
            print(f"  {dataset['name']:<40} | {n_samples:>7}样本 | {n_features:>4}特征 | {n_classes:>3}类 | {size_label}")
        
        print("-" * 80)
        print(f"  {'总计':<40} | {total_samples:>7}样本")
        
        # 按规模分类统计
        size_stats = {}
        for dataset in datasets.values():
            size = dataset.get('size', 'unknown')
            size_stats[size] = size_stats.get(size, 0) + 1
        
        print("\n📈 数据集规模分布:")
        for size, count in size_stats.items():
            print(f"  {size}: {count}个数据集")
            
        print("\n🎯 推荐测试方案:")
        print("  1. 快速测试: 选择 small 规模数据集 (3-4个)")
        print("  2. 标准测试: 选择 small + medium 规模数据集 (6-8个)")
        print("  3. 完整测试: 包含所有数据集 (可能需要较长时间)")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集加载失败: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_dataset_loading()
    if success:
        print("\n✅ 数据集扩展功能测试通过！")
    else:
        print("\n❌ 数据集扩展功能测试失败！") 