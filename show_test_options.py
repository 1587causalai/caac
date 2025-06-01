#!/usr/bin/env python3
"""
CAAC项目测试选项概览
显示所有可用的测试类型和使用方法
"""

def show_test_options():
    """显示所有测试选项"""
    print("🧪 CAAC项目测试选项概览")
    print("=" * 80)
    
    print("\n🚀 **推荐：标签噪声鲁棒性测试**")
    print("-" * 50)
    
    print("1️⃣  **快速验证测试** (3-5分钟) - 🎛️ 参数化支持")
    print("   命令: python run_quick_robustness_test.py")
    print("   数据集: 4个小数据集 (2,694样本)")
    print("   默认噪声: 0%, 10%, 20% (3个水平)")
    print("   用途: 快速验证方法是否正常工作")
    print("   适合: 初次测试、调试代码、原型验证")
    print("   参数化示例:")
    print("     • 自定义噪声: --noise-levels 0.0 0.05")
    print("     • 调整网络: --representation-dim 256")
    print("     • 更少轮数: --epochs 50")
    
    print("\n2️⃣  **标准鲁棒性测试** (15-25分钟) - 🎛️ 参数化支持 ⭐推荐⭐")
    print("   命令: python run_standard_robustness_test.py")
    print("   数据集: 6个数据集 (38,000样本)")
    print("   默认噪声: 0%, 5%, 10%, 15%, 20% (5个水平)")
    print("   用途: 获得可靠的鲁棒性评估结果")
    print("   适合: 论文研究、正式实验、研究报告")
    print("   参数化示例:")
    print("     • 精细噪声: --noise-levels 0.0 0.02 0.05 0.08 0.10")
    print("     • 深度网络: --feature-hidden-dims 128 64 32")
    print("     • 大表征维度: --representation-dim 256")
    
    print("\n3️⃣  **完整交互式测试** (自定义时间)")
    print("   命令: python compare_methods_outlier_robustness.py")
    print("   数据集: 可选择任意组合 (最多74,491样本)")
    print("   用途: 自定义测试配置")
    print("   适合: 特定需求、深入研究")
    
    print("\n📊 **基础性能测试**")
    print("-" * 50)
    
    print("4️⃣  **单数据集测试**")
    print("   命令: cd src/experiments && python run_experiments.py --dataset [数据集名]")
    print("   用途: 测试单个数据集的基础性能")
    print("   支持: iris, wine, breast_cancer, digits")
    
    print("\n5️⃣  **批量性能测试**")
    print("   命令: python run_all_experiments.py [--comparison]")
    print("   用途: 运行所有基础性能测试")
    
    print("\n6️⃣  **数据集加载测试**")
    print("   命令: python test_new_datasets.py")
    print("   用途: 验证数据集扩展功能")
    
    print("\n📈 **数据集信息**")
    print("-" * 50)
    
    datasets_info = [
        ("Iris", "150", "4", "3", "small", "经典平衡数据集"),
        ("Wine", "178", "13", "3", "small", "轻微不平衡"),
        ("Breast Cancer", "569", "30", "2", "small", "医疗诊断"),
        ("Optical Digits", "1,797", "64", "10", "small", "手写数字"),
        ("Digits", "1,797", "64", "10", "medium", "数字识别"),
        ("Synthetic", "5,000", "20", "5", "medium", "合成不平衡"),
        ("Covertype", "10,000", "54", "7", "medium", "森林覆盖"),
        ("Letter Recognition", "20,000", "16", "26", "medium", "26类字母"),
        ("MNIST", "15,000", "784", "10", "large", "手写数字图像"),
        ("Fashion-MNIST", "20,000", "784", "10", "large", "服装图像")
    ]
    
    print(f"{'数据集':<20} {'样本':<8} {'特征':<5} {'类别':<5} {'规模':<8} 特点")
    print("-" * 70)
    for name, samples, features, classes, size, desc in datasets_info:
        print(f"{name:<20} {samples:<8} {features:<5} {classes:<5} {size:<8} {desc}")
    
    print(f"\n总计: 74,491样本 across 10个多样化数据集")
    
    print("\n🎯 **测试配置说明**")
    print("-" * 50)
    print("• 噪声水平: 0%, 5%, 10%, 15%, 20% (模拟真实标签错误)")
    print("• 数据分割: 70% train / 15% val / 15% test (创新分割策略)")
    print("• 对比方法: CAAC(Cauchy), CAAC(Gaussian), MLP(Softmax), MLP(OvR), MLP(Hinge)")
    print("• 噪声策略: Proportional (按类别比例注入，保持统计特性)")
    print("• 评估指标: 准确率、F1分数、训练时间、鲁棒性得分")
    
    print("\n📊 **输出文件**")
    print("-" * 50)
    print("• 详细报告: results/caac_outlier_robustness_report_[时间戳].md")
    print("• 鲁棒性曲线: results/caac_outlier_robustness_curves.png")
    print("• 鲁棒性热力图: results/caac_outlier_robustness_heatmap.png")
    print("• 原始数据: results/caac_outlier_robustness_detailed_[时间戳].csv")
    print("• 汇总数据: results/caac_outlier_robustness_summary_[时间戳].csv")
    
    print("\n🎛️ **参数化功能** (NEW!)")
    print("-" * 50)
    print("两个主要脚本现在支持完全参数化配置，无需修改代码：")
    print("")
    print("📋 **可调参数:**")
    print("• 噪声水平: --noise-levels 0.0 0.05 0.10 ...")
    print("• 表征维度: --representation-dim 64/128/256")
    print("• 特征网络: --feature-hidden-dims 64 或 128 64 32")
    print("• 推断网络: --abduction-hidden-dims 128 64 或 256 128 64")
    print("• 训练参数: --batch-size 32/64/128, --epochs 50/100/200")
    print("• 学习率: --learning-rate 0.0001/0.001/0.01")
    print("• 数据集选择: --datasets iris wine breast_cancer ...")
    print("")
    print("💡 **使用示例:**")
    print("• 快速原型: python run_quick_robustness_test.py --noise-levels 0.0 0.20 --epochs 50")
    print("• 深度网络: python run_standard_robustness_test.py --representation-dim 256 --feature-hidden-dims 128 64")
    print("• 精细噪声: python run_standard_robustness_test.py --noise-levels 0.0 0.02 0.05 0.08 0.10")
    print("• 超参优化: python run_quick_robustness_test.py --learning-rate 0.0001 --batch-size 128")
    print("")
    print("📖 **详细使用指南:** 查看 parameter_usage_guide.md")
    
    print("\n💡 **使用建议**")
    print("-" * 50)
    print("• 首次使用: 运行快速验证测试 (5分钟)")
    print("• 研究论文: 运行标准鲁棒性测试 (25分钟)")
    print("• 参数调优: 使用参数化功能快速迭代")
    print("• 深入分析: 使用交互式测试自定义配置")
    print("• 调试问题: 先运行数据集加载测试")
    
    print("\n🔧 **环境准备**")
    print("-" * 50)
    print("1. conda activate base")
    print("2. pip install torch scikit-learn matplotlib pandas numpy seaborn")
    print("3. 确保网络连接正常 (某些数据集需要在线下载)")
    
    print("\n" + "=" * 80)
    print("💬 需要更多帮助？查看 README.md 或运行具体脚本获取详细说明")
    print("=" * 80)

if __name__ == "__main__":
    show_test_options() 