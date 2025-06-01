#!/usr/bin/env python3
"""
CAAC鲁棒性快速测试 - 参数化5分钟验证版本
支持自定义噪声水平和网络结构参数，只使用小规模数据集进行快速验证
"""

import sys
import argparse
sys.path.append('src')

def run_quick_robustness_test(
    noise_levels=None,
    representation_dim=128,
    feature_hidden_dims=None,
    abduction_hidden_dims=None,
    batch_size=64,
    epochs=100,  # 快速测试使用较少的epochs
    learning_rate=0.001,
    early_stopping_patience=10,
    datasets=None
):
    """
    运行参数化的快速鲁棒性测试
    
    Args:
        noise_levels: List[float] - 噪声水平列表，如 [0.0, 0.10, 0.20]
        representation_dim: int - 表征维度，默认128
        feature_hidden_dims: List[int] - 特征网络隐藏层维度，默认[64]
        abduction_hidden_dims: List[int] - 推断网络隐藏层维度，默认[128, 64]
        batch_size: int - 批量大小，默认64
        epochs: int - 训练轮数，默认100（快速测试）
        learning_rate: float - 学习率，默认0.001
        early_stopping_patience: int - 早停耐心值，默认10（快速测试）
        datasets: List[str] - 数据集列表，默认使用小规模数据集
    """
    # 设置默认参数（快速测试配置）
    if noise_levels is None:
        noise_levels = [0.0, 0.10, 0.20]  # 快速测试只用3个噪声水平
    if feature_hidden_dims is None:
        feature_hidden_dims = [64]
    if abduction_hidden_dims is None:
        abduction_hidden_dims = [128, 64]
    if datasets is None:
        datasets = ['iris', 'wine', 'breast_cancer', 'optical_digits']  # 只用小数据集
    
    print("🚀 CAAC方法Outlier鲁棒性参数化快速测试")
    print("=" * 60)
    print("📊 测试配置:")
    print(f"  • 数据集: 仅小规模数据集 ({len(datasets)}个)")
    print("  • 方法: CAAC(Cauchy), CAAC(Gaussian), MLP(Softmax), MLP(OvR), MLP(Hinge)")
    print(f"  • 噪声水平: {[f'{x:.1%}' for x in noise_levels]}")
    print("  • 数据分割: 70% train / 15% val / 15% test")
    print("📈 网络结构:")
    print(f"  • 表征维度: {representation_dim}")
    print(f"  • 特征网络隐藏层: {feature_hidden_dims}")
    print(f"  • 推断网络隐藏层: {abduction_hidden_dims}")
    print("⚙️ 训练参数 (快速配置):")
    print(f"  • 批量大小: {batch_size}")
    print(f"  • 训练轮数: {epochs}")
    print(f"  • 学习率: {learning_rate}")
    print(f"  • 早停耐心值: {early_stopping_patience}")
    print(f"  • 预计时间: 3-5分钟")
    print("=" * 60)
    
    # 加载数据集并显示选择
    from run_standard_robustness_test import run_parameterized_outlier_robustness_experiments
    from compare_methods_outlier_robustness import load_datasets
    from compare_methods_outlier_robustness import create_robustness_visualizations, create_robustness_heatmap
    from compare_methods_outlier_robustness import analyze_robustness_results, generate_robustness_report
    
    print("\n📊 加载数据集...")
    all_datasets = load_datasets()
    
    # 过滤选择的数据集
    filtered_datasets = {k: v for k, v in all_datasets.items() if k in datasets}
    
    print(f"\n🎯 快速测试将使用以下{len(filtered_datasets)}个数据集:")
    total_samples = 0
    for key, dataset in filtered_datasets.items():
        n_samples, n_features = dataset['data'].shape
        n_classes = len(set(dataset['target']))
        size_label = dataset.get('size', 'unknown')
        total_samples += n_samples
        print(f"  • {dataset['name']}: {n_samples:,}样本, {n_features}特征, {n_classes}类 [{size_label}]")
    
    print(f"\n📈 总计: {total_samples:,}样本 across {len(filtered_datasets)}个数据集")
    
    # 确认开始
    print(f"\n⚠️  注意: 这将运行 {len(filtered_datasets)} × {len(noise_levels)}噪声水平 × 5方法 = {len(filtered_datasets)*len(noise_levels)*5} 个实验")
    
    import time
    print("\n⏰ 实验将在3秒后自动开始...")
    for i in range(3, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    
    print("\n🚀 开始快速测试!")
    print("=" * 60)
    
    try:
        # 运行实验，使用快速配置
        results_df = run_parameterized_outlier_robustness_experiments(
            filtered_datasets,
            noise_levels=noise_levels,
            representation_dim=representation_dim,
            feature_hidden_dims=feature_hidden_dims,
            abduction_hidden_dims=abduction_hidden_dims,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            early_stopping_patience=early_stopping_patience
        )
        
        # 创建可视化
        print("\n📈 生成可视化...")
        create_robustness_visualizations(results_df)
        create_robustness_heatmap(results_df)
        
        # 分析结果
        print("\n🔍 分析结果...")
        robustness_df = analyze_robustness_results(results_df)
        
        # 生成报告
        print("\n📄 生成报告...")
        report_file = generate_robustness_report(results_df, robustness_df)
        
        print("\n" + "=" * 60)
        print("🎉 快速测试完成！")
        print("=" * 60)
        print("📊 生成的文件:")
        print(f"  • 详细报告: {report_file}")
        print("  • 鲁棒性曲线: results/caac_outlier_robustness_curves.png")
        print("  • 鲁棒性热力图: results/caac_outlier_robustness_heatmap.png")
        
        # 显示关键发现
        print("\n🔍 关键发现预览:")
        print(f"  • 最鲁棒方法: {robustness_df.iloc[0]['Method']}")
        print(f"  • 鲁棒性得分: {robustness_df.iloc[0]['Overall_Robustness']:.4f}")
        print(f"  • 性能衰减: {robustness_df.iloc[0]['Performance_Drop']:.1f}%")
        
        print(f"\n💡 这是基于{total_samples:,}样本的快速验证结果")
        print("   如需更可靠的结论，请运行标准测试: python run_standard_robustness_test.py")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description='CAAC鲁棒性参数化快速测试')
    
    # 噪声水平参数（快速测试默认值）
    parser.add_argument('--noise-levels', nargs='+', type=float, 
                       default=[0.0, 0.10, 0.20],
                       help='噪声水平列表 (默认: 0.0 0.10 0.20)')
    
    # 网络结构参数
    parser.add_argument('--representation-dim', type=int, default=128,
                       help='表征维度 (默认: 128)')
    parser.add_argument('--feature-hidden-dims', nargs='+', type=int, default=[64],
                       help='特征网络隐藏层维度 (默认: 64)')
    parser.add_argument('--abduction-hidden-dims', nargs='+', type=int, default=[128, 64],
                       help='推断网络隐藏层维度 (默认: 128 64)')
    
    # 训练参数（快速测试配置）
    parser.add_argument('--batch-size', type=int, default=64,
                       help='批量大小 (默认: 64)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数 (默认: 100)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='学习率 (默认: 0.001)')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                       help='早停耐心值 (默认: 10)')
    
    # 数据集选择（快速测试配置）
    parser.add_argument('--datasets', nargs='+', 
                       default=['iris', 'wine', 'breast_cancer', 'optical_digits'],
                       help='数据集列表 (默认: 使用小规模数据集)')
    
    args = parser.parse_args()
    
    # 验证参数
    for noise in args.noise_levels:
        if not (0.0 <= noise <= 1.0):
            print(f"❌ 错误: 噪声水平 {noise} 必须在 [0.0, 1.0] 范围内")
            sys.exit(1)
    
    if args.representation_dim <= 0:
        print("❌ 错误: 表征维度必须为正整数")
        sys.exit(1)
    
    if any(dim <= 0 for dim in args.feature_hidden_dims):
        print("❌ 错误: 特征网络隐藏层维度必须为正整数")
        sys.exit(1)
    
    if any(dim <= 0 for dim in args.abduction_hidden_dims):
        print("❌ 错误: 推断网络隐藏层维度必须为正整数")
        sys.exit(1)
    
    print("🎯 快速测试使用参数:")
    print(f"  • 噪声水平: {args.noise_levels}")
    print(f"  • 表征维度: {args.representation_dim}")
    print(f"  • 特征网络隐藏层: {args.feature_hidden_dims}")
    print(f"  • 推断网络隐藏层: {args.abduction_hidden_dims}")
    print(f"  • 批量大小: {args.batch_size}")
    print(f"  • 训练轮数: {args.epochs}")
    print(f"  • 学习率: {args.learning_rate}")
    print(f"  • 早停耐心值: {args.early_stopping_patience}")
    print(f"  • 数据集: {args.datasets}")
    print()
    
    success = run_quick_robustness_test(
        noise_levels=args.noise_levels,
        representation_dim=args.representation_dim,
        feature_hidden_dims=args.feature_hidden_dims,
        abduction_hidden_dims=args.abduction_hidden_dims,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        datasets=args.datasets
    )
    
    if not success:
        print("\n💡 如遇到问题，请检查:")
        print("  1. 是否在正确的conda环境 (conda activate base)")
        print("  2. 是否安装了所有依赖包")
        print("  3. 网络连接是否正常 (用于下载某些数据集)")
        print("  4. 参数设置是否合理")
        sys.exit(1)


if __name__ == "__main__":
    # 如果没有命令行参数，使用默认参数运行
    if len(sys.argv) == 1:
        print("🚀 使用快速测试默认参数运行...")
        success = run_quick_robustness_test()
        if not success:
            print("\n💡 如遇到问题，请检查:")
            print("  1. 是否在正确的conda环境 (conda activate base)")
            print("  2. 是否安装了所有依赖包")
            print("  3. 网络连接是否正常 (用于下载某些数据集)")
            sys.exit(1)
    else:
        main() 