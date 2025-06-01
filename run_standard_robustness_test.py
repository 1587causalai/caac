#!/usr/bin/env python3
"""
CAAC鲁棒性标准测试 - 参数化快速启动脚本
支持自定义噪声水平和网络结构参数
"""

import sys
import argparse
sys.path.append('src')

def run_standard_robustness_test(
    noise_levels=None,
    representation_dim=128,
    feature_hidden_dims=None,
    abduction_hidden_dims=None,
    batch_size=64,
    epochs=150,
    learning_rate=0.001,
    early_stopping_patience=15,
    datasets=None
):
    """
    运行参数化的标准鲁棒性测试
    
    Args:
        noise_levels: List[float] - 噪声水平列表，如 [0.0, 0.05, 0.10, 0.15, 0.20]
        representation_dim: int - 表征维度，默认128
        feature_hidden_dims: List[int] - 特征网络隐藏层维度，默认[64]
        abduction_hidden_dims: List[int] - 推断网络隐藏层维度，默认[128, 64]
        batch_size: int - 批量大小，默认64
        epochs: int - 训练轮数，默认150
        learning_rate: float - 学习率，默认0.001
        early_stopping_patience: int - 早停耐心值，默认15
        datasets: List[str] - 数据集列表，默认使用标准数据集
    """
    # 设置默认参数
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    if feature_hidden_dims is None:
        feature_hidden_dims = [64]
    if abduction_hidden_dims is None:
        abduction_hidden_dims = [128, 64]
    if datasets is None:
        datasets = ['breast_cancer', 'optical_digits', 'digits', 'synthetic_imbalanced', 'covertype', 'letter']
    
    print("🧪 CAAC方法Outlier鲁棒性参数化测试")
    print("=" * 60)
    print("📊 测试配置:")
    print(f"  • 数据集: 1个小数据集 + 5个中等规模数据集 (共{len(datasets)}个数据集)")
    print("  • 方法: CAAC(Cauchy), CAAC(Gaussian), MLP(Softmax), MLP(OvR), MLP(Hinge)")
    print(f"  • 噪声水平: {[f'{x:.1%}' for x in noise_levels]}")
    print("  • 数据分割: 70% train / 15% val / 15% test")
    print("📈 网络结构:")
    print(f"  • 表征维度: {representation_dim}")
    print(f"  • 特征网络隐藏层: {feature_hidden_dims}")
    print(f"  • 推断网络隐藏层: {abduction_hidden_dims}")
    print("⚙️ 训练参数:")
    print(f"  • 批量大小: {batch_size}")
    print(f"  • 训练轮数: {epochs}")
    print(f"  • 学习率: {learning_rate}")
    print(f"  • 早停耐心值: {early_stopping_patience}")
    print(f"  • 预计时间: 15-25分钟")
    print("=" * 60)
    
    # 加载数据集并显示选择
    from compare_methods_outlier_robustness import load_datasets, create_robust_comparison_methods
    from compare_methods_outlier_robustness import create_robustness_visualizations, create_robustness_heatmap
    from compare_methods_outlier_robustness import analyze_robustness_results, generate_robustness_report
    
    print("\n📊 加载数据集...")
    all_datasets = load_datasets()
    
    # 过滤选择的数据集
    filtered_datasets = {k: v for k, v in all_datasets.items() if k in datasets}
    
    print(f"\n🎯 标准测试将使用以下{len(filtered_datasets)}个数据集:")
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
    print("\n⏰ 实验将在5秒后自动开始...")
    for i in range(5, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    
    print("\n🚀 开始参数化测试!")
    print("=" * 60)
    
    try:
        # 运行实验，传递参数化配置
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
        print("🎉 参数化测试完成！")
        print("=" * 60)
        print("📊 生成的文件:")
        print(f"  • 详细报告: {report_file}")
        print("  • 鲁棒性曲线: results/caac_outlier_robustness_curves.png")
        print("  • 鲁棒性热力图: results/caac_outlier_robustness_heatmap.png")
        print("  • 原始数据: results/caac_outlier_robustness_detailed_*.csv")
        print("  • 汇总数据: results/caac_outlier_robustness_summary_*.csv")
        
        # 显示关键发现
        print("\n🔍 关键发现预览:")
        print(f"  • 最鲁棒方法: {robustness_df.iloc[0]['Method']}")
        print(f"  • 鲁棒性得分: {robustness_df.iloc[0]['Overall_Robustness']:.4f}")
        print(f"  • 性能衰减: {robustness_df.iloc[0]['Performance_Drop']:.1f}%")
        
        print("\n📖 查看完整报告获取详细分析和结论")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_parameterized_outlier_robustness_experiments(
    datasets,
    noise_levels,
    representation_dim,
    feature_hidden_dims,
    abduction_hidden_dims,
    batch_size,
    epochs,
    learning_rate,
    early_stopping_patience
):
    """运行参数化的outlier鲁棒性对比实验"""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    from src.models.caac_ovr_model import (
        CAACOvRModel, 
        SoftmaxMLPModel,
        OvRCrossEntropyMLPModel,
        CAACOvRGaussianModel,
        CrammerSingerMLPModel
    )
    from src.data.data_processor import DataProcessor
    from compare_methods_outlier_robustness import evaluate_method_with_outliers
    
    print("🔬 开始运行参数化outlier鲁棒性对比实验")
    print("包含方法: CAAC(Cauchy), CAAC(Gaussian), MLP(Softmax), MLP(OvR), MLP(Hinge)")
    print("=" * 80)
    
    # 创建参数化的方法配置
    methods = create_parameterized_comparison_methods(
        representation_dim=representation_dim,
        feature_hidden_dims=feature_hidden_dims,
        abduction_hidden_dims=abduction_hidden_dims,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience
    )
    
    results = []
    
    for dataset_name, dataset in datasets.items():
        print(f"\n📊 正在测试数据集: {dataset['name']}")
        print("-" * 50)
        
        # 数据预处理
        X = dataset['data']
        y = dataset['target']
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 显示原始类别分布
        unique, counts = np.unique(y, return_counts=True)
        print(f"原始类别分布: {dict(zip(unique, counts))}")
        
        for outlier_ratio in noise_levels:
            print(f"\n  🎯 测试outlier比例: {outlier_ratio:.1%}")
            
            # 使用新的数据分割策略
            if outlier_ratio > 0:
                result = DataProcessor.split_classification_data_with_outliers(
                    X_scaled, y,
                    train_size=0.7, val_size=0.15, test_size=0.15,
                    outlier_ratio=outlier_ratio, balance_strategy='proportional',
                    random_state=42
                )
                X_train, X_val, X_test, y_train, y_val, y_test, outlier_info = result
                
                print(f"    Outliers添加: Train={outlier_info['outliers_in_train']}, Val={outlier_info['outliers_in_val']}")
            else:
                # 无outliers的基线
                X_train, X_val, X_test, y_train, y_val, y_test = DataProcessor.split_data(
                    X_scaled, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42
                )
                print(f"    基线实验 (无outliers)")
            
            # 测试每种方法
            for method_key, method_info in methods.items():
                print(f"    🧪 测试方法: {method_info['name']}")
                
                try:
                    metrics = evaluate_method_with_outliers(
                        method_info, 
                        X_train, X_val, X_test, 
                        y_train, y_val, y_test
                    )
                    
                    results.append({
                        'Dataset': dataset['name'],
                        'Dataset_Key': dataset_name,
                        'Outlier_Ratio': outlier_ratio,
                        'Method': method_info['name'],
                        'Method_Key': method_key,
                        'Method_Type': method_info['type'],
                        'Description': method_info.get('description', ''),
                        'Accuracy': metrics['accuracy'],
                        'Precision_Macro': metrics['precision_macro'],
                        'Recall_Macro': metrics['recall_macro'],
                        'F1_Macro': metrics['f1_macro'],
                        'F1_Weighted': metrics['f1_weighted'],
                        'Training_Time': metrics['training_time']
                    })
                    
                    print(f"      ✅ 准确率: {metrics['accuracy']:.4f}, F1: {metrics['f1_macro']:.4f}")
                    
                except Exception as e:
                    print(f"      ❌ 错误: {str(e)}")
                    continue
    
    return pd.DataFrame(results)


def create_parameterized_comparison_methods(
    representation_dim,
    feature_hidden_dims,
    abduction_hidden_dims,
    batch_size,
    epochs,
    learning_rate,
    early_stopping_patience
):
    """创建参数化的用于鲁棒性比较的方法"""
    from src.models.caac_ovr_model import (
        CAACOvRModel, 
        SoftmaxMLPModel,
        OvRCrossEntropyMLPModel,
        CAACOvRGaussianModel,
        CrammerSingerMLPModel
    )
    
    # 参数化的基础网络架构参数
    base_params = {
        'representation_dim': representation_dim,
        'latent_dim': None,  # 默认等于representation_dim
        'feature_hidden_dims': feature_hidden_dims,
        'abduction_hidden_dims': abduction_hidden_dims,
        'lr': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'device': None,
        'early_stopping_patience': early_stopping_patience,
        'early_stopping_min_delta': 0.0001
    }
    
    # CAAC模型专用参数（包含额外的鲁棒性参数）
    caac_params = {
        **base_params,
        'learnable_thresholds': False,
        'uniqueness_constraint': False
    }
    
    # 标准MLP模型参数（不包含CAAC特有参数）
    mlp_params = base_params.copy()
    
    methods = {
        # 核心方法对比 - 根据用户要求选择的5种方法
        'CAAC_Cauchy': {
            'name': 'CAAC OvR (Cauchy)',
            'type': 'unified',
            'model_class': CAACOvRModel,
            'params': caac_params,
            'description': f'柯西分布 + 固定阈值 (表征维度:{representation_dim})'
        },
        'CAAC_Gaussian': {
            'name': 'CAAC OvR (Gaussian)',
            'type': 'unified',
            'model_class': CAACOvRGaussianModel,
            'params': caac_params,
            'description': f'高斯分布 + 固定阈值 (表征维度:{representation_dim})'
        },
        'MLP_Softmax': {
            'name': 'MLP (Softmax)',
            'type': 'unified',
            'model_class': SoftmaxMLPModel,
            'params': mlp_params,
            'description': f'标准多层感知机 (表征维度:{representation_dim})'
        },
        'MLP_OvR_CE': {
            'name': 'MLP (OvR Cross Entropy)',
            'type': 'unified',
            'model_class': OvRCrossEntropyMLPModel,
            'params': mlp_params,
            'description': f'OvR策略 (表征维度:{representation_dim})'
        },
        'MLP_Hinge': {
            'name': 'MLP (Crammer & Singer Hinge)',
            'type': 'unified',
            'model_class': CrammerSingerMLPModel,
            'params': mlp_params,
            'description': f'铰链损失 (表征维度:{representation_dim})'
        }
    }
    return methods


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description='CAAC鲁棒性参数化测试')
    
    # 噪声水平参数
    parser.add_argument('--noise-levels', nargs='+', type=float, 
                       default=[0.0, 0.05, 0.10, 0.15, 0.20],
                       help='噪声水平列表 (默认: 0.0 0.05 0.10 0.15 0.20)')
    
    # 网络结构参数
    parser.add_argument('--representation-dim', type=int, default=128,
                       help='表征维度 (默认: 128)')
    parser.add_argument('--feature-hidden-dims', nargs='+', type=int, default=[64],
                       help='特征网络隐藏层维度 (默认: 64)')
    parser.add_argument('--abduction-hidden-dims', nargs='+', type=int, default=[128, 64],
                       help='推断网络隐藏层维度 (默认: 128 64)')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=64,
                       help='批量大小 (默认: 64)')
    parser.add_argument('--epochs', type=int, default=150,
                       help='训练轮数 (默认: 150)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='学习率 (默认: 0.001)')
    parser.add_argument('--early-stopping-patience', type=int, default=15,
                       help='早停耐心值 (默认: 15)')
    
    # 数据集选择
    parser.add_argument('--datasets', nargs='+', 
                       default=['breast_cancer', 'optical_digits', 'digits', 'synthetic_imbalanced', 'covertype', 'letter'],
                       help='数据集列表 (默认: 使用标准数据集)')
    
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
    
    print("🎯 使用参数:")
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
    
    success = run_standard_robustness_test(
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
        print("🚀 使用默认参数运行...")
        success = run_standard_robustness_test()
        if not success:
            print("\n💡 如遇到问题，请检查:")
            print("  1. 是否在正确的conda环境 (conda activate base)")
            print("  2. 是否安装了所有依赖包")
            print("  3. 网络连接是否正常 (用于下载某些数据集)")
            sys.exit(1)
    else:
        main() 