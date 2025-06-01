"""
🚀 CAAC项目基础使用示例

本文件展示CAAC项目的基本使用方法，包括：
- 模型训练和预测
- 实验运行
- 结果可视化

请在项目根目录运行：python examples/basic_usage.py
"""

import sys
import os
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.caac_ovr_model import CAACOvRModel
from src.experiments.experiment_manager import ExperimentManager
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


def example_1_direct_model_usage():
    """
    示例1: 直接使用CAAC模型
    """
    print("=" * 60)
    print("🎯 示例1: 直接使用CAAC模型")
    print("=" * 60)
    
    # 1. 加载数据
    print("📊 加载Iris数据集...")
    data = load_iris()
    X, y = data.data, data.target
    
    # 2. 数据预处理
    print("🔧 数据预处理...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. 创建和训练模型
    print("🧠 创建CAAC模型...")
    model = CAACOvRModel(
        input_dim=X_train_scaled.shape[1],
        representation_dim=32,
        latent_dim=16,
        n_classes=len(np.unique(y_train)),
        feature_hidden_dims=[32],
        abduction_hidden_dims=[32, 16],
        lr=0.01,
        batch_size=16,
        epochs=50,
        early_stopping_patience=10
    )
    
    print("🔥 开始训练...")
    history = model.fit(X_train_scaled, y_train, verbose=1)
    
    # 4. 模型预测
    print("🎯 进行预测...")
    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)
    
    # 5. 评估结果
    accuracy = accuracy_score(y_test, predictions)
    print(f"✅ 测试准确率: {accuracy:.4f}")
    print("\n📊 详细分类报告:")
    print(classification_report(y_test, predictions, target_names=data.target_names))
    
    # 6. 简单可视化
    if hasattr(plt, 'figure'):
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_losses'], label='Training Loss')
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_accuracies'], label='Training Accuracy')
        plt.title('Training Accuracy Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('basic_usage_training_history.png', dpi=300, bbox_inches='tight')
        print("📈 训练历史已保存为 'basic_usage_training_history.png'")
        plt.close()
    
    return model, accuracy


def example_2_experiment_manager():
    """
    示例2: 使用实验管理器
    """
    print("\n" + "=" * 60)
    print("🔬 示例2: 使用实验管理器")
    print("=" * 60)
    
    # 1. 创建实验管理器
    print("📋 创建实验管理器...")
    manager = ExperimentManager(base_results_dir="examples/results")
    
    # 2. 查看可用实验
    print("📝 可用实验类型:")
    available_experiments = manager.list_available_experiments()
    for exp in available_experiments:
        print(f"  - {exp}")
    
    # 3. 运行快速实验
    print("\n🚀 运行快速鲁棒性测试...")
    try:
        result_dir = manager.run_quick_robustness_test(
            datasets=['iris', 'wine'],  # 限制数据集以加快速度
            epochs=30,  # 减少训练轮数
            noise_levels=[0.0, 0.1, 0.2]  # 减少噪声水平
        )
        print(f"✅ 实验完成! 结果保存在: {result_dir}")
        
        # 4. 创建实验总结
        summary = manager.create_experiment_summary(result_dir)
        print("\n📊 实验总结:")
        print(f"  - 生成文件数: {len(summary.get('files', []))}")
        print(f"  - 实验时间: {summary.get('timestamp', 'N/A')}")
        
        # 5. 显示生成的文件
        if summary.get('files'):
            print("  - 生成的文件:")
            for file in summary['files'][:5]:  # 只显示前5个文件
                print(f"    • {file}")
            if len(summary['files']) > 5:
                print(f"    • ... 还有 {len(summary['files']) - 5} 个文件")
        
    except Exception as e:
        print(f"❌ 实验运行失败: {e}")
        print("请检查实验模块是否正确配置")
    
    return manager


def example_3_custom_configuration():
    """
    示例3: 自定义配置
    """
    print("\n" + "=" * 60)
    print("⚙️ 示例3: 自定义配置")
    print("=" * 60)
    
    # 1. 加载不同数据集
    print("📊 加载Wine数据集...")
    data = load_wine()
    X, y = data.data, data.target
    
    # 2. 创建高级配置的模型
    print("🧠 创建高级配置模型...")
    advanced_model = CAACOvRModel(
        input_dim=X.shape[1],
        representation_dim=64,
        latent_dim=32,
        n_classes=len(np.unique(y)),
        feature_hidden_dims=[128, 64],
        abduction_hidden_dims=[64, 32],
        lr=0.001,
        batch_size=32,
        epochs=100,
        learnable_thresholds=True,  # 启用可学习阈值
        uniqueness_constraint=True,  # 启用唯一性约束
        uniqueness_weight=0.1,
        early_stopping_patience=15
    )
    
    # 3. 数据预处理和分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. 训练模型
    print("🔥 训练高级配置模型...")
    history = advanced_model.fit(X_train_scaled, y_train, verbose=1)
    
    # 5. 评估性能
    predictions = advanced_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"✅ 高级模型准确率: {accuracy:.4f}")
    
    # 6. 参数探索
    print("\n🔧 模型参数:")
    params = advanced_model.get_params()
    key_params = ['representation_dim', 'latent_dim', 'lr', 'learnable_thresholds', 'uniqueness_constraint']
    for param in key_params:
        if param in params:
            print(f"  - {param}: {params[param]}")
    
    return advanced_model, accuracy


def example_4_comparison_multiple_models():
    """
    示例4: 多模型对比
    """
    print("\n" + "=" * 60)
    print("📊 示例4: 多模型对比")
    print("=" * 60)
    
    # 1. 准备数据
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. 定义不同配置的模型
    models = {
        'CAAC Basic': CAACOvRModel(
            input_dim=X.shape[1], n_classes=len(np.unique(y)),
            representation_dim=32, epochs=30
        ),
        'CAAC + Learnable Thresholds': CAACOvRModel(
            input_dim=X.shape[1], n_classes=len(np.unique(y)),
            representation_dim=32, epochs=30, learnable_thresholds=True
        ),
        'CAAC + Uniqueness': CAACOvRModel(
            input_dim=X.shape[1], n_classes=len(np.unique(y)),
            representation_dim=32, epochs=30, uniqueness_constraint=True
        )
    }
    
    # 3. 训练和评估所有模型
    results = {}
    for name, model in models.items():
        print(f"\n🔥 训练 {name}...")
        history = model.fit(X_train_scaled, y_train, verbose=0)
        predictions = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        results[name] = accuracy
        print(f"✅ {name} 准确率: {accuracy:.4f}")
    
    # 4. 显示对比结果
    print("\n" + "=" * 40)
    print("📊 模型对比结果")
    print("=" * 40)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, accuracy) in enumerate(sorted_results, 1):
        print(f"{rank}. {name:<25}: {accuracy:.4f}")
    
    return results


def main():
    """
    运行所有示例
    """
    print("🚀 CAAC项目基础使用示例")
    print("=" * 60)
    print("本示例将演示CAAC项目的基本使用方法")
    print()
    
    try:
        # 示例1: 直接模型使用
        model1, acc1 = example_1_direct_model_usage()
        
        # 示例2: 实验管理器
        manager = example_2_experiment_manager()
        
        # 示例3: 自定义配置
        model3, acc3 = example_3_custom_configuration()
        
        # 示例4: 多模型对比
        comparison_results = example_4_comparison_multiple_models()
        
        # 总结
        print("\n" + "=" * 60)
        print("🎊 所有示例运行完成!")
        print("=" * 60)
        print(f"📊 直接模型使用准确率: {acc1:.4f}")
        print(f"⚙️ 高级配置模型准确率: {acc3:.4f}")
        print(f"🏆 最佳对比模型: {max(comparison_results.items(), key=lambda x: x[1])[0]}")
        print()
        print("💡 下一步建议:")
        print("  1. 查看 docs/api/ 获取详细API文档")
        print("  2. 运行 python run_experiments.py --quick 进行快速实验")
        print("  3. 参考 examples/custom_experiment.py 学习自定义实验")
        
    except Exception as e:
        print(f"❌ 运行示例时出错: {e}")
        print("请检查环境配置和依赖安装")


if __name__ == "__main__":
    # 确保结果目录存在
    os.makedirs("examples/results", exist_ok=True)
    
    # 运行示例
    main() 