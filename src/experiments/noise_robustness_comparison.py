"""
噪声鲁棒性对比实验

比较CAAC模型在不同类型标签噪声下的鲁棒性表现。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
from datetime import datetime
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from experiments.robustness_experiments import RobustnessExperimentRunner
from data.data_processor import DataProcessor

class NoiseRobustnessComparison:
    """噪声鲁棒性对比实验类"""
    
    def __init__(self, results_dir="results/noise_comparison"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 噪声类型配置
        self.noise_types = {
            'random_uniform': 'Random Uniform',
            'proportional': 'Proportional',
            'majority_bias': 'Majority Bias',
            'minority_bias': 'Minority Bias',
            'adjacent': 'Adjacent',
            'flip_pairs': 'Flip Pairs'
        }
        
    def run_noise_comparison_experiment(self, 
                                      datasets=None,
                                      noise_levels=None,
                                      noise_types=None,
                                      representation_dim=128,
                                      epochs=100):
        """
        运行噪声对比实验
        
        Args:
            datasets: 要测试的数据集列表
            noise_levels: 噪声水平列表
            noise_types: 噪声类型列表
            representation_dim: 表示维度
            epochs: 训练轮数
        """
        
        if datasets is None:
            datasets = ['iris', 'wine', 'breast_cancer']
            
        if noise_levels is None:
            noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
            
        if noise_types is None:
            noise_types = list(self.noise_types.keys())
            
        print("🔬 Noise Robustness Comparison Experiment")
        print("=" * 60)
        print(f"📊 Testing {len(datasets)} datasets")
        print(f"🔧 Noise types: {[self.noise_types[nt] for nt in noise_types]}")
        print(f"📈 Noise levels: {noise_levels}")
        print(f"⚙️  Representation dim: {representation_dim}, Epochs: {epochs}")
        print()
        
        # 初始化实验运行器
        exp_runner = RobustnessExperimentRunner()
        
        # 存储所有结果
        all_results = []
        
        # 测试每个数据集
        for dataset_name in datasets:
            print(f"📁 Processing dataset: {dataset_name}")
            
            # 加载数据集
            X, y, target_names, display_name = exp_runner.load_dataset(dataset_name)
            
            # 数据预处理
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 测试每种噪声类型
            for noise_type in noise_types:
                print(f"  🧪 Testing noise type: {self.noise_types[noise_type]}")
                
                # 测试每个噪声水平
                for noise_level in noise_levels:
                    try:
                        # 注入噪声
                        y_train_noisy, noise_info = DataProcessor.inject_label_noise(
                            y_train, noise_level, noise_type, random_state=42
                        )
                        
                        # 设置模型参数
                        from models.caac_models import CAACOvRModel
                        
                        model_params = {
                            'input_dim': X_train_scaled.shape[1],
                            'n_classes': len(np.unique(y)),
                            'representation_dim': representation_dim,
                            'epochs': epochs,
                            'lr': 0.001,
                            'batch_size': 32,
                            'early_stopping_patience': 10
                        }
                        
                        # 训练模型
                        start_time = time.time()
                        model = CAACOvRModel(**model_params)
                        model.fit(X_train_scaled, y_train_noisy, verbose=0)
                        training_time = time.time() - start_time
                        
                        # 评估模型
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                        
                        y_pred = model.predict(X_test_scaled)
                        
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, y_pred, average='weighted')
                        f1 = f1_score(y_test, y_pred, average='weighted')
                        
                        # 记录结果
                        result = {
                            'dataset': display_name,
                            'dataset_key': dataset_name,
                            'noise_type': noise_type,
                            'noise_type_display': self.noise_types[noise_type],
                            'noise_level': noise_level,
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'training_time': training_time,
                            'samples_changed': noise_info['changes'],
                            'total_samples': len(y_train)
                        }
                        
                        all_results.append(result)
                        
                        print(f"    📈 Noise {noise_level:.1%}: Accuracy={accuracy:.3f}, "
                              f"Changed={noise_info['changes']} samples")
                              
                    except Exception as e:
                        print(f"    ❌ Error at noise level {noise_level:.1%}: {str(e)}")
                        continue
        
        # 创建结果DataFrame
        results_df = pd.DataFrame(all_results)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"noise_comparison_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\n💾 Results saved to {results_file}")
        
        # 创建可视化
        self._create_comparison_visualizations(results_df, timestamp)
        
        # 生成分析报告
        self._generate_analysis_report(results_df, timestamp)
        
        return results_df
    
    def _create_comparison_visualizations(self, results_df, timestamp):
        """创建对比可视化图表"""
        
        print("🎨 Creating comparison visualizations...")
        
        # 1. 鲁棒性曲线对比图
        self._create_robustness_curves(results_df, timestamp)
        
        # 2. 噪声类型性能热力图
        self._create_noise_performance_heatmap(results_df, timestamp)
        
        # 3. 数据集间性能对比
        self._create_dataset_comparison(results_df, timestamp)
        
        # 4. 噪声影响统计图
        self._create_noise_impact_analysis(results_df, timestamp)
    
    def _create_robustness_curves(self, results_df, timestamp):
        """创建鲁棒性曲线图"""
        
        fig, axes = plt.subplots(1, len(results_df['dataset'].unique()), 
                                figsize=(5*len(results_df['dataset'].unique()), 6))
        
        if len(results_df['dataset'].unique()) == 1:
            axes = [axes]
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(results_df['noise_type'].unique())))
        
        for idx, dataset in enumerate(results_df['dataset'].unique()):
            ax = axes[idx]
            dataset_data = results_df[results_df['dataset'] == dataset]
            
            for i, noise_type in enumerate(dataset_data['noise_type'].unique()):
                noise_data = dataset_data[dataset_data['noise_type'] == noise_type]
                noise_data = noise_data.sort_values('noise_level')
                
                ax.plot(noise_data['noise_level'], noise_data['accuracy'], 
                       marker='o', linewidth=2, markersize=6, color=colors[i],
                       label=noise_data['noise_type_display'].iloc[0])
            
            ax.set_xlabel('Noise Level', fontsize=12)
            ax.set_ylabel('Test Accuracy', fontsize=12)
            ax.set_title(f'{dataset}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.suptitle('CAAC Model Robustness to Different Noise Types', 
                     fontsize=16, fontweight='bold', y=1.02)
        
        save_path = self.results_dir / f"robustness_curves_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  📊 Robustness curves saved to {save_path}")
        plt.show()
    
    def _create_noise_performance_heatmap(self, results_df, timestamp):
        """创建噪声类型性能热力图"""
        
        # 计算平均性能（跨数据集）
        heatmap_data = results_df.groupby(['noise_type_display', 'noise_level'])['accuracy'].mean().unstack()
        
        plt.figure(figsize=(12, 8))
        
        # 创建热力图
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r',
                   cbar_kws={'label': 'Average Accuracy'}, 
                   linewidths=0.5)
        
        plt.title('Average Performance Across Noise Types and Levels', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Noise Level', fontsize=14)
        plt.ylabel('Noise Type', fontsize=14)
        
        save_path = self.results_dir / f"noise_performance_heatmap_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  📊 Performance heatmap saved to {save_path}")
        plt.show()
    
    def _create_dataset_comparison(self, results_df, timestamp):
        """创建数据集间性能对比图"""
        
        # 计算每个数据集在不同噪声下的平均性能下降
        performance_drop = []
        
        for dataset in results_df['dataset'].unique():
            for noise_type in results_df['noise_type'].unique():
                data = results_df[(results_df['dataset'] == dataset) & 
                                (results_df['noise_type'] == noise_type)]
                
                if len(data) > 1:
                    clean_acc = data[data['noise_level'] == 0]['accuracy'].values
                    max_noise_acc = data[data['noise_level'] > 0]['accuracy'].min()
                    
                    if len(clean_acc) > 0:
                        drop = clean_acc[0] - max_noise_acc
                        performance_drop.append({
                            'dataset': dataset,
                            'noise_type': data['noise_type_display'].iloc[0],
                            'performance_drop': drop
                        })
        
        drop_df = pd.DataFrame(performance_drop)
        
        if not drop_df.empty:
            plt.figure(figsize=(12, 8))
            
            # 创建分组条形图
            pivot_data = drop_df.pivot(index='dataset', columns='noise_type', values='performance_drop')
            
            ax = pivot_data.plot(kind='bar', figsize=(12, 8), width=0.8)
            
            plt.title('Performance Drop by Dataset and Noise Type', 
                      fontsize=16, fontweight='bold')
            plt.xlabel('Dataset', fontsize=14)
            plt.ylabel('Performance Drop (Accuracy)', fontsize=14)
            plt.legend(title='Noise Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            save_path = self.results_dir / f"dataset_comparison_{timestamp}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  📊 Dataset comparison saved to {save_path}")
            plt.show()
    
    def _create_noise_impact_analysis(self, results_df, timestamp):
        """创建噪声影响分析图"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 不同噪声类型的平均性能
        ax1 = axes[0, 0]
        noise_performance = results_df.groupby('noise_type_display')['accuracy'].agg(['mean', 'std'])
        bars = ax1.bar(range(len(noise_performance)), noise_performance['mean'], 
                      yerr=noise_performance['std'], capsize=5, alpha=0.7)
        ax1.set_title('Average Performance by Noise Type', fontweight='bold')
        ax1.set_xlabel('Noise Type')
        ax1.set_ylabel('Average Accuracy')
        ax1.set_xticks(range(len(noise_performance)))
        ax1.set_xticklabels(noise_performance.index, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. 噪声水平影响
        ax2 = axes[0, 1]
        noise_level_performance = results_df.groupby('noise_level')['accuracy'].agg(['mean', 'std'])
        ax2.errorbar(noise_level_performance.index, noise_level_performance['mean'], 
                    yerr=noise_level_performance['std'], marker='o', linewidth=2, markersize=8)
        ax2.set_title('Performance vs Noise Level', fontweight='bold')
        ax2.set_xlabel('Noise Level')
        ax2.set_ylabel('Average Accuracy')
        ax2.grid(True, alpha=0.3)
        
        # 3. 训练时间对比
        ax3 = axes[1, 0]
        time_by_noise = results_df.groupby('noise_type_display')['training_time'].mean()
        bars = ax3.bar(range(len(time_by_noise)), time_by_noise.values, alpha=0.7)
        ax3.set_title('Training Time by Noise Type', fontweight='bold')
        ax3.set_xlabel('Noise Type')
        ax3.set_ylabel('Average Training Time (s)')
        ax3.set_xticks(range(len(time_by_noise)))
        ax3.set_xticklabels(time_by_noise.index, rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. 噪声影响样本数
        ax4 = axes[1, 1]
        changed_samples = results_df[results_df['noise_level'] > 0].groupby('noise_level')['samples_changed'].mean()
        ax4.bar(range(len(changed_samples)), changed_samples.values, alpha=0.7)
        ax4.set_title('Average Samples Changed by Noise Level', fontweight='bold')
        ax4.set_xlabel('Noise Level')
        ax4.set_ylabel('Average Samples Changed')
        ax4.set_xticks(range(len(changed_samples)))
        ax4.set_xticklabels([f'{x:.1%}' for x in changed_samples.index])
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.results_dir / f"noise_impact_analysis_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  📊 Noise impact analysis saved to {save_path}")
        plt.show()
    
    def _generate_analysis_report(self, results_df, timestamp):
        """生成分析报告"""
        
        print("📝 Generating analysis report...")
        
        report_content = f"""# Noise Robustness Comparison Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Experiment Overview
- **Datasets tested**: {', '.join(results_df['dataset'].unique())}
- **Noise types**: {', '.join(results_df['noise_type_display'].unique())}
- **Noise levels**: {', '.join([f'{x:.1%}' for x in sorted(results_df['noise_level'].unique())])}
- **Total experiments**: {len(results_df)}

## Key Findings

### 1. Overall Performance Ranking
"""
        
        # 计算各噪声类型的总体性能
        overall_performance = results_df.groupby('noise_type_display')['accuracy'].agg(['mean', 'std']).round(4)
        overall_performance = overall_performance.sort_values('mean', ascending=False)
        
        report_content += "\nNoise Type Ranking (by mean accuracy):\n"
        for i, (noise_type, stats) in enumerate(overall_performance.iterrows(), 1):
            report_content += f"{i}. **{noise_type}**: {stats['mean']:.3f} ± {stats['std']:.3f}\n"
        
        # 最佳和最差性能
        best_result = results_df.loc[results_df['accuracy'].idxmax()]
        worst_result = results_df.loc[results_df['accuracy'].idxmin()]
        
        report_content += f"""
### 2. Best and Worst Performance
- **Best**: {best_result['noise_type_display']} on {best_result['dataset']} (Noise: {best_result['noise_level']:.1%}) - Accuracy: {best_result['accuracy']:.3f}
- **Worst**: {worst_result['noise_type_display']} on {worst_result['dataset']} (Noise: {worst_result['noise_level']:.1%}) - Accuracy: {worst_result['accuracy']:.3f}

### 3. Robustness Analysis
"""
        
        # 计算鲁棒性（无噪声到最高噪声的性能下降）
        robustness_analysis = []
        for dataset in results_df['dataset'].unique():
            for noise_type in results_df['noise_type'].unique():
                data = results_df[(results_df['dataset'] == dataset) & 
                                (results_df['noise_type'] == noise_type)]
                
                clean_acc = data[data['noise_level'] == 0]['accuracy']
                noisy_acc = data[data['noise_level'] > 0]['accuracy']
                
                if len(clean_acc) > 0 and len(noisy_acc) > 0:
                    avg_drop = clean_acc.iloc[0] - noisy_acc.mean()
                    robustness_analysis.append({
                        'dataset': dataset,
                        'noise_type': noise_type,
                        'noise_type_display': data['noise_type_display'].iloc[0],
                        'performance_drop': avg_drop
                    })
        
        robustness_df = pd.DataFrame(robustness_analysis)
        if not robustness_df.empty:
            avg_robustness = robustness_df.groupby('noise_type_display')['performance_drop'].mean().sort_values()
            
            report_content += "\nMost Robust Noise Types (smallest performance drop):\n"
            for i, (noise_type, drop) in enumerate(avg_robustness.head(3).items(), 1):
                report_content += f"{i}. **{noise_type}**: {drop:.3f} average drop\n"
        
        report_content += f"""
### 4. Dataset-specific Insights
"""
        
        for dataset in results_df['dataset'].unique():
            dataset_data = results_df[results_df['dataset'] == dataset]
            best_noise_type = dataset_data.groupby('noise_type_display')['accuracy'].mean().idxmax()
            worst_noise_type = dataset_data.groupby('noise_type_display')['accuracy'].mean().idxmin()
            
            report_content += f"""
**{dataset}**:
- Best noise type: {best_noise_type}
- Worst noise type: {worst_noise_type}
- Clean accuracy: {dataset_data[dataset_data['noise_level'] == 0]['accuracy'].mean():.3f}
"""
        
        # 保存报告
        report_file = self.results_dir / f"noise_comparison_report_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"  📝 Analysis report saved to {report_file}")

def run_comparison_experiment():
    """运行对比实验的主函数"""
    
    comparison = NoiseRobustnessComparison()
    
    # 运行实验
    results_df = comparison.run_noise_comparison_experiment(
        datasets=['iris', 'wine', 'breast_cancer'],
        noise_levels=[0.0, 0.1, 0.2, 0.3],
        noise_types=['random_uniform', 'proportional', 'majority_bias', 'minority_bias'],
        representation_dim=64,  # 使用较小的维度以加快实验
        epochs=50  # 使用较少的轮数以加快实验
    )
    
    return results_df

if __name__ == "__main__":
    results = run_comparison_experiment()
    print("\n🎉 Noise robustness comparison experiment completed!") 