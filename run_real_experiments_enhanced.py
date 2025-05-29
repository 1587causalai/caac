"""
增强版真实数据集实验运行脚本

在原有基础上添加异常值鲁棒性实验，并优化早停机制
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.datasets import load_iris, load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# 导入CAAC模型
from src.models.caac_model import CAACModelWrapper

def inject_outliers_to_real_data(X, y, outlier_ratio=0.1, random_state=42):
    """
    为真实数据注入异常值
    
    参数:
        X: 特征矩阵
        y: 类别标签
        outlier_ratio: 异常值比例
        random_state: 随机种子
    
    返回:
        X_with_outliers: 包含异常值的特征矩阵
        y_with_outliers: 包含异常值的类别标签
        outlier_mask: 异常值掩码
    """
    rng = np.random.RandomState(random_state)
    X_with_outliers = X.copy()
    y_with_outliers = y.copy()
    
    n_samples = X.shape[0]
    n_outliers = int(n_samples * outlier_ratio)
    
    outlier_indices = rng.choice(n_samples, n_outliers, replace=False)
    outlier_mask = np.zeros(n_samples, dtype=bool)
    outlier_mask[outlier_indices] = True
    
    for idx in outlier_indices:
        # 翻转类别标签
        y_with_outliers[idx] = 1 - y_with_outliers[idx]
        
        # 添加特征噪声（基于每个特征的标准差）
        feature_stds = np.std(X, axis=0)
        noise = rng.randn(X.shape[1]) * feature_stds * 2.0  # 2倍标准差的噪声
        X_with_outliers[idx] += noise
    
    return X_with_outliers, y_with_outliers, outlier_mask

def load_real_datasets():
    """加载所有真实数据集"""
    datasets = {}
    
    # 1. Iris数据集 (多分类，转为二分类：setosa vs 其他)
    print("加载 Iris 数据集...")
    iris = load_iris()
    X_iris = iris.data
    y_iris = (iris.target != 0).astype(int)  # setosa(0) vs others(1,2)
    datasets['iris'] = {
        'X': X_iris,
        'y': y_iris,
        'name': 'Iris',
        'task_type': 'binary',
        'description': '鸢尾花数据集 (setosa vs others)'
    }
    
    # 2. Breast Cancer数据集
    print("加载 Breast Cancer 数据集...")
    bc = load_breast_cancer()
    datasets['breast_cancer'] = {
        'X': bc.data,
        'y': bc.target,
        'name': 'Breast Cancer Wisconsin',
        'task_type': 'binary',
        'description': '乳腺癌诊断数据集'
    }
    
    # 3. Adult数据集（简化版，只取部分特征避免维度过高）
    print("加载 Adult 数据集...")
    try:
        adult = fetch_openml(name='adult', version=2, as_frame=True, parser='auto')
        X_adult = adult.data
        y_adult = adult.target
        
        # 只选择数值特征，避免高维度问题
        numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        X_adult_selected = X_adult[numerical_features]
        
        # 处理缺失值
        X_adult_selected = X_adult_selected.fillna(X_adult_selected.median())
        
        # 标准化
        scaler = StandardScaler()
        X_adult_processed = scaler.fit_transform(X_adult_selected)
        
        # 处理标签 (>50K=1, <=50K=0)
        y_adult_processed = (y_adult == '>50K').astype(int)
        
        # 采样以减少数据量
        if len(X_adult_processed) > 5000:
            X_adult_processed, _, y_adult_processed, _ = train_test_split(
                X_adult_processed, y_adult_processed, 
                train_size=5000, random_state=42, stratify=y_adult_processed
            )
        
        datasets['adult'] = {
            'X': X_adult_processed,
            'y': y_adult_processed,
            'name': 'Adult (Census Income)',
            'task_type': 'binary',
            'description': '成年人收入普查数据集（简化版）'
        }
    except Exception as e:
        print(f"Warning: 无法加载Adult数据集: {e}")
    
    return datasets

def run_baselines(X_train, X_test, y_train, y_test, random_state=42):
    """运行基线方法"""
    baselines = {
        'LogisticRegression': LogisticRegression(random_state=random_state, max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=random_state, n_estimators=100),
        'SVM': SVC(random_state=random_state, probability=True),
        'XGBoost': xgb.XGBClassifier(random_state=random_state, eval_metric='logloss'),
        'MLP': MLPClassifier(random_state=random_state, max_iter=1000, 
                           hidden_layer_sizes=(64, 32), 
                           learning_rate_init=0.001,
                           early_stopping=True,
                           validation_fraction=0.1)
    }
    
    results = {}
    
    for name, model in baselines.items():
        print(f"    训练 {name}...")
        start_time = datetime.now()
        
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            end_time = datetime.now()
            train_time = (end_time - start_time).total_seconds()
            
            # 计算指标
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
            
            try:
                auc_roc = roc_auc_score(y_test, y_pred_proba)
            except:
                auc_roc = 0.5
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc_roc': auc_roc,
                'train_time': train_time
            }
            
        except Exception as e:
            print(f"      Error training {name}: {e}")
            results[name] = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'auc_roc': 0.5,
                'train_time': 0.0
            }
    
    return results

def run_caac_experiment(X_train, X_val, X_test, y_train, y_val, y_test, experiment_name=""):
    """运行CAAC实验"""
    print(f"    训练 CAAC-SPSFT ({experiment_name})...")
    start_time = datetime.now()
    
    # 获取输入维度
    input_dim = X_train.shape[1]
    
    # 设置CAAC模型参数 - 更严格的早停
    model_params = {
        'input_dim': input_dim,
        'representation_dim': 64,
        'latent_dim': 32,
        'n_paths': 2,
        'n_classes': 2,
        'feature_hidden_dims': [64],
        'abduction_hidden_dims': [64, 32],
        'lr': 0.001,
        'batch_size': 32,
        'epochs': 100,  # 减少最大epoch数
        'early_stopping_patience': 10,  # 更严格的早停
        'early_stopping_min_delta': 0.001  # 更大的最小改善阈值
    }
    
    try:
        # 创建并训练CAAC模型
        model = CAACModelWrapper(**model_params)
        model.fit(X_train, y_train, X_val, y_val, verbose=0)
        
        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # 获取正类概率
        
        end_time = datetime.now()
        train_time = (end_time - start_time).total_seconds()
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        # 记录早停信息
        final_epoch = len(model.history['train_loss'])
        best_epoch = model.history.get('best_epoch', final_epoch)
        early_stopped = final_epoch < model_params['epochs']
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc,
            'train_time': train_time,
            'model': model,
            'final_epoch': final_epoch,
            'best_epoch': best_epoch,
            'early_stopped': early_stopped
        }
        
    except Exception as e:
        print(f"      Error training CAAC: {e}")
        end_time = datetime.now()
        train_time = (end_time - start_time).total_seconds()
        
        # 返回默认值
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc_roc': 0.5,
            'train_time': train_time,
            'model': None,
            'final_epoch': 0,
            'best_epoch': 0,
            'early_stopped': False
        }

def save_results(dataset_name, experiments_results, results_dir):
    """保存实验结果"""
    # 为每个实验保存结果
    for exp_name, results in experiments_results.items():
        caac_results = results['caac_results']
        baseline_results = results['baseline_results']
        
        # 合并结果
        all_results = {'CAAC-SPSFT': caac_results, **baseline_results}
        
        # 创建DataFrame
        results_df = pd.DataFrame(all_results).T
        
        # 保存CSV
        csv_file = os.path.join(results_dir, f"comparison_{dataset_name}_{exp_name}.csv")
        results_df.to_csv(csv_file)
        print(f"    结果已保存到: {csv_file}")
        
        # 保存CAAC训练历史图片
        if caac_results.get('model') is not None and hasattr(caac_results['model'], 'history'):
            save_training_history_plot(caac_results['model'].history, f"{dataset_name}_{exp_name}")
        
        # 打印结果
        print(f"\n    === {dataset_name} ({exp_name}) 性能比较 ===")
        display_df = results_df[['accuracy', 'precision', 'recall', 'f1', 'auc_roc']].round(3)
        print(display_df)
        
        # 打印早停信息
        if caac_results.get('early_stopped'):
            print(f"    CAAC早停触发: 第{caac_results['final_epoch']}轮停止，最佳轮次: {caac_results['best_epoch']}")
        else:
            print(f"    CAAC训练完成: 第{caac_results['final_epoch']}轮完成，无早停")
    
    return experiments_results

def save_training_history_plot(history, experiment_name):
    """保存训练历史图片到docs/assets/images/目录"""
    # 创建图片保存目录
    images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs', 'assets', 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # 创建图表
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    if 'train_loss' in history and len(history['train_loss']) > 0:
        plt.plot(history['train_loss'], label='Train Loss', color='blue')
    if 'val_loss' in history and len(history['val_loss']) > 0:
        plt.plot(history['val_loss'], label='Validation Loss', color='orange')
        # 标记最佳轮次
        if 'best_epoch' in history and history['best_epoch'] > 0:
            best_idx = history['best_epoch'] - 1
            if best_idx < len(history['val_loss']):
                plt.axvline(x=best_idx, color='red', linestyle='--', alpha=0.7, label='Best Epoch')
    plt.title(f'{experiment_name.title().replace("_", " ")} - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    if 'train_acc' in history and len(history['train_acc']) > 0:
        plt.plot(history['train_acc'], label='Train Accuracy', color='blue')
    if 'val_acc' in history and len(history['val_acc']) > 0:
        plt.plot(history['val_acc'], label='Validation Accuracy', color='orange')
        # 标记最佳轮次
        if 'best_epoch' in history and history['best_epoch'] > 0:
            best_idx = history['best_epoch'] - 1
            if best_idx < len(history['val_acc']):
                plt.axvline(x=best_idx, color='red', linestyle='--', alpha=0.7, label='Best Epoch')
    plt.title(f'{experiment_name.title().replace("_", " ")} - Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    image_file = os.path.join(images_dir, f"history_{experiment_name}.png")
    plt.savefig(image_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    训练历史图片已保存到: {image_file}")

def run_robustness_experiment(dataset_name, dataset_info, results_dir, random_state=42):
    """运行鲁棒性实验（原始数据 vs 有异常值数据）"""
    print(f"\n=== 运行鲁棒性实验: {dataset_info['name']} ===")
    print(f"描述: {dataset_info['description']}")
    print(f"样本数量: {dataset_info['X'].shape[0]}")
    print(f"特征数量: {dataset_info['X'].shape[1]}")
    
    X, y = dataset_info['X'], dataset_info['y']
    
    # 数据预处理
    if X.dtype == 'object':  # 如果还有未处理的类别特征
        print("  数据预处理...")
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # 确保数据是numpy数组格式
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=int)
    
    experiments_results = {}
    
    # 实验1: 原始数据（无异常值）
    print("\n  实验1: 原始数据（无异常值）")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )
    
    print(f"    训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")
    
    # 运行CAAC实验
    caac_results = run_caac_experiment(X_train, X_val, X_test, y_train, y_val, y_test, "原始数据")
    
    # 运行基线方法
    baseline_results = run_baselines(X_train, X_test, y_train, y_test, random_state)
    
    experiments_results['clean'] = {
        'caac_results': caac_results,
        'baseline_results': baseline_results
    }
    
    # 实验2: 注入异常值数据
    print("\n  实验2: 注入异常值数据（10%异常值）")
    X_outliers, y_outliers, outlier_mask = inject_outliers_to_real_data(X, y, outlier_ratio=0.1, random_state=random_state)
    
    print(f"    注入异常值: {np.sum(outlier_mask)} / {len(outlier_mask)} ({np.sum(outlier_mask)/len(outlier_mask)*100:.1f}%)")
    
    X_train_out, X_temp_out, y_train_out, y_temp_out = train_test_split(
        X_outliers, y_outliers, test_size=0.2, random_state=random_state, stratify=y_outliers
    )
    X_val_out, X_test_out, y_val_out, y_test_out = train_test_split(
        X_temp_out, y_temp_out, test_size=0.5, random_state=random_state, stratify=y_temp_out
    )
    
    # 运行CAAC实验
    caac_results_out = run_caac_experiment(X_train_out, X_val_out, X_test_out, y_train_out, y_val_out, y_test_out, "有异常值")
    
    # 运行基线方法
    baseline_results_out = run_baselines(X_train_out, X_test_out, y_train_out, y_test_out, random_state)
    
    experiments_results['outliers'] = {
        'caac_results': caac_results_out,
        'baseline_results': baseline_results_out
    }
    
    # 保存结果
    save_results(dataset_name, experiments_results, results_dir)
    
    # 计算鲁棒性分析
    print(f"\n  === {dataset_info['name']} 鲁棒性分析 ===")
    models = ['CAAC-SPSFT'] + list(baseline_results.keys())
    
    print("  模型准确率下降分析：")
    print("  模型\t\t原始数据\t有异常值\t下降幅度\t下降百分比")
    print("  " + "-"*60)
    
    for model in models:
        clean_acc = experiments_results['clean']['caac_results']['accuracy'] if model == 'CAAC-SPSFT' else experiments_results['clean']['baseline_results'][model]['accuracy']
        outlier_acc = experiments_results['outliers']['caac_results']['accuracy'] if model == 'CAAC-SPSFT' else experiments_results['outliers']['baseline_results'][model]['accuracy']
        
        acc_drop = clean_acc - outlier_acc
        acc_drop_pct = (acc_drop / clean_acc * 100) if clean_acc > 0 else 0
        
        print(f"  {model:<12}\t{clean_acc:.3f}\t\t{outlier_acc:.3f}\t\t{acc_drop:.3f}\t\t{acc_drop_pct:.1f}%")
    
    return experiments_results

def main():
    """主函数"""
    # 设置随机种子
    random_state = 42
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    
    # 创建结果目录
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_real_enhanced')
    os.makedirs(results_dir, exist_ok=True)
    
    print("=== 增强版真实数据集鲁棒性实验 ===\n")
    
    # 加载数据集
    datasets = load_real_datasets()
    
    if not datasets:
        print("错误: 没有成功加载任何数据集")
        return
    
    print(f"\n成功加载 {len(datasets)} 个数据集")
    
    # 运行鲁棒性实验
    all_experiment_results = {}
    
    for dataset_name, dataset_info in datasets.items():
        experiments_results = run_robustness_experiment(dataset_name, dataset_info, results_dir, random_state)
        all_experiment_results[dataset_name] = experiments_results
    
    # 生成综合鲁棒性报告
    print(f"\n=== 综合鲁棒性实验总结 ===")
    
    summary_data = []
    for dataset_name, experiments in all_experiment_results.items():
        dataset_info = datasets[dataset_name]
        
        # 获取CAAC结果
        caac_clean = experiments['clean']['caac_results']['accuracy']
        caac_outlier = experiments['outliers']['caac_results']['accuracy']
        caac_drop_pct = (caac_clean - caac_outlier) / caac_clean * 100 if caac_clean > 0 else 0
        
        # 获取最佳基线方法结果
        baseline_clean_results = experiments['clean']['baseline_results']
        best_baseline_clean = max(baseline_clean_results.keys(), key=lambda x: baseline_clean_results[x]['accuracy'])
        best_baseline_clean_acc = baseline_clean_results[best_baseline_clean]['accuracy']
        
        baseline_outlier_results = experiments['outliers']['baseline_results']
        best_baseline_outlier_acc = baseline_outlier_results[best_baseline_clean]['accuracy']
        baseline_drop_pct = (best_baseline_clean_acc - best_baseline_outlier_acc) / best_baseline_clean_acc * 100 if best_baseline_clean_acc > 0 else 0
        
        row = {
            'Dataset': dataset_info['name'],
            'Samples': dataset_info['X'].shape[0],
            'Features': dataset_info['X'].shape[1],
            'CAAC_Clean': caac_clean,
            'CAAC_Outlier': caac_outlier,
            'CAAC_Drop%': caac_drop_pct,
            'Best_Baseline': best_baseline_clean,
            'Baseline_Clean': best_baseline_clean_acc,
            'Baseline_Outlier': best_baseline_outlier_acc,
            'Baseline_Drop%': baseline_drop_pct,
            'CAAC_More_Robust': caac_drop_pct < baseline_drop_pct
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.round(3))
    
    # 保存综合报告
    summary_file = os.path.join(results_dir, "robustness_experiment_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"\n鲁棒性实验综合报告已保存到: {summary_file}")
    
    # 统计鲁棒性优势
    caac_wins = summary_df['CAAC_More_Robust'].sum()
    total_datasets = len(summary_df)
    print(f"\nCAAC-SPSFT在 {caac_wins}/{total_datasets} 个数据集上展现了更好的鲁棒性")
    
    return all_experiment_results

if __name__ == "__main__":
    main() 