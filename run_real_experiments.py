"""
真实数据集实验运行脚本

运行CAAC-SPSFT二分类算法在真实数据集上的实验，并与基线方法比较
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
    
    # 3. German Credit数据集
    print("加载 German Credit 数据集...")
    try:
        german = fetch_openml(name='german', version=1, as_frame=True, parser='auto')
        X_german = german.data
        y_german = german.target
        
        # 处理类别特征
        categorical_features = X_german.select_dtypes(include=['object', 'category']).columns
        numerical_features = X_german.select_dtypes(include=['int64', 'float64']).columns
        
        # 预处理
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ])
        
        X_german_processed = preprocessor.fit_transform(X_german)
        
        # 处理标签 (good=1, bad=0)
        le = LabelEncoder()
        y_german_processed = le.fit_transform(y_german)
        
        datasets['german_credit'] = {
            'X': X_german_processed,
            'y': y_german_processed,
            'name': 'German Credit',
            'task_type': 'binary',
            'description': '德国信贷数据集'
        }
    except Exception as e:
        print(f"Warning: 无法加载German Credit数据集: {e}")
    
    # 4. Adult数据集
    print("加载 Adult 数据集...")
    try:
        adult = fetch_openml(name='adult', version=2, as_frame=True, parser='auto')
        X_adult = adult.data
        y_adult = adult.target
        
        # 处理缺失值
        X_adult = X_adult.fillna(X_adult.mode().iloc[0])
        
        # 处理类别特征
        categorical_features = X_adult.select_dtypes(include=['object', 'category']).columns
        numerical_features = X_adult.select_dtypes(include=['int64', 'float64']).columns
        
        # 预处理
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ])
        
        X_adult_processed = preprocessor.fit_transform(X_adult)
        
        # 处理标签 (>50K=1, <=50K=0)
        y_adult_processed = (y_adult == '>50K').astype(int)
        
        # 采样以减少数据量
        if len(X_adult_processed) > 10000:
            X_adult_processed, _, y_adult_processed, _ = train_test_split(
                X_adult_processed, y_adult_processed, 
                train_size=10000, random_state=42, stratify=y_adult_processed
            )
        
        datasets['adult'] = {
            'X': X_adult_processed,
            'y': y_adult_processed,
            'name': 'Adult (Census Income)',
            'task_type': 'binary',
            'description': '成年人收入普查数据集'
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
                           hidden_layer_sizes=(128, 64), 
                           learning_rate_init=0.001,
                           early_stopping=True,
                           validation_fraction=0.1)
    }
    
    results = {}
    
    for name, model in baselines.items():
        print(f"  训练 {name}...")
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
            print(f"    Error training {name}: {e}")
            results[name] = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'auc_roc': 0.5,
                'train_time': 0.0
            }
    
    return results

def run_caac_experiment(X_train, X_val, X_test, y_train, y_val, y_test):
    """运行CAAC实验"""
    print("  训练 CAAC-SPSFT...")
    start_time = datetime.now()
    
    # 获取输入维度
    input_dim = X_train.shape[1]
    
    # 设置CAAC模型参数
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
        'epochs': 200,
        'early_stopping_patience': 20
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
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc,
            'train_time': train_time,
            'model': model
        }
        
    except Exception as e:
        print(f"    Error training CAAC: {e}")
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
            'model': None
        }

def save_results(dataset_name, caac_results, baseline_results, results_dir):
    """保存实验结果"""
    # 合并结果
    all_results = {'CAAC-SPSFT': caac_results, **baseline_results}
    
    # 创建DataFrame
    results_df = pd.DataFrame(all_results).T
    
    # 保存CSV
    csv_file = os.path.join(results_dir, f"comparison_{dataset_name}.csv")
    results_df.to_csv(csv_file)
    print(f"  结果已保存到: {csv_file}")
    
    # 保存CAAC训练历史图片
    if caac_results.get('model') is not None and hasattr(caac_results['model'], 'history'):
        save_training_history_plot(caac_results['model'].history, dataset_name)
    
    # 打印结果
    print(f"\n  === {dataset_name} 性能比较 ===")
    print(results_df.round(3))
    
    return results_df

def save_training_history_plot(history, dataset_name):
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
    plt.title(f'{dataset_name.title()} - Training Loss')
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
    plt.title(f'{dataset_name.title()} - Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    image_file = os.path.join(images_dir, f"history_{dataset_name}.png")
    plt.savefig(image_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  训练历史图片已保存到: {image_file}")

def main():
    """主函数"""
    # 设置随机种子
    random_state = 42
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    
    # 创建结果目录
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_real')
    os.makedirs(results_dir, exist_ok=True)
    
    print("=== 真实数据集实验 ===\n")
    
    # 加载数据集
    datasets = load_real_datasets()
    
    if not datasets:
        print("错误: 没有成功加载任何数据集")
        return
    
    print(f"\n成功加载 {len(datasets)} 个数据集")
    
    # 运行实验
    all_experiment_results = []
    
    for dataset_name, dataset_info in datasets.items():
        print(f"\n=== 运行实验: {dataset_info['name']} ===")
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
        
        # 划分数据集
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
        )
        
        print(f"  训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")
        
        # 运行CAAC实验
        caac_results = run_caac_experiment(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # 运行基线方法
        baseline_results = run_baselines(X_train, X_test, y_train, y_test, random_state)
        
        # 保存结果
        results_df = save_results(dataset_name, caac_results, baseline_results, results_dir)
        
        # 记录实验结果
        experiment_result = {
            'dataset_name': dataset_name,
            'dataset_info': dataset_info,
            'caac_results': caac_results,
            'baseline_results': baseline_results,
            'results_df': results_df
        }
        all_experiment_results.append(experiment_result)
    
    # 生成综合报告
    print(f"\n=== 实验总结 ===")
    summary_data = []
    for exp in all_experiment_results:
        row = {
            'Dataset': exp['dataset_info']['name'],
            'Samples': exp['dataset_info']['X'].shape[0],
            'Features': exp['dataset_info']['X'].shape[1],
            'CAAC_Accuracy': exp['caac_results']['accuracy'],
            'CAAC_F1': exp['caac_results']['f1'],
            'Best_Baseline': max(exp['baseline_results'].keys(), 
                               key=lambda x: exp['baseline_results'][x]['accuracy']),
            'Best_Baseline_Accuracy': max(exp['baseline_results'].values(), 
                                        key=lambda x: x['accuracy'])['accuracy']
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.round(3))
    
    # 保存综合报告
    summary_file = os.path.join(results_dir, "experiment_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"\n综合报告已保存到: {summary_file}")
    
    return all_experiment_results

if __name__ == "__main__":
    main() 