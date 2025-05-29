"""
多分类实验模块

提供多分类实验的完整流程，包括：
1. 多类别合成数据生成
2. CAAC-SPSFT多分类模型训练
3. 基线方法比较
4. 性能评估和可视化
"""

import numpy as np
import pandas as pd
import torch
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.caac_model import CAACModelWrapper
from ..utils.metrics import evaluate_multiclass_classification

def generate_multiclass_data(
    n_samples=1000,
    n_features=10,
    n_classes=3,
    n_informative=8,
    n_redundant=2,
    n_clusters_per_class=1,
    class_sep=1.0,
    flip_y=0.01,
    random_state=42
):
    """
    生成多分类合成数据
    
    参数:
        n_samples: 样本数量
        n_features: 特征数量
        n_classes: 类别数量
        n_informative: 信息特征数量
        n_redundant: 冗余特征数量
        n_clusters_per_class: 每个类别的簇数
        class_sep: 类别分离度
        flip_y: 标签噪声比例
        random_state: 随机种子
        
    返回:
        X: 特征矩阵
        y: 标签向量
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        class_sep=class_sep,
        flip_y=flip_y,
        random_state=random_state
    )
    
    return X, y

def add_outliers_multiclass(X, y, outlier_ratio=0.1, random_state=42):
    """
    为多分类数据添加异常值
    
    参数:
        X: 特征矩阵
        y: 标签向量
        outlier_ratio: 异常值比例
        random_state: 随机种子
        
    返回:
        X_with_outliers: 包含异常值的特征矩阵
        y_with_outliers: 包含翻转标签的标签向量
        outlier_mask: 异常值标记
    """
    np.random.seed(random_state)
    n_samples = X.shape[0]
    n_outliers = int(n_samples * outlier_ratio)
    n_classes = len(np.unique(y))
    
    # 复制数据
    X_with_outliers = X.copy()
    y_with_outliers = y.copy()
    
    # 随机选择异常值样本
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    outlier_mask = np.zeros(n_samples, dtype=bool)
    outlier_mask[outlier_indices] = True
    
    # 为异常值添加噪声
    noise = np.random.randn(n_outliers, X.shape[1]) * np.std(X, axis=0) * 2
    X_with_outliers[outlier_indices] += noise
    
    # 随机翻转异常值的标签到其他类别
    for idx in outlier_indices:
        current_label = y_with_outliers[idx]
        other_labels = [i for i in range(n_classes) if i != current_label]
        y_with_outliers[idx] = np.random.choice(other_labels)
    
    return X_with_outliers, y_with_outliers, outlier_mask

def run_multiclass_classification_experiment(
    n_samples=1000,
    n_features=10,
    n_classes=3,
    test_size=0.2,
    val_size=0.2,
    class_sep=1.0,
    outlier_ratio=0.0,
    random_state=42,
    model_params=None
):
    """
    运行多分类实验
    
    参数:
        n_samples: 样本数量
        n_features: 特征数量
        n_classes: 类别数量
        test_size: 测试集比例
        val_size: 验证集比例（从训练集中划分）
        class_sep: 类别分离度
        outlier_ratio: 异常值比例
        random_state: 随机种子
        model_params: 模型参数字典
        
    返回:
        results: 实验结果字典
    """
    # 生成数据
    X, y = generate_multiclass_data(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        class_sep=class_sep,
        random_state=random_state
    )
    
    # 添加异常值（如果需要）
    if outlier_ratio > 0:
        X, y, outlier_mask = add_outliers_multiclass(X, y, outlier_ratio, random_state)
    else:
        outlier_mask = np.zeros(len(y), dtype=bool)
    
    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 划分数据集
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size/(1-test_size), 
        random_state=random_state, stratify=y_train_val
    )
    
    # 设置默认模型参数
    default_model_params = {
        'input_dim': n_features,
        'representation_dim': 64,
        'latent_dim': 32,
        'n_paths': n_classes,  # 路径数量设为类别数量
        'n_classes': n_classes,
        'feature_hidden_dims': [64],
        'abduction_hidden_dims': [64, 32],
        'lr': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'early_stopping_patience': 10
    }
    
    # 更新模型参数
    if model_params is not None:
        default_model_params.update(model_params)
    
    # 创建并训练模型
    model = CAACModelWrapper(**default_model_params)
    model.fit(X_train, y_train, X_val, y_val, verbose=1)
    
    # 评估模型
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    metrics = evaluate_multiclass_classification(y_test, y_pred, y_pred_proba)
    
    # 返回结果
    results = {
        'model': model,
        'metrics': metrics,
        'history': model.history,
        'data': {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'outlier_mask': outlier_mask
        },
        'experiment_config': {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_classes': n_classes,
            'test_size': test_size,
            'val_size': val_size,
            'class_sep': class_sep,
            'outlier_ratio': outlier_ratio,
            'random_state': random_state,
            'model_params': default_model_params
        }
    }
    
    return results

def compare_multiclass_with_baselines(
    X_train, y_train, X_test, y_test,
    caac_model=None,
    n_classes=3,
    random_state=42
):
    """
    与基线方法比较（多分类）
    
    参数:
        X_train: 训练集特征
        y_train: 训练集标签
        X_test: 测试集特征
        y_test: 测试集标签
        caac_model: 已训练的CAAC模型 (可选)
        n_classes: 类别数量
        random_state: 随机种子
        
    返回:
        comparison_results: 比较结果字典
    """
    results = {}
    
    # 如果提供了CAAC模型，评估它
    if caac_model is not None:
        y_pred = caac_model.predict(X_test)
        y_pred_proba = caac_model.predict_proba(X_test)
        caac_metrics = evaluate_multiclass_classification(y_test, y_pred, y_pred_proba)
        caac_metrics['train_time'] = caac_model.history.get('train_time', 0.0)
        results['CAAC'] = caac_metrics
    
    # 逻辑回归（多分类）
    start_time = time.time()
    lr = LogisticRegression(multi_class='multinomial', random_state=random_state)
    lr.fit(X_train, y_train)
    lr_time = time.time() - start_time
    
    y_pred = lr.predict(X_test)
    y_pred_proba = lr.predict_proba(X_test)
    lr_metrics = evaluate_multiclass_classification(y_test, y_pred, y_pred_proba)
    lr_metrics['train_time'] = lr_time
    results['LogisticRegression'] = lr_metrics
    
    # 随机森林
    start_time = time.time()
    rf = RandomForestClassifier(random_state=random_state)
    rf.fit(X_train, y_train)
    rf_time = time.time() - start_time
    
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)
    rf_metrics = evaluate_multiclass_classification(y_test, y_pred, y_pred_proba)
    rf_metrics['train_time'] = rf_time
    results['RandomForest'] = rf_metrics
    
    # SVM（多分类）
    start_time = time.time()
    svm = SVC(probability=True, decision_function_shape='ovr', random_state=random_state)
    svm.fit(X_train, y_train)
    svm_time = time.time() - start_time
    
    y_pred = svm.predict(X_test)
    y_pred_proba = svm.predict_proba(X_test)
    svm_metrics = evaluate_multiclass_classification(y_test, y_pred, y_pred_proba)
    svm_metrics['train_time'] = svm_time
    results['SVM'] = svm_metrics
    
    # 转换为DataFrame以便比较
    comparison_df = pd.DataFrame(results).T
    
    return {
        'results': results,
        'comparison_df': comparison_df
    }

def visualize_multiclass_results(results, save_path=None):
    """
    可视化多分类实验结果
    
    参数:
        results: 实验结果字典
        save_path: 保存路径（可选）
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 训练历史
    ax = axes[0, 0]
    history = results['history']
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True)
    
    # 2. 准确率历史
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    if 'val_acc' in history:
        ax.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy History')
    ax.legend()
    ax.grid(True)
    
    # 3. 混淆矩阵
    ax = axes[1, 0]
    y_test = results['data']['y_test']
    y_pred = results['model'].predict(results['data']['X_test'])
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    # 4. 类别概率分布
    ax = axes[1, 1]
    y_pred_proba = results['model'].predict_proba(results['data']['X_test'])
    n_classes = y_pred_proba.shape[1]
    
    # 为每个真实类别绘制预测概率分布
    for true_class in range(n_classes):
        mask = y_test == true_class
        if np.any(mask):
            probs = y_pred_proba[mask, true_class]
            ax.hist(probs, bins=20, alpha=0.6, label=f'Class {true_class}')
    
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Count')
    ax.set_title('Predicted Probabilities for True Classes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig 