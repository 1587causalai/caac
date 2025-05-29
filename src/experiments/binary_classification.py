"""
实验流程模块

提供以下组件：
1. run_binary_classification_experiment - 运行二分类实验
2. compare_with_baselines - 与基线方法比较
"""

import numpy as np
import pandas as pd
import time
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from ..data.synthetic import SyntheticBinaryClassificationGenerator, split_data
from ..models.caac_model import CAACModelWrapper
from ..utils.metrics import evaluate_binary_classification

def run_binary_classification_experiment(
    n_samples=1000, 
    n_features=10, 
    test_size=0.2,
    val_size=0.2,
    data_type='linear',
    outlier_ratio=0.1,
    random_state=42,
    model_params=None
):
    """
    运行二分类实验
    
    参数:
        n_samples: 样本数量
        n_features: 特征数量
        test_size: 测试集比例
        val_size: 验证集比例
        data_type: 数据类型 ('linear' 或 'nonlinear')
        outlier_ratio: 异常值比例
        random_state: 随机种子
        model_params: 模型参数字典
        
    返回:
        results: 实验结果字典
    """
    # 生成数据
    generator = SyntheticBinaryClassificationGenerator(
        n_samples_total=n_samples, 
        n_features=n_features, 
        random_state=random_state
    )
    
    if data_type == 'linear':
        X, y = generator.generate_linear()
    elif data_type == 'nonlinear':
        X, y = generator.generate_nonlinear()
    else:
        raise ValueError(f"不支持的数据类型: {data_type}")
    
    # 注入异常值
    if outlier_ratio > 0:
        X, y, outlier_mask = generator.inject_outliers(X, y, outlier_ratio)
    else:
        outlier_mask = np.zeros(len(y), dtype=bool)
    
    # 划分数据集
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=test_size, val_size=val_size, random_state=random_state
    )
    
    # 设置默认模型参数
    default_model_params = {
        'input_dim': n_features,
        'representation_dim': 64,
        'latent_dim': 32,
        'n_paths': 2,
        'n_classes': 2,
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
    metrics = evaluate_binary_classification(y_test, y_pred, y_pred_proba)
    
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
            'test_size': test_size,
            'val_size': val_size,
            'data_type': data_type,
            'outlier_ratio': outlier_ratio,
            'random_state': random_state,
            'model_params': default_model_params
        }
    }
    
    return results

def compare_with_baselines(
    X_train, y_train, X_test, y_test,
    caac_model=None,
    random_state=42
):
    """
    与基线方法比较
    
    参数:
        X_train: 训练集特征
        y_train: 训练集标签
        X_test: 测试集特征
        y_test: 测试集标签
        caac_model: 已训练的CAAC模型 (可选)
        random_state: 随机种子
        
    返回:
        comparison_results: 比较结果字典
    """
    results = {}
    
    # 如果提供了CAAC模型，评估它
    if caac_model is not None:
        y_pred = caac_model.predict(X_test)
        y_pred_proba = caac_model.predict_proba(X_test)
        caac_metrics = evaluate_binary_classification(y_test, y_pred, y_pred_proba)
        # 添加训练时间
        caac_metrics['train_time'] = caac_model.history.get('train_time', 0.0)
        results['CAAC'] = caac_metrics
    
    # 逻辑回归
    start_time = time.time()
    lr = LogisticRegression(random_state=random_state)
    lr.fit(X_train, y_train)
    lr_time = time.time() - start_time
    
    y_pred = lr.predict(X_test)
    y_pred_proba = lr.predict_proba(X_test)
    lr_metrics = evaluate_binary_classification(y_test, y_pred, y_pred_proba)
    lr_metrics['train_time'] = lr_time
    results['LogisticRegression'] = lr_metrics
    
    # 随机森林
    start_time = time.time()
    rf = RandomForestClassifier(random_state=random_state)
    rf.fit(X_train, y_train)
    rf_time = time.time() - start_time
    
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)
    rf_metrics = evaluate_binary_classification(y_test, y_pred, y_pred_proba)
    rf_metrics['train_time'] = rf_time
    results['RandomForest'] = rf_metrics
    
    # SVM
    start_time = time.time()
    svm = SVC(probability=True, random_state=random_state)
    svm.fit(X_train, y_train)
    svm_time = time.time() - start_time
    
    y_pred = svm.predict(X_test)
    y_pred_proba = svm.predict_proba(X_test)
    svm_metrics = evaluate_binary_classification(y_test, y_pred, y_pred_proba)
    svm_metrics['train_time'] = svm_time
    results['SVM'] = svm_metrics
    
    # 转换为DataFrame以便比较
    comparison_df = pd.DataFrame(results).T
    
    return {
        'results': results,
        'comparison_df': comparison_df
    }
