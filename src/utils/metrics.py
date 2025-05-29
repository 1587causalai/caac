"""
评估指标模块

提供以下组件：
1. evaluate_binary_classification - 评估二分类模型性能
2. evaluate_multiclass_classification - 评估多分类模型性能
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

def evaluate_binary_classification(y_true, y_pred, y_pred_proba=None):
    """
    评估二分类模型性能
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        y_pred_proba: 预测概率 (可选，用于计算AUC)
        
    返回:
        metrics: 包含所有评估指标的字典
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    # 如果提供了预测概率，计算AUC
    if y_pred_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        except:
            metrics['auc'] = float('nan')  # 处理可能的错误
    
    return metrics

def evaluate_multiclass_classification(y_true, y_pred, y_pred_proba=None):
    """
    评估多分类模型性能
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        y_pred_proba: 预测概率矩阵 [n_samples, n_classes] (可选，用于计算AUC)
        
    返回:
        metrics: 包含所有评估指标的字典
    """
    # 基本指标
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # 每个类别的指标
    n_classes = len(np.unique(y_true))
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    for i in range(n_classes):
        metrics[f'precision_class_{i}'] = precision_per_class[i]
        metrics[f'recall_class_{i}'] = recall_per_class[i]
        metrics[f'f1_class_{i}'] = f1_per_class[i]
    
    # 如果提供了预测概率，计算多分类AUC
    if y_pred_proba is not None:
        try:
            # One-vs-Rest AUC
            metrics['auc_ovr'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            # One-vs-One AUC
            metrics['auc_ovo'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovo')
        except:
            metrics['auc_ovr'] = float('nan')
            metrics['auc_ovo'] = float('nan')
    
    # 混淆矩阵
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics
