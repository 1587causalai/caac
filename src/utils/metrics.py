"""
评估指标模块

提供以下组件：
1. evaluate_binary_classification - 评估二分类模型性能
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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
