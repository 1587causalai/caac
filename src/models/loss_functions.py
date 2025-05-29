"""
损失函数模块

提供以下组件：
1. compute_nll_loss - 计算负对数似然损失
"""

import torch
import torch.nn.functional as F

def compute_nll_loss(y_true, y_pred_probs):
    """
    计算负对数似然损失
    
    参数:
        y_true: 真实标签，形状为 [batch_size]，值为0或1
        y_pred_probs: 预测概率，形状为 [batch_size, n_classes]
    
    返回:
        loss: 负对数似然损失
    """
    batch_size = y_true.shape[0]
    
    # 将真实标签转换为one-hot编码
    y_true_one_hot = F.one_hot(y_true, num_classes=y_pred_probs.shape[1]).float()
    
    # 计算负对数似然损失
    # -log(p_i) 其中 i 是真实类别
    # 为了数值稳定性，使用 log_softmax 和 nll_loss
    log_probs = torch.log(torch.clamp(y_pred_probs, min=1e-10))
    loss = -torch.sum(y_true_one_hot * log_probs) / batch_size
    
    return loss
