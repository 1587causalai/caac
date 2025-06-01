"""
OvR概率计算层 (OvR Probability Layer)

计算样本属于每个类别的概率。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class OvRProbabilityLayer(nn.Module):
    """
    OvR概率计算层 (OvR Probability Layer)
    
    基于类别得分随机变量的柯西分布参数，计算样本属于每个类别的概率。
    使用柯西分布的CDF计算概率。
    """
    def __init__(self, n_classes, threshold=0.0):
        """
        初始化OvR概率计算层
        
        Args:
            n_classes: 类别数量
            threshold: 判决阈值，默认为0.0
        """
        super(OvRProbabilityLayer, self).__init__()
        self.n_classes = n_classes
        self.threshold = threshold
        
    def forward(self, loc, scale):
        """
        前向传播
        
        Args:
            loc: 类别得分随机变量的位置参数 [batch_size, n_classes]
            scale: 类别得分随机变量的尺度参数 [batch_size, n_classes]
            
        Returns:
            class_probs: 类别概率 [batch_size, n_classes]
        """
        # 确保scale为正且数值稳定
        scale_stable = torch.clamp(scale, min=1e-6)
        
        # 计算每个类别的概率
        # P_k(z) = 0.5 - (1/pi) * arctan((C_k - loc(S_k; z)) / scale(S_k; z))
        normalized_diff = (self.threshold - loc) / scale_stable
        class_probs = 0.5 - (1.0 / torch.pi) * torch.atan(normalized_diff)
        
        # 确保概率在[0,1]范围内
        class_probs = torch.clamp(class_probs, min=1e-6, max=1.0-1e-6)
        
        return class_probs
