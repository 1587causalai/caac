"""
固定阈值机制模块

提供以下组件：
1. FixedThresholdMechanism - 固定阈值机制，确保机制不变性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FixedThresholdMechanism(nn.Module):
    """
    固定阈值机制 (Fixed Threshold Mechanism)
    
    实现全局固定的分类阈值，确保机制不变性。
    对于二分类问题，只需要一个阈值θ*。
    """
    def __init__(self, n_classes=2):
        super(FixedThresholdMechanism, self).__init__()
        self.n_classes = n_classes
        self.n_thresholds = n_classes - 1
        
        # 对于二分类，只需要一个阈值
        if self.n_thresholds > 0:
            # 第一个阈值的原始参数
            self.raw_theta_1 = nn.Parameter(torch.zeros(1))
            
            # 如果是多分类，还需要阈值间的差值参数
            if self.n_thresholds > 1:
                self.raw_delta_thetas = nn.Parameter(torch.zeros(self.n_thresholds - 1))
        
        # 初始化参数
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        # 对于二分类，初始化第一个阈值为0
        with torch.no_grad():
            self.raw_theta_1.data.fill_(0.0)
            
        # 如果是多分类，初始化阈值间的差值
        if self.n_thresholds > 1:
            with torch.no_grad():
                self.raw_delta_thetas.data.fill_(1.0)
    
    def forward(self):
        # 对于二分类，直接返回第一个阈值
        if self.n_thresholds == 1:
            return self.raw_theta_1
        
        # 对于多分类，计算所有阈值
        thresholds = [self.raw_theta_1]
        
        # 通过累加正差值计算后续阈值
        current_threshold = self.raw_theta_1
        for i in range(self.n_thresholds - 1):
            delta = torch.exp(self.raw_delta_thetas[i])  # 确保差值为正
            current_threshold = current_threshold + delta
            thresholds.append(current_threshold)
        
        return torch.cat(thresholds)
