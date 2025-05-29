"""
固定阈值机制模块

提供以下组件：
1. FixedThresholdMechanism - 固定阈值机制，确保机制不变性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FixedThresholdMechanism(nn.Module):
    """
    固定阈值机制 (Fixed Threshold Mechanism)
    
    实现全局固定的分类阈值，确保机制不变性。
    对于N分类问题，需要N-1个阈值。
    
    参数化方式：
    - 第一个阈值：直接学习 raw_theta_1
    - 后续阈值：通过正差值累加，θ_k = θ_{k-1} + exp(raw_delta_k)
    """
    def __init__(self, n_classes=2):
        super(FixedThresholdMechanism, self).__init__()
        self.n_classes = n_classes
        self.n_thresholds = n_classes - 1
        
        if self.n_thresholds > 0:
            # 第一个阈值的原始参数
            self.raw_theta_1 = nn.Parameter(torch.zeros(1))
            
            # 多分类时，阈值间的差值参数
            if self.n_thresholds > 1:
                self.raw_delta_thetas = nn.Parameter(torch.zeros(self.n_thresholds - 1))
        
        # 初始化参数
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """初始化阈值参数，使用均匀分布的初始值"""
        with torch.no_grad():
            if self.n_thresholds == 1:
                # 二分类：初始化阈值为0
                self.raw_theta_1.data.fill_(0.0)
            else:
                # 多分类：使用标准柯西分布的分位点初始化
                # 计算均匀分布的分位点
                quantiles = torch.linspace(1.0 / self.n_classes, 
                                         1.0 - 1.0 / self.n_classes, 
                                         self.n_thresholds)
                
                # 转换为柯西分布的分位点
                # 柯西分布的逆CDF: F^{-1}(p) = tan(π(p - 0.5))
                initial_thresholds = torch.tan(np.pi * (quantiles - 0.5))
                
                # 设置第一个阈值
                self.raw_theta_1.data = initial_thresholds[0:1]
                
                # 设置差值参数（取对数因为要通过exp转换）
                if self.n_thresholds > 1:
                    deltas = initial_thresholds[1:] - initial_thresholds[:-1]
                    self.raw_delta_thetas.data = torch.log(deltas)
    
    def forward(self):
        """
        返回有序的分类阈值
        
        返回:
            thresholds: 形状为 [n_thresholds] 的阈值张量
        """
        if self.n_thresholds == 0:
            # 特殊情况：单类别分类（虽然不太可能）
            return torch.tensor([], device=self.raw_theta_1.device)
        
        if self.n_thresholds == 1:
            # 二分类：直接返回第一个阈值
            return self.raw_theta_1
        
        # 多分类：通过累加正差值计算所有阈值
        thresholds = [self.raw_theta_1]
        
        current_threshold = self.raw_theta_1
        for i in range(self.n_thresholds - 1):
            # 使用exp确保差值为正，保证阈值递增
            delta = torch.exp(self.raw_delta_thetas[i])
            current_threshold = current_threshold + delta
            thresholds.append(current_threshold)
        
        # 返回连接的阈值张量
        return torch.cat(thresholds)
    
    def get_thresholds_info(self):
        """
        获取阈值的详细信息（用于调试和可视化）
        
        返回:
            dict: 包含阈值和相关信息的字典
        """
        with torch.no_grad():
            thresholds = self.forward()
            info = {
                'n_classes': self.n_classes,
                'n_thresholds': self.n_thresholds,
                'thresholds': thresholds.cpu().numpy() if self.n_thresholds > 0 else [],
            }
            
            if self.n_thresholds > 1:
                info['deltas'] = torch.exp(self.raw_delta_thetas).cpu().numpy()
            
            return info
