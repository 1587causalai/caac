"""
分类概率计算模块

提供以下组件：
1. ClassificationHead - 分类头，计算最终的分类概率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassificationHead(nn.Module):
    """
    分类头 (Classification Head)
    
    基于柯西CDF和固定阈值计算每条路径的分类概率，
    然后通过路径选择概率进行加权平均。
    
    支持二分类和多分类任务。
    """
    def __init__(self, n_classes=2):
        super(ClassificationHead, self).__init__()
        self.n_classes = n_classes
    
    def forward(self, mu_scores, gamma_scores, path_probs, thresholds):
        batch_size, n_paths = mu_scores.shape
        
        # 计算柯西CDF: F(θ) = 0.5 + (1/π) * arctan((θ - μ) / γ)
        def cauchy_cdf(threshold, mu, gamma):
            normalized = (threshold - mu) / gamma
            return 0.5 + (1.0 / np.pi) * torch.atan(normalized)
        
        if self.n_classes == 2:
            # 二分类：只需要一个阈值
            # [batch_size, n_paths]
            cdf_values = cauchy_cdf(thresholds, mu_scores, gamma_scores)
            
            # 计算每条路径的类别概率
            # 类别0的概率: P(Y=0|M=j,x) = F(θ)
            # 类别1的概率: P(Y=1|M=j,x) = 1 - F(θ)
            class_0_probs = cdf_values
            class_1_probs = 1.0 - cdf_values
            
            # 堆叠类别概率
            # [batch_size, n_paths, n_classes]
            path_class_probs = torch.stack([class_0_probs, class_1_probs], dim=2)
            
        else:
            # 多分类：需要 n_classes-1 个阈值
            # thresholds: [n_classes-1]
            
            # 初始化路径类别概率
            # [batch_size, n_paths, n_classes]
            path_class_probs = torch.zeros(batch_size, n_paths, self.n_classes, device=mu_scores.device)
            
            # 扩展阈值维度以便计算
            # mu_scores: [batch_size, n_paths] -> [batch_size, n_paths, 1]
            # gamma_scores: [batch_size, n_paths] -> [batch_size, n_paths, 1]
            mu_expanded = mu_scores.unsqueeze(2)
            gamma_expanded = gamma_scores.unsqueeze(2)
            
            # 计算每个阈值的CDF值
            # thresholds: [n_classes-1] -> [1, 1, n_classes-1]
            thresholds_expanded = thresholds.view(1, 1, -1)
            
            # 计算所有阈值的CDF
            # [batch_size, n_paths, n_classes-1]
            cdf_values = cauchy_cdf(thresholds_expanded, mu_expanded, gamma_expanded)
            
            # 计算每个类别的概率
            # P(Y=0|M=j,x) = F(θ_0)，其中 F(θ_0) = F(θ_1)（第一个阈值）
            path_class_probs[:, :, 0] = cdf_values[:, :, 0]
            
            # P(Y=k|M=j,x) = F(θ_k) - F(θ_{k-1}) for k=1,...,n_classes-2
            for k in range(1, self.n_classes-1):
                path_class_probs[:, :, k] = cdf_values[:, :, k] - cdf_values[:, :, k-1]
            
            # P(Y=n_classes-1|M=j,x) = 1 - F(θ_{n_classes-2})
            path_class_probs[:, :, self.n_classes-1] = 1.0 - cdf_values[:, :, -1]
        
        # 通过路径选择概率加权平均
        # [batch_size, n_paths, 1] * [batch_size, n_paths, n_classes]
        # -> [batch_size, n_paths, n_classes] -> [batch_size, n_classes]
        weighted_probs = path_probs.unsqueeze(2) * path_class_probs
        final_probs = weighted_probs.sum(dim=1)
        
        # 确保概率和为1（数值稳定性）
        final_probs = final_probs / final_probs.sum(dim=1, keepdim=True)
        
        return final_probs
