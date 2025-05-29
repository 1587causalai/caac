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
    """
    def __init__(self, n_classes=2):
        super(ClassificationHead, self).__init__()
        self.n_classes = n_classes
    
    def forward(self, mu_scores, gamma_scores, path_probs, thresholds):
        batch_size, n_paths = mu_scores.shape
        
        # 对于二分类，计算每条路径的类别概率
        if self.n_classes == 2:
            # 计算柯西CDF: F(θ) = 0.5 + (1/π) * arctan((θ - μ) / γ)
            # [batch_size, n_paths]
            normalized_threshold = (thresholds - mu_scores) / gamma_scores
            cdf_values = 0.5 + (1.0 / np.pi) * torch.atan(normalized_threshold)
            
            # 计算每条路径的类别概率
            # 类别1的概率: P(Y=1|M=j,x) = F(θ)
            # 类别2的概率: P(Y=2|M=j,x) = 1 - F(θ)
            class_1_probs = cdf_values
            class_2_probs = 1.0 - cdf_values
            
            # 堆叠类别概率
            # [batch_size, n_paths, n_classes]
            path_class_probs = torch.stack([class_1_probs, class_2_probs], dim=2)
            
            # 通过路径选择概率加权平均
            # [batch_size, n_paths, 1] * [batch_size, n_paths, n_classes]
            # -> [batch_size, n_paths, n_classes] -> [batch_size, n_classes]
            weighted_probs = path_probs.unsqueeze(2) * path_class_probs
            final_probs = weighted_probs.sum(dim=1)
            
            return final_probs
        else:
            # 多分类的实现（未来扩展）
            raise NotImplementedError("多分类尚未实现")
