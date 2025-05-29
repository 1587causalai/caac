"""
多路径网络模块

提供以下组件：
1. PathwayNetwork - 多路径网络，包含多条并行的"解读路径"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PathwayNetwork(nn.Module):
    """
    多路径网络 (Pathway Network)
    
    包含K条并行的"解读路径"，每条路径有自己的线性变换参数，
    从因果表征生成路径特定的柯西得分参数。
    """
    def __init__(self, latent_dim, n_paths):
        super(PathwayNetwork, self).__init__()
        self.latent_dim = latent_dim
        self.n_paths = n_paths
        
        # 每条路径的位置参数线性变换
        self.mu_weights = nn.Parameter(torch.randn(n_paths, latent_dim))
        self.mu_biases = nn.Parameter(torch.zeros(n_paths))
        
        # 每条路径的尺度参数线性变换
        self.gamma_weights = nn.Parameter(torch.randn(n_paths, latent_dim) * 0.1)
        self.gamma_biases = nn.Parameter(torch.zeros(n_paths))
        
        # 路径选择概率的原始参数
        self.path_logits = nn.Parameter(torch.zeros(n_paths))
        
        # 初始化参数
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        # 位置参数权重初始化
        nn.init.xavier_normal_(self.mu_weights)
        
        # 位置参数偏置初始化 - 使不同路径有不同的基线
        for j in range(self.n_paths):
            self.mu_biases.data[j] = j * 0.5
            
        # 尺度参数权重初始化 - 小值初始化
        with torch.no_grad():
            self.gamma_weights.data.uniform_(-0.1, 0.1)
            
        # 尺度参数偏置初始化 - 确保正值
        with torch.no_grad():
            self.gamma_biases.data.fill_(0.0)
            
        # 路径选择概率初始化 - 均匀分布
        with torch.no_grad():
            self.path_logits.data.fill_(0.0)
    
    def forward(self, location_param, scale_param):
        batch_size = location_param.shape[0]
        
        # 计算每条路径的位置参数
        # [batch_size, latent_dim] @ [n_paths, latent_dim].t() -> [batch_size, n_paths]
        mu_scores = torch.matmul(location_param, self.mu_weights.t()) + self.mu_biases
        
        # 计算每条路径的尺度参数
        # [batch_size, latent_dim] @ [n_paths, latent_dim].t() -> [batch_size, n_paths]
        gamma_base = torch.matmul(scale_param, torch.abs(self.gamma_weights).t())
        gamma_scores = gamma_base + torch.exp(self.gamma_biases)
        
        # 计算路径选择概率
        # [n_paths] -> [batch_size, n_paths]
        path_probs = F.softmax(self.path_logits, dim=0).expand(batch_size, -1)
        
        return mu_scores, gamma_scores, path_probs
