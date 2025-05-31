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
    
    包含K条并行的"解读路径"。
    对于Softmax分类模式：每条路径从潜在表征生成n_classes个logits。
    同时计算路径选择概率。
    """
    def __init__(self, latent_dim, n_paths, n_classes):
        super(PathwayNetwork, self).__init__()
        self.latent_dim = latent_dim
        self.n_paths = n_paths
        self.n_classes = n_classes
        
        # 每条路径的类别 Logits 生成层
        self.path_class_logit_layers = nn.ModuleList(
            [nn.Linear(latent_dim, n_classes) for _ in range(n_paths)]
        )
        
        # 路径选择概率的原始参数
        self.path_logits_param = nn.Parameter(torch.zeros(n_paths))
        
        # 初始化参数
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        # 初始化每个路径的 Logit 生成层
        for layer in self.path_class_logit_layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            
        # 路径选择概率初始化 - 均匀分布
        with torch.no_grad():
            self.path_logits_param.data.fill_(0.0)
    
    def forward(self, latent_representation):
        # latent_representation: [batch_size, latent_dim]
        batch_size = latent_representation.shape[0]
        
        all_path_class_logits = []
        for i in range(self.n_paths):
            current_path_logits = self.path_class_logit_layers[i](latent_representation)
            all_path_class_logits.append(current_path_logits)
        
        # Stack to get [batch_size, n_paths, n_classes]
        path_class_logits = torch.stack(all_path_class_logits, dim=1)
        
        # 计算路径选择概率
        # [n_paths] -> [batch_size, n_paths]
        path_probs = F.softmax(self.path_logits_param, dim=0).expand(batch_size, -1)
        
        return path_class_logits, path_probs
