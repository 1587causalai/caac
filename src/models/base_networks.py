"""
基础网络模块

提供以下组件：
1. FeatureNetwork - 特征提取网络
2. AbductionNetwork - 推断网络，生成柯西分布参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureNetwork(nn.Module):
    """
    特征网络 (Feature Network)
    
    将输入特征转换为高级表征
    """
    def __init__(self, input_dim, representation_dim, hidden_dims=[64]):
        super(FeatureNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim_i in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim_i))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim_i
        layers.append(nn.Linear(prev_dim, representation_dim))
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.hidden_dims = hidden_dims 

    def forward(self, x):
        return self.network(x)

class AbductionNetwork(nn.Module):
    """
    推断网络 (Abduction Network)
    
    生成柯西分布的因果表征参数
    """
    def __init__(self, representation_dim, latent_dim, hidden_dims=[64, 32]):
        super(AbductionNetwork, self).__init__()
        shared_layers_list = []
        prev_dim = representation_dim
        for hidden_dim_i in hidden_dims:
            shared_layers_list.append(nn.Linear(prev_dim, hidden_dim_i))
            shared_layers_list.append(nn.ReLU())
            prev_dim = hidden_dim_i
        shared_output_dim = prev_dim 
        self.location_head = nn.Linear(shared_output_dim, latent_dim)
        self.scale_head = nn.Linear(shared_output_dim, latent_dim)
        self.shared_mlp = nn.Sequential(*shared_layers_list)
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
            
    def forward(self, representation):
        shared_features = self.shared_mlp(representation)
        location_param = self.location_head(shared_features)
        scale_param = F.softplus(self.scale_head(shared_features))
        return location_param, scale_param
