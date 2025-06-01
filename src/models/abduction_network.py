"""
推断网络 (Abduction Network)

推断共享潜在柯西随机向量的参数。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AbductionNetwork(nn.Module):
    """
    推断网络 (Abduction Network)
    
    基于高维表征推断共享潜在柯西随机向量的参数。
    """
    def __init__(self, representation_dim, latent_dim, hidden_dims=[64, 32]):
        """
        初始化推断网络
        
        Args:
            representation_dim: 输入表征维度
            latent_dim: 潜在柯西向量维度
            hidden_dims: 隐藏层维度列表
        """
        super(AbductionNetwork, self).__init__()
        shared_layers = []
        prev_dim = representation_dim
        for hidden_dim in hidden_dims:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.shared_mlp = nn.Sequential(*shared_layers)
        
        self.location_head = nn.Linear(prev_dim, latent_dim)
        self.scale_head = nn.Linear(prev_dim, latent_dim)
        
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
    def forward(self, representation):
        """
        前向传播
        
        Args:
            representation: 高维表征 [batch_size, representation_dim]
            
        Returns:
            location_param: 潜在柯西向量的位置参数 [batch_size, latent_dim]
            scale_param: 潜在柯西向量的尺度参数 [batch_size, latent_dim]
        """
        shared_features = self.shared_mlp(representation)
        location_param = self.location_head(shared_features)
        # 使用softplus确保尺度参数为正
        scale_param = F.softplus(self.scale_head(shared_features))
        return location_param, scale_param
