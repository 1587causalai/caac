"""
Abduction Network component for CAAC models.

This module implements the AbductionNetwork class responsible for 
inferring latent distribution parameters from representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class AbductionNetwork(nn.Module):
    """
    Abduction Network - 推断网络
    
    负责从表征向量推断潜在柯西向量的分布参数。网络输出位置参数(location)
    和尺度参数(scale)，用于定义潜在空间中的概率分布。
    
    Args:
        representation_dim (int): 输入表征维度
        latent_dim (int): 潜在向量维度
        hidden_dims (List[int]): 隐藏层维度列表，默认为[64, 32]
        
    Example:
        >>> abduction_net = AbductionNetwork(
        ...     representation_dim=64,
        ...     latent_dim=32,
        ...     hidden_dims=[128, 64]
        ... )
        >>> representation = torch.randn(32, 64)
        >>> location, scale = abduction_net(representation)
        >>> print(location.shape, scale.shape)  # (32, 32), (32, 32)
    """
    
    def __init__(self, representation_dim: int, latent_dim: int, 
                 hidden_dims: List[int] = None):
        super(AbductionNetwork, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
            
        # 构建共享的MLP层
        shared_layers = []
        prev_dim = representation_dim
        
        for hidden_dim in hidden_dims:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
            prev_dim = hidden_dim
            
        self.shared_mlp = nn.Sequential(*shared_layers)
        shared_output_dim = prev_dim
        
        # 创建专门的输出头
        self.location_head = nn.Linear(shared_output_dim, latent_dim)
        self.scale_head = nn.Linear(shared_output_dim, latent_dim)
        
        # 保存配置参数
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

    def forward(self, representation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，推断潜在分布参数
        
        Args:
            representation (torch.Tensor): 输入表征，shape: (batch_size, representation_dim)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - location_param: 位置参数，shape: (batch_size, latent_dim)
                - scale_param: 尺度参数，shape: (batch_size, latent_dim)
        """
        # 通过共享MLP提取特征
        shared_features = self.shared_mlp(representation)
        
        # 分别计算位置和尺度参数
        location_param = self.location_head(shared_features)
        scale_param = F.softplus(self.scale_head(shared_features))  # 确保尺度参数为正
        
        return location_param, scale_param
    
    def sample_latent(self, representation: torch.Tensor, 
                     n_samples: int = 1) -> torch.Tensor:
        """
        从推断的分布中采样潜在向量
        
        Args:
            representation (torch.Tensor): 输入表征，shape: (batch_size, representation_dim)
            n_samples (int): 每个样本的采样次数
            
        Returns:
            torch.Tensor: 采样的潜在向量，shape: (batch_size, n_samples, latent_dim)
        """
        location_param, scale_param = self.forward(representation)
        batch_size = representation.size(0)
        
        # 使用重参数化技巧采样柯西分布
        # u = loc + scale * tan(π(ε - 0.5)), 其中 ε ~ Uniform(0,1)
        uniform_samples = torch.rand(batch_size, n_samples, self.latent_dim, 
                                   device=representation.device)
        cauchy_samples = location_param.unsqueeze(1) + scale_param.unsqueeze(1) * \
                        torch.tan(torch.pi * (uniform_samples - 0.5))
                        
        return cauchy_samples
    
    def get_config(self) -> dict:
        """
        获取网络配置
        
        Returns:
            dict: 包含网络结构参数的字典
        """
        return {
            'representation_dim': self.representation_dim,
            'latent_dim': self.latent_dim,
            'hidden_dims': self.hidden_dims
        } 