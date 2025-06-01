"""
Feature Network component for CAAC models.

This module implements the FeatureNetwork class responsible for 
feature extraction and representation learning.
"""

import torch
import torch.nn as nn
from typing import List


class FeatureNetwork(nn.Module):
    """
    Feature Network - 特征网络
    
    负责从原始输入特征提取高维表征。使用多层感知机(MLP)架构，
    通过可配置的隐藏层将输入映射到固定维度的表征空间。
    
    Args:
        input_dim (int): 输入特征维度
        representation_dim (int): 输出表征维度
        hidden_dims (List[int]): 隐藏层维度列表，默认为[64]
        
    Example:
        >>> feature_net = FeatureNetwork(
        ...     input_dim=10,
        ...     representation_dim=64,
        ...     hidden_dims=[128, 64]
        ... )
        >>> x = torch.randn(32, 10)  # batch_size=32
        >>> representation = feature_net(x)  # shape: (32, 64)
    """
    
    def __init__(self, input_dim: int, representation_dim: int, 
                 hidden_dims: List[int] = None):
        super(FeatureNetwork, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [64]
            
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        # 添加隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
            
        # 添加输出层
        layers.append(nn.Linear(prev_dim, representation_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 保存配置参数
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.hidden_dims = hidden_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征，shape: (batch_size, input_dim)
            
        Returns:
            torch.Tensor: 表征向量，shape: (batch_size, representation_dim)
        """
        return self.network(x)
    
    def get_config(self) -> dict:
        """
        获取网络配置
        
        Returns:
            dict: 包含网络结构参数的字典
        """
        return {
            'input_dim': self.input_dim,
            'representation_dim': self.representation_dim,
            'hidden_dims': self.hidden_dims
        } 