"""
特征网络 (Feature Network)

提取输入特征的高级表征。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureNetwork(nn.Module):
    """
    特征网络 (Feature Network)
    
    将原始输入特征转换为高维表征。
    """
    def __init__(self, input_dim, representation_dim, hidden_dims=[64]):
        """
        初始化特征网络
        
        Args:
            input_dim: 输入特征维度
            representation_dim: 输出表征维度
            hidden_dims: 隐藏层维度列表
        """
        super(FeatureNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, representation_dim))
        self.network = nn.Sequential(*layers)
        
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.hidden_dims = hidden_dims
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            
        Returns:
            representation: 高维表征 [batch_size, representation_dim]
        """
        return self.network(x)
