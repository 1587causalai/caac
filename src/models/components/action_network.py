"""
Action Network component for CAAC models.

This module implements the ActionNetwork class responsible for 
mapping from latent space to class score distributions.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ActionNetwork(nn.Module):
    """
    Action Network - 行动网络
    
    处理潜在表征随机变量，通过线性变换输出每个类别的得分分布参数。
    概念上：输入随机变量U_j，通过线性变换输出Score随机变量S_k的参数。
    
    Args:
        latent_dim (int): 潜在向量维度
        n_classes (int): 类别数量
        
    Example:
        >>> action_net = ActionNetwork(latent_dim=32, n_classes=3)
        >>> location_param = torch.randn(16, 32)  # batch_size=16
        >>> scale_param = torch.randn(16, 32)
        >>> class_locations, class_scales = action_net.compute_class_distribution_params(
        ...     location_param, scale_param, distribution_type='cauchy'
        ... )
        >>> print(class_locations.shape, class_scales.shape)  # (16, 3), (16, 3)
    """
    
    def __init__(self, latent_dim: int, n_classes: int):
        super(ActionNetwork, self).__init__()
        
        self.linear = nn.Linear(latent_dim, n_classes)
        self.latent_dim = latent_dim
        self.n_classes = n_classes
    
    def forward(self, location_param: torch.Tensor) -> torch.Tensor:
        """
        前向传播（兼容性接口）
        
        Args:
            location_param (torch.Tensor): 位置参数，shape: (batch_size, latent_dim)
            
        Returns:
            torch.Tensor: 线性变换结果，shape: (batch_size, n_classes)
            
        Note:
            这里输入location_param是为了兼容现有架构。
            概念上应该处理随机变量，但实际通过权重矩阵在损失函数中计算分布参数。
        """
        return self.linear(location_param)
    
    def get_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取线性变换的权重和偏置
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - weight: 权重矩阵A，shape: (n_classes, latent_dim)
                - bias: 偏置向量B，shape: (n_classes,)
        """
        weight = self.linear.weight.data  # [n_classes, latent_dim] - 线性变换矩阵A
        bias = self.linear.bias.data      # [n_classes] - 偏置B
        return weight, bias
    
    def compute_class_distribution_params(self, location_param: torch.Tensor, 
                                        scale_param: torch.Tensor, 
                                        distribution_type: str = 'cauchy') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算每个类别Score随机变量的分布参数
        
        不同分布类型使用不同的线性组合规则：
        - Cauchy: scale(S_k) = |W_k| @ scale_param
        - Gaussian: std(S_k) = sqrt(W_k^2 @ scale_param^2)
        
        Args:
            location_param (torch.Tensor): 潜在位置参数，shape: (batch_size, latent_dim)
            scale_param (torch.Tensor): 潜在尺度参数，shape: (batch_size, latent_dim)
            distribution_type (str): 分布类型，'cauchy' 或 'gaussian'
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - class_locations: 类别位置参数，shape: (batch_size, n_classes)
                - class_scales: 类别尺度参数，shape: (batch_size, n_classes)
                
        Raises:
            ValueError: 如果分布类型不支持
        """
        W, b = self.get_weights()
        
        # 位置参数：loc(S_k) = W_k @ location_param + b_k (所有分布相同)
        class_locations = torch.matmul(location_param, W.T) + b.unsqueeze(0)
        
        if distribution_type == 'cauchy':
            # 柯西分布：scale(S_k) = |W_k| @ scale_param
            W_abs = torch.abs(W)
            class_scales = torch.matmul(scale_param, W_abs.T)
            return class_locations, torch.clamp(class_scales, min=1e-6)
            
        elif distribution_type == 'gaussian':
            # 高斯分布：var(S_k) = W_k^2 @ scale_param^2, std(S_k) = sqrt(var)
            W_squared = W ** 2
            class_variances = torch.matmul(scale_param ** 2, W_squared.T)
            class_stds = torch.sqrt(torch.clamp(class_variances, min=1e-6))
            return class_locations, class_stds
            
        else:
            raise ValueError(f"Unsupported distribution type: {distribution_type}")

    def compute_scores_from_samples(self, samples: torch.Tensor) -> torch.Tensor:
        """
        直接从采样的潜在向量计算得分
        
        Args:
            samples (torch.Tensor): 采样的潜在向量，shape: (batch_size, n_samples, latent_dim)
        
        Returns:
            torch.Tensor: 每个采样的得分，shape: (batch_size, n_samples, n_classes)
        """
        W, b = self.get_weights()
        # samples: [batch_size, n_samples, latent_dim]
        # W: [n_classes, latent_dim]
        # 结果: [batch_size, n_samples, n_classes]
        scores = torch.matmul(samples, W.T) + b.unsqueeze(0).unsqueeze(0)
        return scores
    
    def get_config(self) -> dict:
        """
        获取网络配置
        
        Returns:
            dict: 包含网络结构参数的字典
        """
        return {
            'latent_dim': self.latent_dim,
            'n_classes': self.n_classes
        } 