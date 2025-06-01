"""
线性变换层 (Linear Transformation Layer)

将潜在柯西向量映射到各类别得分随机变量。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LinearTransformationLayer(nn.Module):
    """
    线性变换层 (Linear Transformation Layer)
    
    将潜在柯西向量映射到各类别得分随机变量。
    利用柯西分布的线性组合特性，确保类别得分随机变量仍然服从柯西分布。
    """
    def __init__(self, latent_dim, n_classes):
        """
        初始化线性变换层
        
        Args:
            latent_dim: 潜在柯西向量维度
            n_classes: 类别数量
        """
        super(LinearTransformationLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_classes, latent_dim))
        self.bias = nn.Parameter(torch.Tensor(n_classes))
        self.reset_parameters()
        
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        
    def reset_parameters(self):
        """
        重置参数
        
        使用Kaiming初始化权重，均匀分布初始化偏置。
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, location_param, scale_param):
        """
        前向传播
        
        Args:
            location_param: 潜在柯西向量的位置参数 [batch_size, latent_dim]
            scale_param: 潜在柯西向量的尺度参数 [batch_size, latent_dim]
            
        Returns:
            loc: 类别得分随机变量的位置参数 [batch_size, n_classes]
            scale: 类别得分随机变量的尺度参数 [batch_size, n_classes]
        """
        # 计算类别得分随机变量的位置参数
        # loc(S_k; z) = sum_{j=1}^M A_{kj} * mu_j(z) + B_k
        loc = F.linear(location_param, self.weight, self.bias)
        
        # 计算类别得分随机变量的尺度参数
        # scale(S_k; z) = sum_{j=1}^M |A_{kj}| * sigma_j(z)
        weight_abs = torch.abs(self.weight)
        scale = F.linear(scale_param, weight_abs)
        
        return loc, scale
