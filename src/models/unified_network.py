"""
统一分类网络 (Unified Classification Network)

整合特征网络、推断网络、线性变换层和OvR概率计算层，
构建完整的共享潜在柯西向量的OvR多分类器。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_network import FeatureNetwork
from .abduction_network import AbductionNetwork
from .linear_transformation import LinearTransformationLayer
from .ovr_probability import OvRProbabilityLayer

class UnifiedClassificationNetwork(nn.Module):
    """
    统一分类网络 (Unified Classification Network)
    
    整合特征网络、推断网络、线性变换层和OvR概率计算层，
    构建完整的共享潜在柯西向量的OvR多分类器。
    """
    def __init__(self, input_dim, representation_dim, latent_dim, n_classes,
                 feature_hidden_dims=[64], abduction_hidden_dims=[128, 64],
                 threshold=0.0):
        """
        初始化统一分类网络
        
        Args:
            input_dim: 输入特征维度
            representation_dim: 表征维度
            latent_dim: 潜在柯西向量维度
            n_classes: 类别数量
            feature_hidden_dims: 特征网络隐藏层维度列表
            abduction_hidden_dims: 推断网络隐藏层维度列表
            threshold: 判决阈值，默认为0.0
        """
        super(UnifiedClassificationNetwork, self).__init__()
        
        self.feature_net = FeatureNetwork(input_dim, representation_dim, feature_hidden_dims)
        self.abduction_net = AbductionNetwork(representation_dim, latent_dim, abduction_hidden_dims)
        self.linear_transform = LinearTransformationLayer(latent_dim, n_classes)
        self.ovr_probability = OvRProbabilityLayer(n_classes, threshold)
        
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.threshold = threshold
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            
        Returns:
            class_probs: 类别概率 [batch_size, n_classes]
            loc: 类别得分随机变量的位置参数 [batch_size, n_classes]
            scale: 类别得分随机变量的尺度参数 [batch_size, n_classes]
            location_param: 潜在柯西向量的位置参数 [batch_size, latent_dim]
            scale_param: 潜在柯西向量的尺度参数 [batch_size, latent_dim]
        """
        # 特征提取
        representation = self.feature_net(x)
        
        # 推断潜在柯西向量参数
        location_param, scale_param = self.abduction_net(representation)
        
        # 线性变换到类别得分随机变量
        loc, scale = self.linear_transform(location_param, scale_param)
        
        # 计算类别概率
        class_probs = self.ovr_probability(loc, scale)
        
        return class_probs, loc, scale, location_param, scale_param
    
    def predict(self, x):
        """
        预测类别概率
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            
        Returns:
            class_probs: 类别概率 [batch_size, n_classes]
        """
        class_probs, _, _, _, _ = self.forward(x)
        return class_probs
