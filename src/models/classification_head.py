"""
分类概率计算模块

提供以下组件：
1. ClassificationHead - 分类头，计算最终的分类概率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassificationHead(nn.Module):
    """
    分类头模块

    根据路径得分、路径概率和阈值计算最终的类别概率。
    Adapting for Softmax-based multi-class (n_classes > 2) 
    and keeping existing logic for binary (n_classes == 2).
    """
    def __init__(self, n_classes):
        super(ClassificationHead, self).__init__()
        self.n_classes = n_classes

    def forward(self, path_class_logits=None, path_probs=None, mu_scores=None, gamma_scores=None, thresholds=None):
        """
        计算最终类别概率。

        Args (for n_classes > 2):
            path_class_logits: 各路径的类别 Logits [batch_size, n_paths, n_classes]
            path_probs: 各路径的选择概率 [batch_size, n_paths]
        
        Args (for n_classes == 2, current logic):
            mu_scores: 各路径的位置参数 [batch_size, n_paths]
            gamma_scores: 各路径的尺度参数 [batch_size, n_paths]
            path_probs: 各路径的选择概率 [batch_size, n_paths]
            thresholds: 类别阈值 [n_classes-1] (effectively [1] for binary)

        Returns:
            class_probs: 最终类别概率 [batch_size, n_classes]
        """
        if self.n_classes > 2:
            if path_class_logits is None or path_probs is None:
                raise ValueError("For n_classes > 2, path_class_logits and path_probs must be provided.")
            
            # 加权平均路径的类别Logits
            # path_class_logits: [batch_size, n_paths, n_classes]
            # path_probs: [batch_size, n_paths]
            # Unsqueeze path_probs to [batch_size, n_paths, 1] for broadcasting
            weighted_logits = torch.sum(path_class_logits * path_probs.unsqueeze(-1), dim=1)
            # weighted_logits: [batch_size, n_classes]
            
            class_probs = F.softmax(weighted_logits, dim=1)
            
        elif self.n_classes == 2:
            if mu_scores is None or gamma_scores is None or path_probs is None or thresholds is None:
                raise ValueError("For n_classes == 2, mu_scores, gamma_scores, path_probs, and thresholds must be provided.")

            batch_size = mu_scores.shape[0]
            n_paths = mu_scores.shape[1]

            # 确保 gamma_scores 为正
            gamma_scores_positive = F.softplus(gamma_scores)

            # 计算每个路径对每个类别的贡献 (使用柯西CDF)
            # thresholds: [1] for binary case (class 0 vs class 1)
            # threshold_val typically 0 for distinguishing <0 and >0
            threshold_val = thresholds[0] # For binary classification, there's one threshold
            
            # Prob of being in class 1 for each path
            # P(X > threshold_val) = 1 - P(X <= threshold_val) = 1 - CDF(threshold_val)
            # CDF_cauchy(x; mu, gamma) = 0.5 + (1/pi) * atan((x - mu) / gamma)
            prob_class1_per_path = 0.5 - (1.0 / torch.pi) * torch.atan((threshold_val - mu_scores) / gamma_scores_positive)
            prob_class1_per_path = prob_class1_per_path.clamp(min=1e-6, max=1.0-1e-6) # Clamp for stability

            # 加权平均路径贡献
            # path_probs: [batch_size, n_paths]
            # prob_class1_per_path: [batch_size, n_paths]
            final_prob_class1 = torch.sum(prob_class1_per_path * path_probs, dim=1, keepdim=True)
            final_prob_class0 = 1.0 - final_prob_class1
            
            class_probs = torch.cat([final_prob_class0, final_prob_class1], dim=1)
        else:
            raise ValueError(f"Unsupported n_classes: {self.n_classes}")
            
        return class_probs
