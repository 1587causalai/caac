"""
合成数据生成模块

提供以下组件：
1. SyntheticBinaryClassificationGenerator - 二分类合成数据生成器
"""

import numpy as np
from sklearn.model_selection import train_test_split

class SyntheticBinaryClassificationGenerator:
    """
    二分类合成数据生成器
    
    生成用于二分类任务的合成数据，支持线性和非线性数据，以及异常值注入。
    """
    def __init__(self, n_samples_total=1000, n_features=10, random_state=None):
        self.n_samples_total = n_samples_total
        self.n_features = n_features
        self.rng = np.random.RandomState(random_state)
    
    def generate_linear(self, separation=1.0):
        """
        生成线性可分的二分类数据
        
        参数:
            separation: 类别间的分离度
            
        返回:
            X: 特征矩阵
            y: 类别标签 (0或1)
        """
        X = self.rng.randn(self.n_samples_total, self.n_features)
        
        # 生成随机权重向量
        w = self.rng.randn(self.n_features)
        w = w / np.linalg.norm(w) * separation  # 归一化并调整分离度
        
        # 计算决策值
        decision_values = np.dot(X, w)
        
        # 根据决策值确定类别
        y = (decision_values > 0).astype(int)
        
        return X, y
    
    def generate_nonlinear(self, complexity=1.0):
        """
        生成非线性二分类数据
        
        参数:
            complexity: 非线性复杂度
            
        返回:
            X: 特征矩阵
            y: 类别标签 (0或1)
        """
        X = self.rng.randn(self.n_samples_total, self.n_features)
        
        # 生成非线性决策边界
        decision_values = np.sum(X**2, axis=1) - self.n_features * complexity
        
        # 根据决策值确定类别
        y = (decision_values > 0).astype(int)
        
        return X, y
    
    def inject_outliers(self, X, y, outlier_ratio=0.1):
        """
        注入异常值
        
        参数:
            X: 特征矩阵
            y: 类别标签
            outlier_ratio: 异常值比例
            
        返回:
            X_with_outliers: 包含异常值的特征矩阵
            y_with_outliers: 包含异常值的类别标签
            outlier_mask: 异常值掩码
        """
        X_with_outliers = X.copy()
        y_with_outliers = y.copy()
        
        n_samples = X.shape[0]
        n_outliers = int(n_samples * outlier_ratio)
        
        outlier_indices = self.rng.choice(n_samples, n_outliers, replace=False)
        outlier_mask = np.zeros(n_samples, dtype=bool)
        outlier_mask[outlier_indices] = True
        
        for idx in outlier_indices:
            # 翻转类别标签
            y_with_outliers[idx] = 1 - y_with_outliers[idx]
            
            # 添加特征噪声
            X_with_outliers[idx] += self.rng.randn(self.n_features) * 3.0
        
        return X_with_outliers, y_with_outliers, outlier_mask

def split_data(X, y, test_size=0.2, val_size=0.2, random_state=None):
    """
    将数据集划分为训练集、验证集和测试集
    
    参数:
        X: 特征矩阵
        y: 标签
        test_size: 测试集比例
        val_size: 验证集比例
        random_state: 随机种子
        
    返回:
        X_train: 训练集特征
        X_val: 验证集特征
        X_test: 测试集特征
        y_train: 训练集标签
        y_val: 验证集标签
        y_test: 测试集标签
    """
    # 首先划分出测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 从剩余数据中划分出验证集
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
