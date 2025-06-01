"""
数据处理模块

提供以下功能：
1. 数据集加载
2. 数据预处理
3. 数据分割
4. 数据转换
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    """
    数据处理类
    
    提供数据集加载、预处理、分割和转换功能。
    """
    
    @staticmethod
    def load_dataset(dataset_name, random_state=42):
        """
        加载数据集
        
        Args:
            dataset_name: 数据集名称，支持 'iris', 'wine', 'digits', 'breast_cancer'
            random_state: 随机种子
            
        Returns:
            X: 特征
            y: 标签
            feature_names: 特征名称
            target_names: 标签名称
        """
        if dataset_name == 'iris':
            data = load_iris()
        elif dataset_name == 'wine':
            data = load_wine()
        elif dataset_name == 'digits':
            data = load_digits()
        elif dataset_name == 'breast_cancer':
            data = load_breast_cancer()
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        X = data.data
        y = data.target
        feature_names = data.feature_names
        target_names = data.target_names
        
        return X, y, feature_names, target_names
    
    @staticmethod
    def preprocess_data(X, standardize=True):
        """
        预处理数据
        
        Args:
            X: 特征
            standardize: 是否标准化
            
        Returns:
            X_processed: 预处理后的特征
            scaler: 标准化器（如果standardize=True）
        """
        if standardize:
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X)
            return X_processed, scaler
        else:
            return X, None
    
    @staticmethod
    def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
        """
        分割数据
        
        Args:
            X: 特征
            y: 标签
            test_size: 测试集比例
            val_size: 验证集比例（相对于训练集）
            random_state: 随机种子
            
        Returns:
            X_train: 训练特征
            X_val: 验证特征
            X_test: 测试特征
            y_train: 训练标签
            y_val: 验证标签
            y_test: 测试标签
        """
        # 首先分割出测试集
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 然后从剩余数据中分割出验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=val_size/(1-test_size),  # 调整验证集比例
            random_state=random_state,
            stratify=y_train_val
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    @staticmethod
    def get_class_distribution(y):
        """
        获取类别分布
        
        Args:
            y: 标签
            
        Returns:
            class_counts: 类别计数
            class_distribution: 类别分布百分比
        """
        unique_classes, class_counts = np.unique(y, return_counts=True)
        class_distribution = class_counts / len(y) * 100
        
        # 转换为Python原生数据类型以避免JSON序列化问题
        class_counts_dict = {int(cls): int(count) for cls, count in zip(unique_classes, class_counts)}
        class_distribution_dict = {int(cls): float(dist) for cls, dist in zip(unique_classes, class_distribution)}
        
        return class_counts_dict, class_distribution_dict
