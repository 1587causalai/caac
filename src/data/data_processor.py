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
    def split_data(X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
        """
        分割数据为训练、验证和测试集
        
        Args:
            X: 特征
            y: 标签
            train_size: 训练集比例 (默认0.7)
            val_size: 验证集比例 (默认0.15) 
            test_size: 测试集比例 (默认0.15)
            random_state: 随机种子
            
        Returns:
            X_train: 训练特征
            X_val: 验证特征
            X_test: 测试特征
            y_train: 训练标签
            y_val: 验证标签
            y_test: 测试标签
        """
        # 确保比例总和为1
        total_size = train_size + val_size + test_size
        if abs(total_size - 1.0) > 1e-6:
            raise ValueError(f"train_size + val_size + test_size should equal 1.0, got {total_size}")
        
        # 首先分割出测试集 (15%)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # 从剩余数据中分割出验证集
        # val在train_val中的比例 = val_size / (train_size + val_size)
        val_ratio_in_train_val = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=val_ratio_in_train_val,
            random_state=random_state,
            stratify=y_train_val if len(np.unique(y_train_val)) > 1 else None
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
    
    @staticmethod
    def add_outliers_to_regression_data(X, y, outlier_ratio=0.1, outlier_strength=3.0, random_state=42):
        """
        向回归数据添加outliers
        
        Args:
            X: 特征数据
            y: 目标变量 
            outlier_ratio: outlier比例 (0.0-1.0)
            outlier_strength: outlier强度倍数 (越大偏离越远)
            random_state: 随机种子
            
        Returns:
            X_with_outliers: 含outliers的特征数据
            y_with_outliers: 含outliers的目标变量
            outlier_indices: outlier的索引
        """
        np.random.seed(random_state)
        
        X_outliers = X.copy()
        y_outliers = y.copy()
        
        n_samples = len(y)
        n_outliers = max(1, int(n_samples * outlier_ratio))
        
        # 随机选择outlier位置
        outlier_indices = np.random.choice(n_samples, size=n_outliers, replace=False)
        
        # 计算目标变量的统计信息
        y_mean = np.mean(y)
        y_std = np.std(y)
        
        # 方法1: 目标变量outliers - 加入极值
        for idx in outlier_indices:
            # 随机选择偏离方向 (正或负)
            direction = np.random.choice([-1, 1])
            # 生成强outlier值
            y_outliers[idx] = y_mean + direction * outlier_strength * y_std
        
        # 方法2: 特征outliers - 对部分特征添加噪声
        if X.shape[1] > 0:
            for idx in outlier_indices:
                # 随机选择1-3个特征进行扰动
                n_features_to_perturb = np.random.randint(1, min(4, X.shape[1] + 1))
                features_to_perturb = np.random.choice(X.shape[1], size=n_features_to_perturb, replace=False)
                
                for feat_idx in features_to_perturb:
                    feat_std = np.std(X[:, feat_idx])
                    feat_mean = np.mean(X[:, feat_idx])
                    
                    # 添加强噪声
                    direction = np.random.choice([-1, 1])
                    X_outliers[idx, feat_idx] = feat_mean + direction * outlier_strength * feat_std
        
        return X_outliers, y_outliers, outlier_indices
    
    @staticmethod
    def add_outliers_to_classification_data(X, y, outlier_ratio=0.1, random_state=42, balance_strategy='proportional'):
        """
        向分类数据添加outliers（通过随机改变标签）
        
        Args:
            X: 特征数据
            y: 分类标签
            outlier_ratio: outlier比例 (0.0-1.0)
            random_state: 随机种子
            balance_strategy: 处理类别不平衡的策略
                - 'proportional': 按原始类别比例分配outliers
                - 'uniform': 各类别均匀分配outliers
                - 'majority_to_minority': 主要类别样本改为少数类别
                - 'random': 完全随机改变标签
                
        Returns:
            X_with_outliers: 含outliers的特征数据 (与原始相同，因为只改标签)
            y_with_outliers: 含outliers的标签
            outlier_indices: outlier的索引
            outlier_info: outlier的详细信息
        """
        np.random.seed(random_state)
        
        X_outliers = X.copy()
        y_outliers = y.copy()
        
        n_samples = len(y)
        n_outliers = max(1, int(n_samples * outlier_ratio))
        
        # 获取类别信息
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        
        if n_classes < 2:
            raise ValueError("Need at least 2 classes for classification outliers")
        
        # 计算原始类别分布
        class_counts = {}
        for cls in unique_classes:
            class_counts[cls] = np.sum(y == cls)
        
        # 随机选择outlier位置
        outlier_indices = np.random.choice(n_samples, size=n_outliers, replace=False)
        
        # 根据不同策略改变标签
        if balance_strategy == 'random':
            # 完全随机改变标签
            for idx in outlier_indices:
                original_label = y_outliers[idx]
                # 从其他类别中随机选择
                other_classes = unique_classes[unique_classes != original_label]
                y_outliers[idx] = np.random.choice(other_classes)
                
        elif balance_strategy == 'majority_to_minority':
            # 找到多数类和少数类
            majority_class = max(class_counts, key=class_counts.get)
            minority_class = min(class_counts, key=class_counts.get)
            
            for idx in outlier_indices:
                original_label = y_outliers[idx]
                if original_label == majority_class:
                    # 多数类改为少数类
                    y_outliers[idx] = minority_class
                else:
                    # 其他类别改为多数类
                    y_outliers[idx] = majority_class
                    
        elif balance_strategy == 'uniform':
            # 均匀分配到各个错误类别
            for i, idx in enumerate(outlier_indices):
                original_label = y_outliers[idx]
                other_classes = unique_classes[unique_classes != original_label]
                # 轮流分配到其他类别
                target_class = other_classes[i % len(other_classes)]
                y_outliers[idx] = target_class
                
        elif balance_strategy == 'proportional':
            # 按原始类别比例分配outliers
            class_probs = np.array([class_counts[cls] for cls in unique_classes]) / n_samples
            
            for idx in outlier_indices:
                original_label = y_outliers[idx]
                # 移除原始类别，重新归一化概率
                other_classes = unique_classes[unique_classes != original_label]
                other_indices = [i for i, cls in enumerate(unique_classes) if cls != original_label]
                other_probs = class_probs[other_indices]
                other_probs = other_probs / np.sum(other_probs)  # 重新归一化
                
                # 按概率选择新类别
                target_class = np.random.choice(other_classes, p=other_probs)
                y_outliers[idx] = target_class
        
        else:
            raise ValueError(f"Unknown balance_strategy: {balance_strategy}")
        
        # 统计outlier信息
        outlier_changes = {}
        for idx in outlier_indices:
            original = y[idx]
            new = y_outliers[idx]
            if original not in outlier_changes:
                outlier_changes[original] = {}
            if new not in outlier_changes[original]:
                outlier_changes[original][new] = 0
            outlier_changes[original][new] += 1
        
        outlier_info = {
            'total_outliers': len(outlier_indices),
            'outlier_ratio': outlier_ratio,
            'balance_strategy': balance_strategy,
            'outlier_indices': outlier_indices,
            'class_changes': outlier_changes,
            'original_class_distribution': class_counts,
            'new_class_distribution': {cls: np.sum(y_outliers == cls) for cls in unique_classes}
        }
        
        return X_outliers, y_outliers, outlier_indices, outlier_info
    
    @staticmethod
    def split_data_with_outliers(X, y, train_size=0.7, val_size=0.15, test_size=0.15, 
                                outlier_ratio=0.1, outlier_strength=3.0, random_state=42):
        """
        分割数据并在train+val中添加outliers，保持test干净
        
        Args:
            X: 特征
            y: 标签/目标变量
            train_size: 训练集比例 (默认0.7)
            val_size: 验证集比例 (默认0.15)
            test_size: 测试集比例 (默认0.15)
            outlier_ratio: outlier比例 (仅在train+val中)
            outlier_strength: outlier强度倍数
            random_state: 随机种子
            
        Returns:
            X_train: 训练特征 (含outliers)
            X_val: 验证特征 (含outliers) 
            X_test: 测试特征 (干净)
            y_train: 训练标签 (含outliers)
            y_val: 验证标签 (含outliers)
            y_test: 测试标签 (干净)
            outlier_info: outlier信息字典
        """
        # 首先进行正常的数据分割
        X_train, X_val, X_test, y_train, y_val, y_test = DataProcessor.split_data(
            X, y, train_size, val_size, test_size, random_state
        )
        
        # 合并train和val数据来添加outliers
        X_train_val_combined = np.vstack([X_train, X_val])
        y_train_val_combined = np.concatenate([y_train, y_val])
        
        # 在合并的train+val数据中添加outliers
        X_train_val_outliers, y_train_val_outliers, outlier_indices = DataProcessor.add_outliers_to_regression_data(
            X_train_val_combined, y_train_val_combined, 
            outlier_ratio, outlier_strength, random_state
        )
        
        # 重新分割含outliers的train+val数据
        n_train = len(X_train)
        X_train_final = X_train_val_outliers[:n_train]
        X_val_final = X_train_val_outliers[n_train:]
        y_train_final = y_train_val_outliers[:n_train]
        y_val_final = y_train_val_outliers[n_train:]
        
        # 计算outlier统计信息
        outlier_info = {
            'total_outliers': len(outlier_indices),
            'outlier_ratio': outlier_ratio,
            'outlier_strength': outlier_strength,
            'outliers_in_train': np.sum(outlier_indices < n_train),
            'outliers_in_val': np.sum(outlier_indices >= n_train),
            'outlier_indices_global': outlier_indices,  # 在合并数据中的索引
            'train_size': len(X_train_final),
            'val_size': len(X_val_final), 
            'test_size': len(X_test)
        }
        
        return X_train_final, X_val_final, X_test, y_train_final, y_val_final, y_test, outlier_info
    
    @staticmethod
    def split_classification_data_with_outliers(X, y, train_size=0.7, val_size=0.15, test_size=0.15, 
                                              outlier_ratio=0.1, balance_strategy='proportional', random_state=42):
        """
        分割分类数据并在train+val中添加outliers（通过改变标签），保持test干净
        
        Args:
            X: 特征
            y: 分类标签
            train_size: 训练集比例 (默认0.7)
            val_size: 验证集比例 (默认0.15)
            test_size: 测试集比例 (默认0.15)
            outlier_ratio: outlier比例 (仅在train+val中)
            balance_strategy: 处理类别不平衡的策略
            random_state: 随机种子
            
        Returns:
            X_train: 训练特征 (原始，只有标签被修改)
            X_val: 验证特征 (原始，只有标签被修改)
            X_test: 测试特征 (干净)
            y_train: 训练标签 (含outliers)
            y_val: 验证标签 (含outliers)
            y_test: 测试标签 (干净)
            outlier_info: outlier信息字典
        """
        # 首先进行正常的数据分割
        X_train, X_val, X_test, y_train, y_val, y_test = DataProcessor.split_data(
            X, y, train_size, val_size, test_size, random_state
        )
        
        # 合并train和val数据来添加outliers
        X_train_val_combined = np.vstack([X_train, X_val])
        y_train_val_combined = np.concatenate([y_train, y_val])
        
        # 在合并的train+val数据中添加outliers (只改变标签)
        X_train_val_outliers, y_train_val_outliers, outlier_indices, outlier_details = DataProcessor.add_outliers_to_classification_data(
            X_train_val_combined, y_train_val_combined, 
            outlier_ratio, random_state, balance_strategy
        )
        
        # 重新分割含outliers的train+val数据
        n_train = len(X_train)
        X_train_final = X_train_val_outliers[:n_train]  # 特征不变
        X_val_final = X_train_val_outliers[n_train:]    # 特征不变
        y_train_final = y_train_val_outliers[:n_train]  # 标签可能被修改
        y_val_final = y_train_val_outliers[n_train:]    # 标签可能被修改
        
        # 计算各数据集中的outlier统计
        outliers_in_train = np.sum(outlier_indices < n_train)
        outliers_in_val = np.sum(outlier_indices >= n_train)
        
        # 增强outlier信息
        outlier_info = {
            **outlier_details,  # 包含原有的详细信息
            'outliers_in_train': outliers_in_train,
            'outliers_in_val': outliers_in_val,
            'train_size': len(X_train_final),
            'val_size': len(X_val_final), 
            'test_size': len(X_test),
            'train_class_distribution': {cls: np.sum(y_train_final == cls) for cls in np.unique(y)},
            'val_class_distribution': {cls: np.sum(y_val_final == cls) for cls in np.unique(y)},
            'test_class_distribution': {cls: np.sum(y_test == cls) for cls in np.unique(y)}
        }
        
        return X_train_final, X_val_final, X_test, y_train_final, y_val_final, y_test, outlier_info

    @staticmethod
    def inject_label_noise(y, noise_level, noise_type='random_uniform', random_state=42):
        """
        向标签中注入噪声的统一接口
        
        Args:
            y: 原始标签
            noise_level: 噪声比例 (0.0-1.0)
            noise_type: 噪声类型
                - 'random_uniform': 随机均匀注入 - 随机选择非真实标签
                - 'proportional': 按原始类别比例选择错误标签
                - 'majority_bias': 偏向多数类的错误标签
                - 'minority_bias': 偏向少数类的错误标签
                - 'adjacent': 选择相邻标签（仅适用于有序标签）
                - 'flip_pairs': 成对翻转（两个类别互相翻转）
            random_state: 随机种子
            
        Returns:
            y_noisy: 含噪声的标签
            noise_info: 噪声注入信息
        """
        if noise_level == 0:
            return y.copy(), {'noise_level': 0, 'noise_type': noise_type, 'changes': 0}
        
        np.random.seed(random_state)
        
        y_noisy = y.copy()
        n_samples = len(y)
        n_noisy = int(n_samples * noise_level)
        
        # 获取类别信息
        unique_labels = np.unique(y)
        n_classes = len(unique_labels)
        
        if n_classes < 2:
            raise ValueError("Need at least 2 classes for label noise injection")
        
        # 计算类别分布
        class_counts = {label: np.sum(y == label) for label in unique_labels}
        total_samples = len(y)
        
        # 随机选择要添加噪声的样本
        noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
        
        # 记录标签变化
        label_changes = {}
        
        for idx in noisy_indices:
            original_label = y_noisy[idx]
            
            if noise_type == 'random_uniform':
                # 随机均匀注入：从其他标签中随机选择
                possible_labels = unique_labels[unique_labels != original_label]
                new_label = np.random.choice(possible_labels)
                
            elif noise_type == 'proportional':
                # 按原始类别比例选择错误标签
                other_labels = unique_labels[unique_labels != original_label]
                other_counts = np.array([class_counts[label] for label in other_labels])
                other_probs = other_counts / np.sum(other_counts)
                new_label = np.random.choice(other_labels, p=other_probs)
                
            elif noise_type == 'majority_bias':
                # 偏向多数类
                majority_label = max(class_counts, key=class_counts.get)
                if original_label == majority_label:
                    # 如果原标签就是多数类，随机选择其他
                    other_labels = unique_labels[unique_labels != original_label]
                    new_label = np.random.choice(other_labels)
                else:
                    # 否则改为多数类
                    new_label = majority_label
                    
            elif noise_type == 'minority_bias':
                # 偏向少数类
                minority_label = min(class_counts, key=class_counts.get)
                if original_label == minority_label:
                    # 如果原标签就是少数类，随机选择其他
                    other_labels = unique_labels[unique_labels != original_label]
                    new_label = np.random.choice(other_labels)
                else:
                    # 否则改为少数类
                    new_label = minority_label
                    
            elif noise_type == 'adjacent':
                # 相邻标签噪声（假设标签是有序的）
                if len(unique_labels) > 2:
                    # 找到当前标签在排序后的位置
                    sorted_labels = np.sort(unique_labels)
                    current_pos = np.where(sorted_labels == original_label)[0][0]
                    
                    # 选择相邻位置
                    adjacent_positions = []
                    if current_pos > 0:
                        adjacent_positions.append(current_pos - 1)
                    if current_pos < len(sorted_labels) - 1:
                        adjacent_positions.append(current_pos + 1)
                    
                    if adjacent_positions:
                        chosen_pos = np.random.choice(adjacent_positions)
                        new_label = sorted_labels[chosen_pos]
                    else:
                        # 如果没有相邻标签，随机选择
                        other_labels = unique_labels[unique_labels != original_label]
                        new_label = np.random.choice(other_labels)
                else:
                    # 只有两个类别，直接翻转
                    other_labels = unique_labels[unique_labels != original_label]
                    new_label = other_labels[0]
                    
            elif noise_type == 'flip_pairs':
                # 成对翻转（主要针对二分类）
                if n_classes == 2:
                    # 二分类直接翻转
                    other_labels = unique_labels[unique_labels != original_label]
                    new_label = other_labels[0]
                else:
                    # 多分类情况下，随机配对翻转
                    other_labels = unique_labels[unique_labels != original_label]
                    new_label = np.random.choice(other_labels)
                    
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")
            
            y_noisy[idx] = new_label
            
            # 记录标签变化
            if original_label not in label_changes:
                label_changes[original_label] = {}
            if new_label not in label_changes[original_label]:
                label_changes[original_label][new_label] = 0
            label_changes[original_label][new_label] += 1
        
        # 计算噪声后的类别分布
        new_class_counts = {label: np.sum(y_noisy == label) for label in unique_labels}
        
        noise_info = {
            'noise_level': noise_level,
            'noise_type': noise_type,
            'total_samples': n_samples,
            'noisy_samples': n_noisy,
            'noisy_indices': noisy_indices,
            'label_changes': label_changes,
            'original_distribution': class_counts,
            'new_distribution': new_class_counts,
            'changes': len(noisy_indices)
        }
        
        return y_noisy, noise_info
