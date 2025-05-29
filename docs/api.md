# API文档

本页面提供了CAAC-SPSFT模型的API文档，包括核心模块的类和函数说明。

## 1. 模型核心组件

### 1.1 FeatureNetwork

特征网络，将输入特征转换为高级表征。

```python
class FeatureNetwork(nn.Module):
    def __init__(self, input_dim, representation_dim, hidden_dims=[64]):
        """
        参数:
            input_dim: 输入特征维度
            representation_dim: 输出表征维度
            hidden_dims: 隐藏层维度列表
        """
        
    def forward(self, x):
        """
        参数:
            x: 输入特征，形状为 [batch_size, input_dim]
            
        返回:
            表征，形状为 [batch_size, representation_dim]
        """
```

### 1.2 AbductionNetwork

推断网络，生成柯西分布的因果表征参数。

```python
class AbductionNetwork(nn.Module):
    def __init__(self, representation_dim, latent_dim, hidden_dims=[64, 32]):
        """
        参数:
            representation_dim: 输入表征维度
            latent_dim: 潜在变量维度
            hidden_dims: 隐藏层维度列表
        """
        
    def forward(self, representation):
        """
        参数:
            representation: 输入表征，形状为 [batch_size, representation_dim]
            
        返回:
            location_param: 位置参数，形状为 [batch_size, latent_dim]
            scale_param: 尺度参数，形状为 [batch_size, latent_dim]
        """
```

### 1.3 PathwayNetwork

多路径网络，包含多条并行的"解读路径"。

```python
class PathwayNetwork(nn.Module):
    def __init__(self, latent_dim, n_paths):
        """
        参数:
            latent_dim: 潜在变量维度
            n_paths: 路径数量
        """
        
    def forward(self, location_param, scale_param):
        """
        参数:
            location_param: 位置参数，形状为 [batch_size, latent_dim]
            scale_param: 尺度参数，形状为 [batch_size, latent_dim]
            
        返回:
            mu_scores: 路径特定的位置参数，形状为 [batch_size, n_paths]
            gamma_scores: 路径特定的尺度参数，形状为 [batch_size, n_paths]
            path_probs: 路径选择概率，形状为 [batch_size, n_paths]
        """
```

### 1.4 FixedThresholdMechanism

固定阈值机制，确保机制不变性。

```python
class FixedThresholdMechanism(nn.Module):
    def __init__(self, n_classes=2):
        """
        参数:
            n_classes: 类别数量
        """
        
    def forward(self):
        """
        返回:
            thresholds: 分类阈值，形状为 [n_classes-1]
        """
```

### 1.5 ClassificationHead

分类头，计算最终的分类概率。

```python
class ClassificationHead(nn.Module):
    def __init__(self, n_classes=2):
        """
        参数:
            n_classes: 类别数量
        """
        
    def forward(self, mu_scores, gamma_scores, path_probs, thresholds):
        """
        参数:
            mu_scores: 路径特定的位置参数，形状为 [batch_size, n_paths]
            gamma_scores: 路径特定的尺度参数，形状为 [batch_size, n_paths]
            path_probs: 路径选择概率，形状为 [batch_size, n_paths]
            thresholds: 分类阈值，形状为 [n_classes-1]
            
        返回:
            final_probs: 分类概率，形状为 [batch_size, n_classes]
        """
```

## 2. 完整模型

### 2.1 CAACModel

完整的CAAC-SPSFT二分类模型。

```python
class CAACModel(nn.Module):
    def __init__(self, input_dim, representation_dim=64, latent_dim=64, 
                 n_paths=2, n_classes=2,
                 feature_hidden_dims=[64], abduction_hidden_dims=[128, 64]):
        """
        参数:
            input_dim: 输入特征维度
            representation_dim: 表征维度
            latent_dim: 潜在变量维度
            n_paths: 路径数量
            n_classes: 类别数量
            feature_hidden_dims: 特征网络隐藏层维度列表
            abduction_hidden_dims: 推断网络隐藏层维度列表
        """
        
    def forward(self, x):
        """
        参数:
            x: 输入特征，形状为 [batch_size, input_dim]
            
        返回:
            class_probs: 分类概率，形状为 [batch_size, n_classes]
            location_param: 位置参数，形状为 [batch_size, latent_dim]
            scale_param: 尺度参数，形状为 [batch_size, latent_dim]
            mu_scores: 路径特定的位置参数，形状为 [batch_size, n_paths]
            gamma_scores: 路径特定的尺度参数，形状为 [batch_size, n_paths]
            path_probs: 路径选择概率，形状为 [batch_size, n_paths]
            thresholds: 分类阈值，形状为 [n_classes-1]
        """
```

### 2.2 CAACModelWrapper

模型包装类，封装训练、验证和预测功能。

```python
class CAACModelWrapper:
    def __init__(self, input_dim, 
                 representation_dim=64, 
                 latent_dim=64, 
                 n_paths=2,
                 n_classes=2,
                 feature_hidden_dims=[64], 
                 abduction_hidden_dims=[128, 64], 
                 lr=0.001, batch_size=32, epochs=100, device=None,
                 early_stopping_patience=None, early_stopping_min_delta=0.0001):
        """
        参数:
            input_dim: 输入特征维度
            representation_dim: 表征维度
            latent_dim: 潜在变量维度
            n_paths: 路径数量
            n_classes: 类别数量
            feature_hidden_dims: 特征网络隐藏层维度列表
            abduction_hidden_dims: 推断网络隐藏层维度列表
            lr: 学习率
            batch_size: 批量大小
            epochs: 训练轮数
            device: 设备（'cuda'或'cpu'）
            early_stopping_patience: 早停耐心值
            early_stopping_min_delta: 早停最小增量
        """
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        """
        训练模型
        
        参数:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征
            y_val: 验证集标签
            verbose: 详细程度
            
        返回:
            self
        """
        
    def predict_proba(self, X):
        """
        预测概率
        
        参数:
            X: 特征
            
        返回:
            预测概率，形状为 [n_samples, n_classes]
        """
        
    def predict(self, X):
        """
        预测类别
        
        参数:
            X: 特征
            
        返回:
            预测类别，形状为 [n_samples]
        """
```

## 3. 实验工具

### 3.1 SyntheticBinaryClassificationGenerator

二分类合成数据生成器。

```python
class SyntheticBinaryClassificationGenerator:
    def __init__(self, n_samples_total=1000, n_features=10, random_state=None):
        """
        参数:
            n_samples_total: 样本总数
            n_features: 特征数量
            random_state: 随机种子
        """
        
    def generate_linear(self, separation=1.0):
        """
        生成线性可分的二分类数据
        
        参数:
            separation: 类别间的分离度
            
        返回:
            X: 特征矩阵
            y: 类别标签 (0或1)
        """
        
    def generate_nonlinear(self, complexity=1.0):
        """
        生成非线性二分类数据
        
        参数:
            complexity: 非线性复杂度
            
        返回:
            X: 特征矩阵
            y: 类别标签 (0或1)
        """
        
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
```

### 3.2 实验函数

```python
def run_binary_classification_experiment(
    n_samples=1000, 
    n_features=10, 
    test_size=0.2,
    val_size=0.2,
    data_type='linear',
    outlier_ratio=0.1,
    random_state=42,
    model_params=None
):
    """
    运行二分类实验
    
    参数:
        n_samples: 样本数量
        n_features: 特征数量
        test_size: 测试集比例
        val_size: 验证集比例
        data_type: 数据类型 ('linear' 或 'nonlinear')
        outlier_ratio: 异常值比例
        random_state: 随机种子
        model_params: 模型参数字典
        
    返回:
        results: 实验结果字典
    """
    
def compare_with_baselines(
    X_train, y_train, X_test, y_test,
    caac_model=None,
    random_state=42
):
    """
    与基线方法比较
    
    参数:
        X_train: 训练集特征
        y_train: 训练集标签
        X_test: 测试集特征
        y_test: 测试集标签
        caac_model: 已训练的CAAC模型 (可选)
        random_state: 随机种子
        
    返回:
        comparison_results: 比较结果字典
    """
```

## 4. 使用示例

### 4.1 基本使用

```python
import numpy as np
from src.models.caac_model import CAACModelWrapper
from src.data.synthetic import SyntheticBinaryClassificationGenerator, split_data

# 生成数据
generator = SyntheticBinaryClassificationGenerator(n_samples_total=1000, n_features=10)
X, y = generator.generate_linear()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# 创建并训练模型
model = CAACModelWrapper(input_dim=10, n_paths=2)
model.fit(X_train, y_train, X_val, y_val)

# 预测
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# 评估
from src.utils.metrics import evaluate_binary_classification
metrics = evaluate_binary_classification(y_test, y_pred, y_pred_proba)
print(metrics)
```

### 4.2 运行完整实验

```python
from src.experiments.binary_classification import run_binary_classification_experiment, compare_with_baselines

# 运行实验
results = run_binary_classification_experiment(
    n_samples=1000,
    n_features=10,
    data_type='linear',
    outlier_ratio=0.1
)

# 与基线方法比较
comparison = compare_with_baselines(
    results['data']['X_train'],
    results['data']['y_train'],
    results['data']['X_test'],
    results['data']['y_test'],
    caac_model=results['model']
)

# 打印比较结果
print(comparison['comparison_df'])
```
