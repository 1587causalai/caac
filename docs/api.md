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

固定阈值机制，确保机制不变性。**支持多分类**。

```python
class FixedThresholdMechanism(nn.Module):
    def __init__(self, n_classes=2):
        """
        参数:
            n_classes: 类别数量（2-N的任意整数）
        """
        
    def forward(self):
        """
        返回:
            thresholds: 分类阈值，形状为 [n_classes-1]
        """
        
    def get_thresholds_info(self):
        """
        获取阈值的详细信息（用于调试和可视化）
        
        返回:
            dict: 包含阈值和相关信息的字典
        """
```

### 1.5 ClassificationHead

分类头，计算最终的分类概率。**支持多分类**。

```python
class ClassificationHead(nn.Module):
    def __init__(self, n_classes=2):
        """
        参数:
            n_classes: 类别数量（2-N的任意整数）
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

完整的CAAC-SPSFT分类模型，支持二分类和多分类。

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
            n_classes: 类别数量（2-N的任意整数）
            feature_hidden_dims: 特征网络隐藏层维度
            abduction_hidden_dims: 推断网络隐藏层维度
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
            n_classes: 类别数量（2-N的任意整数）
            feature_hidden_dims: 特征网络隐藏层维度
            abduction_hidden_dims: 推断网络隐藏层维度
            lr: 学习率
            batch_size: 批次大小
            epochs: 训练轮次
            device: 计算设备
            early_stopping_patience: 早停耐心值
            early_stopping_min_delta: 早停最小改善值
        """
```

## 3. 评估指标

### 3.1 二分类评估

```python
def evaluate_binary_classification(y_true, y_pred, y_pred_proba=None):
    """
    评估二分类模型性能
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        y_pred_proba: 预测概率 (可选，用于计算AUC)
        
    返回:
        metrics: 包含准确率、精确率、召回率、F1分数、AUC的字典
    """
```

### 3.2 多分类评估

```python
def evaluate_multiclass_classification(y_true, y_pred, y_pred_proba=None):
    """
    评估多分类模型性能
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        y_pred_proba: 预测概率矩阵 [n_samples, n_classes] (可选)
        
    返回:
        metrics: 包含以下指标的字典：
            - accuracy: 准确率
            - precision_macro/weighted: 宏平均/加权平均精确率
            - recall_macro/weighted: 宏平均/加权平均召回率
            - f1_macro/weighted: 宏平均/加权平均F1分数
            - precision_class_i: 每个类别的精确率
            - recall_class_i: 每个类别的召回率
            - f1_class_i: 每个类别的F1分数
            - auc_ovr: One-vs-Rest AUC
            - auc_ovo: One-vs-One AUC
            - confusion_matrix: 混淆矩阵
    """
```

## 4. 实验模块

### 4.1 二分类实验

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
    
    参数和返回值详见源码
    """
```

### 4.2 多分类实验

```python
def run_multiclass_classification_experiment(
    n_samples=1000,
    n_features=10,
    n_classes=3,
    test_size=0.2,
    val_size=0.2,
    class_sep=1.0,
    outlier_ratio=0.0,
    random_state=42,
    model_params=None
):
    """
    运行多分类实验
    
    参数:
        n_samples: 样本数量
        n_features: 特征数量
        n_classes: 类别数量（3-N）
        test_size: 测试集比例
        val_size: 验证集比例
        class_sep: 类别分离度
        outlier_ratio: 异常值比例
        random_state: 随机种子
        model_params: 模型参数字典
        
    返回:
        results: 实验结果字典
    """
```

## 5. 使用示例

### 5.1 二分类任务

```python
from src.models.caac_model import CAACModelWrapper
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 生成数据
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 创建并训练模型
model = CAACModelWrapper(
    input_dim=20,
    n_classes=2,
    n_paths=2
)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
```

### 5.2 多分类任务

```python
# 生成5分类数据
X, y = make_classification(n_samples=2000, n_features=20, n_classes=5, n_informative=15)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 创建并训练模型
model = CAACModelWrapper(
    input_dim=20,
    n_classes=5,
    n_paths=5,  # 建议路径数等于类别数
    epochs=150,
    early_stopping_patience=15
)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# 评估
from src.utils.metrics import evaluate_multiclass_classification
metrics = evaluate_multiclass_classification(y_test, y_pred, y_proba)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1 Macro: {metrics['f1_macro']:.3f}")
```

---
*最后更新时间: 2025-01-20*
*已支持多分类任务*
