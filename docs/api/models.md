# 🧠 模型API文档

本文档详细描述CAAC项目中核心模型的API接口和使用方法。

## 📋 概览

CAAC模型实现位于 `src/models/caac_ovr_model.py`，提供基于共享潜在柯西向量的One-vs-Rest多分类器。

```python
from src.models.caac_ovr_model import (
    CAACOvRModel, 
    CAACOvRGaussianModel,
    SoftmaxMLPModel,
    OvRCrossEntropyMLPModel,
    CrammerSingerMLPModel
)
```

## 🎯 核心模型：CAACOvRModel

### 类定义

```python
class CAACOvRModel:
    """
    基于共享潜在柯西向量的One-vs-Rest多分类器
    
    核心特性：
    - 柯西分布建模决策不确定性
    - 共享潜在向量捕捉类别相关性
    - OvR策略支持大规模类别
    - 可学习阈值和唯一性约束
    """
```

### 初始化

```python
def __init__(self, 
             input_dim: int,
             representation_dim: int = 64, 
             latent_dim: Optional[int] = None,
             n_classes: int = 2,
             feature_hidden_dims: List[int] = [64], 
             abduction_hidden_dims: List[int] = [128, 64], 
             lr: float = 0.001, 
             batch_size: int = 32, 
             epochs: int = 100, 
             device: Optional[str] = None,
             early_stopping_patience: Optional[int] = None, 
             early_stopping_min_delta: float = 0.0001,
             learnable_thresholds: bool = False,
             uniqueness_constraint: bool = False,
             uniqueness_samples: int = 10,
             uniqueness_weight: float = 0.1):
    """
    初始化CAAC OvR模型
    
    Args:
        input_dim: 输入特征维度
        representation_dim: 表征层维度
        latent_dim: 潜在向量维度 (默认等于representation_dim)
        n_classes: 类别数量
        feature_hidden_dims: 特征网络隐藏层维度
        abduction_hidden_dims: 推断网络隐藏层维度
        lr: 学习率
        batch_size: 批量大小
        epochs: 训练轮数
        device: 计算设备 ('cpu' 或 'cuda')
        early_stopping_patience: 早停耐心值
        early_stopping_min_delta: 早停最小改善值
        learnable_thresholds: 是否使用可学习阈值
        uniqueness_constraint: 是否启用唯一性约束
        uniqueness_samples: 唯一性约束采样次数
        uniqueness_weight: 唯一性约束损失权重
    """
```

**示例**:
```python
# 基础模型
model = CAACOvRModel(
    input_dim=20,
    n_classes=3,
    representation_dim=64
)

# 高级配置
model = CAACOvRModel(
    input_dim=784,  # MNIST
    n_classes=10,
    representation_dim=128,
    latent_dim=64,
    feature_hidden_dims=[256, 128],
    abduction_hidden_dims=[128, 64],
    learnable_thresholds=True,
    uniqueness_constraint=True,
    early_stopping_patience=10
)
```

### 主要方法

#### 🔧 模型训练

```python
def fit(self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_val: Optional[np.ndarray] = None, 
        y_val: Optional[np.ndarray] = None, 
        verbose: int = 1) -> Dict:
    """
    训练CAAC模型
    
    Args:
        X_train: 训练特征 (n_samples, n_features)
        y_train: 训练标签 (n_samples,)
        X_val: 验证特征 (可选)
        y_val: 验证标签 (可选)
        verbose: 详细程度 (0=静默, 1=进度条, 2=详细)
        
    Returns:
        Dict: 训练历史信息
            - train_losses: List[float] - 训练损失
            - val_losses: List[float] - 验证损失 (如果提供验证集)
            - train_accuracies: List[float] - 训练准确率
            - val_accuracies: List[float] - 验证准确率
            - best_epoch: int - 最佳轮次
            - total_time: float - 总训练时间
    """
```

**示例**:
```python
# 基础训练
history = model.fit(X_train, y_train)

# 带验证集训练
history = model.fit(
    X_train, y_train,
    X_val, y_val,
    verbose=2
)

# 访问训练历史
print(f"最佳轮次: {history['best_epoch']}")
print(f"最终训练准确率: {history['train_accuracies'][-1]:.4f}")
```

#### 🎯 模型预测

```python
def predict(self, X: np.ndarray) -> np.ndarray:
    """
    进行类别预测
    
    Args:
        X: 输入特征 (n_samples, n_features)
        
    Returns:
        np.ndarray: 预测类别 (n_samples,)
    """

def predict_proba(self, X: np.ndarray) -> np.ndarray:
    """
    预测类别概率
    
    Args:
        X: 输入特征 (n_samples, n_features)
        
    Returns:
        np.ndarray: 类别概率 (n_samples, n_classes)
    """
```

**示例**:
```python
# 类别预测
predictions = model.predict(X_test)

# 概率预测
probabilities = model.predict_proba(X_test)

# 获取最高置信度的预测
max_proba = np.max(probabilities, axis=1)
confident_predictions = predictions[max_proba > 0.8]
```

#### ⚙️ 参数管理

```python
def get_params(self, deep: bool = True) -> Dict:
    """
    获取模型参数
    
    Args:
        deep: 是否包含嵌套参数
        
    Returns:
        Dict: 模型参数字典
    """

def set_params(self, **params) -> 'CAACOvRModel':
    """
    设置模型参数
    
    Args:
        **params: 要设置的参数
        
    Returns:
        CAACOvRModel: 返回自身以支持链式调用
    """
```

**示例**:
```python
# 获取所有参数
params = model.get_params()
print(f"当前学习率: {params['lr']}")

# 设置新参数
model.set_params(lr=0.01, epochs=200)
```

### 网络架构组件

#### 🔗 特征网络 (FeatureNetwork)

```python
class FeatureNetwork(nn.Module):
    """
    特征提取网络
    
    将原始输入映射到表征空间
    """
    
    def __init__(self, input_dim: int, representation_dim: int, hidden_dims: List[int] = [64]):
        """
        Args:
            input_dim: 输入维度
            representation_dim: 输出表征维度
            hidden_dims: 隐藏层维度列表
        """
```

#### 🎲 推断网络 (AbductionNetwork)

```python
class AbductionNetwork(nn.Module):
    """
    潜在柯西向量参数推断网络
    
    从表征预测柯西分布的位置和尺度参数
    """
    
    def __init__(self, representation_dim: int, latent_dim: int, hidden_dims: List[int] = [64, 32]):
        """
        Args:
            representation_dim: 输入表征维度
            latent_dim: 潜在向量维度
            hidden_dims: 隐藏层维度列表
        """
```

#### 🎯 动作网络 (ActionNetwork)

```python
class ActionNetwork(nn.Module):
    """
    类别决策网络
    
    从潜在向量计算各类别的得分和概率
    """
    
    def __init__(self, latent_dim: int, n_classes: int):
        """
        Args:
            latent_dim: 潜在向量维度
            n_classes: 类别数量
        """
```

## 🔄 对比模型

### CAACOvRGaussianModel

```python
class CAACOvRGaussianModel(CAACOvRModel):
    """
    CAAC的高斯分布变体
    
    使用高斯分布替代柯西分布进行建模
    """
```

### SoftmaxMLPModel

```python
class SoftmaxMLPModel:
    """
    传统Softmax多层感知机
    
    用于基准对比的标准神经网络分类器
    """
```

### OvRCrossEntropyMLPModel

```python
class OvRCrossEntropyMLPModel:
    """
    One-vs-Rest交叉熵MLP模型
    
    使用交叉熵损失的OvR策略
    """
```

### CrammerSingerMLPModel

```python
class CrammerSingerMLPModel:
    """
    Crammer & Singer铰链损失MLP模型
    
    使用多类铰链损失的分类器
    """
```

## 📊 损失函数

### CAAC损失计算

```python
def compute_loss(self, y_true: torch.Tensor, logits: torch.Tensor, 
                location_param: torch.Tensor, scale_param: torch.Tensor) -> torch.Tensor:
    """
    计算CAAC模型的组合损失
    
    包含：
    - One-vs-Rest二元交叉熵损失
    - 唯一性约束损失 (如果启用)
    
    Args:
        y_true: 真实标签
        logits: 模型输出logits
        location_param: 柯西分布位置参数
        scale_param: 柯西分布尺度参数
        
    Returns:
        torch.Tensor: 总损失值
    """
```

## 🎛️ 高级特性

### 可学习阈值

当 `learnable_thresholds=True` 时，决策阈值变为可学习参数：

```python
# 启用可学习阈值
model = CAACOvRModel(
    input_dim=20,
    n_classes=3,
    learnable_thresholds=True
)
```

### 唯一性约束

当 `uniqueness_constraint=True` 时，添加唯一性损失：

```python
# 启用唯一性约束
model = CAACOvRModel(
    input_dim=20,
    n_classes=3,
    uniqueness_constraint=True,
    uniqueness_samples=10,
    uniqueness_weight=0.1
)
```

## 📈 模型评估

### 性能指标

```python
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 基础指标
accuracy = accuracy_score(y_true, predictions)
f1 = f1_score(y_true, predictions, average='macro')

# 详细报告
report = classification_report(y_true, predictions)
print(report)
```

### 不确定性分析

```python
# 获取预测概率
probabilities = model.predict_proba(X_test)

# 计算预测不确定性 (熵)
from scipy.stats import entropy
uncertainties = [entropy(prob) for prob in probabilities]

# 识别高不确定性样本
high_uncertainty_indices = np.where(np.array(uncertainties) > threshold)[0]
```

## 🔧 模型配置最佳实践

### 小数据集配置
```python
model = CAACOvRModel(
    input_dim=input_dim,
    representation_dim=32,
    latent_dim=16,
    feature_hidden_dims=[32],
    abduction_hidden_dims=[32, 16],
    lr=0.01,
    batch_size=16,
    epochs=100
)
```

### 大数据集配置
```python
model = CAACOvRModel(
    input_dim=input_dim,
    representation_dim=256,
    latent_dim=128,
    feature_hidden_dims=[512, 256],
    abduction_hidden_dims=[256, 128],
    lr=0.001,
    batch_size=128,
    epochs=200,
    early_stopping_patience=10
)
```

### 高鲁棒性配置
```python
model = CAACOvRModel(
    input_dim=input_dim,
    representation_dim=128,
    learnable_thresholds=True,
    uniqueness_constraint=True,
    uniqueness_weight=0.1,
    early_stopping_patience=15
)
```

## 🚨 注意事项

### 数据预处理
- 建议对输入特征进行标准化
- 类别标签应为0开始的连续整数
- 确保训练集中包含所有类别

### 内存和计算
- 唯一性约束会增加计算开销
- 大的`latent_dim`会增加内存使用
- GPU加速建议用于大规模数据

### 超参数调优
- `representation_dim`通常设为输入维度的1/2到2倍
- `learnable_thresholds`在不平衡数据上效果更好
- `uniqueness_constraint`在多类别任务中更有效

## 📞 技术支持

更多详细信息请参考：
1. `docs/theory/motivation.md` - 理论基础
2. `examples/` - 使用示例
3. `docs/api/experiments.md` - 实验API 