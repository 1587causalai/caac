# API文档

## 模型模块 (models)

### CAACOvRModel

```python
class CAACOvRModel:
    """
    CAAC OvR 模型包装类
    
    封装训练、验证和预测功能。
    """
    def __init__(self, input_dim, 
                 representation_dim=64, 
                 latent_dim=64, 
                 n_classes=2,
                 feature_hidden_dims=[64], 
                 abduction_hidden_dims=[128, 64], 
                 threshold=0.0,
                 lr=0.001, batch_size=32, epochs=100, device=None,
                 early_stopping_patience=None, early_stopping_min_delta=0.0001):
        """
        初始化CAAC OvR模型包装类
        
        Args:
            input_dim: 输入特征维度
            representation_dim: 表征维度
            latent_dim: 潜在柯西向量维度
            n_classes: 类别数量
            feature_hidden_dims: 特征网络隐藏层维度列表
            abduction_hidden_dims: 推断网络隐藏层维度列表
            threshold: 判决阈值，默认为0.0
            lr: 学习率
            batch_size: 批量大小
            epochs: 训练轮数
            device: 设备（CPU/GPU）
            early_stopping_patience: 早停耐心值
            early_stopping_min_delta: 早停最小增益
        """
        pass
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        """
        训练模型
        
        Args:
            X_train: 训练特征 [n_samples, input_dim]
            y_train: 训练标签 [n_samples]
            X_val: 验证特征 [n_val_samples, input_dim]
            y_val: 验证标签 [n_val_samples]
            verbose: 是否打印训练过程
            
        Returns:
            self: 训练后的模型
        """
        pass
        
    def predict_proba(self, X):
        """
        预测类别概率
        
        Args:
            X: 特征 [n_samples, input_dim]
            
        Returns:
            probs: 类别概率 [n_samples, n_classes]
        """
        pass
        
    def predict(self, X):
        """
        预测类别
        
        Args:
            X: 特征 [n_samples, input_dim]
            
        Returns:
            predictions: 预测类别 [n_samples]
        """
        pass
        
    def get_params(self, deep=True):
        """
        获取模型参数
        
        Args:
            deep: 是否深拷贝
            
        Returns:
            params: 模型参数字典
        """
        pass
        
    def set_params(self, **params):
        """
        设置模型参数
        
        Args:
            **params: 模型参数
            
        Returns:
            self: 更新参数后的模型
        """
        pass
```

### UnifiedClassificationNetwork

```python
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
        pass
        
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
        pass
        
    def predict(self, x):
        """
        预测类别概率
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            
        Returns:
            class_probs: 类别概率 [batch_size, n_classes]
        """
        pass
```

### FeatureNetwork

```python
class FeatureNetwork(nn.Module):
    """
    特征网络 (Feature Network)
    
    将原始输入特征转换为高维表征。
    """
    def __init__(self, input_dim, representation_dim, hidden_dims=[64]):
        """
        初始化特征网络
        
        Args:
            input_dim: 输入特征维度
            representation_dim: 输出表征维度
            hidden_dims: 隐藏层维度列表
        """
        pass
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            
        Returns:
            representation: 高维表征 [batch_size, representation_dim]
        """
        pass
```

### AbductionNetwork

```python
class AbductionNetwork(nn.Module):
    """
    推断网络 (Abduction Network)
    
    基于高维表征推断共享潜在柯西随机向量的参数。
    """
    def __init__(self, representation_dim, latent_dim, hidden_dims=[64, 32]):
        """
        初始化推断网络
        
        Args:
            representation_dim: 输入表征维度
            latent_dim: 潜在柯西向量维度
            hidden_dims: 隐藏层维度列表
        """
        pass
        
    def forward(self, representation):
        """
        前向传播
        
        Args:
            representation: 高维表征 [batch_size, representation_dim]
            
        Returns:
            location_param: 潜在柯西向量的位置参数 [batch_size, latent_dim]
            scale_param: 潜在柯西向量的尺度参数 [batch_size, latent_dim]
        """
        pass
```

### LinearTransformationLayer

```python
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
        pass
        
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
        pass
```

### OvRProbabilityLayer

```python
class OvRProbabilityLayer(nn.Module):
    """
    OvR概率计算层 (OvR Probability Layer)
    
    基于类别得分随机变量的柯西分布参数，计算样本属于每个类别的概率。
    使用柯西分布的CDF计算概率。
    """
    def __init__(self, n_classes, threshold=0.0):
        """
        初始化OvR概率计算层
        
        Args:
            n_classes: 类别数量
            threshold: 判决阈值，默认为0.0
        """
        pass
        
    def forward(self, loc, scale):
        """
        前向传播
        
        Args:
            loc: 类别得分随机变量的位置参数 [batch_size, n_classes]
            scale: 类别得分随机变量的尺度参数 [batch_size, n_classes]
            
        Returns:
            class_probs: 类别概率 [batch_size, n_classes]
        """
        pass
```

## 数据处理模块 (data)

### DataProcessor

```python
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
        pass
    
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
        pass
    
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
        pass
    
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
        pass
```

## 实验评估模块 (experiments)

### ModelEvaluator

```python
class ModelEvaluator:
    """
    模型评估类
    
    提供模型评估、结果可视化和实验配置功能。
    """
    
    def __init__(self, results_dir='../../results'):
        """
        初始化模型评估类
        
        Args:
            results_dir: 结果保存目录
        """
        pass
    
    def evaluate_model(self, model, X_test, y_test):
        """
        评估模型
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试标签
            
        Returns:
            metrics: 评估指标字典
            y_pred_proba: 预测概率
            y_pred: 预测类别
            cm: 混淆矩阵
        """
        pass
    
    def visualize_confusion_matrix(self, cm, class_names, title='Confusion Matrix', save_path=None):
        """
        可视化混淆矩阵
        
        Args:
            cm: 混淆矩阵
            class_names: 类别名称
            title: 图表标题
            save_path: 保存路径
            
        Returns:
            fig: 图表对象
        """
        pass
    
    def visualize_roc_curve(self, y_test, y_pred_proba, n_classes, save_path=None):
        """
        可视化ROC曲线
        
        Args:
            y_test: 测试标签
            y_pred_proba: 预测概率
            n_classes: 类别数量
            save_path: 保存路径
            
        Returns:
            fig: 图表对象
        """
        pass
    
    def visualize_uncertainty(self, model, X_test, y_test, n_samples=10, save_path=None):
        """
        可视化不确定性
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试标签
            n_samples: 样本数量
            save_path: 保存路径
            
        Returns:
            fig: 图表对象
        """
        pass
    
    def save_experiment_results(self, experiment_name, dataset_name, model_params, metrics, 
                               train_time, n_samples, n_features, n_classes, class_distribution):
        """
        保存实验结果
        
        Args:
            experiment_name: 实验名称
            dataset_name: 数据集名称
            model_params: 模型参数
            metrics: 评估指标
            train_time: 训练时间
            n_samples: 样本数量
            n_features: 特征数量
            n_classes: 类别数量
            class_distribution: 类别分布
            
        Returns:
            result_path: 结果保存路径
        """
        pass
```

### 实验脚本

```python
def run_experiment(dataset_name, model_params=None, results_dir='../../results', random_state=42):
    """
    运行实验
    
    Args:
        dataset_name: 数据集名称
        model_params: 模型参数
        results_dir: 结果保存目录
        random_state: 随机种子
        
    Returns:
        experiment_results: 实验结果
    """
    pass
```
