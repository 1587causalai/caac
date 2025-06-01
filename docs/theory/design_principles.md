py# 设计原则

## 💡 核心设计理念

本项目的设计遵循以下核心原则，这些原则指导了我们在架构设计、算法选择和实现方式上的决策。

## 🎯 1. 单一职责原则

### 模块化设计
每个组件都有明确、单一的职责：

- **FeatureNetwork**: 专注于特征提取和表征学习
- **AbductionNetwork**: 专门负责潜在柯西向量参数的推理
- **ActionNetwork**: 处理从潜在空间到类别概率的映射
- **Trainer**: 专注于训练流程和优化
- **Evaluator**: 专门负责模型评估和指标计算

### 优势
- 代码更易理解和维护
- 便于单独测试和调试
- 支持组件的独立开发和优化

## 🔗 2. 接口抽象原则

### 统一的模型接口
所有模型都实现相同的基础接口：

```python
class BaseModel:
    def fit(self, X_train, y_train, X_val=None, y_val=None)
    def predict(self, X)
    def predict_proba(self, X)
    def get_params(self, deep=True)
    def set_params(self, **params)
```

### 好处
- 不同模型之间可以无缝替换
- 实验比较更加公平和一致
- 便于集成到更大的机器学习流水线

## 📊 3. 配置驱动原则

### 实验配置外部化
将实验参数从代码中分离出来：

```yaml
# experiment_config.yaml
model:
  type: "CAACOvRModel"
  params:
    representation_dim: 64
    latent_dim: 32
    
training:
  learning_rate: 0.001
  batch_size: 32
  epochs: 100
  
evaluation:
  metrics: ["accuracy", "f1_macro", "roc_auc"]
```

### 优势
- 实验可重现性
- 参数调优更加系统化
- 便于批量运行和网格搜索

## 🔄 4. 可扩展性原则

### 插件式架构
支持新组件的轻松添加：

```python
# 添加新的损失函数
class CustomLoss(BaseLoss):
    def __init__(self, weight=1.0):
        self.weight = weight
    
    def compute(self, y_pred, y_true):
        # 自定义损失计算
        pass

# 注册到损失函数工厂
LossFactory.register("custom_loss", CustomLoss)
```

### 新模型添加
通过继承基础类轻松添加新模型：

```python
class NewCAACVariant(BaseCAACModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 自定义初始化
    
    def _compute_distribution_params(self, latent_params):
        # 实现新的分布参数计算逻辑
        pass
```

## 🧪 5. 测试驱动原则

### 分层测试策略
- **单元测试**: 测试每个组件的独立功能
- **集成测试**: 测试组件间的交互
- **端到端测试**: 测试完整的训练和预测流程

```python
# 示例单元测试
def test_feature_network():
    network = FeatureNetwork(input_dim=10, output_dim=64)
    x = torch.randn(32, 10)
    output = network(x)
    assert output.shape == (32, 64)

# 示例集成测试
def test_model_training():
    model = CAACOvRModel(input_dim=10, n_classes=3)
    X_train, y_train = generate_dummy_data(100, 10, 3)
    model.fit(X_train, y_train)
    assert model.is_fitted
```

## 📈 6. 性能优化原则

### 计算效率
- **向量化操作**: 优先使用PyTorch的批量操作
- **内存效率**: 避免不必要的张量复制
- **GPU加速**: 支持CUDA训练和推理

### 算法效率
- **早停机制**: 避免过拟合和计算浪费
- **梯度裁剪**: 防止梯度爆炸
- **学习率调度**: 优化收敛速度

## 🔍 7. 可观测性原则

### 丰富的日志和监控
```python
logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
logger.info(f"Training time: {training_time:.2f}s")

# 指标跟踪
metrics_tracker.log("train_accuracy", train_acc, step=epoch)
metrics_tracker.log("validation_f1", val_f1, step=epoch)
```

### 可视化支持
- 训练曲线自动绘制
- 混淆矩阵可视化
- 不确定性分析图表
- ROC/PR曲线

## 🎨 8. 用户友好原则

### 直观的API设计
```python
# 简单易用的接口
model = CAACOvRModel()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# 丰富的参数说明
model = CAACOvRModel(
    representation_dim=64,      # 表征维度
    latent_dim=32,             # 潜在向量维度  
    learnable_thresholds=True,  # 是否学习阈值
    early_stopping_patience=10  # 早停等待轮数
)
```

### 完善的错误处理
```python
try:
    model.fit(X_train, y_train)
except ValueError as e:
    logger.error(f"训练数据格式错误: {e}")
    raise DataFormatError("请检查输入数据的维度和类型")
```

## 🔄 9. 向后兼容原则

### API稳定性
- 保持公共接口的稳定性
- 废弃功能提供迁移路径
- 版本化的配置格式

### 数据格式兼容
- 支持多种输入数据格式（numpy, pandas, torch）
- 自动类型转换和验证
- 清晰的错误提示

## 📚 10. 文档优先原则

### 自文档化代码
```python
def compute_cauchy_cdf(x: torch.Tensor, loc: torch.Tensor, 
                      scale: torch.Tensor) -> torch.Tensor:
    """
    计算柯西分布的累积分布函数值
    
    Args:
        x: 输入值张量, shape: (batch_size, n_classes)
        loc: 位置参数, shape: (batch_size, n_classes)  
        scale: 尺度参数, shape: (batch_size, n_classes)
        
    Returns:
        CDF值, shape: (batch_size, n_classes)
        
    Note:
        使用数值稳定的实现避免除零错误
    """
```

### 完整的使用示例
- 每个功能都有对应的使用示例
- 从简单到复杂的教程序列
- 常见问题的解决方案

---

这些设计原则确保了项目的高质量、可维护性和用户友好性。它们在代码实现的每个层面都得到了体现，为项目的长期发展奠定了坚实的基础。 