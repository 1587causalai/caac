# 统一架构分类方法代码设计文档

**文档版本**: v1.1  
**创建日期**: 2024年  
**更新日期**: 2024年（新增可学习阈值变体）  
**对应代码**: `src/models/caac_ovr_model.py`  
**数学原理**: `docs/unified_methods_mathematical_principles.md`

## 1. 整体架构概览

基于共享潜在表征的统一架构设计，支持**七种**不同的分类方法。所有方法采用相同的网络结构，仅在损失函数、概率建模策略和阈值参数设定上有所差异，确保公平的算法比较。

### 1.0 支持的方法列表

**CAAC系列方法（4种）**：
1. `CAACOvRModel(learnable_thresholds=False)` - 柯西分布 + 固定阈值
2. `CAACOvRModel(learnable_thresholds=True)` - 柯西分布 + 可学习阈值
3. `CAACOvRGaussianModel(learnable_thresholds=False)` - 高斯分布 + 固定阈值  
4. `CAACOvRGaussianModel(learnable_thresholds=True)` - 高斯分布 + 可学习阈值

**MLP系列方法（3种）**：
5. `SoftmaxMLPModel` - 标准Softmax多层感知机
6. `OvRCrossEntropyMLPModel` - OvR交叉熵多层感知机
7. `CrammerSingerMLPModel` - Crammer & Singer铰链损失

### 1.0.1 唯一性约束扩展功能

**新增功能**: CAAC系列方法支持可选的**潜在向量采样唯一性约束**

**⚠️ 实验发现**: 唯一性约束在实际应用中倾向于降低分类准确率，主要用作**理论对照研究**。

**功能参数**:
- `uniqueness_constraint`: 布尔值，是否启用唯一性约束 (默认 `False`)
- `uniqueness_samples`: 整数，每个样本的采样次数 (建议 `3`)
- `uniqueness_weight`: 浮点数，约束损失权重 (建议 `0.05`)

**实现原理**:
- 对每个样本从其潜在分布中采样多个实例化向量
- 使用最大-次大间隔约束确保每个采样实例的决策唯一性
- 采样次数和权重共同控制约束强度

**推荐配置**:
```python
# 理论对照研究配置
model = CAACOvRModel(
    uniqueness_constraint=True,
    uniqueness_samples=3,        # 最小采样次数
    uniqueness_weight=0.05       # 较低权重，减少对准确率的影响
)
```

**使用建议**:
- **主要用途**: 理论研究和方法论比较
- **不推荐**: 追求最高准确率的生产环境
- **适用**: 需要严格决策一致性的特殊场景

### 1.1 架构设计原则

- **统一性**: 所有方法共享相同的网络架构
- **模块化**: 清晰的模块边界，便于扩展和维护
- **可配置性**: 支持灵活的参数配置
- **可扩展性**: 为未来改进预留接口
- **概念对齐**: `d_latent = d_repr`，体现因果表征概念

### 1.2 核心模块组成

```
输入特征 x ∈ ℝᴰ
    ↓
FeatureNetwork (特征提取)
    ↓  
确定性特征表征 z ∈ ℝᵈʳᵉᵖʳ
    ↓
AbductionNetwork (推理网络)
    ↓
因果表征随机变量参数 (μ, σ) ∈ ℝᵈˡᵃᵗᵉⁿᵗ × ℝᵈˡᵃᵗᵉⁿᵗ
    ↓
ActionNetwork (行动网络)
    ↓
类别得分/概率 ∈ ℝᴺ
    ↓
预测类别 = argmax
```

**关键约定**: `d_latent = d_repr`（因果表征维度 = 特征表征维度）

---

## 2. 核心网络模块设计

### 2.1 FeatureNetwork - 特征提取网络

**功能**: 将原始输入映射为确定性高维特征表征

```python
class FeatureNetwork(nn.Module):
    """
    特征网络 - 与回归模型完全一致
    输入: x ∈ ℝᴰ (原始特征)
    输出: z ∈ ℝᵈʳᵉᵖʳ (确定性特征表征)
    """
    def __init__(self, input_dim, representation_dim, hidden_dims=[64]):
        super(FeatureNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim_i in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim_i))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim_i
        layers.append(nn.Linear(prev_dim, representation_dim))
        self.network = nn.Sequential(*layers)
        
        # 存储架构信息
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.hidden_dims = hidden_dims 

    def forward(self, x):
        return self.network(x)
```

**设计特点**:
- 多层感知机结构，支持可配置隐藏层
- ReLU激活函数，确保非线性变换
- 输出确定性特征表征向量

### 2.2 AbductionNetwork - 推理网络  

**功能**: 从确定性特征表征推理因果表征随机变量的分布参数

```python
class AbductionNetwork(nn.Module):
    """
    统一推断网络 - 输出因果表征随机变量的位置和尺度参数
    输入: z ∈ ℝᵈʳᵉᵖʳ (确定性特征表征)
    输出: μ(z) ∈ ℝᵈˡᵃᵗᵉⁿᵗ, σ(z) ∈ ℝᵈˡᵃᵗᵉⁿᵗ (随机变量参数)
    """
    def __init__(self, representation_dim, latent_dim, hidden_dims=[64, 32]):
        super(AbductionNetwork, self).__init__()
        
        # 共享特征提取层
        shared_layers_list = []
        prev_dim = representation_dim
        for hidden_dim_i in hidden_dims:
            shared_layers_list.append(nn.Linear(prev_dim, hidden_dim_i))
            shared_layers_list.append(nn.ReLU())
            prev_dim = hidden_dim_i
        shared_output_dim = prev_dim 
        
        # 位置和尺度参数的独立头部
        self.location_head = nn.Linear(shared_output_dim, latent_dim)
        self.scale_head = nn.Linear(shared_output_dim, latent_dim)
        self.shared_mlp = nn.Sequential(*shared_layers_list)
        
        # 存储架构信息
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
            
    def forward(self, representation):
        shared_features = self.shared_mlp(representation)
        location_param = self.location_head(shared_features)
        scale_param = F.softplus(self.scale_head(shared_features))  # 确保正性
        return location_param, scale_param
```

**设计特点**:
- 共享MLP主干 + 双头设计（位置/尺度）
- Softplus激活确保尺度参数正性
- 支持概念对齐：`d_latent = d_repr`

### 2.3 ActionNetwork - 行动网络

**功能**: 将因果表征随机变量线性变换为类别得分随机变量

```python
class ActionNetwork(nn.Module):
    """
    行动网络 - 处理因果表征随机变量，输出类别得分分布参数
    概念: 输入随机变量 U_j，通过线性变换输出类别得分随机变量 S_k 的参数
    """
    def __init__(self, latent_dim, n_classes):
        super(ActionNetwork, self).__init__()
        self.linear = nn.Linear(latent_dim, n_classes)
        self.latent_dim = latent_dim
        self.n_classes = n_classes
    
    def forward(self, location_param):
        # 注：输入location_param是为了兼容现有架构
        # 概念上应该处理随机变量，实际通过权重矩阵在损失函数中计算分布参数
        return self.linear(location_param)
    
    def get_weights(self):
        """获取线性变换参数"""
        weight = self.linear.weight.data  # [n_classes, latent_dim] - 线性变换矩阵A
        bias = self.linear.bias.data      # [n_classes] - 偏置B
        return weight, bias
    
    def compute_class_distribution_params(self, location_param, scale_param, distribution_type='cauchy'):
        """
        计算每个类别Score随机变量的分布参数
        不同分布类型使用不同的线性组合规则
        
        Args:
            location_param: 因果表征位置参数 μ(z) ∈ ℝᵈˡᵃᵗᵉⁿᵗ
            scale_param: 因果表征尺度参数 σ(z) ∈ ℝᵈˡᵃᵗᵉⁿᵗ  
            distribution_type: 分布类型 ('cauchy' | 'gaussian')
            
        Returns:
            class_locations: 类别得分位置参数 ∈ ℝᴺ
            class_scales: 类别得分尺度参数 ∈ ℝᴺ
        """
        W, b = self.get_weights()
        batch_size = location_param.size(0)
        
        # 位置参数：loc(S_k) = W_k @ μ(z) + b_k (所有分布相同)
        class_locations = torch.matmul(location_param, W.T) + b.unsqueeze(0)
        
        if distribution_type == 'cauchy':
            # 柯西分布：scale(S_k) = |W_k| @ σ(z)
            W_abs = torch.abs(W)
            class_scales = torch.matmul(scale_param, W_abs.T)
            return class_locations, torch.clamp(class_scales, min=1e-6)
            
        elif distribution_type == 'gaussian':
            # 高斯分布：var(S_k) = W_k^2 @ σ(z)^2, std(S_k) = sqrt(var)
            W_squared = W ** 2
            class_variances = torch.matmul(scale_param ** 2, W_squared.T)
            class_stds = torch.sqrt(torch.clamp(class_variances, min=1e-6))
            return class_locations, class_stds
            
        else:
            raise ValueError(f"Unsupported distribution type: {distribution_type}")

    def compute_scores_from_samples(self, samples):
        """
        直接从采样的潜在向量计算得分 (唯一性约束专用)
        
        Args:
            samples: [batch_size, n_samples, latent_dim] - 采样的潜在向量实例
        
        Returns:
            scores: [batch_size, n_samples, n_classes] - 每个采样的确定性得分
        """
        W, b = self.get_weights()
        # 批量矩阵运算：samples @ W.T + b
        scores = torch.matmul(samples, W.T) + b.unsqueeze(0).unsqueeze(0)
        return scores
```

**设计特点**:
- 简单线性变换层，体现线性组合概念
- 支持不同分布类型的参数计算
- 数值稳定性保障（clamp操作）

### 2.4 UnifiedClassificationNetwork - 统一分类网络

**功能**: 整合三个核心模块，构成完整的分类网络

```python
class UnifiedClassificationNetwork(nn.Module):
    """
    统一分类网络 - 整合 FeatureNetwork → AbductionNetwork → ActionNetwork
    所有分类方法共享此架构，仅损失函数不同
    """
    def __init__(self, input_dim, representation_dim, latent_dim, n_classes,
                 feature_hidden_dims, abduction_hidden_dims):
        super(UnifiedClassificationNetwork, self).__init__()
        
        # 三个核心模块
        self.feature_net = FeatureNetwork(input_dim, representation_dim, feature_hidden_dims)
        self.abduction_net = AbductionNetwork(representation_dim, latent_dim, abduction_hidden_dims)
        self.action_net = ActionNetwork(latent_dim, n_classes)
        
        # 存储架构信息
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.feature_hidden_dims = feature_hidden_dims
        self.abduction_hidden_dims = abduction_hidden_dims

    def forward(self, x):
        """
        前向传播
        
        Returns:
            logits: 用于传统MLP方法的logits
            location_param: 因果表征位置参数，用于CAAC方法
            scale_param: 因果表征尺度参数，用于CAAC方法
        """
        representation = self.feature_net(x)
        location_param, scale_param = self.abduction_net(representation)
        logits = self.action_net(location_param)
        return logits, location_param, scale_param

    def predict_proba(self, x):
        """预测概率（用于传统MLP方法）"""
        logits, _, _ = self.forward(x)
        return F.softmax(logits, dim=1)
```

**设计特点**:
- 模块化组装，清晰的数据流
- 同时输出logits和分布参数，支持不同方法
- 兼容传统softmax预测接口

---

## 3. 七种分类方法实现

### 3.1 方法实现模式

所有方法采用相同的设计模式：

```python
class MethodModel:
    """分类方法基础模式"""
    
    def __init__(self, input_dim, representation_dim=64, latent_dim=None, ...):
        # 概念对齐：d_latent = d_repr
        self.latent_dim = latent_dim if latent_dim is not None else representation_dim
        self._setup_model_optimizer()
    
    def _setup_model_optimizer(self):
        """设置统一网络和优化器"""
        self.model = UnifiedClassificationNetwork(...)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def compute_loss(self, y_true, logits, location_param, scale_param):
        """核心差异：不同的损失函数实现"""
        # 各方法的具体实现不同
        pass
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        """统一的训练流程"""
        # 相同的训练循环逻辑
        pass
```

### 3.2 CAAC OvR (柯西分布) - CAACOvRModel

**支持变体**: 固定阈值版本 & 可学习阈值版本  
**特色**: 使用柯西分布CDF计算类别概率

```python
class CAACOvRModel:
    def __init__(self, input_dim, n_classes, learnable_thresholds=False, ...):
        """
        Args:
            learnable_thresholds: 是否使用可学习阈值参数
                - False: 阈值固定为0 (传统版本)
                - True: 阈值为可学习参数 (新增版本)
        """
        super().__init__()
        # ... 网络初始化 ...
        
        # 阈值参数设置
        if learnable_thresholds:
            self.thresholds = nn.Parameter(torch.zeros(n_classes))
        else:
            self.register_buffer('thresholds', torch.zeros(n_classes))

def compute_loss(self, y_true, logits, location_param, scale_param):
    """
    CAAC柯西损失函数：体现因果表征随机变量到Score随机变量的分布变换
    核心步骤：
    1. 通过ActionNetwork计算每个类别Score的柯西分布参数
    2. 用柯西CDF计算每个类别的概率 P_k  
    3. 用P_k计算OvR二元交叉熵损失
    4. 支持固定/可学习阈值两种模式
    """
    batch_size = y_true.size(0)
    n_classes = self.n_classes
    device = y_true.device
    
    # 计算类别得分的柯西分布参数
    class_locations, class_scales = self.model.action_net.compute_class_distribution_params(
        location_param, scale_param, distribution_type='cauchy'
    )
    
    # 使用阈值参数（自动支持固定/可学习）
    thresholds = self.thresholds
    
    # 柯西分布CDF计算类别概率
    pi = torch.tensor(np.pi, device=device)
    normalized_thresholds = (thresholds.unsqueeze(0) - class_locations) / class_scales
    P_k = 0.5 - (1/pi) * torch.atan(normalized_thresholds)
    P_k = torch.clamp(P_k, min=1e-7, max=1-1e-7)  # 数值稳定性
    
    # OvR二元交叉熵损失
    y_binary = torch.zeros(batch_size, n_classes, device=device)
    y_binary.scatter_(1, y_true.unsqueeze(1), 1)
    
    bce_loss = -(y_binary * torch.log(P_k) + (1 - y_binary) * torch.log(1 - P_k))
    return torch.mean(bce_loss)
```

### 3.3 CAAC OvR (高斯分布) - CAACOvRGaussianModel

**支持变体**: 固定阈值版本 & 可学习阈值版本  
**特色**: 使用高斯分布CDF计算类别概率

```python
class CAACOvRGaussianModel:
    def __init__(self, input_dim, n_classes, learnable_thresholds=False, ...):
        """
        Args:
            learnable_thresholds: 是否使用可学习阈值参数
                - False: 阈值固定为0 (传统版本)
                - True: 阈值为可学习参数 (新增版本)
        """
        super().__init__()
        # ... 网络初始化 ...
        
        # 阈值参数设置（与柯西版本完全相同）
        if learnable_thresholds:
            self.thresholds = nn.Parameter(torch.zeros(n_classes))
        else:
            self.register_buffer('thresholds', torch.zeros(n_classes))

def compute_loss(self, y_true, logits, location_param, scale_param):
    """
    CAAC高斯损失函数：与柯西版本相同的逻辑，但使用高斯分布的线性组合规则和CDF
    支持固定/可学习阈值两种模式
    """
    # ... 前置设置相同 ...
    
    # 计算类别得分的高斯分布参数
    class_locations, class_stds = self.model.action_net.compute_class_distribution_params(
        location_param, scale_param, distribution_type='gaussian'
    )
    
    # 使用阈值参数（自动支持固定/可学习）
    thresholds = self.thresholds
    
    # 高斯分布CDF计算类别概率
    normalized_thresholds = (thresholds.unsqueeze(0) - class_locations) / class_stds
    standard_normal = Normal(0, 1)
    P_k = 1 - standard_normal.cdf(normalized_thresholds)
    P_k = torch.clamp(P_k, min=1e-7, max=1-1e-7)
    
    # ... 相同的BCE损失计算 ...
```

### 3.4 MLP (Softmax) - SoftmaxMLPModel

**特色**: 标准Softmax + 交叉熵，忽略尺度参数

```python
def compute_loss(self, y_true, logits, location_param, scale_param):
    """
    标准Softmax交叉熵损失函数
    仅使用logits，完全忽略尺度参数 scale_param
    """
    return F.cross_entropy(logits, y_true)
```

### 3.5 MLP (OvR Cross Entropy) - OvRCrossEntropyMLPModel  

**特色**: OvR策略 + 交叉熵，忽略尺度参数

```python
def compute_loss(self, y_true, logits, location_param, scale_param):
    """
    OvR交叉熵损失函数
    仅使用logits，不使用尺度参数
    这是与CAAC方法的核心区别：相同OvR策略但不使用尺度参数
    """
    return F.cross_entropy(logits, y_true)
```

### 3.6 MLP (Crammer & Singer Hinge) - CrammerSingerMLPModel

**特色**: 多分类铰链损失，基于margin最大化

```python
def compute_loss(self, y_true, logits, location_param, scale_param):
    """
    Crammer & Singer 多分类铰链损失函数
    仅使用logits，完全忽略尺度参数
    """
    batch_size, n_classes = logits.shape
    
    # 获取正确类别的分数
    correct_scores = logits.gather(1, y_true.unsqueeze(1)).squeeze(1)
    
    # 计算margin违反
    margins = logits - correct_scores.unsqueeze(1) + 1.0
    margins.scatter_(1, y_true.unsqueeze(1), 0)  # 排除正确类别
    
    # 铰链损失
    max_margins, _ = margins.max(dim=1)
    hinge_loss = F.relu(max_margins)
    
    return hinge_loss.mean()
```

---

## 4. 模块接口设计

### 4.1 统一配置接口

```python
# 统一参数配置
common_params = {
    'representation_dim': 64,
    'latent_dim': None,  # 默认等于representation_dim，体现概念对齐
    'feature_hidden_dims': [64],
    'abduction_hidden_dims': [128, 64],
    'lr': 0.001,
    'batch_size': 32,
    'learnable_thresholds': False,  # 新增：控制阈值参数模式
    'epochs': 100,
    'device': None,
    'early_stopping_patience': 10,
    'early_stopping_min_delta': 0.0001
}
```

### 4.2 训练接口

```python
def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
    """
    统一训练接口
    
    Args:
        X_train: 训练特征 [N, D]
        y_train: 训练标签 [N]
        X_val: 验证特征 [N_val, D] (可选)
        y_val: 验证标签 [N_val] (可选)
        verbose: 输出详细程度 (0|1|2)
    
    Returns:
        self: 支持链式调用
    """
    # 统一的训练流程实现
```

### 4.3 预测接口

```python
def predict_proba(self, X):
    """预测类别概率"""
    
def predict(self, X):
    """预测类别标签"""
    
def get_params(self, deep=True):
    """获取模型参数"""
    
def set_params(self, **params):
    """设置模型参数"""
```

---

## 5. 扩展机制设计

### 5.1 新方法扩展接口

```python
class NewMethodModel:
    """新方法扩展模板"""
    
    def __init__(self, input_dim, representation_dim=64, latent_dim=None, ...):
        # 遵循统一配置模式
        self.latent_dim = latent_dim if latent_dim is not None else representation_dim
        # ... 其他参数设置 ...
        self._setup_model_optimizer()
    
    def _setup_model_optimizer(self):
        """使用统一网络架构"""
        self.model = UnifiedClassificationNetwork(
            self.input_dim, self.representation_dim, self.latent_dim, self.n_classes,
            self.feature_hidden_dims, self.abduction_hidden_dims
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def compute_loss(self, y_true, logits, location_param, scale_param):
        """实现新的损失函数逻辑"""
        # 新方法的创新点在这里
        pass
    
    # 继承统一的 fit, predict_proba, predict 等方法
```

### 5.2 架构组件扩展

```python
# 扩展新的网络组件
class EnhancedFeatureNetwork(FeatureNetwork):
    """增强特征网络 - 预留扩展点"""
    
class AttentionAbductionNetwork(AbductionNetwork):
    """注意力推理网络 - 预留扩展点"""
    
class DynamicActionNetwork(ActionNetwork):
    """动态行动网络 - 预留扩展点"""
```

### 5.3 分布类型扩展

```python
def compute_class_distribution_params(self, location_param, scale_param, distribution_type='cauchy'):
    """支持更多分布类型"""
    
    if distribution_type == 'cauchy':
        # 现有柯西分布逻辑
    elif distribution_type == 'gaussian':
        # 现有高斯分布逻辑
    elif distribution_type == 'student_t':
        # 预留：Student-t分布扩展
    elif distribution_type == 'laplace':
        # 预留：拉普拉斯分布扩展
    elif distribution_type == 'mixture':
        # 预留：混合分布扩展
    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")
```

---

## 6. 实现细节与最佳实践

### 6.1 数值稳定性

```python
# 1. 尺度参数正性约束
scale_param = F.softplus(self.scale_head(shared_features))

# 2. 概率值截断
P_k = torch.clamp(P_k, min=1e-7, max=1-1e-7)

# 3. 分母稳定性
class_scales = torch.clamp(class_scales, min=1e-6)
```

### 6.2 内存优化

```python
# 1. 及时释放中间变量
del intermediate_tensor

# 2. 梯度累积支持大批量
if (batch_idx + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()

# 3. 混合精度训练支持
with torch.cuda.amp.autocast():
    loss = compute_loss(...)
```

### 6.3 设备兼容性

```python
# 自动设备检测
if device is None:
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 统一设备迁移
def to_device(self, device):
    self.model = self.model.to(device)
    self.device = device
    return self
```

---

## 7. 使用示例

### 7.1 创建七种方法实例

```python
# 统一参数配置
common_params = {
    'input_dim': 20,
    'n_classes': 3,
    'representation_dim': 64,
    'latent_dim': None,  # 默认等于representation_dim
    'feature_hidden_dims': [64],
    'abduction_hidden_dims': [128, 64],
    'lr': 0.001,
    'batch_size': 32,
    'epochs': 100
}

# 创建七种方法的实例
methods = {
    # CAAC系列 - 固定阈值版本
    'CAAC_Cauchy': CAACOvRModel(**{**common_params, 'learnable_thresholds': False}),
    'CAAC_Gaussian': CAACOvRGaussianModel(**{**common_params, 'learnable_thresholds': False}),
    
    # CAAC系列 - 可学习阈值版本（新增）
    'CAAC_Cauchy_Learnable': CAACOvRModel(**{**common_params, 'learnable_thresholds': True}),
    'CAAC_Gaussian_Learnable': CAACOvRGaussianModel(**{**common_params, 'learnable_thresholds': True}),
    
    # MLP系列
    'MLP_Softmax': SoftmaxMLPModel(**common_params),
    'MLP_OvR_CE': OvRCrossEntropyMLPModel(**common_params),
    'MLP_Hinge': CrammerSingerMLPModel(**common_params)
}
```

### 7.2 训练与评估示例

```python
# 训练所有方法
for name, model in methods.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train, X_val, y_val, verbose=1)
    
    # 评估性能
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
```

### 7.3 可学习阈值分析

```python
# 分析可学习阈值的学习结果
learnable_models = ['CAAC_Cauchy_Learnable', 'CAAC_Gaussian_Learnable']

for name in learnable_models:
    model = methods[name]
    thresholds = model.thresholds.data.cpu().numpy()
    print(f"{name} learned thresholds: {thresholds}")
    
    # 可视化阈值分布
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(thresholds)), thresholds)
    plt.title(f'{name} - Learned Thresholds by Class')
    plt.xlabel('Class Index')
    plt.ylabel('Threshold Value')
    plt.show()
```

---

## 8. 测试与验证

### 8.1 单元测试设计

```python
def test_feature_network():
    """测试特征网络"""
    
def test_abduction_network():
    """测试推理网络"""
    
def test_action_network():
    """测试行动网络"""
    
def test_unified_network():
    """测试统一网络"""

def test_concept_alignment():
    """测试概念对齐：d_latent = d_repr"""

def test_learnable_thresholds():
    """测试可学习阈值功能"""
    # 验证固定阈值模式
    model_fixed = CAACOvRModel(learnable_thresholds=False)
    assert not model_fixed.thresholds.requires_grad
    
    # 验证可学习阈值模式
    model_learnable = CAACOvRModel(learnable_thresholds=True)
    assert model_learnable.thresholds.requires_grad
```

### 8.2 集成测试

```python
def test_method_compatibility():
    """测试七种方法的兼容性"""
    
def test_training_pipeline():
    """测试训练流程"""
    
def test_prediction_consistency():
    """测试预测一致性"""
```

---

## 9. 部署与优化

### 9.1 模型保存与加载

```python
def save_model(self, filepath):
    """保存完整模型状态"""
    
def load_model(cls, filepath):
    """加载模型状态"""
    
def export_onnx(self, filepath, input_shape):
    """导出ONNX格式"""
```

### 9.2 推理优化

```python
def optimize_for_inference(self):
    """推理优化"""
    self.model.eval()
    # 可选：模型量化、剪枝等
    
def batch_predict(self, X, batch_size=1000):
    """批量预测优化"""
```

---

## 10. 未来扩展规划

### 10.1 架构扩展方向

- [ ] **多模态输入支持**: 扩展FeatureNetwork支持图像、文本等
- [ ] **动态架构**: 自适应调整网络深度和宽度
- [ ] **元学习集成**: 支持快速适应新任务
- [ ] **联邦学习**: 分布式训练支持

### 10.2 算法创新方向

- [ ] **自适应分布选择**: 根据数据特性动态选择最优分布
- [ ] **贝叶斯集成**: 引入贝叶斯神经网络增强不确定性
- [ ] **对抗训练**: 提升模型鲁棒性
- [ ] **知识蒸馏**: 模型压缩与加速

### 10.3 应用场景扩展

- [ ] **大规模分类**: 支持百万级类别分类
- [ ] **实时系统**: 低延迟推理优化
- [ ] **边缘计算**: 轻量化部署方案
- [ ] **可解释AI**: 增强决策透明度

---

**文档结束**

*本文档详细描述了统一架构分类方法的代码设计与实现。文档将随着代码演进持续更新。*

**相关文档**:
- 📊 **数学原理**: `docs/unified_methods_mathematical_principles.md`
- 🧠 **研究动机**: `docs/motivation.md`
- 🔬 **实验对比**: `compare_methods.py`
- 🏗️ **参考设计**: `architecture_design.md` 