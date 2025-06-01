# 共享潜在柯西向量的 One-vs-Rest (OvR) 多分类器设计

## 1. 整体架构设计

基于对CAAR和CAAC项目的分析，结合理论设计文档，我们设计一个全新的共享潜在柯西向量的OvR多分类器架构。该架构保留CAAR的模块化三段式结构，同时引入共享潜在柯西向量和OvR策略，以解决CAAC在分类任务上的不足。

### 1.1 模块组成

新架构由以下核心模块组成：

1. **特征网络 (FeatureNetwork)**
   - 输入：原始特征 x ∈ ℝᴰ
   - 输出：高维表征 z ∈ ℝᴸ
   - 功能：提取输入特征的高级表征

2. **推断网络 (AbductionNetwork)**
   - 输入：高维表征 z ∈ ℝᴸ
   - 输出：潜在柯西向量参数 (μ₁...μₘ, σ₁...σₘ)
   - 功能：推断共享潜在柯西随机向量的参数

3. **线性变换层 (LinearTransformationLayer)**
   - 输入：潜在柯西向量参数 (μ₁...μₘ, σ₁...σₘ)
   - 输出：N个类别的得分随机变量参数 (loc₁...locₙ, scale₁...scaleₙ)
   - 功能：将潜在柯西向量映射到各类别得分随机变量

4. **OvR概率计算层 (OvRProbabilityLayer)**
   - 输入：类别得分随机变量参数 (loc₁...locₙ, scale₁...scaleₙ)
   - 输出：N个类别的概率 P₁...Pₙ
   - 功能：计算样本属于每个类别的概率

5. **损失计算模块 (LossComputation)**
   - 输入：预测概率 P₁...Pₙ 和真实标签 y
   - 输出：损失值
   - 功能：计算OvR策略下的二元交叉熵损失

### 1.2 数据流图

```
输入特征 x
    ↓
FeatureNetwork
    ↓
高维表征 z
    ↓
AbductionNetwork
    ↓
潜在柯西向量参数 (μ₁...μₘ, σ₁...σₘ)
    ↓
LinearTransformationLayer
    ↓
类别得分随机变量参数 (loc₁...locₙ, scale₁...scaleₙ)
    ↓
OvRProbabilityLayer
    ↓
类别概率 P₁...Pₙ
    ↓
预测类别 = argmax(P₁...Pₙ)
```

## 2. 模块详细设计

### 2.1 特征网络 (FeatureNetwork)

```python
class FeatureNetwork(nn.Module):
    def __init__(self, input_dim, representation_dim, hidden_dims=[64]):
        super(FeatureNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, representation_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
```

### 2.2 推断网络 (AbductionNetwork)

```python
class AbductionNetwork(nn.Module):
    def __init__(self, representation_dim, latent_dim, hidden_dims=[64, 32]):
        super(AbductionNetwork, self).__init__()
        shared_layers = []
        prev_dim = representation_dim
        for hidden_dim in hidden_dims:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.shared_mlp = nn.Sequential(*shared_layers)
        
        self.location_head = nn.Linear(prev_dim, latent_dim)
        self.scale_head = nn.Linear(prev_dim, latent_dim)
        
    def forward(self, representation):
        shared_features = self.shared_mlp(representation)
        location_param = self.location_head(shared_features)
        scale_param = F.softplus(self.scale_head(shared_features))
        return location_param, scale_param
```

### 2.3 线性变换层 (LinearTransformationLayer)

```python
class LinearTransformationLayer(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super(LinearTransformationLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_classes, latent_dim))
        self.bias = nn.Parameter(torch.Tensor(n_classes))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, location_param, scale_param):
        # 计算类别得分随机变量的位置参数
        # loc(S_k; z) = sum_{j=1}^M A_{kj} * mu_j(z) + B_k
        loc = F.linear(location_param, self.weight, self.bias)
        
        # 计算类别得分随机变量的尺度参数
        # scale(S_k; z) = sum_{j=1}^M |A_{kj}| * sigma_j(z)
        weight_abs = torch.abs(self.weight)
        scale = F.linear(scale_param, weight_abs)
        
        return loc, scale
```

### 2.4 OvR概率计算层 (OvRProbabilityLayer)

```python
class OvRProbabilityLayer(nn.Module):
    def __init__(self, n_classes, threshold=0.0):
        super(OvRProbabilityLayer, self).__init__()
        self.n_classes = n_classes
        self.threshold = threshold
        
    def forward(self, loc, scale):
        # 确保scale为正且数值稳定
        scale_stable = torch.clamp(scale, min=1e-6)
        
        # 计算每个类别的概率
        # P_k(z) = 0.5 - (1/pi) * arctan((C_k - loc(S_k; z)) / scale(S_k; z))
        normalized_diff = (self.threshold - loc) / scale_stable
        class_probs = 0.5 - (1.0 / torch.pi) * torch.atan(normalized_diff)
        
        # 确保概率在[0,1]范围内
        class_probs = torch.clamp(class_probs, min=1e-6, max=1.0-1e-6)
        
        return class_probs
```

### 2.5 损失计算模块 (LossComputation)

```python
def compute_ovr_loss(y_true, class_probs):
    """
    计算OvR策略下的二元交叉熵损失
    
    Args:
        y_true: 真实标签 [batch_size]
        class_probs: 预测概率 [batch_size, n_classes]
        
    Returns:
        loss: 损失值
    """
    batch_size, n_classes = class_probs.shape
    
    # 将真实标签转换为one-hot编码
    y_true_one_hot = F.one_hot(y_true, n_classes).float()
    
    # 计算每个类别的二元交叉熵损失
    # L = -[y_true * log(P) + (1-y_true) * log(1-P)]
    bce_loss = -y_true_one_hot * torch.log(class_probs) - (1 - y_true_one_hot) * torch.log(1 - class_probs)
    
    # 对所有类别和样本求平均
    loss = torch.mean(bce_loss)
    
    return loss
```

## 3. 完整模型设计

### 3.1 统一分类网络 (UnifiedClassificationNetwork)

```python
class UnifiedClassificationNetwork(nn.Module):
    def __init__(self, input_dim, representation_dim, latent_dim, n_classes,
                 feature_hidden_dims=[64], abduction_hidden_dims=[128, 64],
                 threshold=0.0):
        super(UnifiedClassificationNetwork, self).__init__()
        
        self.feature_net = FeatureNetwork(input_dim, representation_dim, feature_hidden_dims)
        self.abduction_net = AbductionNetwork(representation_dim, latent_dim, abduction_hidden_dims)
        self.linear_transform = LinearTransformationLayer(latent_dim, n_classes)
        self.ovr_probability = OvRProbabilityLayer(n_classes, threshold)
        
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.threshold = threshold
        
    def forward(self, x):
        # 特征提取
        representation = self.feature_net(x)
        
        # 推断潜在柯西向量参数
        location_param, scale_param = self.abduction_net(representation)
        
        # 线性变换到类别得分随机变量
        loc, scale = self.linear_transform(location_param, scale_param)
        
        # 计算类别概率
        class_probs = self.ovr_probability(loc, scale)
        
        return class_probs, loc, scale, location_param, scale_param
    
    def predict(self, x):
        class_probs, _, _, _, _ = self.forward(x)
        return class_probs
```

### 3.2 模型包装类 (CAACOvRModel)

```python
class CAACOvRModel:
    def __init__(self, input_dim, 
                 representation_dim=64, 
                 latent_dim=64, 
                 n_classes=2,
                 feature_hidden_dims=[64], 
                 abduction_hidden_dims=[128, 64], 
                 threshold=0.0,
                 lr=0.001, batch_size=32, epochs=100, device=None,
                 early_stopping_patience=None, early_stopping_min_delta=0.0001):
        
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.feature_hidden_dims = feature_hidden_dims
        self.abduction_hidden_dims = abduction_hidden_dims
        self.threshold = threshold
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self._setup_model_optimizer()
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 
                        'train_time': 0, 'best_epoch': 0}
    
    def _setup_model_optimizer(self):
        self.model = UnifiedClassificationNetwork(
            self.input_dim, 
            self.representation_dim, 
            self.latent_dim, 
            self.n_classes,
            self.feature_hidden_dims, 
            self.abduction_hidden_dims,
            self.threshold
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def compute_loss(self, y_true, class_probs):
        return compute_ovr_loss(y_true, class_probs)
    
    def compute_accuracy(self, y_true, class_probs):
        _, predicted = torch.max(class_probs, 1)
        correct = (predicted == y_true).sum().item()
        total = y_true.size(0)
        return correct / total
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        # 训练流程实现
        # ...
        
    def predict_proba(self, X):
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            class_probs = self.model.predict(X_tensor)
        return class_probs.cpu().numpy()
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
```

## 4. 模型特性与优势

1. **共享潜在柯西向量**：所有类别共享同一个潜在柯西随机向量，隐式捕获类别间的相关性。

2. **OvR策略**：将N类问题分解为N个二分类问题，提高并行性和可扩展性。

3. **精确的不确定性建模**：通过柯西分布的位置参数和尺度参数，为每个类别判决提供"中心"和"模糊度"量化。

4. **线性变换特性**：利用柯西分布的线性组合特性，确保类别得分随机变量仍然服从柯西分布。

5. **模块化设计**：清晰的模块化结构，便于扩展和维护。

## 5. 实现注意事项

1. **数值稳定性**：在计算概率和损失时，需要确保数值稳定性，避免除零或对数溢出。

2. **参数初始化**：线性变换层的权重初始化对模型性能有重要影响，需要合理设置。

3. **批量处理**：确保所有操作支持批量处理，提高训练效率。

4. **设备兼容性**：确保模型在CPU和GPU上都能正常运行。

5. **早停机制**：实现早停机制，避免过拟合。
