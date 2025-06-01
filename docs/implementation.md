# 实现细节

## 项目结构

CAAC项目采用模块化的结构组织代码，主要包括以下目录：

```
caac_project/
├── docs/                 # docsify文档
├── src/                  # 源代码
│   ├── data/             # 数据处理模块
│   ├── experiments/      # 实验和评估模块
│   ├── models/           # 模型实现
│   └── utils/            # 工具函数
├── results/              # 实验结果
└── tests/                # 测试代码
```

## 核心模块实现

### 特征网络 (FeatureNetwork)

特征网络负责从原始输入特征中提取高级表征，实现如下：

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

### 推断网络 (AbductionNetwork)

推断网络负责从高维表征中推断共享潜在柯西随机向量的参数：

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

### 线性变换层 (LinearTransformationLayer)

线性变换层负责将潜在柯西向量映射到各类别得分随机变量：

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

### OvR概率计算层 (OvRProbabilityLayer)

OvR概率计算层负责计算样本属于每个类别的概率：

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

### 统一分类网络 (UnifiedClassificationNetwork)

统一分类网络整合了上述所有模块，构建完整的分类器：

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
```

## 损失函数实现

模型使用OvR策略的二元交叉熵损失：

```python
def compute_loss(self, y_true, class_probs):
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

## 训练流程实现

训练流程包括数据加载、模型训练、验证和早停机制：

```python
def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
    X_train_tensor = torch.FloatTensor(X_train).to(self.device)
    y_train_tensor = torch.LongTensor(y_train).to(self.device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state_dict = None
    
    has_validation = False
    if X_val is not None and y_val is not None:
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        has_validation = True
    
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=self.batch_size, shuffle=True
    )
    
    start_time = time.time()
    
    for epoch in range(self.epochs):
        self.model.train()
        epoch_train_loss = 0
        epoch_train_acc = 0
        
        for batch_X, batch_y in train_loader:
            class_probs, *_ = self.model(batch_X)
            loss = self.compute_loss(batch_y, class_probs)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_train_loss += loss.item()
            epoch_train_acc += self.compute_accuracy(batch_y, class_probs)
        
        # 计算平均损失和准确率
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_acc = epoch_train_acc / len(train_loader)
        self.history['train_loss'].append(avg_train_loss)
        self.history['train_acc'].append(avg_train_acc)
        
        # 验证
        if has_validation:
            self.model.eval()
            with torch.no_grad():
                val_class_probs, *_ = self.model(X_val_tensor)
                val_loss = self.compute_loss(y_val_tensor, val_class_probs)
                val_acc = self.compute_accuracy(y_val_tensor, val_class_probs)
            
            self.history['val_loss'].append(val_loss.item())
            self.history['val_acc'].append(val_acc)
            
            # 早停
            if self.early_stopping_patience is not None:
                if val_loss < best_val_loss - self.early_stopping_min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state_dict = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= self.early_stopping_patience:
                    break
    
    # 加载最佳模型
    if best_model_state_dict is not None:
        self.model.load_state_dict(best_model_state_dict)
    
    return self
```

## 实验评估实现

实验评估包括模型评估、结果可视化和实验配置：

```python
def evaluate_model(self, model, X_test, y_test):
    # 预测概率和类别
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    
    # 对于二分类问题，计算精确率、召回率和F1分数
    if len(np.unique(y_test)) == 2:
        metrics['precision'] = precision_score(y_test, y_pred)
        metrics['recall'] = recall_score(y_test, y_pred)
        metrics['f1'] = f1_score(y_test, y_pred)
    else:
        # 对于多分类问题，计算宏平均和加权平均
        metrics['precision_macro'] = precision_score(y_test, y_pred, average='macro')
        metrics['recall_macro'] = recall_score(y_test, y_pred, average='macro')
        metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro')
        metrics['precision_weighted'] = precision_score(y_test, y_pred, average='weighted')
        metrics['recall_weighted'] = recall_score(y_test, y_pred, average='weighted')
        metrics['f1_weighted'] = f1_score(y_test, y_pred, average='weighted')
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    return metrics, y_pred_proba, y_pred, cm
```

## 不确定性可视化实现

不确定性可视化展示了模型对每个类别的"中心判断"和"模糊度"：

```python
def visualize_uncertainty(self, model, X_test, y_test, n_samples=10, save_path=None):
    # 获取模型内部的统一分类网络
    unified_net = model.model
    
    # 随机选择n_samples个样本
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    X_samples = X_test[indices]
    y_samples = y_test[indices]
    
    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X_samples).to(model.device)
    
    # 获取预测结果和不确定性参数
    unified_net.eval()
    with torch.no_grad():
        class_probs, loc, scale, _, _ = unified_net(X_tensor)
    
    # 转换为NumPy数组
    class_probs = class_probs.cpu().numpy()
    loc = loc.cpu().numpy()
    scale = scale.cpu().numpy()
    
    # 可视化
    n_classes = class_probs.shape[1]
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4 * n_samples))
    
    for i, ax in enumerate(axes):
        # 绘制每个类别的位置参数和尺度参数
        ax.bar(range(n_classes), loc[i], alpha=0.6, label='Location (Center)')
        ax.errorbar(range(n_classes), loc[i], yerr=scale[i], fmt='o', capsize=5, label='Scale (Uncertainty)')
        
        # 标记真实类别
        true_class = y_samples[i]
        ax.axvline(x=true_class, color='r', linestyle='--', label=f'True Class: {true_class}')
        
        # 标记预测类别
        pred_class = np.argmax(class_probs[i])
        ax.axvline(x=pred_class, color='g', linestyle=':', label=f'Predicted Class: {pred_class}')
        
        ax.set_xticks(range(n_classes))
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title(f'Sample {i+1}: Uncertainty Visualization')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        
    return fig
```

## 数值稳定性考量

在实现过程中，我们特别关注了数值稳定性：

1. 使用`softplus`函数确保尺度参数为正值
2. 在计算概率时，使用`torch.clamp`函数确保尺度参数不为零，避免除零错误
3. 在计算概率时，使用`torch.clamp`函数确保概率在[0,1]范围内，避免对数溢出
4. 在计算损失时，使用`torch.log`函数的稳定版本，避免对数溢出

## 参数初始化策略

线性变换层的权重初始化对模型性能有重要影响，我们使用Kaiming初始化：

```python
def reset_parameters(self):
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in)
    nn.init.uniform_(self.bias, -bound, bound)
```

这种初始化方法考虑了输入维度，有助于保持前向传播中的方差稳定，避免梯度消失或爆炸问题。
