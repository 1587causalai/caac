# 使用指南

## 安装

要使用共享潜在柯西向量的OvR多分类器，首先需要安装必要的依赖：

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn
```

然后，克隆项目仓库：

```bash
git clone https://github.com/1587causalai/caac.git
cd caac
```

## 快速开始

以下是一个简单的示例，展示如何使用共享潜在柯西向量的OvR多分类器：

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 导入模型
from src.models.caac_ovr_model import CAACOvRModel

# 加载数据集
X, y = load_iris(return_X_y=True)

# 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 创建模型
model = CAACOvRModel(
    input_dim=X.shape[1],
    representation_dim=64,
    latent_dim=32,
    n_classes=3,
    feature_hidden_dims=[128, 64],
    abduction_hidden_dims=[64, 32],
    threshold=0.0,
    lr=0.001,
    batch_size=32,
    epochs=100,
    early_stopping_patience=10
)

# 训练模型
model.fit(X_train, y_train, verbose=1)

# 预测
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# 评估
from sklearn.metrics import accuracy_score, classification_report
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

## 运行实验

要复现论文中的实验结果，可以使用提供的实验脚本：

```bash
cd src/experiments
python run_experiments.py --dataset iris
python run_experiments.py --dataset wine
python run_experiments.py --dataset breast_cancer
python run_experiments.py --dataset digits
```

实验结果将保存在`results`目录中。

## 可视化不确定性

共享潜在柯西向量的OvR多分类器的一个关键特性是能够量化决策的不确定性。以下是如何可视化模型的不确定性：

```python
from src.experiments.model_evaluator import ModelEvaluator

# 创建评估器
evaluator = ModelEvaluator()

# 可视化不确定性
uncertainty_fig = evaluator.visualize_uncertainty(
    model, 
    X_test, 
    y_test, 
    n_samples=5,
    save_path='uncertainty.png'
)
```

## 模型参数调整

以下是一些关键参数的调整建议：

- **representation_dim**：表征维度，通常设置为输入维度的2-4倍
- **latent_dim**：潜在柯西向量维度，通常设置为类别数量的2-4倍
- **feature_hidden_dims**：特征网络隐藏层维度，根据数据复杂度调整
- **abduction_hidden_dims**：推断网络隐藏层维度，根据数据复杂度调整
- **threshold**：判决阈值，默认为0.0，可以根据需要调整
- **lr**：学习率，通常设置为0.001，可以根据需要调整
- **batch_size**：批量大小，通常设置为32，可以根据数据量调整
- **epochs**：训练轮数，通常设置为100，可以根据需要调整
- **early_stopping_patience**：早停耐心值，通常设置为10，可以根据需要调整

## 高级用法

### 自定义特征网络

可以自定义特征网络，以适应不同的数据类型：

```python
import torch.nn as nn

# 自定义特征网络
class CustomFeatureNetwork(nn.Module):
    def __init__(self, input_dim, representation_dim):
        super(CustomFeatureNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, representation_dim)
        )
        
    def forward(self, x):
        return self.network(x)

# 使用自定义特征网络
from src.models.unified_network import UnifiedClassificationNetwork

# 创建统一分类网络
unified_net = UnifiedClassificationNetwork(
    input_dim=X.shape[1],
    representation_dim=64,
    latent_dim=32,
    n_classes=3,
    feature_hidden_dims=[128, 64],
    abduction_hidden_dims=[64, 32],
    threshold=0.0
)

# 替换特征网络
unified_net.feature_net = CustomFeatureNetwork(X.shape[1], 64)

# 创建模型包装类
model = CAACOvRModel(
    input_dim=X.shape[1],
    representation_dim=64,
    latent_dim=32,
    n_classes=3
)

# 替换模型
model.model = unified_net
```

### 分析类别间相关性

可以通过分析线性变换层的权重矩阵，观察类别间的相关性：

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 获取线性变换层的权重矩阵
weight = model.model.linear_transform.weight.data.cpu().numpy()

# 计算类别间的相关性
correlation = np.corrcoef(weight)

# 可视化类别间的相关性
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Class Correlation')
plt.savefig('class_correlation.png')
```

### 分析不确定性与性能的关系

可以分析模型的不确定性（尺度参数）与性能的关系：

```python
# 获取预测结果和不确定性参数
model.model.eval()
with torch.no_grad():
    class_probs, loc, scale, _, _ = model.model(torch.FloatTensor(X_test).to(model.device))

# 转换为NumPy数组
class_probs = class_probs.cpu().numpy()
loc = loc.cpu().numpy()
scale = scale.cpu().numpy()

# 获取预测类别
y_pred = np.argmax(class_probs, axis=1)

# 计算正确分类和错误分类的平均尺度参数
correct_mask = (y_pred == y_test)
correct_scale = scale[correct_mask].mean()
incorrect_scale = scale[~correct_mask].mean()

print(f"Correct classification average scale: {correct_scale:.4f}")
print(f"Incorrect classification average scale: {incorrect_scale:.4f}")
print(f"Ratio: {incorrect_scale / correct_scale:.4f}")
```
