# CAAC-SPSFT 多分类扩展

本文档详细说明了CAAC-SPSFT模型从二分类到多分类的扩展实现。

## 1. 理论背景

根据设计文档，CAAC-SPSFT通过固定阈值机制实现多分类：

### 1.1 核心思想

对于 $N_{cl}$ 个类别的分类任务：
- 需要 $N_{cl}-1$ 个有序的固定阈值：$\theta_1^* < \theta_2^* < ... < \theta_{N_{cl}-1}^*$
- 每个类别 $k$ 的概率由相邻阈值之间的柯西CDF差值决定

### 1.2 分类概率计算

对于类别 $k$ 的概率：
$$P(Y=k|M=j,x) = F_{S_j}(\theta_k^*) - F_{S_j}(\theta_{k-1}^*)$$

其中：
- $F_{S_j}$ 是路径 $j$ 的柯西CDF
- $\theta_0^* = -\infty$，$\theta_{N_{cl}}^* = +\infty$

最终概率通过路径混合得到：
$$P(Y=k|x) = \sum_{j=1}^{K_{paths}} \pi_j \cdot P(Y=k|M=j,x)$$

## 2. 实现细节

### 2.1 更新的组件

#### 2.1.1 ClassificationHead

扩展了分类头以支持多分类：

```python
def forward(self, mu_scores, gamma_scores, path_probs, thresholds):
    if self.n_classes == 2:
        # 二分类逻辑（保持不变）
    else:
        # 多分类逻辑
        # 1. 计算所有阈值的CDF值
        # 2. 计算类别概率：P(Y=k) = F(θ_k) - F(θ_{k-1})
        # 3. 通过路径概率加权平均
```

#### 2.1.2 FixedThresholdMechanism

改进了阈值机制的初始化和参数化：

- **参数化方式**：
  - 第一个阈值：直接学习 `raw_theta_1`
  - 后续阈值：通过正差值累加，$\theta_k = \theta_{k-1} + \exp(\text{raw\_delta}_k)$

- **初始化策略**：
  - 使用标准柯西分布的分位点初始化
  - 确保阈值均匀分布在概率空间

#### 2.1.3 评估指标

添加了多分类专用的评估函数：

```python
def evaluate_multiclass_classification(y_true, y_pred, y_pred_proba=None):
    # 宏平均和加权平均的精确率、召回率、F1分数
    # 每个类别的单独指标
    # One-vs-Rest 和 One-vs-One AUC
    # 混淆矩阵
```

### 2.2 模型配置建议

对于多分类任务的参数设置：

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `n_paths` | 等于类别数 | 每条路径可以专门化处理一个类别 |
| `latent_dim` | 32-64 | 因果表征维度 |
| `representation_dim` | 64-128 | 编码器输出维度 |
| `epochs` | 150-200 | 多分类通常需要更多训练轮次 |
| `early_stopping_patience` | 15-20 | 给模型更多机会收敛 |

## 3. 使用示例

### 3.1 运行多分类实验

```python
from src.experiments.multiclass_classification import run_multiclass_classification_experiment

# 运行3分类实验
results = run_multiclass_classification_experiment(
    n_samples=1500,
    n_features=10,
    n_classes=3,
    class_sep=1.0,
    outlier_ratio=0.1,  # 10%异常值
    model_params={
        'n_paths': 3,
        'representation_dim': 64,
        'latent_dim': 32
    }
)

# 查看结果
print(f"Accuracy: {results['metrics']['accuracy']:.3f}")
print(f"F1 (Macro): {results['metrics']['f1_macro']:.3f}")
```

### 3.2 批量实验

使用提供的脚本运行完整的多分类实验：

```bash
python run_multiclass_experiments.py
```

这将自动运行：
- 3类、4类、5类分类实验
- 有无异常值的对比
- 与基线方法的比较
- 生成可视化和分析报告

## 4. 实验结果解读

### 4.1 性能指标

多分类实验输出以下指标：
- **准确率**：整体分类正确率
- **F1分数**（宏平均和加权平均）：综合考虑精确率和召回率
- **AUC**（One-vs-Rest和One-vs-One）：多分类ROC曲线下面积
- **混淆矩阵**：详细展示各类别间的分类情况

### 4.2 鲁棒性分析

实验会自动生成鲁棒性分析图表，展示：
- 各模型在注入异常值后的准确率下降
- CAAC-SPSFT相对于基线方法的鲁棒性优势

### 4.3 可视化输出

每个实验生成4个子图：
1. **训练历史**：损失函数变化
2. **准确率历史**：训练和验证准确率
3. **混淆矩阵**：分类详细情况
4. **概率分布**：各类别的预测概率分布

## 5. 主要发现和建议

### 5.1 模型特点

1. **柯西分布的优势**：在多分类任务中，柯西分布的重尾特性继续提供鲁棒性
2. **路径专门化**：多条路径倾向于专门化处理不同类别
3. **阈值学习**：固定阈值通过梯度下降自动调整到合适位置

### 5.2 应用建议

1. **类别数量**：模型在3-5类分类任务上表现良好，更多类别可能需要调整架构
2. **路径数量**：建议设置为类别数量，但可以尝试更多路径以提高表达能力
3. **异常值处理**：CAAC-SPSFT在存在标签噪声和特征异常值时表现稳定

### 5.3 局限性

1. **计算复杂度**：随着类别数增加，模型参数和计算量线性增长
2. **类别不平衡**：当前实现假设类别相对平衡，严重不平衡可能需要额外处理
3. **高维问题**：对于类别数很多（>10）的任务，可能需要考虑层次化方法

## 6. 未来工作

1. **层次化多分类**：对于大量类别，实现层次化的分类策略
2. **自适应路径数**：根据任务复杂度自动确定最优路径数量
3. **类别不平衡处理**：集成针对不平衡数据的专门技术
4. **实际应用验证**：在更多真实多分类数据集上验证模型性能

---
*最后更新时间: 2025-01-20* 