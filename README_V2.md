# Shared Cauchy OvR Classifier V2 🚀

## 概述

这是基于共享潜在柯西向量的 One-vs-Rest (OvR) 多分类器的全新实现，遵循 `docs/theory_v2.md` 中的理论设计。该架构结合了 OvR 策略的可扩展性与柯西分布的不确定性建模能力，特别适用于大规模分类任务。

## 🌟 核心特性

### 1. 创新架构
- **共享潜在柯西向量**: 学习低维潜在表示，隐式捕捉类别间相关性
- **线性变换映射**: 将潜在向量映射到类别特定的得分随机变量
- **柯西分布建模**: 为每个类别提供位置参数（中心判断）和尺度参数（不确定性）
- **OvR 策略**: 高效处理大规模类别数，支持并行训练和推理

### 2. 不确定性量化
- **精确建模**: 通过柯西分布参数直接量化决策不确定性
- **可解释性**: 清晰区分"中心判断"和"模糊度"
- **多种指标**: 提供概率、熵、尺度等多维度不确定性度量

### 3. 灵活的损失函数
- **标准 OvR BCE**: 基础二元交叉熵损失
- **加权 BCE**: 自动处理类别不平衡
- **Focal Loss**: 关注困难样本
- **不确定性正则化**: 引导合理的不确定性量化

## 📁 项目结构

```
src/models/
├── shared_cauchy_ovr.py      # 核心分类器实现
├── loss_functions.py         # 多种损失函数
├── trainer.py               # 训练、验证、评估框架
└── __init__.py              # 模块导出

test_v2_implementation.py    # 完整测试脚本
docs/theory_v2.md           # 理论文档
README_V2.md               # 本文档
```

## 🔧 核心组件

### SharedCauchyOvRClassifier

主要分类器类，实现完整的前向传播流程：

```python
from src.models import SharedCauchyOvRClassifier

model = SharedCauchyOvRClassifier(
    input_dim=100,        # 输入特征维度
    num_classes=50,       # 类别数量
    latent_dim=20,        # 潜在向量维度
    hidden_dims=[256, 128] # 特征提取器隐藏层
)

# 前向传播
outputs = model(x)
# 返回: probabilities, class_locations, class_scales, 
#       latent_locations, latent_scales, features

# 预测
predictions = model.predict(x)

# 不确定性分析
uncertainty_metrics = model.get_uncertainty_metrics(x)
```

### 多样化损失函数

```python
from src.models import create_loss_function

# 标准 OvR BCE
loss_fn = create_loss_function('ovr_bce')

# 加权 BCE (处理类别不平衡)
loss_fn = create_loss_function('weighted_ovr_bce', alpha=0.5)

# Focal Loss (关注困难样本)
loss_fn = create_loss_function('focal_ovr', gamma=2.0)

# 不确定性正则化
loss_fn = create_loss_function('uncertainty_reg', scale_regularizer_weight=0.01)
```

### 完整训练框架

```python
from src.models import SharedCauchyOvRTrainer
import torch.optim as optim

# 创建训练器
trainer = SharedCauchyOvRTrainer(
    model=model,
    loss_function=loss_fn,
    optimizer=optim.Adam(model.parameters(), lr=1e-3),
    device='auto'
)

# 训练
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    early_stopping_patience=10
)

# 评估
results = trainer.evaluate(test_loader)

# 不确定性分析
uncertainty_analysis = trainer.analyze_uncertainty(test_loader)

# 可视化
trainer.plot_training_history()
```

## 🧪 快速验证

运行完整测试确保实现正确：

```bash
python test_v2_implementation.py
```

测试包括：
- ✅ 基础功能测试 (模型前向传播、形状检查、概率约束)
- ✅ 损失函数测试 (各种损失函数的计算验证)  
- ✅ 训练流程测试 (完整的训练-验证-测试-不确定性分析)

## 📊 实验结果

在合成 5 类数据集上的测试结果：
- **数据集**: 1000 样本，20 特征，5 类别
- **模型参数**: 4,047 个参数
- **训练**: 10 epochs，早停机制
- **结果**: 84% 测试准确率
- **不确定性**: 平均不确定性 0.061，平均置信度 0.800

## 🎯 核心优势

### 1. 可扩展性
- **并行性**: OvR 策略支持类别间独立训练和推理
- **模块化**: 新增类别只需添加对应的线性变换行
- **效率**: 避免 Softmax 在大规模类别下的昂贵归一化

### 2. 可解释性
- **参数语义**: `class_locations` 表示中心判断，`class_scales` 表示不确定性
- **不确定性可视化**: 提供多种不确定性指标和可视化工具
- **决策透明**: 每个类别的判决过程完全可追踪

### 3. 鲁棒性
- **厚尾分布**: 柯西分布对异常值和噪声更鲁棒
- **梯度稳定**: 基于标准 BCE 损失，优化过程稳定
- **正则化**: 内置不确定性正则化防止过拟合

### 4. 类别关联建模
- **共享表示**: 通过共享潜在向量隐式学习类别相关性
- **参数共享**: 当 `latent_dim < num_classes` 时实现参数降维
- **泛化能力**: 相关类别间的知识可以相互迁移

## 🔬 理论基础

详细的数学原理和理论推导请参考 [`docs/theory_v2.md`](docs/theory_v2.md)，包括：

1. **柯西分布性质**: 线性组合封闭性、厚尾特性
2. **模型架构**: 潜在向量→线性变换→类别得分→概率计算
3. **损失函数**: OvR 策略下的二元交叉熵优化
4. **不确定性建模**: 通过柯西分布参数量化决策模糊度

## 🚀 使用场景

特别适用于：
- **大规模分类**: 成千上万个类别的分类任务
- **不确定性敏感**: 需要量化预测置信度的场景
- **可解释性要求**: 需要理解模型决策过程的应用
- **类别不平衡**: 长尾分布的分类问题
- **在线学习**: 需要动态添加新类别的场景

## 🔄 与 V1 的主要差异

| 方面 | V1 (旧版) | V2 (新版) |
|------|-----------|-----------|
| **核心思想** | 复杂的路径机制 | 共享潜在柯西向量 |
| **数学基础** | 多层嵌套逻辑 | 柯西分布线性组合性质 |
| **可解释性** | 间接、复杂 | 直接、清晰 |
| **计算效率** | 较低 | 高效并行 |
| **不确定性** | 隐式 | 显式量化 |
| **可扩展性** | 受限 | 优秀 |

## 📈 后续发展

计划中的改进方向：
- **可学习阈值**: 将固定阈值 C_k 设为可学习参数
- **动态潜在维度**: 根据类别数量自适应调整潜在维度
- **层次化类别**: 支持类别间的层次化关系建模
- **实时推理**: 针对大规模在线推理的优化
- **多模态扩展**: 支持多模态输入的融合

---

*本实现完全遵循 theory_v2.md 的理论设计，为高可解释性、大规模多分类任务提供了优雅的解决方案。* 🎯 