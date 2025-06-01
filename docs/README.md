# CAAC 项目文档

基于共享潜在柯西向量的 One-vs-Rest (OvR) 多分类器

## 📖 关于本项目

本项目实现了一种新颖的多分类器架构——**基于共享潜在柯西向量的 One-vs-Rest (OvR) 分类器**。该模型结合了：

- 🚀 **OvR 策略的效率和可扩展性**
- 🎯 **柯西分布对决策不确定性的显式建模**
- 🔗 **共享潜在向量对类别相关性的隐式捕捉**

## 🗂️ 文档导航

### 理论基础
深入了解项目的理论背景和数学原理：

- [**项目动机**](theory/motivation.md) - 核心思想和设计理念
- [**数学基础**](theory/mathematical_foundations.md) - 详细的数学推导
- [**设计原则**](theory/design_principles.md) - 架构设计的考量

### 使用教程
快速上手和深入使用：

- [**快速开始**](tutorials/quickstart.md) - 5分钟快速体验
- [**安装指南**](tutorials/installation.md) - 详细的环境配置

### 实现细节
了解代码架构和API：

- [**架构概述**](implementation/architecture.md) - 高层设计概览
- [**API文档**](implementation/api/) - 详细的接口说明

### 实验结果
查看性能表现和分析：

- [**基准测试**](experiments/benchmark_results.md) - 标准数据集性能
- [**鲁棒性分析**](experiments/robustness_analysis.md) - 噪声和异常值测试

## 🚀 快速开始

```bash
# 克隆项目
git clone <repository-url>
cd caac_project

# 安装依赖
conda activate base
pip install -r requirements.txt

# 运行快速测试
python run_quick_robustness_test.py
```

详细说明请参见 [快速开始指南](tutorials/quickstart.md)。

## 📊 主要特性

- ✅ **高效并行训练** - OvR策略支持大规模类别
- ✅ **不确定性量化** - 柯西分布参数提供决策置信度
- ✅ **类别关系建模** - 共享潜在空间捕捉相关性
- ✅ **完整实验框架** - 从训练到评估的端到端支持
- ✅ **丰富的可视化** - 混淆矩阵、ROC曲线、不确定性分析

## 🏆 性能亮点

基于10个数据集、74,000+样本的测试显示：

| 方法 | 鲁棒性得分 | 噪声下性能衰减 |
|------|-----------|---------------|
| **CAAC (Cauchy)** | **0.9539** | **1.7%** |
| CAAC (Gaussian) | 0.9378 | 2.7% |
| MLP (OvR) | 0.9376 | 4.0% |
| MLP (Softmax) | 0.9331 | 7.0% |

详细结果请参见 [鲁棒性分析](experiments/robustness_analysis.md)。

## 🤝 贡献指南

欢迎贡献！请先阅读我们的：
- 代码风格指南
- 测试要求
- 文档标准

## 📝 许可证

MIT License

---

**开始探索：** 建议从 [项目动机](theory/motivation.md) 开始了解理论背景，然后跟随 [快速开始](tutorials/quickstart.md) 进行实践。
