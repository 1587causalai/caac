# CAAC - 共享潜在柯西向量的 OvR 多分类器

> 🧠 一种新颖且高度可解释的多分类器架构

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7+-green.svg)](https://python.org)
[![Framework](https://img.shields.io/badge/framework-PyTorch-red.svg)](https://pytorch.org)

## 🎯 项目概述

**基于共享潜在柯西向量的 One-vs-Rest (OvR) 多分类器** 是一个机器学习研究项目，旨在通过柯西分布的特殊性质来改善多分类任务的鲁棒性和不确定性建模。

### 核心创新点

- 🎯 **柯西分布建模**: 利用柯西分布的厚尾特性增强对异常值的鲁棒性
- 🔗 **共享潜在向量**: 通过共享表示捕捉类别间的内在相关性  
- 🚀 **OvR策略**: 可扩展到大规模类别数的高效分类架构
- 📊 **不确定性量化**: 显式建模决策过程中的不确定性

## ⚡ 快速开始

### 环境准备
```bash
# 激活conda环境
conda activate base

# 安装依赖
pip install torch scikit-learn matplotlib pandas numpy seaborn
```

### 运行第一个实验
```bash
# 查看所有实验选项
python run_experiments.py

# 快速鲁棒性测试 (3-5分钟)
python run_experiments.py --quick

# 交互式实验设计
python run_experiments.py --interactive
```

### 实验结果示例

![实验结果](assets/experiment_demo.png)

## 🏆 核心性能

基于 **74,000+ 样本** 的综合测试结果：

| 排名 | 方法 | 鲁棒性得分 | 基线准确率 | 性能衰减 |
|-----|------|-----------|-----------|---------|
| 🥇 | **CAAC OvR (Cauchy)** | **0.9539** | 0.9623 | **1.7%** |
| 🥈 | CAAC OvR (Gaussian) | 0.9378 | 0.9538 | 2.7% |
| 🥉 | MLP (OvR Cross Entropy) | 0.9376 | 0.9661 | 4.0% |

## 📊 实验能力

### 支持的数据集 (10个)
- **小型**: Iris, Wine, Breast Cancer, Optical Digits
- **中型**: Digits, Synthetic, Covertype, Letter Recognition  
- **大型**: MNIST, Fashion-MNIST

### 对比方法 (5种)
1. **CAAC OvR (Cauchy)** - 本项目核心方法
2. **CAAC OvR (Gaussian)** - 高斯分布变体
3. **MLP (Softmax)** - 传统神经网络
4. **MLP (OvR Cross Entropy)** - OvR交叉熵
5. **MLP (Crammer & Singer Hinge)** - 铰链损失

## 🔬 数学原理

详细的数学推导见 [理论基础](theory/motivation.md)

**核心思想**:
1. **潜在柯西向量**: $\mathbf{U} \sim \text{Cauchy}(\boldsymbol{\mu}(x), \boldsymbol{\sigma}(x))$
2. **线性变换**: $\mathbf{S} = \mathbf{A}\mathbf{U} + \mathbf{B}$  
3. **类别概率**: $P_k = P(S_k > C_k) = F_{\text{Cauchy}}^{-1}(C_k)$

## 🚀 主要功能

### 实验类型
| 命令 | 时间 | 数据集 | 用途 |
|------|------|--------|------|
| `--quick` | 3-5分钟 | 4个小数据集 | 环境验证 |
| `--standard` | 15-25分钟 | 8个数据集 | 标准评估 |
| `--comparison` | 5-10分钟 | 4个经典数据集 | 方法对比 |
| `--interactive` | 自定义 | 用户选择 | 定制实验 |

### 输出结果
- 📊 可视化图表 (鲁棒性曲线、热力图)
- 📈 详细数据 (CSV格式，便于进一步分析)
- 📝 实验报告 (Markdown格式，包含完整分析)

## 🔄 项目架构

```
caac_project/
├── 📄 run_experiments.py         # 🎯 统一实验入口
├── 📄 QUICK_START.md             # 📖 快速开始指南
├── 
├── src/                          # 源代码
│   ├── models/                   # 🧠 核心算法实现
│   │   └── caac_ovr_model.py    # CAAC算法主体
│   ├── experiments/              # 🔬 实验模块
│   │   ├── experiment_manager.py # 实验管理器
│   │   ├── robustness_experiments.py # 鲁棒性测试
│   │   └── comparison_experiments.py # 方法对比
│   └── ...
├── 
├── docs/                         # 📚 文档网站
├── results/                      # 📁 实验结果
└── tests/                        # 🧪 测试代码
```

## 📚 文档导航

- **📖 [快速开始](tutorials/quickstart.md)** - 5分钟上手指南
- **🔬 [理论基础](theory/motivation.md)** - 数学原理和动机
- **🛠️ [API文档](api/caac_ovr_model.md)** - 详细的API参考
- **📊 [实验结果](experiments/benchmark_results.md)** - 基准测试报告
- **🎯 [用户指南](tutorials/user_guide.md)** - 完整的使用说明

## 🎯 适用场景

- **研究人员**: 探索新的多分类方法和不确定性建模
- **开发者**: 需要鲁棒性强的分类器的实际应用
- **学习者**: 理解柯西分布在机器学习中的应用

## 🤝 贡献指南

我们欢迎各种形式的贡献！请查看 [贡献指南](development/contributing.md) 了解详情。

## 📞 获取帮助

- 📖 **文档**: 查看本网站的详细文档
- 🐛 **问题报告**: 在GitHub Issues中提交
- 💬 **讨论**: 加入我们的讨论社区

---

*🎯 让多分类任务更加鲁棒和可解释*
