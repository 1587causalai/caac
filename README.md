# CAAC Project - Shared Latent Cauchy Vector OvR Classifier

基于共享潜在柯西向量的 One-vs-Rest (OvR) 多分类器项目。

## 项目概述

本项目实现了一种新颖的多分类器架构——**基于共享潜在柯西向量的 One-vs-Rest (OvR) 分类器**。该模型结合了 OvR 策略的效率和可扩展性，以及柯西分布对决策不确定性的显式建模能力，并引入共享的潜在随机向量来隐式捕捉类别间的相关性。

## 核心特性

- 🚀 **高效的 OvR 策略**：可并行训练，易于扩展到大量类别
- 🎯 **不确定性量化**：通过柯西分布参数量化决策的"中心"和"模糊度"
- 🔗 **类别相关性建模**：通过共享潜在向量捕捉类别间的内在联系
- 📊 **完整的实验框架**：包含训练、评估、可视化的完整流程

## 项目结构

```
caac_project/
├── 📄 run_experiments.py         # 🎯 统一实验入口 (主要使用)
├── 📄 QUICK_START.md             # 📖 快速开始指南
├── 
├── src/                          # 源代码
│   ├── models/                   # 🧠 核心算法实现
│   │   └── caac_ovr_model.py    # CAAC算法主体
│   ├── experiments/              # 🔬 实验模块
│   │   ├── experiment_manager.py # 实验管理器
│   │   ├── robustness_experiments.py # 鲁棒性测试
│   │   ├── comparison_experiments.py # 方法对比
│   │   └── outlier_experiments.py # 离群值测试
│   ├── data/                     # 📊 数据处理
│   ├── evaluators/               # 📈 评估器
│   └── utils/                    # 🛠️ 工具函数
├── 
├── docs/                         # 📚 文档
│   └── theory/motivation.md     # 理论基础
├── results/                      # 📁 实验结果
└── tests/                        # 🧪 测试代码
```

## 🚀 快速开始

### ⚡ 3步开始使用

```bash
# 1. 激活环境
conda activate base

# 2. 查看所有实验选项
python run_experiments.py

# 3. 运行你的第一个实验 (3-5分钟)
python run_experiments.py --quick
```

📖 **详细指南**: 查看 [`QUICK_START.md`](QUICK_START.md) 获取完整的快速开始指南

### 🔬 主要实验类型

```bash
python run_experiments.py --quick         # 快速鲁棒性测试 (3-5分钟)
python run_experiments.py --standard      # 标准鲁棒性测试 (15-25分钟)  
python run_experiments.py --comparison    # 方法对比分析
python run_experiments.py --interactive   # 交互式实验设计
```

### 📋 环境要求

- Python 3.7+ 
- PyTorch, scikit-learn, matplotlib, pandas, numpy, seaborn
- 推荐使用 `base` conda环境

### 🔄 项目架构

> **最新版本**: 项目已完成模块化重构，代码结构更加清晰。
>
> - ✅ **统一入口**: `python run_experiments.py` 提供所有实验功能
> - ✅ **模块化设计**: 实验逻辑已整理到 `src/experiments/` 目录
> - ✅ **简化使用**: 一键运行各类实验，支持交互式配置

## 📊 支持的数据集与实验

### 数据集概览
项目支持 **10个多样化数据集**，涵盖不同规模和特点：

- **小型数据集**: Iris, Wine, Breast Cancer, Optical Digits  
- **中型数据集**: Digits, Synthetic Imbalanced, Forest Covertype, Letter Recognition
- **大型数据集**: MNIST, Fashion-MNIST

总计 **74,000+ 样本**，适合不同规模的实验需求。

### 实验配置
- **鲁棒性测试**: 0%, 5%, 10%, 15%, 20% 标签噪声
- **方法对比**: CAAC(Cauchy), CAAC(Gaussian), MLP(Softmax), MLP(OvR), MLP(Hinge)
- **数据分割**: 70% train / 15% val / 15% test
- **输出格式**: JSON数据 + 可视化图表

## 实验结果

### 🎯 标签噪声鲁棒性测试结果

基于扩展的10个数据集的鲁棒性测试表明，**CAAC (Cauchy) 方法在标签噪声环境下表现最佳**：

#### 鲁棒性排名 (综合74,000+样本):

| 排名 | 方法 | 鲁棒性得分 | 基线准确率 | 性能衰减 | 特点 |
|-----|------|-----------|-----------|---------|------|
| 🥇 | **CAAC OvR (Cauchy)** | **0.9539** | 0.9623 | **1.7%** | 最稳定 |
| 🥈 | CAAC OvR (Gaussian) | 0.9378 | 0.9538 | 2.7% | 次佳鲁棒性 |
| 🥉 | MLP (OvR Cross Entropy) | 0.9376 | 0.9661 | 4.0% | 传统方法最佳 |
| 4 | MLP (Crammer & Singer Hinge) | 0.9368 | 0.9269 | 2.4% | 铰链损失 |
| 5 | MLP (Softmax) | 0.9331 | 0.9615 | **7.0%** | 衰减最大 |

#### 关键发现：

- ✅ **CAAC (Cauchy) 最鲁棒**: 在20%标签噪声下仅衰减1.7%
- ✅ **柯西分布优于高斯分布**: 鲁棒性得分高1.6%
- ✅ **CAAC方法整体优秀**: 在5种方法中占据前2名

### 📊 基础性能测试结果

模型在四个标准分类数据集上的表现：

| Dataset | Accuracy | F1 Score (Macro) | Training Time |
|---------|----------|------------------|---------------|
| Iris | 96.67% | 96.67% | ~0.1s |
| Wine | 94.44% | 94.44% | ~0.1s |
| Breast Cancer | 97.37% | 97.18% | ~0.2s |
| Digits | 95.56% | 95.53% | ~0.5s |

#### 与其他方法的比较

| Dataset | 本方法 | Softmax | 标准OvR | 原始CAAC |
|---------|--------|---------|---------|----------|
| Iris | **96.67%** | 93.33% | 90.00% | 86.67% |
| Wine | **94.44%** | 91.67% | 88.89% | 83.33% |
| Breast Cancer | **97.37%** | 95.61% | 94.74% | 92.98% |
| Digits | **95.56%** | 94.44% | 92.22% | 88.89% |

## 实验输出

每次实验会在 `results/` 目录下生成一个包含以下内容的文件夹：

- `experiment_results.json` - 详细的实验结果和配置
- `training_history.json` - 训练历史数据
- `confusion_matrix.png` - 混淆矩阵可视化
- `roc_curve.png` - ROC曲线图
- `uncertainty.png` - 不确定性分析图
- `training_history.png` - 训练过程可视化

## 模型配置

主要的模型参数：

- **表征维度 (representation_dim)**: 64
- **潜在柯西向量维度 (latent_dim)**: 32
- **特征网络隐藏层**: [128, 64]
- **推断网络隐藏层**: [64, 32]
- **判决阈值**: 0.0
- **学习率**: 0.001
- **批量大小**: 32
- **训练轮数**: 100（含早停机制）

## 理论基础

详细的理论动机和数学原理请参见 [`docs/motivation.md`](docs/motivation.md)。

关键创新点：
1. **共享潜在柯西向量**：通过学习低维柯西随机向量的参数，再线性变换到各类别得分
2. **OvR策略**：将多分类问题分解为多个独立的二分类问题
3. **不确定性建模**：柯西分布的位置和尺度参数提供决策的"中心"和"模糊度"信息

## 贡献

本项目的主要贡献：

1. 提出了基于共享潜在柯西向量的新型OvR多分类器架构
2. 实现了完整的训练、评估和可视化框架
3. 在多个标准数据集上验证了方法的有效性
4. 提供了详细的不确定性分析和可解释性研究

## 🚀 快速使用指南

### 主要命令 (推荐使用)
```bash
# 查看所有实验选项
python run_experiments.py

# 快速验证环境 (3-5分钟)
python run_experiments.py --quick

# 标准鲁棒性测试 (15-25分钟)
python run_experiments.py --standard

# 交互式实验设计
python run_experiments.py --interactive

# 方法对比分析
python run_experiments.py --comparison
```

### 环境准备
```bash
# 激活conda环境
conda activate base

# 确保依赖已安装
pip install torch scikit-learn matplotlib pandas numpy seaborn
```

### 结果查看
实验完成后，结果保存在 `results/` 目录：
- 📊 可视化图表 (`.png` 文件)
- 📈 详细数据 (`.csv` 文件) 
- 📝 实验报告 (`.md` 文件)

## 📁 核心文件说明

- **主入口**: `run_experiments.py` - 统一的实验运行接口
- **核心算法**: `src/models/caac_ovr_model.py` - CAAC算法实现
- **理论基础**: `docs/theory/motivation.md` - 数学原理和动机
- **快速指南**: `QUICK_START.md` - 详细的使用说明

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交 Issue 或联系项目维护者。 