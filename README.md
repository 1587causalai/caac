# CAAC Project - 柯西推断行动分类 (Cauchy Abduction Action Classification)

基于柯西分布的推断行动分类方法项目。

## 项目概述

本项目实现了一种新颖的分类器架构——**柯西推断行动分类(CAAC)**。该模型采用柯西分布对决策不确定性进行显式建模，通过因果推理机制实现更加鲁棒的分类性能，特别是在含有标签噪声的真实环境中表现出卓越的稳定性。

## 核心特性

- 🚀 **因果推理机制**：通过因果表征学习提升分类性能
- 🎯 **不确定性量化**：通过柯西分布参数量化决策的"中心"和"模糊度"
- 🛡️ **强鲁棒性**：在标签噪声环境下表现出卓越的稳定性
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
项目支持 **8个多样化数据集**，涵盖不同规模和特点：

- **标准基准数据集**: Iris, Wine, Breast Cancer, Optical Digits  
- **计算机视觉数据集**: Digits, Synthetic Imbalanced
- **大规模数据集**: Forest Cover Type, Letter Recognition

总计 **8个数据集**，涵盖从小规模到大规模的不同实验需求。

### 实验配置
- **鲁棒性测试**: 0%, 10%, 20% 标签噪声
- **方法对比**: CAAC(Cauchy), CAAC(Gaussian), MLP(Softmax), MLP(OvR Cross Entropy), MLP(Crammer & Singer Hinge)
- **数据分割**: 标准分割策略
- **输出格式**: JSON数据 + 可视化图表

## 实验结果

### 🎯 最新鲁棒性测试结果 (2025-06-02)

基于8个数据集的最新鲁棒性测试表明，**CAAC (Cauchy) 方法在标签噪声环境下表现最佳**：

#### 整体鲁棒性排名:

| 排名 | 方法 | 鲁棒性得分 | 基线准确率 | 最差准确率 | 性能衰减 |
|-----|------|-----------|-----------|-----------|---------|
| 🥇 | **CAAC (Cauchy)** | **0.8266** | 0.8887 | 0.7661 | **13.8%** |
| 🥈 | MLP (Crammer & Singer Hinge) | 0.8069 | 0.8819 | 0.7314 | 17.1% |
| 🥉 | MLP (OvR Cross Entropy) | 0.8039 | 0.8884 | 0.7269 | 18.2% |
| 4 | MLP (Softmax) | 0.8021 | 0.8842 | 0.7205 | 18.5% |
| 5 | CAAC (Gaussian) | 0.7922 | 0.8855 | 0.6767 | 23.6% |

#### 关键发现：

- ✅ **CAAC (Cauchy) 最鲁棒**: 在20%标签噪声下性能衰减最小(13.8%)
- ✅ **柯西分布优于高斯分布**: 鲁棒性得分高4.3%
- ✅ **显著超越传统方法**: 比最佳传统方法(MLP Hinge)高2.4%

### 📊 代表性数据集详细结果

#### Iris数据集鲁棒性表现
| 方法 | 0%噪声 | 10%噪声 | 20%噪声 |
|------|--------|---------|---------|
| **CAAC (Cauchy)** | **96.67%** | **86.67%** | **93.33%** |
| CAAC (Gaussian) | 93.33% | 86.67% | 66.67% |
| MLP (Crammer & Singer Hinge) | 96.67% | 96.67% | 93.33% |

#### Wine数据集鲁棒性表现
| 方法 | 0%噪声 | 10%噪声 | 20%噪声 |
|------|--------|---------|---------|
| **CAAC (Cauchy)** | **97.22%** | **91.67%** | **83.33%** |
| CAAC (Gaussian) | 97.22% | 83.33% | 69.44% |
| MLP (Softmax) | 97.22% | 94.44% | 80.56% |

#### Breast Cancer数据集鲁棒性表现
| 方法 | 0%噪声 | 10%噪声 | 20%噪声 |
|------|--------|---------|---------|
| **CAAC (Cauchy)** | **97.37%** | **87.72%** | **80.70%** |
| CAAC (Gaussian) | 94.74% | 87.72% | 78.07% |
| MLP (OvR Cross Entropy) | 95.61% | 82.46% | 81.58% |

### 🧪 平均性能分析（跨所有数据集）

| 方法 | 平均准确率 | 标准差 | 平均F1分数 | 平均训练时间 |
|------|-----------|--------|-----------|-------------|
| **CAAC (Cauchy)** | **82.66%** | 14.48% | **82.42%** | 2.53s |
| MLP (Crammer & Singer Hinge) | 80.69% | 15.83% | 80.75% | 2.27s |
| MLP (OvR Cross Entropy) | 80.39% | 14.86% | 80.32% | 1.93s |
| MLP (Softmax) | 80.21% | 15.61% | 80.24% | 2.00s |
| CAAC (Gaussian) | 79.22% | 14.80% | 79.19% | 2.65s |

## 实验输出

每次实验会在 `results/` 目录下生成一个包含以下内容的文件夹：

- `experiment_results.json` - 详细的实验结果和配置
- `training_history.json` - 训练历史数据
- `confusion_matrix.png` - 混淆矩阵可视化
- `roc_curve.png` - ROC曲线图
- `uncertainty.png` - 不确定性分析图
- `training_history.png` - 训练过程可视化

## 模型配置

根据`docs/theory/motivation.md`理论基础，实际实现包含完整的三层网络架构：

### 核心网络架构（严格按照理论实现）

1. **特征网络 (FeatureNetwork)**: 输入特征 $x$ → 高维确定性表征 $z$
2. **推断网络 (AbductionNetwork)**: 确定性表征 $z$ → 潜在柯西分布参数 $\mu(z), \sigma(z)$
3. **随机变量采样**: 从柯西分布采样潜在向量 $U \sim \text{Cauchy}(\mu(z), \sigma(z))$
4. **行动网络 (ActionNetwork)**: **关键线性变换** $S_k = ∑ A_{kj} * U_j + B_k$
5. **概率计算**: $P_k = P(S_k > C_k)$ 通过柯西CDF计算

**理论核心**: 步骤4的线性变换 $S_k = A*U + B$ 是整个方法的关键，体现了从因果表征随机变量到类别得分随机变量的直接映射。

### 标准鲁棒性测试配置
- **表征维度 (representation_dim)**: 128
- **潜在因果向量维度 (latent_dim)**: 128（默认等于representation_dim）
- **特征网络结构**: input_dim → [64] → 128 (representation_dim)
- **推断网络结构**: 128 → [128, 64] → {location_head:64→128, scale_head:64→128}
- **行动网络结构**: **核心线性变换** A[n_classes×128] + B[n_classes] 
  - 实现理论公式: $S_k = ∑ A_{kj} * U_j + B_k$
  - 从潜在随机变量U直接映射到得分随机变量S
- **决策阈值 (threshold)**: 0.0（固定），支持可学习阈值
- **学习率**: 0.001
- **批量大小**: 64  
- **训练轮数**: 150（含早停机制，patience=15）


### 高级功能配置
- **可学习阈值 (learnable_thresholds)**: False（默认）
- **唯一性约束 (uniqueness_constraint)**: False（默认）
- **唯一性采样数 (uniqueness_samples)**: 10
- **唯一性权重 (uniqueness_weight)**: 0.1


## 理论基础

详细的理论动机和数学原理请参见 [`docs/theory/motivation.md`](docs/theory/motivation.md)。

### CAAC核心创新点：
1. **因果推理机制**：通过因果表征学习捕捉数据的内在因果结构
2. **柯西分布建模**：利用柯西分布的厚尾特性增强对异常值的鲁棒性  
3. **不确定性量化**：柯西分布的位置和尺度参数提供决策的"中心"和"模糊度"信息
4. **渐进式学习**：通过端到端优化实现特征提取、因果推理、决策预测的联合学习

### 方法优势
- **强鲁棒性**: 在标签噪声环境下性能衰减最小
- **理论基础**: 基于因果推理和概率建模的坚实理论基础
- **实用性**: 保持与传统方法相当的计算效率

## 贡献

本项目的主要贡献：

1. 提出了基于柯西分布的因果推理分类方法(CAAC)
2. 验证了柯西分布在鲁棒性方面相对高斯分布的优势
3. 在8个标准数据集上进行了全面的鲁棒性评估
4. 提供了完整的实验框架和详细的性能分析
5. 实现了端到端的因果推理分类架构

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

## 引用

如果您在研究中使用了CAAC模型，请引用我们的工作：

```bibtex
@misc{caac2025,
  author = {Heyang Gong},
  title = {CAAC: Cauchy Abduction Action Classification},
  year = {2025},
  howpublished = {\url{https://github.com/1587causalai/caac}}
}
```

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交 Issue 或联系项目维护者。 