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
├── src/                          # 源代码
│   ├── models/                   # 模型实现
│   │   ├── caac_ovr_model.py    # 主模型类
│   │   ├── unified_network.py    # 统一网络架构
│   │   └── ...
│   ├── experiments/              # 实验脚本
│   │   ├── run_experiments.py    # 单个实验运行
│   │   └── model_evaluator.py    # 模型评估器
│   └── data/                     # 数据处理模块
├── docs/                         # 文档
│   ├── motivation.md            # 理论动机
│   └── experiments.md           # 实验结果
├── results/                      # 实验结果
├── run_all_experiments.py       # 批量运行所有实验
├── generate_experiment_report.py # 生成实验报告
└── README.md                    # 项目说明
```

## 快速开始

### 环境要求

- Python 3.7+
- PyTorch
- scikit-learn
- matplotlib
- pandas
- numpy

### 安装依赖

```bash
pip install torch scikit-learn matplotlib pandas numpy seaborn
```

### 运行实验

#### 1. 运行单个数据集实验

```bash
# 进入实验目录
cd src/experiments

# 运行 Iris 数据集实验
python run_experiments.py --dataset iris

# 运行 Wine 数据集实验
python run_experiments.py --dataset wine

# 运行 Breast Cancer 数据集实验
python run_experiments.py --dataset breast_cancer

# 运行 Digits 数据集实验
python run_experiments.py --dataset digits
```

#### 2. 批量运行所有实验

```bash
# 在项目根目录运行
python run_all_experiments.py

# 包含方法比较报告
python run_all_experiments.py --comparison
```

#### 3. 生成实验报告

```bash
# 分析已有实验结果并生成详细报告
python generate_experiment_report.py

# 指定结果目录
python generate_experiment_report.py --results_dir results --output_dir reports
```

## 实验结果

模型在四个标准分类数据集上的表现：

| Dataset | Accuracy | F1 Score (Macro) | Training Time |
|---------|----------|------------------|---------------|
| Iris | 96.67% | 96.67% | ~0.1s |
| Wine | 94.44% | 94.44% | ~0.1s |
| Breast Cancer | 97.37% | 97.18% | ~0.2s |
| Digits | 95.56% | 95.53% | ~0.5s |

### 与其他方法的比较

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

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交 Issue 或联系项目维护者。 