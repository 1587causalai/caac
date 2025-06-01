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
- seaborn

### 安装依赖

```bash
# 推荐使用conda环境
conda activate base
pip install torch scikit-learn matplotlib pandas numpy seaborn
```

### 运行实验

#### 🚀 标签噪声鲁棒性测试 (推荐)

**新增：基于扩展数据集的鲁棒性测试**

本项目现在支持在**10个多样化数据集**上进行标签噪声鲁棒性测试，总计74,000+样本！

##### 快速选择测试模式：

```bash
# 1. 快速验证 (3-5分钟) - 4个小数据集
python run_quick_robustness_test.py

# 2. 标准测试 (15-25分钟) - 8个数据集 (推荐)
python run_standard_robustness_test.py

# 3. 完整交互式测试 (自选数据集)
python compare_methods_outlier_robustness.py
```

##### 支持的数据集：

| 数据集 | 样本数 | 特征数 | 类别数 | 规模 | 特点 |
|-------|-------|-------|-------|------|-----|
| Iris | 150 | 4 | 3 | small | 经典平衡数据集 |
| Wine | 178 | 13 | 3 | small | 轻微不平衡 |
| Breast Cancer | 569 | 30 | 2 | small | 医疗诊断数据 |
| Optical Digits | 1,797 | 64 | 10 | small | 手写数字识别 |
| Digits | 1,797 | 64 | 10 | medium | 数字识别 |
| Synthetic Imbalanced | 5,000 | 20 | 5 | medium | 合成不平衡数据 |
| Forest Covertype | 10,000 | 54 | 7 | medium | 森林覆盖预测 |
| Letter Recognition | 20,000 | 16 | 26 | medium | 26类字母识别 |
| MNIST | 15,000 | 784 | 10 | large | 手写数字图像 |
| Fashion-MNIST | 20,000 | 784 | 10 | large | 服装图像分类 |

##### 测试配置：

- **噪声水平**: 0%, 5%, 10%, 15%, 20%
- **数据分割**: 70% train / 15% val / 15% test
- **方法对比**: CAAC(Cauchy), CAAC(Gaussian), MLP(Softmax), MLP(OvR), MLP(Hinge)
- **输出**: 详细报告、鲁棒性曲线、热力图、原始数据

#### 🔬 基础性能测试

##### 1. 运行单个数据集实验

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

##### 2. 批量运行所有实验

```bash
# 在项目根目录运行
python run_all_experiments.py

# 包含方法比较报告
python run_all_experiments.py --comparison
```

##### 3. 生成实验报告

```bash
# 分析已有实验结果并生成详细报告
python generate_experiment_report.py

# 指定结果目录
python generate_experiment_report.py --results_dir results --output_dir reports
```

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

## 🚀 快速命令参考

### 查看所有测试选项
```bash
python show_test_options.py  # 显示详细的测试选项和使用说明
```

### 鲁棒性测试 (推荐)
```bash
# 快速验证 (5分钟)
python run_quick_robustness_test.py

# 标准测试 (25分钟, 推荐用于研究)
python run_standard_robustness_test.py

# 交互式自定义测试
python compare_methods_outlier_robustness.py
```

### 数据集和环境测试
```bash
# 测试数据集加载
python test_new_datasets.py

# 检查环境依赖
conda activate base
pip install torch scikit-learn matplotlib pandas numpy seaborn
```

### 基础性能测试
```bash
# 批量基础测试
python run_all_experiments.py --comparison

# 单数据集测试
cd src/experiments
python run_experiments.py --dataset iris
```

## 📁 重要文件说明

- `compare_methods_outlier_robustness.py` - 主要的鲁棒性测试脚本
- `run_standard_robustness_test.py` - 一键标准测试
- `run_quick_robustness_test.py` - 一键快速测试  
- `show_test_options.py` - 测试选项概览
- `test_new_datasets.py` - 数据集功能测试
- `results/` - 所有实验结果和报告
- `src/` - 源代码和模型实现

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交 Issue 或联系项目维护者。 