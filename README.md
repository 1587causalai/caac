# CAAC - Cauchy Abduction Action Classification

基于柯西分布的因果推断分类模型，是对CAAR（Cauchy Abduction Action Regression）回归模型的扩展。

## 项目简介

CAAC-SPSFT（Cauchy Abduction Action Classification with Stochastic Pathway Selection and Fixed Thresholds）是一种基于柯西分布的因果推断分类模型。该模型保留了柯西分布的重尾特性和因果解释性，同时引入了固定阈值机制和多路径混合策略，确保机制不变性。

**2025年1月更新**：模型已扩展支持多分类任务！

## 核心特点

- **因果表征**：通过柯西分布建模潜在变量，提高模型对异常值的鲁棒性
- **机制不变性**：输入数据仅影响因果表征生成，而因果表征到分类结果的映射机制是固定的
- **多路径混合**：通过多条并行的"解读路径"增强模型表达能力
- **固定阈值**：全局固定的分类阈值确保机制不变性
- **多分类支持**：从二分类扩展到任意数量类别的分类任务

## 项目结构

```
caac/
├── src/
│   ├── models/
│   │   ├── base_networks.py      # 因果表征生成模块
│   │   ├── pathway_network.py    # 多路径网络
│   │   ├── threshold_mechanism.py # 固定阈值机制（支持多分类）
│   │   ├── classification_head.py # 分类概率计算（支持多分类）
│   │   ├── loss_functions.py     # NLL损失函数
│   │   └── caac_model.py         # 完整模型架构
│   ├── data/
│   │   └── synthetic.py          # 合成数据生成
│   ├── utils/
│   │   └── metrics.py            # 评估指标（含多分类指标）
│   └── experiments/
│       ├── binary_classification.py    # 二分类实验
│       └── multiclass_classification.py # 多分类实验
├── docs/                         # 文档网站
├── results/                      # 实验结果
├── run_experiments.py            # 二分类实验脚本
├── run_multiclass_experiments.py # 多分类实验脚本
└── run_real_experiments.py       # 真实数据集实验
```

## 安装与使用

### 安装依赖

```bash
pip install torch pandas matplotlib scikit-learn xgboost seaborn
```

### 运行二分类实验

```bash
python run_experiments.py
```

### 运行多分类实验

```bash
python run_multiclass_experiments.py
```

这将运行3类、4类和5类分类实验，包括有无异常值的对比。

### 运行真实数据集实验

```bash
python run_real_experiments.py
```

## 多分类扩展

### 理论基础

对于 N 类分类任务，CAAC-SPSFT使用 N-1 个有序的固定阈值：

$$\theta_1^* < \theta_2^* < ... < \theta_{N-1}^*$$

每个类别的概率通过相邻阈值间的柯西CDF差值计算：

$$P(Y=k|x) = \sum_{j=1}^{K} \pi_j \cdot [F_{S_j}(\theta_k^*) - F_{S_j}(\theta_{k-1}^*)]$$

### 使用示例

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
```

## 实验结果

### 二分类性能

CAAC-SPSFT模型在多种数据集上（线性/非线性，有/无异常值）与基线方法（逻辑回归、随机森林、SVM）进行了比较，展现出优秀的性能和鲁棒性。

### 多分类性能

在3-5类分类任务上的主要发现：
- **3类分类**：准确率达97%（清洁数据），86.3%（10%异常值）
- **鲁棒性**：在所有测试场景中，CAAC-SPSFT表现出稳定的异常值鲁棒性
- **计算效率**：训练时间随类别数线性增长，但仍在可接受范围内

详细实验结果请查看[文档网站](https://1587causalai.github.io/caac/)。

## 文档

- [理论设计](docs/theory.md)
- [API文档](docs/api.md)
- [多分类扩展说明](docs/multiclass_extension.md)
- [完整文档网站](https://1587causalai.github.io/caac/)

## 引用

如果您在研究中使用了CAAC-SPSFT模型，请引用我们的工作：

```bibtex
@misc{caac2025,
  author = {Heyang Gong},
  title = {CAAC: Cauchy Abduction Action Classification},
  year = {2025},
  howpublished = {\url{https://github.com/1587causalai/caac}}
}
```

## 许可证

MIT
