# CAAC - Cauchy Abduction Action Classification

基于柯西分布的因果推断分类模型，是对CAAR（Cauchy Abduction Action Regression）回归模型的扩展。

## 项目简介

CAAC-SPSFT（Cauchy Abduction Action Classification with Stochastic Pathway Selection and Fixed Thresholds）是一种基于柯西分布的因果推断分类模型。该模型保留了柯西分布的重尾特性和因果解释性，同时引入了固定阈值机制和多路径混合策略，确保机制不变性。

## 核心特点

- **因果表征**：通过柯西分布建模潜在变量，提高模型对异常值的鲁棒性
- **机制不变性**：输入数据仅影响因果表征生成，而因果表征到分类结果的映射机制是固定的
- **多路径混合**：通过多条并行的"解读路径"增强模型表达能力
- **固定阈值**：全局固定的分类阈值确保机制不变性

## 项目结构

```
caac/
├── src/
│   ├── models/
│   │   ├── base_networks.py      # 因果表征生成模块
│   │   ├── pathway_network.py    # 多路径网络
│   │   ├── threshold_mechanism.py # 固定阈值机制
│   │   ├── classification_head.py # 分类概率计算
│   │   ├── loss_functions.py     # NLL损失函数
│   │   └── caac_model.py         # 完整模型架构
│   ├── data/
│   │   └── synthetic.py          # 合成数据生成
│   ├── utils/
│   │   └── metrics.py            # 评估指标
│   └── experiments/
│       └── binary_classification.py # 实验流程
├── docs/                         # 文档网站
├── results/                      # 实验结果
└── run_experiments.py            # 主实验运行脚本
```

## 安装与使用

### 安装依赖

```bash
pip install torch pandas matplotlib scikit-learn
```

### 运行实验

```bash
python run_experiments.py
```

## 实验结果

CAAC-SPSFT模型在多种数据集上（线性/非线性，有/无异常值）与基线方法（逻辑回归、随机森林、SVM）进行了比较，展现出优秀的性能和鲁棒性。

详细实验结果请查看[文档网站](https://1587causalai.github.io/caac/)。

## 文档

完整文档请访问：[https://1587causalai.github.io/caac/](https://1587causalai.github.io/caac/)

## 引用

如果您在研究中使用了CAAC-SPSFT模型，请引用我们的工作：

```
@misc{caac2025,
  author = {1587causalai},
  title = {CAAC: Cauchy Abduction Action Classification},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/1587causalai/caac}}
}
```

## 许可证

MIT
