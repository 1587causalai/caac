# CAAC - Cauchy Abduction Action Classification

欢迎访问CAAC-SPSFT（Cauchy Abduction Action Classification with Stochastic Pathway Selection and Fixed Thresholds）的文档网站。

## 项目简介

CAAC-SPSFT是一种基于柯西分布的因果推断分类模型，是对CAAR（Cauchy Abduction Action Regression）回归模型的扩展。该模型保留了柯西分布的重尾特性和因果解释性，同时引入了固定阈值机制和多路径混合策略，确保机制不变性。

**🎉 2025年1月更新：模型已扩展支持多分类任务！**

## 核心特点

- **因果表征**：通过柯西分布建模潜在变量，提高模型对异常值的鲁棒性
- **机制不变性**：输入数据仅影响因果表征生成，而因果表征到分类结果的映射机制是固定的
- **多路径混合**：通过多条并行的"解读路径"增强模型表达能力
- **固定阈值**：全局固定的分类阈值确保机制不变性
- **多分类支持**：从二分类扩展到任意数量类别的分类任务

## 快速开始

### 安装

```bash
pip install torch pandas matplotlib scikit-learn xgboost seaborn
```

### 基础示例（二分类）

```python
from src.models.caac_model import CAACModelWrapper
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 生成数据
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 创建并训练模型
model = CAACModelWrapper(input_dim=20, n_classes=2, n_paths=2)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
```

### 多分类示例

```python
# 生成5分类数据
X, y = make_classification(n_samples=2000, n_features=20, n_classes=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 创建并训练模型
model = CAACModelWrapper(
    input_dim=20,
    n_classes=5,
    n_paths=5,  # 建议路径数等于类别数
    epochs=150
)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
```

## 实验结果

### 二分类性能

在合成数据集上的表现：

| 数据类型 | 条件 | 准确率 | F1分数 | AUC |
|---------|------|--------|--------|-----|
| 线性数据 | 无异常值 | 0.985 | 0.985 | 0.996 |
| 线性数据 | 10%异常值 | 0.915 | 0.913 | 0.967 |
| 非线性数据 | 无异常值 | 0.930 | 0.929 | 0.972 |
| 非线性数据 | 10%异常值 | 0.815 | 0.812 | 0.880 |

### 多分类性能

#### 合成数据集

| 类别数 | 数据类型 | 准确率 | F1 (Macro) | AUC (OvR) | 训练时间 |
|--------|----------|--------|------------|-----------|----------|
| 3类 | 清洁数据 | 0.970 | 0.970 | 0.994 | 1.82s |
| 3类 | 10%异常值 | 0.863 | 0.864 | 0.945 | 1.24s |
| 4类 | 清洁数据 | 0.845 | 0.845 | 0.951 | 1.49s |
| 4类 | 10%异常值 | 0.772 | 0.771 | 0.899 | 1.10s |
| 5类 | 清洁数据 | 0.762 | 0.761 | 0.910 | 3.14s |
| 5类 | 10%异常值 | 0.638 | 0.640 | 0.846 | 2.15s |

#### 真实数据集

| 数据集 | 类别数 | 准确率 | F1 (Macro) | AUC (OvR) |
|--------|--------|--------|------------|-----------|
| Wine | 3 | 0.917 | 0.923 | 0.997 |
| Digits (0-4) | 5 | 0.961 | 0.961 | 0.988 |

### 鲁棒性分析

CAAC-SPSFT在面对异常值时表现出稳定的鲁棒性：
- 3类分类：准确率下降10.7%（相比逻辑回归的12.6%下降）
- 4类分类：准确率下降7.3%（相比逻辑回归的6.2%下降）
- 5类分类：准确率下降12.4%（相比逻辑回归的6.6%下降）

## 技术细节

### 多分类理论

对于 N 类分类任务，CAAC-SPSFT使用 N-1 个有序的固定阈值：

$$\theta_1^* < \theta_2^* < ... < \theta_{N-1}^*$$

每个类别的概率通过相邻阈值间的柯西CDF差值计算：

$$P(Y=k|x) = \sum_{j=1}^{K} \pi_j \cdot [F_{S_j}(\theta_k^*) - F_{S_j}(\theta_{k-1}^*)]$$

其中：
- $F_{S_j}$ 是路径 $j$ 的柯西累积分布函数
- $\pi_j$ 是路径 $j$ 的选择概率
- $\theta_0^* = -\infty$，$\theta_N^* = +\infty$

## 文档导航

- [理论设计](theory.md) - 详细的理论背景和数学推导
- [API文档](api.md) - 完整的API参考
- [二分类实验](experiments.md) - 详细的二分类实验结果
- [多分类扩展](multiclass_extension.md) - 多分类的实现细节和实验结果
- [真实数据集实验](experiments_real.md) - 在真实数据集上的表现

## 项目链接

- [GitHub仓库](https://github.com/1587causalai/caac)
- [PyPI包](https://pypi.org/project/caac/) (即将发布)

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

---
*最后更新时间: 2025-01-20* 