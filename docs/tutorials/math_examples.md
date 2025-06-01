# 🔬 数学公式示例

本页面展示 CAAC 项目中使用的数学公式渲染效果，验证 KaTeX 支持。

## 📐 基础数学公式

### 行内公式

柯西分布的概率密度函数为 $f(x) = \frac{1}{\pi} \frac{\gamma}{(x-x_0)^2 + \gamma^2}$，其中 $x_0$ 是位置参数，$\gamma$ 是尺度参数。

CAAC模型的核心思想是学习潜在柯西向量 $\mathbf{U} \sim \text{Cauchy}(\boldsymbol{\mu}(x), \boldsymbol{\sigma}(x))$。

### 独立公式块

**柯西分布的概率密度函数：**

$$f(x; x_0, \gamma) = \frac{1}{\pi\gamma}\left[1 + \left(\frac{x-x_0}{\gamma}\right)^2\right]^{-1}$$

**CAAC模型的前向传播：**

$$\begin{align}
\mathbf{R} &= \text{FeatureNet}(\mathbf{x}) \\
\boldsymbol{\mu}(\mathbf{x}) &= \text{MLP}_\mu(\mathbf{R}) \\
\boldsymbol{\sigma}(\mathbf{x}) &= \text{Softplus}(\text{MLP}_\sigma(\mathbf{R})) \\
\mathbf{U} &\sim \text{Cauchy}(\boldsymbol{\mu}(\mathbf{x}), \boldsymbol{\sigma}(\mathbf{x})) \\
\mathbf{S} &= \mathbf{A}\mathbf{U} + \mathbf{B}
\end{align}$$

## 🎯 CAAC 核心数学原理

### 1. 潜在向量建模

CAAC 使用柯西分布对潜在向量进行建模：

$$\mathbf{U} = [\mathbf{U}_1, \mathbf{U}_2, \ldots, \mathbf{U}_d]^T$$

其中每个分量独立地服从柯西分布：

$$U_i \sim \text{Cauchy}(\mu_i(\mathbf{x}), \sigma_i(\mathbf{x}))$$

### 2. 线性变换

通过可学习的线性变换将潜在向量映射到分类分数：

$$\mathbf{S} = \mathbf{A}\mathbf{U} + \mathbf{B}$$

其中：
- $\mathbf{A} \in \mathbb{R}^{K \times d}$ 是变换矩阵
- $\mathbf{B} \in \mathbb{R}^K$ 是偏置向量
- $K$ 是类别数量

### 3. 类别概率计算

对于类别 $k$，其概率通过柯西分布的累积分布函数计算：

$$P(y = k | \mathbf{x}) = P(S_k > C_k) = 1 - F_{\text{Cauchy}}(C_k; \text{loc}_k, \text{scale}_k)$$

其中 $F_{\text{Cauchy}}$ 是柯西分布的累积分布函数：

$$F_{\text{Cauchy}}(x; x_0, \gamma) = \frac{1}{\pi}\arctan\left(\frac{x-x_0}{\gamma}\right) + \frac{1}{2}$$

## 📊 鲁棒性分析

### 损失函数

CAAC 使用负对数似然损失：

$$\mathcal{L} = -\sum_{i=1}^{N} \log P(y_i | \mathbf{x}_i)$$

### 不确定性量化

模型的不确定性通过尺度参数 $\boldsymbol{\sigma}(\mathbf{x})$ 来量化：

$$\text{Uncertainty}(\mathbf{x}) = \frac{1}{d}\sum_{i=1}^{d} \sigma_i(\mathbf{x})$$

### 鲁棒性度量

在噪声水平 $\epsilon$ 下的性能定义为：

$$\text{Robustness}(\epsilon) = \frac{1}{|\mathcal{D}_\epsilon|} \sum_{(\mathbf{x}, y) \in \mathcal{D}_\epsilon} \mathbb{I}[f(\mathbf{x}) = y]$$

其中 $\mathcal{D}_\epsilon$ 是添加噪声后的测试集。

## 🔄 优化算法

### 梯度计算

对于柯西分布参数的梯度：

$$\frac{\partial \mathcal{L}}{\partial \mu_i} = \frac{\partial \mathcal{L}}{\partial U_i} \cdot \frac{\partial U_i}{\partial \mu_i}$$

$$\frac{\partial \mathcal{L}}{\partial \sigma_i} = \frac{\partial \mathcal{L}}{\partial U_i} \cdot \frac{\partial U_i}{\partial \sigma_i}$$

### 重参数化技巧

为了实现反向传播，使用重参数化：

$$U_i = \mu_i(\mathbf{x}) + \sigma_i(\mathbf{x}) \cdot \tan(\pi(\mathcal{U} - 0.5))$$

其中 $\mathcal{U} \sim \text{Uniform}(0, 1)$。

## 🏆 性能指标

### 准确率

$$\text{Accuracy} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{I}[f(\mathbf{x}_i) = y_i]$$

### F1分数

对于多分类任务的宏平均F1分数：

$$\text{F1} = \frac{1}{K}\sum_{k=1}^{K} \frac{2 \cdot \text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$

### 鲁棒性得分

综合鲁棒性得分：

$$\text{Robustness Score} = \frac{\sum_{\epsilon} \text{Accuracy}(\epsilon)}{\sum_{\epsilon} \text{Baseline Accuracy}}$$

## 🧮 矩阵运算

### 协方差矩阵

类别间相关性通过协方差矩阵分析：

$$\mathbf{C} = \frac{1}{N-1}\sum_{i=1}^{N}(\mathbf{s}_i - \bar{\mathbf{s}})(\mathbf{s}_i - \bar{\mathbf{s}})^T$$

### 特征值分解

变换矩阵的谱分析：

$$\mathbf{A} = \mathbf{U}\mathbf{\Lambda}\mathbf{V}^T$$

其中 $\mathbf{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_r)$ 是奇异值矩阵。

## 🔍 示例计算

### 简单示例

假设输入 $\mathbf{x} = [1.0, 2.0]^T$，经过特征网络得到表征 $\mathbf{R} = [0.5, -0.3, 0.8]^T$。

通过参数网络计算：
- $\boldsymbol{\mu} = [0.2, -0.1]^T$
- $\boldsymbol{\sigma} = [0.5, 0.3]^T$

采样得到潜在向量：
$$\mathbf{U} = [0.1, 0.4]^T$$

应用线性变换：
$$\mathbf{S} = \begin{bmatrix} 0.8 & -0.2 \\ 0.3 & 0.9 \\ -0.1 & 0.6 \end{bmatrix} \begin{bmatrix} 0.1 \\ 0.4 \end{bmatrix} + \begin{bmatrix} 0.1 \\ -0.2 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 0.02 \\ 0.17 \\ 0.23 \end{bmatrix}$$

---

💡 **提示**: 这些数学公式展示了 CAAC 算法的完整数学框架。所有公式都应该在支持 KaTeX 的环境中正确渲染。

> 🔬 **验证**: 如果您能正确看到上述数学公式，说明 KaTeX 配置成功！ 