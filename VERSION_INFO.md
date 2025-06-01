# CAAC v2 - Unified Architecture Comparison Framework

## 版本说明

**分支名称**: `v2`  
**创建日期**: 2025年1月31日  
**主要特点**: 基于统一架构的公平对比实验框架

## 核心改进

### 1. 统一网络架构设计
- **FeatureNet**: 特征提取网络 (输入维度 → 64维表示)
- **AbductionNet**: 溯因推理网络 (64维 → 潜在空间64维)  
- **ActionNet**: 行动决策网络 (64维 → 类别数量)

### 2. 公平对比实验
所有神经网络方法采用相同架构，仅损失函数不同：
- CAAC OvR (柯西分布) 
- CAAC OvR (高斯分布)
- MLP (Softmax)
- MLP (OvR Cross Entropy)

### 3. 综合基准测试
与经典机器学习方法全面对比：
- Multinomial Logistic Regression
- One-vs-Rest Strategy  
- SVM (RBF kernel)
- Random Forest
- Sklearn MLP

### 4. 完整实验框架
- 标准化数据预处理
- 统一评估指标
- 自动化报告生成
- 英文可视化图表

## 核心研究问题

**验证柯西分布尺度参数是否能提升分类性能**

通过统一架构确保对比公平性，专注分析分布选择对性能的影响。

## 文件结构

```
caac_project/
├── src/models/               # 核心模型实现
├── compare_methods.py        # 主要对比实验脚本  
├── docs/                     # 完整文档
├── README.md                 # 项目说明
└── architecture_design.md    # 架构设计文档
```

## 运行实验

```bash
# 使用base conda环境
conda activate base

# 运行完整对比实验
python compare_methods.py
```

## 主要贡献

1. **理论贡献**: 验证柯西分布在分类任务中的有效性
2. **方法贡献**: 提出统一架构对比框架
3. **实验贡献**: 全面的性能基准测试
4. **工程贡献**: 完整的开源实现和文档

---

*此版本专注于公平对比和性能验证，为CAAC方法提供科学严谨的实验证据。* 