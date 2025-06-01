# 🚀 5分钟快速开始

欢迎使用 CAAC！本指南将帮助您在5分钟内完成环境搭建并运行第一个实验。

## ⚡ 一键快速体验

如果您只想快速体验项目功能，可以直接运行：

```bash
# 克隆项目
git clone https://github.com/1587causalai/caac.git
cd caac_project

# 激活环境并运行快速测试
conda activate base
python run_experiments.py --quick
```

> 🎯 **预期时间**: 3-5分钟  
> 🔍 **测试内容**: 4个小数据集上的鲁棒性测试

## 📋 详细安装步骤

### 第1步: 环境准备

```bash
# 确保使用推荐的conda环境
conda activate base

# 检查Python版本 (需要3.7+)
python --version
```

### 第2步: 安装依赖

```bash
# 安装核心依赖
pip install torch scikit-learn matplotlib pandas numpy seaborn

# 验证安装
python -c "import torch, sklearn, matplotlib; print('依赖安装成功!')"
```

### 第3步: 验证安装

```bash
# 查看所有可用实验
python run_experiments.py

# 应该看到类似这样的输出：
```

```
🧠 CAAC Project - Shared Latent Cauchy Vector OvR Classifier
============================================================
🔬 Available Experiments:

  🚀 --quick        Quick robustness test (3-5 minutes)
  🔬 --standard     Standard robustness test (15-25 minutes)  
  📊 --comparison   Basic method comparison
  🎯 --outlier      Outlier robustness test
  🎮 --interactive  Interactive experiment designer
```

## 🎯 运行第一个实验

### 快速鲁棒性测试 (推荐)

```bash
python run_experiments.py --quick
```

该命令将：
- ✅ 在4个经典数据集上测试CAAC算法
- ✅ 对比5种不同的分类方法
- ✅ 测试在0%-20%标签噪声下的鲁棒性
- ✅ 生成详细的可视化报告

**预期输出示例：**
```
🚀 Starting Quick Robustness Test...
📊 Loading datasets...
🔬 Testing CAAC OvR (Cauchy)...
📈 Accuracy: 0.9565, F1: 0.9582
✅ Experiment completed successfully!
📁 Results saved to: results/
```

### 交互式实验设计

如果您想自定义实验参数：

```bash
python run_experiments.py --interactive
```

系统会引导您：
1. 选择实验类型
2. 配置参数（数据集、噪声水平等）
3. 运行自定义实验

## 📊 查看结果

实验完成后，结果保存在 `results/` 目录：

```
results/
├── 📊 caac_outlier_robustness_curves.png      # 鲁棒性曲线图
├── 📊 caac_outlier_robustness_heatmap.png     # 热力图对比
├── 📈 caac_outlier_robustness_detailed.csv    # 详细数据
├── 📝 caac_outlier_robustness_report.md       # 实验报告
└── 📋 caac_outlier_robustness_summary.csv     # 结果摘要
```

### 结果解读

**鲁棒性得分**: 越接近1.0表示在噪声环境下性能越稳定
**基线准确率**: 无噪声情况下的分类准确率  
**性能衰减**: 在最高噪声水平下的性能下降程度

## 🎮 更多实验类型

### 标准鲁棒性测试 (15-25分钟)

```bash
python run_experiments.py --standard
```
- 覆盖8个数据集，74,000+样本
- 更全面的性能评估
- 适合研究和发表

### 方法对比分析 (5-10分钟)

```bash
python run_experiments.py --comparison
```
- 专注于不同方法的性能对比
- 生成对比图表和统计分析
- 适合方法选择和评估

### 离群值鲁棒性测试 (10-20分钟)

```bash
python run_experiments.py --outlier
```
- 测试对极端异常值的抵抗能力
- 模拟真实世界的数据质量问题
- 适合鲁棒性研究

## 🔧 自定义配置

您可以通过修改配置来自定义实验：

```python
from src.experiments.experiment_manager import ExperimentManager

# 创建实验管理器
manager = ExperimentManager()

# 自定义配置
custom_config = {
    'datasets': ['iris', 'wine'],           # 选择特定数据集
    'noise_levels': [0.0, 0.1, 0.2],      # 自定义噪声水平
    'epochs': 50,                           # 调整训练轮数
    'representation_dim': 64                # 修改表征维度
}

# 运行自定义实验
result_dir = manager.run_quick_robustness_test(**custom_config)
```

## 🚨 常见问题解决

### Q: 出现导入错误？
```bash
# 确保在项目根目录
cd /path/to/caac_project
python run_experiments.py --quick
```

### Q: 依赖包缺失？
```bash
# 重新安装依赖
conda activate base
pip install torch scikit-learn matplotlib pandas numpy seaborn
```

### Q: 内存不足？
```bash
# 运行小规模测试
python run_experiments.py --quick
# 或减少数据集数量（使用交互式模式）
python run_experiments.py --interactive
```

### Q: 训练时间太长？
```bash
# 使用快速配置
python run_experiments.py --quick
# 或在交互式模式中减少epochs
```

## 🎯 下一步建议

完成快速开始后，建议您：

1. **📖 深入理解**: 阅读 [理论基础](../theory/motivation.md) 了解算法原理
2. **🛠️ 探索API**: 查看 [API文档](../api/caac_ovr_model.md) 了解详细接口
3. **📊 分析结果**: 学习 [结果解读](result_interpretation.md) 深入分析
4. **🔬 自定义实验**: 使用 [实验配置](experiment_config.md) 设计专属实验

## 💡 使用技巧

- 🚀 **新手推荐**: 从 `--quick` 开始，验证环境和功能
- 🔬 **研究使用**: 运行 `--standard` 获取完整的性能数据
- 🎮 **高级用户**: 使用 `--interactive` 进行精确的实验控制
- 📊 **结果分析**: 结合多种实验类型获得全面认知

---

🎉 **恭喜！** 您已成功完成CAAC项目的快速入门。现在可以开始探索这个强大的多分类器了！

> 💬 **需要帮助？** 查看 [常见问题](faq.md) 或在GitHub Issues中提问。
