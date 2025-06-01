# 🚀 CAAC Project 快速开始

基于共享潜在柯西向量的 One-vs-Rest 多分类器项目的快速使用指南。

## ⚡ 5分钟快速体验

### 1. 环境准备
```bash
# 激活conda环境
conda activate base

# 检查依赖 (如果缺少请安装)
pip install torch scikit-learn matplotlib pandas numpy seaborn
```

### 2. 运行第一个实验
```bash
# 查看所有可用实验
python run_experiments.py

# 运行快速鲁棒性测试 (3-5分钟)
python run_experiments.py --quick
```

![20250601224346](https://s2.loli.net/2025/06/01/craV6wnQljB3HCp.png)

### 3. 查看结果
实验完成后，结果会保存在 `results/` 目录下，包含：
- 📊 可视化图表 (`.png` 文件)
- 📈 详细指标 (`experiment_results.json`)
- 📝 训练历史 (`training_history.json`)

## 🎯 推荐实验流程

### 新用户入门
```bash
# 1. 快速验证环境 (3-5分钟)
python run_experiments.py --quick

# 2. 深入测试性能 (15-25分钟)
python run_experiments.py --standard

# 3. 方法对比分析
python run_experiments.py --comparison
```

### 研究人员深度使用
```bash
# 交互式实验设计 (自定义参数)
python run_experiments.py --interactive

# 离群值鲁棒性测试
python run_experiments.py --outlier
```

## 🔧 核心文件说明

- **理论基础**: `docs/theory/motivation.md` - 数学原理和动机
- **核心算法**: `src/models/caac_ovr_model.py` - CAAC算法实现
- **实验入口**: `run_experiments.py` - 统一实验接口 (推荐)
- **重构指南**: `REFACTOR_GUIDE.md` - 项目重构说明

## 🎮 交互式实验设计

```bash
python run_experiments.py --interactive
```

交互式模式允许您：
- 📋 选择实验类型
- ⚙️ 自定义参数配置  
- 🎯 选择特定数据集
- 🔧 调整训练参数

## 📊 实验类型说明

| 实验类型 | 时间 | 数据集数量 | 主要目的 |
|---------|------|-----------|----------|
| `--quick` | 3-5分钟 | 4个小数据集 | 快速验证和环境测试 |
| `--standard` | 15-25分钟 | 8个数据集 | 标准鲁棒性评估 |
| `--comparison` | 5-10分钟 | 4个经典数据集 | 方法对比分析 |
| `--outlier` | 10-20分钟 | 可配置 | 离群值鲁棒性测试 |

## 🚨 常见问题

### Q: 运行时出现导入错误？
```bash
# 确保在项目根目录运行
cd /path/to/caac_project
python run_experiments.py --quick
```

### Q: 缺少依赖包？
```bash
# 安装所需依赖
conda activate base
pip install torch scikit-learn matplotlib pandas numpy seaborn
```

### Q: 想使用原有脚本？
```bash
# 原有脚本依然可用
python run_quick_robustness_test.py
python compare_methods.py
```

### Q: 需要自定义实验配置？
```bash
# 使用交互式模式
python run_experiments.py --interactive

# 或直接编辑ExperimentManager的default_configs
```

## 📈 结果解读

### 鲁棒性测试结果
- **鲁棒性得分**: 越接近1.0越好
- **基线准确率**: 无噪声情况下的准确率
- **性能衰减**: 噪声影响下的性能下降程度

### 可视化文件
- `robustness_curves.png` - 鲁棒性曲线图
- `robustness_heatmap.png` - 热力图对比
- `method_comparison.png` - 方法对比图
- `uncertainty_analysis.png` - 不确定性分析

## 🔄 项目结构

```
caac_project/
├── run_experiments.py          # 🆕 主入口 (推荐)
├── QUICK_START.md              # 🆕 快速开始指南
├── REFACTOR_GUIDE.md           # 🆕 重构指南
├── 
├── src/models/caac_ovr_model.py # 核心算法
├── docs/theory/motivation.md   # 理论基础
├── 
├── # 原有脚本 (向后兼容)
├── run_quick_robustness_test.py
├── compare_methods.py
└── ...
```

## 🎯 下一步

1. **阅读理论**: 查看 `docs/theory/motivation.md` 了解算法原理
2. **查看代码**: 研究 `src/models/caac_ovr_model.py` 了解实现细节  
3. **运行实验**: 使用 `python run_experiments.py --interactive` 设计实验
4. **分析结果**: 查看生成的可视化和数据文件

---

💡 **提示**: 如果您是研究人员，建议先运行 `--quick` 验证环境，然后使用 `--interactive` 模式进行深度实验。 