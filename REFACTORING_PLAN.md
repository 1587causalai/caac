# CAAC 项目重构计划

## 📋 概述

本文档概述了 CAAC 项目的重构计划，旨在解决以下主要问题：
1. 历史文档过时
2. 代码、实验设计和文档职责不清
3. 代码结构需要优化

## 🎯 重构目标

1. **清晰的职责分离**
   - 核心算法代码
   - 实验框架
   - 文档系统
   
2. **模块化设计**
   - 单一职责原则
   - 高内聚低耦合
   - 易于扩展和维护

3. **完善的文档体系**
   - 理论文档（不变的核心概念）
   - 实现文档（代码设计和API）
   - 使用文档（如何使用）
   - 实验文档（结果和分析）

## 📅 分阶段实施计划

### 第一阶段：清理和整理（1-2天）

#### 1.1 文档清理
- [ ] 审查 `docs/` 目录中的所有文档
- [ ] 识别过时内容并归档到 `docs/archive/`
- [ ] 保留核心理论文档（如 `motivation.md`）
- [ ] 创建新的文档结构

#### 1.2 代码清理
- [ ] 审查所有实验脚本
- [ ] 归档临时文件（如 `tmp.py`）
- [ ] 整理 `results/` 目录

#### 1.3 创建清晰的项目结构
```
caac_project/
├── src/
│   ├── models/           # 核心模型实现
│   ├── trainers/         # 训练器和优化器
│   ├── evaluators/       # 评估和指标
│   ├── data/            # 数据加载和处理
│   └── utils/           # 工具函数
├── experiments/         # 实验脚本（独立于src）
│   ├── configs/         # 实验配置
│   ├── scripts/         # 运行脚本
│   └── analysis/        # 结果分析
├── docs/
│   ├── theory/          # 理论和数学原理
│   ├── implementation/  # 实现细节
│   ├── api/            # API文档
│   └── tutorials/      # 使用教程
└── results/            # 实验结果
```

### 第二阶段：代码重构（3-4天）

#### 2.1 拆分 `caac_ovr_model.py`
- [ ] 将基础网络组件移到 `src/models/components/`
  - `feature_network.py`
  - `abduction_network.py`
  - `action_network.py`
- [ ] 将不同的模型类拆分到独立文件
  - `caac_cauchy.py` - CAACOvRModel
  - `caac_gaussian.py` - CAACOvRGaussianModel
  - `baseline_softmax.py` - SoftmaxMLPModel
  - `baseline_ovr.py` - OvRCrossEntropyMLPModel
  - `baseline_hinge.py` - CrammerSingerMLPModel
- [ ] 创建统一的模型接口 `base_model.py`

#### 2.2 创建训练器模块
- [ ] 提取训练逻辑到 `src/trainers/trainer.py`
- [ ] 实现早停机制 `src/trainers/early_stopping.py`
- [ ] 分离损失函数到 `src/trainers/losses.py`

#### 2.3 创建评估器模块
- [ ] 标准评估器 `src/evaluators/standard_evaluator.py`
- [ ] 鲁棒性评估器 `src/evaluators/robustness_evaluator.py`
- [ ] 可视化工具 `src/evaluators/visualizer.py`

### 第三阶段：文档更新（2-3天）

#### 3.1 理论文档
- [ ] 更新 `docs/theory/motivation.md`
- [ ] 创建 `docs/theory/mathematical_foundations.md`
- [ ] 创建 `docs/theory/design_principles.md`

#### 3.2 实现文档
- [ ] 创建 `docs/implementation/architecture.md`
- [ ] 创建 `docs/implementation/model_details.md`
- [ ] 自动生成 API 文档

#### 3.3 使用文档
- [ ] 创建 `docs/tutorials/quickstart.md`
- [ ] 创建 `docs/tutorials/advanced_usage.md`
- [ ] 创建 `docs/tutorials/custom_experiments.md`

### 第四阶段：实验框架改进（2-3天）

#### 4.1 标准化实验配置
- [ ] 创建 YAML/JSON 配置文件系统
- [ ] 实现实验配置验证
- [ ] 支持配置继承和覆盖

#### 4.2 创建实验管道
- [ ] 统一的实验运行器
- [ ] 自动化结果收集
- [ ] 标准化的报告生成

#### 4.3 基准测试套件
- [ ] 定义标准数据集集合
- [ ] 创建性能基准
- [ ] 实现自动化对比分析

## 🛠️ 技术决策

### 代码风格
- 使用 Type Hints
- 遵循 PEP 8
- 完善的 docstrings

### 测试策略
- 单元测试覆盖核心组件
- 集成测试覆盖训练流程
- 端到端测试覆盖实验

### 文档工具
- 继续使用 Docsify 作为文档框架
- 使用 Sphinx 生成 API 文档
- Jupyter notebooks 作为交互式教程

## 📊 成功指标

1. **代码质量**
   - 单个文件不超过 300 行
   - 测试覆盖率 > 80%
   - 无循环依赖

2. **文档完整性**
   - 所有公共 API 都有文档
   - 有完整的使用示例
   - 理论和实现一致

3. **实验可重现性**
   - 所有实验可通过配置文件重现
   - 结果自动保存和版本控制
   - 清晰的实验报告

## 🚀 下一步行动

1. 创建 `docs/archive/` 目录并开始归档过时文档
2. 创建新的目录结构
3. 开始拆分 `caac_ovr_model.py`

---

*本计划将根据实施过程中的发现进行调整和更新。* 