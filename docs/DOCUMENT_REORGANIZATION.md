# 文档重组计划

## 📂 现有文档分析

### 需要保留和更新的核心文档
1. **motivation.md** - 核心理论文档，需要保留但可能需要轻微更新
2. **README.md** - 文档入口，需要更新以反映新结构

### 需要归档的文档
1. **unified_code_architecture.md** - 过于详细的代码实现，应该由代码本身和API文档替代
2. **unified_methods_mathematical_principles.md** - 包含太多实现细节，应该分离理论和实现
3. **architecture.md** - 可能与新的架构不符
4. **implementation.md** - 需要根据新的代码结构重写

### 需要更新的文档
1. **api.md** - 应该自动生成，而不是手动维护
2. **usage.md** - 需要更新以反映新的API
3. **experiments.md** - 应该由实验结果自动生成
4. **theory.md** - 可能需要与motivation.md合并

### Docsify相关文件（保留）
- **index.html** - Docsify入口
- **_sidebar.md** - 需要更新以反映新结构
- **_coverpage.md** - 可能需要更新

## 📋 新文档结构

```
docs/
├── archive/                     # 归档的旧文档
│   ├── unified_code_architecture.md
│   ├── unified_methods_mathematical_principles.md
│   └── old_implementation.md
├── theory/                      # 理论和数学原理
│   ├── motivation.md           # 核心动机（从根目录移动）
│   ├── mathematical_foundations.md  # 数学基础
│   └── design_principles.md    # 设计原则
├── implementation/              # 实现相关
│   ├── architecture.md         # 高层架构设计
│   └── api/                   # 自动生成的API文档
├── tutorials/                   # 使用教程
│   ├── quickstart.md          # 快速开始
│   ├── installation.md        # 安装指南
│   └── examples/              # 示例代码
├── experiments/                 # 实验相关
│   ├── benchmark_results.md   # 基准测试结果
│   └── robustness_analysis.md # 鲁棒性分析
├── index.html                  # Docsify配置
├── _sidebar.md                 # 侧边栏配置
├── _coverpage.md              # 封面页
└── README.md                   # 文档首页
```

## 🚀 执行步骤

### 第一步：归档旧文档
```bash
# 移动需要归档的文档
mv docs/unified_code_architecture.md docs/archive/
mv docs/unified_methods_mathematical_principles.md docs/archive/
mv docs/implementation.md docs/archive/old_implementation.md
```

### 第二步：创建新目录结构
```bash
mkdir -p docs/theory
mkdir -p docs/implementation/api
mkdir -p docs/tutorials/examples
mkdir -p docs/experiments
```

### 第三步：重组现有文档
```bash
# 移动理论文档
mv docs/motivation.md docs/theory/
mv docs/theory.md docs/theory/mathematical_foundations.md

# 更新使用文档
mv docs/usage.md docs/tutorials/quickstart.md
```

### 第四步：更新配置文件
- 更新 `_sidebar.md` 以反映新结构
- 更新 `README.md` 作为文档首页
- 更新 `_coverpage.md` 如果需要

## 📝 新文档创建计划

### 需要创建的新文档
1. **docs/theory/design_principles.md** - 解释设计决策
2. **docs/tutorials/installation.md** - 详细安装指南
3. **docs/implementation/architecture.md** - 新的架构概述
4. **docs/experiments/benchmark_results.md** - 标准化的实验结果

### 需要更新的现有文档
1. **docs/theory/motivation.md** - 检查并更新以确保与当前实现一致
2. **docs/tutorials/quickstart.md** - 基于新API更新示例

## 📊 成功标准

1. 文档结构清晰，易于导航
2. 理论文档与实现分离
3. 使用文档包含可运行的示例
4. API文档自动生成，减少维护负担
5. 实验结果有标准化的展示格式 