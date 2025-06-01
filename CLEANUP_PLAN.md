# 🧹 CAAC Project 清理计划 - 下一阶段

## 📋 当前状态

✅ **已完成**:
- 模块化重构：实验脚本整合到 `src/experiments/`
- 代码分离：按功能创建专门模块
- 历史保留：原脚本移到 `legacy/` 目录
- 文档更新：README 反映新架构

## 🎯 下一步清理优先级

### **优先级1: 依赖和导入优化** ⭐⭐⭐

#### 问题现状
- `experiment_manager.py` 中的导入路径指向新模块，但可能存在循环依赖
- 一些工具函数可能在多个模块中重复

#### 清理方案
```bash
# 1. 检查和修复导入问题
src/experiments/
├── __init__.py              # 统一导入接口
├── experiment_manager.py    # 主管理器
├── robustness_experiments.py
├── comparison_experiments.py  
├── outlier_experiments.py
└── utils.py                 # 🆕 共享工具函数
```

#### 具体行动
- [ ] 创建 `src/experiments/utils.py` 统一工具函数
- [ ] 重构各实验模块，移除重复代码
- [ ] 完善 `__init__.py` 提供清晰的模块接口
- [ ] 测试所有导入路径正常工作

### **优先级2: 配置管理系统** ⭐⭐⭐

#### 目标
统一所有实验的配置管理，避免硬编码参数

#### 清理方案
```bash
# 2. 创建配置管理
src/
├── config/
│   ├── __init__.py
│   ├── default_config.py    # 🆕 默认配置
│   ├── experiment_configs.py # 🆕 实验特定配置
│   └── dataset_configs.py   # 🆕 数据集配置
```

#### 具体行动
- [ ] 抽取所有硬编码配置到配置文件
- [ ] 创建配置验证和加载机制
- [ ] 支持YAML/JSON配置文件
- [ ] 环境变量配置覆盖

### **优先级3: 测试框架完善** ⭐⭐

#### 目标
确保重构后的代码质量和稳定性

#### 清理方案
```bash
# 3. 完善测试
tests/
├── __init__.py
├── test_experiments/        # 🆕 实验测试
│   ├── test_robustness.py
│   ├── test_comparison.py
│   └── test_outlier.py
├── test_models/            # 🆕 模型测试
│   └── test_caac_model.py
└── test_integration/       # 🆕 集成测试
    └── test_full_pipeline.py
```

#### 具体行动
- [ ] 为每个实验模块添加单元测试
- [ ] 创建集成测试验证完整流程
- [ ] 添加性能基准测试
- [ ] 建立CI/CD流程

### **优先级4: 文档和示例** ⭐⭐

#### 目标
为新架构提供完整的文档和示例

#### 清理方案
```bash
# 4. 完善文档
docs/
├── theory/motivation.md    # 保持不变
├── api/                    # 🆕 API文档
│   ├── experiments.md      # 实验模块API
│   └── models.md          # 模型API
├── examples/              # 🆕 使用示例
│   ├── basic_usage.py     # 基础使用
│   ├── custom_experiment.py # 自定义实验
│   └── advanced_config.py # 高级配置
└── development/           # 🆕 开发指南
    └── contributing.md    # 贡献指南
```

### **优先级5: 性能优化** ⭐

#### 目标
优化代码性能和资源使用

#### 清理方案
- [ ] 分析实验流程性能瓶颈
- [ ] 优化数据加载和预处理
- [ ] 实现实验结果缓存
- [ ] 支持并行实验执行

## 📅 实施时间表

### 第1周: 依赖优化
- 创建工具函数模块
- 修复导入问题
- 测试模块功能

### 第2周: 配置管理
- 设计配置系统
- 重构硬编码配置
- 支持配置文件

### 第3周: 测试框架
- 添加基础测试
- 集成测试
- CI/CD设置

### 第4周: 文档完善
- API文档
- 使用示例
- 开发指南

## 🧪 清理验证标准

每个阶段完成后的验证：

### 功能验证
```bash
# 所有实验类型正常运行
python run_experiments.py --quick    ✅
python run_experiments.py --standard ✅ 
python run_experiments.py --comparison ✅
python run_experiments.py --interactive ✅
```

### 性能验证
- [ ] 快速实验 < 5分钟
- [ ] 标准实验 < 25分钟
- [ ] 内存使用合理
- [ ] 无内存泄漏

### 质量验证
- [ ] 所有测试通过
- [ ] 代码覆盖率 > 80%
- [ ] 文档完整性检查
- [ ] 性能基准达标

## 🚨 风险控制

### 备份策略
- `legacy/` 目录保留原始功能
- 每个清理阶段创建git分支
- 重要变更前创建标签

### 回滚机制
- 如果新模块出现问题，可快速回到 `legacy/` 版本
- 保持向后兼容的接口
- 渐进式迁移，避免破坏性变更

## 🎯 最终目标

清理完成后的项目特征：
- ✅ **模块化**: 清晰的代码组织结构
- ✅ **可配置**: 灵活的配置管理系统  
- ✅ **可测试**: 完整的测试覆盖
- ✅ **可维护**: 详细的文档和示例
- ✅ **高性能**: 优化的执行效率 