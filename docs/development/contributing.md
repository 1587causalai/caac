# 🤝 CAAC项目贡献指南

欢迎为CAAC项目做出贡献！本指南将帮助您了解如何参与项目开发。

## 📋 目录

- [项目概览](#项目概览)
- [开发环境设置](#开发环境设置)
- [代码规范](#代码规范)
- [贡献流程](#贡献流程)
- [测试指南](#测试指南)
- [文档维护](#文档维护)
- [问题报告](#问题报告)
- [功能请求](#功能请求)

## 🎯 项目概览

### 核心架构

CAAC项目采用模块化设计，主要组件包括：

```
caac_project/
├── src/
│   ├── models/           # 核心算法实现
│   ├── experiments/      # 实验管理模块
│   ├── data/            # 数据处理
│   ├── evaluators/      # 评估器
│   └── utils/           # 工具函数
├── docs/                # 项目文档
├── tests/               # 测试代码
├── examples/            # 使用示例
└── results/             # 实验结果
```

### 技术栈

- **机器学习**: PyTorch, scikit-learn
- **数据处理**: NumPy, Pandas
- **可视化**: Matplotlib, Seaborn
- **配置管理**: JSON, YAML
- **测试**: pytest (推荐)
- **代码质量**: flake8, black (推荐)

## 🛠️ 开发环境设置

### 1. 克隆仓库

```bash
git clone <repository-url>
cd caac_project
```

### 2. 环境准备

推荐使用conda环境：

```bash
# 创建虚拟环境
conda create -n caac-dev python=3.8
conda activate caac-dev

# 安装依赖
pip install torch scikit-learn matplotlib pandas numpy seaborn
pip install pytest flake8 black jupyter  # 开发工具
```

### 3. 验证安装

```bash
# 运行快速测试
python run_experiments.py --quick

# 运行示例
python examples/basic_usage.py
```

## 📏 代码规范

### Python代码风格

遵循PEP 8规范，推荐使用以下工具：

```bash
# 代码格式化
black src/ tests/ examples/

# 代码检查
flake8 src/ tests/ examples/
```

### 命名约定

- **类名**: `PascalCase` (如 `CAACOvRModel`)
- **函数名**: `snake_case` (如 `run_experiment`)
- **变量名**: `snake_case` (如 `noise_level`)
- **常量**: `UPPER_SNAKE_CASE` (如 `DEFAULT_EPOCHS`)

### 文档字符串

使用Google风格的docstring：

```python
def example_function(param1: int, param2: str = "default") -> bool:
    """
    简短的函数描述
    
    Args:
        param1: 第一个参数说明
        param2: 第二个参数说明，有默认值
        
    Returns:
        bool: 返回值说明
        
    Raises:
        ValueError: 异常情况说明
        
    Example:
        >>> result = example_function(42, "test")
        >>> print(result)
        True
    """
    return True
```

### 注释规范

```python
# 使用中文注释解释复杂逻辑
def complex_algorithm():
    # 第一步：初始化参数
    params = initialize_params()
    
    # 第二步：迭代优化
    for epoch in range(epochs):
        # 计算梯度
        gradients = compute_gradients()
        
        # 更新参数
        params = update_params(params, gradients)
    
    return params
```

## 🔄 贡献流程

### 1. 创建Issue

在开始开发前，请先创建Issue：

- **Bug报告**: 使用Bug报告模板
- **功能请求**: 描述新功能的用途和期望
- **改进建议**: 说明现有功能的改进点

### 2. 分支管理

```bash
# 创建功能分支
git checkout -b feature/your-feature-name

# 或创建修复分支
git checkout -b fix/bug-description
```

### 3. 开发流程

1. **编写代码**：遵循代码规范
2. **添加测试**：确保新功能有测试覆盖
3. **更新文档**：同步更新相关文档
4. **运行测试**：确保所有测试通过

```bash
# 运行测试
python -m pytest tests/

# 检查代码质量
flake8 src/
black --check src/
```

### 4. 提交规范

使用清晰的提交信息：

```bash
# 功能提交
git commit -m "feat: add uncertainty quantification for predictions"

# 修复提交
git commit -m "fix: resolve memory leak in batch processing"

# 文档提交
git commit -m "docs: update API documentation for ExperimentManager"

# 重构提交
git commit -m "refactor: optimize data loading pipeline"
```

### 5. Pull Request

创建PR时请：

- 使用描述性标题
- 详细说明改动内容
- 关联相关Issue
- 添加必要的截图或示例

## 🧪 测试指南

### 测试结构

```
tests/
├── unit/                # 单元测试
│   ├── test_models.py
│   ├── test_experiments.py
│   └── test_utils.py
├── integration/         # 集成测试
│   ├── test_full_pipeline.py
│   └── test_experiment_manager.py
└── fixtures/           # 测试数据和工具
    ├── data/
    └── conftest.py
```

### 编写测试

#### 单元测试示例

```python
import pytest
import numpy as np
from src.models.caac_ovr_model import CAACOvRModel

class TestCAACOvRModel:
    """CAAC OvR模型单元测试"""
    
    def test_model_initialization(self):
        """测试模型初始化"""
        model = CAACOvRModel(input_dim=10, n_classes=3)
        assert model.input_dim == 10
        assert model.n_classes == 3
        
    def test_model_fit(self):
        """测试模型训练"""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)
        
        model = CAACOvRModel(input_dim=10, n_classes=3, epochs=5)
        history = model.fit(X, y)
        
        assert 'train_losses' in history
        assert len(history['train_losses']) > 0
        
    def test_model_predict(self):
        """测试模型预测"""
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 3, 100)
        X_test = np.random.randn(20, 10)
        
        model = CAACOvRModel(input_dim=10, n_classes=3, epochs=5)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        assert predictions.shape == (20,)
        assert probabilities.shape == (20, 3)
        assert np.all(predictions >= 0) and np.all(predictions < 3)
```

#### 集成测试示例

```python
import pytest
from src.experiments.experiment_manager import ExperimentManager

class TestExperimentIntegration:
    """实验管理器集成测试"""
    
    def test_quick_robustness_experiment(self):
        """测试快速鲁棒性实验完整流程"""
        manager = ExperimentManager(base_results_dir="tests/temp_results")
        
        # 运行实验
        result_dir = manager.run_quick_robustness_test(
            datasets=['iris'],
            noise_levels=[0.0, 0.1],
            epochs=5
        )
        
        # 验证结果
        assert result_dir is not None
        assert os.path.exists(result_dir)
        
        # 清理
        import shutil
        shutil.rmtree("tests/temp_results", ignore_errors=True)
```

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试文件
python -m pytest tests/unit/test_models.py

# 运行特定测试类
python -m pytest tests/unit/test_models.py::TestCAACOvRModel

# 运行特定测试方法
python -m pytest tests/unit/test_models.py::TestCAACOvRModel::test_model_initialization

# 显示覆盖率
python -m pytest tests/ --cov=src --cov-report=html
```

## 📚 文档维护

### 文档结构

- **API文档**: `docs/api/` - 详细的API参考
- **理论文档**: `docs/theory/` - 算法原理和数学基础
- **使用示例**: `examples/` - 实际使用示例
- **开发文档**: `docs/development/` - 开发和贡献指南

### 文档更新原则

1. **同步更新**: 代码变更时同步更新文档
2. **示例验证**: 确保文档中的示例可以正常运行
3. **清晰简洁**: 使用清晰的语言和适当的示例
4. **多语言**: 核心文档提供中英文版本

### Markdown规范

```markdown
# 一级标题

## 二级标题

### 三级标题

- 使用bullet points列表
- 保持一致的格式

1. 使用数字列表
2. 表示有序步骤

**粗体文本**用于强调
*斜体文本*用于术语

`代码片段`使用反引号

```python
# 代码块使用三反引号
def example():
    return "Hello, World!"
```

> 引用文本用于重要提示
```

## 🐛 问题报告

### Bug报告模板

```markdown
## Bug描述
简要描述遇到的问题

## 复现步骤
1. 执行步骤1
2. 执行步骤2
3. 观察到的错误

## 期望行为
描述期望的正确行为

## 实际行为
描述实际观察到的行为

## 环境信息
- Python版本：
- 操作系统：
- 相关依赖版本：

## 错误信息
```
粘贴错误堆栈信息
```

## 额外信息
其他可能有用的信息
```

### 调试技巧

1. **开启详细输出**：
```python
# 使用详细模式运行
python run_experiments.py --quick --verbose

# 在代码中添加调试信息
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **使用断点调试**：
```python
import pdb; pdb.set_trace()  # 设置断点
```

3. **检查中间结果**：
```python
# 保存中间结果用于调试
np.save('debug_data.npy', intermediate_result)
```

## 💡 功能请求

### 功能请求模板

```markdown
## 功能描述
简要描述请求的新功能

## 使用场景
描述什么情况下需要这个功能

## 期望接口
描述希望的API接口设计

## 实现建议
（可选）如何实现这个功能的建议

## 替代方案
（可选）其他可能的解决方案
```

### 功能开发流程

1. **需求分析**: 明确功能需求和使用场景
2. **接口设计**: 设计用户友好的API接口
3. **实现规划**: 制定实现计划和时间表
4. **原型开发**: 创建功能原型
5. **测试验证**: 编写测试确保质量
6. **文档完善**: 更新相关文档

## 🏆 最佳实践

### 代码质量

1. **保持简洁**: 避免不必要的复杂性
2. **单一职责**: 每个函数只做一件事
3. **错误处理**: 优雅处理异常情况
4. **性能考虑**: 注意内存和计算效率

### 团队协作

1. **及时沟通**: 遇到问题及时讨论
2. **代码审查**: 认真参与代码审查
3. **知识分享**: 主动分享经验和最佳实践
4. **持续学习**: 保持对新技术的学习

### 版本管理

1. **小步提交**: 频繁提交小的改动
2. **清晰历史**: 保持清晰的提交历史
3. **分支策略**: 合理使用分支管理
4. **标签管理**: 为重要版本打标签

## 📞 获取帮助

如果您在贡献过程中遇到问题：

1. **查看文档**: 首先查看相关文档
2. **搜索Issue**: 查看是否有类似问题
3. **创建Issue**: 描述具体问题寻求帮助
4. **联系维护者**: 直接联系项目维护者

## 🙏 致谢

感谢所有为CAAC项目做出贡献的开发者！您的贡献让这个项目变得更好。

---

**最后更新**: 2024年12月
**文档版本**: v1.0