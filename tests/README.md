# Tests Directory

本目录包含 CAAC 项目的所有测试文件。

## 测试文件说明

### 核心测试文件

- **`test_classification_outliers.py`** - 分类数据outlier功能测试
  - 测试分类数据的outlier添加功能
  - 展示70%/15%/15%数据分割策略
  - 测试不同的标签噪声添加策略

- **`test_new_data_split.py`** - 新数据分割策略测试
  - 测试回归数据的outlier添加功能
  - 验证train+val含outliers，test保持干净的设计
  
- **`test_new_datasets.py`** - 数据集加载功能测试
  - 测试扩展的数据集加载功能
  - 验证各种规模数据集的加载
  
- **`test_crammer_singer.py`** - Crammer-Singer模型测试
  - 测试Crammer-Singer MLP模型功能

### 运行测试

#### 运行单个测试
```bash
cd tests
python test_new_datasets.py
```

#### 运行所有测试
```bash
cd tests
python run_all_tests.py
```

#### 只运行快速测试
```bash
cd tests
python run_all_tests.py --quick
```

## 测试分类

### 快速测试 (--quick)
- `test_new_datasets.py` - 数据集加载测试
- `test_crammer_singer.py` - Crammer-Singer模型测试

这些测试运行较快，适合开发过程中的快速验证。

### 完整测试
包含所有测试文件，其中一些测试可能需要较长时间运行，包括：
- 完整的outlier添加和可视化测试
- 大规模数据处理测试

## 目录结构

```
tests/
├── README.md                           # 本文档
├── __init__.py                        # Python包初始化
├── run_all_tests.py                   # 测试运行脚本
├── test_classification_outliers.py   # 分类outlier测试
├── test_new_data_split.py            # 数据分割测试
├── test_new_datasets.py              # 数据集加载测试
└── test_crammer_singer.py            # Crammer-Singer测试
```

## 注意事项

1. **导入路径**: 所有测试文件已调整导入路径以适应新的目录结构
2. **依赖关系**: 确保在运行测试前已经安装了所有必要的依赖
3. **数据下载**: 某些测试可能需要下载数据集，首次运行可能较慢
4. **图像输出**: 部分测试会生成可视化图像，请确保在支持图形界面的环境中运行

## 故障排除

如果遇到导入错误，请确保：
1. 在项目根目录下运行测试
2. Python路径设置正确
3. 所有依赖包已安装

如果需要调试特定测试，可以直接运行对应的测试文件并查看详细输出。 