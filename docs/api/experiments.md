# 🔬 实验模块 API 文档

本文档详细描述CAAC项目中实验模块的API接口和使用方法。

## 📋 概览

实验模块位于 `src/experiments/` 目录，提供统一的实验管理和执行接口：

```python
from src.experiments.experiment_manager import ExperimentManager
from src.experiments import robustness_experiments, comparison_experiments, outlier_experiments
```

## 🎯 核心类：ExperimentManager

### 类定义

```python
class ExperimentManager:
    """
    CAAC项目的统一实验管理器
    
    提供以下功能：
    - 鲁棒性测试 (快速/标准)
    - 方法对比实验
    - 离群值鲁棒性测试
    - 自定义实验配置
    """
```

### 初始化

```python
def __init__(self, 
             base_results_dir: str = "results",
             config_file: Optional[str] = None):
    """
    初始化实验管理器
    
    Args:
        base_results_dir: 结果保存的基础目录
        config_file: 可选的配置文件路径
    """
```

**示例**:
```python
# 使用默认配置
manager = ExperimentManager()

# 指定结果目录
manager = ExperimentManager(base_results_dir="my_results")

# 使用配置文件
manager = ExperimentManager(config_file="config.json")
```

### 主要方法

#### 🚀 快速鲁棒性测试

```python
def run_quick_robustness_test(self, **kwargs) -> str:
    """
    运行快速鲁棒性测试 (3-5分钟)
    
    Args:
        **kwargs: 覆盖默认参数
            - noise_levels: List[float] = [0.0, 0.05, 0.10, 0.15, 0.20]
            - representation_dim: int = 128
            - epochs: int = 100
            - datasets: List[str] = ['iris', 'wine', 'breast_cancer', 'optical_digits']
            
    Returns:
        str: 实验结果目录路径
        
    Raises:
        ImportError: 如果无法导入鲁棒性实验模块
    """
```

**示例**:
```python
# 使用默认配置
result_dir = manager.run_quick_robustness_test()

# 自定义配置
result_dir = manager.run_quick_robustness_test(
    noise_levels=[0.0, 0.1, 0.2],
    epochs=50,
    datasets=['iris', 'wine']
)
```

#### 🔬 标准鲁棒性测试

```python
def run_standard_robustness_test(self, **kwargs) -> str:
    """
    运行标准鲁棒性测试 (15-25分钟)
    
    Args:
        **kwargs: 覆盖默认参数
            - noise_levels: List[float] = [0.0, 0.05, 0.10, 0.15, 0.20]
            - representation_dim: int = 128
            - epochs: int = 150
            - datasets: List[str] = 8个数据集的完整列表
            
    Returns:
        str: 实验结果目录路径
    """
```

#### 📊 基础方法对比

```python
def run_basic_comparison(self, **kwargs) -> str:
    """
    运行基础方法对比实验
    
    Args:
        **kwargs: 覆盖默认参数
            - datasets: List[str] = ['iris', 'wine', 'breast_cancer', 'digits']
            - representation_dim: int = 64
            - epochs: int = 100
            
    Returns:
        str: 实验结果目录路径
    """
```

#### 🎯 离群值鲁棒性测试

```python
def run_outlier_robustness_test(self, **kwargs) -> str:
    """
    运行离群值鲁棒性实验
    
    Args:
        **kwargs: 实验参数
            - outlier_ratios: List[float] - 离群值比例
            - datasets: List[str] - 测试数据集
            - methods: List[str] - 对比方法
            
    Returns:
        str: 实验结果目录路径
    """
```

#### ⚙️ 自定义实验

```python
def run_custom_experiment(self, 
                        experiment_type: str,
                        config: Dict,
                        save_name: Optional[str] = None) -> str:
    """
    运行自定义配置的实验
    
    Args:
        experiment_type: 实验类型 ('robustness', 'comparison', 'outlier_robustness')
        config: 实验配置字典
        save_name: 自定义保存名称
        
    Returns:
        str: 实验结果目录路径
        
    Raises:
        ValueError: 如果实验类型不支持
    """
```

**示例**:
```python
# 自定义鲁棒性实验
custom_config = {
    'noise_levels': [0.0, 0.05, 0.15],
    'datasets': ['iris', 'digits'],
    'epochs': 200
}
result_dir = manager.run_custom_experiment(
    'robustness', 
    custom_config, 
    'high_epochs_test'
)
```

### 工具方法

#### 📋 获取可用实验列表

```python
def list_available_experiments(self) -> List[str]:
    """
    返回所有可用的实验类型
    
    Returns:
        List[str]: ['quick_robustness', 'standard_robustness', 
                   'basic_comparison', 'outlier_robustness', 'custom']
    """
```

#### ⚙️ 获取默认配置

```python
def get_experiment_config(self, experiment_type: str) -> Dict:
    """
    获取指定实验类型的默认配置
    
    Args:
        experiment_type: 实验类型名称
        
    Returns:
        Dict: 默认配置字典
    """
```

#### 📊 创建实验总结

```python
def create_experiment_summary(self, results_dir: str) -> Dict:
    """
    创建实验结果总结
    
    Args:
        results_dir: 实验结果目录
        
    Returns:
        Dict: 包含文件列表、时间戳等信息的总结
    """
```

## 🔬 专门实验模块

### robustness_experiments.py

```python
def run_quick_robustness_test(**config) -> str:
    """快速鲁棒性测试实现"""

def run_standard_robustness_test(**config) -> str:
    """标准鲁棒性测试实现"""
```

### comparison_experiments.py

```python
def run_comparison_experiments(**config) -> str:
    """方法对比实验实现"""
```

### outlier_experiments.py

```python
def run_outlier_robustness_experiments(**config) -> str:
    """离群值鲁棒性实验实现"""
```

## 📊 默认配置

### 快速鲁棒性测试
```python
{
    'noise_levels': [0.0, 0.05, 0.10, 0.15, 0.20],
    'representation_dim': 128,
    'epochs': 100,
    'datasets': ['iris', 'wine', 'breast_cancer', 'optical_digits']
}
```

### 标准鲁棒性测试
```python
{
    'noise_levels': [0.0, 0.05, 0.10, 0.15, 0.20],
    'representation_dim': 128,
    'epochs': 150,
    'datasets': ['iris', 'wine', 'breast_cancer', 'optical_digits', 
               'digits', 'synthetic_imbalanced', 'covertype', 'letter']
}
```

### 基础方法对比
```python
{
    'datasets': ['iris', 'wine', 'breast_cancer', 'digits'],
    'representation_dim': 64,
    'epochs': 100
}
```

## 🚨 异常处理

### 常见异常

- **ImportError**: 实验模块导入失败
- **ValueError**: 无效的实验类型或配置参数
- **FileNotFoundError**: 配置文件不存在
- **RuntimeError**: 实验执行过程中的错误

### 错误处理示例

```python
try:
    result_dir = manager.run_quick_robustness_test()
    print(f"实验完成，结果保存在: {result_dir}")
except ImportError as e:
    print(f"模块导入错误: {e}")
    print("请检查实验模块是否正确安装")
except Exception as e:
    print(f"实验执行错误: {e}")
    print("请检查配置和环境设置")
```

## 📈 返回值说明

所有实验方法都返回字符串类型的结果目录路径，该目录包含：

- **可视化文件**: `*.png` - 图表和曲线
- **数据文件**: `*.csv` - 详细的实验数据
- **报告文件**: `*.md` - 格式化的实验报告
- **配置文件**: `*_config.json` - 使用的实验配置

## 🔄 版本兼容性

- **当前版本**: 支持Python 3.7+
- **依赖要求**: PyTorch, scikit-learn, matplotlib, pandas, numpy, seaborn
- **向后兼容**: 与legacy模块完全兼容

## 📞 技术支持

如需获得更多帮助：
1. 查看 `QUICK_START.md` 获取快速上手指南
2. 参考 `examples/` 目录中的使用示例
3. 运行 `python run_experiments.py --help` 查看命令行帮助 