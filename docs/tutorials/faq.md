# 🚨 常见问题解答 (FAQ)

这里汇总了用户在使用 CAAC 项目时遇到的常见问题和解决方案。

## 🛠️ 安装和环境问题

### Q: 如何选择合适的Python版本？

**A:** 推荐使用 Python 3.8 或 3.9：

```bash
# 检查当前版本
python --version

# 使用conda安装指定版本
conda create -n caac python=3.8
conda activate caac
```

**为什么推荐3.8-3.9？**
- 所有依赖包的最佳兼容性
- PyTorch的稳定支持
- 性能和稳定性平衡

### Q: pip安装依赖时总是超时？

**A:** 使用国内镜像源：

```bash
# 临时使用清华源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch scikit-learn matplotlib

# 永久设置镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: conda安装很慢怎么办？

**A:** 配置国内镜像源：

```bash
# 添加清华镜像源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes

# 清理并重新安装
conda clean --all
conda install package_name
```

### Q: 出现"ModuleNotFoundError"错误？

**A:** 检查以下几点：

```bash
# 1. 确认在正确的环境中
conda activate base

# 2. 确认在项目根目录
cd /path/to/caac_project
python run_experiments.py --help

# 3. 重新安装依赖
pip install torch scikit-learn matplotlib pandas numpy seaborn

# 4. 检查Python路径
python -c "import sys; print('\n'.join(sys.path))"
```

### Q: GPU支持问题？

**A:** 分步骤检查：

```bash
# 1. 检查GPU驱动
nvidia-smi

# 2. 检查CUDA版本
nvcc --version

# 3. 安装对应PyTorch版本
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 4. 验证GPU可用性
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 🔬 实验运行问题

### Q: 实验运行很慢？

**A:** 优化策略：

```bash
# 1. 使用快速模式
python run_experiments.py --quick

# 2. 减少数据集 (交互式模式)
python run_experiments.py --interactive
# 选择1-2个小数据集

# 3. 降低训练参数
# 在交互式模式中设置 epochs=50
```

**性能对比参考：**
| 配置 | 快速模式 | 标准模式 | 自定义(小) |
|------|---------|----------|-----------|
| 时间 | 3-5分钟 | 15-25分钟 | 1-3分钟 |
| 数据集 | 4个小 | 8个全 | 用户选择 |

### Q: 内存不足错误？

**A:** 减少内存使用：

```python
# 在交互式模式中使用较小的配置
custom_config = {
    'datasets': ['iris', 'wine'],           # 只选择小数据集
    'representation_dim': 32,               # 降低维度
    'epochs': 50,                           # 减少训练轮数
    'batch_size': 16                        # 减小批量大小
}
```

### Q: 训练过程中断？

**A:** 检查和恢复：

```bash
# 1. 检查磁盘空间
df -h

# 2. 检查内存使用
free -h

# 3. 重新运行（结果会保存到新目录）
python run_experiments.py --quick

# 4. 查看历史结果
ls -la results/
```

### Q: 结果文件在哪里？

**A:** 结果保存位置：

```
caac_project/
└── results/
    ├── 20250601_143025_quick_robustness/      # 时间戳命名
    │   ├── 📊 robustness_curves.png          # 鲁棒性曲线
    │   ├── 📊 robustness_heatmap.png         # 热力图
    │   ├── 📈 detailed_results.csv           # 详细数据
    │   └── 📝 experiment_report.md           # 实验报告
    └── latest/                                # 最新结果链接
```

## 📊 结果理解问题

### Q: 如何理解鲁棒性得分？

**A:** 鲁棒性得分解释：

| 得分范围 | 质量评级 | 说明 |
|---------|---------|------|
| 0.95+ | 🥇 优秀 | 在噪声环境下表现非常稳定 |
| 0.90-0.95 | 🥈 良好 | 具有较强的鲁棒性 |
| 0.85-0.90 | 🥉 一般 | 基本的抗噪声能力 |
| <0.85 | ❌ 较差 | 对噪声敏感 |

**计算公式：**
```
鲁棒性得分 = Σ(准确率_噪声水平) / Σ(最大可能准确率)
```

### Q: 性能衰减如何计算？

**A:** 性能衰减指标：

```
性能衰减 = (基线准确率 - 最高噪声准确率) / 基线准确率 × 100%
```

**示例：**
- 基线准确率: 96.23%
- 20%噪声准确率: 94.60%  
- 性能衰减: (96.23 - 94.60) / 96.23 × 100% = 1.7%

### Q: 不确定性参数如何解读？

**A:** 柯西分布参数含义：

- **位置参数 (μ)**: 预测的中心位置
- **尺度参数 (σ)**: 不确定性程度
  - σ 越大 → 不确定性越高
  - σ 越小 → 预测越自信

**实用解读：**
```python
# 在结果文件中查看
import pandas as pd
results = pd.read_csv('results/latest/detailed_results.csv')

# 分析不确定性
uncertainty_avg = results.groupby('method')['uncertainty_avg'].mean()
print("平均不确定性:", uncertainty_avg)
```

## 🎮 高级使用问题

### Q: 如何自定义实验参数？

**A:** 使用实验管理器：

```python
from src.experiments.experiment_manager import ExperimentManager

# 创建管理器
manager = ExperimentManager()

# 自定义配置
config = {
    'datasets': ['iris', 'wine', 'breast_cancer'],
    'noise_levels': [0.0, 0.1, 0.2, 0.3],
    'representation_dim': 128,
    'epochs': 200,
    'learning_rate': 0.001
}

# 运行实验
result_dir = manager.run_quick_robustness_test(**config)
```

### Q: 如何添加新数据集？

**A:** 扩展数据集支持：

```python
# 在 src/data/ 中添加新的数据加载器
from sklearn.datasets import load_your_dataset

def load_custom_dataset():
    """加载自定义数据集"""
    X, y = load_your_dataset()
    return X, y, "custom_dataset"

# 在实验配置中使用
config = {
    'datasets': ['iris', 'wine', 'custom_dataset'],
    # ... 其他参数
}
```

### Q: 如何对比自己的方法？

**A:** 扩展方法对比：

```python
# 在 src/experiments/comparison_experiments.py 中添加
from your_module import YourClassifier

def add_custom_method(X_train, y_train, X_test, y_test):
    """添加自定义方法"""
    clf = YourClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)
```

## 🔧 开发和调试问题

### Q: 如何调试模型训练？

**A:** 使用调试模式：

```python
# 启用详细输出
import logging
logging.basicConfig(level=logging.DEBUG)

# 减少数据量进行快速调试
debug_config = {
    'datasets': ['iris'],  # 最小数据集
    'epochs': 10,          # 快速训练
    'representation_dim': 16,  # 小维度
}
```

### Q: 如何查看训练过程？

**A:** 监控训练进度：

```python
# 训练历史保存在结果目录中
import json
with open('results/latest/training_history.json', 'r') as f:
    history = json.load(f)

# 查看损失变化
losses = history['train_losses']
print(f"初始损失: {losses[0]:.4f}")
print(f"最终损失: {losses[-1]:.4f}")
```

### Q: 模型参数如何调优？

**A:** 参数调优建议：

| 参数 | 默认值 | 调优建议 |
|------|-------|----------|
| representation_dim | 128 | 数据复杂度×2-4 |
| epochs | 100 | 观察收敛情况调整 |
| learning_rate | 0.001 | 0.0001-0.01范围 |
| batch_size | 32 | 根据内存调整 |

## 🚀 性能优化问题

### Q: 如何加速实验？

**A:** 性能优化技巧：

```bash
# 1. 使用GPU (如果可用)
export CUDA_VISIBLE_DEVICES=0

# 2. 并行数据加载
export OMP_NUM_THREADS=4

# 3. 优化内存使用
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Q: 批量运行实验？

**A:** 自动化脚本：

```bash
#!/bin/bash
# run_all_experiments.sh

echo "运行快速测试..."
python run_experiments.py --quick

echo "运行标准测试..."
python run_experiments.py --standard

echo "运行方法对比..."
python run_experiments.py --comparison

echo "所有实验完成！"
```

## 📞 获取帮助

### 问题解决流程

1. **🔍 搜索FAQ** - 查看本页面相关问题
2. **📖 查看文档** - 阅读详细的使用指南
3. **🧪 最小化测试** - 用最简单的配置复现问题
4. **📝 收集信息** - 记录错误信息和环境配置
5. **💬 寻求帮助** - 在GitHub Issues中提问

### 提问模板

```markdown
**环境信息：**
- 操作系统: macOS 12.0
- Python版本: 3.8.10
- 依赖版本: pip list | grep torch

**问题描述：**
清晰描述遇到的问题

**复现步骤：**
1. 运行命令: python run_experiments.py --quick
2. 出现错误: ...

**预期结果：**
期望得到的结果

**实际结果：**
实际发生的情况

**错误信息：**
```python
完整的错误堆栈
```
```

---

💡 **提示**: 大多数问题都可以通过重新安装依赖或检查环境配置解决。如果问题持续存在，请在GitHub Issues中详细描述。

🎯 **快速解决**: 
- 环境问题 → 重新创建conda环境
- 运行问题 → 使用 `--quick` 模式测试
- 结果问题 → 查看 `results/latest/` 目录

> 💬 **还有问题？** [提交Issue](https://github.com/1587causalai/caac/issues) 或查看其他文档页面。 