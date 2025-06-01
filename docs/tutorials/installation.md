# 安装指南

## 📋 系统要求

### 操作系统
- macOS 10.14+ 
- Ubuntu 18.04+
- Windows 10+

### Python 版本
- Python 3.7 或更高版本
- 推荐使用 Python 3.8 或 3.9

### 硬件要求
- 最小内存：4GB RAM
- 推荐内存：8GB+ RAM
- GPU：可选，支持CUDA 10.2+的NVIDIA GPU可显著加速训练

## 🚀 快速安装

### 方法一：使用 Conda（推荐）

```bash
# 1. 克隆项目
git clone <repository-url>
cd caac_project

# 2. 激活base环境
conda activate base

# 3. 安装依赖
pip install torch scikit-learn matplotlib pandas numpy seaborn

# 4. 安装项目
pip install -e .

# 5. 验证安装
python -c "from src.models.caac_ovr_model import CAACOvRModel; print('Installation successful!')"
```

### 方法二：使用虚拟环境

```bash
# 1. 克隆项目
git clone <repository-url>
cd caac_project

# 2. 创建虚拟环境
python -m venv caac_env

# 3. 激活虚拟环境
# macOS/Linux:
source caac_env/bin/activate
# Windows:
caac_env\Scripts\activate

# 4. 升级pip
pip install --upgrade pip

# 5. 安装依赖
pip install -r requirements.txt

# 6. 安装项目
pip install -e .
```

## 📦 详细依赖说明

### 核心依赖

| 包名 | 版本要求 | 用途 |
|------|----------|------|
| torch | >=1.9.0 | 深度学习框架 |
| scikit-learn | >=1.0.0 | 机器学习工具和数据集 |
| numpy | >=1.19.0 | 数值计算 |
| pandas | >=1.3.0 | 数据处理 |

### 可视化依赖

| 包名 | 版本要求 | 用途 |
|------|----------|------|
| matplotlib | >=3.3.0 | 基础绘图 |
| seaborn | >=0.11.0 | 统计可视化 |
| plotly | >=5.0.0 | 交互式可视化（可选）|

### 开发依赖

| 包名 | 版本要求 | 用途 |
|------|----------|------|
| pytest | >=6.0.0 | 单元测试 |
| pytest-cov | >=2.10.0 | 测试覆盖率 |
| black | >=21.0.0 | 代码格式化 |
| flake8 | >=3.8.0 | 代码检查 |

## 🔧 创建 requirements.txt

如果项目中没有 `requirements.txt` 文件，可以创建一个：

```txt
# requirements.txt
torch>=1.9.0
scikit-learn>=1.0.0
numpy>=1.19.0
pandas>=1.3.0
matplotlib>=3.3.0
seaborn>=0.11.0

# 可选依赖
plotly>=5.0.0

# 开发依赖
pytest>=6.0.0
pytest-cov>=2.10.0
black>=21.0.0
flake8>=3.8.0
```

## 🎮 GPU 支持（可选）

如果您有NVIDIA GPU并希望加速训练：

### 检查CUDA版本
```bash
nvidia-smi
```

### 安装对应的PyTorch版本
```bash
# CUDA 11.1
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 11.3
pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# 或访问 https://pytorch.org 获取最新安装命令
```

### 验证GPU可用性
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
```

## ✅ 安装验证

### 基础功能测试
```bash
# 运行快速测试
python run_quick_robustness_test.py
```

### 完整测试套件
```bash
# 如果安装了pytest
pytest tests/ -v

# 或运行特定测试
python -m pytest tests/test_models.py -v
```

### 数据集下载测试
```bash
# 测试所有数据集是否可以正常加载
python test_new_datasets.py
```

## 🐛 常见问题

### 问题1：torch安装失败
```bash
# 解决方案：使用conda安装
conda install pytorch torchvision torchaudio -c pytorch
```

### 问题2：scikit-learn版本冲突
```bash
# 解决方案：强制更新
pip install --upgrade scikit-learn
```

### 问题3：matplotlib中文显示问题
```bash
# 安装中文字体支持
pip install fonttools
# 然后重启Python环境
```

### 问题4：导入错误
```bash
# 确保项目已正确安装
pip install -e .

# 或者将项目路径添加到PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/caac_project"
```

## 🔄 开发环境设置

如果您计划开发或修改代码：

### 1. 安装开发依赖
```bash
pip install -r requirements-dev.txt
```

### 2. 设置pre-commit hooks
```bash
pip install pre-commit
pre-commit install
```

### 3. 代码格式化设置
```bash
# 格式化代码
black src/ tests/

# 检查代码风格
flake8 src/ tests/
```

### 4. 测试环境
```bash
# 运行所有测试
pytest tests/ --cov=src --cov-report=html

# 查看覆盖率报告
open htmlcov/index.html
```

## 📱 Docker支持（高级）

如果您偏好使用Docker：

### Dockerfile 示例
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python", "run_quick_robustness_test.py"]
```

### 构建和运行
```bash
# 构建镜像
docker build -t caac-project .

# 运行容器
docker run --rm caac-project

# 交互式运行
docker run -it --rm -v $(pwd):/app caac-project bash
```

## 🆘 获取帮助

如果遇到安装问题：

1. **检查系统兼容性**：确认Python版本和操作系统
2. **查看错误日志**：完整的错误信息有助于诊断
3. **尝试干净安装**：在新的虚拟环境中重新安装
4. **查看FAQ**：常见问题可能已有解决方案
5. **提交Issue**：在GitHub上报告问题

## 📝 下一步

安装完成后，建议：

1. 阅读 [快速开始](quickstart.md) 教程
2. 运行示例实验验证安装
3. 查看 [项目动机](../theory/motivation.md) 了解理论背景

---

**提示**：建议使用conda管理Python环境，可以避免大多数依赖冲突问题。 