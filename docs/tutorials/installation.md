# 🛠️ 安装指南

本指南详细介绍如何在不同环境中安装和配置 CAAC 项目。

## 📋 系统要求

### 最低要求
- **操作系统**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **Python**: 3.7 或更高版本
- **内存**: 4GB RAM (推荐 8GB+)
- **存储**: 2GB 可用空间

### 推荐配置
- **Python**: 3.8+ 
- **内存**: 16GB RAM
- **GPU**: NVIDIA GPU with CUDA support (可选，加速训练)

## 🎯 安装方式选择

根据您的使用场景选择合适的安装方式：

| 使用场景 | 推荐方式 | 时间 | 难度 |
|---------|---------|------|------|
| 快速体验 | [Conda环境](#conda环境安装-推荐) | 5分钟 | ⭐ |
| 开发使用 | [源码安装](#源码安装) | 10分钟 | ⭐⭐ |
| 生产部署 | [Docker安装](#docker安装) | 15分钟 | ⭐⭐⭐ |

## 🐍 Conda环境安装 (推荐)

### 第1步: 安装Conda

如果您还没有安装Conda：

```bash
# macOS/Linux
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Windows (PowerShell)
Invoke-WebRequest -Uri https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -OutFile miniconda.exe
./miniconda.exe
```

### 第2步: 创建虚拟环境

```bash
# 创建新环境 (如果没有base环境)
conda create -n caac python=3.8
conda activate caac

# 或使用推荐的base环境
conda activate base
```

### 第3步: 安装依赖

```bash
# 安装核心依赖
pip install torch scikit-learn matplotlib pandas numpy seaborn

# 验证安装
python -c "import torch, sklearn, matplotlib; print('✅ 安装成功!')"
```

### 第4步: 获取项目

```bash
# 克隆项目
git clone https://github.com/1587causalai/caac.git
cd caac_project

# 测试安装
python run_experiments.py --help
```

## 📦 源码安装

适合需要修改代码或贡献开发的用户。

### 第1步: 克隆仓库

```bash
git clone https://github.com/1587causalai/caac.git
cd caac_project
```

### 第2步: 创建虚拟环境

```bash
# 使用venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 或使用conda
conda create -n caac python=3.8
conda activate caac
```

### 第3步: 安装依赖

```bash
# 安装核心依赖
pip install -r requirements.txt

# 开发模式安装 (可编辑)
pip install -e .
```

### 第4步: 运行测试

```bash
# 验证安装
python -m pytest tests/
python run_experiments.py --quick
```

## 🐳 Docker安装

适合生产环境或避免环境冲突。

### 第1步: 安装Docker

参考 [Docker官方文档](https://docs.docker.com/get-docker/) 安装Docker。

### 第2步: 构建镜像

```bash
# 克隆项目
git clone https://github.com/1587causalai/caac.git
cd caac_project

# 构建Docker镜像
docker build -t caac:latest .
```

### 第3步: 运行容器

```bash
# 运行容器
docker run -it --rm -v $(pwd)/results:/app/results caac:latest

# 在容器中运行实验
python run_experiments.py --quick
```

## ⚡ GPU支持安装

如果您有NVIDIA GPU并希望加速训练：

### 第1步: 安装CUDA

```bash
# 检查CUDA版本
nvidia-smi

# 安装CUDA Toolkit (以CUDA 11.8为例)
# 访问 https://developer.nvidia.com/cuda-toolkit 下载
```

### 第2步: 安装PyTorch GPU版本

```bash
# 卸载CPU版本
pip uninstall torch

# 安装GPU版本 (根据CUDA版本选择)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 第3步: 验证GPU支持

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU devices: {torch.cuda.device_count()}")
```

## 🔧 依赖详解

### 核心依赖

```
torch>=1.9.0          # 深度学习框架
scikit-learn>=1.0.0   # 机器学习工具
matplotlib>=3.3.0     # 可视化
pandas>=1.3.0         # 数据处理
numpy>=1.20.0         # 数值计算
seaborn>=0.11.0       # 统计可视化
```

### 可选依赖

```
jupyter>=1.0.0        # 交互式开发
pytest>=6.0.0         # 测试框架
sphinx>=4.0.0         # 文档生成
black>=21.0.0         # 代码格式化
```

### 安装可选依赖

```bash
# 开发工具
pip install jupyter pytest sphinx black

# 或使用extras安装
pip install -e ".[dev]"
```

## 🚨 常见问题解决

### Q: pip安装超时？

```bash
# 使用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch scikit-learn matplotlib pandas numpy seaborn
```

### Q: torch安装失败？

```bash
# 清除缓存重试
pip cache purge
pip install torch --no-cache-dir
```

### Q: 内存不足？

```bash
# 使用CPU版本
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Q: 权限错误？

```bash
# 使用用户安装
pip install --user torch scikit-learn matplotlib pandas numpy seaborn
```

### Q: conda安装慢？

```bash
# 添加国内源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
```

## ✅ 验证安装

运行以下脚本验证安装是否成功：

```python
# test_installation.py
import sys
import importlib

def test_package(package_name):
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name}")
        return False

# 测试核心依赖
packages = ['torch', 'sklearn', 'matplotlib', 'pandas', 'numpy', 'seaborn']
all_ok = all(test_package(pkg) for pkg in packages)

if all_ok:
    print("\n🎉 所有依赖安装成功！")
    print("运行: python run_experiments.py --quick")
else:
    print("\n❌ 部分依赖缺失，请重新安装")
```

## 🚀 下一步

安装完成后，您可以：

1. **🎯 快速体验**: 运行 `python run_experiments.py --quick`
2. **📖 学习使用**: 查看 [快速开始](quickstart.md)
3. **🔬 深入了解**: 阅读 [理论基础](../theory/motivation.md)
4. **🛠️ 自定义实验**: 使用 [实验配置](experiment_config.md)

## 💡 环境管理技巧

### 环境隔离

```bash
# 为不同项目创建独立环境
conda create -n caac_dev python=3.8    # 开发环境
conda create -n caac_prod python=3.8   # 生产环境
```

### 环境备份

```bash
# 导出环境配置
conda env export > environment.yml
pip freeze > requirements.txt

# 从配置恢复环境
conda env create -f environment.yml
pip install -r requirements.txt
```

### 清理环境

```bash
# 清理conda缓存
conda clean --all

# 删除不用的环境
conda env remove -n old_env
```

---

🎉 **安装完成！** 现在您可以开始探索CAAC的强大功能了。

> 💬 **遇到问题？** 查看 [常见问题](faq.md) 或在GitHub Issues中寻求帮助。 