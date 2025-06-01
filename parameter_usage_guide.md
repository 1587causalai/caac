# CAAC鲁棒性测试参数化使用指南

## 概述

`run_standard_robustness_test.py` 现在支持完全参数化配置，你可以轻松调整噪声水平和网络结构参数，无需修改代码。

## 基本使用

### 1. 使用默认参数运行（推荐）

**标准测试** (15-25分钟):
```bash
python run_standard_robustness_test.py
```

**快速测试** (3-5分钟):
```bash
python run_quick_robustness_test.py
```

**默认配置对比:**

| 参数 | 标准测试 | 快速测试 |
|------|----------|----------|
| 噪声水平 | 0%, 5%, 10%, 15%, 20% | 0%, 10%, 20% |
| 数据集 | 6个（小+中等规模） | 4个（仅小规模） |
| 训练轮数 | 150 | 100 |
| 早停耐心值 | 15 | 10 |
| 表征维度 | 128 | 128 |
| 特征网络隐藏层 | [64] | [64] |
| 推断网络隐藏层 | [128, 64] | [128, 64] |
| 批量大小 | 64 | 64 |
| 学习率 | 0.001 | 0.001 |

### 2. 自定义噪声水平

**标准测试:**
```bash
# 只测试轻度噪声
python run_standard_robustness_test.py --noise-levels 0.0 0.05 0.10

# 测试极端噪声场景
python run_standard_robustness_test.py --noise-levels 0.0 0.10 0.20 0.30 0.40
```

**快速测试:**
```bash
# 快速验证轻度噪声
python run_quick_robustness_test.py --noise-levels 0.0 0.05

# 快速验证极端噪声
python run_quick_robustness_test.py --noise-levels 0.0 0.20 0.40
```

### 3. 调整网络结构
```bash
# 使用更大的表征维度
python run_standard_robustness_test.py --representation-dim 256

# 使用更深的特征网络
python run_standard_robustness_test.py --feature-hidden-dims 128 64 32

# 使用更复杂的推断网络
python run_standard_robustness_test.py --abduction-hidden-dims 256 128 64 32
```

### 4. 调整训练参数
```bash
# 快速测试（较少训练轮数）
python run_standard_robustness_test.py --epochs 50 --batch-size 128

# 精细调优（更多训练轮数，更小学习率）
python run_standard_robustness_test.py --epochs 300 --learning-rate 0.0001 --early-stopping-patience 25
```

## 常用参数组合

### 快速原型验证
```bash
# 3分钟快速验证 - 只测试核心噪声水平
python run_standard_robustness_test.py \
    --noise-levels 0.0 0.10 0.20 \
    --epochs 50 \
    --datasets breast_cancer optical_digits
```

### 标准研究配置（推荐）
```bash
# 使用默认参数的完整测试
python run_standard_robustness_test.py
```

### 深度网络实验
```bash
# 测试更深更宽的网络对鲁棒性的影响
python run_standard_robustness_test.py \
    --representation-dim 256 \
    --feature-hidden-dims 128 64 \
    --abduction-hidden-dims 256 128 64 \
    --batch-size 32 \
    --epochs 200
```

### 精细噪声分析
```bash
# 更细粒度的噪声水平测试
python run_standard_robustness_test.py \
    --noise-levels 0.0 0.02 0.05 0.08 0.10 0.15 0.20 0.25
```

### 小规模快速测试
```bash
# 只使用小数据集进行快速测试
python run_standard_robustness_test.py \
    --datasets breast_cancer optical_digits \
    --epochs 100
```

## 参数详细说明

### 噪声水平参数
- `--noise-levels`: 噪声水平列表，值在 [0.0, 1.0] 范围内
- 例如: `--noise-levels 0.0 0.05 0.10 0.15 0.20`

### 网络结构参数
- `--representation-dim`: 表征维度，影响模型容量
- `--feature-hidden-dims`: 特征网络隐藏层维度列表
- `--abduction-hidden-dims`: 推断网络隐藏层维度列表

### 训练参数
- `--batch-size`: 批量大小，影响训练稳定性和速度
- `--epochs`: 最大训练轮数
- `--learning-rate`: 学习率
- `--early-stopping-patience`: 早停耐心值

### 数据集选择
- `--datasets`: 选择要测试的数据集
- 可选数据集: `breast_cancer`, `optical_digits`, `digits`, `synthetic_imbalanced`, `covertype`, `letter`, `iris`, `wine`, `mnist`, `fashion_mnist`

## 实验建议

### 日常快速验证
```bash
# 5分钟验证
python run_standard_robustness_test.py \
    --noise-levels 0.0 0.10 0.20 \
    --datasets breast_cancer optical_digits \
    --epochs 75
```

### 论文发表级别实验
```bash
# 完整标准测试（20-30分钟）
python run_standard_robustness_test.py
```

### 网络结构敏感性分析
```bash
# 测试不同表征维度
for dim in 64 128 256; do
    python run_standard_robustness_test.py --representation-dim $dim
done
```

### 超参数优化
```bash
# 测试不同学习率
for lr in 0.0001 0.001 0.01; do
    python run_standard_robustness_test.py --learning-rate $lr
done
```

## 输出文件

每次运行都会生成：
- **详细报告**: `results/caac_outlier_robustness_report_*.md`
- **鲁棒性曲线**: `results/caac_outlier_robustness_curves.png`
- **鲁棒性热力图**: `results/caac_outlier_robustness_heatmap.png`
- **原始数据**: `results/caac_outlier_robustness_detailed_*.csv`
- **汇总数据**: `results/caac_outlier_robustness_summary_*.csv`

## 性能建议

- **快速测试**: 减少epochs和数据集数量
- **标准测试**: 使用默认参数
- **深度分析**: 增加表征维度和网络深度
- **大规模实验**: 增加批量大小和早停耐心值

## 错误处理

脚本包含完整的参数验证：
- 噪声水平必须在 [0.0, 1.0] 范围内
- 所有维度参数必须为正整数
- 自动检测无效参数组合

## 兼容性

- 完全向后兼容：无参数运行使用默认配置
- 支持部分参数覆盖：只需指定想要修改的参数
- 保持所有原有功能：可视化、报告生成等 