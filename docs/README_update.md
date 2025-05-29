# 自动更新实验文档

## 概述

项目中包含了一个自动更新实验文档的脚本 `update_experiments_doc.py`，可以从 `results/` 目录中的CSV文件自动读取最新的实验结果，并更新 `docs/experiments_synthetic.md` 文档中的所有结果表格。

## 使用方法

### 方法1：使用Makefile（推荐）

```bash
# 仅更新文档
make update-docs

# 运行完整实验流程（运行实验 + 更新文档）
make full-experiment

# 查看所有可用命令
make help
```

### 方法2：直接运行脚本

```bash
python update_experiments_doc.py
```

## 工作原理

1. **读取实验结果**：脚本自动从以下CSV文件读取数据：
   - `results/comparison_linear_outlier0.0.csv`
   - `results/comparison_linear_outlier0.1.csv`
   - `results/comparison_nonlinear_outlier0.0.csv`
   - `results/comparison_nonlinear_outlier0.1.csv`

2. **更新文档表格**：使用正则表达式匹配和替换 `docs/experiments_synthetic.md` 中的四个实验结果表格

3. **添加时间戳**：在文档末尾添加更新时间，方便追踪文档版本

## 文件说明

- `update_experiments_doc.py`：自动更新脚本
- `docs/experiments_synthetic.md`：合成数据集实验结果文档（会被自动更新）
- `results/`：实验输出目录，包含CSV和图像文件

## 注意事项

- 确保已经运行过 `run_experiments.py` 生成了实验结果文件
- 脚本会保留文档的原始格式和结构，只更新数据表格
- 建议在Git中查看更新前后的差异，确认更新正确

## 示例输出

```
开始更新实验文档...
找到 4 个实验的结果数据
已更新 linear_outlier0.0 的实验结果表格
已更新 linear_outlier0.1 的实验结果表格
已更新 nonlinear_outlier0.0 的实验结果表格
已更新 nonlinear_outlier0.1 的实验结果表格
实验文档已更新: docs/experiments_synthetic.md
实验文档更新完成！ 