# 实验框架结构说明

本文档说明了CAAC项目的实验框架组织结构，以及如何运行和管理实验。

## 📁 文件结构

### 🔬 实验脚本
```
caac/
├── run_experiments.py              # 合成数据集实验脚本
├── run_real_experiments.py         # 真实数据集实验脚本
├── update_experiments_doc.py       # 合成数据实验文档更新脚本
└── update_real_experiments_doc.py  # 真实数据实验文档更新脚本
```

### 📊 实验结果目录
```
caac/
├── results/                        # 合成数据集实验结果
│   ├── comparison_linear_outlier0.0.csv
│   ├── comparison_linear_outlier0.1.csv
│   ├── comparison_nonlinear_outlier0.0.csv
│   ├── comparison_nonlinear_outlier0.1.csv
│   ├── experiment_summary.csv
│   └── *.png                       # 训练历史图和比较图
└── results_real/                   # 真实数据集实验结果
    ├── comparison_iris.csv
    ├── comparison_breast_cancer.csv
    ├── comparison_adult.csv
    ├── comparison_german_credit.csv (待生成)
    ├── comparison_covertype.csv (待生成)
    ├── comparison_credit_fraud.csv (待生成)
    └── experiment_summary.csv
```

### 📖 文档
```
docs/
├── experiments_synthetic.md        # 合成数据集实验结果文档
├── experiments_real.md             # 真实数据集实验结果文档
├── _sidebar.md                     # 导航菜单 (已更新)
└── README.md                       # 主文档 (已更新)
```

## 🚀 使用指南

### 1. 运行合成数据集实验

```bash
# 运行合成数据集实验 (4种场景)
python run_experiments.py

# 更新合成数据实验文档
python update_experiments_doc.py
```

**合成数据集包括：**
- 线性数据，无异常值
- 线性数据，有异常值 (10%)
- 非线性数据，无异常值
- 非线性数据，有异常值 (10%)

### 2. 运行真实数据集实验

```bash
# 运行真实数据集实验 (6个经典数据集)
python run_real_experiments.py

# 更新真实数据实验文档
python update_real_experiments_doc.py
```

**真实数据集包括：**
- **Iris** 🌷: 鸢尾花数据集 (多分类转二分类)
- **Breast Cancer Wisconsin** 🔬: 乳腺癌诊断数据集
- **Adult (Census Income)** 💰: 成年人收入普查数据集
- **German Credit** 🇩🇪: 德国信贷数据集
- **Covertype** 🌲: 森林植被类型数据集
- **Credit Card Fraud** 💳: 信用卡欺诈检测数据集

## 🔧 技术细节

### 基线方法
- **LogisticRegression**: 逻辑回归
- **RandomForest**: 随机森林
- **SVM**: 支持向量机
- **XGBoost**: 梯度提升 (仅真实数据集实验)
- **MLP**: 多层感知机 (仅真实数据集实验)

### 评估指标
- **准确率 (Accuracy)**
- **精确率 (Precision)**
- **召回率 (Recall)**
- **F1分数 (F1 Score)**
- **AUC-ROC**: ROC曲线下面积
- **AUC-PR**: Precision-Recall曲线下面积 (真实数据集)
- **训练时间**: 模型训练耗时

### 数据预处理
- **数值特征标准化**: StandardScaler Z-score标准化
- **类别特征编码**: OneHotEncoder独热编码
- **数据集划分**: 8:1:1 (训练:验证:测试)
- **分层采样**: 保持类别分布一致

## 📈 自动文档更新

### 工作原理
1. **实验脚本**运行后生成CSV结果文件
2. **文档更新脚本**读取CSV文件中的最新结果
3. **自动替换**文档中的结果表格
4. **添加时间戳**追踪更新时间

### 更新机制
- **正则表达式匹配**: 精确定位文档中的表格位置
- **表格格式自动生成**: 根据数据集类型生成对应格式的表格
- **错误处理**: 优雅处理缺失文件和数据

## 🔍 实验状态

### ✅ 已完成
- [x] 合成数据集实验框架
- [x] 真实数据集实验框架
- [x] 自动文档更新系统
- [x] Iris数据集实验
- [x] Breast Cancer数据集实验
- [x] Adult数据集实验

### 🔄 进行中/待完成
- [ ] German Credit数据集 (数据加载问题)
- [ ] Covertype大规模数据集实验
- [ ] Credit Card Fraud极不平衡数据集实验
- [ ] CAAC-SPSFT真实模型集成 (当前使用模拟结果)

## 🛠️ 故障排除

### 常见问题

1. **数据集加载失败**
   ```bash
   Warning: 无法加载German Credit数据集: Dataset german with version 1 not found.
   ```
   **解决**: 检查网络连接，或使用本地数据集文件

2. **CAAC模型结果为模拟值**
   - 当前CAAC-SPSFT使用模拟结果进行框架测试
   - 需要集成真实的CAAC模型实现

3. **文档更新失败**
   ```bash
   警告: 无法找到 xxx 的表格位置
   ```
   **解决**: 检查文档格式，确保标题和表格结构正确

### 依赖要求
```bash
pip install numpy pandas matplotlib scikit-learn xgboost torch
```

## 📝 贡献指南

### 添加新数据集
1. 在 `run_real_experiments.py` 的 `load_real_datasets()` 中添加数据集加载逻辑
2. 在 `update_real_experiments_doc.py` 的 `dataset_info` 中添加表格映射
3. 在 `docs/experiments_real.md` 中添加相应的数据集描述和表格结构

### 添加新基线方法
1. 在 `run_baselines()` 函数中添加新模型
2. 更新文档更新脚本中的 `model_mapping`
3. 确保CSV输出包含所有必要的评估指标

---

*最后更新时间: 2025-01-20* 