#!/usr/bin/env python3
"""
自动更新experiments_real.md文档中的实验结果表格
从results_real目录中的CSV文件读取最新的实验数据并更新文档
"""

import os
import pandas as pd
import re
from datetime import datetime

def load_real_experiment_results():
    """从CSV文件加载真实数据集实验结果"""
    results_dir = "results_real"
    
    if not os.path.exists(results_dir):
        print(f"错误: 结果目录 {results_dir} 不存在")
        return {}
    
    # 自动发现所有CSV文件
    experiment_data = {}
    
    for filename in os.listdir(results_dir):
        if filename.startswith("comparison_") and filename.endswith(".csv"):
            exp_name = filename.replace("comparison_", "").replace(".csv", "")
            filepath = os.path.join(results_dir, filename)
            
            try:
                df = pd.read_csv(filepath, index_col=0)
                experiment_data[exp_name] = df
                print(f"成功加载: {exp_name} ({filename})")
            except Exception as e:
                print(f"警告: 无法加载文件 {filepath}: {e}")
    
    return experiment_data

def format_real_table_row(model_name, accuracy, precision, recall, f1, auc_roc, auc_pr=None, train_time=None):
    """格式化真实数据集表格行"""
    if auc_pr is not None and train_time is not None:
        # 完整表格 (6列指标)
        return f"| {model_name} | {accuracy:.3f} | {precision:.3f} | {recall:.3f} | {f1:.3f} | {auc_roc:.3f} | {auc_pr:.3f} |"
    elif train_time is not None:
        # 多分类表格 (包含训练时间)
        return f"| {model_name} | {accuracy:.3f} | {f1:.3f} | {train_time:.1f} |"
    else:
        # 标准二分类表格 (5列指标)
        return f"| {model_name} | {accuracy:.3f} | {precision:.3f} | {recall:.3f} | {f1:.3f} | {auc_roc:.3f} |"

def generate_real_experiment_table(df, table_type="binary"):
    """生成真实数据集实验的结果表格"""
    table_lines = []
    
    if table_type == "binary_full":
        # 完整二分类表格 (包含AUC-PR)
        table_lines.append("| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | AUC-ROC | AUC-PR |")
        table_lines.append("|------|--------|--------|--------|--------|---------|--------|")
    elif table_type == "multiclass":
        # 多分类表格
        table_lines.append("| 模型 | 准确率 | 宏平均F1 | 训练时间(s) |")
        table_lines.append("|------|--------|----------|------------|")
    else:
        # 标准二分类表格
        table_lines.append("| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | AUC-ROC |")
        table_lines.append("|------|--------|--------|--------|--------|---------|")
    
    # 模型名称映射
    model_mapping = {
        "CAAC-SPSFT": "CAAC-SPSFT",
        "LogisticRegression": "逻辑回归",
        "RandomForest": "随机森林", 
        "SVM": "SVM",
        "XGBoost": "XGBoost",
        "MLP": "MLP"
    }
    
    for model_key, model_name in model_mapping.items():
        if model_key in df.index:
            row = df.loc[model_key]
            
            if table_type == "binary_full":
                # 假设有AUC-PR列
                auc_pr = row.get('auc_pr', row['auc_roc'])  # 如果没有AUC-PR，使用AUC-ROC
                table_lines.append(format_real_table_row(
                    model_name, row['accuracy'], row['precision'], 
                    row['recall'], row['f1'], row['auc_roc'], auc_pr
                ))
            elif table_type == "multiclass":
                table_lines.append(f"| {model_name} | {row['accuracy']:.3f} | {row['f1']:.3f} | {row.get('train_time', 0.0):.1f} |")
            else:
                table_lines.append(format_real_table_row(
                    model_name, row['accuracy'], row['precision'], 
                    row['recall'], row['f1'], row['auc_roc']
                ))
    
    return "\n".join(table_lines)

def update_real_experiments_md(experiment_data):
    """更新experiments_real.md文件"""
    doc_file = "docs/experiments_real.md"
    
    if not os.path.exists(doc_file):
        print(f"错误: 文档文件 {doc_file} 不存在")
        return
    
    with open(doc_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 定义数据集对应的表格信息
    dataset_info = {
        "iris": {
            "title": "#### 2.1.1 Iris数据集",
            "pattern": r"(#### 2\.1\.1 Iris数据集.*?\n\n.*?\n\n)\| 模型.*?\n\|.*?\n(?:\|.*?\n)*(?=\n|####|$)",
            "table_type": "multiclass"
        },
        "breast_cancer": {
            "title": "#### 2.1.2 Breast Cancer Wisconsin数据集", 
            "pattern": r"(#### 2\.1\.2 Breast Cancer Wisconsin数据集.*?\n\n.*?\n\n)\| 模型.*?\n\|.*?\n(?:\|.*?\n)*(?=\n|####|$)",
            "table_type": "binary_full"
        },
        "german_credit": {
            "title": "#### 2.1.3 German Credit数据集",
            "pattern": r"(#### 2\.1\.3 German Credit数据集.*?\n\n.*?\n\n)\| 模型.*?\n\|.*?\n(?:\|.*?\n)*(?=\n|####|$)",
            "table_type": "binary_full"
        },
        "adult": {
            "title": "#### 2.2.1 Adult数据集",
            "pattern": r"(#### 2\.2\.1 Adult数据集.*?\n\n.*?\n\n)\| 模型.*?\n\|.*?\n(?:\|.*?\n)*(?=\n|####|$)",
            "table_type": "binary_full"
        },
        "covertype": {
            "title": "#### 2.3.1 Covertype数据集",
            "pattern": r"(#### 2\.3\.1 Covertype数据集.*?\n\n.*?\n\n)\| 模型.*?\n\|.*?\n(?:\|.*?\n)*(?=\n|####|$)",
            "table_type": "multiclass"
        },
        "credit_fraud": {
            "title": "#### 2.4.1 Credit Card Fraud数据集",
            "pattern": r"(#### 2\.4\.1 Credit Card Fraud数据集.*?\n\n.*?\n\n)\| 模型.*?\n\|.*?\n(?:\|.*?\n)*(?=\n|####|$)",
            "table_type": "binary_full"
        }
    }
    
    # 只更新实际存在数据的表格
    updated_count = 0
    for exp_name, df in experiment_data.items():
        if exp_name in dataset_info:
            section_info = dataset_info[exp_name]
            new_table = generate_real_experiment_table(df, section_info["table_type"])
            
            # 查找并替换表格
            pattern = section_info["pattern"]
            match = re.search(pattern, content, re.DOTALL)
            
            if match:
                prefix = match.group(1)
                replacement = prefix + new_table + "\n"
                content = re.sub(pattern, replacement, content, flags=re.DOTALL)
                print(f"已更新 {exp_name} 的实验结果表格")
                updated_count += 1
            else:
                print(f"警告: 无法找到 {exp_name} 的表格位置")
        else:
            print(f"警告: 未定义 {exp_name} 的表格信息")
    
    if updated_count > 0:
        # 添加或更新时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp_pattern = r"\n\n---\n\*最后更新时间: .*?\*"
        timestamp_text = f"\n\n---\n*最后更新时间: {timestamp}*"
        
        if re.search(timestamp_pattern, content):
            content = re.sub(timestamp_pattern, timestamp_text, content)
        else:
            content += timestamp_text
        
        # 写回文件
        with open(doc_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"真实数据集实验文档已更新: {doc_file} (更新了 {updated_count} 个表格)")
    else:
        print("没有找到可更新的表格")

def main():
    """主函数"""
    print("开始更新真实数据集实验文档...")
    
    # 加载实验结果
    experiment_data = load_real_experiment_results()
    
    if not experiment_data:
        print("错误: 没有找到真实数据集实验结果数据")
        return
    
    print(f"找到 {len(experiment_data)} 个数据集的结果数据")
    
    # 更新文档
    update_real_experiments_md(experiment_data)
    
    print("真实数据集实验文档更新完成！")

if __name__ == "__main__":
    main() 