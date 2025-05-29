#!/usr/bin/env python3
"""
自动更新experiments.md文档中的实验结果表格
从results目录中的CSV文件读取最新的实验数据并更新文档
"""

import os
import pandas as pd
import re
from datetime import datetime

def load_experiment_results():
    """从CSV文件加载实验结果"""
    results_dir = "results"
    
    # 读取各个详细结果文件
    experiments = {
        "linear_outlier0.0": "comparison_linear_outlier0.0.csv",
        "linear_outlier0.1": "comparison_linear_outlier0.1.csv", 
        "nonlinear_outlier0.0": "comparison_nonlinear_outlier0.0.csv",
        "nonlinear_outlier0.1": "comparison_nonlinear_outlier0.1.csv"
    }
    
    experiment_data = {}
    
    for exp_name, filename in experiments.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0)
            experiment_data[exp_name] = df
        else:
            print(f"警告: 文件 {filepath} 不存在")
    
    return experiment_data

def format_table_row(model_name, accuracy, precision, recall, f1, auc):
    """格式化表格行"""
    return f"| {model_name} | {accuracy:.3f} | {precision:.3f} | {recall:.3f} | {f1:.3f} | {auc:.3f} |"

def generate_experiment_table(df):
    """生成单个实验的结果表格"""
    table_lines = []
    table_lines.append("| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | AUC |")
    table_lines.append("|------|--------|--------|--------|--------|-----|")
    
    # 按照文档中的顺序添加模型结果
    model_mapping = {
        "CAAC": "CAAC",
        "LogisticRegression": "逻辑回归",
        "RandomForest": "随机森林", 
        "SVM": "SVM"
    }
    
    for model_key, model_name in model_mapping.items():
        if model_key in df.index:
            row = df.loc[model_key]
            table_lines.append(format_table_row(
                model_name,
                row['accuracy'],
                row['precision'], 
                row['recall'],
                row['f1'],
                row['auc']
            ))
    
    return "\n".join(table_lines)

def update_experiments_md(experiment_data):
    """更新experiments.md文件"""
    doc_file = "docs/experiments.md"
    
    if not os.path.exists(doc_file):
        print(f"错误: 文档文件 {doc_file} 不存在")
        return
    
    with open(doc_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 定义需要替换的表格部分 - 修复正则表达式
    table_sections = {
        "linear_outlier0.0": {
            "title": "#### 2.2.1 线性数据，无异常值",
            "pattern": r"(#### 2\.2\.1 线性数据，无异常值.*?\n\n)\| 模型.*?\n\|.*?\n(?:\|.*?\n)*(?=\n|####|$)"
        },
        "linear_outlier0.1": {
            "title": "#### 2.2.2 线性数据，有异常值", 
            "pattern": r"(#### 2\.2\.2 线性数据，有异常值.*?\n\n)\| 模型.*?\n\|.*?\n(?:\|.*?\n)*(?=\n|####|$)"
        },
        "nonlinear_outlier0.0": {
            "title": "#### 2.2.3 非线性数据，无异常值",
            "pattern": r"(#### 2\.2\.3 非线性数据，无异常值.*?\n\n)\| 模型.*?\n\|.*?\n(?:\|.*?\n)*(?=\n|####|$)"
        },
        "nonlinear_outlier0.1": {
            "title": "#### 2.2.4 非线性数据，有异常值",
            "pattern": r"(#### 2\.2\.4 非线性数据，有异常值.*?\n\n)\| 模型.*?\n\|.*?\n(?:\|.*?\n)*(?=\n|####|$)"
        }
    }
    
    # 更新每个表格
    for exp_name, df in experiment_data.items():
        if exp_name in table_sections:
            section_info = table_sections[exp_name]
            new_table = generate_experiment_table(df)
            
            # 查找并替换表格
            pattern = section_info["pattern"]
            match = re.search(pattern, content, re.DOTALL)
            
            if match:
                prefix = match.group(1)
                # 确保表格后面有正确的换行符
                replacement = prefix + new_table + "\n"
                content = re.sub(pattern, replacement, content, flags=re.DOTALL)
                print(f"已更新 {exp_name} 的实验结果表格")
            else:
                print(f"警告: 无法找到 {exp_name} 的表格位置")
                print(f"尝试的模式: {pattern}")
    
    # 在文档末尾添加更新时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 查找是否已有更新时间戳，如果有则替换，如果没有则添加
    timestamp_pattern = r"\n\n---\n\*最后更新时间: .*?\*"
    timestamp_text = f"\n\n---\n*最后更新时间: {timestamp}*"
    
    if re.search(timestamp_pattern, content):
        content = re.sub(timestamp_pattern, timestamp_text, content)
    else:
        content += timestamp_text
    
    # 写回文件
    with open(doc_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"实验文档已更新: {doc_file}")

def main():
    """主函数"""
    print("开始更新实验文档...")
    
    # 加载实验结果
    experiment_data = load_experiment_results()
    
    if not experiment_data:
        print("错误: 没有找到实验结果数据")
        return
    
    print(f"找到 {len(experiment_data)} 个实验的结果数据")
    
    # 更新文档
    update_experiments_md(experiment_data)
    
    print("实验文档更新完成！")

if __name__ == "__main__":
    main() 