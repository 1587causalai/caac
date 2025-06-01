"""
实验评估模块

提供以下功能：
1. 模型评估
2. 结果可视化
3. 实验配置
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import time
import json
import os

class ModelEvaluator:
    """
    模型评估类
    
    提供模型评估、结果可视化和实验配置功能。
    """
    
    def __init__(self, results_dir='../../results'):
        """
        初始化模型评估类
        
        Args:
            results_dir: 结果保存目录
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def evaluate_model(self, model, X_test, y_test):
        """
        评估模型
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试标签
            
        Returns:
            metrics: 评估指标字典
        """
        # 预测概率和类别
        y_pred_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        
        # 计算评估指标
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        
        # 对于二分类问题，计算精确率、召回率和F1分数
        if len(np.unique(y_test)) == 2:
            metrics['precision'] = precision_score(y_test, y_pred)
            metrics['recall'] = recall_score(y_test, y_pred)
            metrics['f1'] = f1_score(y_test, y_pred)
        else:
            # 对于多分类问题，计算宏平均和加权平均
            metrics['precision_macro'] = precision_score(y_test, y_pred, average='macro')
            metrics['recall_macro'] = recall_score(y_test, y_pred, average='macro')
            metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro')
            metrics['precision_weighted'] = precision_score(y_test, y_pred, average='weighted')
            metrics['recall_weighted'] = recall_score(y_test, y_pred, average='weighted')
            metrics['f1_weighted'] = f1_score(y_test, y_pred, average='weighted')
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        return metrics, y_pred_proba, y_pred, cm
    
    def visualize_confusion_matrix(self, cm, class_names, title='Confusion Matrix', save_path=None):
        """
        可视化混淆矩阵
        
        Args:
            cm: 混淆矩阵
            class_names: 类别名称
            title: 图表标题
            save_path: 保存路径
            
        Returns:
            fig: 图表对象
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            
        return plt.gcf()
    
    def visualize_roc_curve(self, y_test, y_pred_proba, n_classes, save_path=None):
        """
        可视化ROC曲线
        
        Args:
            y_test: 测试标签
            y_pred_proba: 预测概率
            n_classes: 类别数量
            save_path: 保存路径
            
        Returns:
            fig: 图表对象
        """
        plt.figure(figsize=(10, 8))
        
        if n_classes == 2:
            # 二分类ROC曲线
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
        else:
            # 多分类ROC曲线（一对多）
            from sklearn.preprocessing import label_binarize
            
            # 将标签二值化
            y_test_bin = label_binarize(y_test, classes=range(n_classes))
            
            # 计算每个类别的ROC曲线和AUC
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (area = {roc_auc[i]:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (One-vs-Rest)')
            plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            
        return plt.gcf()
    
    def visualize_uncertainty(self, model, X_test, y_test, n_samples=10, save_path=None):
        """
        可视化不确定性
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试标签
            n_samples: 样本数量
            save_path: 保存路径
            
        Returns:
            fig: 图表对象
        """
        # 获取模型内部的统一分类网络
        unified_net = model.model
        
        # 随机选择n_samples个样本
        indices = np.random.choice(len(X_test), n_samples, replace=False)
        X_samples = X_test[indices]
        y_samples = y_test[indices]
        
        # 转换为PyTorch张量
        import torch
        X_tensor = torch.FloatTensor(X_samples).to(model.device)
        
        # 获取预测结果和不确定性参数
        unified_net.eval()
        with torch.no_grad():
            class_probs, loc, scale, _, _ = unified_net(X_tensor)
        
        # 转换为NumPy数组
        class_probs = class_probs.cpu().numpy()
        loc = loc.cpu().numpy()
        scale = scale.cpu().numpy()
        
        # 可视化
        n_classes = class_probs.shape[1]
        fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4 * n_samples))
        
        for i, ax in enumerate(axes):
            # 绘制每个类别的位置参数和尺度参数
            ax.bar(range(n_classes), loc[i], alpha=0.6, label='Location (Center)')
            ax.errorbar(range(n_classes), loc[i], yerr=scale[i], fmt='o', capsize=5, label='Scale (Uncertainty)')
            
            # 标记真实类别
            true_class = y_samples[i]
            ax.axvline(x=true_class, color='r', linestyle='--', label=f'True Class: {true_class}')
            
            # 标记预测类别
            pred_class = np.argmax(class_probs[i])
            ax.axvline(x=pred_class, color='g', linestyle=':', label=f'Predicted Class: {pred_class}')
            
            ax.set_xticks(range(n_classes))
            ax.set_xlabel('Class')
            ax.set_ylabel('Score')
            ax.set_title(f'Sample {i+1}: Uncertainty Visualization')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            
        return fig
    
    def save_experiment_results(self, experiment_name, dataset_name, model_params, metrics, 
                               train_time, n_samples, n_features, n_classes, class_distribution):
        """
        保存实验结果
        
        Args:
            experiment_name: 实验名称
            dataset_name: 数据集名称
            model_params: 模型参数
            metrics: 评估指标
            train_time: 训练时间
            n_samples: 样本数量
            n_features: 特征数量
            n_classes: 类别数量
            class_distribution: 类别分布
            
        Returns:
            result_path: 结果保存路径
        """
        # 创建结果目录
        experiment_dir = os.path.join(self.results_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # 转换所有值为JSON可序列化的类型
        def convert_to_serializable(obj):
            """递归转换numpy类型为Python原生类型"""
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        # 创建结果字典
        result = {
            'experiment_name': experiment_name,
            'dataset_name': dataset_name,
            'model_params': convert_to_serializable(model_params),
            'metrics': convert_to_serializable(metrics),
            'train_time': float(train_time),
            'n_samples': int(n_samples),
            'n_features': int(n_features),
            'n_classes': int(n_classes),
            'class_distribution': convert_to_serializable(class_distribution),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存结果
        result_path = os.path.join(experiment_dir, f'{dataset_name}_results.json')
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=4)
        
        return result_path
