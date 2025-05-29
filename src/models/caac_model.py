"""
CAAC-SPSFT 模型实现

提供以下组件：
1. CAACModel - 完整的CAAC-SPSFT二分类模型
2. CAACModelWrapper - 模型包装类，封装训练、验证和预测功能
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.utils.data import Dataset, DataLoader

from .base_networks import FeatureNetwork, AbductionNetwork
from .pathway_network import PathwayNetwork
from .threshold_mechanism import FixedThresholdMechanism
from .classification_head import ClassificationHead
from .loss_functions import compute_nll_loss

class CAACModel(nn.Module):
    """
    CAAC-SPSFT 二分类模型
    
    结合了因果表征生成、多路径网络、固定阈值机制和分类概率计算等核心组件。
    """
    def __init__(self, input_dim, representation_dim=64, latent_dim=64, 
                 n_paths=2, n_classes=2,
                 feature_hidden_dims=[64], abduction_hidden_dims=[128, 64]):
        super(CAACModel, self).__init__()
        
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim
        self.n_paths = n_paths
        self.n_classes = n_classes
        
        # 特征提取网络
        self.feature_net = FeatureNetwork(input_dim, representation_dim, feature_hidden_dims)
        
        # 推断网络
        self.abduction_net = AbductionNetwork(representation_dim, latent_dim, abduction_hidden_dims)
        
        # 多路径网络
        self.pathway_net = PathwayNetwork(latent_dim, n_paths)
        
        # 固定阈值机制
        self.threshold_mechanism = FixedThresholdMechanism(n_classes)
        
        # 分类头
        self.classification_head = ClassificationHead(n_classes)
    
    def forward(self, x):
        # 特征提取
        representation = self.feature_net(x)
        
        # 推断因果表征参数
        location_param, scale_param = self.abduction_net(representation)
        
        # 多路径处理
        mu_scores, gamma_scores, path_probs = self.pathway_net(location_param, scale_param)
        
        # 获取固定阈值
        thresholds = self.threshold_mechanism()
        
        # 计算分类概率
        class_probs = self.classification_head(mu_scores, gamma_scores, path_probs, thresholds)
        
        return class_probs, location_param, scale_param, mu_scores, gamma_scores, path_probs, thresholds

class CAACModelWrapper:
    """
    CAAC-SPSFT 模型包装类
    
    封装训练、验证和预测功能。
    """
    def __init__(self, input_dim, 
                 representation_dim=64, 
                 latent_dim=64, 
                 n_paths=2,
                 n_classes=2,
                 feature_hidden_dims=[64], 
                 abduction_hidden_dims=[128, 64], 
                 lr=0.001, batch_size=32, epochs=100, device=None,
                 early_stopping_patience=None, early_stopping_min_delta=0.0001):
        
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim
        self.n_paths = n_paths
        self.n_classes = n_classes
        self.feature_hidden_dims = feature_hidden_dims
        self.abduction_hidden_dims = abduction_hidden_dims
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self._setup_model_optimizer()
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 
                        'train_time': 0, 'best_epoch': 0}
    
    def _setup_model_optimizer(self):
        self.model = CAACModel(
            self.input_dim, 
            self.representation_dim, 
            self.latent_dim, 
            self.n_paths,
            self.n_classes,
            self.feature_hidden_dims, 
            self.abduction_hidden_dims
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def compute_loss(self, y_true, class_probs):
        return compute_nll_loss(y_true, class_probs)
    
    def compute_accuracy(self, y_true, class_probs):
        _, predicted = torch.max(class_probs, 1)
        correct = (predicted == y_true).sum().item()
        total = y_true.size(0)
        return correct / total
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state_dict = None
        
        has_validation = False
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val).to(self.device)
            has_validation = True
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_train_loss = 0
            epoch_train_acc = 0
            
            for batch_X, batch_y in train_loader:
                class_probs, *_ = self.model(batch_X)
                loss = self.compute_loss(batch_y, class_probs)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_train_loss += loss.item()
                epoch_train_acc += self.compute_accuracy(batch_y, class_probs)
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            avg_train_acc = epoch_train_acc / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(avg_train_acc)
            
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    val_class_probs, *_ = self.model(X_val_tensor)
                    val_loss = self.compute_loss(y_val_tensor, val_class_probs)
                    val_acc = self.compute_accuracy(y_val_tensor, val_class_probs)
                
                self.history['val_loss'].append(val_loss.item())
                self.history['val_acc'].append(val_acc)
                
                if verbose and (epoch + 1) % (self.epochs // 10 if self.epochs >= 10 else 1) == 0:
                    print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}, '
                          f'Train Acc: {avg_train_acc:.4f}, Val Loss: {val_loss.item():.4f}, '
                          f'Val Acc: {val_acc:.4f}')
                
                if self.early_stopping_patience is not None:
                    if val_loss < best_val_loss - self.early_stopping_min_delta:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_model_state_dict = self.model.state_dict().copy()
                        self.history['best_epoch'] = epoch + 1
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= self.early_stopping_patience:
                        if verbose:
                            print(f'Early stopping triggered at epoch {epoch+1}')
                        break
            else:
                if verbose and (epoch + 1) % (self.epochs // 10 if self.epochs >= 10 else 1) == 0:
                    print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}, '
                          f'Train Acc: {avg_train_acc:.4f}')
        
        self.history['train_time'] = time.time() - start_time
        
        if best_model_state_dict is not None:
            self.model.load_state_dict(best_model_state_dict)
            if verbose:
                print(f'Loaded best model from epoch {self.history["best_epoch"]}')
        
        return self
    
    def predict_proba(self, X):
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            class_probs, *_ = self.model(X_tensor)
        return class_probs.cpu().numpy()
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
