"""
CAAC OvR 分类模型
参考回归模型设计，实现统一网络架构但不同损失函数的分类模型，确保公平比较
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class FeatureNetwork(nn.Module):
    """
    特征网络 (Feature Network) - 与回归模型完全一致
    """
    def __init__(self, input_dim, representation_dim, hidden_dims=[64]):
        super(FeatureNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim_i in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim_i))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim_i
        layers.append(nn.Linear(prev_dim, representation_dim))
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.hidden_dims = hidden_dims 

    def forward(self, x):
        return self.network(x)

class AbductionNetwork(nn.Module):
    """
    统一推断网络 (Abduction Network) - 与回归模型完全一致
    """
    def __init__(self, representation_dim, latent_dim, hidden_dims=[64, 32]):
        super(AbductionNetwork, self).__init__()
        shared_layers_list = []
        prev_dim = representation_dim
        for hidden_dim_i in hidden_dims:
            shared_layers_list.append(nn.Linear(prev_dim, hidden_dim_i))
            shared_layers_list.append(nn.ReLU())
            prev_dim = hidden_dim_i
        shared_output_dim = prev_dim 
        self.location_head = nn.Linear(shared_output_dim, latent_dim)
        self.scale_head = nn.Linear(shared_output_dim, latent_dim)
        self.shared_mlp = nn.Sequential(*shared_layers_list)
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
            
    def forward(self, representation):
        shared_features = self.shared_mlp(representation)
        location_param = self.location_head(shared_features)
        scale_param = F.softplus(self.scale_head(shared_features))
        return location_param, scale_param

class ActionNetwork(nn.Module):
    """
    行动网络 (Action Network) - 处理因果表征随机变量，输出Score分布参数
    概念上：输入随机变量U_j，通过线性变换输出Score随机变量S_k的参数
    """
    def __init__(self, latent_dim, n_classes):
        super(ActionNetwork, self).__init__()
        self.linear = nn.Linear(latent_dim, n_classes)
        self.latent_dim = latent_dim
        self.n_classes = n_classes
    
    def forward(self, location_param):
        # 注：这里输入location_param是为了兼容现有架构
        # 概念上应该处理随机变量，但实际通过权重矩阵在损失函数中计算分布参数
        return self.linear(location_param)
    
    def get_weights(self):
        weight = self.linear.weight.data  # [n_classes, latent_dim] - 线性变换矩阵A
        bias = self.linear.bias.data      # [n_classes] - 偏置B
        return weight, bias
    
    def compute_class_distribution_params(self, location_param, scale_param, distribution_type='cauchy'):
        """
        计算每个类别Score随机变量的分布参数
        不同分布类型使用不同的线性组合规则
        """
        W, b = self.get_weights()
        batch_size = location_param.size(0)
        
        # 位置参数：loc(S_k) = W_k @ location_param + b_k (所有分布相同)
        class_locations = torch.matmul(location_param, W.T) + b.unsqueeze(0)
        
        if distribution_type == 'cauchy':
            # 柯西分布：scale(S_k) = |W_k| @ scale_param
            W_abs = torch.abs(W)
            class_scales = torch.matmul(scale_param, W_abs.T)
            return class_locations, torch.clamp(class_scales, min=1e-6)
            
        elif distribution_type == 'gaussian':
            # 高斯分布：var(S_k) = W_k^2 @ scale_param^2, std(S_k) = sqrt(var)
            W_squared = W ** 2
            class_variances = torch.matmul(scale_param ** 2, W_squared.T)
            class_stds = torch.sqrt(torch.clamp(class_variances, min=1e-6))
            return class_locations, class_stds
            
        else:
            raise ValueError(f"Unsupported distribution type: {distribution_type}")

    def compute_scores_from_samples(self, samples):
        """
        直接从采样的潜在向量计算得分
        
        Args:
            samples: [batch_size, n_samples, latent_dim] - 采样的潜在向量
        
        Returns:
            scores: [batch_size, n_samples, n_classes] - 每个采样的得分
        """
        W, b = self.get_weights()
        # samples: [batch_size, n_samples, latent_dim]
        # W: [n_classes, latent_dim]
        # 结果: [batch_size, n_samples, n_classes]
        scores = torch.matmul(samples, W.T) + b.unsqueeze(0).unsqueeze(0)
        return scores

class UnifiedClassificationNetwork(nn.Module):
    """
    统一分类网络 (Unified Classification Network)
    包含 FeatureNetwork -> AbductionNetwork -> ActionNetwork
    """
    def __init__(self, input_dim, representation_dim, latent_dim, n_classes,
                 feature_hidden_dims, abduction_hidden_dims):
        super(UnifiedClassificationNetwork, self).__init__()
        self.feature_net = FeatureNetwork(input_dim, representation_dim, feature_hidden_dims)
        self.abduction_net = AbductionNetwork(representation_dim, latent_dim, abduction_hidden_dims)
        self.action_net = ActionNetwork(latent_dim, n_classes)
        
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.feature_hidden_dims = feature_hidden_dims
        self.abduction_hidden_dims = abduction_hidden_dims

    def forward(self, x):
        representation = self.feature_net(x)
        location_param, scale_param = self.abduction_net(representation)
        logits = self.action_net(location_param)
        return logits, location_param, scale_param

    def predict_proba(self, x):
        logits, _, _ = self.forward(x)
        return F.softmax(logits, dim=1)

class CAACOvRModel:
    """
    CAAC OvR分类模型 - 使用柯西分布损失函数
    
    支持可学习阈值变体：
    - learnable_thresholds=False: 固定阈值 C_k = 0 (默认)
    - learnable_thresholds=True: 可学习阈值 C_k 作为模型参数
    """
    def __init__(self, input_dim, 
                 representation_dim=64, 
                 latent_dim=None,  # 默认等于representation_dim，体现d_latent = d_repr的概念
                 n_classes=2,
                 feature_hidden_dims=[64], 
                 abduction_hidden_dims=[128, 64], 
                 lr=0.001, batch_size=32, epochs=100, device=None,
                 early_stopping_patience=None, early_stopping_min_delta=0.0001,
                 learnable_thresholds=False,  # 新增：是否使用可学习阈值
                 uniqueness_constraint=False,  # 新增：是否启用唯一性约束
                 uniqueness_samples=10,  # 新增：每个样本的采样次数
                 uniqueness_weight=0.1):  # 新增：唯一性约束损失权重
        
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        # 概念对齐：因果表征维度默认等于特征表征维度 (d_latent = d_repr)
        self.latent_dim = latent_dim if latent_dim is not None else representation_dim
        self.n_classes = n_classes
        self.feature_hidden_dims = feature_hidden_dims
        self.abduction_hidden_dims = abduction_hidden_dims
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.learnable_thresholds = learnable_thresholds
        self.uniqueness_constraint = uniqueness_constraint  # 新增：唯一性约束标志
        self.uniqueness_samples = uniqueness_samples  # 新增：采样次数
        self.uniqueness_weight = uniqueness_weight  # 新增：约束损失权重
        self.device_str = str(device)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        self.label_encoder = LabelEncoder()
        self._setup_model_optimizer()
        self.history = {'train_loss': [], 'val_loss': [], 'train_time': 0, 'best_epoch': 0}
    
    def _setup_model_optimizer(self):
        self.model = UnifiedClassificationNetwork(
            self.input_dim, self.representation_dim, self.latent_dim, self.n_classes,
            self.feature_hidden_dims, self.abduction_hidden_dims
        ).to(self.device)
        
        # 如果使用可学习阈值，创建阈值参数
        if self.learnable_thresholds:
            self.thresholds = nn.Parameter(torch.zeros(self.n_classes, device=self.device))
        else:
            self.thresholds = None
            
        # 设置优化器参数，包含模型参数和可能的阈值参数
        params_to_optimize = list(self.model.parameters())
        if self.learnable_thresholds and self.thresholds is not None:
            params_to_optimize.append(self.thresholds)
        self.optimizer = torch.optim.Adam(params_to_optimize, lr=self.lr)
    
    def compute_loss(self, y_true, logits, location_param, scale_param):
        """
        CAAC柯西损失函数：体现因果表征随机变量到Score随机变量的分布变换
        1. 通过ActionNetwork计算每个类别Score的柯西分布参数
        2. 用柯西CDF计算每个类别的概率 P_k
        3. 用P_k计算二元交叉熵损失
        4. 如果启用，添加唯一性约束损失
        """
        batch_size = y_true.size(0)
        n_classes = self.n_classes
        device = y_true.device
        
        # 使用ActionNetwork计算每个类别Score的分布参数 (体现分布类型差异)
        class_locations, class_scales = self.model.action_net.compute_class_distribution_params(
            location_param, scale_param, distribution_type='cauchy'
        )
        
        # 根据设置使用固定阈值或可学习阈值
        if self.learnable_thresholds and self.thresholds is not None:
            thresholds = self.thresholds  # 使用可学习阈值参数
        else:
            thresholds = torch.zeros(n_classes, device=device)  # 使用固定阈值 C_k = 0
        
        # 计算柯西分布CDF：P_k = P(S_k > C_k) = 1 - F(C_k; loc, scale)
        pi = torch.tensor(np.pi, device=device)
        normalized_thresholds = (thresholds.unsqueeze(0) - class_locations) / class_scales
        P_k = 0.5 - (1/pi) * torch.atan(normalized_thresholds)
        P_k = torch.clamp(P_k, min=1e-7, max=1-1e-7)
        
        # 转换标签为二元标签并计算BCE损失
        y_binary = torch.zeros(batch_size, n_classes, device=device)
        y_binary.scatter_(1, y_true.unsqueeze(1), 1)
        
        bce_loss = -(y_binary * torch.log(P_k) + (1 - y_binary) * torch.log(1 - P_k))
        total_loss = torch.mean(bce_loss)
        
        # 如果启用唯一性约束，添加最大-次大间隔约束损失
        if self.uniqueness_constraint and self.uniqueness_samples > 0:
            # 从柯西分布采样多个潜在向量实例
            # 使用重参数化技巧进行采样：u = location + scale * tan(pi * (p - 0.5))
            # 其中 p ~ Uniform(0, 1)
            n_samples = self.uniqueness_samples
            
            # 生成均匀分布的随机数
            p = torch.rand(batch_size, n_samples, self.latent_dim, device=device)
            # 转换为标准柯西分布
            standard_cauchy = torch.tan(pi * (p - 0.5))
            # 缩放和平移得到目标柯西分布的采样
            # samples: [batch_size, n_samples, latent_dim]
            samples = location_param.unsqueeze(1) + scale_param.unsqueeze(1) * standard_cauchy
            
            # 计算每个采样的得分
            # scores: [batch_size, n_samples, n_classes]
            scores = self.model.action_net.compute_scores_from_samples(samples)
            
            # 实现最大-次大间隔约束
            # 对每个采样，找到最大和次大的得分
            top2_scores, top2_indices = torch.topk(scores, k=min(2, n_classes), dim=2)
            
            if n_classes > 1:
                max_scores = top2_scores[:, :, 0]  # [batch_size, n_samples]
                second_max_scores = top2_scores[:, :, 1]  # [batch_size, n_samples]
                max_indices = top2_indices[:, :, 0]  # [batch_size, n_samples]
                second_max_indices = top2_indices[:, :, 1]  # [batch_size, n_samples]
                
                # 获取对应的阈值
                max_thresholds = thresholds[max_indices]  # [batch_size, n_samples]
                second_max_thresholds = thresholds[second_max_indices]  # [batch_size, n_samples]
                
                # 计算约束损失：
                # 1. 最大得分应该超过其阈值：max(0, threshold_max - score_max)
                # 2. 次大得分应该低于其阈值：max(0, score_second - threshold_second)
                max_violation = F.relu(max_thresholds - max_scores)
                second_violation = F.relu(second_max_scores - second_max_thresholds)
                
                # 计算平均违反损失（而不是累加）
                total_violations = max_violation + second_violation
                uniqueness_loss = torch.mean(total_violations)  # 对所有采样取平均
                
                # 添加权重并加到总损失中
                total_loss = total_loss + self.uniqueness_weight * uniqueness_loss
        
        return total_loss

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        # 编码标签
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        self.n_classes = len(self.label_encoder.classes_)
        
        # 重新设置模型以匹配正确的类别数
        self._setup_model_optimizer()
        
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train_encoded).to(self.device)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state_dict = None
        final_epoch_count = self.epochs 

        has_validation = False
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val_encoded).to(self.device)
            has_validation = True
        
        effective_early_stopping_patience = self.early_stopping_patience
        if not has_validation or self.early_stopping_patience is None or self.early_stopping_patience <= 0:
            effective_early_stopping_patience = None
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            final_epoch_count = epoch + 1 
            self.model.train()
            epoch_train_loss = 0
            for batch_X, batch_y_true in train_loader:
                logits, location_param, scale_param = self.model(batch_X)
                loss = self.compute_loss(batch_y_true, logits, location_param, scale_param)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0
            self.history['train_loss'].append(avg_train_loss)
            
            current_val_loss = float('inf')
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    logits_val, loc_param_val, scale_param_val = self.model(X_val_tensor)
                    current_val_loss = self.compute_loss(y_val_tensor, logits_val, loc_param_val, scale_param_val).item()
                self.history['val_loss'].append(current_val_loss)
                
                # Early stopping logic
                if effective_early_stopping_patience is not None:
                    if current_val_loss < best_val_loss - self.early_stopping_min_delta:
                        best_val_loss = current_val_loss
                        patience_counter = 0
                        best_model_state_dict = self.model.state_dict().copy()
                        self.history['best_epoch'] = epoch + 1
                    else:
                        patience_counter += 1
                        if patience_counter >= effective_early_stopping_patience:
                            if verbose >= 1:
                                print(f"Early stopping at epoch {epoch+1}")
                            break
                else:
                    best_model_state_dict = self.model.state_dict().copy()
                    self.history['best_epoch'] = epoch + 1
            else:
                best_model_state_dict = self.model.state_dict().copy()
                self.history['best_epoch'] = epoch + 1
            
            if verbose >= 2 or (verbose == 1 and (epoch+1) % 20 == 0):
                msg = f"Epoch {epoch+1:4d}/{self.epochs}, Train Loss: {avg_train_loss:.6f}"
                if has_validation:
                    msg += f", Val Loss: {current_val_loss:.6f}"
                print(msg)
        
        end_time = time.time()
        self.history['train_time'] = end_time - start_time
        
        # Load best model
        if best_model_state_dict is not None:
            self.model.load_state_dict(best_model_state_dict)
        
        if verbose >= 1:
            print(f"Training completed in {self.history['train_time']:.2f}s, Best epoch: {self.history['best_epoch']}")

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            probs = self.model.predict_proba(X_tensor)
        return probs.cpu().numpy()

    def predict(self, X):
        probs = self.predict_proba(X)
        y_pred_encoded = np.argmax(probs, axis=1)
        return self.label_encoder.inverse_transform(y_pred_encoded)

    def get_params(self, deep=True):
        return {
            'input_dim': self.input_dim,
            'representation_dim': self.representation_dim,
            'latent_dim': self.latent_dim,
            'n_classes': self.n_classes,
            'feature_hidden_dims': self.feature_hidden_dims,
            'abduction_hidden_dims': self.abduction_hidden_dims,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'device': self.device_str,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta,
            'learnable_thresholds': self.learnable_thresholds,
            'uniqueness_constraint': self.uniqueness_constraint,
            'uniqueness_samples': self.uniqueness_samples,
            'uniqueness_weight': self.uniqueness_weight
        }

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

class SoftmaxMLPModel:
    """
    Softmax MLP模型 - 使用标准Softmax策略和交叉熵损失
    仅使用位置参数（location_param），不使用尺度参数（scale_param）
    """
    def __init__(self, input_dim, 
                 representation_dim=64, 
                 latent_dim=None,  # 默认等于representation_dim，体现d_latent = d_repr的概念
                 n_classes=2,
                 feature_hidden_dims=[64], 
                 abduction_hidden_dims=[128, 64], 
                 lr=0.001, batch_size=32, epochs=100, device=None,
                 early_stopping_patience=None, early_stopping_min_delta=0.0001):
        
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        # 概念对齐：因果表征维度默认等于特征表征维度 (d_latent = d_repr)
        self.latent_dim = latent_dim if latent_dim is not None else representation_dim
        self.n_classes = n_classes
        self.feature_hidden_dims = feature_hidden_dims
        self.abduction_hidden_dims = abduction_hidden_dims
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device_str = str(device)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        self.label_encoder = LabelEncoder()
        self._setup_model_optimizer()
        self.history = {'train_loss': [], 'val_loss': [], 'train_time': 0, 'best_epoch': 0}
    
    def _setup_model_optimizer(self):
        self.model = UnifiedClassificationNetwork(
            self.input_dim, self.representation_dim, self.latent_dim, self.n_classes,
            self.feature_hidden_dims, self.abduction_hidden_dims
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def compute_loss(self, y_true, logits, location_param, scale_param):
        """
        标准Softmax交叉熵损失函数 - 仅使用logits，不使用尺度参数
        """
        return F.cross_entropy(logits, y_true)

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        # 编码标签
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        self.n_classes = len(self.label_encoder.classes_)
        
        # 重新设置模型以匹配正确的类别数
        self._setup_model_optimizer()
        
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train_encoded).to(self.device)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state_dict = None
        final_epoch_count = self.epochs 

        has_validation = False
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val_encoded).to(self.device)
            has_validation = True
        
        effective_early_stopping_patience = self.early_stopping_patience
        if not has_validation or self.early_stopping_patience is None or self.early_stopping_patience <= 0:
            effective_early_stopping_patience = None
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            final_epoch_count = epoch + 1 
            self.model.train()
            epoch_train_loss = 0
            for batch_X, batch_y_true in train_loader:
                logits, location_param, scale_param = self.model(batch_X)
                loss = self.compute_loss(batch_y_true, logits, location_param, scale_param)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0
            self.history['train_loss'].append(avg_train_loss)
            
            current_val_loss = float('inf')
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    logits_val, loc_param_val, scale_param_val = self.model(X_val_tensor)
                    current_val_loss = self.compute_loss(y_val_tensor, logits_val, loc_param_val, scale_param_val).item()
                self.history['val_loss'].append(current_val_loss)
                
                # Early stopping logic
                if effective_early_stopping_patience is not None:
                    if current_val_loss < best_val_loss - self.early_stopping_min_delta:
                        best_val_loss = current_val_loss
                        patience_counter = 0
                        best_model_state_dict = self.model.state_dict().copy()
                        self.history['best_epoch'] = epoch + 1
                    else:
                        patience_counter += 1
                        if patience_counter >= effective_early_stopping_patience:
                            if verbose >= 1:
                                print(f"Early stopping at epoch {epoch+1}")
                            break
                else:
                    best_model_state_dict = self.model.state_dict().copy()
                    self.history['best_epoch'] = epoch + 1
            else:
                best_model_state_dict = self.model.state_dict().copy()
                self.history['best_epoch'] = epoch + 1
            
            if verbose >= 2 or (verbose == 1 and (epoch+1) % 20 == 0):
                msg = f"Epoch {epoch+1:4d}/{self.epochs}, Train Loss: {avg_train_loss:.6f}"
                if has_validation:
                    msg += f", Val Loss: {current_val_loss:.6f}"
                print(msg)
        
        end_time = time.time()
        self.history['train_time'] = end_time - start_time
        
        # Load best model
        if best_model_state_dict is not None:
            self.model.load_state_dict(best_model_state_dict)
        
        if verbose >= 1:
            print(f"Training completed in {self.history['train_time']:.2f}s, Best epoch: {self.history['best_epoch']}")

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            probs = self.model.predict_proba(X_tensor)
        return probs.cpu().numpy()

    def predict(self, X):
        probs = self.predict_proba(X)
        y_pred_encoded = np.argmax(probs, axis=1)
        return self.label_encoder.inverse_transform(y_pred_encoded)

    def get_params(self, deep=True):
        return {
            'input_dim': self.input_dim,
            'representation_dim': self.representation_dim,
            'latent_dim': self.latent_dim,
            'n_classes': self.n_classes,
            'feature_hidden_dims': self.feature_hidden_dims,
            'abduction_hidden_dims': self.abduction_hidden_dims,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'device': self.device_str,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta
        }

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

class OvRCrossEntropyMLPModel:
    """
    OvR交叉熵MLP模型 - 使用One-vs-Rest策略和交叉熵损失
    仅使用位置参数（location_param），不使用尺度参数（scale_param）
    这是对照CAAC方法的核心基线：相同的OvR策略但不使用尺度参数
    """
    def __init__(self, input_dim, 
                 representation_dim=64, 
                 latent_dim=None,  # 默认等于representation_dim，体现d_latent = d_repr的概念
                 n_classes=2,
                 feature_hidden_dims=[64], 
                 abduction_hidden_dims=[128, 64], 
                 lr=0.001, batch_size=32, epochs=100, device=None,
                 early_stopping_patience=None, early_stopping_min_delta=0.0001):
        
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        # 概念对齐：因果表征维度默认等于特征表征维度 (d_latent = d_repr)
        self.latent_dim = latent_dim if latent_dim is not None else representation_dim
        self.n_classes = n_classes
        self.feature_hidden_dims = feature_hidden_dims
        self.abduction_hidden_dims = abduction_hidden_dims
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device_str = str(device)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        self.label_encoder = LabelEncoder()
        self._setup_model_optimizer()
        self.history = {'train_loss': [], 'val_loss': [], 'train_time': 0, 'best_epoch': 0}
    
    def _setup_model_optimizer(self):
        self.model = UnifiedClassificationNetwork(
            self.input_dim, self.representation_dim, self.latent_dim, self.n_classes,
            self.feature_hidden_dims, self.abduction_hidden_dims
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def compute_loss(self, y_true, logits, location_param, scale_param):
        """
        OvR交叉熵损失函数 - 仅使用logits，不使用尺度参数
        这是与CAAC方法的核心区别：相同OvR策略但不使用尺度参数
        """
        return F.cross_entropy(logits, y_true)

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        # 编码标签
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        self.n_classes = len(self.label_encoder.classes_)
        
        # 重新设置模型以匹配正确的类别数
        self._setup_model_optimizer()
        
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train_encoded).to(self.device)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state_dict = None
        final_epoch_count = self.epochs 

        has_validation = False
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val_encoded).to(self.device)
            has_validation = True
        
        effective_early_stopping_patience = self.early_stopping_patience
        if not has_validation or self.early_stopping_patience is None or self.early_stopping_patience <= 0:
            effective_early_stopping_patience = None
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            final_epoch_count = epoch + 1 
            self.model.train()
            epoch_train_loss = 0
            for batch_X, batch_y_true in train_loader:
                logits, location_param, scale_param = self.model(batch_X)
                loss = self.compute_loss(batch_y_true, logits, location_param, scale_param)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0
            self.history['train_loss'].append(avg_train_loss)
            
            current_val_loss = float('inf')
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    logits_val, loc_param_val, scale_param_val = self.model(X_val_tensor)
                    current_val_loss = self.compute_loss(y_val_tensor, logits_val, loc_param_val, scale_param_val).item()
                self.history['val_loss'].append(current_val_loss)
                
                # Early stopping logic
                if effective_early_stopping_patience is not None:
                    if current_val_loss < best_val_loss - self.early_stopping_min_delta:
                        best_val_loss = current_val_loss
                        patience_counter = 0
                        best_model_state_dict = self.model.state_dict().copy()
                        self.history['best_epoch'] = epoch + 1
                    else:
                        patience_counter += 1
                        if patience_counter >= effective_early_stopping_patience:
                            if verbose >= 1:
                                print(f"Early stopping at epoch {epoch+1}")
                            break
                else:
                    best_model_state_dict = self.model.state_dict().copy()
                    self.history['best_epoch'] = epoch + 1
            else:
                best_model_state_dict = self.model.state_dict().copy()
                self.history['best_epoch'] = epoch + 1
            
            if verbose >= 2 or (verbose == 1 and (epoch+1) % 20 == 0):
                msg = f"Epoch {epoch+1:4d}/{self.epochs}, Train Loss: {avg_train_loss:.6f}"
                if has_validation:
                    msg += f", Val Loss: {current_val_loss:.6f}"
                print(msg)
        
        end_time = time.time()
        self.history['train_time'] = end_time - start_time
        
        # Load best model
        if best_model_state_dict is not None:
            self.model.load_state_dict(best_model_state_dict)
        
        if verbose >= 1:
            print(f"Training completed in {self.history['train_time']:.2f}s, Best epoch: {self.history['best_epoch']}")

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            probs = self.model.predict_proba(X_tensor)
        return probs.cpu().numpy()

    def predict(self, X):
        probs = self.predict_proba(X)
        y_pred_encoded = np.argmax(probs, axis=1)
        return self.label_encoder.inverse_transform(y_pred_encoded)

    def get_params(self, deep=True):
        return {
            'input_dim': self.input_dim,
            'representation_dim': self.representation_dim,
            'latent_dim': self.latent_dim,
            'n_classes': self.n_classes,
            'feature_hidden_dims': self.feature_hidden_dims,
            'abduction_hidden_dims': self.abduction_hidden_dims,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'device': self.device_str,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta
        }

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

class CAACOvRGaussianModel:
    """
    CAAC OvR (Gaussian Distribution) - 使用高斯分布而非柯西分布的CAAC分类模型
    对比研究：相同的CAAC框架但使用不同的概率分布
    
    支持可学习阈值变体：
    - learnable_thresholds=False: 固定阈值 C_k = 0 (默认)
    - learnable_thresholds=True: 可学习阈值 C_k 作为模型参数
    """
    def __init__(self, input_dim, 
                 representation_dim=64, 
                 latent_dim=None,  # 默认等于representation_dim，体现d_latent = d_repr的概念
                 n_classes=2,
                 feature_hidden_dims=[64], 
                 abduction_hidden_dims=[128, 64], 
                 lr=0.001, batch_size=32, epochs=100, device=None,
                 early_stopping_patience=None, early_stopping_min_delta=0.0001,
                 learnable_thresholds=False,  # 新增：是否使用可学习阈值
                 uniqueness_constraint=False,  # 新增：是否启用唯一性约束
                 uniqueness_samples=10,  # 新增：每个样本的采样次数
                 uniqueness_weight=0.1):  # 新增：唯一性约束损失权重
        
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        # 概念对齐：因果表征维度默认等于特征表征维度 (d_latent = d_repr)
        self.latent_dim = latent_dim if latent_dim is not None else representation_dim
        self.n_classes = n_classes
        self.feature_hidden_dims = feature_hidden_dims
        self.abduction_hidden_dims = abduction_hidden_dims
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.learnable_thresholds = learnable_thresholds
        self.uniqueness_constraint = uniqueness_constraint  # 新增：唯一性约束标志
        self.uniqueness_samples = uniqueness_samples  # 新增：采样次数
        self.uniqueness_weight = uniqueness_weight  # 新增：约束损失权重
        self.device_str = str(device)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        self.label_encoder = LabelEncoder()
        self._setup_model_optimizer()
        self.history = {'train_loss': [], 'val_loss': [], 'train_time': 0, 'best_epoch': 0}
    
    def _setup_model_optimizer(self):
        self.model = UnifiedClassificationNetwork(
            self.input_dim, self.representation_dim, self.latent_dim, self.n_classes,
            self.feature_hidden_dims, self.abduction_hidden_dims
        ).to(self.device)
        
        # 如果使用可学习阈值，创建阈值参数
        if self.learnable_thresholds:
            self.thresholds = nn.Parameter(torch.zeros(self.n_classes, device=self.device))
        else:
            self.thresholds = None
            
        # 设置优化器参数，包含模型参数和可能的阈值参数
        params_to_optimize = list(self.model.parameters())
        if self.learnable_thresholds and self.thresholds is not None:
            params_to_optimize.append(self.thresholds)
        self.optimizer = torch.optim.Adam(params_to_optimize, lr=self.lr)
    
    def compute_loss(self, y_true, logits, location_param, scale_param):
        """
        CAAC高斯损失函数：体现因果表征随机变量到Score随机变量的分布变换
        与柯西版本相同的逻辑，但使用高斯分布的线性组合规则和CDF
        """
        batch_size = y_true.size(0)
        n_classes = self.n_classes
        device = y_true.device
        
        # 使用ActionNetwork计算每个类别Score的分布参数 (体现分布类型差异)
        class_locations, class_stds = self.model.action_net.compute_class_distribution_params(
            location_param, scale_param, distribution_type='gaussian'
        )
        
        # 根据设置使用固定阈值或可学习阈值
        if self.learnable_thresholds and self.thresholds is not None:
            thresholds = self.thresholds  # 使用可学习阈值参数
        else:
            thresholds = torch.zeros(n_classes, device=device)  # 使用固定阈值 C_k = 0
        
        # 计算高斯分布CDF：P_k = P(S_k > C_k) = 1 - Φ((C_k - mu) / sigma)
        normalized_thresholds = (thresholds.unsqueeze(0) - class_locations) / class_stds
        
        from torch.distributions import Normal
        standard_normal = Normal(0, 1)
        P_k = 1 - standard_normal.cdf(normalized_thresholds)
        P_k = torch.clamp(P_k, min=1e-7, max=1-1e-7)
        
        # 转换标签为二元标签并计算BCE损失
        y_binary = torch.zeros(batch_size, n_classes, device=device)
        y_binary.scatter_(1, y_true.unsqueeze(1), 1)
        
        bce_loss = -(y_binary * torch.log(P_k) + (1 - y_binary) * torch.log(1 - P_k))
        total_loss = torch.mean(bce_loss)
        
        # 如果启用唯一性约束，添加最大-次大间隔约束损失
        if self.uniqueness_constraint and self.uniqueness_samples > 0:
            # 从高斯分布采样多个潜在向量实例
            n_samples = self.uniqueness_samples
            
            # 生成标准正态分布的随机数
            standard_normal_samples = torch.randn(batch_size, n_samples, self.latent_dim, device=device)
            # 缩放和平移得到目标高斯分布的采样
            # samples: [batch_size, n_samples, latent_dim]
            samples = location_param.unsqueeze(1) + scale_param.unsqueeze(1) * standard_normal_samples
            
            # 计算每个采样的得分
            # scores: [batch_size, n_samples, n_classes]
            scores = self.model.action_net.compute_scores_from_samples(samples)
            
            # 实现最大-次大间隔约束
            # 对每个采样，找到最大和次大的得分
            top2_scores, top2_indices = torch.topk(scores, k=min(2, n_classes), dim=2)
            
            if n_classes > 1:
                max_scores = top2_scores[:, :, 0]  # [batch_size, n_samples]
                second_max_scores = top2_scores[:, :, 1]  # [batch_size, n_samples]
                max_indices = top2_indices[:, :, 0]  # [batch_size, n_samples]
                second_max_indices = top2_indices[:, :, 1]  # [batch_size, n_samples]
                
                # 获取对应的阈值
                max_thresholds = thresholds[max_indices]  # [batch_size, n_samples]
                second_max_thresholds = thresholds[second_max_indices]  # [batch_size, n_samples]
                
                # 计算约束损失：
                # 1. 最大得分应该超过其阈值：max(0, threshold_max - score_max)
                # 2. 次大得分应该低于其阈值：max(0, score_second - threshold_second)
                max_violation = F.relu(max_thresholds - max_scores)
                second_violation = F.relu(second_max_scores - second_max_thresholds)
                
                # 计算平均违反损失（而不是累加）
                total_violations = max_violation + second_violation
                uniqueness_loss = torch.mean(total_violations)  # 对所有采样取平均
                
                # 添加权重并加到总损失中
                total_loss = total_loss + self.uniqueness_weight * uniqueness_loss
        
        return total_loss

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        # 编码标签
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        self.n_classes = len(self.label_encoder.classes_)
        
        # 重新设置模型以匹配正确的类别数
        self._setup_model_optimizer()
        
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train_encoded).to(self.device)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state_dict = None
        final_epoch_count = self.epochs 

        has_validation = False
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val_encoded).to(self.device)
            has_validation = True
        
        effective_early_stopping_patience = self.early_stopping_patience
        if not has_validation or self.early_stopping_patience is None or self.early_stopping_patience <= 0:
            effective_early_stopping_patience = None
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            final_epoch_count = epoch + 1 
            self.model.train()
            epoch_train_loss = 0
            for batch_X, batch_y_true in train_loader:
                logits, location_param, scale_param = self.model(batch_X)
                loss = self.compute_loss(batch_y_true, logits, location_param, scale_param)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0
            self.history['train_loss'].append(avg_train_loss)
            
            current_val_loss = float('inf')
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    logits_val, loc_param_val, scale_param_val = self.model(X_val_tensor)
                    current_val_loss = self.compute_loss(y_val_tensor, logits_val, loc_param_val, scale_param_val).item()
                self.history['val_loss'].append(current_val_loss)
                
                # Early stopping logic
                if effective_early_stopping_patience is not None:
                    if current_val_loss < best_val_loss - self.early_stopping_min_delta:
                        best_val_loss = current_val_loss
                        patience_counter = 0
                        best_model_state_dict = self.model.state_dict().copy()
                        self.history['best_epoch'] = epoch + 1
                    else:
                        patience_counter += 1
                        if patience_counter >= effective_early_stopping_patience:
                            if verbose >= 1:
                                print(f"Early stopping at epoch {epoch+1}")
                            break
                else:
                    best_model_state_dict = self.model.state_dict().copy()
                    self.history['best_epoch'] = epoch + 1
            else:
                best_model_state_dict = self.model.state_dict().copy()
                self.history['best_epoch'] = epoch + 1
            
            if verbose >= 2 or (verbose == 1 and (epoch+1) % 20 == 0):
                msg = f"Epoch {epoch+1:4d}/{self.epochs}, Train Loss: {avg_train_loss:.6f}"
                if has_validation:
                    msg += f", Val Loss: {current_val_loss:.6f}"
                print(msg)
        
        end_time = time.time()
        self.history['train_time'] = end_time - start_time
        
        # Load best model
        if best_model_state_dict is not None:
            self.model.load_state_dict(best_model_state_dict)
        
        if verbose >= 1:
            print(f"Training completed in {self.history['train_time']:.2f}s, Best epoch: {self.history['best_epoch']}")

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            probs = self.model.predict_proba(X_tensor)
        return probs.cpu().numpy()

    def predict(self, X):
        probs = self.predict_proba(X)
        y_pred_encoded = np.argmax(probs, axis=1)
        return self.label_encoder.inverse_transform(y_pred_encoded)

    def get_params(self, deep=True):
        return {
            'input_dim': self.input_dim,
            'representation_dim': self.representation_dim,
            'latent_dim': self.latent_dim,
            'n_classes': self.n_classes,
            'feature_hidden_dims': self.feature_hidden_dims,
            'abduction_hidden_dims': self.abduction_hidden_dims,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'device': self.device_str,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta,
            'learnable_thresholds': self.learnable_thresholds,
            'uniqueness_constraint': self.uniqueness_constraint,
            'uniqueness_samples': self.uniqueness_samples,
            'uniqueness_weight': self.uniqueness_weight
        }

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

class CrammerSingerMLPModel:
    """
    方法五：MLP (Crammer & Singer 多分类铰链损失) - CrammerSingerMLPModel
    
    核心思想：
    - 采用Crammer & Singer多分类铰链损失，基于margin最大化原理
    - 要求正确类别的分数比任何错误类别的分数至少高出固定margin值1
    - 仅使用位置参数，通过几何margin优化进行训练
    - 基于统计学习理论和VC维理论，泛化保证较强
    """
    
    def __init__(self, input_dim, 
                 representation_dim=64, 
                 latent_dim=None,  # 默认等于representation_dim，体现d_latent = d_repr的概念
                 n_classes=2,
                 feature_hidden_dims=[64], 
                 abduction_hidden_dims=[128, 64], 
                 lr=0.001, batch_size=32, epochs=100, device=None,
                 early_stopping_patience=None, early_stopping_min_delta=0.0001):
        
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.latent_dim = latent_dim if latent_dim is not None else representation_dim
        self.n_classes = n_classes
        self.feature_hidden_dims = feature_hidden_dims
        self.abduction_hidden_dims = abduction_hidden_dims
        
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device_str = str(device)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        self.label_encoder = LabelEncoder()
        self._setup_model_optimizer()
        self.history = {'train_loss': [], 'val_loss': [], 'train_time': 0, 'best_epoch': 0}
    
    def _setup_model_optimizer(self):
        self.model = UnifiedClassificationNetwork(
            self.input_dim, self.representation_dim, self.latent_dim, self.n_classes,
            self.feature_hidden_dims, self.abduction_hidden_dims
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def compute_loss(self, y_true, logits, location_param, scale_param):
        """
        Crammer & Singer 多分类铰链损失
        
        数学公式：
        L = (1/|B|) * Σ_i max(0, max_{k≠y_i}(S_k^(i) - S_{y_i}^(i) + 1))
        
        其中：
        - S_k^(i) 是第i个样本对类别k的分数（logits）
        - y_i 是第i个样本的真实标签
        - margin = 1 是固定的间隔要求
        
        物理含义：
        - 要求正确类别分数比最高错误类别分数至少高出1个单位
        - 当满足margin条件时损失为0，体现稀疏性
        - 基于几何margin最大化，提供更好的泛化边界
        
        注意：仅使用logits（基于位置参数），完全忽略尺度参数scale_param
        """
        batch_size, n_classes = logits.shape
        
        # 第一步：获取正确类别的分数 S_y
        correct_scores = logits.gather(1, y_true.unsqueeze(1)).squeeze(1)  # [batch_size]
        
        # 第二步：计算 margin_violation = S_k - S_y + 1 for all k ≠ y
        margins = logits - correct_scores.unsqueeze(1) + 1.0  # [batch_size, n_classes]
        
        # 第三步：排除正确类别自身（避免 S_y - S_y + 1 = 1）
        margins.scatter_(1, y_true.unsqueeze(1), 0)
        
        # 第四步：找到最大的margin违反 max_{k≠y}(S_k - S_y + 1)
        max_margins, _ = margins.max(dim=1)  # [batch_size]
        
        # 第五步：应用ReLU得到铰链损失 max(0, margin_violation)
        hinge_loss = F.relu(max_margins)
        
        return hinge_loss.mean()

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        # 编码标签
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        self.n_classes = len(self.label_encoder.classes_)
        
        # 重新设置模型以匹配正确的类别数
        self._setup_model_optimizer()
        
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train_encoded).to(self.device)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state_dict = None
        final_epoch_count = self.epochs 

        has_validation = False
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val_encoded).to(self.device)
            has_validation = True
        
        effective_early_stopping_patience = self.early_stopping_patience
        if not has_validation or self.early_stopping_patience is None or self.early_stopping_patience <= 0:
            effective_early_stopping_patience = None
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            final_epoch_count = epoch + 1 
            self.model.train()
            epoch_train_loss = 0
            active_samples_count = 0  # 统计违反margin的样本数量
            
            for batch_X, batch_y_true in train_loader:
                logits, location_param, scale_param = self.model(batch_X)
                loss = self.compute_loss(batch_y_true, logits, location_param, scale_param)
                
                # 统计active samples（违反margin的样本）
                with torch.no_grad():
                    correct_scores = logits.gather(1, batch_y_true.unsqueeze(1)).squeeze(1)
                    margins = logits - correct_scores.unsqueeze(1) + 1.0
                    margins.scatter_(1, batch_y_true.unsqueeze(1), 0)
                    max_margins, _ = margins.max(dim=1)
                    active_samples_count += (max_margins > 0).sum().item()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0
            self.history['train_loss'].append(avg_train_loss)
            
            current_val_loss = float('inf')
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    logits_val, loc_param_val, scale_param_val = self.model(X_val_tensor)
                    current_val_loss = self.compute_loss(y_val_tensor, logits_val, loc_param_val, scale_param_val).item()
                self.history['val_loss'].append(current_val_loss)
                
                # Early stopping logic
                if effective_early_stopping_patience is not None:
                    if current_val_loss < best_val_loss - self.early_stopping_min_delta:
                        best_val_loss = current_val_loss
                        patience_counter = 0
                        best_model_state_dict = self.model.state_dict().copy()
                        self.history['best_epoch'] = epoch + 1
                    else:
                        patience_counter += 1
                        if patience_counter >= effective_early_stopping_patience:
                            if verbose >= 1:
                                print(f"Early stopping at epoch {epoch+1}")
                            break
                else:
                    best_model_state_dict = self.model.state_dict().copy()
                    self.history['best_epoch'] = epoch + 1
            else:
                best_model_state_dict = self.model.state_dict().copy()
                self.history['best_epoch'] = epoch + 1
            
            if verbose >= 2 or (verbose == 1 and (epoch+1) % 20 == 0):
                msg = f"Epoch {epoch+1:4d}/{self.epochs}, Train Loss: {avg_train_loss:.6f}"
                if has_validation:
                    msg += f", Val Loss: {current_val_loss:.6f}"
                # 显示active samples比例，体现稀疏性特点
                total_samples = len(X_train_tensor)
                active_ratio = active_samples_count / total_samples if total_samples > 0 else 0
                msg += f", Active: {active_ratio:.1%}"
                print(msg)
        
        end_time = time.time()
        self.history['train_time'] = end_time - start_time
        
        # Load best model
        if best_model_state_dict is not None:
            self.model.load_state_dict(best_model_state_dict)
        
        if verbose >= 1:
            print(f"Training completed in {self.history['train_time']:.2f}s, Best epoch: {self.history['best_epoch']}")

    def predict_proba(self, X):
        """
        预测概率输出
        
        注意：铰链损失训练时不直接依赖概率，此方法仅为API兼容性
        通过Softmax将类别分数转换为概率分布（可选的概率化输出）
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            logits, _, _ = self.model(X_tensor)
            # 使用Softmax将分数转为概率（仅为兼容性，训练时不依赖此概率）
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def predict(self, X):
        """
        预测类别标签
        
        基于argmax决策规则：ŷ = argmax_k S_k
        这是铰链损失的原生预测方式，不依赖概率计算
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            logits, _, _ = self.model(X_tensor)
            # 直接基于分数进行argmax预测（铰链损失的原生方式）
            y_pred_encoded = torch.argmax(logits, dim=1)
        return self.label_encoder.inverse_transform(y_pred_encoded.cpu().numpy())

    def get_params(self, deep=True):
        return {
            'input_dim': self.input_dim,
            'representation_dim': self.representation_dim,
            'latent_dim': self.latent_dim,
            'n_classes': self.n_classes,
            'feature_hidden_dims': self.feature_hidden_dims,
            'abduction_hidden_dims': self.abduction_hidden_dims,
            
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'device': self.device_str,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta
        }

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

