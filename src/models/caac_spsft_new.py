"""
新的 CAAC-SPSFT 模型实现

基于设计文档重新实现的CAAC-SPSFT模型和包装器。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader
from scipy.stats import norm

class CAAC_SPSFT(nn.Module):
    """
    CAAC-SPSFT (Cauchy Abduction Action Classification - Stochastic Pathway Selection with Fixed Thresholds) 模型。
    柯西溯因行动分类 - 带固定阈值的随机路径选择模型。
    """
    def __init__(self, input_dim, encoder_hidden_dim, encoder_output_dim, d_c, num_paths, num_classes, init_config=None):
        """
        初始化 CAAC-SPSFT 模型。

        参数:
            input_dim (int): 输入特征 x 的维度。
            encoder_hidden_dim (int): MLP 编码器的隐藏层维度。
            encoder_output_dim (int): 编码器 h(x) 的输出维度。
            d_c (int): 因果表征 C 的维度。
            num_paths (int): 并行解释路径的数量 (K_paths)。
            num_classes (int): 目标类别的数量 (N_cl)。
            init_config (dict, optional): 参数初始化配置。默认为 None。
        """
        super().__init__()
        
        assert num_classes >= 2, "类别数量必须至少为2。"

        self.input_dim = input_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_output_dim = encoder_output_dim
        self.d_c = d_c
        self.num_paths = num_paths
        self.num_classes = num_classes

        if init_config is None:
            init_config = {}
        self.init_config = init_config

        # 1. 编码器: x -> h(x)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, encoder_output_dim)
        )

        # 2. 因果表征参数: h(x) -> mu_C(h), gamma_C(h)
        self.fc_mu_C = nn.Linear(encoder_output_dim, d_c)
        self.fc_gamma_C_raw = nn.Linear(encoder_output_dim, d_c) # 原始 gamma, 应用 softplus 保证其为正

        # 3. 路径特定的得分参数 (可学习, 对于给定模型是固定的)
        # 对于 mu_S_j(h) = W_mu^(j)^T mu_C(h) + b_mu^(j)
        self.W_mu_paths = nn.Parameter(torch.Tensor(num_paths, d_c)) # 形状: (K_paths, d_c)
        self.b_mu_paths = nn.Parameter(torch.Tensor(num_paths))      # 形状: (K_paths)

        # 对于 gamma_S_j(h) = sum_i |W_gamma_i^(j)| gamma_C_i(h) + gamma_epsilon0^(j)
        self.W_gamma_paths = nn.Parameter(torch.Tensor(num_paths, d_c)) # 形状: (K_paths, d_c)
        self.raw_gamma_epsilon0_paths = nn.Parameter(torch.Tensor(num_paths)) # 原始值, 应用 exp 保证其为正

        # 4. 路径选择概率 (可学习的固定 logits l_j)
        self.path_logits = nn.Parameter(torch.Tensor(num_paths)) # l_j

        # 5. 固定决策阈值 (可学习的原始参数)
        # theta_1^* = raw_theta_1_star
        # delta_k^* = exp(raw_delta_k_star) for k=2,...,N_cl-1
        # theta_k^* = theta_{k-1}^* + delta_k^*
        self.raw_theta_1_star = nn.Parameter(torch.Tensor(1))
        if num_classes > 2:
            self.raw_delta_k_star = nn.Parameter(torch.Tensor(num_classes - 2))
        
        self.init_weights()

    def init_weights(self):
        """根据提供的 init_config 或默认值初始化模型参数。"""
        # 编码器权重
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.fc_mu_C.weight)
        nn.init.zeros_(self.fc_mu_C.bias)
        nn.init.xavier_uniform_(self.fc_gamma_C_raw.weight)
        nn.init.zeros_(self.fc_gamma_C_raw.bias)

        # 路径特定的得分参数
        std_W_mu = self.init_config.get('std_W_mu', 1.0 / np.sqrt(self.d_c))
        nn.init.normal_(self.W_mu_paths, mean=0.0, std=std_W_mu)
        
        delta_b_mu = self.init_config.get('delta_b_mu', 0.5)
        self.b_mu_paths.data = torch.arange(self.num_paths, dtype=torch.float32) * delta_b_mu
        
        W_gamma_abs_max = self.init_config.get('W_gamma_abs_max', 0.1)
        nn.init.uniform_(self.W_gamma_paths, -W_gamma_abs_max, W_gamma_abs_max)
        
        # 初始化 raw_gamma_epsilon0_paths 使得 exp(raw_gamma_epsilon0_paths) 大约在 1.0
        # 所以, raw_gamma_epsilon0_paths 大约在 log(1.0) = 0.0
        mean_raw_gamma_eps = self.init_config.get('mean_raw_gamma_epsilon0', np.log(1.0))
        std_raw_gamma_eps = self.init_config.get('std_raw_gamma_epsilon0', 0.1)
        nn.init.normal_(self.raw_gamma_epsilon0_paths, mean=mean_raw_gamma_eps, std=std_raw_gamma_eps)

        # 路径 logits (l_j) - 初始化为0以获得均匀的 pi_j
        nn.init.zeros_(self.path_logits)

        # 阈值参数
        self._initialize_threshold_params()

    def _initialize_threshold_params(self):
        """为固定阈值初始化 raw_theta_1_star 和 raw_delta_k_star。"""
        # 初始策略: 使用更稳定的阈值初始化
        # 使用简单的等间距分布而不是tan函数，避免极值
        
        # 使用标准正态分布的分位数作为初始阈值
        k_vals = torch.arange(1, self.num_classes, dtype=torch.float32) # 1, 2, ..., N_cl-1
        F_k = k_vals / self.num_classes
        
        # 使用标准正态分布的分位数，限制在合理范围内
        initial_thetas = torch.tensor([norm.ppf(f.item()) for f in F_k], dtype=torch.float32)
        # 限制阈值范围，避免极值
        initial_thetas = torch.clamp(initial_thetas, -3.0, 3.0)

        self.raw_theta_1_star.data.fill_(initial_thetas[0])
        
        if self.num_classes > 2:
            # initial_deltas_k = theta_k^{*(init)} - theta_{k-1}^{*(init)} for k=2,...,N_cl-1
            initial_deltas = initial_thetas[1:] - initial_thetas[:-1] # 形状: (N_cl - 2)
            # 确保 deltas 有最小值，避免log(0)
            initial_deltas = torch.clamp(initial_deltas, min=0.1)
            # raw_delta_k_star = log(initial_deltas_k)
            self.raw_delta_k_star.data.copy_(torch.log(initial_deltas)) # 现在安全了，因为initial_deltas >= 0.1

    def get_thresholds(self):
        """
        从原始参数计算实际的决策阈值 theta_k^*。
        确保 theta_1^* < theta_2^* < ... < theta_{N_cl-1}^*。
        返回:
            torch.Tensor:排序后的阈值，形状 (num_classes - 1)。
        """
        if self.num_classes == 1: # 根据断言不应该发生
             return torch.empty(0, device=self.raw_theta_1_star.device)

        thresholds = torch.empty(self.num_classes - 1, device=self.raw_theta_1_star.device)
        # 限制第一个阈值的范围
        thresholds[0] = torch.clamp(self.raw_theta_1_star, -5.0, 5.0)
        
        if self.num_classes > 2:
            deltas_exp = torch.exp(torch.clamp(self.raw_delta_k_star, -5.0, 5.0)) # 限制exp输入范围
            # 确保deltas有最小值
            deltas_exp = torch.clamp(deltas_exp, min=0.01)
            # thresholds[1] = raw_theta_1_star + delta_2^*
            # thresholds[2] = raw_theta_1_star + delta_2^* + delta_3^*
            # ...
            thresholds[1:] = thresholds[0] + torch.cumsum(deltas_exp, dim=0)
        return thresholds

    def cauchy_cdf(self, s, mu, gamma):
        """
        计算柯西分布的累积分布函数 (CDF)。
        F_S(s; mu, gamma) = 0.5 + (1/pi) * arctan((s - mu) / gamma)
        参数:
            s (torch.Tensor): 计算 CDF 的值。
            mu (torch.Tensor): 位置参数。
            gamma (torch.Tensor): 尺度参数 (必须 > 0)。
        返回:
            torch.Tensor: CDF 值。
        """
        # 确保gamma有最小值，避免除零
        gamma_safe = torch.clamp(gamma, min=1e-6)
        # 限制atan的输入范围，避免数值不稳定
        atan_input = (s - mu) / gamma_safe
        atan_input = torch.clamp(atan_input, -1e6, 1e6)
        return 0.5 + (1.0 / torch.pi) * torch.atan(atan_input)

    def forward(self, x):
        """
        执行 CAAC-SPSFT 模型的前向传播。
        参数:
            x (torch.Tensor): 输入特征，形状 (batch_size, input_dim)。
        返回:
            torch.Tensor: 预测的类别概率，形状 (batch_size, num_classes)。
        """
        batch_size = x.shape[0]

        # 1. 编码输入: x -> h(x)
        h_x = self.encoder(x) # (batch_size, encoder_output_dim)

        # 2. 生成因果表征参数: mu_C(h), gamma_C(h)
        mu_C = self.fc_mu_C(h_x) # (batch_size, d_c)
        gamma_C_raw = self.fc_gamma_C_raw(h_x)
        # 使用更稳定的softplus，并限制最大值
        gamma_C = F.softplus(gamma_C_raw) # 确保 gamma_C > 0; (batch_size, d_c)
        gamma_C = torch.clamp(gamma_C, min=1e-4, max=10.0)  # 限制范围

        # 3. 计算路径特定的得分参数: mu_S_j(h), gamma_S_j(h)
        # mu_S_j(h) = W_mu^(j)^T mu_C(h) + b_mu^(j)
        # W_mu_paths: (num_paths, d_c), mu_C: (batch_size, d_c)
        # mu_S_paths = torch.einsum('bd,pd->bp', mu_C, self.W_mu_paths) + self.b_mu_paths.unsqueeze(0) # einsum 写法
        mu_S_paths = mu_C @ self.W_mu_paths.T + self.b_mu_paths # (batch_size, num_paths)
        # 限制位置参数的范围
        mu_S_paths = torch.clamp(mu_S_paths, -10.0, 10.0)

        # gamma_S_j(h) = sum_i |W_gamma_i^(j)| gamma_C_i(h) + gamma_epsilon0^(j)
        # W_gamma_paths: (num_paths, d_c), gamma_C: (batch_size, d_c)
        abs_W_gamma_paths = torch.abs(self.W_gamma_paths) # (num_paths, d_c)
        # term1 = torch.einsum('bd,pd->bp', gamma_C, abs_W_gamma_paths) # einsum 写法
        term1 = gamma_C @ abs_W_gamma_paths.T # (batch_size, num_paths)
        
        # 限制exp的输入范围
        raw_gamma_clamped = torch.clamp(self.raw_gamma_epsilon0_paths, -5.0, 5.0)
        gamma_epsilon0_paths = torch.exp(raw_gamma_clamped) # 确保 > 0; (num_paths)
        gamma_S_paths = term1 + gamma_epsilon0_paths.unsqueeze(0) # (batch_size, num_paths)
        # 确保尺度参数有最小值
        gamma_S_paths = torch.clamp(gamma_S_paths, min=1e-4, max=10.0)

        # 4. 路径选择概率 pi_j (对于模型是固定的, 不依赖于批次)
        # 限制logits范围，使用温度调节
        path_logits_clamped = torch.clamp(self.path_logits, -10.0, 10.0)
        pi_paths = torch.softmax(path_logits_clamped, dim=0) # (num_paths)

        # 5. 获取固定决策阈值 theta_k^*
        # thresholds_star: (num_classes - 1), 例如, 对于3个类别是 [theta_1^*, theta_2^*]
        thresholds_star = self.get_thresholds() 
        
        # 使用 -inf 和 +inf 扩展阈值以便于 CDF 计算范围
        # full_thresholds: 对于3个类别是 [-inf, theta_1^*, theta_2^*, +inf]
        # 形状: (num_classes + 1)
        # 使用有限的大数值而不是无穷大，避免数值问题
        theta_0_star = torch.tensor([-1e6], device=x.device)
        theta_Ncl_star = torch.tensor([1e6], device=x.device)
        full_thresholds = torch.cat([theta_0_star, thresholds_star, theta_Ncl_star], dim=0)

        # 6. 计算每个路径和类别的 P(Y=k | M=j, x)
        # 扩展 mu_S_paths 和 gamma_S_paths 以便与阈值进行广播
        mu_S_expanded = mu_S_paths.unsqueeze(2)     # (batch_size, num_paths, 1)
        gamma_S_expanded = gamma_S_paths.unsqueeze(2) # (batch_size, num_paths, 1)
        
        # full_thresholds 扩展以便广播: (1, 1, num_classes + 1)
        # cdf_at_thresholds 形状: (batch_size, num_paths, num_classes + 1)
        cdf_at_thresholds = self.cauchy_cdf(full_thresholds.view(1, 1, -1), mu_S_expanded, gamma_S_expanded)
        
        # P(Y=k | M=j, x) = F_Sj(theta_k^*) - F_Sj(theta_{k-1}^*)
        # probs_path_class 形状: (batch_size, num_paths, num_classes)
        probs_path_class = cdf_at_thresholds[..., 1:] - cdf_at_thresholds[..., :-1]
        
        # 确保概率为正并且有最小值
        probs_path_class = torch.clamp(probs_path_class, min=1e-8, max=1.0)
        
        # 确保每个路径的概率总和为1 (由于潜在的数值问题)
        path_class_sum = torch.sum(probs_path_class, dim=2, keepdim=True)
        path_class_sum = torch.clamp(path_class_sum, min=1e-8)
        probs_path_class = probs_path_class / path_class_sum

        # 7. 计算最终混合概率 P(Y=k | x)
        # P(Y=k|x) = sum_j pi_j * P(Y=k | M=j, x)
        # pi_paths: (num_paths) -> (1, num_paths, 1) 以便广播
        # probs_class 形状: (batch_size, num_classes)
        probs_class = torch.sum(probs_path_class * pi_paths.view(1, -1, 1), dim=1)
        
        # 确保最终概率为正并且有最小值
        probs_class = torch.clamp(probs_class, min=1e-8, max=1.0)
        
        # 确保最终概率总和为1 (由于潜在的数值问题)
        final_sum = torch.sum(probs_class, dim=1, keepdim=True)
        final_sum = torch.clamp(final_sum, min=1e-8)
        probs_class = probs_class / final_sum

        return probs_class

    def compute_nll_loss(self, pred_probs, true_labels):
        """
        计算负对数似然 (NLL) 损失。
        NLL = - sum_i log P(Y=y_i | x_i) / N
        参数:
            pred_probs (torch.Tensor): 来自 forward() 的预测类别概率，形状 (batch_size, num_classes)。
            true_labels (torch.Tensor): 真实的类别标签 (整数)，形状 (batch_size)。
        返回:
            torch.Tensor: 标量 NLL 损失。
        """
        # 为每个样本选择真实类别的概率
        # pred_probs[i, true_labels[i]]
        p_y_i = pred_probs[torch.arange(pred_probs.size(0)), true_labels]
        
        # 确保概率在合理范围内，添加epsilon以保证数值稳定性
        p_y_i = torch.clamp(p_y_i, min=1e-8, max=1.0)
        
        # 计算log，现在是安全的
        log_p_y_i = torch.log(p_y_i)
        
        nll = -log_p_y_i.mean()
        
        # 检查NaN并返回有效值
        if torch.isnan(nll):
            print("Warning: NaN detected in NLL loss, returning large finite value")
            return torch.tensor(10.0, device=nll.device, requires_grad=True)
        
        return nll


class NewCAACModelWrapper:
    """
    新的 CAAC-SPSFT 模型包装类
    
    使用设计文档中的实现，封装训练、验证和预测功能。
    """
    def __init__(self, input_dim, 
                 encoder_hidden_dim=64,
                 encoder_output_dim=32,
                 d_c=16,
                 n_paths=2,
                 n_classes=2,
                 lr=0.001, 
                 batch_size=32, 
                 epochs=100, 
                 device=None,
                 early_stopping_patience=None, 
                 early_stopping_min_delta=0.0001,
                 init_config=None,
                 **kwargs):  # 接受额外的参数以保持兼容性
        
        self.input_dim = input_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_output_dim = encoder_output_dim
        self.d_c = d_c
        self.n_paths = n_paths
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.init_config = init_config
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self._setup_model_optimizer()
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 
                        'train_time': 0, 'best_epoch': 0}
    
    def _setup_model_optimizer(self):
        self.model = CAAC_SPSFT(
            input_dim=self.input_dim,
            encoder_hidden_dim=self.encoder_hidden_dim,
            encoder_output_dim=self.encoder_output_dim,
            d_c=self.d_c,
            num_paths=self.n_paths,
            num_classes=self.n_classes,
            init_config=self.init_config
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
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
                class_probs = self.model(batch_X)
                loss = self.model.compute_nll_loss(class_probs, batch_y)
                
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
                    val_class_probs = self.model(X_val_tensor)
                    val_loss = self.model.compute_nll_loss(val_class_probs, y_val_tensor)
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
            class_probs = self.model(X_tensor)
        return class_probs.cpu().numpy()
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1) 