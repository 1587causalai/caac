import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        # 初始策略: 标准柯西分布的均匀分位数
        # theta_k^{*(init)} = tan(pi * ( (k_val)/N_cl - 0.5 )) for k_val = 1, ..., N_cl-1
        
        k_vals = torch.arange(1, self.num_classes, dtype=torch.float32) # 1, 2, ..., N_cl-1
        F_k = k_vals / self.num_classes
        initial_thetas = torch.tan(torch.pi * (F_k - 0.5)) # 形状: (N_cl - 1)

        self.raw_theta_1_star.data.fill_(initial_thetas[0])
        
        if self.num_classes > 2:
            # initial_deltas_k = theta_k^{*(init)} - theta_{k-1}^{*(init)} for k=2,...,N_cl-1
            initial_deltas = initial_thetas[1:] - initial_thetas[:-1] # 形状: (N_cl - 2)
            # 在取对数前确保 deltas 是正的 (由于 tan 的单调性，这应该是成立的)
            # raw_delta_k_star = log(initial_deltas_k)
            self.raw_delta_k_star.data.copy_(torch.log(initial_deltas + 1e-9)) # 添加 epsilon 以保证稳定性

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
        thresholds[0] = self.raw_theta_1_star
        
        if self.num_classes > 2:
            deltas_exp = torch.exp(self.raw_delta_k_star) # 这些是 delta_2^*, delta_3^*, ...
            # thresholds[1] = raw_theta_1_star + delta_2^*
            # thresholds[2] = raw_theta_1_star + delta_2^* + delta_3^*
            # ...
            thresholds[1:] = self.raw_theta_1_star + torch.cumsum(deltas_exp, dim=0)
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
        return 0.5 + (1.0 / torch.pi) * torch.atan((s - mu) / (gamma + 1e-9)) # 对 gamma 添加 epsilon 以保证稳定性

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
        gamma_C = F.softplus(gamma_C_raw) # 确保 gamma_C > 0; (batch_size, d_c)

        # 3. 计算路径特定的得分参数: mu_S_j(h), gamma_S_j(h)
        # mu_S_j(h) = W_mu^(j)^T mu_C(h) + b_mu^(j)
        # W_mu_paths: (num_paths, d_c), mu_C: (batch_size, d_c)
        # mu_S_paths = torch.einsum('bd,pd->bp', mu_C, self.W_mu_paths) + self.b_mu_paths.unsqueeze(0) # einsum 写法
        mu_S_paths = mu_C @ self.W_mu_paths.T + self.b_mu_paths # (batch_size, num_paths)

        # gamma_S_j(h) = sum_i |W_gamma_i^(j)| gamma_C_i(h) + gamma_epsilon0^(j)
        # W_gamma_paths: (num_paths, d_c), gamma_C: (batch_size, d_c)
        abs_W_gamma_paths = torch.abs(self.W_gamma_paths) # (num_paths, d_c)
        # term1 = torch.einsum('bd,pd->bp', gamma_C, abs_W_gamma_paths) # einsum 写法
        term1 = gamma_C @ abs_W_gamma_paths.T # (batch_size, num_paths)
        
        gamma_epsilon0_paths = torch.exp(self.raw_gamma_epsilon0_paths) # 确保 > 0; (num_paths)
        gamma_S_paths = term1 + gamma_epsilon0_paths.unsqueeze(0) # (batch_size, num_paths)

        # 4. 路径选择概率 pi_j (对于模型是固定的, 不依赖于批次)
        pi_paths = torch.softmax(self.path_logits, dim=0) # (num_paths)

        # 5. 获取固定决策阈值 theta_k^*
        # thresholds_star: (num_classes - 1), 例如, 对于3个类别是 [theta_1^*, theta_2^*]
        thresholds_star = self.get_thresholds() 
        
        # 使用 -inf 和 +inf 扩展阈值以便于 CDF 计算范围
        # full_thresholds: 对于3个类别是 [-inf, theta_1^*, theta_2^*, +inf]
        # 形状: (num_classes + 1)
        theta_0_star = torch.tensor([float('-inf')], device=x.device)
        theta_Ncl_star = torch.tensor([float('inf')], device=x.device)
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
        
        # 确保每个路径的概率总和为1 (由于潜在的数值问题)
        probs_path_class = probs_path_class / (torch.sum(probs_path_class, dim=2, keepdim=True) + 1e-9)

        # 7. 计算最终混合概率 P(Y=k | x)
        # P(Y=k|x) = sum_j pi_j * P(Y=k | M=j, x)
        # pi_paths: (num_paths) -> (1, num_paths, 1) 以便广播
        # probs_class 形状: (batch_size, num_classes)
        probs_class = torch.sum(probs_path_class * pi_paths.view(1, -1, 1), dim=1)
        
        # 确保最终概率总和为1 (由于潜在的数值问题)
        probs_class = probs_class / (torch.sum(probs_class, dim=1, keepdim=True) + 1e-9)

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
        
        # 添加 epsilon 以保证数值稳定性 (避免 log(0))
        log_p_y_i = torch.log(p_y_i + 1e-9)
        
        nll = -log_p_y_i.mean()
        return nll

if __name__ == '__main__':
    # 示例用法
    batch_size = 10
    input_dim = 20
    encoder_hidden_dim = 64
    encoder_output_dim = 32 # h(x) 的维度
    d_c = 16                # 因果表征 C 的维度
    num_paths = 5           # 解释路径的数量 (K_paths)
    num_classes = 3         # 目标类别的数量 (N_cl)

    # 实例化模型
    model = CAAC_SPSFT(input_dim, encoder_hidden_dim, encoder_output_dim, d_c, num_paths, num_classes)
    print("CAAC-SPSFT 模型已实例化:")
    # print(model) # 取消注释以查看模型结构

    # 创建虚拟输入数据
    dummy_x = torch.randn(batch_size, input_dim)
    dummy_labels = torch.randint(0, num_classes, (batch_size,))

    # 执行前向传播
    try:
        pred_probs = model(dummy_x)
        print(f"\n输入形状: {dummy_x.shape}")
        print(f"预测概率形状: {pred_probs.shape}")
        print(f"预测概率示例 (前2个样本):\n{pred_probs[:2]}")
        print(f"第一个样本的概率总和: {torch.sum(pred_probs[0])}")

        # 计算损失
        loss = model.compute_nll_loss(pred_probs, dummy_labels)
        print(f"\n计算得到的 NLL 损失: {loss.item()}")

        # 检查阈值
        thresholds = model.get_thresholds()
        print(f"\n学习到的阈值 (theta_k^*): {thresholds.data}")
        if num_classes > 2:
             print(f"   原始 theta_1_star: {model.raw_theta_1_star.data}")
             print(f"   原始 delta_k_star: {model.raw_delta_k_star.data}")
             print(f"   exp(原始 delta_k_star): {torch.exp(model.raw_delta_k_star.data)}")
        else:
             print(f"   原始 theta_1_star: {model.raw_theta_1_star.data}")


        # 检查路径概率
        path_probs = torch.softmax(model.path_logits, dim=0)
        print(f"\n路径选择概率 (pi_j): {path_probs.data}")
        print(f"路径概率总和: {torch.sum(path_probs)}")


    except Exception as e:
        print(f"\n示例用法中发生错误: {e}")
        import traceback
        traceback.print_exc()

    # --- 使用 init_config 的示例 ---
    print("\n--- 使用 init_config 的示例 ---")
    custom_init_config = {
        'std_W_mu': 0.1,
        'delta_b_mu': 0.2,
        'W_gamma_abs_max': 0.05,
        'mean_raw_gamma_epsilon0': np.log(0.5), # 目标 gamma_epsilon0 约等于 0.5
        'std_raw_gamma_epsilon0': 0.05
    }
    model_custom_init = CAAC_SPSFT(input_dim, encoder_hidden_dim, encoder_output_dim, 
                                   d_c, num_paths, num_classes, init_config=custom_init_config)
    
    print("具有自定义初始化配置的模型已实例化。")
    # 您可以检查其参数或运行前向传播
    # custom_pred_probs = model_custom_init(dummy_x)
    # print(f"自定义初始化后 W_mu_paths 示例 (第一个路径):\n{model_custom_init.W_mu_paths.data[0,:5]}")
    # print(f"自定义初始化后 b_mu_paths 示例:\n{model_custom_init.b_mu_paths.data[:5]}")

