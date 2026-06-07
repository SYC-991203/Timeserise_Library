import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Dynamic Hypergraph Structure Learning (DHSL) [cite: 4, 10]
    严格适配 Time-Series-Library 接口标准
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_nodes = configs.enc_in
        self.d_model = configs.d_model
        self.k_neighbors = getattr(configs, 'k_neighbors', 3) # KNN 参数 [cite: 11, 191]
        self.beta_add = getattr(configs, 'beta_add', 0.05)     # 节点增强比例 [cite: 291]
        self.tau = getattr(configs, 'tau', 1.0)               # Gumbel Softmax 温度 [cite: 250, 279]
        self.norm_layer = nn.LayerNorm(configs.d_model)

        # 1. 变量嵌入层 (用于将时间序列映射到高维空间)
        self.embedding = nn.Linear(configs.seq_len, configs.d_model)

        # 2. 动态超图结构学习模块 (HSL) [cite: 176, 216]
        self.z_e = nn.Parameter(torch.randn(self.num_nodes)) # 超边缘重要性参数 [cite: 237, 241]
        self.fc_v = nn.Linear(self.d_model, self.d_model)
        self.fc_e = nn.Linear(self.d_model, self.d_model)
        self.fc_score = nn.Linear(self.d_model, 1) # 计算 Z_v Eq.(7) [cite: 275]
        self.multi_attn = nn.MultiheadAttention(self.d_model, num_heads=4, batch_first=True) # Eq.(10) [cite: 288]
        self.alpha = nn.Parameter(torch.tensor(0.1)) # 动态演化系数 Eq.(13) [cite: 304]

        # 3. 时空超图神经网络 (STHGNN) [cite: 188, 307]
        self.W_H = nn.Linear(self.d_model, self.d_model) # 超图卷积参数 Eq.(14) [cite: 313]
        self.W_A = nn.Linear(self.d_model, self.d_model) # 预定义图卷积参数 Eq.(15) [cite: 322]
        self.fusion_attn = nn.MultiheadAttention(self.d_model, num_heads=2, batch_first=True) # Eq.(16) [cite: 328]

        # 4. 时间层 (GRU) 与预测头 [cite: 331, 337]
        self.gru = nn.GRU(self.d_model, self.d_model, batch_first=True) # Eq.(17) [cite: 333]
        self.out_proj = nn.Linear(self.d_model, configs.pred_len) # Eq.(18) [cite: 337]

    def _gumbel_softmax(self, logits, tau):
        """ 离散采样可微化处理 Eq.(5) & Eq.(8) [cite: 252, 279] """
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        return torch.sigmoid((logits + gumbel_noise) / tau)

    def _generate_knn_hypergraph(self, x):
        """ 基于距离的超图生成 Eq.(2)  """
        B, N, D = x.shape
        dist = torch.cdist(x, x, p=2) # [B, N, N]
        _, knn_idx = torch.topk(dist, self.k_neighbors, largest=False) # [B, N, K]
        H = torch.zeros(B, N, N).to(x.device)
        H.scatter_(2, knn_idx, 1.0)
        return H

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, A_pre=None):
        """ 核心长程预测逻辑 [cite: 179] """
        B, L, N = x_enc.shape
        # 输入转换: [Batch, Nodes, Seq_Len]
        x = x_enc.transpose(1, 2)
        node_emb = self.embedding(x) # [B, N, d]
        edge_emb = node_emb.clone()  # 初始超边缘特征

        # --- 步骤 1: 动态超图优化 (HSL) [cite: 233] ---
        H_init = self._generate_knn_hypergraph(node_emb)
        
        # 超边缘去噪 Eq.(6) [cite: 262]
        m_e = self._gumbel_softmax(self.z_e.expand(B, N), self.tau)
        H_prime = H_init * m_e.unsqueeze(1)
        
        # 节点去噪 Eq.(9) [cite: 281]
        v_feat = self.fc_v(node_emb).unsqueeze(2).repeat(1, 1, N, 1)
        e_feat = self.fc_e(edge_emb).unsqueeze(1).repeat(1, N, 1, 1)
        Z_v = torch.sigmoid(self.fc_score(torch.relu(v_feat + e_feat))).squeeze(-1)
        Z_v = torch.clamp(Z_v, 1e-7, 1 - 1e-7)
        M_v = self._gumbel_softmax(Z_v, self.tau)
        H_tilde = H_prime * M_v
        
        # 节点增强 Eq.(11) [cite: 294]
        _, S = self.multi_attn(node_emb, edge_emb, edge_emb)
        k_add = int(N * N * self.beta_add)
        delta_H = torch.zeros_like(H_init)
        if k_add > 0:
            top_vals, _ = torch.topk(S.view(B, -1), k_add, dim=-1)
            threshold = top_vals[:, -1].view(B, 1, 1)
            delta_H = (S >= threshold).float()
        
        H_dyn = H_tilde + delta_H # Eq.(12) [cite: 298]

        # --- 步骤 2: 时空特征聚合 (STHGNN) [cite: 308] ---
        # 超图卷积 Eq.(14) [cite: 313]
        dv = torch.sum(H_dyn, dim=2)
        de = torch.sum(H_dyn, dim=1)

        # 不仅加 epsilon，还要对结果进行最大限幅
        # 1e-9 防止除零，1e4 防止求逆后的数值过大
        Dv_inv = torch.diag_embed(torch.pow(dv + 1e-9, -0.5)).clamp(max=1e4)
        De_inv = torch.diag_embed(torch.pow(de + 1e-9, -1.0)).clamp(max=1e4)

        X_H = Dv_inv @ H_dyn @ De_inv @ H_dyn.transpose(1, 2) @ Dv_inv @ node_emb

        # 必须在激活函数前加入 LayerNorm，这是防止数据驱动模型崩掉的工业级做法
        # 如果 configs 里没定义，可以手动加一个 nn.LayerNorm(d_model)
        X_H = self.norm_layer(X_H) 
        X_H = torch.relu(self.W_H(X_H))

        # 预定义图卷积 Eq.(15) [cite: 322]
        # 如果没有提供预定义 A_pre，则降级为自环图
        if A_pre is None:
            A_pre = torch.eye(N).to(x_enc.device).expand(B, N, N)
        da = torch.sum(A_pre, dim=2) + 1e-5
        Da_inv = torch.diag_embed(torch.pow(da, -0.5))
        X_A = Da_inv @ A_pre @ Da_inv @ node_emb
        X_A = torch.relu(self.W_A(X_A))

        # 注意力融合 Eq.(16) [cite: 328]
        combined = torch.cat([X_H, X_A], dim=1)
        fused, _ = self.fusion_attn(combined, combined, combined)
        spatial_feat = fused[:, :N, :] + fused[:, N:, :]

        # 时间维度聚合 Eq.(17) [cite: 333]
        # 为每个节点跑 GRU
        gru_in = spatial_feat.reshape(B * N, 1, -1)
        out, _ = self.gru(gru_in)
        
        # 步骤 3: 投影输出 Eq.(18) [cite: 337]
        dec_out = self.out_proj(out.view(B, N, -1)) # [B, N, Pred_Len]
        return dec_out.transpose(1, 2) # 返回 [B, Pred_Len, Nodes]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """ TSlib 标准 forward 接口 """
        if self.task_name == 'long_term_forecast':
            # 在 TSlib 框架下，A_pre 通常作为静态属性或通过 configs 传入
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # 满足 TSlib 的切片返回标准：返回最后 pred_len 部分
            return dec_out[:, -self.pred_len:, :]
        return None