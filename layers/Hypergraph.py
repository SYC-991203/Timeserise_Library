import torch
import torch.nn as nn
import torch.nn.functional as F
import json
class BioHypergraph_Encoder(nn.Module):
    """
    BioHypergraph_Encoder (基于生物机理的超图编码器) - TSlib Style
    
    [Design Philosophy]
    1. Structure Init: 利用 LLM 的分组打分 (Logits) 初始化超图关联矩阵 H。
    2. Projection: 变量 -> 功能组 (Node to Hyperedge)。
    3. Interaction: 功能组之间的信息交互 (Hyperedge Mixing)。
    4. Reconstruction: 功能组 -> 变量 (Hyperedge to Node)。
    
    [Robustness]
    - H 是 Parameter，允许梯度下降修正 LLM 的幻觉。
    - 强制检查 configs 参数，避免隐式错误。
    """
    def __init__(self, configs):
        super(BioHypergraph_Encoder, self).__init__()
        
        # === 1. 基础配置 (Fail Fast: 直接访问属性，缺参即报错) ===
        self.d_model = configs.d_model
        self.num_vars = configs.enc_in
        
        # 如果 configs 里没写这些，直接抛出 AttributeError，方便调试
        self.dropout = configs.dropout 
        self.n_heads = configs.n_heads
        self.init_temperature = configs.temp
        
        # 必须显式定义超边(组)的数量，通常为 3 (物理/代谢/生化)
        # 如果 configs 中没有这个参数，请在 Configs 类中添加
        self.num_hyperedges =3
        
        # === 2. LLM 先验结构解析 ===
        # 必须传入 LLM 打分字典，否则报错。
        # 格式: {"0": {0: 5.0, ...}, "1": {...}}
        if not hasattr(configs, 'moe_logits_init'):
            raise ValueError("Config missing 'moe_logits_init'! Please provide LLM scores.")
            
        llm_scores = configs.moe_logits_init
        
        # === 3. 核心创新：可学习的关联矩阵 H (Incidence Matrix) ===
        # 初始化 H (Logits 空间)
        # H 的形状: [Num_Vars, Num_Hyperedges]
        H_init = self._init_incidence_matrix(llm_scores)
        
        # 将 H 定义为 Parameter，允许模型在百万级数据上微调 LLM 的先验
        # 这就是对抗 "LLM幻觉" 的核心防线
        self.H_logits = nn.Parameter(H_init, requires_grad=True)

        # === 4. 神经网络层 ===
        
        # A. 节点变换层 (Node -> Hidden)
        self.node_proj = nn.Linear(self.d_model, self.d_model)
        
        # B. 超边交互层 (Hyperedge Self-Attention)
        # 实现 "Group 0 <-> Group 1 <-> Group 2" 的机理交互
        self.hyperedge_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=self.dropout,
            batch_first=True
        )
        self.norm_hyper = nn.LayerNorm(self.d_model)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.log_counter = 0
        # C. 输出层 (Reconstruction)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.norm_out = nn.LayerNorm(self.d_model)

    def _init_incidence_matrix(self, llm_scores):
        """
        利用 LLM 打分初始化 H 矩阵，并进行温度缩放以防止梯度饱和。
        Args:
            llm_scores: LLM 提供的分数 (Dict or JSON String)
            init_temperature: 温度系数 (除数)。
                              建议值: 
                              - 1.0 (保留原始 5.0/-5.0，极度自信，梯度难传)
                              - 3.0~5.0 (压缩到 1.0 左右，比较温和，梯度健康)
        """
        # 1. 初始化底板：
        # 原本是 -5.0，现在直接根据温度缩放
        # 比如 temp=5.0, base_value 就变成 -1.0。这样未提及的点也不是"绝对死刑"，而是"弱连接"。
        base_val = -5.0
        H = torch.full((self.num_vars, self.num_hyperedges), base_val)

        # ---------------- 解析逻辑 (保持不变) ----------------
        if isinstance(llm_scores, str):
            try:
                llm_scores = json.loads(llm_scores)
            except json.JSONDecodeError:
                try:
                    import ast
                    llm_scores = ast.literal_eval(llm_scores)
                except Exception as e:
                    print(f"[Error] Failed to parse llm_scores string: {e}")
                    # 解析失败，返回缩放后的底板
                    return H / self.init_temperature
        # ---------------------------------------------------

        # 2. 填充 LLM 打分
        if isinstance(llm_scores, dict):
            for var_idx_str, scores_dict in llm_scores.items():
                try:
                    var_idx = int(var_idx_str)
                    if var_idx >= self.num_vars:
                        print(f"[Warning] Index {var_idx} out of bounds. Ignored.")
                        continue
                    
                    for g_idx, score in scores_dict.items():
                        g_idx = int(g_idx)
                        if g_idx < self.num_hyperedges:
                            # 这里填入原始分 (如 5.0)
                            H[var_idx, g_idx] = float(score)
                            
                except ValueError:
                     raise ValueError(f"Invalid format key: {var_idx_str}")
        else:
             # 如果不是dict且没解析成功，这里可能要处理一下，或者上面return了
             pass

        # 3. 【关键步骤】全局温度缩放 (Scaling)
        # 这一步把 5.0 变成 1.6 (如果 temp=3)，把 -5.0 变成 -1.6
        # 这样进入 Softmax 后，模型既有先验知识，又能动起来。
        H = H / self.init_temperature
        

        return H

    def forward(self, x):
        """
        Input: [Batch, Num_Vars, D_Model]
        """
        B, N, D = x.shape
        residual = x
        
        # === Step 1: 计算动态拓扑权重 ===
        # H_matrix: [N, G]
        H_matrix = F.softmax(self.H_logits, dim=1)
        
        # === [简洁版] 核心参数 Trace 日志 ===
        # 直接用 if 判断，去掉 with torch.no_grad()
        if self.training and (self.log_counter % 100 == 0):
             # 传入 H_matrix 即可
            self._print_topology_trace(H_matrix)
            
        if self.training:
            self.log_counter += 1
        
        # ... [后续计算逻辑不变] ...
        x_node = self.node_proj(x)
        x_hyperedge = torch.einsum('bnd,ng->bgd', x_node, H_matrix)
        attn_out, _ = self.hyperedge_attention(x_hyperedge, x_hyperedge, x_hyperedge)
        x_hyperedge = self.norm_hyper(x_hyperedge + self.dropout_layer(attn_out))
        x_reconstructed = torch.einsum('bgd,ng->bnd', x_hyperedge, H_matrix)
        output = self.out_proj(x_reconstructed)
        return self.norm_out(output + residual)

    def _print_topology_trace(self, H_matrix):
            """
            单行打印所有变量的分布，避免多卡训练时日志换行冲突。
            格式示例: [Step 100] v0:G1[0.1,0.9] | v1:G0[0.8,0.2] ...
            """
            # 1. 准备数据: 断开梯度, 转CPU, 转列表
            # shape: [num_vars, num_groups]
            all_probs = H_matrix.detach().cpu().tolist()
            
            # 2. 构建每各变量的字符串片段
            # 格式: v{索引}:G{主组}[权重1,权重2...]
            var_logs = []
            for var_idx, probs in enumerate(all_probs):
                max_group = probs.index(max(probs)) # 找到主导组
                # 简化权重显示，保留3-4位小数即可，用逗号分隔
                probs_str = ",".join([f"{p:.3f}" for p in probs]) 
                var_logs.append(f"v{var_idx}:G{max_group}[{probs_str}]")
                
            # 3. 拼接成完整的一行日志
            # 使用 " | " 作为变量间的分隔符
            full_log = f"[TopoTrace Step {self.log_counter}] " + " | ".join(var_logs)
            
            # 4. 单次打印 (原子操作，不会被其他进程的输出打断中间内容)
            print(full_log)


class DataDrivenGNN_Encoder(nn.Module):
    def __init__(self, configs, print_freq=100):
        super(DataDrivenGNN_Encoder, self).__init__()
        self.num_vars = configs.enc_in
        self.d_model = configs.d_model
        
        # 打印控制
        self.print_freq = print_freq
        self.batch_count = 0
        
        # 1. 结构学习参数 (Structure Learning)
        self.node_emb1 = nn.Parameter(torch.randn(self.num_vars, 64))
        self.node_emb2 = nn.Parameter(torch.randn(self.num_vars, 64))
        
        # 2. GCN 层
        self.gcn_linear = nn.Linear(self.d_model, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(configs.dropout)

    def _print_topology_trace(self, adj_matrix):
        """
        GNN版单行拓扑监控。
        adj_matrix: [num_vars, num_vars]
        每一行代表一个变量(Receiver)，列代表它关注的邻居(Sender)
        """
        # 1. 准备数据
        all_probs = adj_matrix.detach().cpu().tolist()
        
        var_logs = []
        for var_idx, probs in enumerate(all_probs):
            # 找到权重最大的那个"邻居" (Max Neighbor)
            max_neighbor = probs.index(max(probs))
            
            # 格式化权重字符串 (保留2位小数即可，太长看不清)
            # 比如: [0.01, 0.90, 0.05, ...]
            probs_str = ",".join([f"{p:.2f}" for p in probs])
            
            # 格式: v0->v2[权重详情]
            # 含义: 变量0 主要在听 变量2 的信息
            var_logs.append(f"v{var_idx}->v{max_neighbor}[{probs_str}]")
            
        # 3. 拼接
        full_log = f"[GNNTrace {self.batch_count}] " + " | ".join(var_logs)
        print(full_log)

    def forward(self, x):
        # x: [Batch, Vars, D]
        
        # [Step 1] 生成邻接矩阵 (Adjacency Matrix)
        # result: [Vars, Vars]
        logits = torch.matmul(self.node_emb1, self.node_emb2.transpose(1, 0))
        adj = torch.softmax(torch.relu(logits), dim=-1)
        
        # ================== 监控模块 ==================
        if self.training:
            self.batch_count += 1
            if (self.batch_count % self.print_freq == 0) or (self.batch_count==1) :
                self._print_topology_trace(adj)
        # ============================================

        # [Step 2] 聚合
        out = torch.matmul(adj, x)
        
        # [Step 3] 变换与残差
        out = self.gcn_linear(out)
        out = self.dropout(out)
        
        return self.norm(x + out)
    


class PearsonGNN_Encoder(nn.Module):
    """
    Ablation Model: Fixed Pearson Correlation Encoder
    
    Mechanism:
    - 内置 sub1/sub2/sub3 的静态皮尔森矩阵。
    - 初始化时根据 configs.data_path 自动加载对应矩阵。
    - 矩阵被注册为 Buffer，严格不可训练 (Frozen)。
    """
    def __init__(self, configs):
        super(PearsonGNN_Encoder, self).__init__()
        self.d_model = configs.d_model
        
        # 1. 硬编码预计算的 Pearson 矩阵
        # 来源: User provided static calculation
        pearson_data = {
            'sub1': [
                [ 1.000,  0.226, -0.049,  0.143, -0.035],
                [ 0.226,  1.000, -0.196,  0.431, -0.131],
                [-0.049, -0.196,  1.000, -0.634, -0.672],
                [ 0.143,  0.431, -0.634,  1.000,  0.345],
                [-0.035, -0.131, -0.672,  0.345,  1.000]
            ],
            'sub2': [
                [ 1.000, -0.615, -0.458,  0.008,  0.513],
                [-0.615,  1.000,  0.496, -0.103, -0.850],
                [-0.458,  0.496,  1.000, -0.123, -0.515],
                [ 0.008, -0.103, -0.123,  1.000,  0.227],
                [ 0.513, -0.850, -0.515,  0.227,  1.000]
            ],
            'sub3': [
                [ 1.000,  0.226, -0.199, -0.015,  0.006],
                [ 0.226,  1.000, -0.615,  0.008, -0.005],
                [-0.199, -0.615,  1.000, -0.103, -0.459],
                [-0.015,  0.008, -0.103,  1.000,  0.288],
                [ 0.006, -0.005, -0.459,  0.288,  1.000]
            ]
        }

        # 2. 根据 data_path 自动匹配
        # 假设 data_path 类似于 "dataset/DYG_data_3_sub2.csv"
        target_key = None
        for key in ['sub1', 'sub2', 'sub3']:
            if key in configs.data_path:
                target_key = key
                break
        
        if target_key is None:
            # 如果没匹配到，报错或者默认单位阵 (防止实验跑偏建议报错)
            raise ValueError(f"CRITICAL ERROR: Could not match 'sub1/2/3' in data_path: {configs.data_path}")

        print(f"\n[PearsonGNN] ✅ Detected Dataset: {target_key}")
        print(f"[PearsonGNN] 🧊 Matrix Loaded & FROZEN (Grad=False).")

        # 3. 转换为 Tensor 并注册为 Buffer
        # register_buffer 会自动处理 device 转移，且不会被 optimizer 更新
        adj_matrix = torch.tensor(pearson_data[target_key], dtype=torch.float32)
        self.register_buffer('adj', adj_matrix)

        # 4. 打印拓扑结构供检查
        self._print_static_topology()

        # 5. 特征变换层 (这是网络唯一能学的部分: Linear Projection)
        # 即使图结构固定，特征维度 D 的混合还是需要学习的
        self.gcn_linear = nn.Linear(self.d_model, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(configs.dropout)

    def _print_static_topology(self):
        """仅在初始化时打印一次，确认加载正确"""
        print("=== Static Pearson Topology Preview ===")
        all_probs = self.adj.tolist()
        var_logs = []
        for var_idx, probs in enumerate(all_probs):
            # 找到除了自己(1.0)以外最大的相关性
            # 简单的逻辑：找绝对值最大的邻居
            others = [(i, p) for i, p in enumerate(probs) if i != var_idx]
            max_neighbor, max_val = max(others, key=lambda x: abs(x[1]))
            
            # 格式化: v0->v1[0.85]
            var_logs.append(f"v{var_idx}->v{max_neighbor}[{max_val:.2f}]")
        print(" | ".join(var_logs))
        print("=======================================\n")

    def forward(self, x):
        # x shape: [Batch, Vars, D_Model]
        
        # [Step 1] 静态图聚合
        # Matmul: [Vars, Vars] @ [Batch, Vars, D] -> [Batch, Vars, D]
        # self.adj 里的值已经在 init 里固定死了
        out = torch.matmul(self.adj, x)
        
        # [Step 2] 线性变换 + 残差
        out = self.gcn_linear(out)
        out = self.dropout(out)
        
        return self.norm(x + out)


######
class PearsonHKGNN_Encoder(nn.Module):
    """
    Ablation Model: KNN-based Static Pearson Hypergraph Encoder
    
    逻辑：
    1. 加载静态 Pearson 矩阵作为相似度度量。
    2. 使用 KNN 将 5 个变量聚合成超边（组数 K 由 configs.k_groups 决定）。
    3. 执行超图卷积进行变量间特征聚合。
    """
    def __init__(self, configs):
        super(PearsonHKGNN_Encoder, self).__init__()
        self.d_model = configs.d_model
        # 获取超边内的节点数 K (1-5)
        self.k_groups = int(configs.temp)  # 这里复用 temp 参数来传递 K 的值，避免修改 Configs 结构。使用时请确保 configs.temp 是整数且在合理范围内。
        # 1. 静态数据定义 (保持原样)
        pearson_data = {
            'sub1': [[1.000, 0.226, -0.049, 0.143, -0.035], [0.226, 1.000, -0.196, 0.431, -0.131], [-0.049, -0.196, 1.000, -0.634, -0.672], [0.143, 0.431, -0.634, 1.000, 0.345], [-0.035, -0.131, -0.672, 0.345, 1.000]],
            'sub2': [[1.000, -0.615, -0.458, 0.008, 0.513], [-0.615, 1.000, 0.496, -0.103, -0.850], [-0.458, 0.496, 1.000, -0.123, -0.515], [0.008, -0.103, -0.123, 1.000, 0.227], [0.513, -0.850, -0.515, 0.227, 1.000]],
            'sub3': [[1.000, 0.226, -0.199, -0.015, 0.006], [0.226, 1.000, -0.615, 0.008, -0.005], [-0.199, -0.615, 1.000, -0.103, -0.459], [-0.015, 0.008, -0.103, 1.000, 0.288], [0.006, -0.005, -0.459, 0.288, 1.000]]
        }

        # 2. 匹配数据集
        target_key = next((k for k in ['sub1', 'sub2', 'sub3'] if k in configs.data_path), None)
        if target_key is None:
            raise ValueError(f"DataPath {configs.data_path} matched no sub-dataset.")

        # 3. 构建关联矩阵 H (Incidence Matrix)
        # H shape: [Num_Nodes, Num_Hyperedges]
        # 这里每个节点作为中心点生成一个包含其 K 个近邻的超边
        adj = torch.tensor(pearson_data[target_key], dtype=torch.float32)
        num_nodes = adj.shape[0]
        H = torch.zeros((num_nodes, num_nodes))
        
        # KNN 逻辑：对每一列（每个潜在超边），选取相关性绝对值最大的前 K 个变量
        # 因为 adj 是相似度矩阵，直接用 topk
        _, indices = torch.topk(torch.abs(adj), self.k_groups, dim=1)
        for i in range(num_nodes):
            H[indices[i], i] = 1.0 # 将选中的 K 个变量关联到第 i 条超边

        # 4. 预计算超图拉普拉斯算子的组件
        # D_v: 节点度矩阵 (N, N)
        # D_e: 超边度矩阵 (E, E)
        dv = torch.sum(H, dim=1) # 节点参与了多少条超边
        de = torch.sum(H, dim=0) # 超边包含了多少个节点
        
        # 归一化项 (防止数值爆炸，加 1e-9)
        Dv_inv_sqrt = torch.diag(torch.pow(dv + 1e-9, -0.5))
        De_inv = torch.diag(torch.pow(de + 1e-9, -1.0))
        
        # 注册为 Buffer 保证不被训练且随模型移动
        self.register_buffer('H', H)
        self.register_buffer('G', Dv_inv_sqrt @ H @ De_inv @ H.t() @ Dv_inv_sqrt)

        # 5. 网络层
        self.fc = nn.Linear(configs.d_model, configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.norm = nn.LayerNorm(configs.d_model)

        print(f"[HyperPearson] Dataset: {target_key} | K-Groups: {self.k_groups} | Hypergraph Built.")

    def forward(self, x):
        # x shape: [B, Vars, D]
        # 超图卷积公式: X = G @ X @ W
        # 这里 G = D_v^-0.5 * H * D_e^-1 * H^T * D_v^-0.5
        out = torch.matmul(self.G, x) 
        out = self.fc(out)
        out = self.dropout(out)
        
        return self.norm(x + out) # 残差连接