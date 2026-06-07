import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class ChannelMoE_GroupRouter(nn.Module):
    """
    MoE 路由模块，LLM 引导 Logits 作为内部可学习参数。
    接收组特定的初始化 Logits。
    """
    def __init__(self, configs, specific_init_logits: List[float]):
        super(ChannelMoE_GroupRouter, self).__init__()
        
        self.d_model = configs.d_model
        self.n_experts = 2 
        
        # 修复：使用传入的组特定的 Logits [length 2] 初始化可学习参数
        initial_logits = torch.tensor(
            specific_init_logits, 
            dtype=torch.float32
        ).view(1, 1, self.n_experts)
        
        # G_LLM (可训练的 LLM 引导先验) - 形状 [1, 1, 2]
        self.guidance_logits = nn.Parameter(initial_logits, requires_grad=True) 
        
        # 鲁棒性参数: 控制引导比例 (Logit 形式，通过 Sigmoid 激活到 [0, 1])
        self.alpha = nn.Parameter(
            torch.tensor(getattr(configs, 'llm_robustness_init_alpha', -10.0)),
            requires_grad=True
        )
        
        # 专家列表 (Experts) - 保持二元决策
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(self.d_model, self.d_model * 2), nn.GELU(), nn.Linear(self.d_model * 2, self.d_model)),
            nn.Sequential(nn.Linear(self.d_model, self.d_model * 2), nn.GELU(), nn.Linear(self.d_model * 2, self.d_model))
        ])
        
        # 数据驱动门控 (G_Data)
        self.data_gate = nn.Linear(self.d_model, self.n_experts)

    def forward(self, x):
        """
        x: [Batch, Seq_Len, d_model] - 输入特征
        """
        B, L, D = x.shape
        
        # 1. 专家计算
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        # expert_outputs: [B, L, D, n_experts]

        # 2. 门控计算 (Gating Mechanism)
        
        # 数据驱动门控 G_Data: [B, L, D] -> [B, L, n_experts]
        data_routing_logits = self.data_gate(x) 
        
        # 获取内部 LLM 引导 G_LLM (从 [1, 1, 2] 扩展到 [B, L, 2])
        llm_guidance = self.guidance_logits.expand(B, L, self.n_experts)
        
        # 鲁棒性混合门控 (Robust Blended Gate)
        alpha = torch.sigmoid(self.alpha) # 将 Logit 转化为混合系数 [0, 1]
        
        # 混合 logits: G_Final_logits = alpha * G_Data + (1-alpha) * G_LLM
        final_routing_logits = alpha * data_routing_logits + (1.0 - alpha) * llm_guidance
        
        final_routing_weights = F.softmax(final_routing_logits, dim=-1)
        
        # 3. 结果加权聚合
        output = torch.sum(expert_outputs * final_routing_weights.unsqueeze(-2), dim=-1)
        
        return output, final_routing_weights
    


class ChannelDiffMoE_GroupRouter(nn.Module):
    def __init__(self, configs, specific_init_logits: List[float]):
        super(ChannelDiffMoE_GroupRouter, self).__init__()
        
        self.d_model = configs.d_model
        self.n_experts = 2 
        
        # 初始化 Logits (保持不变)
        initial_logits = torch.tensor(specific_init_logits, dtype=torch.float32).view(1, 1, self.n_experts)
        self.guidance_logits = nn.Parameter(initial_logits, requires_grad=True) 
        self.alpha = nn.Parameter(torch.tensor(getattr(configs, 'llm_robustness_init_alpha', 0.0)), requires_grad=True)
        
        # === 核心修改：异构专家 ===
        self.experts = nn.ModuleList([
            # Expert 0: 卷积专家 (Conv Expert) - 擅长局部特征/高频
            # 使用 kernel_size=3 的一维卷积来捕获邻近时间步的关系
            nn.Sequential(
                nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model * 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(in_channels=self.d_model * 2, out_channels=self.d_model, kernel_size=1),
            ),
            
            # Expert 1: 线性专家 (Linear Expert) - 擅长全局映射/低频
            # 也就是原来的 MLP 结构
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model * 2),
                nn.GELU(),
                nn.Linear(self.d_model * 2, self.d_model)
            )
        ])
        
        self.data_gate = nn.Linear(self.d_model, self.n_experts)

    def forward(self, x):
        """
        x: [Batch, Seq_Len, d_model]
        """
        B, L, D = x.shape
        
        # === Expert 0 (Conv) 处理 ===
        # Conv1d 需要输入 [B, C, L]，所以需要 permute
        x_perm = x.permute(0, 2, 1) # [B, D, L]
        out0 = self.experts[0](x_perm).permute(0, 2, 1) # -> [B, L, D]
        
        # === Expert 1 (Linear) 处理 ===
        out1 = self.experts[1](x) # [B, L, D]
        
        # 堆叠
        expert_outputs = torch.stack([out0, out1], dim=-1)
        
        # === 门控逻辑 (保持不变) ===
        data_routing_logits = self.data_gate(x) 
        llm_guidance = self.guidance_logits.expand(B, L, self.n_experts)
        alpha = torch.sigmoid(self.alpha)
        
        final_routing_logits = alpha * data_routing_logits + (1.0 - alpha) * llm_guidance
        final_routing_weights = F.softmax(final_routing_logits, dim=-1)
        
        output = torch.sum(expert_outputs * final_routing_weights.unsqueeze(-2), dim=-1)
        
        return output, final_routing_weights