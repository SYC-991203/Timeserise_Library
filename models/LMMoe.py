import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from typing import List, Dict
from layers.Embed import DataEmbedding
from layers.Moe import ChannelMoE_GroupRouter
from layers.SelfAttention_Family import AttentionLayer, FullAttention, HalfRouterAttentionLayer
# ... 并假设 ChannelMoE_GroupRouter 已经定义如上



# 导入我们修复后的 MoE 路由器
# from layers.moe import ChannelMoE_GroupRouter 
# (此处省略 ChannelMoE_GroupRouter 定义，因为它在 layers/moe.py 中)

# -----------------------------------------------------------------------

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # TSLib 基本配置...
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = nn.Dropout(configs.dropout)
        
        # LLM 引导的通道分组 (静态架构先验)
        self.direction = getattr(configs, 'direction', [0] * configs.enc_in)
        self.group_labels = sorted(list(set(self.direction)))

        # 架构层定义 (共享层)
        # 修复：使用真实的 DataEmbedding
        self.embedding_global = DataEmbedding(self.enc_in, self.d_model, configs.dropout)
        self.projection = nn.Linear(self.d_model, 1) ## 因为是moe的分组处理，所以一个通道上输出一个值
        
        # 共享 FFN/Norm 层 (在 MoE 路由后使用)
        self.MLP1 = nn.Linear(self.d_model, self.d_model)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)

        # 修复 1: 获取 MoE 引导初始化 Logits 字典
        init_logits_str: Dict[str, List[float]] = getattr(configs, 'moe_logits_init', {})
        init_logits_dict = json.loads(init_logits_str)
        
        self.group_embeddings = nn.ModuleDict()
        self.group_moe_routers = nn.ModuleDict()
        self.attention_results = {} # 用于存储路由权重

        # 通道分组 MoE 初始化
        for group_label in self.group_labels:
            group_label_str = str(group_label)
            
            # 确定该组包含的通道数 c_in_group
            indices = np.where(np.array(self.direction) == group_label)[0]
            c_in_group = len(indices)

            # 修复：使用真实的 DataEmbedding (Group_C -> d_model)
            self.group_embeddings[group_label_str] = DataEmbedding(c_in_group, self.d_model, configs.dropout)
            
            # 修复 2: 提取组特定的初始化 Logits
            
            specific_init_logits = init_logits_dict.get(group_label_str, [0.0, 0.0])

            # 修复 3: 初始化 MoE 路由器并传入组特定的 Logits
            self.group_moe_routers[group_label_str] = ChannelMoE_GroupRouter(
                configs, 
                specific_init_logits=specific_init_logits
            )


    # 修复 4: 恢复 TSLib 标准 forward 签名
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # 移除 llm_direction 参数
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # 假设输出路由权重是 TSLib Exp 要求的
            if getattr(self, 'output_moe_weights', False):
                return dec_out, self.attention_results['group_routing_weights']
            else:
                return dec_out
        
        return None

    # 修复 5: 完整的 forecast 逻辑 (移除 Time Attention/Segmentation)
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        batch_size = x_enc.shape[0]
        seq_len = x_enc.shape[1]

        # 1. 全局 DataEmbedding (用于残差和 FFN 的起始点)
        x_global = self.embedding_global(x_enc, x_mark_enc)  # [B, L, d_model]
        
        # ---------------------------------------------
        # 移除 Time Attention 及其相关 FFN/Norm
        # ---------------------------------------------
        
        final_moe_out = torch.zeros(batch_size, seq_len, self.enc_in, self.d_model).to(x_enc.device)
        self.attention_results['group_routing_weights'] = {} 
        
        direction_tensor = torch.tensor(self.direction).to(x_enc.device)
        group_labels = self.group_labels
        
        # 2. LLM 引导的 MoE 路由 (核心逻辑)
        for group_label in group_labels:
            group_label_str = str(group_label)
            
            indices = torch.nonzero(direction_tensor == group_label).squeeze(-1)
            num_channels_in_group = indices.shape[0] if indices.dim() > 0 else 1
            
            # 提取原始输入数据
            group_x = x_enc[:, :, indices] 
            group_x_mark = x_mark_enc 
            
            # DataEmbedding (Group_C -> d_model)
            group_embedding_layer = self.group_embeddings[group_label_str]
            group_embedding = group_embedding_layer(group_x, group_x_mark) # [B, L, d_model]

            # MoE 路由 (内部自带 LLM 引导)
            moe_router = self.group_moe_routers[group_label_str]
            moe_out, routing_weights = moe_router(group_embedding)
            
            self.attention_results['group_routing_weights'][group_label_str] = routing_weights

            # FFN / 归一化 (精炼 MoE 输出)
            local_out = group_embedding + self.dropout(moe_out)
            local_out = self.norm1(local_out)
            local_out = local_out + self.dropout(self.MLP1(local_out))
            local_out = self.norm2(local_out) # local_out: [B, L, d_model]

            # 扩展并赋值回 final_moe_out
            local_out_expanded = local_out.unsqueeze(2).expand(-1, -1, num_channels_in_group, -1)
            
            for idx_in_group, channel_idx in enumerate(indices):
                final_moe_out[:, :, channel_idx, :] = local_out_expanded[:, :, idx_in_group, :]

        # 3. 投影层
        final_out = final_moe_out.view(-1, self.d_model) # [B*L*C,D]      
        final_out = self.projection(final_out)  #[B*L*C,5]          

        output_dim = self.projection.out_features
        final_out = final_out.view(batch_size, seq_len,-1) 
        
        dec_out = final_out[:, -self.pred_len:, :]

        return dec_out  # [batch_size, pred_len, enc_in]