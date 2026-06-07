import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from typing import Dict, List

# 引入 TSlib 的核心组件
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding

# 引入你的 MoE 模块 (假设文件名为 layers.moe)
from layers.Moe import ChannelMoE_GroupRouter

class Model(nn.Module):
    """
    MoE-Transformer:
    1. ChannelMoE: 处理通道分组与交互 (Spatial/Channel Dimension)
    2. Transformer Encoder: 处理全局时间依赖 (Temporal Dimension)
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # 1. 基本配置
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = nn.Dropout(configs.dropout)
        
        # 2. LLM 引导配置
        self.direction = getattr(configs, 'direction', [0] * configs.enc_in)
        self.group_labels = sorted(list(set(self.direction)))
        
        # 3. MoE 模块初始化 (用于替代普通的 Embedding 层做特征提取)
        init_logits_str: Dict[str, List[float]] = getattr(configs, 'moe_guidance_init_logits', {})
        # 兼容处理：如果是字符串则解析，如果是字典则直接使用
        if isinstance(init_logits_str, str):
            init_logits_dict = json.loads(init_logits_str) if init_logits_str else {}
        else:
            init_logits_dict = init_logits_str

        self.group_embeddings = nn.ModuleDict()
        self.group_moe_routers = nn.ModuleDict()
        self.attention_results = {} 

        # MoE 组件构建
        for group_label in self.group_labels:
            group_label_str = str(group_label)
            indices = np.where(np.array(self.direction) == group_label)[0]
            c_in_group = len(indices)

            # 使用 TSlib 的 DataEmbedding (包含 Positional Embedding)
            self.group_embeddings[group_label_str] = DataEmbedding(c_in_group, self.d_model, configs.dropout)
            
            specific_init_logits = init_logits_dict.get(group_label_str, [0.0, 0.0])
            self.group_moe_routers[group_label_str] = ChannelMoE_GroupRouter(
                configs, specific_init_logits=specific_init_logits
            )
            
        # -----------------------------------------------------------
        # 4. 新增：Transformer Encoder (借鉴 Vanilla Transformer)
        # -----------------------------------------------------------
        # 我们使用 Encoder 来捕捉 MoE 提取特征后的时间依赖
        # 这里的 d_model 必须与 MoE 输出的 d_model 一致
        self.transformer_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), 
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # 5. MoE 后处理层 (MLP) - 保留作为局部特征精炼
        self.MLP1 = nn.Linear(self.d_model, self.d_model)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)

        # 6. 投影层
        # MoE + Transformer 输出维度为 d_model，我们将其映射回 1 (单通道预测)
        self.projection = nn.Linear(self.d_model, 1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            if getattr(self, 'output_moe_weights', False):
                return dec_out, self.attention_results['group_routing_weights']
            else:
                return dec_out
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        batch_size = x_enc.shape[0]
        seq_len = x_enc.shape[1]

        # 初始化 MoE 输出容器 [B, L, C, D]
        final_moe_out = torch.zeros(batch_size, seq_len, self.enc_in, self.d_model).to(x_enc.device)
        self.attention_results['group_routing_weights'] = {} 
        
        direction_tensor = torch.tensor(self.direction).to(x_enc.device)
        
        # ==========================================
        # Stage 1: LLM-Guided MoE (Spatial/Channel Processing)
        # ==========================================
        for group_label in self.group_labels:
            group_label_str = str(group_label)
            indices = torch.nonzero(direction_tensor == group_label).squeeze(-1)
            num_channels_in_group = indices.shape[0] if indices.dim() > 0 else 1
            
            # 1.1 提取与 Embedding
            group_x = x_enc[:, :, indices] 
            group_x_mark = x_mark_enc 
            
            # [B, L, D] - 此时已经包含了 Positional Embedding
            group_embedding = self.group_embeddings[group_label_str](group_x, group_x_mark) 

            # 1.2 MoE 路由
            moe_router = self.group_moe_routers[group_label_str]
            moe_out, routing_weights = moe_router(group_embedding)
            self.attention_results['group_routing_weights'][group_label_str] = routing_weights

            # 1.3 局部 FFN 精炼 (保留原有逻辑)
            local_out = group_embedding + self.dropout(moe_out)
            local_out = self.norm1(local_out)
            local_out = local_out + self.dropout(self.MLP1(local_out))
            local_out = self.norm2(local_out) 

            # 1.4 填充回大张量
            local_out_expanded = local_out.unsqueeze(2).expand(-1, -1, num_channels_in_group, -1)
            for idx_in_group, channel_idx in enumerate(indices):
                final_moe_out[:, :, channel_idx, :] = local_out_expanded[:, :, idx_in_group, :]

        # final_moe_out 形状: [B, L, C, D]
        
        # ==========================================
        # Stage 2: Transformer Encoder (Temporal Processing)
        # ==========================================
        
        # 关键策略: Channel Independence (CI)
        # 我们希望 Transformer 学习每个通道的时间模式，但不混淆不同通道的 MoE 特征。
        # 操作: 将 Batch 和 Channel 维度合并 -> [B * C, L, D]
        
        # 1. 维度变换: [B, L, C, D] -> [B, C, L, D] -> [B*C, L, D]
        enc_in = final_moe_out.permute(0, 2, 1, 3).reshape(batch_size * self.enc_in, seq_len, self.d_model)
        
        # 2. Transformer Encoder 编码
        # 输入包含 Positional Embedding (来自 DataEmbedding)，所以直接进 Attention
        # enc_out: [B*C, L, D]
        enc_out, attns = self.transformer_encoder(enc_in, attn_mask=None)
        
        # 3. 维度还原: [B*C, L, D] -> [B, C, L, D] -> [B, L, C, D]
        enc_out = enc_out.reshape(batch_size, self.enc_in, seq_len, self.d_model).permute(0, 2, 1, 3)

        # ==========================================
        # Stage 3: Projection & Output
        # ==========================================
        
        # 展平: [B, L, C, D] -> [B*L*C, D]
        output = enc_out.reshape(-1, self.d_model)
        
        # 投影: [B*L*C, D] -> [B*L*C, 1]
        output = self.projection(output)
        
        # 还原: [B*L*C, 1] -> [B, L, C]
        output = output.view(batch_size, seq_len, self.enc_in)
        
        # 截取预测部分
        dec_out = output[:, -self.pred_len:, :]
        
        return dec_out