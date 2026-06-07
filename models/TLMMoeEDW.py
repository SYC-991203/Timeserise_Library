import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from typing import Dict, List

from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
from layers.Moe import ChannelMoE_GroupRouter

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # ... (基础配置保持不变) ...
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.enc_in = configs.enc_in
        self.dec_in = configs.dec_in
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.dropout = nn.Dropout(configs.dropout)

        # 1. LLM 引导配置
        self.direction = getattr(configs, 'direction', [0] * configs.enc_in)
        self.group_labels = sorted(list(set(self.direction)))
        self.num_groups = len(self.group_labels)
        
        # 只保留 Routing 引导，移除 Fusion 引导，降低幻觉风险
        init_logits_str = getattr(configs, 'moe_guidance_init_logits', {})
        init_logits_dict = json.loads(init_logits_str) if isinstance(init_logits_str, str) else init_logits_str

        self.group_embeddings = nn.ModuleDict()
        self.group_moe_routers = nn.ModuleDict()
        self.attention_results = {}

        # 2. MoE 组件初始化
        for group_label in self.group_labels:
            group_label_str = str(group_label)
            indices = np.where(np.array(self.direction) == group_label)[0]
            c_in_group = len(indices)

            self.group_embeddings[group_label_str] = DataEmbedding(c_in_group, self.d_model, configs.dropout)
            
            # 使用 LLM 给出的 Routing Prior
            specific_init_logits = init_logits_dict.get(group_label_str, [0.0, 0.0])
            self.group_moe_routers[group_label_str] = ChannelMoE_GroupRouter(configs, specific_init_logits=specific_init_logits)

        # 3. 可学习的组融合层 (均等初始化 -> 数据驱动)
        # 初始化为全 0 -> Softmax 后变为全 1/N (均等权重)
        # 让模型完全根据 MoE 处理后的特征质量来学习谁更重要
        self.group_fusion_logits = nn.Parameter(torch.zeros(self.num_groups), requires_grad=True)
        
        self.moe_norm = nn.LayerNorm(self.d_model)

        # 4. Decoder Embedding & Transformer Encoder/Decoder (保持不变)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.encoder = Encoder(
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

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # ... (Stage 1 MoE 提取逻辑保持不变) ...
        batch_size = x_enc.shape[0]
        seq_len = x_enc.shape[1]
        enc_out_list = [] 
        self.attention_results['group_routing_weights'] = {}
        direction_tensor = torch.tensor(self.direction).to(x_enc.device)

        for group_label in self.group_labels:
            group_label_str = str(group_label)
            indices = torch.nonzero(direction_tensor == group_label).squeeze(-1)
            
            group_x = x_enc[:, :, indices]
            group_x_mark = x_mark_enc
            group_embedding = self.group_embeddings[group_label_str](group_x, group_x_mark)
            
            moe_router = self.group_moe_routers[group_label_str]
            moe_out, routing_weights = moe_router(group_embedding)
            self.attention_results['group_routing_weights'][group_label_str] = routing_weights
            
            # Residual & Norm
            group_out = group_embedding + self.dropout(moe_out)
            group_out = self.moe_norm(group_out)
            enc_out_list.append(group_out)

        # ==========================================
        # Stage 2: Data-Driven Learnable Fusion
        # ==========================================
        
        stacked_features = torch.stack(enc_out_list, dim=2) # [B, L, G, D]
        
        # 这里完全由训练数据决定哪个组重要
        # 初始阶段是公平的 (1/G)，随着 loss 下降，重要组的权重会上升
        fusion_weights = F.softmax(self.group_fusion_logits, dim=0)
        
        enc_out = torch.sum(stacked_features * fusion_weights.view(1, 1, -1, 1), dim=2)
        
        # ... (Stage 3 & 4 Transformer Encoder/Decoder 逻辑保持不变) ...
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        
        return dec_out[:, -self.pred_len:, :]
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            if getattr(self, 'output_moe_weights', False):
                return dec_out, self.attention_results['group_routing_weights']
            else:
                return dec_out[:, -self.pred_len:, :]
        return None