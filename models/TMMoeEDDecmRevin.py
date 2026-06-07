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

# === 1. 序列分解模块 (核心组件) ===
class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
        self.kernel_size = kernel_size

    def forward(self, x):
        """
        x: [Batch, Seq_Len, Channels]
        """
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        
        x_trend = self.moving_avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
        x_seasonal = x - x_trend
        return x_seasonal, x_trend

# === 2. 修复后的 RevIN ===
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        self._get_statistics(x) # 必须调用这个！
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.affine_weight)
        x = x * self.stdev
        x = x + self.mean
        return x

class Model(nn.Module):
    """
    Decomposition-Aware MoE Transformer
    结构: RevIN -> Decomposition -> [Trend Stream (Linear) + Seasonal Stream (MoE+Transformer)] -> Sum -> RevIN_Inverse
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # 1. 基础配置
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.dec_in = configs.dec_in
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.dropout = nn.Dropout(configs.dropout)

        # === 关键组件 1: RevIN ===
        self.revin = RevIN(configs.enc_in)

        # === 关键组件 2: Series Decomposition ===
        kernel_size = 25 # 移动平均窗口，通常 25 左右适合长期预测
        self.decomp = SeriesDecomp(kernel_size)

        # 2. LLM 引导配置
        self.direction = getattr(configs, 'direction', [0] * configs.enc_in)
        self.group_labels = sorted(list(set(self.direction)))
        self.num_groups = len(self.group_labels)
        
        init_logits_str = getattr(configs, 'moe_guidance_init_logits', {})
        init_logits_dict = json.loads(init_logits_str) if isinstance(init_logits_str, str) else init_logits_str

        # 3. MoE 组件 (处理 Seasonal 部分)
        self.group_embeddings = nn.ModuleDict()
        self.group_moe_routers = nn.ModuleDict()
        self.attention_results = {}

        for group_label in self.group_labels:
            group_label_str = str(group_label)
            indices = np.where(np.array(self.direction) == group_label)[0]
            c_in_group = len(indices)

            self.group_embeddings[group_label_str] = DataEmbedding(c_in_group, self.d_model, configs.dropout)
            
            specific_init_logits = init_logits_dict.get(group_label_str, [0.0, 0.0])
            self.group_moe_routers[group_label_str] = ChannelMoE_GroupRouter(configs, specific_init_logits=specific_init_logits)

        # 融合层 (改进：使用 Linear 融合而不是简单的 Sum，保留更多信息)
        # 将 [Num_Groups * d_model] 映射回 [d_model]
        self.fusion_layer = nn.Linear(self.num_groups * self.d_model, self.d_model)
        self.moe_norm = nn.LayerNorm(self.d_model)

        # 4. Transformer Backbone (处理 Seasonal 部分)
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

        # === 关键组件 3: Trend Projection (处理 Trend 部分) ===
        # 使用简单的 Linear 层直接映射 Trend
        # 输入: [Batch, Seq_Len, Channels] -> 输出: [Batch, Pred_Len, Channels]
        self.trend_projection = nn.Linear(configs.seq_len, configs.pred_len)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            if getattr(self, 'output_moe_weights', False):
                return dec_out, self.attention_results['group_routing_weights']
            else:
                return dec_out
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 1. RevIN Normalization (全数据)
        x_enc = self.revin._normalize(x_enc)
        
        # 2. Decomposition (分解为 Seasonal 和 Trend)
        seasonal_init, trend_init = self.decomp(x_enc)
        
        # ==========================================
        # Stream A: Seasonal Handling (LLM-MoE + Transformer)
        # ==========================================
        # 注意：这里我们只处理 seasonal_init
        
        batch_size = seasonal_init.shape[0]
        seq_len = seasonal_init.shape[1]
        enc_out_list = [] 
        self.attention_results['group_routing_weights'] = {}
        direction_tensor = torch.tensor(self.direction).to(x_enc.device)

        # MoE Loop
        for group_label in self.group_labels:
            group_label_str = str(group_label)
            indices = torch.nonzero(direction_tensor == group_label).squeeze(-1)
            
            # 使用 seasonal 部分作为输入
            group_x = seasonal_init[:, :, indices]
            group_x_mark = x_mark_enc
            
            group_embedding = self.group_embeddings[group_label_str](group_x, group_x_mark)
            
            moe_router = self.group_moe_routers[group_label_str]
            moe_out, routing_weights = moe_router(group_embedding)
            self.attention_results['group_routing_weights'][group_label_str] = routing_weights
            
            group_out = group_embedding + self.dropout(moe_out)
            group_out = self.moe_norm(group_out)
            enc_out_list.append(group_out)

        # Fusion: Concat + Linear 
        # (比 Sum 更强，保留了组间差异信息)
        stacked_features = torch.cat(enc_out_list, dim=-1) # [B, L, Num_Groups * D]
        enc_out = self.fusion_layer(stacked_features)      # [B, L, D]
        
        # Transformer Encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Transformer Decoder (注意：Dec输入通常用全零初始化，或者 trend 的一部分，这里沿用标准做法)
        # 为了简单，这里 x_dec 可以不做分解，或者做同样的变换
        # 标准做法：Decoder 输入通常包含 label_len 的真实值 + pred_len 的占位符
        # 我们简单对 x_dec 也做 RevIN (如果不做会 scale 不匹配)
        # 但通常 x_dec 是从 x_enc 构建的，所以不需要重复 normalize
        
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        seasonal_part = dec_out[:, -self.pred_len:, :]

        # ==========================================
        # Stream B: Trend Handling (Simple Linear)
        # ==========================================
        # trend_init: [B, L, C] -> permute -> [B, C, L]
        # Linear: [B, C, L] -> [B, C, Pred_Len]
        # permute -> [B, Pred_Len, C]
        trend_part = self.trend_projection(trend_init.permute(0, 2, 1)).permute(0, 2, 1)

        # ==========================================
        # Final Sum & Denormalization
        # ==========================================
        
        final_out = seasonal_part + trend_part
        
        final_out = self.revin._denormalize(final_out)
        
        return final_out