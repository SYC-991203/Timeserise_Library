import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from typing import Dict, List

# 引入 TSlib 组件
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
from layers.Moe import ChannelMoE_GroupRouter,ChannelDiffMoE_GroupRouter
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
        # >>> 关键修复：先计算统计量，再进行标准化 <<<
        self._get_statistics(x) 
        
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
    改进版：MoE-Enhanced Transformer (Encoder-Decoder Architecture)
    思路：
    1. 保留 TSLib 原生 Transformer 的 Encoder-Decoder 骨架。
    2. 将 LLM-MoE 作为 "特征增强层" 插入在 Embedding 之后，Encoder 之前。
    3. 引入位置编码重注入机制，防止 MoE 破坏时序信息。
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # 1. 基础配置
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.enc_in = configs.enc_in
        self.dec_in = configs.dec_in
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.dropout = nn.Dropout(configs.dropout)

        # 2. LLM 引导配置
        self.direction = getattr(configs, 'direction', [0] * configs.enc_in)
        self.group_labels = sorted(list(set(self.direction)))
        
        init_logits_str = getattr(configs, 'moe_logits_init', {})
        init_logits_dict = json.loads(init_logits_str) if isinstance(init_logits_str, str) else init_logits_str

        # 3. Embedding 层
        # 我们使用各自独立的 Embedding，因为后续会有 Encoder/Decoder
        # Encoder 的 Embedding 改为由 MoE 处理，这里只定义 Decoder 的
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        
        # 4. MoE 组件 (替代 Encoder 的 Embedding 部分)
        self.group_embeddings = nn.ModuleDict()
        self.group_moe_routers = nn.ModuleDict()
        self.attention_results = {}

        for group_label in self.group_labels:
            group_label_str = str(group_label)
            indices = np.where(np.array(self.direction) == group_label)[0]
            c_in_group = len(indices)

            # MoE 输入端的 Embedding
            self.group_embeddings[group_label_str] = DataEmbedding(c_in_group, self.d_model, configs.dropout)
            
            specific_init_logits = init_logits_dict.get(group_label_str, [0.0, 0.0])
            # self.group_moe_routers[group_label_str] = ChannelDiffMoE_GroupRouter(configs, specific_init_logits=specific_init_logits)

            self.group_moe_routers[group_label_str] = ChannelMoE_GroupRouter(configs, specific_init_logits=specific_init_logits)

        # MoE 后的特征融合层
        self.moe_projection = nn.Linear(self.d_model, self.d_model)
        self.moe_norm = nn.LayerNorm(self.d_model)

        # 5. Transformer Encoder (处理时序)
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

        # 6. Transformer Decoder (恢复 Decoder!)
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
        # self.revin = RevIN(configs.enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            if getattr(self, 'output_moe_weights', False):
                return dec_out, self.attention_results['group_routing_weights']
            else:
                return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc = self.revin._normalize(x_enc)
        batch_size = x_enc.shape[0]
        seq_len = x_enc.shape[1]

        # ==========================================
        # Stage 1: LLM-MoE Feature Extraction (Encoder Side)
        # ==========================================
        # 我们用 MoE 替代普通的 enc_embedding 过程，或者说增强它
        
        # 初始化一个用于 Encoder 输入的大张量
        # 注意：原生 Transformer Encoder 输入通常是 [B, L, D] (混合通道)
        # 但为了利用 MoE 的分组特性，我们在 MoE 之后再合并
        
        # 这里我们需要处理一个小问题：TSlib Transformer 通常假设 global embedding。
        # 我们将重建 global embedding 的效果。
        
        enc_out_list = [] # 暂存各组的输出
        
        # 为了按正确的通道顺序拼回去，我们需要创建一个全零张量
        # 原生 Transformer 输入维度通常是 d_model，不区分通道。
        # 但是！原生 Transformer 的 DataEmbedding 是把 (B, L, C) 映射为 (B, L, D)。
        # 这意味着通道信息被混合在 D 中了。
        
        # 为了兼容，我们采取策略：
        # MoE 输出 [B, L, C_group, D] -> Sum/Mean -> [B, L, D] ???
        # 不，这样会丢失通道独立性。
        
        # === 关键修正 ===
        # 原生 Transformer 处理多元预测时，DataEmbedding(c_in, d_model) 其实是
        # Linear(c_in, d_model)。这意味着所有通道在第一层就混合了。
        # 你的 MoE 也是为了混合/独立。
        
        # 我们的策略：
        # 1. 各组分别过 Embedding 和 MoE -> [B, L, D] (每个组一个特征向量)
        # 2. 将各组的 [B, L, D] 融合为全局 [B, L, D] 喂给 Encoder
        
        # 存储每个通道位置的特征
        # 我们需要构建一个 [B, L, enc_in, D] 的张量，然后聚合
        feature_map = torch.zeros(batch_size, seq_len, self.enc_in, self.d_model).to(x_enc.device)
        self.attention_results['group_routing_weights'] = {}
        
        direction_tensor = torch.tensor(self.direction).to(x_enc.device)

        for group_label in self.group_labels:
            group_label_str = str(group_label)
            indices = torch.nonzero(direction_tensor == group_label).squeeze(-1)
            
            # 1. Group Embedding
            group_x = x_enc[:, :, indices]
            group_x_mark = x_mark_enc
            
            # [B, L, D]
            group_embedding = self.group_embeddings[group_label_str](group_x, group_x_mark)
            
            # 2. Group MoE
            moe_router = self.group_moe_routers[group_label_str]
            moe_out, routing_weights = moe_router(group_embedding)
            self.attention_results['group_routing_weights'][group_label_str] = routing_weights
            
            # 3. Residual Connection & Norm (MoE output refinement)
            # 这里的 group_embedding 含有位置信息！
            # 我们通过残差连接 group_embedding + moe_out，确保位置信息保留
            group_out = group_embedding + self.dropout(moe_out)
            group_out = self.moe_norm(group_out) # [B, L, D]
            
            # 4. 将组特征分配回对应的通道位置 (Broadcast)
            # 因为 group_out 是该组的聚合特征，我们将其赋予组内每个通道
            # 或者，更简单的方法：我们不再区分通道，而是将各组特征 加权求和 或 拼接？
            
            # 为了适配 Transformer 的输入 [B, L, D]，我们需要把所有组的信息压缩到 D 中。
            # 原生 Transformer: Linear(5, 512) -> [B, L, 512]
            # 我们的 MoE: 3个组 -> 3个 [B, L, 512]
            
            # 方案：将各组特征相加 (类似于 Multi-head 的融合)
            # 但为了保留通道特异性，我们最好使用 "Masked Sum" 或者直接将它们填回去再映射
            
            # 简化方案：直接相加所有组的特征 (假设 D 维度足够容纳所有组的信息)
            # 但这要求各组特征是对齐的。
            
            # === 更稳妥的方案：类似 iTransformer 的倒置处理 ===
            # 但为了保持原生 Transformer 结构，我们采用：
            # 将所有组的特征 [B, L, D] 放入列表，最后通过一个 Linear 层融合
            enc_out_list.append(group_out)

        # 融合所有组特征: Stack [B, L, Num_Groups, D] -> Sum -> [B, L, D]
        # 这是一个简单的融合，意味着 Transformer 看到的是所有组特征的叠加
        enc_out = torch.stack(enc_out_list, dim=-2).sum(dim=-2)
        
        # 重注入/强化位置编码 (可选，但推荐)
        # 因为上面的 Sum 可能削弱了 PE
        # enc_out = enc_out + self.dec_embedding.position_embedding(x_enc) # 伪代码，需 DataEmbedding 支持
        
        # ==========================================
        # Stage 2: Transformer Encoder
        # ==========================================
        # enc_out: [B, L, D]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # ==========================================
        # Stage 3: Transformer Decoder
        # ==========================================
        # 准备 Decoder 输入
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        
        # Cross Attention: Decoder 查询 Encoder 的输出
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        # ==========================================
        # Stage 4: Output
        # ==========================================
        # Decoder 内部已经有了 Projection: Linear(D, c_out)
        # dec_out: [B, L, c_out]  
        out = dec_out[:, -self.pred_len:, :]

        # out = self.revin._denormalize(out)
        if self.training:
            # 遍历模型中定义的所有组标签
            for group_label in self.group_labels:
                group_id = str(group_label)
                
                # 1. 获取路由概率 (Prob of Expert 0)
                # 确保该组在本次 forward 中产生了权重记录
                if group_id in self.attention_results.get('group_routing_weights', {}):
                    weights = self.attention_results['group_routing_weights'][group_id]
                    # 计算当前 Batch 对 Expert 0 的平均选择概率
                    prob_exp0 = weights[..., 0].mean().detach().item()
                    
                    # 2. 获取 Alpha 值 (Data-driven weight)
                    # 从对应的 Router 模块中获取 alpha 参数并 sigmoid
                    if group_id in self.group_moe_routers:
                        router = self.group_moe_routers[group_id]
                        # alpha 是 Parameter，需要 detach 切断梯度
                        current_alpha = torch.sigmoid(router.alpha).detach().item()
                    else:
                        current_alpha = -1.0 # 异常值占位
                    
                    # 3. 打印日志
                    # 格式：[MOE_TRACE] Group:{ID} Prob:{Prob} Alpha:{Alpha}
                    # 这样一行 log 就包含了一个组的所有关键状态
                    print(f"[MOE_TRACE] Group:{group_id} Prob:{prob_exp0:.5f} Alpha:{current_alpha:.5f}")
        return out