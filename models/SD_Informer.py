import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding

# =============================================================================
# 1. 语义蒸馏层 (Semantic Distilling) - 【核心创新】
#    替代原始的 ConvLayer，增加 LLM 门控
# =============================================================================
# class SemanticDistilling(nn.Module):
# attention based
#     def __init__(self, c_in, llm_dim=2048):
#         super(SemanticDistilling, self).__init__()
        
#         # === 1. Cross Attention 模块 (把 LLM 当 Memory) ===
#         # 先把 LLM 维度投影到和时间序列一致
#         self.llm_proj = nn.Linear(llm_dim, c_in)
        
#         # Cross Attention: Query=TS, Key/Value=LLM
#         self.cross_attn = nn.MultiheadAttention(embed_dim=c_in, num_heads=4, batch_first=True)
#         self.norm_attn = nn.LayerNorm(c_in) # 融合后的 Norm 很重要

#         # === 2. 原始 Informer 的卷积蒸馏 (保持不变) ===
#         self.conv = nn.Conv1d(c_in, c_in, kernel_size=3, padding=1, padding_mode='circular')
#         self.norm_conv = nn.BatchNorm1d(c_in)
#         self.activation = nn.ELU()
#         self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

#     def forward(self, x, llm_emb):
#         """
#         x: [Batch, Length_Current, Channel] (长度会随着层数变短)
#         llm_emb: [Batch, Length_Original, LLM_Dim] (长度始终保持原始长度)
#         """
#         # --- Step 1: Cross Attention Fusion ---
#         # 投影 LLM 特征作为 Memory
#         memory = self.llm_proj(llm_emb) # [Batch, Length_Original, Channel]
        
#         # Attention: TS 查阅 Memory
#         # Query = x (当前层的时间序列特征)
#         # Key, Value = memory (原始的 LLM 特征)
#         # 注意：即使 x 变短了，cross_attn 也能处理，因为它只关心 Key/Value 的长度一致
#         attn_out, _ = self.cross_attn(query=x, key=memory, value=memory)
        
#         # 残差连接 + Norm (保留原始 TS 特征为主体，LLM 信息为辅助)
#         x_guided = self.norm_attn(x + attn_out)
        
#         # --- Step 2: 蒸馏 (Downsampling) ---
#         x_guided = x_guided.transpose(1, 2) # [B, C, L]
        
#         x_out = self.conv(x_guided)
#         x_out = self.norm_conv(x_out)
#         x_out = self.activation(x_out)
#         x_out = self.max_pool(x_out)
        
#         x_out = x_out.transpose(1, 2) # [B, L', C]
#         return x_out
class SemanticDistilling(nn.Module):
    def __init__(self, c_in, llm_dim=2048):
        super(SemanticDistilling, self).__init__()
        
        # 1. 只需要两个简单的映射，生成 gamma (缩放) 和 beta (平移)
        # 这里的 c_in 对应时间序列的 Channel 数
        self.scale_proj = nn.Linear(llm_dim, c_in)
        self.shift_proj = nn.Linear(llm_dim, c_in)
        
        # 2. 原始的蒸馏层 (保持不变)
        self.conv = nn.Conv1d(c_in, c_in, kernel_size=3, padding=1, padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x, llm_emb):
        """
        x: [Batch, Length, Channel]
        llm_emb: [Batch, Length, LLM_Dim] 或者 [Batch, 1, LLM_Dim]
        """
        # 如果 llm_emb 是 [Batch, Length, Dim]，我们先取平均变成全局特征 [Batch, Dim]
        # 因为语义通常是全局的，不需要每个时间步都不一样
        if llm_emb.dim() == 3:
            llm_emb_global = llm_emb.mean(dim=1) 
        else:
            llm_emb_global = llm_emb

        # 1. 计算调制参数,scale 和shift 参数保留一下
        scale = self.scale_proj(llm_emb_global).unsqueeze(1) # [Batch, 1, Channel]
        shift = self.shift_proj(llm_emb_global).unsqueeze(1) # [Batch, 1, Channel]
        
        # 2. FiLM 调制 (核心步骤)
        # 让时间序列根据语义进行自适应调整
        x_modulated = x * (1 + scale) + shift
        
        # 3. 正常的蒸馏流程
        x_in = x_modulated.permute(0, 2, 1) # [B, C, L]
        
        x_out = self.conv(x_in)
        x_out = self.norm(x_out)
        x_out = self.activation(x_out)
        x_out = self.max_pool(x_out)
        
        x_out = x_out.transpose(1, 2) # [B, L', C]
        return x_out

# =============================================================================
# 2. SD_Encoder - 【必须重写】
#    标准 Encoder 不支持传 llm_emb 参数，必须修改 forward
# =============================================================================
class SD_Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(SD_Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, llm_emb=None):
        if self.conv_layers is not None:
            # 这里的 zip 逻辑是 Informer 的特征：一层 Attention 后面接一层 Conv
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                x, _ = attn_layer(x, attn_mask=attn_mask)
                # 【关键修改】把 llm_emb 传给 SemanticDistilling
                x = conv_layer(x, llm_emb) 

            x, _ = self.attn_layers[-1](x, attn_mask=attn_mask)
        else:
            for attn_layer in self.attn_layers:
                x, _ = attn_layer(x, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x

# =============================================================================
# 3. 主模型 Model (SD-Informer)
# =============================================================================
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        
        # 请根据您使用的 LLM 维度修改这里 (Qwen-1.8B=2048)
        self.time_feat_dim = 4
        self.llm_dim = 2048 

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.llm_projection = nn.Linear(self.llm_dim, configs.d_model)
        # Encoder
        # 这里我们使用自定义的 SD_Encoder，但内部的 AttentionLayer 和 ProbAttention 是标准的
        self.encoder = SD_Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        # 直接实例化 ProbAttention，符合您的习惯
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                # 【关键替换】用 SemanticDistilling 替换 ConvLayer
                SemanticDistilling(configs.d_model, llm_dim=self.llm_dim)
                for l in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # Decoder (保持标准写法不变)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
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

    # 4. Forward 函数需要增加入口参数 llm_embedding
    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        real_x_mark_enc = x_mark_enc[:, :, :self.time_feat_dim] 
        
        # 取剩余部分为 LLM Embedding
        llm_emb = x_mark_enc[:, :, self.time_feat_dim:]
        enc_out = self.enc_embedding(x_enc, real_x_mark_enc)
        # llm_feat = self.llm_projection(llm_emb)
        # enc_out = enc_out + llm_feat
        enc_out = self.encoder(enc_out, attn_mask=None,llm_emb =llm_emb)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        return dec_out  # [B, L, D]
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None