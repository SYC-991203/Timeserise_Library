import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.d_model = configs.d_model
        self.d_llm = 2048
        self.num_nodes = configs.enc_in # 对应变量数 N
        self.time_feat_dim = 4

        # --- 1. Embedding 层 ---
        # 遵循 iTransformer 逻辑，对变量进行转置嵌入 [cite: 737-739]
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # --- 2. Time Series Encoder (原生逻辑) [cite: 725-729] ---
        self.TS_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # --- 3. Prompt (LLM) Encoder (原生逻辑)  ---
        # 即使 Dataloader 处理完了，原生 TimeMKG 依然会对语义向量进行一次编码
        self.Prompt_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # 空间对齐层：将 LLM 的维度映射到模型的 d_model [cite: 735]
        self.LLMtodim = nn.Linear(self.d_llm, configs.d_model)

        # --- 4. Cross Attention (多模态融合层) [cite: 755] ---
        self.cross_attention = Decoder(
            [
                DecoderLayer(
                    # 第1个参数：Self-Attention 层
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=False), configs.d_model, configs.n_heads),
                    # 第2个参数：Cross-Attention 层 (这才是报错的关键)
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=False), configs.d_model, configs.n_heads),
                    # 后面才是数值参数
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        # --- 5. Decoder & Projection [cite: 757-761] ---
        self.decoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation
                ) for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def long_forecast(self, x_enc, x_mark_enc):
        # --- 标准归一化流程 ---
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        B, T, N = x_enc.size()

        # --- 1. 获取干净的 LLM Embedding ---
        # 既然是干净的 B, L, D，我们取第一个时间步（因为语义是静态的）[cite: 2041-2042]
        # 这里的 D = N * d_llm (假设你 Dataloader 将所有变量的语义向量拼在了一起)
        llm_full = x_mark_enc[:, 0, self.time_feat_dim:].unsqueeze(1) 
        # 将其还原为每个变量独立的向量形状: [B, N, d_llm]
        prompt_out = llm_full.repeat(1, N, 1)
        
        # 空间维度对齐与编码 [cite: 735, 754-755]
        prompt_out = self.LLMtodim(prompt_out) 
        prompt_out, _ = self.Prompt_encoder(prompt_out, attn_mask=None)

        # --- 2. 时序特征编码 ---
        # 提取时间特征：Month, Day, Hour 等 [cite: 960]
        real_x_mark = x_mark_enc[:, :, :self.time_feat_dim]
        enc_out = self.enc_embedding(x_enc, real_x_mark)
        enc_out, _ = self.TS_encoder(enc_out, attn_mask=None)

        # --- 3. 跨模态融合 (TimeMKG 核心) ---
        # 使用 Decoder 结构实现时序 Q 与语义 K,V 的交互 [cite: 755-756]
        dec_out = self.cross_attention(enc_out, prompt_out, x_mask=None, cross_mask=None)
        dec_out, _ = self.decoder(dec_out, attn_mask=None)
        
        # 映射回变量预测维度
        dec_out = self.projection(dec_out).permute(0, 2, 1)[:, :, :N]

        # --- 4. 反归一化 ---
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.long_forecast(x_enc, x_mark_enc)
            return dec_out[:, -self.pred_len:, :]
        return None