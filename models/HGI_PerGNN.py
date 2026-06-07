import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Hypergraph import BioHypergraph_Encoder,DataDrivenGNN_Encoder,PearsonGNN_Encoder
from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import ProbAttention, AttentionLayer,FullAttention
from layers.Transformer_EncDec import Decoder, DecoderLayer

class Model(nn.Module):
    """
    BioInformer (Lean Version)
    仅保留 Long Term Forecast 任务接口。
    
    Structure:
    - Encoder: BioHypergraph_Encoder (处理变量维度的机理交互) [B, Vars, D]
    - Decoder: Informer Decoder (处理时间维度的序列生成) [B, Time, D]
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # === 1. Encoder 部分: 空间机理编码 ===
        # [Step 1] Embedding: 将时间窗口 Seq_Len 投影为特征维度 D_Model
        # Input: [Batch, Num_Vars, Seq_Len] -> Output: [Batch, Num_Vars, D_Model]
        # 这取代了原来的 DataEmbedding，因为我们在 Encoder 阶段不再看"序列"，而是看"变量"
        self.enc_embedding = nn.Linear(configs.seq_len, configs.d_model)
        
        # [Step 2] BioHypergraph: 在变量之间进行超图交互
        # Input: [Batch, Num_Vars, D_Model] -> Output: [Batch, Num_Vars, D_Model]
        self.encoder = PearsonGNN_Encoder(configs)
        
        # === 2. Decoder 部分: Informer 时序生成 ===
        # Decoder 依然处理时间序列，所以保留完整的 DataEmbedding (Value + Position + Temporal)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Informer Decoder: 使用 ProbSparse Attention 生成未来序列
        self.decoder = Decoder(
            [
                DecoderLayer(
                    # [修改点 2] Self-Attention: 使用 FullAttention
                    # output_attention=True/False 取决于你是否需要可视化 Attention Map
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                        
                    # [修改点 3] Cross-Attention: 使用 FullAttention
                    # Query: Time, Key/Value: Variables
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

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # === A. Encoder 流程 (Variable-Centric) ===
        
        # 1. 维度置换: [B, Seq_Len, Num_Vars] -> [B, Num_Vars, Seq_Len]
        # 我们把变量维度移到序列长度的位置，把"Variable"视为新的"Token"
        x_enc_transposed = x_enc.permute(0, 2, 1)
        
        # 2. 线性投影: [B, Vars, Seq_Len] -> [B, Vars, D_Model]
        # 将历史时间窗口压缩为一个特征向量
        enc_out = self.enc_embedding(x_enc_transposed)
        
        # 3. 超图交互: [B, Vars, D_Model]
        # 变量之间根据 LLM 定义的机理进行交互
        enc_out = self.encoder(enc_out)

        # === B. Decoder 流程 (Time-Centric) ===
        
        # 4. Decoder Embedding: [B, Label+Pred, Num_Vars] -> [B, Label+Pred, D_Model]
        # Decoder 依然按照传统方式处理时间步
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        
        # 5. Cross-Attention 解码
        # x_mask 和 cross_mask 设为 None，Informer 内部会自动处理因果 Mask
        # 此时 Cross Attention 的物理意义：用"未来时刻"去查询"变量机理状态"
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        
        # Output shape: [B, Label+Pred, C_Out]
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # 切片只返回预测部分，符合 TSlib 标准接口
            return dec_out[:, -self.pred_len:, :]
            
        return None