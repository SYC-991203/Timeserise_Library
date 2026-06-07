import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
from layers.Hypergraph import BioHypergraph_Encoder

class Model(nn.Module):
    """
    Ablation Model: w/o Hypergraph Encoder
    (去掉超图模块，仅保留 Linear Projection 和 Decoder)
    
    Structure:
    - Encoder: 仅做维度变换和线性投影 [B, Vars, Seq_Len] -> [B, Vars, D]
               (没有 BioHypergraph 层的交互)
    - Decoder: 保持不变，Cross-Attention 直接去查阅“孤立”的变量特征
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # === 1. Encoder 部分 (简化版) ===
        # [Step 1] Embedding: 依然需要！
        # 必须把时间长度 Seq_Len 压成 D_Model，否则无法和 Decoder 进行 Attention
        self.enc_embedding = nn.Linear(configs.seq_len, configs.d_model)
        
        # 【关键修改】: 移除了 self.encoder = BioHypergraph_Encoder(configs)
        # 这里什么都不加，或者你可以加一个简单的 Dropout 防止过拟合
        self.dropout = nn.Dropout(configs.dropout)
        
        # === 2. Decoder 部分 (保持原样) ===
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

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

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # === A. Encoder 流程 (Ablation) ===
        
        # 1. 维度置换: [B, Seq_Len, Num_Vars] -> [B, Num_Vars, Seq_Len]
        x_enc_transposed = x_enc.permute(0, 2, 1)
        
        # 2. 线性投影: [B, Vars, Seq_Len] -> [B, Vars, D_Model]
        # 这一步必须保留，为了对齐维度
        enc_out = self.enc_embedding(x_enc_transposed)
        
        # 【关键修改】: 跳过了超图交互
        # enc_out = self.encoder(enc_out)  <-- 这行被删掉了
        # 现在的 enc_out 代表的是“完全独立的变量特征”
        enc_out = self.dropout(enc_out)

        # === B. Decoder 流程 (保持不变) ===
        
        # 4. Decoder Embedding
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        
        # 5. Cross-Attention
        # Decoder (Time) 去查询 Encoder (Independent Variables)
        # 此时模型只能利用变量自身的历史信息，无法利用其他变量的信息
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None