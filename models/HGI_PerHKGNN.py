import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Hypergraph import BioHypergraph_Encoder,DataDrivenGNN_Encoder,PearsonHKGNN_Encoder
from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import ProbAttention, AttentionLayer,FullAttention
from layers.Transformer_EncDec import Decoder, DecoderLayer


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Encoder: 空间机理编码 (已更换为超图版本)
        self.enc_embedding = nn.Linear(configs.seq_len, configs.d_model)
        # 此时传入的 configs 中应包含 k_groups 参数
        self.encoder = PearsonHKGNN_Encoder(configs)
        
        # Decoder: 处理时间维度
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                                   configs.d_model, configs.n_heads),
                    AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                                   configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation,
                ) for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 1. 空间变换
        x_enc_transposed = x_enc.permute(0, 2, 1) # [B, 5, 96]
        enc_out = self.enc_embedding(x_enc_transposed) # [B, 5, 512]
        
        # 2. 超图卷积提取变量间机理特征
        enc_out = self.encoder(enc_out) # [B, 5, 512]

        # 3. 解码生成
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None