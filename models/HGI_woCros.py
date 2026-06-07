import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
# 假设 BioHypergraph_Encoder 在 layers 文件夹中
from layers.Hypergraph import BioHypergraph_Encoder

class Model(nn.Module):
    """
    Ablation Model: w/o Cross-Attention (LHC-Net w/o Retrieval)
    
    Structure:
    - Encoder: BioHypergraph (正常工作，构建知识库)
    - Decoder: 仅保留 Self-Attention (类似于 GPT 的纯自回归结构，但不看 Encoder)
    
    Purpose:
    证明"查阅知识库"是必要的。如果这个模型效果差，说明 Encoder 里的机理信息
    必须通过 Cross-Attention 才能传输给预测步骤。
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # === 1. Encoder 部分 (保持完整) ===
        # 我们依然计算它，证明"即使有知识库，不查也没用"
        self.enc_embedding = nn.Linear(configs.seq_len, configs.d_model)
        self.encoder = BioHypergraph_Encoder(configs)
        
        # === 2. Decoder 部分 (去掉了 Cross-Attention) ===
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # 【关键修改】
        # 使用 Encoder 类的结构来充当"只有 Self-Attention 的 Decoder"
        # 这里的 Encoder 实际上处理的是 x_dec，起到了 Decoder 的作用，只是没有 Cross-Attention 接口
        self.decoder_self_only = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        # 仅保留 Self-Attention
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.predcition=nn.Linear(configs.d_model, configs.c_out, bias=True)


    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # === A. Encoder 流程 (正常运行) ===
        x_enc_transposed = x_enc.permute(0, 2, 1)
        enc_out = self.enc_embedding(x_enc_transposed)
        
        # [知识库构建]
        # enc_out 包含了丰富的机理特征
        enc_out = self.encoder(enc_out) 

        # === B. Decoder 流程 (断连) ===
        
        # 1. Embedding
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        
        # 2. Self-Attention Only
        # 【关键点】这里我们完全忽略了上面的 enc_out
        # 也就是：预测过程完全不看知识库，只看自己过去的序列片段(x_dec)
        dec_out, attns = self.decoder_self_only(dec_out, attn_mask=None)
        dec_out  = self.predcition(dec_out)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None