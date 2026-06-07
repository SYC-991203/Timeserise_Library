import torch
import torch.nn as nn
# 假设 BioHypergraph_Encoder 在 layers 文件夹中
from layers.Hypergraph import BioHypergraph_Encoder

class Model(nn.Module):
    """
    Ablation Model: w/o Decoder (Encoder-Only Variant)
    
    Structure:
    - Encoder: 保留完整的 BioHypergraph (LLM + 超图交互)
    - Decoder: 移除！使用简单的 Linear Projection 直接输出预测
    
    Flow:
    Input [B, S, V] -> Transpose [B, V, S] -> Embed [B, V, D] 
    -> Hypergraph Fusion [B, V, D] -> Project [B, V, P] -> Transpose [B, P, V]
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # === 1. Encoder 部分 (核心创新保留) ===
        # [Step 1] Embedding: 时间压缩
        # Input: [B, V, Seq_Len] -> Output: [B, V, D_Model]
        self.enc_embedding = nn.Linear(configs.seq_len, configs.d_model)
        
        # [Step 2] BioHypergraph: 超图交互 (保留!)
        # 这是你的核心 Story，必须保留，否则就变成了普通的 iTransformer
        self.encoder = BioHypergraph_Encoder(configs)
        
        # === 2. Prediction 部分 (替代 Decoder) ===
        # [修改点] 移除原来的 Decoder 和 Cross-Attention
        # 直接使用一个线性层，将特征维度 D 映射为 预测长度 P
        # Input: [B, Vars, D_Model] -> Output: [B, Vars, Pred_Len]
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # === Encoder Flow ===
        
        # 1. 维度置换 (Inverted): [B, S, V] -> [B, V, S]
        x_enc_transposed = x_enc.permute(0, 2, 1)
        
        # 2. Embedding: 压缩历史信息
        enc_out = self.enc_embedding(x_enc_transposed)
        
        # 3. 超图融合: [B, V, D]
        # 在这里，变量之间交换了信息（底物告诉了产物）
        enc_out = self.encoder(enc_out)

        # === Prediction Flow (Direct Projection) ===
        
        # 4. 直接投影: [B, V, D] -> [B, V, P]
        # 模型利用融合后的特征，直接"画"出未来的曲线
        dec_out = self.projector(enc_out)
        
        # 5. 恢复维度: [B, V, P] -> [B, P, V]
        # 为了符合 Loss 计算的标准格式 [Batch, Pred_Len, Channels]
        dec_out = dec_out.permute(0, 2, 1)
        
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out  # 不需要切片了，因为出来的直接就是 pred_len
            
        return None