import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from layers.HalfRouterformer_EnDec import scale_block, Encoder, Decoder, DecoderLayer
from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention, HalfRouterAttentionLayer
from models.PatchTST import FlattenHead
from math import ceil
import numpy as np


import torch
import torch.nn as nn
import numpy as np
from math import ceil, sqrt
from einops import rearrange, repeat

# 假设辅助资料代码中的类已导入，例如 FullAttention, DataEmbedding, 等
def detect_change_points(sequence, num_splits):
    # 计算相邻元素之间的差异
    diffs = np.diff(sequence)
    
    # 计算差异的绝对值
    abs_diffs = np.abs(diffs)

    # 找到最大的 num_splits - 1 个差值位置作为切割点
    if num_splits > 1:
        change_points = np.argsort(abs_diffs)[- (num_splits - 1):] + 1  # +1 是因为 diff 减少了一个元素
        change_points.sort()  # 保持切割点的顺序
    else:
        change_points = []

    return change_points

class Model(nn.Module):
    """
    自定义模型 AAA，将《分割算法》集成到深度模型中，按照指定的逻辑进行修改
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.task_name = configs.task_name

        self.d_model = configs.d_model

        # 获取 Direction 参数
        self.direction = getattr(configs, 'direction', None)
        if self.direction is None:
            raise ValueError("Direction parameter must be provided in the config.")
        self.direction = configs.direction  # 例如 [0, 0, 1, 1, 1]
        self.num_groups = len(set(self.direction))  # 计算组的数量

        # 为每个组创建一个 DataEmbedding
        self.group_embeddings = nn.ModuleDict()
        for group_label in set(self.direction):
            num_channels_in_group = (np.array(self.direction) == group_label).sum()
            self.group_embeddings[str(group_label)] = DataEmbedding(
                c_in=num_channels_in_group,
                d_model=configs.d_model,
                dropout=configs.dropout
            )

        # 全局的 DataEmbedding，用于对整个输入进行嵌入
        self.embedding_global = DataEmbedding(
            c_in=configs.enc_in,
            d_model=configs.d_model,
            dropout=configs.dropout
        )

        # Time Attention Layer
        self.time_attention = AttentionLayer(
            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                          output_attention=True), configs.d_model, configs.n_heads)

        # 创建局部注意力层，为每个组创建一个
        self.local_full_attentions = nn.ModuleDict()
        for group_label in set(self.direction):
            self.local_full_attentions[str(group_label)] = AttentionLayer(
                FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                              output_attention=True), configs.d_model, configs.n_heads)

        # Dropout and Normalization layers
        self.dropout = nn.Dropout(configs.dropout)
        self.norm1 = nn.LayerNorm(configs.d_model)
        self.norm2 = nn.LayerNorm(configs.d_model)

        # MLP layers
        d_ff = configs.d_ff or 4 * configs.d_model
        self.MLP1 = nn.Sequential(
            nn.Linear(configs.d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, configs.d_model)
        )

        # 投影层，将隐藏维度映射到输出维度
        self.projection = nn.Linear(
            in_features=configs.d_model,
            out_features=1,
            bias=True
        )

        # 用于记录每一层的 attention 结果
        self.attention_results = {
            'time_attention': None,
            'local_full_attention': {}
        }

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        batch_size = x_enc.shape[0]
        seq_len = x_enc.shape[1]

        # 1. 对输入所有数据进行 DataEmbedding，不进行 self.seg_len 的处理
        x_global = self.embedding_global(x_enc, x_mark_enc)  # [batch_size, seq_len, d_model]

        # 2. 对 embedding 的结果，模型进行 time_attention 的计算
        time_in = x_global  # [batch_size, seq_len, d_model]
        time_enc, time_attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None
        )
        # 记录 time_attention 结果
        self.attention_results['time_attention'] = time_attn

        # 残差连接和归一化
        x = time_in + self.dropout(time_enc)
        x = self.norm1(x)
        x = x + self.dropout(self.MLP1(x))
        x = self.norm2(x)

        # x 形状: [batch_size, seq_len, d_model]

        # 3. 对于 time_attention 计算结果，利用《分割算法》对 time_attention 进行分割，得到 segments
        avg_attn = time_attn.mean(dim=(0, 1)).detach().cpu().numpy()  # [seq_len, seq_len]
        attention_sequence = np.diag(avg_attn)
        num_splits = 2  # 根据需要调整
        change_points = detect_change_points(attention_sequence, num_splits)

        # 准备分割点列表，包括起始和结束位置
        split_points = [0] + change_points.tolist() + [seq_len]
        segments = []
        segment_indices = []

        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            segments.append(x[:, start:end, :])  # x 是经过 time_attention 后的输出
            segment_indices.append((start, end))

        # 4. 对分割以后的 segments，根据 Direction 进行局部 MSA
        processed_segments = []
        direction = torch.tensor(self.direction).to(x_enc.device)  # Shape: [enc_in]
        group_labels = torch.unique(direction)

        for seg, (start, end) in zip(segments, segment_indices):
            # seg 形状: [batch_size, segment_length, d_model]
            segment_length = end - start
            # 创建一个张量来存储 segment 的输出 [batch_size, segment_length, enc_in, d_model]
            segment_out = torch.zeros(batch_size, segment_length, self.enc_in, self.d_model).to(x_enc.device)

            for group_label in group_labels:
                # 获取属于该组的通道索引
                indices = torch.nonzero(direction == group_label).squeeze()
                # 选择这些通道的数据
                group_x = x_enc[:, start:end, :][:, :, indices]  # [batch_size, segment_length, num_channels_in_group]
                group_x_mark = x_mark_enc[:, start:end, :]
                # 获取对应的 DataEmbedding
                group_embedding_layer = self.group_embeddings[str(group_label.item())]
                # 对这些通道进行 DataEmbedding
                group_embedding = group_embedding_layer(group_x, group_x_mark)  # [batch_size, segment_length, d_model]
                # 获取对应的局部注意力层
                local_attention = self.local_full_attentions[str(group_label.item())]
                # 应用局部 MSA
                local_enc, local_attn = local_attention(
                    group_embedding, group_embedding, group_embedding, attn_mask=None
                )
                # 记录每个组的 local_full_attention 结果
                self.attention_results['local_full_attention'][f'group_{group_label.item()}'] = local_attn

                # 残差连接和归一化
                local_out = group_embedding + self.dropout(local_enc)
                local_out = self.norm1(local_out)
                local_out = local_out + self.dropout(self.MLP1(local_out))
                local_out = self.norm2(local_out)

                # 扩展维度以匹配通道数
                num_channels_in_group = len(indices)
                local_out_expanded = local_out.unsqueeze(2).expand(-1, -1, num_channels_in_group, -1)  # [batch_size, segment_length, num_channels_in_group, d_model]

                # 修正赋值操作，避免形状不匹配
                for idx_in_group, channel_idx in enumerate(indices):
                    # 将处理后的数据赋值回 segment_out
                    segment_out[:, :, channel_idx, :] = local_out_expanded[:, :, idx_in_group, :]

            processed_segments.append(segment_out)

        # 将所有处理后的 segments 连接起来
        final_out = torch.cat(processed_segments, dim=1)  # [batch_size, seq_len, enc_in, d_model]

        # 5. 应用投影层，得到最终输出
        final_out = final_out.view(-1, self.d_model)  # [batch_size * seq_len * enc_in, d_model]
        final_out = self.projection(final_out)  # [batch_size * seq_len * enc_in, 1]
        final_out = final_out.view(batch_size, -1, self.enc_in)  # [batch_size, seq_len, enc_in]

        # 截取预测长度部分
        final_out = final_out[:, -self.pred_len:, :]  # [batch_size, pred_len, enc_in]

        return final_out  # [batch_size, pred_len, enc_in]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out  # [batch_size, pred_len, enc_in]
        return None
