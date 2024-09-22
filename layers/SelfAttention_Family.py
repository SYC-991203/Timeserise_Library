import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat


class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=configs.output_attention), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out


class HalfRouterAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(HalfRouterAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.local_full_attention =  AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=configs.output_attention), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        ts_d = x.shape[1] 
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Split the input into two halves
        dim_in_first_half = dim_in[:3*batch, :, :]
        dim_in_second_half = dim_in[3*batch:, :, :]

        ## 对于前三通道有机理关系的做局部的MSA
        loacl_dim_in_first_half = rearrange(dim_in_first_half, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        local_dim_enc,local_atten = self.local_full_attention(loacl_dim_in_first_half,loacl_dim_in_first_half,loacl_dim_in_first_half,attn_mask=None, tau=None, delta=None)

        local_dim_in = loacl_dim_in_first_half + self.dropout(local_dim_enc)
        local_dim_in = self.norm1(local_dim_in)
        local_dim_in = local_dim_in + self.dropout(self.MLP1(local_dim_in))
        local_dim_in = self.norm2(local_dim_in)

        # 上述的输出拼接处理
        local_dim_out = rearrange(local_dim_in, '(b seg_num) ts_d d_model -> (b ts_d) seg_num d_model', b=batch)
        local_out_cat = torch.cat([local_dim_out, dim_in_second_half], dim=0)

        ## 再整体做Router attention
        dim_send = rearrange(local_out_cat, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        # 检查和打印维度以确认匹配

        dim_buffer_first_half, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive_first_half, attn = self.dim_receiver(dim_send, dim_buffer_first_half, dim_buffer_first_half, attn_mask=None, tau=None, delta=None)
        dim_enc_first_half = dim_send + self.dropout(dim_receive_first_half)  # 残差连接
        dim_enc_first_half = self.norm3(dim_enc_first_half)  # 应用 LayerNorm
        dim_enc_first_half = dim_enc_first_half + self.dropout(self.MLP2(dim_enc_first_half))  # MLP 层和残差连接
        dim_enc_first_half = self.norm4(dim_enc_first_half)  # 再次应用 LayerNorm

        # 将处理后的前3个通道和未处理的后2个通道拼接
        dim_enc_first_half = rearrange(dim_enc_first_half, '(b seg_num) ts_d d_model -> (b ts_d) seg_num d_model', b=batch)
        final_out = rearrange(dim_enc_first_half, '(b ts_d) seg_num d_model -> b ts_d seg_num d_model', b=batch)


        return final_out


class DirectionAttentionLayerV0(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1,direction = None):
        super(DirectionAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        if direction is None:
            self.direction = [1] * configs.enc_in  # 或者其他默认值
        else:
            self.direction = direction
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.local_full_attention =  AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=configs.output_attention), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x shape: [batch_size, ts_d, seg_num, d_model]
        batch_size = x.shape[0]
        ts_d = x.shape[1]
        seg_num = x.shape[2]
        d_model = x.shape[3]

        # 初始的时间维度注意力
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # 将 dim_in 重新变形为 [batch_size, ts_d, seg_num, d_model]
        dim_in = dim_in.view(batch_size, ts_d, seg_num, d_model)

        # 将 self.direction 转换为张量并获取参与和不参与局部 MSA 的索引
        direction = torch.tensor(self.direction).to(x.device)
        indices_1 = torch.nonzero(direction == 1).squeeze()
        indices_0 = torch.nonzero(direction == 0).squeeze()

        # 分别获取参与和不参与局部 MSA 的通道
        dim_in_first_half = dim_in[:, indices_1, :, :]  # 形状：[batch_size, ts_d', seg_num, d_model]
        dim_in_second_half = dim_in[:, indices_0, :, :]  # 形状：[batch_size, ts_d'', seg_num, d_model]

        # 对参与局部 MSA 的通道进行处理
        local_dim_in_first_half = rearrange(dim_in_first_half, 'b ts_d seg_num d_model -> (b seg_num) ts_d d_model')
        local_dim_enc, local_attn = self.local_full_attention(
            local_dim_in_first_half, local_dim_in_first_half, local_dim_in_first_half, attn_mask=None, tau=None, delta=None
        )

        local_dim_in = local_dim_in_first_half + self.dropout(local_dim_enc)
        local_dim_in = self.norm1(local_dim_in)
        local_dim_in = local_dim_in + self.dropout(self.MLP1(local_dim_in))
        local_dim_in = self.norm2(local_dim_in)

        # 将处理后的结果重新变形回原来的形状
        local_dim_out = rearrange(local_dim_in, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch_size)

        # 准备用于 Router Attention 的输入
        dim_in_processed = torch.zeros_like(dim_in)
        dim_in_processed[:, indices_1, :, :] = local_dim_out  # 放入处理过的通道
        dim_in_processed[:, indices_0, :, :] = dim_in[:, indices_0, :, :]  # 放入未处理的通道

        # 进行 Router Attention
        dim_send = rearrange(dim_in_processed, 'b ts_d seg_num d_model -> (b seg_num) ts_d d_model')

        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch_size)

        dim_buffer_first_half, attn = self.dim_sender(
            batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None
        )
        dim_receive_first_half, attn = self.dim_receiver(
            dim_send, dim_buffer_first_half, dim_buffer_first_half, attn_mask=None, tau=None, delta=None
        )
        dim_enc_first_half = dim_send + self.dropout(dim_receive_first_half)
        dim_enc_first_half = self.norm3(dim_enc_first_half)
        dim_enc_first_half = dim_enc_first_half + self.dropout(self.MLP2(dim_enc_first_half))
        dim_enc_first_half = self.norm4(dim_enc_first_half)

        # 将结果重新变形回原来的形状
        final_out = rearrange(dim_enc_first_half, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch_size)

        return final_out

class DirectionAttentionLayer(nn.Module):
    '''
    Modified HalfRouterAttentionLayer according to the specified logic.
    Input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs, seg_num, factor, d_model, n_heads, direction, d_ff=None, dropout=0.1):
        super(DirectionAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.direction = direction  # Direction is a list of group labels, e.g., [0,0,1,1,1]
        self.num_groups = len(set(direction))  # Number of unique groups

        self.time_attention = AttentionLayer(
            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                          output_attention=configs.output_attention), d_model, n_heads)

        # Create a local attention layer for each group
        self.local_full_attentions = nn.ModuleDict()
        for group_label in set(direction):
            self.local_full_attentions[str(group_label)] = AttentionLayer(
                FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                              output_attention=configs.output_attention), d_model, n_heads)

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x shape: [batch_size, ts_d, seg_num, d_model]
        batch_size, ts_d, seg_num, d_model = x.shape

        # Cross Time Stage: Directly apply MSA to each dimension (channel)
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')

        # Apply time attention
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Reshape back to [batch_size, ts_d, seg_num, d_model]
        dim_in = dim_in.view(batch_size, ts_d, seg_num, d_model)

        # Convert direction to tensor on the same device
        direction = torch.tensor(self.direction).to(x.device)  # Shape: [ts_d]

        # Get unique group labels
        group_labels = torch.unique(direction)

        group_outputs = []  # To collect outputs for each group

        for group_label in group_labels:
            # Get indices of channels belonging to this group
            indices = torch.nonzero(direction == group_label).squeeze()
            # Select data for these channels
            group_dim_in = dim_in[:, indices, :, :]  # Shape: [batch_size, num_channels_in_group, seg_num, d_model]

            # Rearrange for local attention
            # Since we are applying local MSA across channels for each segment,
            # we need to rearrange to [ (batch_size * seg_num), num_channels_in_group, d_model ]
            group_dim_in_reshaped = rearrange(group_dim_in, 'b ts_d seg_num d_model -> (b seg_num) ts_d d_model')

            # Get the local attention layer for this group
            local_attention = self.local_full_attentions[str(group_label.item())]

            # Apply local attention
            local_dim_enc, local_attn = local_attention(
                group_dim_in_reshaped, group_dim_in_reshaped, group_dim_in_reshaped, attn_mask=None, tau=None, delta=None
            )

            # Residual connection and normalization
            local_dim_in = group_dim_in_reshaped + self.dropout(local_dim_enc)
            local_dim_in = self.norm1(local_dim_in)
            local_dim_in = local_dim_in + self.dropout(self.MLP1(local_dim_in))
            local_dim_in = self.norm2(local_dim_in)

            # Reshape back to original shape
            group_dim_out = rearrange(local_dim_in, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch_size)

            # Collect outputs and indices
            group_outputs.append((indices, group_dim_out))

        # Now assemble the outputs back
        final_out = torch.zeros_like(dim_in).to(x.device)

        for indices, group_dim_out in group_outputs:
            # Place the group outputs back into final_out
            final_out[:, indices, :, :] = group_dim_out  # Indices along ts_d dimension

        # final_out is of shape [batch_size, ts_d, seg_num, d_model]
        return final_out
