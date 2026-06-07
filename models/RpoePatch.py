import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# 依赖 thuml/Time-Series-Library 的层实现（与示例风格保持一致）
from layers.Embed import DataEmbedding  # 仅作兜底备用，主流程用 CLIPRoPEEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer


# =====================
# 数值/时间 RoPE 特征
# =====================
class NumericRoPEFeatures(nn.Module):
    """将标量 x 映射为 2*n_freq 维 [cos(x*θ), sin(x*θ)]；支持可学习 base 与可选缩放。
    输入形状：[B,L,D]；输出：[B,L,D,2F]
    """
    def __init__(self, n_freq: int = 16, base: float = 10000.0, learnable_base: bool = True,
                 value_scale: Optional[float] = None):
        super().__init__()
        self.n_freq = n_freq
        self.value_scale = value_scale
        if learnable_base:
            self.log_base = nn.Parameter(torch.tensor(math.log(base), dtype=torch.float32))
        else:
            self.register_buffer('log_base', torch.tensor(math.log(base), dtype=torch.float32), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.value_scale is not None:
            x = x / self.value_scale
        B, L, D = x.shape
        device = x.device
        Freq = self.n_freq
        base = torch.exp(self.log_base)
        i = torch.arange(Freq, device=device, dtype=torch.float32)
        theta = torch.pow(base, -2 * i / max(1.0, float(Freq)))  # [F]
        P = x.unsqueeze(-1) * theta.view(1, 1, 1, -1)            # [B,L,D,F]
        return torch.cat([torch.cos(P), torch.sin(P)], dim=-1)   # [B,L,D,2F]


# =====================
# 训练期来自 numeric_clip_align 的“数值分词器” → 时序中的逐时刻 token
# =====================
class NumericTokenizerFromCKPT(nn.Module):
    """把每个时间步的 C 维数值编码为 d_model 维 token。
    - 复用 NumericEncoder 的：field_emb、value_rope.log_base、value_proj
    - pool 选项：'attn'（字段注意力池化）/ 'mean'（均值）/ 'none'（不聚合，返回 [B,L,C,d]）
    """
    def __init__(self, num_fields: int, d_model: int, n_value_freq: int, value_scale: Optional[float] = None,
                 pool: str = 'attn'):
        super().__init__()
        self.num_fields = num_fields
        self.d_model = d_model
        self.n_value_freq = n_value_freq
        self.value_rope = NumericRoPEFeatures(n_freq=n_value_freq, learnable_base=True, value_scale=value_scale)
        self.value_proj = nn.Linear(2 * n_value_freq, d_model)
        self.field_emb = nn.Embedding(num_fields, d_model)
        self.pool = pool
        if pool == 'attn':
            self.query = nn.Parameter(torch.randn(1, 1, d_model))  # [1,1,d]

    @staticmethod
    def from_ckpt(ckpt_path: str, pool: str = 'attn', value_scale: Optional[float] = None):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        st = ckpt['numeric_encoder']
        num_fields = ckpt['num_fields']
        d_model = st['field_emb.weight'].shape[1]
        in_dim = st['value_proj.weight'].shape[1]
        n_value_freq = in_dim // 2
        tok = NumericTokenizerFromCKPT(num_fields, d_model, n_value_freq, value_scale=value_scale, pool=pool)
        tok.field_emb.load_state_dict({'weight': st['field_emb.weight']})
        tok.value_proj.load_state_dict({'weight': st['value_proj.weight'], 'bias': st['value_proj.bias']})
        with torch.no_grad():
            tok.value_rope.log_base.copy_(st['value_rope.log_base'])
        return tok

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,L,C] → out: [B,L,d] 或 [B,L,C,d]（当 pool='none' 时）"""
        B, L, C = x.shape
        field_ids = torch.arange(C, device=x.device).view(1, 1, C).expand(B, L, C)
        rope_feat = self.value_rope(x)                          # [B,L,C,2F]
        vtok = self.value_proj(rope_feat)                       # [B,L,C,d]
        ftok = self.field_emb(field_ids)                        # [B,L,C,d]
        tokens = vtok + ftok                                    # [B,L,C,d]
        if self.pool == 'none':
            return tokens                                       # [B,L,C,d]
        if self.pool == 'mean':
            return tokens.mean(dim=2)                           # [B,L,d]
        # attn pooling
        q = self.query.expand(B, L, -1)                         # [B,L,d]
        attn = (tokens * q.unsqueeze(2)).sum(dim=-1) / math.sqrt(self.d_model)  # [B,L,C]
        attn = attn.softmax(dim=-1)
        out = (tokens * attn.unsqueeze(-1)).sum(dim=2)          # [B,L,d]
        return out


# =====================
# 时间 RoPE 投影
# =====================
class TimeRoPEProjector(nn.Module):
    def __init__(self, d_model: int, n_time_freq: int = 8):
        super().__init__()
        self.time_rope = NumericRoPEFeatures(n_freq=n_time_freq, learnable_base=True, value_scale=None)
        self.proj = nn.Linear(2 * n_time_freq, d_model)

    def forward(self, x_mark: torch.Tensor) -> torch.Tensor:
        # x_mark: [B,L,K]（thuml 的时间特征），数值化后直接做 RoPE
        tfeat = self.time_rope(x_mark)   # [B,L,K,2F]
        tfeat = tfeat.mean(dim=2)        # 聚合 K
        return self.proj(tfeat)          # [B,L,d]


# =====================
# 融合后的 Embedding（数值 token + 时间 token）
# =====================
class CLIPRoPEEmbedding(nn.Module):
    def __init__(self, num_fields: int, d_model: int, n_value_freq: int, n_time_freq: int,
                 numeric_ckpt: Optional[str] = None, freeze_numeric: bool = True, pool: str = 'attn',
                 value_scale: Optional[float] = None, dropout: float = 0.0, per_channel: bool = False):
        super().__init__()
        # 通道独立时强制返回每通道 token（pool='none'），否则按指定 pool 聚合
        effective_pool = 'none' if per_channel else pool
        if numeric_ckpt is not None:
            self.numtok = NumericTokenizerFromCKPT.from_ckpt(numeric_ckpt, pool=effective_pool, value_scale=value_scale)
        else:
            self.numtok = NumericTokenizerFromCKPT(num_fields, d_model, n_value_freq, value_scale=value_scale, pool=effective_pool)
        if freeze_numeric:
            for p in self.numtok.parameters():
                p.requires_grad = False
        self.timeproj = TimeRoPEProjector(d_model, n_time_freq)
        self.dropout = nn.Dropout(dropout)
        self.per_channel = per_channel

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor) -> torch.Tensor:
        # x: [B,L,C], x_mark: [B,L,K]
        vtok = self.numtok(x)                       # [B,L,d] 或 [B,L,C,d]
        ttok = self.timeproj(x_mark)                # [B,L,d]
        if self.per_channel:
            ttok = ttok.unsqueeze(2)                # [B,L,1,d]，对每通道广播
        etok = vtok + ttok                          # [B,L,d] 或 [B,L,C,d]
        return self.dropout(etok)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, rope_base: float = 10000.0, learnable: bool = True):
        super().__init__()
        if learnable:
            self.log_base = nn.Parameter(torch.tensor(math.log(rope_base), dtype=torch.float32))
        else:
            self.register_buffer('log_base', torch.tensor(math.log(rope_base), dtype=torch.float32), persistent=False)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).view_as(x)

    def apply(self, q: torch.Tensor, k: torch.Tensor):
        B, H, L, Dh = q.shape
        assert Dh % 2 == 0
        base = torch.exp(self.log_base)
        half = Dh // 2
        i = torch.arange(half, device=q.device, dtype=torch.float32)
        theta = torch.pow(base, -2 * i / max(1.0, float(half)))
        pos = torch.arange(L, device=q.device, dtype=torch.float32).unsqueeze(1)
        ang = pos * theta.view(1, -1)
        cos = torch.cos(ang); sin = torch.sin(ang)
        cos = torch.stack([cos, cos], dim=-1).view(L, -1)
        sin = torch.stack([sin, sin], dim=-1).view(L, -1)
        while cos.dim() < q.dim():
            cos = cos.unsqueeze(0); sin = sin.unsqueeze(0)
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot
    def __init__(self, rope_base: float = 10000.0, learnable: bool = True):
        super().__init__()
        if learnable:
            self.log_base = nn.Parameter(torch.tensor(math.log(rope_base), dtype=torch.float32))
        else:
            self.register_buffer('log_base', torch.tensor(math.log(rope_base), dtype=torch.float32), persistent=False)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).view_as(x)

    def apply(self, q: torch.Tensor, k: torch.Tensor):
        # q,k: [B, H, L, Dh]
        B, H, L, Dh = q.shape
        assert Dh % 2 == 0
        base = torch.exp(self.log_base)
        half = Dh // 2
        i = torch.arange(half, device=q.device, dtype=torch.float32)
        theta = torch.pow(base, -2 * i / max(1.0, float(half)))  # [half]
        pos = torch.arange(L, device=q.device, dtype=torch.float32).unsqueeze(1)  # [L,1]
        ang = pos * theta.view(1, -1)  # [L,half]
        cos = torch.cos(ang)
        sin = torch.sin(ang)
        cos = torch.stack([cos, cos], dim=-1).view(L, -1)  # [L,Dh]
        sin = torch.stack([sin, sin], dim=-1).view(L, -1)
        # broadcast 到 [B,H,L,Dh]
        while cos.dim() < q.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot


class RoPEFullAttention(nn.Module):
    """与库内 FullAttention 接口一致：先对 Q/K 施加旋转，再调用 FullAttention。
    用法：AttentionLayer(RoPEFullAttention(FullAttention(...)), d_model, n_heads)
    """
    def __init__(self, inner_full_attn: FullAttention, rope_base: float = 10000.0, learnable: bool = True):
        super().__init__()
        self.inner = inner_full_attn
        self.rope = RotaryPositionalEmbedding(rope_base=rope_base, learnable=learnable)

    def forward(self, queries, keys, values, attn_mask, **kwargs):
        # 期望形状与库保持一致：[B, H, L, Dh]
        q_rot, k_rot = self.rope.apply(queries, keys)
        return self.inner(q_rot, k_rot, values, attn_mask, **kwargs)


# =====================
# 主模型（与 thuml 示例风格保持一致的接口）
# =====================
class Model(nn.Module):
    """
    CLIP 数值分词器 + 时间 RoPE 的 DataEmbedding，
    编码器/解码器自注意力替换为 RoPEFullAttention（可配），其余保持与库一致。

    兼容任务：long_term_forecast / short_term_forecast / imputation / anomaly_detection / classification
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # 额外超参（给默认值以兼容原主程序）
        self.numeric_ckpt = getattr(configs, 'numeric_ckpt', None)
        self.freeze_numeric = getattr(configs, 'freeze_numeric', True)
        self.n_value_freq = getattr(configs, 'n_value_freq', 16)
        self.n_time_freq = getattr(configs, 'n_time_freq', 8)
        self.pool = getattr(configs, 'pool', 'attn')              # 'attn' / 'mean'
        self.value_scale = getattr(configs, 'value_scale', None)  # 可传入稳健尺度（如 MAD 均值）
        self.use_rope_attn = getattr(configs, 'use_rope_attn', True)
        self.channel_independent = getattr(configs, 'channel_independent', False)

        # ============ Embedding ============
        # 仅在预测任务上使能“通道独立”的逐通道 token；其它任务默认通道融合
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.enc_embedding = CLIPRoPEEmbedding(
                num_fields=configs.enc_in, d_model=configs.d_model,
                n_value_freq=self.n_value_freq, n_time_freq=self.n_time_freq,
                numeric_ckpt=self.numeric_ckpt, freeze_numeric=self.freeze_numeric,
                pool=self.pool, value_scale=self.value_scale, dropout=configs.dropout,
                per_channel=self.channel_independent
            )
            self.dec_embedding = CLIPRoPEEmbedding(
                num_fields=configs.dec_in, d_model=configs.d_model,
                n_value_freq=self.n_value_freq, n_time_freq=self.n_time_freq,
                numeric_ckpt=self.numeric_ckpt, freeze_numeric=self.freeze_numeric,
                pool=self.pool, value_scale=self.value_scale, dropout=configs.dropout,
                per_channel=self.channel_independent
            )
        else:
            self.enc_embedding = CLIPRoPEEmbedding(
                num_fields=configs.enc_in, d_model=configs.d_model,
                n_value_freq=self.n_value_freq, n_time_freq=self.n_time_freq,
                numeric_ckpt=self.numeric_ckpt, freeze_numeric=self.freeze_numeric,
                pool=self.pool, value_scale=self.value_scale, dropout=configs.dropout,
                per_channel=False
            )

        # ============ Encoder ============
        def make_attn(mask_flag):
            inner = FullAttention(mask_flag, configs.factor, attention_dropout=configs.dropout,
                                  output_attention=configs.output_attention)
            if self.use_rope_attn:
                return RoPEFullAttention(inner)
            return inner

        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(make_attn(False), configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation
            ) for _ in range(configs.e_layers)
        ], norm_layer=nn.LayerNorm(configs.d_model))

        # ============ Decoder / Heads ============
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.decoder = Decoder([
                DecoderLayer(
                    AttentionLayer(make_attn(True), configs.d_model, configs.n_heads),
                    AttentionLayer(make_attn(False), configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation
                ) for _ in range(configs.d_layers)
            ], norm_layer=nn.LayerNorm(configs.d_model),
               projection=(nn.Identity() if self.channel_independent else nn.Linear(configs.d_model, configs.c_out, bias=True)))
            if self.channel_independent:
                self.ci_out = nn.Linear(configs.d_model, 1, bias=True)

        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    # ============ Task impls ============
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if getattr(self, 'channel_independent', False):
            B, L, C = x_enc.shape
            enc_tok = self.enc_embedding(x_enc, x_mark_enc)      # [B,L,C,d]
            enc_tok = enc_tok.permute(0, 2, 1, 3).reshape(B * C, L, -1)
            enc_out, attns = self.encoder(enc_tok, attn_mask=None)

            Ld = x_dec.size(1)
            dec_tok = self.dec_embedding(x_dec, x_mark_dec)      # [B,Ld,C,d]
            dec_tok = dec_tok.permute(0, 2, 1, 3).reshape(B * C, Ld, -1)
            dec_out = self.decoder(dec_tok, enc_out, x_mask=None, cross_mask=None)  # [B*C,Ld,d] (Identity)
            y = self.ci_out(dec_out)                             # [B*C,Ld,1]
            y = y.view(B, C, Ld, 1)[:, :, -self.pred_len:, 0]   # [B,C,pred_len]
            return y.permute(0, 2, 1)                           # [B,pred_len,C]
        # 通道融合路径（默认）
        enc_out = self.enc_embedding(x_enc, x_mark_enc)      # [B, L, d]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        return self.projection(enc_out)

    def anomaly_detection(self, x_enc):
        enc_out = self.enc_embedding(x_enc, torch.zeros(x_enc.size(0), x_enc.size(1), 1, device=x_enc.device))
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        return self.projection(enc_out)

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        output = F.gelu(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # 遵循原实现：对 padding 位置置零
        output = output.reshape(output.shape[0], -1)
        return self.projection(output)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]   # [B, L_pred, D]
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None
