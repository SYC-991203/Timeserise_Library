import torch
import torch.nn as nn
import torch.nn.functional as F
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None):
        # Self-attention layer
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        # Feed-forward layer
        y = x + self.dropout(self.ffn(x))
        y = self.norm2(y)
        return y, attn

class Encoder(nn.Module):
    def __init__(self, attn_layers):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
    
    def forward(self, x, attn_mask=None):
        attentions = []
        for layer in self.attn_layers:
            x, attn = layer(x, attn_mask=attn_mask)
            attentions.append(attn)
        return x, attentions
    
class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention  # Using HalfRouterAttentionLayer
        self.cross_attention = cross_attention
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, x_mask=None, cross_mask=None):
        # Self-attention using HalfRouterAttentionLayer
        new_x = self.self_attention(x)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        # Cross-attention layer
        new_x, _ = self.cross_attention(
            x, enc_out, enc_out,
            attn_mask=cross_mask
        )
        x = x + self.dropout(new_x)
        x = self.norm2(x)
        # Feed-forward layer
        y = x + self.dropout(self.ffn(x))
        y = self.norm3(y)
        return y

class Decoder(nn.Module):
    def __init__(self, layers):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, enc_out, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, enc_out, x_mask=x_mask, cross_mask=cross_mask)
        return x

class DecoderLayerWithHalfRouterWo2(nn.Module):
    def __init__(self, self_attention_layer, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(DecoderLayerWithHalfRouterWo2, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.self_attention_layer = self_attention_layer  # Instance of HalfRouterAttentionLayerWo1
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        # Self-attention with HalfRouterAttentionLayerWo1
        x2 = self.self_attention_layer(x, attn_mask=x_mask, tau=tau, delta=delta)
        x = x + self.dropout(x2)
        x = self.norm1(x)

        x2 = self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau,
            delta=delta
        )[0]
        x = x + self.dropout(x2)
        x = self.norm2(x)

        y = x
        y = y.transpose(-1, 1)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        y = y.transpose(-1, 1)
        x = x + y
        x = self.norm3(x)
        return x
