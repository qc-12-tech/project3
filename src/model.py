"""从零实现的 Transformer Encoder 文本分类器（外卖评价三分类）。

结构：字符 Embedding + 正弦位置编码 + N 层 Encoder + Masked Mean Pooling + 线性分类头。
参考 testcode/transformer.py 的实现风格（Post-LN）。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config as C


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=C.MAX_LEN, dropout=C.DROPOUT):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))          # [1, max_len, d_model]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return torch.matmul(attn, v), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=C.D_MODEL, num_heads=C.NUM_HEADS, dropout=C.DROPOUT):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, Lq = query.size(0), query.size(1)
        Lk = key.size(1)
        q = self.w_q(query).view(B, Lq, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(B, Lk, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(B, Lk, self.h, self.d_k).transpose(1, 2)
        out, attn = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.w_o(out), attn


class AddNorm(nn.Module):
    def __init__(self, d_model=C.D_MODEL, dropout=C.DROPOUT):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sub_out):
        return self.norm(x + self.dropout(sub_out))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=C.D_MODEL, d_ff=C.D_FF, dropout=C.DROPOUT):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model=C.D_MODEL, num_heads=C.NUM_HEADS,
                 d_ff=C.D_FF, dropout=C.DROPOUT):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, x, attn_mask):
        attn_out, _ = self.self_attn(x, x, x, attn_mask)
        x = self.add_norm1(x, attn_out)
        x = self.add_norm2(x, self.ffn(x))
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=C.D_MODEL, num_heads=C.NUM_HEADS,
                 num_layers=C.NUM_LAYERS, d_ff=C.D_FF, dropout=C.DROPOUT,
                 num_classes=C.NUM_CLASSES, max_len=C.MAX_LEN, pad_id=C.PAD_ID):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def encode(self, input_ids, attention_mask):
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(2)      # [B,1,1,L]
        x = self.embed(input_ids) * math.sqrt(self.d_model)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, pad_mask)
        x = self.norm(x)
        # Masked mean pooling
        mask = attention_mask.unsqueeze(-1).float()              # [B,L,1]
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return x

    def forward(self, input_ids, attention_mask):
        pooled = self.encode(input_ids, attention_mask)
        return self.classifier(self.dropout(pooled))
