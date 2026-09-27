"""从零实现的 decoder-only Transformer（GPT 式），无 encoder、无预训练。

结构要点：
  - Token Embedding（与 LM Head 权重共享）+ **RoPE** 旋转位置编码（无可学习位置表）
  - Pre-RMSNorm + Causal Self-Attention（多头，用 SDPA 的 is_causal 掩码）
  - SwiGLU 前馈 + 残差
  - 生成式输出：在受限词表 {好,差} 与 {1..5} 上取概率分布
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, POS_CHAR, NEG_CHAR, RATING_CHARS


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def build_rope_cache(head_dim: int, max_len: int, theta: float, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, inv)                      # [L, head_dim/2]
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x, cos, sin):
    """x: [B, H, L, D]，旋转前半/后半维度。"""
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    out = torch.stack([o1, o2], dim=-1)
    return out.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.resid_scale = 1.0 / math.sqrt(2 * cfg.n_layer)

    def forward(self, x, cos, sin, past=None, use_cache: bool = False):
        B, L, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.cfg.d_model, dim=-1)
        q = q.view(B, L, self.cfg.n_head, self.cfg.head_dim).transpose(1, 2)
        k = k.view(B, L, self.cfg.n_head, self.cfg.head_dim).transpose(1, 2)
        v = v.view(B, L, self.cfg.n_head, self.cfg.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        new_cache = None
        if past is not None:
            pk, pv = past
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
            # 增量解码：q 只对全部 k 做注意力，无需因果掩码
            o = F.scaled_dot_product_attention(q, k, v)
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if use_cache:
            new_cache = (k, v)
        o = o.transpose(1, 2).contiguous().view(B, L, -1)
        o = self.drop(self.proj(o))
        return o * self.resid_scale, new_cache


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w3 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.resid_scale = 1.0 / math.sqrt(2 * cfg.n_layer)

    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x))) * self.resid_scale


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, past=None, use_cache: bool = False):
        a, cache = self.attn(self.n1(x), cos, sin, past=past, use_cache=use_cache)
        x = x + a
        x = x + self.ffn(self.n2(x))
        return x, cache


class GPT(nn.Module):
    """decoder-only 因果语言模型 + 两个受限生成头（情感 / 评分）。"""

    def __init__(self, cfg: ModelConfig, stoi: dict):
        super().__init__()
        self.cfg = cfg
        self.stoi = stoi
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w3.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
        # 受限词表 id
        self.pos_id = stoi[POS_CHAR]
        self.neg_id = stoi[NEG_CHAR]
        self.rating_ids = [stoi[c] for c in RATING_CHARS]
        self.sent_threshold = 0.5          # 训练后用验证集 F1 校准（见 train.py）
        self.register_buffer("rope_cos", torch.zeros(1), persistent=False)
        self.register_buffer("rope_sin", torch.zeros(1), persistent=False)
        self._rope_len = 0

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _rope(self, length: int, device, dtype):
        if self._rope_len < length or self.rope_cos.device != device:
            cos, sin = build_rope_cache(self.cfg.head_dim, max(length, self.cfg.max_len),
                                        self.cfg.rope_theta, device, torch.float32)
            self.rope_cos, self.rope_sin = cos, sin
            self._rope_len = cos.shape[0]
        return self.rope_cos[:length].to(dtype), self.rope_sin[:length].to(dtype)

    def encode(self, ids, past=None, use_cache: bool = False, pos_offset: int = 0):
        """[B, L] -> 隐状态 [B, L, d]（未经 final norm）+ 可选 KV cache。"""
        B, L = ids.shape
        x = self.drop(self.tok_emb(ids))
        cos, sin = self._rope(pos_offset + L, ids.device, x.dtype)
        cos, sin = cos[pos_offset:], sin[pos_offset:]
        caches = [] if use_cache else None
        for i, blk in enumerate(self.blocks):
            p = past[i] if past is not None else None
            x, c = blk(x, cos, sin, past=p, use_cache=use_cache)
            if use_cache:
                caches.append(c)
        return x, caches

    def head(self, hidden):
        """隐状态 -> 词表 logits（训练/评估时只对少量位置调用，省显存与算力）。"""
        return self.lm_head(self.norm_f(hidden))

    def forward(self, ids, past=None, use_cache: bool = False, pos_offset: int = 0, last_only: bool = False):
        """ids: [B, L] -> logits [B, L, V]（last_only=True 时只算最后一位，[B, 1, V]）。"""
        x, caches = self.encode(ids, past=past, use_cache=use_cache, pos_offset=pos_offset)
        if last_only:
            x = x[:, -1:]
        logits = self.head(x)
        return (logits, caches) if use_cache else logits

    # ------------------------------------------------------------ 受限解码
    def sentiment_probs(self, logits_last):
        """最后一位的 logits -> P(好)。"""
        pair = torch.stack([logits_last[:, self.pos_id], logits_last[:, self.neg_id]], dim=-1)
        return torch.softmax(pair.float(), dim=-1)[:, 0]

    def rating_dist(self, logits_last):
        """最后一位的 logits -> 5 维评分分布。"""
        idx = torch.tensor(self.rating_ids, device=logits_last.device)
        return torch.softmax(logits_last.index_select(-1, idx).float(), dim=-1)

    @torch.no_grad()
    def generate_sentiment_rating(self, ids, sep_ids):
        """两步自回归解码（要求同批内所有序列等长、无 padding）：

        1) 前缀 `...\\n情感：` -> 受限词表 {好, 差} 的 P(好)；
        2) 追加 argmax 情感 token + `␣评分：` -> 受限词表 {1..5} 的评分分布。

        返回 (p_pos, sent_id, exp_rating, arg_rating, rating_dist)。
        """
        device = ids.device
        B, L = ids.shape
        sep_ids = sep_ids.to(device)
        logits, caches = self.forward(ids, use_cache=True, last_only=True)
        last = logits[:, -1]                                     # [B, V]
        p_pos = self.sentiment_probs(last)
        thr = float(getattr(self, "sent_threshold", 0.5))
        sent_id = torch.where(p_pos >= thr,
                              torch.full((B,), self.pos_id, device=device, dtype=ids.dtype),
                              torch.full((B,), self.neg_id, device=device, dtype=ids.dtype))
        # 第 2 步：位置从 L 继续（KV cache 里已有 0..L-1）
        step = torch.cat([sent_id[:, None], sep_ids.expand(B, -1)], dim=1)
        logits2 = self.forward(step, past=caches, pos_offset=L, last_only=True)
        dist = self.rating_dist(logits2[:, -1])
        vals = torch.arange(1, 6, device=device, dtype=torch.float32)
        exp_rating = (dist * vals).sum(-1)
        arg_rating = dist.argmax(-1) + 1
        return p_pos, sent_id, exp_rating, arg_rating, dist


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
