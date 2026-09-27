"""推理：用两步自回归解码给影评打分（P(好) + 期望评分）。

- `run_inference`：全量 212 万条流式推理，按前缀长度等长分组以避免 padding，分片写出 parquet。
- `score_texts`：任意文本批量打分（供方面词挖掘 / API / 演示使用）。
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import (ART_DIR, CACHE_DIR, PRED_DIR, ModelConfig, SEP, PAD_ID, POS_CHAR, NEG_CHAR,
                     pick_device)
from .data import PrefixStore, make_prefix_ids
from .model import GPT
from .vocab import Vocab


def load_model(ckpt=None, device: str = "auto"):
    ckpt = Path(ckpt or (ART_DIR / "model.pt"))
    vocab = Vocab.load(ART_DIR / "vocab.json")
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**{k: v for k, v in blob["model"].items()})
    mcfg.vocab_size = len(vocab)
    model = GPT(mcfg, vocab.stoi)
    model.load_state_dict(blob["state_dict"])
    model.sent_threshold = float(blob.get("sent_threshold", 0.5))
    dev = pick_device(device)
    model.to(dev).eval()
    return model, vocab, mcfg, dev


@torch.inference_mode()
def score_texts(model, vocab, cfg, texts, device=None, batch_size: int = 256, progress: bool = False):
    """任意文本 -> dict(p_pos, sent, exp_rating, arg_rating, p_rating)。变长，等长分组无 padding。"""
    device = device or next(model.parameters()).device
    seqs = [make_prefix_ids(vocab, t, cfg) for t in texts]
    n = len(seqs)
    out = {"p_pos": np.zeros(n, dtype=np.float32), "sent": np.zeros(n, dtype=np.int8),
           "exp_rating": np.zeros(n, dtype=np.float32), "arg_rating": np.zeros(n, dtype=np.int8),
           "p_rating": np.zeros(n, dtype=np.float32)}
    if n == 0:
        return out
    sep_ids = torch.tensor([vocab.id(c) for c in SEP], dtype=torch.long, device=device)
    order = sorted(range(n), key=lambda i: len(seqs[i]))
    i = 0
    done = 0
    while i < n:
        L = len(seqs[order[i]])
        j = i
        while j < n and len(seqs[order[j]]) == L:
            j += 1
        for a in range(i, j, batch_size):
            idx = order[a:min(j, a + batch_size)]
            ids = torch.tensor(np.stack([seqs[k] for k in idx]), dtype=torch.long, device=device)
            pp, st, er, ar, dist = model.generate_sentiment_rating(ids, sep_ids)
            for t, k in enumerate(idx):
                out["p_pos"][k] = float(pp[t])
                out["sent"][k] = 1 if int(st[t]) == model.pos_id else 0
                out["exp_rating"][k] = float(er[t])
                out["arg_rating"][k] = int(ar[t])
                out["p_rating"][k] = float(dist[t].max())
            done += len(idx)
            if progress and done % 20000 < batch_size:
                print(f"    scored {done}/{n}", flush=True)
        i = j
    return out


def run_inference(batch_size: int = 128, limit=None, num_shards: int = 1, shard: int = 0,
                  ckpt=None, device="auto", start: int = 0, end=None, flush_every: int = 200_000,
                  token_budget: int = 32768, verbose: bool = True):
    """全量/分段推理。

    - `start`/`end`：按行区间分段，方便续跑（每段一个独立进程，MPS 分配器状态全新）；
    - 每 `flush_every` 行增量落盘一个 parquet，崩了也不用整段重跑；
    - 批大小按 token 预算收缩（长序列自动小批），并把 shape 收敛，避免 MPS 分配器内存只涨不降。
    """
    model, vocab, cfg, dev = load_model(ckpt, device)
    store = PrefixStore(CACHE_DIR)
    total_rows = len(store)
    end = total_rows if end is None else min(int(end), total_rows)
    if limit:
        end = min(end, start + limit)
    n = end - start
    lengths = store.lengths()[start:end]
    order = np.argsort(lengths, kind="stable")          # 等长分组，避免 padding 影响 RoPE 位置
    sep_ids = torch.tensor([vocab.id(c) for c in SEP], dtype=torch.long, device=dev)

    p_pos = np.zeros(n, dtype=np.float32)
    sent = np.zeros(n, dtype=np.int8)
    exp_r = np.zeros(n, dtype=np.float32)
    arg_r = np.zeros(n, dtype=np.int8)
    p_top = np.zeros(n, dtype=np.float32)
    written = np.zeros(n, dtype=bool)
    done_mask = np.zeros(n, dtype=bool)
    for old in PRED_DIR.glob(f"preds_{start:08d}_*.parquet"):    # 清理该区间旧分片，避免重复行
        old.unlink()
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    meta = store.meta[start:end]

    def _flush(mask, part):
        keep = np.where(mask)[0]
        pd.DataFrame({
            "row_id": (keep + start).astype(np.int32),
            "movie_id": meta[keep, 2].astype(np.int16),
            "star": meta[keep, 0].astype(np.int8),
            "likes": meta[keep, 3].astype(np.int32),
            "p_pos": p_pos[keep].astype(np.float32),
            "pred_sent": sent[keep],
            "exp_rating": exp_r[keep].astype(np.float32),
            "pred_rating": arg_r[keep],
            "p_rating": p_top[keep].astype(np.float32),
        }).to_parquet(PRED_DIR / f"preds_{start:08d}_{part:03d}.parquet", index=False)
        return len(keep)

    t0 = time.time()
    i, done, part, since_flush, saved = 0, 0, 0, 0, 0
    while i < n:
        L = int(lengths[order[i]])
        j = i
        while j < n and int(lengths[order[j]]) == L:
            j += 1
        bs = max(16, min(batch_size, token_budget // max(1, L)))       # token 预算动态批
        for a in range(i, j, bs):
            idx = order[a:min(j, a + bs)]
            if num_shards > 1:
                idx = idx[idx % num_shards == shard]
                if len(idx) == 0:
                    continue
            ids = torch.tensor(np.stack([store.get(int(k) + start) for k in idx]), dtype=torch.long, device=dev)
            pp, st, er, ar, dist = model.generate_sentiment_rating(ids, sep_ids)
            pp, st, er, ar, ptop = pp.cpu().numpy(), st.cpu().numpy(), er.cpu().numpy(), ar.cpu().numpy(), dist.cpu().numpy()
            p_pos[idx], sent[idx] = pp, (st == model.pos_id).astype(np.int8)
            exp_r[idx], arg_r[idx] = er, ar.astype(np.int8)
            p_top[idx] = ptop.max(1)
            done_mask[idx] = True
            done += len(idx)
            since_flush += len(idx)
            if done % 50_000 < len(idx) and hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()          # 精确长度分组 shape 多，定期释放分配器缓存
            if since_flush >= flush_every:
                m = done_mask & ~written
                saved += _flush(m, part)
                written |= m
                part += 1
                since_flush = 0
            if verbose and done % 100_000 < len(idx):
                el = time.time() - t0
                print(f"  [infer] {start+done}/{end} ({done/max(1e-9,el):.0f} 条/s, 已用 {el/60:.1f} min)",
                      flush=True)
        i = j

    m = ~written
    if m.any():
        saved += _flush(m, part)
    if verbose:
        print(f"  [infer] 区间 [{start}, {end}) 共 {saved} 行写出到 {PRED_DIR}，"
              f"用时 {(time.time()-t0)/60:.2f} min", flush=True)
    return saved


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--flush-every", type=int, default=200_000)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    a = ap.parse_args()
    run_inference(a.batch_size, a.limit, a.num_shards, a.shard, device=a.device,
                  start=a.start, end=a.end, flush_every=a.flush_every)
