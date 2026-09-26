"""数据集与 DataLoader：文本 -> 字符 id，带磁盘编码缓存（memory-map）。

100 万条数据如果每次现场编码/全部载入内存都会很慢，这里在首次运行时
把整份语料编码成 int32 的 .npy 文件，之后用 mmap 只读映射，内存占用极低。
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src import config as C
from src.generate_data import read_jsonl
from src.vocab import Vocab


def _file_signature(path):
    st = os.stat(path)
    return {"path": os.path.abspath(path), "size": st.st_size, "mtime": int(st.st_mtime)}


def _cache_meta_path(cache_prefix=None):
    return (cache_prefix or C.CACHE_PREFIX) + "_meta.json"


def build_or_load_cache(data_path, vocab, max_len=C.MAX_LEN, limit=None, verbose=True,
                        cache_prefix=None):
    """把语料编码成 ids/labels 的 npy 缓存并返回 (ids, labels)。

    ids:    np.memmap  [N, max_len] int32
    labels: np.ndarray [N]          int64
    cache_prefix: 自定义缓存前缀（多任务时用独立前缀，避免互相覆盖缓存）。
    """
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    prefix = cache_prefix or C.CACHE_PREFIX
    meta_path = _cache_meta_path(prefix)
    sig = _file_signature(data_path)
    sig.update({"vocab_size": len(vocab), "max_len": max_len, "limit": limit})

    ids_path = prefix + "_ids.npy"
    labels_path = prefix + "_labels.npy"

    if os.path.exists(meta_path) and os.path.exists(ids_path) and os.path.exists(labels_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached == sig:
            if verbose:
                print(f"[cache] 命中编码缓存 {ids_path}")
            ids = np.load(ids_path, mmap_mode="r")
            labels = np.load(labels_path)
            if limit is not None:
                ids, labels = ids[:limit], labels[:limit]
            return ids, labels

    if verbose:
        print("[cache] 未命中，开始编码语料（首次较慢）...")
    texts, labels_list = read_jsonl(data_path, limit=limit)
    n = len(texts)
    ids = np.zeros((n, max_len), dtype=np.int32)
    labels = np.asarray(labels_list, dtype=np.int64)
    for i, text in enumerate(texts):
        ids[i] = vocab.encode(text, max_len)

    np.save(ids_path, ids)
    np.save(labels_path, labels)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(sig, f, ensure_ascii=False)
    if verbose:
        print(f"[cache] 已缓存 {n} 条 -> {ids_path}")

    ids = np.load(ids_path, mmap_mode="r")
    return ids, labels


class ReviewDataset(Dataset):
    def __init__(self, ids, labels, indices=None):
        self.ids = ids
        self.labels = labels
        self.indices = np.arange(len(labels)) if indices is None else indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return torch.from_numpy(np.asarray(self.ids[idx], dtype=np.int64)), int(self.labels[idx])


def collate_fn(batch):
    ids = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    # padding 位置 mask：[B, L]，True 表示有效 token
    attention_mask = ids != C.PAD_ID
    return ids, attention_mask, labels


def make_dataloaders(data_path, vocab, max_len=C.MAX_LEN, batch_size=C.BATCH_SIZE,
                     val_ratio=C.VAL_RATIO, max_samples=None, seed=C.SEED, verbose=True,
                     cache_prefix=None):
    ids, labels = build_or_load_cache(data_path, vocab, max_len, limit=max_samples,
                                      verbose=verbose, cache_prefix=cache_prefix)
    n = len(labels)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_ratio))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = ReviewDataset(ids, labels, train_idx)
    val_ds = ReviewDataset(ids, labels, val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)
    return train_loader, val_loader
