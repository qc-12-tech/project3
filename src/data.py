"""数据管线：清洗 -> 去重 -> 生成式序列编码 -> 磁盘缓存（int16 flat + offsets，mmap 只读）。

每条影评编码成：

  训练/验证（带标签，因果 LM 监督）:
    <bos> 影评：{评论} \\n 情感：{好|差} ␣ 评分：{1-5} <eos>
    只在 情感位 / 评分位 / <eos> 位上算 loss；star==3 的中评屏蔽情感位。

  推理（只有前缀）:
    <bos> 影评：{评论} \\n 情感：

划分用行号哈希分桶，保证流式处理下可复现，无需把 200 万条全读进内存。
"""
import argparse
import json
import re
from array import array
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import (DATA_CSV, CACHE_DIR, ART_DIR, PREFIX, MID, SEP, EOS_ID, BOS_ID, PAD_ID,
                     POS_CHAR, NEG_CHAR, STAR_POS_MIN, STAR_NEG_MAX, ModelConfig)
from .vocab import Vocab

_URL = re.compile(r"https?://\S+|www\.\S+")
_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

VAL_BUCKET = 10      # 行号哈希 %1000 < 10 -> 验证集（约 1%）
TRAIN_BUCKET = 200   # [10, 200) -> 训练集候选（约 19%），其余只用于全量推理
HASH_MUL = 2654435761
COLS = ["Movie_Name_EN", "Movie_Name_CN", "Star", "Username", "Date", "Comment", "Like"]


def clean_text(s: str) -> str:
    s = _URL.sub(" ", str(s))
    s = _CTRL.sub("", s)
    s = s.replace("\u3000", " ").strip()
    return _WS.sub(" ", s)


def star_to_sentiment(star: int) -> int:
    """1=好评 0=差评 -1=中评"""
    if star >= STAR_POS_MIN:
        return 1
    if star <= STAR_NEG_MAX:
        return 0
    return -1


def load_clean(limit=None, chunksize: int = 200_000) -> pd.DataFrame:
    """清洗 + 去重后的 DataFrame（EDA / 建词表用，全量约 1GB 内存，谨慎使用）。"""
    out, seen = [], set()
    for ch in pd.read_csv(DATA_CSV, chunksize=chunksize, usecols=COLS):
        ch["comment"] = ch["Comment"].map(clean_text)
        ch = ch[(ch["comment"].str.len() >= 4) & ch["Star"].between(1, 5)]
        key = (ch["Movie_Name_CN"].astype(str) + "|" + ch["Username"].astype(str) + "|" + ch["comment"])
        h = pd.util.hash_array(key.to_numpy(dtype=object))
        keep = np.array([x not in seen for x in h])
        for x in h[keep]:
            seen.add(x)
        ch = ch[keep]
        out.append(ch[["Movie_Name_EN", "Movie_Name_CN", "Star", "Username", "Date", "comment", "Like"]]
                   .rename(columns={"Movie_Name_EN": "movie_en", "Movie_Name_CN": "movie",
                                    "Star": "star", "Username": "user", "Date": "date",
                                    "Like": "likes"}))
        if limit and sum(len(x) for x in out) >= limit:
            break
    df = pd.concat(out, ignore_index=True)
    if limit:
        df = df.head(limit).copy()
    df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))
    return df


def sample_texts(target: int = 300_000, per_chunk: int = 15_000, chunksize: int = 100_000):
    """跨整份文件均匀抽样文本，避免只取到前几部电影（数据按电影排序）。"""
    texts = []
    for i, ch in enumerate(pd.read_csv(DATA_CSV, chunksize=chunksize, usecols=["Comment"])):
        s = ch["Comment"].map(clean_text)
        s = s[s.str.len() >= 4]
        if len(s) > per_chunk:
            s = s.sample(n=per_chunk, random_state=i)
        texts.extend(s.tolist())
        if len(texts) >= target:
            break
    return texts[:target]


def collect_movies() -> list:
    """扫描电影列，返回稳定排序的 (movie_cn, movie_en) 列表。"""
    pairs = {}
    for ch in pd.read_csv(DATA_CSV, chunksize=500_000, usecols=["Movie_Name_CN", "Movie_Name_EN"]):
        for cn, en in zip(ch["Movie_Name_CN"].astype(str), ch["Movie_Name_EN"].astype(str)):
            pairs.setdefault(cn, en)
    return sorted(pairs.items())


def text_budget(cfg: ModelConfig, templ_len: int) -> int:
    return max(1, cfg.max_len - templ_len)


def make_full_ids(v: Vocab, text: str, star: int, cfg: ModelConfig):
    """返回 (ids, labels)：仅在情感位/评分位/<eos> 位有标签。

    序列总长 = len(text) + 1(bos) + len(PREFIX) + len(MID) + 1(情感) + len(SEP) + 1(评分) + 1(eos)
    所以文本预算必须多留一个 token，否则整条序列会比 max_len 长 1。
    """
    extra = 1 + len(PREFIX) + len(MID) + 1 + len(SEP) + 2      # bos + 模板 + 情感 + 评分 + eos
    text = text[:text_budget(cfg, extra)]
    ids = [BOS_ID] + v.encode(PREFIX) + v.encode(text) + v.encode(MID)
    labels = [-100] * len(ids)
    sent = star_to_sentiment(star)
    if sent >= 0:                                     # 中评不监督情感位
        ids.append(v.id(POS_CHAR if sent == 1 else NEG_CHAR))
        labels.append(ids[-1])
    sep = v.encode(SEP)
    ids += sep
    labels += [-100] * len(sep)
    ids.append(v.id(str(int(star))))
    labels.append(ids[-1])
    ids.append(EOS_ID)
    labels.append(EOS_ID)
    return ids, labels


def make_prefix_ids(v: Vocab, text: str, cfg: ModelConfig):
    """推理用前缀：<bos> 影评：{评论}\\n情感："""
    text = text[:text_budget(cfg, 1 + len(PREFIX) + len(MID))]
    return [BOS_ID] + v.encode(PREFIX) + v.encode(text) + v.encode(MID)


class _FlatWriter:
    """int16 flat 数组 + int32 offsets（array 模块，内存占用仅为 Python list 的 1/15）。"""

    def __init__(self):
        self.ids = array("h")
        self.offsets = array("i", [0])

    def add(self, seq):
        self.ids.extend(seq)
        self.offsets.append(len(self.ids))

    def save(self, prefix):
        Path(prefix).parent.mkdir(parents=True, exist_ok=True)
        np.save(f"{prefix}.ids.npy", np.frombuffer(self.ids, dtype=np.int16))
        np.save(f"{prefix}.off.npy", np.frombuffer(self.offsets, dtype=np.int32))


FIELDS = ("star", "sent", "movie", "likes")


class _Meta:
    """三个 split 的元信息（int32 数组）。"""

    def __init__(self):
        self.data = {k: {f: array("i") for f in FIELDS} for k in ("train", "val", "predict")}

    def add(self, split, star, sent, movie, likes):
        d = self.data[split]
        d["star"].append(star)
        d["sent"].append(sent)
        d["movie"].append(movie)
        d["likes"].append(likes)

    def save(self):
        np.savez(CACHE_DIR / "meta.npz",
                 **{f"{k}_{f}": np.frombuffer(self.data[k][f], dtype=np.int32) for k in self.data for f in FIELDS})


def prepare(cfg: ModelConfig, vocab: Vocab, chunksize: int = 100_000, limit=None, verbose: bool = True):
    """流式构建全部缓存：train / val / predict（全量）+ reviews.parquet。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    movies = collect_movies()
    m2i = {cn: i for i, (cn, _) in enumerate(movies)}
    (ART_DIR / "movies.json").write_text(
        json.dumps({"movies": [{"id": i, "cn": cn, "en": en} for i, (cn, en) in enumerate(movies)]},
                   ensure_ascii=False, indent=1), encoding="utf-8")

    w = {k: _FlatWriter() for k in ("train", "train.labels", "val", "val.labels", "predict")}
    meta = _Meta()
    seen, row_id = set(), 0
    schema = pa.schema([("row_id", pa.int32()), ("movie_id", pa.int16()), ("star", pa.int8()),
                        ("likes", pa.int32()), ("comment", pa.string())])
    writer = pq.ParquetWriter(ART_DIR / "reviews.parquet", schema)
    try:
        for ci, ch in enumerate(pd.read_csv(DATA_CSV, chunksize=chunksize, usecols=COLS)):
            if limit and row_id >= limit:
                break
            ch["comment"] = ch["Comment"].map(clean_text)
            ch = ch[(ch["comment"].str.len() >= 4) & ch["Star"].between(1, 5)]
            key = (ch["Movie_Name_CN"].astype(str) + "|" + ch["Username"].astype(str) + "|" + ch["comment"])
            h = pd.util.hash_array(key.to_numpy(dtype=object))
            keep = np.array([x not in seen for x in h])
            for x in h[keep]:
                seen.add(x)
            ch = ch[keep]
            if limit:
                ch = ch.head(max(0, limit - row_id))
            if not len(ch):
                continue
            stars = ch["Star"].to_numpy(dtype=np.int64)
            likes = ch["Like"].to_numpy(dtype=np.int64)
            mid = np.array([m2i.get(x, -1) for x in ch["Movie_Name_CN"].astype(str)], dtype=np.int64)
            texts = ch["comment"].tolist()
            idx = np.arange(row_id, row_id + len(ch), dtype=np.int64)
            buckets = (idx * HASH_MUL) % 1000
            for j, txt in enumerate(texts):
                st, lk, mv, bk = int(stars[j]), int(likes[j]), int(mid[j]), int(buckets[j])
                sent = star_to_sentiment(st)
                w["predict"].add(make_prefix_ids(vocab, txt, cfg))
                meta.add("predict", st, sent, mv, lk)
                if bk < VAL_BUCKET:
                    ids, lab = make_full_ids(vocab, txt, st, cfg)
                    w["val"].add(ids), w["val.labels"].add(lab)
                    meta.add("val", st, sent, mv, lk)
                elif bk < TRAIN_BUCKET:
                    ids, lab = make_full_ids(vocab, txt, st, cfg)
                    w["train"].add(ids), w["train.labels"].add(lab)
                    meta.add("train", st, sent, mv, lk)
            writer.write_table(pa.Table.from_pydict(
                {"row_id": idx.astype(np.int32), "movie_id": mid.astype(np.int16),
                 "star": stars.astype(np.int8), "likes": likes.astype(np.int32), "comment": texts},
                schema=schema))
            row_id += len(ch)
            if verbose and ci % 5 == 0:
                print(f"  [prepare] 已处理 {row_id} 条（chunk {ci}）", flush=True)
    finally:
        writer.close()

    for k, ww in w.items():
        ww.save(CACHE_DIR / k)
        del ww
    meta.save()
    summary = {"n_train": len(meta.data["train"]["star"]), "n_val": len(meta.data["val"]["star"]),
               "n_predict": len(meta.data["predict"]["star"]), "n_movies": len(movies),
               "vocab_size": len(vocab)}
    (CACHE_DIR / "meta.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    if verbose:
        print("  [prepare]", summary, flush=True)
    return summary


class LMDataset:
    """训练/验证集：flat ids + labels，按需 pad 到批内最长。"""

    def __init__(self, cache_dir, split: str):
        cache_dir = Path(cache_dir)
        self.split = split
        self.ids = np.load(cache_dir / f"{split}.ids.npy", mmap_mode="r")
        self.off = np.load(cache_dir / f"{split}.off.npy", mmap_mode="r")
        self.lab = np.load(cache_dir / f"{split}.labels.ids.npy", mmap_mode="r")
        self.lab_off = np.load(cache_dir / f"{split}.labels.off.npy", mmap_mode="r")
        self._meta = np.load(cache_dir / "meta.npz", mmap_mode="r")
        self.meta = np.stack([self._meta[f"{split}_{f}"] for f in FIELDS], axis=1)   # [N,4]

    def __len__(self):
        return len(self.off) - 1

    def __getitem__(self, i):
        return (np.asarray(self.ids[self.off[i]:self.off[i + 1]], dtype=np.int64),
                np.asarray(self.lab[self.lab_off[i]:self.lab_off[i + 1]], dtype=np.int64))


def collate(batch, pad_id: int = PAD_ID, multiple: int = 16):
    """按批内最长序列 pad；宽度向上对齐到 multiple 的整数倍。

    对齐是为了把「张量形状」收敛到少数几档：MPS 的缓存分配器遇到大量不同 shape 时
    会不断保留新的 block，内存只涨不降，最终 OOM。对齐后形状只有 ~10 档，分配器稳定。
    """
    import torch
    maxlen = max(len(x[0]) for x in batch)
    maxlen = ((maxlen + multiple - 1) // multiple) * multiple
    ids = np.full((len(batch), maxlen), pad_id, dtype=np.int64)
    lab = np.full((len(batch), maxlen), -100, dtype=np.int64)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)] = x
        lab[i, :len(y)] = y
    return torch.from_numpy(ids), torch.from_numpy(lab)


class PrefixStore:
    """推理集：全量前缀（等长分组在 infer 里做，保证无 padding）。"""

    def __init__(self, cache_dir):
        cache_dir = Path(cache_dir)
        self.ids = np.load(cache_dir / "predict.ids.npy", mmap_mode="r")
        self.off = np.load(cache_dir / "predict.off.npy", mmap_mode="r")
        m = np.load(cache_dir / "meta.npz", mmap_mode="r")
        self.meta = np.stack([m[f"predict_{f}"] for f in FIELDS], axis=1)            # star,sent,movie,likes

    def __len__(self):
        return len(self.off) - 1

    def lengths(self):
        return np.diff(self.off)

    def get(self, i):
        return np.asarray(self.ids[self.off[i]:self.off[i + 1]], dtype=np.int64)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 条（冒烟测试）")
    ap.add_argument("--vocab-sample", type=int, default=300_000)
    ap.add_argument("--vocab-size", type=int, default=8000)
    ap.add_argument("--min-freq", type=int, default=3)
    a = ap.parse_args()
    cfg = ModelConfig()
    from .vocab import build_and_save
    texts = sample_texts(a.vocab_sample) if not a.limit else load_clean(limit=a.limit)["comment"].tolist()
    v = build_and_save(texts, max_size=a.vocab_size, min_freq=a.min_freq)
    cfg.vocab_size = len(v)
    print(f"词表大小 {len(v)}")
    prepare(cfg, v, limit=a.limit)
