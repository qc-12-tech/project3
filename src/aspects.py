"""优缺点挖掘：把「每条影评的情感」下沉到「方面词（aspect）」粒度，再聚合成每部电影的优点/缺点。

三步：
1. 抽样（每部电影最多 N 条）→ 按标点切成小句；
2. 小句命中方面词词典 → 交给 decoder-only 模型打情感分（P(好)）；
3. 每个方面聚合 提及数 / 好评率 / 点赞加权好评率 + 代表句（证据），并用 informative Dirichlet
   对数几率比（log-odds ratio）在「模型判好评」与「模型判差评」两个词袋间挖关键词。
"""
import argparse
import heapq
import json
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .config import ART_DIR, PRED_DIR, AggregateConfig, ModelConfig, STAR_POS_MIN, STAR_NEG_MAX
from .infer import load_model, score_texts
from .lexicon import ASPECTS, CLAUSE_SPLIT

try:
    import jieba
    jieba.setLogLevel(60)
except Exception:                                     # pragma: no cover
    jieba = None

_SPLIT = re.compile(CLAUSE_SPLIT)
_WORD = re.compile(r"[\u4e00-\u9fa5]{2,6}|[A-Za-z]{3,12}")
STOP = set("""电影 片子 一部 这部 那部 整个 完全 真的 就是 还是 觉得 感觉 没有 什么 怎么 因为 所以 但是 不过
如果 这个 那个 我们 你们 他们 自己 一个 一直 一样 时候 已经 可以 不是 不能 不会 知道 看到 看完 看完之后
很多 有点 有些 非常 特别 比较 这么 那么 出来 出来 起来 上去 下来 东西 事情 地方 演员们 导演们 而且 然后
虽然 虽然 于是 于是乎 以及 或者 还有 其实 确实 简直 居然 竟然 真的 属实 最后 开始 时候 一切 那些 这些
影评 电影票 电影院 观众 大银幕 国内 中国 国产 好莱坞""".split())


def split_clauses(text: str):
    return [c.strip() for c in _SPLIT.split(text) if len(c.strip()) >= 3]


def match_aspects(clause: str):
    hit = []
    low = clause.lower()
    for name, kws in ASPECTS.items():
        for kw in kws:
            if (kw.lower() in low) if kw.isascii() else (kw in clause):
                hit.append(name)
                break
    return hit


class TopK:
    """按打分保留 TopK 证据句。"""

    def __init__(self, k: int = 60):
        self.k, self.heap = k, []

    def add(self, score, item):
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, (score, item))
        elif score > self.heap[0][0]:
            heapq.heapreplace(self.heap, (score, item))

    def top(self, n=None, reverse=True):
        items = sorted(self.heap, key=lambda x: -x[0])
        return [it for _, it in items[: (n or self.k)]]


def mine(aspect_sample_per_movie: int = 4000, seed: int = 42, limit: int = None, verbose: bool = True):
    acfg = AggregateConfig(aspect_sample_per_movie=aspect_sample_per_movie)
    movies = json.loads((ART_DIR / "movies.json").read_text(encoding="utf-8"))["movies"]
    name = {m["id"]: (m["cn"], m["en"]) for m in movies}

    preds = pd.concat([pd.read_parquet(p, columns=["row_id", "movie_id", "p_pos", "pred_sent"])
                       for p in sorted(PRED_DIR.glob("preds_*.parquet"))], ignore_index=True)
    if limit:
        preds = preds.head(limit)
    rng = np.random.default_rng(seed)
    sampled = preds.groupby("movie_id", group_keys=True).apply(
        lambda g: g.sample(min(len(g), acfg.aspect_sample_per_movie), random_state=seed), include_groups=False)
    sampled = sampled.reset_index()
    # 只读被抽样到的行，避免把 212 万条正文全部载入内存
    rev = pd.read_parquet(ART_DIR / "reviews.parquet", columns=["row_id", "comment", "likes"],
                          filters=[("row_id", "in", sampled["row_id"].tolist())])
    sampled = sampled.merge(rev, on="row_id", how="inner")
    if verbose:
        print(f"  [aspects] 抽样 {len(sampled)} 条影评（每部 ≤{acfg.aspect_sample_per_movie}）", flush=True)

    # ---- 1) 小句 + 方面词命中
    clause_texts, clause_meta = [], []           # meta: (movie_id, aspect, review_likes, review_p_pos)
    for mv, txt, likes, pp in zip(sampled["movie_id"].to_numpy(), sampled["comment"].tolist(),
                                  sampled["likes"].to_numpy(), sampled["p_pos"].to_numpy()):
        seen_clause = set()
        used = 0
        for cl in split_clauses(txt):
            if cl in seen_clause:
                continue
            seen_clause.add(cl)
            hits = match_aspects(cl)
            if not hits:
                continue
            for a in hits:
                clause_texts.append(cl)
                clause_meta.append((int(mv), a, int(likes), float(pp)))
            used += 1
            if used >= 3:                        # 每条影评最多取 3 个小句，控制成本
                break
    if verbose:
        print(f"  [aspects] 命中方面词的小句 {len(clause_texts)} 条", flush=True)

    model, vocab, cfg, dev = load_model()
    scored = score_texts(model, vocab, cfg, clause_texts, device=dev, batch_size=256, progress=verbose)

    # ---- 2) 方面聚合
    agg = defaultdict(lambda: {"n": 0, "sum_p": 0.0, "sum_w": 0.0, "sum_wp": 0.0})
    evidence = defaultdict(lambda: {"pos": TopK(8), "neg": TopK(8)})
    for (mv, a, likes, rpp), cl, pp in zip(clause_meta, clause_texts, scored["p_pos"]):
        w = 1.0 + np.log1p(max(0, likes))
        s = agg[(mv, a)]
        s["n"] += 1
        s["sum_p"] += float(pp)
        s["sum_w"] += float(w)
        s["sum_wp"] += float(w) * float(pp)
        score = abs(float(pp) - 0.5) * 2 * (1 + np.log1p(max(0, likes)))
        evidence[(mv, a)]["pos" if pp >= 0.5 else "neg"].add(score, {"clause": cl, "p_pos": round(float(pp), 3),
                                                                     "likes": int(likes)})

    # ---- 3) log-odds 关键词（模型判好评 vs 判差评 两个词袋）
    key_pool = {"pos": defaultdict(Counter), "neg": defaultdict(Counter)}
    bg = Counter()
    if jieba is not None:
        for mv, txt, sent in zip(sampled["movie_id"].to_numpy(), sampled["comment"].tolist(),
                                 sampled["pred_sent"].to_numpy()):
            words = [w for w in jieba.lcut(txt) if w not in STOP and len(w) >= 2]
            pool = key_pool["pos" if sent == 1 else "neg"][int(mv)]
            for w in words:
                if len(w) >= 2:
                    pool[w] += 1
                    bg[w] += 1
    keywords = {}
    for mv in sampled["movie_id"].unique():
        c_pos, c_neg = key_pool["pos"][int(mv)], key_pool["neg"][int(mv)]
        n_pos, n_neg = sum(c_pos.values()), sum(c_neg.values())
        n_bg, vocab_bg = sum(bg.values()), len(bg)
        a0 = 0.01 * max(1, n_bg)
        out = []
        for w in set(c_pos) | set(c_neg):
            if c_pos[w] + c_neg[w] < 5:
                continue
            y, x = c_pos[w], c_neg[w]
            ai = a0 * (bg[w] / max(1, n_bg))
            d = (np.log((y + ai) / max(1e-9, n_pos + a0 - y - ai)) - np.log((x + ai) / max(1e-9, n_neg + a0 - x - ai)))
            var = 1.0 / (y + ai) + 1.0 / (x + ai)
            out.append({"word": w, "z": round(float(d / np.sqrt(var)), 3), "n_pos": int(y), "n_neg": int(x)})
        out.sort(key=lambda r: -r["z"])
        keywords[int(mv)] = {"pos": out[:acfg.top_keywords], "neg": out[-acfg.top_keywords:][::-1]}

    # ---- 4) 组装输出
    result = {}
    for mv in sorted({k[0] for k in agg}):
        rows = []
        for (m, a), s in agg.items():
            if m != mv:
                continue
            pos_rate = s["sum_p"] / max(1, s["n"])
            rows.append({"aspect": a, "mentions": s["n"], "pos_rate": round(pos_rate, 4),
                         "like_weighted_pos_rate": round(s["sum_wp"] / max(1e-9, s["sum_w"]), 4),
                         "evidence_pos": evidence[(mv, a)]["pos"].top(3),
                         "evidence_neg": evidence[(mv, a)]["neg"].top(3)})
        rows.sort(key=lambda r: -r["mentions"])
        result[str(mv)] = {"movie_id": mv, "movie_cn": name[mv][0], "movie_en": name[mv][1],
                           "n_sampled": int((sampled["movie_id"] == mv).sum()),
                           "aspects": rows, "keywords": keywords.get(mv, {"pos": [], "neg": []})}
    (ART_DIR / "aspects.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    if verbose:
        print(f"  [aspects] -> {ART_DIR/'aspects.json'}（{len(result)} 部电影）", flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-movie", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    mine(a.per_movie, limit=a.limit)
