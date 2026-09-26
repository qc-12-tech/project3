"""评价查重 / 刷评论检测。

判断一个商品的评价里是否存在刷评，主要看两类信号：
1. 完全重复：一字不差的评价出现多次（复制粘贴刷评）；
2. 近似重复：换几个字/加标点的模板化评价（字符 n-gram Jaccard 相似度识别）。

重复率 = 1 - 聚类后唯一评价数 / 总评价数，超过阈值判定为疑似刷评论。
"""
import re
from collections import Counter

from src import config as C

# 归一化时去掉的空白与标点
_PUNCT = re.compile(r"[\s，。！？、；：,.!?~·…—\-_\"'【】\[\]()（）]+")


def _normalize(text):
    return _PUNCT.sub("", text)


def _ngrams(s, n=2):
    """字符 n-gram 集合（用于衡量两句话的表面相似度）。默认 2-gram 对短句更敏感。"""
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def check_duplicates(texts, sim_threshold=None, ngram=2):
    """对同一商品的多条评价做查重，返回结构化报告。"""
    sim_threshold = sim_threshold if sim_threshold is not None else C.DUP_SIM_THRESHOLD
    total = len(texts)
    if total == 0:
        return {
            "total": 0, "unique_count": 0, "dup_rate": 0.0,
            "verdict": "无数据", "level": "normal",
            "exact_dup_groups": [], "near_dup_groups": [],
        }

    norm = [_normalize(t) for t in texts]
    # 归一化文本 -> 首次出现的原始文本（用于展示）
    norm_to_first = {}
    for t, n in zip(texts, norm):
        norm_to_first.setdefault(n, t)

    # ---- 1) 完全重复 ----
    counter = Counter(norm)
    exact_dup_groups = [
        {"text": norm_to_first[k], "count": c}
        for k, c in counter.most_common() if c >= 2
    ]

    # ---- 2) 近似重复贪心聚类（对完全去重后的文本）----
    unique_norms = list(counter.keys())
    ngram_sets = [_ngrams(s, ngram) for s in unique_norms]
    clusters = []  # [(代表 ngram 集合, [成员下标])]
    for i, s in enumerate(ngram_sets):
        placed = False
        for rep_set, members in clusters:
            if _jaccard(s, rep_set) >= sim_threshold:
                members.append(i)
                placed = True
                break
        if not placed:
            clusters.append((s, [i]))

    unique_count = len(clusters)
    dup_rate = round(1 - unique_count / total, 4)

    # 近似重复组：聚类内成员 >= 2 且归一化文本不完全相同（否则属于完全重复）
    near_dup_groups = []
    for rep_set, members in clusters:
        if len(members) < 2:
            continue
        member_norms = [unique_norms[m] for m in members]
        if len(set(member_norms)) < 2:
            continue
        sim = max(_jaccard(ngram_sets[a], ngram_sets[b])
                  for a in members for b in members if a != b)
        near_dup_groups.append({
            "similarity": round(sim, 3),
            "count": len(members),
            "texts": [norm_to_first[unique_norms[m]] for m in members[:5]],
        })
    near_dup_groups.sort(key=lambda g: (-g["count"], -g["similarity"]))

    # ---- 3) 结论 ----
    if dup_rate >= C.DUP_RATE_HIGH:
        verdict, level = "重复率偏高，疑似刷评论", "high"
    elif dup_rate >= C.DUP_RATE_WARN:
        verdict, level = "存在一定重复，建议关注", "medium"
    else:
        verdict, level = "重复率正常", "normal"

    return {
        "total": total,
        "unique_count": unique_count,
        "dup_rate": dup_rate,
        "verdict": verdict,
        "level": level,
        "exact_dup_groups": exact_dup_groups,
        "near_dup_groups": near_dup_groups,
    }


if __name__ == "__main__":
    demo = [
        "味道很好，下次还来",
        "味道很好，下次还来",
        "味道很好，下次还来",
        "味道很好 下次还来！",
        "味道不错，下次再来",
        "配送很快，包装完好",
        "分量很足，性价比高",
        "太咸了，不好吃",
    ]
    import json
    print(json.dumps(check_duplicates(demo), ensure_ascii=False, indent=2))
