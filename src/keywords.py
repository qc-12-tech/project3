"""差评高频词 / 主要问题分析。

思路：
1. jieba 分词 + 去停用词，统计差评中的词频 -> 高频词；
2. 对比“差评词频占比”与“整体词频占比”，得到 lift（区分度），
   避免把“外卖/好吃”这类普遍词误当问题词；
3. 按维度（口味/配送/包装/份量/温度/价格/服务/卫生）聚合，得出主要问题分布。
"""
import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict

import jieba

from src import config as C

STOPWORDS = set("""
的 了 我 你 他 她 它 们 是 在 有 和 与 及 也 都 就 还 又 很 太 更 最 不 没 无
这 那 这个 那个 一个 一下 一点 什么 怎么 为什么 因为 所以 但是 不过 而且 然后
感觉 觉得 就是 真的 挺 特别 非常 有点 比较 一般 可以 还是 已经 应该 现在
外卖 这家 这单 商家 骑手 店 店家 东西 时候 地方 一次 今天 昨天 之前 之后
根本 完全 没有 不是 不会 这么 那么 只是 出来 的话 一起 之后 之前 整体 来说
第一次 评分 朋友 这次 推荐 直接 实在 真的 整个 那种 这种 一点 出来 上去 下来
""".split())

# 单字中仍具情感/问题含义的，保留
SINGLE_OK = set("贵 慢 凉 咸 油 少 差 烫 腥 馊 酸 苦 干 柴 破 漏 洒 脏 虫 硬 烂")

ASPECT_WORDS = {
    "口味": ["味", "咸", "淡", "油腻", "难吃", "好吃", "香", "鲜", "入味", "怪味",
             "腥", "苦", "酸", "柴", "干", "口感", "调料", "齁"],
    "配送": ["配送", "骑手", "送到", "超时", "出餐", "餐", "小时", "分钟", "等",
             "慢", "迟到", "速度"],
    "包装": ["包装", "盒子", "洒", "漏", "破", "撒", "袋子", "压扁"],
    "份量": ["分量", "量", "少", "饱", "缺斤少两", "小", "份"],
    "温度": ["凉", "热", "冷", "温度", "热气"],
    "价格": ["贵", "价格", "性价比", "划算", "坑", "不值", "便宜", "收费"],
    "服务": ["服务", "态度", "客服", "商家", "老板", "不理", "回复", "投诉"],
    "卫生": ["卫生", "干净", "头发", "虫", "异物", "新鲜", "变质", "脏"],
}


def _valid_token(word):
    if word in STOPWORDS:
        return False
    if not word.strip():
        return False
    if any(ch in "，。！？、；：,.!?~ \t\n" for ch in word):
        return False
    if len(word) >= 2:
        return True
    return word in SINGLE_OK


def tokenize(text):
    return [w for w in jieba.cut(text) if _valid_token(w)]


def count_words(texts, token_cache=None):
    counter = Counter()
    for i, text in enumerate(texts):
        tokens = tokenize(text)
        if token_cache is not None:
            token_cache[i] = tokens
        counter.update(tokens)
    return counter


def extract_keywords(neg_texts, overall_texts=None, top_k=C.KEYWORD_TOP_K,
                     min_count=C.KEYWORD_MIN_COUNT, min_lift=C.KEYWORD_MIN_LIFT):
    """返回按重要度排序的关键词列表。

    重要度 = 词频 * log2(1 + lift)，lift = 差评占比 / 整体占比。
    只保留 lift >= min_lift 的词，避免把“外卖/整体”这类两类都常见的词当成问题词。
    """
    neg_counter = count_words(neg_texts)
    neg_total = sum(neg_counter.values()) or 1
    overall_counter = count_words(overall_texts) if overall_texts else neg_counter
    overall_total = sum(overall_counter.values()) or 1

    results = []
    for word, cnt in neg_counter.items():
        if cnt < min_count:
            continue
        neg_rate = cnt / neg_total
        overall_rate = overall_counter.get(word, 1) / overall_total
        lift = neg_rate / max(overall_rate, 1e-9)
        if lift < min_lift:
            continue
        score = cnt * math.log2(1 + lift)
        results.append({
            "word": word,
            "count": int(cnt),
            "neg_ratio": round(neg_rate, 5),
            "lift": round(float(lift), 3),
            "score": round(float(score), 2),
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def compute_aspects(neg_texts, top_per_aspect=8):
    word_counter = count_words(neg_texts)
    aspect_counter = Counter()
    aspect_words = defaultdict(list)
    for word, cnt in word_counter.items():
        matched = None
        # 先精确匹配，避免“干”误配到“干净”这类子串误判
        for aspect, keys in ASPECT_WORDS.items():
            if word in keys:
                matched = aspect
                break
        if matched is None:
            for aspect, keys in ASPECT_WORDS.items():
                if any(len(k) >= 2 and k in word for k in keys):
                    matched = aspect
                    break
        if matched:
            aspect_counter[matched] += cnt
            aspect_words[matched].append((word, cnt))
    total = sum(aspect_counter.values()) or 1
    aspects = []
    for aspect, cnt in aspect_counter.most_common():
        words = sorted(aspect_words[aspect], key=lambda x: x[1], reverse=True)[:top_per_aspect]
        aspects.append({
            "aspect": aspect,
            "count": int(cnt),
            "ratio": round(cnt / total, 4),
            "words": [{"word": w, "count": c} for w, c in words],
        })
    return aspects


def analyze_negatives(neg_texts, overall_texts=None, top_k=C.KEYWORD_TOP_K,
                      min_count=C.KEYWORD_MIN_COUNT, min_lift=C.KEYWORD_MIN_LIFT):
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_negative": len(neg_texts),
        "n_overall": len(overall_texts) if overall_texts else len(neg_texts),
        "top_keywords": extract_keywords(neg_texts, overall_texts, top_k,
                                         min_count, min_lift),
        "aspects": compute_aspects(neg_texts),
    }


def build_from_dataset(data_path=C.RAW_DATA, scan_limit=C.KEYWORD_SCAN_LIMIT,
                       out_path=C.KEYWORDS_PATH, top_k=C.KEYWORD_TOP_K,
                       min_count=C.KEYWORD_MIN_COUNT, min_lift=C.KEYWORD_MIN_LIFT):
    """从数据集中抽取差评（label==2）和整体样本，生成关键词报告。"""
    import json as _json

    neg_texts, overall_texts = [], []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if scan_limit and scan_limit > 0 and i >= scan_limit:
                break
            obj = _json.loads(line)
            overall_texts.append(obj["text"])
            if obj["label"] == C.LABEL2ID["差评"]:
                neg_texts.append(obj["text"])
    report = analyze_negatives(neg_texts, overall_texts, top_k, min_count, min_lift)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def load_report():
    if os.path.exists(C.KEYWORDS_PATH):
        with open(C.KEYWORDS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    p = argparse.ArgumentParser(description="差评高频词/主要问题分析")
    p.add_argument("--data", default=C.RAW_DATA)
    p.add_argument("--scan-limit", type=int, default=C.KEYWORD_SCAN_LIMIT)
    p.add_argument("--top-k", type=int, default=C.KEYWORD_TOP_K)
    p.add_argument("--min-lift", type=float, default=C.KEYWORD_MIN_LIFT)
    p.add_argument("--out", default=C.KEYWORDS_PATH)
    args = p.parse_args()

    report = build_from_dataset(args.data, args.scan_limit, args.out,
                                top_k=args.top_k, min_lift=args.min_lift)
    print(f"扫描差评 {report['n_negative']} 条 / 整体 {report['n_overall']} 条")
    print(f"\n高频问题词 Top-15（lift>={args.min_lift}）：")
    for kw in report["top_keywords"][:15]:
        print(f"  {kw['word']:<6} 次数={kw['count']:<5} lift={kw['lift']:<6} score={kw['score']}")
    print("\n主要问题维度：")
    for a in report["aspects"]:
        print(f"  {a['aspect']}: {a['ratio']*100:.1f}%  代表词 "
              f"{[w['word'] for w in a['words'][:5]]}")
    print(f"\n已保存 -> {args.out}")


if __name__ == "__main__":
    main()
