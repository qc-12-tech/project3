"""按「具体评价内容」派生购买建议标签（不依赖原情感标签）。

读取 data/reviews.jsonl，逐条分析评价文本里的好评/差评描述，给出购买建议：
    - 卫生/安全问题（虫/头发/异物）→ 一票否决 = 不建议购买(2)
    - 好评描述 >= 2 且无差评描述        → 推荐购买(0)
    - 好评与差评并存（褒贬并存）        → 谨慎购买(1)
    - 有差评描述且无好评描述            → 不建议购买(2)
    - 其余（中性/无法判断）             → 谨慎购买(1)

输出 data/advice.jsonl，供 src/train_advice.py 训练。

用法：
    python -m src.label_advice                # 全量
    python -m src.label_advice --limit 100000 # 只处理前 N 条
"""
import argparse
import json
import os
from collections import Counter

from src import config as C
from src.generate_data import MILD_NEG, MILD_POS, NEG, POS

# 卫生/安全类“一票否决”词：只要差评里出现，直接判为不建议购买
SAFETY_WORDS = ["虫", "头发", "异物"]


def _count(text, pool):
    n = 0
    for phrases in pool.values():
        for ph in phrases:
            if ph in text:
                n += 1
    return n


def derive_advice_label(text):
    """根据评价内容给出购买建议标签：0=推荐购买 / 1=谨慎购买 / 2=不建议购买。"""
    # 卫生/安全一票否决
    for phrases in NEG.values():
        for ph in phrases:
            if ph in text and any(w in ph for w in SAFETY_WORDS):
                return 2

    pos = _count(text, POS) + sum(1 for ph in MILD_POS if ph in text)
    neg = _count(text, NEG) + sum(1 for ph in MILD_NEG if ph in text)

    if pos >= 2 and neg == 0:
        return 0
    if neg >= 1 and pos == 0:
        return 2
    if pos >= 1 and neg >= 1:
        return 1
    return 1


def build_labeled(in_path=C.RAW_DATA, out_path=C.ADVICE_DATA, limit=None):
    texts, labels = [], []
    with open(in_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            obj = json.loads(line)
            t = obj["text"]
            texts.append(t)
            labels.append(derive_advice_label(t))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for t, l in zip(texts, labels):
            f.write(json.dumps({"text": t, "label": l}, ensure_ascii=False) + "\n")
    return texts, labels


def main():
    p = argparse.ArgumentParser(description="按内容派生购买建议标签")
    p.add_argument("--in", dest="in_path", default=C.RAW_DATA)
    p.add_argument("--out", default=C.ADVICE_DATA)
    p.add_argument("--limit", type=int, default=0, help="只处理前 N 条，0=全部")
    args = p.parse_args()

    texts, labels = build_labeled(args.in_path, args.out, args.limit or None)
    dist = Counter(labels)
    print(f"已写入 {args.out}，共 {len(labels)} 条")
    print("建议分布：", {C.ADVICE_LABELS[k]: v for k, v in sorted(dist.items())})
    print("\n示例：")
    for i in range(8):
        print(f"  [{C.ADVICE_LABELS[labels[i]]}] {texts[i]}")


if __name__ == "__main__":
    main()
