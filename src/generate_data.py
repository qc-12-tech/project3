"""模拟生成外卖评价数据集（好评 / 中评 / 差评）。

用法：
    python -m src.generate_data --n 1000000 --out data/reviews.jsonl
    python -m src.generate_data --n 1000 --out data/sample_1000.jsonl

生成方式：从各维度（口味/配送/包装/份量/温度/价格/服务/卫生）的短语池中
随机组合成句，并混入少量“情绪矛盾”的难例，使模型的不确定度有真实意义。
"""
import argparse
import json
import os
import random
import time

from src import config as C


# ============================================================
# 短语池
# ============================================================
POS = {
    "口味": ["味道很赞", "味道超棒", "口味非常好", "特别好吃", "味道正宗",
             "很入味", "香气十足", "鲜香可口", "咸淡刚刚好", "越吃越香"],
    "配送": ["配送很快", "送得特别及时", "骑手很给力", "出餐很快",
             "半小时就送到了", "比预计时间还早", "送得飞快"],
    "包装": ["包装很精致", "包装完好无损", "一点都没洒", "包装严实", "打包很用心"],
    "份量": ["分量很足", "量特别大", "吃得特别饱", "份量实在", "满满一大盒"],
    "温度": ["送到还是热乎的", "打开还冒着热气", "温度刚刚好", "热腾腾的"],
    "价格": ["性价比很高", "价格很实惠", "特别划算", "一点都不贵", "物超所值"],
    "服务": ["服务态度很好", "商家特别贴心", "客服回复很快", "老板人很好"],
    "卫生": ["食材很新鲜", "干净又卫生", "看着就很放心"],
}

NEG = {
    "口味": ["味道很难吃", "太咸了", "太油腻了", "一点味道都没有", "难以下咽",
             "食材不新鲜", "有一股怪味", "咸得发苦", "又干又柴", "完全没入味"],
    "配送": ["送了两个小时", "配送太慢了", "骑手绕了很久", "等了快一个半小时",
             "严重超时", "饭点过了才送到", "送得也太慢了"],
    "包装": ["包装都破了", "汤全洒出来了", "洒得到处都是", "盒子压扁了", "漏了一袋子"],
    "份量": ["分量少得可怜", "量太少了", "根本吃不饱", "缺斤少两", "就一点点东西"],
    "温度": ["送到都凉透了", "冷冰冰的", "一点热气都没有", "凉得像剩饭"],
    "价格": ["太贵了", "完全不值这个价", "性价比太低", "又贵又难吃", "被坑了"],
    "服务": ["服务态度很差", "商家根本不理人", "客服态度恶劣", "问什么都不回"],
    "卫生": ["吃出头发了", "里面有虫子", "太不干净了", "卫生堪忧", "菜里有异物"],
}

NEU = [
    "味道一般", "中规中矩", "还行吧", "没有特别惊艳", "就那样", "普通水平",
    "一般般", "无功无过", "凑合能吃", "说不上好也说不上差", "马马虎虎",
    "普普通通", "没什么记忆点", "中规中矩的一家店",
]

# 缓和/倾向性连接词
CONNECTORS = ["不过", "但是", "就是", "而且", "另外", "总的来说", "说实话",
              "讲真", "感觉", "整体来说", "客观地说"]
# 按维度选择主语，避免出现“商家咸淡刚刚好”这类不搭配
SUBJECTS_BY_ASPECT = {
    "口味": ["这家店", "这家的菜", "菜", "外卖"],
    "配送": ["骑手", "外卖", "这单外卖"],
    "包装": ["", "这单外卖", "外卖"],
    "份量": ["这家店", "这家的菜", "菜", "外卖"],
    "温度": ["外卖", "这单外卖", "菜"],
    "价格": ["这家店", "外卖", "这单外卖"],
    "服务": ["商家", "老板", "客服"],
    "卫生": ["这家店", "菜", "外卖"],
}
PREFIXES = ["", "这次", "第一次点", "朋友推荐来的", "看评分挺高就点了", "整体来说"]

# 好评收尾
POS_TAIL = ["好评", "会回购", "还会再点", "推荐大家试试", "五星好评",
            "下次还来", "很满意的一次", "值得推荐", "爱了爱了"]
# 差评收尾
NEG_TAIL = ["差评", "不会再点了", "要退款", "太失望了", "避雷", "无语",
            "必须投诉", "给一星", "再也不会来", "气死了"]
# 中评收尾
NEU_TAIL = ["就这样吧", "暂时不会回购", "看情况吧", "希望能改进", "下次再观望"]

# 边界样本：混入轻微反向描述，制造真正模糊的评价
MILD_NEG = ["就是配送有点慢", "包装稍微有点简陋", "价格有点小贵", "份量一般",
            "送得比预计晚了一点", "味道比较普通"]
MILD_POS = ["味道倒是还可以", "配送速度还行", "包装还算完好",
            "其实也没有那么差", "商家态度还可以", "分量还算足"]

PUNCT = ["，", "，", "，", "。", "！"]


def _sample(phrase_pool):
    """随机取一个维度及该维度下的短语，返回 (维度, 短语)。"""
    aspect = random.choice(list(phrase_pool.keys()))
    return aspect, random.choice(phrase_pool[aspect])


def _clause(phrase_pool):
    """取一个短语并配上合适的主语。"""
    aspect, phrase = _sample(phrase_pool)
    subject = random.choice(SUBJECTS_BY_ASPECT.get(aspect, [""]))
    if not subject or phrase.startswith(subject):
        return phrase
    return f"{subject}{phrase}"


def _join(clauses):
    """把分句拼接成自然句子。"""
    text = clauses[0]
    for c in clauses[1:]:
        if random.random() < 0.12:
            text += random.choice(CONNECTORS) + c
        else:
            text += random.choice(["，", "，", "。"]) + c
    if not text.endswith(("！", "。", "~")):
        text += random.choice(["。", "！", ""])
    return text


def make_positive():
    n = random.choices([1, 2, 3], weights=[0.08, 0.45, 0.47])[0]
    clauses = [_clause(POS) for _ in range(n)]
    if random.random() < C.MIX_RATIO:
        clauses.insert(random.randint(0, len(clauses)), random.choice(MILD_NEG))
    if random.random() < 0.55:
        clauses.append(random.choice(POS_TAIL))
    return _join(clauses)


def make_negative():
    n = random.choices([1, 2, 3], weights=[0.10, 0.42, 0.48])[0]
    clauses = [_clause(NEG) for _ in range(n)]
    if random.random() < C.MIX_RATIO:
        clauses.insert(random.randint(0, len(clauses)), random.choice(MILD_POS))
    if random.random() < 0.60:
        clauses.append(random.choice(NEG_TAIL))
    return _join(clauses)


def make_neutral():
    r = random.random()
    if r < 0.45:
        n = random.choices([1, 2], weights=[0.4, 0.6])[0]
        clauses = [random.choice(NEU) for _ in range(n)]
    elif r < 0.80:
        # 一褒一贬 -> 真正的“中评”
        clauses = [_clause(POS), _clause(NEG)]
    else:
        clauses = [_clause(POS), random.choice(NEU)]
    if random.random() < 0.4:
        clauses.append(random.choice(NEU_TAIL))
    return _join(clauses)


GENERATORS = [make_positive, make_neutral, make_negative]  # 对应 好评/中评/差评


def generate_one(label_id):
    text = GENERATORS[label_id]()
    prefix = random.choice(PREFIXES)
    if prefix:
        text = prefix + "，" + text
    return text


def generate_dataset(n, label_ratio=None, seed=C.SEED, label_noise=C.LABEL_NOISE,
                     show_every=100_000):
    """返回 (texts, labels) 两个列表。label_noise 模拟标注噪声。"""
    ratio = label_ratio or C.LABEL_RATIO
    random.seed(seed)
    counts = [int(n * r) for r in ratio]
    counts[-1] = n - sum(counts[:-1])  # 补齐

    texts, labels = [], []
    for label_id, cnt in enumerate(counts):
        for _ in range(cnt):
            texts.append(generate_one(label_id))
            labels.append(label_id)
        if show_every:
            print(f"  生成 {C.LABELS[label_id]}: {cnt} 条")

    # 标注噪声：随机翻转一部分标签（真实标注也会有错/主观分歧）
    if label_noise > 0:
        flipped = 0
        for i in range(n):
            if random.random() < label_noise:
                labels[i] = random.choice([x for x in range(len(C.LABELS))
                                           if x != labels[i]])
                flipped += 1
        if show_every:
            print(f"  注入标注噪声: {flipped} 条 ({label_noise*100:.1f}%)")

    # 打乱
    order = list(range(n))
    random.shuffle(order)
    texts = [texts[i] for i in order]
    labels = [labels[i] for i in order]
    return texts, labels


def write_jsonl(texts, labels, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for text, label in zip(texts, labels):
            f.write(json.dumps({"text": text, "label": label},
                               ensure_ascii=False) + "\n")


def read_jsonl(path, limit=None):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            obj = json.loads(line)
            texts.append(obj["text"])
            labels.append(obj["label"])
    return texts, labels


def main():
    parser = argparse.ArgumentParser(description="生成模拟外卖评价数据")
    parser.add_argument("--n", type=int, default=C.TOTAL_SAMPLES)
    parser.add_argument("--out", type=str, default=C.RAW_DATA)
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--label-noise", type=float, default=C.LABEL_NOISE,
                        help="标注噪声比例（0~1）")
    args = parser.parse_args()

    t0 = time.time()
    print(f"开始生成 {args.n} 条模拟评价 ...")
    texts, labels = generate_dataset(args.n, seed=args.seed,
                                     label_noise=args.label_noise)
    write_jsonl(texts, labels, args.out)
    print(f"已写入 {args.out}，耗时 {time.time() - t0:.1f}s")

    # 打印几条示例
    print("\n示例：")
    for i in range(6):
        print(f"  [{C.LABELS[labels[i]]}] {texts[i]}")


if __name__ == "__main__":
    main()
