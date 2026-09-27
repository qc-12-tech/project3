"""字符级词表：无需任何预训练权重，从影评语料统计高频字符。

模板字符（影评：/情感：/评分：/好差1-5 等）强制入表，保证 encode 不会退化成 <unk>。
"""
import json
from collections import Counter
from pathlib import Path

from .config import (PAD, BOS, EOS, UNK, SPECIALS, TEMPLATE_CHARS, POS_CHAR, NEG_CHAR,
                     RATING_CHARS, ART_DIR)


def build_vocab(texts, min_freq: int = 3, max_size: int = 8000) -> list:
    """统计字符频次，返回 itos 列表（id -> char）。"""
    cnt = Counter()
    for t in texts:
        cnt.update(t)
    forced = list(SPECIALS) + list(dict.fromkeys(TEMPLATE_CHARS + POS_CHAR + NEG_CHAR + "".join(RATING_CHARS)))
    itos = list(dict.fromkeys(forced))                     # 特殊符 + 模板字符优先
    budget = max(0, max_size - len(itos))
    for ch, c in cnt.most_common():
        if len(itos) >= max_size or budget <= 0:
            break
        if ch in forced or c < min_freq:
            continue
        itos.append(ch)
        budget -= 1
    return itos


class Vocab:
    def __init__(self, itos: list):
        self.itos = list(itos)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text: str) -> list:
        stoi, unk = self.stoi, self.stoi[UNK]
        return [stoi.get(c, unk) for c in text]

    def id(self, ch: str) -> int:
        return self.stoi.get(ch, self.stoi[UNK])

    def save(self, path: Path):
        Path(path).write_text(json.dumps({"itos": self.itos}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Vocab":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["itos"])


def build_and_save(sample_texts, out_path=None, min_freq: int = 3, max_size: int = 8000) -> Vocab:
    out_path = Path(out_path or (ART_DIR / "vocab.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    v = Vocab(build_vocab(sample_texts, min_freq=min_freq, max_size=max_size))
    v.save(out_path)
    return v


if __name__ == "__main__":
    import sys
    from .data import load_clean
    df = load_clean(limit=int(sys.argv[1]) if len(sys.argv) > 1 else 200_000)
    v = build_and_save(df["comment"].tolist())
    print(f"vocab size = {len(v)} -> {ART_DIR/'vocab.json'}")
    print("前 60 个 token:", "".join(v.itos[:60]))
