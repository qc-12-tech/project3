"""字符级词表：构建 / 保存 / 加载 / 编码。"""
import json
from collections import Counter

from src import config as C


class Vocab:
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, char2id):
        self.char2id = char2id
        self.id2char = {i: c for c, i in char2id.items()}
        self.pad_id = char2id[self.PAD_TOKEN]
        self.unk_id = char2id[self.UNK_TOKEN]

    def __len__(self):
        return len(self.char2id)

    @classmethod
    def build(cls, texts, min_freq=C.MIN_CHAR_FREQ):
        counter = Counter()
        for text in texts:
            counter.update(text)
        char2id = {cls.PAD_TOKEN: C.PAD_ID, cls.UNK_TOKEN: C.UNK_ID}
        for ch, freq in counter.most_common():
            if freq < min_freq:
                continue
            if ch in char2id:
                continue
            char2id[ch] = len(char2id)
        return cls(char2id)

    def encode(self, text, max_len=C.MAX_LEN):
        ids = [self.char2id.get(ch, self.unk_id) for ch in text][:max_len]
        if len(ids) < max_len:
            ids += [self.pad_id] * (max_len - len(ids))
        return ids

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"char2id": self.char2id}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return cls(obj["char2id"])
