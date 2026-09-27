"""共享测试夹具：小词表 + 微型 decoder-only 模型（CPU，秒级）。"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ModelConfig, PREFIX, MID, SEP, POS_CHAR, NEG_CHAR, RATING_CHARS, TEMPLATE_CHARS  # noqa: E402
from src.model import GPT                                                                         # noqa: E402
from src.vocab import Vocab                                                                       # noqa: E402


class TinyVocab(Vocab):
    pass


@pytest.fixture(scope="session")
def vocab():
    """固定小词表：特殊符 + 模板字符 + 常用汉字，保证 encode 总能命中。"""
    chars = list(dict.fromkeys(
        ["<pad>", "<bos>", "<eos>", "<unk>"] + list(TEMPLATE_CHARS) +
        list(POS_CHAR + NEG_CHAR + "".join(RATING_CHARS)) +
        list("这部电影很好看不好看剧情拖沓特效演员表演结局仓促牛逼棒烂垃圾一般还行喜欢讨厌笑点泪点画面音乐节奏"
             "的了我是不在有个和都很就也没都说可以一二三四五六七八九十")))
    return TinyVocab(chars)


@pytest.fixture(scope="session")
def tiny_cfg(vocab):
    return ModelConfig(vocab_size=len(vocab), d_model=32, n_layer=2, n_head=4, d_ff=64, max_len=64, dropout=0.0)


@pytest.fixture(scope="session")
def tiny_model(tiny_cfg, vocab):
    torch.manual_seed(0)
    m = GPT(tiny_cfg, vocab.stoi)
    m.eval()
    return m


@pytest.fixture(scope="session")
def sep_ids(vocab):
    return torch.tensor([vocab.id(c) for c in SEP], dtype=torch.long)
