"""词表 / 数据编码 / 生成式模板 / 分桶采样器 的单元测试。

其中 test_bucket_sampler_* 是针对真实踩过的 bug 的回归测试：
采样器早期用 `range(0, len-bs+1, bs)` 切批，桶内样本不足一批时**整桶被静默丢弃**，
导致验证集只评测到 128 条且长度偏斜。
"""
import numpy as np
import pytest
import torch

from src.config import (PREFIX, MID, SEP, EOS_ID, BOS_ID, PAD_ID, POS_CHAR, NEG_CHAR, ModelConfig)
from src.data import collate, make_full_ids, make_prefix_ids, star_to_sentiment
from src.train import BucketSampler
from src.vocab import build_vocab


def test_vocab_contains_template_and_labels(vocab):
    for c in ["影", "评", "：", "\n", "情", "感", " ", "分", "好", "差", "1", "5"]:
        assert c in vocab.stoi, f"模板字符 {c!r} 缺失"
    assert vocab.itos[0] == "<pad>" and vocab.itos[1] == "<bos>"


def test_build_vocab_orders_by_frequency_and_respects_max_size():
    texts = ["好好好", "好看", "差", "很差"]
    itos = build_vocab(texts, min_freq=1, max_size=30)
    assert len(itos) <= 30
    assert "好" in itos and "差" in itos
    assert itos.index("好") < itos.index("差")          # 高频在前


def test_star_to_sentiment():
    assert [star_to_sentiment(s) for s in (1, 2, 3, 4, 5)] == [0, 0, -1, 1, 1]


def test_prefix_is_a_prefix_of_full_sequence(vocab, tiny_cfg):
    text = "剧情拖沓但演员表演在线"
    ids, _ = make_full_ids(vocab, text, 5, tiny_cfg)
    pre = make_prefix_ids(vocab, text, tiny_cfg)
    assert ids[:len(pre)] == pre
    assert pre[0] == BOS_ID and pre[-1] == vocab.id("：")   # 前缀以「情感：」结尾


def test_full_labels_land_on_generative_targets(vocab, tiny_cfg):
    ids, labels = make_full_ids(vocab, "很好看", 5, tiny_cfg)
    assert len(ids) == len(labels)
    sent_i = labels.index(vocab.id(POS_CHAR))
    assert ids[sent_i - 1] == vocab.id("：")               # 情感位紧跟「情感：」
    rate_i = labels.index(vocab.id("5"))
    assert ids[rate_i - 1] == vocab.id("：")               # 评分位紧跟「评分：」
    assert labels[-1] == EOS_ID                            # 学会终止
    assert set(l for l in labels if l != -100) == {vocab.id(POS_CHAR), vocab.id("5"), EOS_ID}


def test_neutral_review_masks_sentiment_but_keeps_rating(vocab, tiny_cfg):
    ids, labels = make_full_ids(vocab, "一般般", 3, tiny_cfg)
    assert vocab.id(POS_CHAR) not in labels and vocab.id(NEG_CHAR) not in labels
    assert vocab.id("3") in labels


def test_long_text_is_truncated_within_max_len(vocab, tiny_cfg):
    ids, labels = make_full_ids(vocab, "很好看" * 200, 5, tiny_cfg)
    assert len(ids) <= tiny_cfg.max_len and len(labels) == len(ids)
    assert labels[-1] == EOS_ID                            # 截断后标签仍完整


def test_collate_pads_to_multiple_of_16_and_masks_pad_labels():
    batch = [(np.arange(5) + 4, np.array([4, -100, -100, -100, 7])),
             (np.arange(33) + 4, np.array([4] + [-100] * 32))]
    ids, lab = collate(batch)
    assert ids.shape[1] % 16 == 0 and ids.shape[1] >= 33   # 形状对齐到 16 的倍数（MPS 分配器友好）
    assert int(lab[0, 5:].max()) == -100                   # 右 padding 的标签全为 -100
    assert (ids[0, 5:] == PAD_ID).all()


def test_bucket_sampler_covers_every_sample_exactly_once():
    lengths = np.array([20, 21, 60, 61, 62, 130, 131, 144, 8, 9, 200, 201] * 6)
    s = BucketSampler(lengths, batch_size=8, shuffle=False, token_budget=8192, min_batch=4)
    seen = [i for b in s for i in b]
    assert sorted(seen) == list(range(len(lengths))), "有样本被丢弃或重复（回归：桶内不足一批时整桶丢弃）"


def test_bucket_sampler_keeps_batch_lengths_homogeneous():
    lengths = np.array([20] * 40 + [140] * 40)
    s = BucketSampler(lengths, batch_size=8, shuffle=False)
    for b in s:
        assert max(lengths[b]) - min(lengths[b]) <= 16, "批内长度差异过大，padding 浪费"


def test_bucket_sampler_long_sequences_use_smaller_batches():
    lengths = np.full(80, 144)
    s = BucketSampler(lengths, batch_size=128, shuffle=False, token_budget=8192)
    sizes = {len(b) for b in s}
    assert max(sizes) <= 8192 // 144 + 1, "长序列未按 token 预算收缩批大小（显存会爆）"
