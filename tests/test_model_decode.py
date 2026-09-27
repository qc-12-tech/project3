"""decoder-only 模型与两步自回归解码的单元测试。

核心是 test_two_step_decode_matches_full_forward：它同时验证 RoPE 位置偏移、
KV cache 拼接、以及「生成式读数 = 受限词表 softmax」三件事是否自洽。
早期实现里 `evaluate()` 用 `pair.argmax(-1)` 与 `gold==pos_id` 比较，语义相反导致
准确率被算成 1-acc，这里也一并加了回归测试。
"""
import math

import numpy as np
import torch

from src.config import ModelConfig, SEP, POS_CHAR, NEG_CHAR, RATING_CHARS
from src.data import make_prefix_ids
from src.model import GPT, RMSNorm, apply_rope, build_rope_cache
from src.train import best_threshold, forward_loss


def test_model_param_count_and_shapes(tiny_model, tiny_cfg, vocab):
    assert sum(p.numel() for p in tiny_model.parameters()) < 200_000
    ids = torch.randint(4, len(vocab), (3, 12))
    out = tiny_model(ids)
    assert out.shape == (3, 12, len(vocab))
    last = tiny_model(ids, last_only=True)
    assert last.shape == (3, 1, len(vocab))            # 只算最后一位，省掉 [B,L,V] 的显存


def test_causal_attention_does_not_look_ahead(tiny_model, vocab):
    """改动序列尾部不应影响前面的输出（因果性）。"""
    ids = torch.randint(4, len(vocab), (1, 10))
    a = tiny_model(ids)[0, :5]
    ids2 = ids.clone()
    ids2[0, 6:] = torch.randint(4, len(vocab), (4,))
    b = tiny_model(ids2)[0, :5]
    assert torch.allclose(a, b, atol=1e-5)


def test_rmsnorm_and_rope_are_sane():
    x = torch.randn(2, 5, 8)
    assert RMSNorm(8)(x).shape == x.shape
    cos, sin = build_rope_cache(8, 6, 10000.0, "cpu", torch.float32)
    assert cos.shape == (6, 4)
    q = torch.randn(1, 2, 6, 8)
    rotated = apply_rope(q, cos, sin)
    assert rotated.shape == q.shape
    # 位置 0 的旋转是恒等变换
    assert torch.allclose(rotated[:, :, 0], q[:, :, 0], atol=1e-5)


def test_sentiment_probs_are_pairwise_softmax(tiny_model):
    logits = torch.randn(4, 20) * 3
    p = tiny_model.sentiment_probs(logits)
    pair = torch.softmax(logits[:, [tiny_model.pos_id, tiny_model.neg_id]].float(), -1)
    assert torch.allclose(p, pair[:, 0], atol=1e-6)
    assert ((p >= 0) & (p <= 1)).all()


def test_two_step_decode_matches_full_forward(tiny_model, tiny_cfg, vocab, sep_ids):
    """第 2 步带 KV cache + 位置偏移的结果，必须与「整条序列一次前向」完全一致。"""
    texts = ["非常好看，剧情精彩", "很烂，浪费时间"]
    for t in texts:
        ids = torch.tensor([make_prefix_ids(vocab, t, tiny_cfg)], dtype=torch.long)
        p_pos, sent_id, exp_r, arg_r, dist = tiny_model.generate_sentiment_rating(ids, sep_ids)
        # ① 第 1 步：P(好) 应等于前缀末位受限 softmax
        last = tiny_model(ids, last_only=True)[0, -1]
        p_ref = torch.softmax(last[[tiny_model.pos_id, tiny_model.neg_id]].float(), 0)[0]
        assert torch.allclose(p_pos[0], p_ref, atol=1e-5)
        # ② 第 2 步：把情感 token + 评分前缀拼上去整条前向，评分分布应完全一致
        #    （两条路径的 SDPA 分支不同：带 cache 时不传 is_causal，浮点累加顺序不同，
        #     实测最大差异 ~8e-6，所以容差取 1e-4；另外再校验 argmax 完全一致）
        full = torch.cat([ids, torch.tensor([[int(sent_id[0])]]), sep_ids[None, :]], dim=1)
        ref = torch.softmax(tiny_model(full, last_only=True)[0, -1][tiny_model.rating_ids].float(), 0)
        assert torch.allclose(dist[0], ref, atol=1e-4), "KV cache / RoPE 位置偏移与整条前向不一致"
        assert float((dist[0] - ref).abs().max()) < 1e-4
        assert int(dist[0].argmax()) == int(ref.argmax())
        assert 1 <= float(exp_r[0]) <= 5 and int(arg_r[0]) in range(1, 6)
        assert abs(float((dist[0] * torch.arange(1, 6)).sum()) - float(exp_r[0])) < 1e-5


def test_rating_dist_is_over_five_tokens(tiny_model):
    logits = torch.randn(3, 100)
    d = tiny_model.rating_dist(logits)
    assert d.shape == (3, 5) and torch.allclose(d.sum(-1), torch.ones(3), atol=1e-6)


def test_forward_loss_only_supervises_label_positions(tiny_model, vocab, tiny_cfg):
    from src.data import make_full_ids
    ids, labels = make_full_ids(vocab, "很好看", 5, tiny_cfg)
    batch_ids = torch.tensor([ids, ids])
    batch_lab = torch.tensor([labels, labels])
    loss, logits, gold = forward_loss(tiny_model, batch_ids, batch_lab, 0.0)
    n_sup = int((batch_lab != -100).sum())
    assert logits.shape == (n_sup, len(vocab)) and gold.shape == (n_sup,)
    assert loss.item() > 0


def test_unlabeled_batch_returns_none(tiny_model):
    ids = torch.randint(4, 20, (2, 8))
    lab = torch.full((2, 8), -100)
    loss, logits, gold = forward_loss(tiny_model, ids, lab, 0.0)
    assert loss is None and logits is None


def test_best_threshold_recovers_separable_labels():
    """回归：校准必须基于 P(好) 原始概率；同时验证方向没有反。"""
    probs = np.concatenate([np.random.uniform(0.6, 0.99, 500), np.random.uniform(0.01, 0.4, 200)])
    golds = np.concatenate([np.ones(500, dtype=np.int8), np.zeros(200, dtype=np.int8)])
    thr, cal = best_threshold(probs, golds)
    assert cal["acc"] > 0.98 and cal["f1_pos"] > 0.98
    assert 0.3 < thr < 0.65
    # 方向反了的话准确率会掉到 ~0.3 以下
    inverted = best_threshold(probs, 1 - golds)[1]["acc"]
    assert cal["acc"] > inverted


def test_evaluate_accuracy_matches_manual_computation(tiny_model, vocab, tiny_cfg):
    """回归：修复前的 `pair.argmax(-1)` 与 `gold==pos_id` 语义相反，会把 acc 算成 1-acc。"""
    from src.data import make_full_ids
    from src.train import evaluate
    samples = []
    for text, star in [("很好看", 5), ("很烂", 1), ("剧情拖沓", 2), ("表演在线", 4)] * 4:
        ids, lab = make_full_ids(vocab, text, star, tiny_cfg)
        samples.append((np.array(ids), np.array(lab)))

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, i):
            return samples[i]

    from src.data import collate
    loader = torch.utils.data.DataLoader(DS(), batch_size=4, collate_fn=collate)
    met = evaluate(tiny_model, loader, "cpu")
    probs, golds = met["_probs"], met["_golds"]
    manual = float(((probs >= 0.5).astype(int) == golds).mean())
    assert abs(met["sent_acc"] - manual) < 1e-9, "evaluate 的准确率与手工计算不一致（方向反了）"
    assert abs(met["sent_acc"] - (1 - manual)) > 0.1 or manual == 0.5
