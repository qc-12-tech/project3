"""训练 decoder-only 因果 LM：情感位 + 评分位双目标（生成式），中评屏蔽情感位。

用法：
    python -m src.train --epochs 2 --max-train 400000
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .config import (ART_DIR, CACHE_DIR, ModelConfig, TrainConfig, pick_device, POS_CHAR, NEG_CHAR,
                     RATING_CHARS, PAD_ID)
from .data import LMDataset, collate
from .model import GPT, count_params
from .vocab import Vocab


class BucketSampler(torch.utils.data.Sampler):
    """按序列长度分桶 + 按 token 预算动态批大小。

    - 分桶：按「长度 / bucket_width 向上取整」分桶，批内长度接近，padding 浪费从 ~2.75x 降到 ~1.15x；
    - 动态批：批大小 = token_budget / 该桶宽度（上限 batch_size），
      这样长序列自动变小批，反向传播的激活内存被钉在常数上（本机统一内存只有 8.6GB，
      否则 L=144、B=128 时 MPS 会 OOM）。
    """

    def __init__(self, lengths, batch_size: int, shuffle: bool = True, seed: int = 0,
                 bucket_width: int = 16, token_budget: int = 8192, min_batch: int = 8):
        self.lengths = np.asarray(lengths, dtype=np.float64)
        self.bs = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.width = bucket_width
        self.tokens = token_budget
        self.min_bs = min_batch
        self.epoch = 0
        self.bucket = np.ceil(self.lengths / self.width).astype(np.int64)
        self.plan = self._plan()

    def _bs_for(self, bucket_id: int) -> int:
        return int(max(self.min_bs, min(self.bs, self.tokens // max(1, bucket_id * self.width))))

    def _plan(self):
        counts = np.bincount(self.bucket)
        return [(b, self._bs_for(b), int(np.ceil(c / self._bs_for(b)))) for b, c in enumerate(counts) if c > 0]

    def __len__(self):
        return sum(nb for _, _, nb in self.plan)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        buckets = list(range(len(self.bucket)))
        if self.shuffle:
            rng.shuffle(buckets)
        for b in buckets:
            idx = np.where(self.bucket == b)[0]
            if self.shuffle:
                rng.shuffle(idx)
            bs = self._bs_for(b)
            # 注意：末尾不足一批也要发出，否则样本会被静默丢弃（曾导致验证集只评测 128 条）
            batches = [idx[i:i + bs].tolist() for i in range(0, len(idx), bs)]
            if self.shuffle:
                rng.shuffle(batches)
            yield from batches


def build_loaders(tcfg: TrainConfig, batch_size: int):
    tr = LMDataset(CACHE_DIR, "train")
    va = LMDataset(CACHE_DIR, "val")
    rng = np.random.default_rng(tcfg.seed)

    def sub(ds, n):
        n = min(n, len(ds))
        return Subset(ds, rng.choice(len(ds), size=n, replace=False).tolist())

    tr, va = sub(tr, tcfg.max_train_samples), sub(va, tcfg.max_val_samples)

    def lengths(ds):
        """直接从 offsets 算长度，避免逐条 mmap 读取。"""
        idx = np.asarray(ds.indices, dtype=np.int64)
        off = np.asarray(ds.dataset.off)
        return off[idx + 1] - off[idx]

    kw = dict(bucket_width=16, token_budget=tcfg.token_budget)
    tr_loader = DataLoader(tr, batch_sampler=BucketSampler(lengths(tr), batch_size, True, tcfg.seed, **kw),
                           collate_fn=collate)
    va_loader = DataLoader(va, batch_sampler=BucketSampler(lengths(va), batch_size, False, tcfg.seed, **kw),
                           collate_fn=collate)
    return tr_loader, va_loader, len(tr), len(va)


def forward_loss(model: GPT, ids, labels, smoothing: float):
    """只在监督位（情感位/评分位/<eos>）上算交叉熵。

    decoder-only 的 next-token 对齐：用 hidden[:, :-1] 预测 labels[:, 1:]。
    只把监督位的隐状态送进 LM Head（而不是整条序列），显存/算力都省 ~100 倍。
    """
    hidden, _ = model.encode(ids)                      # [B, L, d]
    lab = labels[:, 1:]
    mask = lab != -100
    if not bool(mask.any()):
        return None, None, None
    logits = model.head(hidden[:, :-1][mask])          # [M, V]
    gold = lab[mask]
    return torch.nn.functional.cross_entropy(logits, gold, label_smoothing=smoothing), logits, gold


@torch.no_grad()
def evaluate(model: GPT, loader, device, max_batches=None):
    model.eval()
    pos_id, neg_id = model.pos_id, model.neg_id
    rating_ids = torch.tensor(model.rating_ids, device=device)
    n_sent = n_sent_ok = 0
    n_rate = n_rate_ok = 0
    abs_err = 0.0
    tp = fp = fn = 0
    probs, golds = [], []
    for bi, (ids, labels) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        ids, labels = ids.to(device), labels.to(device)
        _, logits, gold = forward_loss(model, ids, labels, 0.0)
        if logits is None:
            continue
        sent_mask = (gold == pos_id) | (gold == neg_id)
        rate_mask = torch.isin(gold, rating_ids)
        if sent_mask.any():
            pair = torch.softmax(logits[sent_mask][:, [pos_id, neg_id]].float(), -1)   # [:,0]=P(好)
            probs.append(pair[:, 0].cpu().numpy())
            golds.append((gold[sent_mask] == pos_id).cpu().numpy().astype(np.int8))
            # 注意：pair 第 0 列才是 P(好)，不能直接用 argmax 与「gold==pos_id」比较，
            # 否则 1 表示“更倾向差”而 g=1 表示“标签是好”，语义相反会把准确率算成 1-acc。
            pred = (pair[:, 0] >= 0.5).long()
            g = (gold[sent_mask] == pos_id).long()
            n_sent += g.numel()
            n_sent_ok += int((pred == g).sum())
            tp += int(((pred == 1) & (g == 1)).sum())
            fp += int(((pred == 1) & (g == 0)).sum())
            fn += int(((pred == 0) & (g == 1)).sum())
        if rate_mask.any():
            pred = logits[rate_mask][:, rating_ids].argmax(-1) + 1
            g = gold[rate_mask] - rating_ids[0] + 1
            n_rate += g.numel()
            n_rate_ok += int((pred == g).sum())
            abs_err += float((pred - g).abs().sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {"sent_acc": n_sent_ok / max(1, n_sent), "sent_precision_pos": prec, "sent_recall_pos": rec,
            "sent_f1_pos": 2 * prec * rec / max(1e-9, prec + rec),
            "rating_acc": n_rate_ok / max(1, n_rate), "rating_mae": abs_err / max(1, n_rate),
            "n_sent_eval": n_sent, "n_rate_eval": n_rate,
            "_probs": np.concatenate(probs) if probs else np.zeros(0, dtype=np.float32),
            "_golds": np.concatenate(golds) if golds else np.zeros(0, dtype=np.int8)}


def best_threshold(probs: np.ndarray, golds: np.ndarray):
    """在验证集上扫描阈值，最大化「好评」类的 F1（修正正类偏多导致的阈值漂移）。"""
    if len(probs) == 0:
        return 0.5, {}
    order = np.argsort(-probs)
    p, g = probs[order], golds[order].astype(np.int64)
    tp = np.cumsum(g)
    fp = np.cumsum(1 - g)
    P = max(1, int(g.sum()))
    prec = tp / np.maximum(1, tp + fp)
    rec = tp / P
    f1 = 2 * prec * rec / np.maximum(1e-9, prec + rec)
    i = int(np.argmax(f1))
    thr = float(p[i])
    pred = probs >= thr
    tp2 = int(((pred == 1) & (golds == 1)).sum())
    fp2 = int(((pred == 1) & (golds == 0)).sum())
    fn2 = int(((pred == 0) & (golds == 1)).sum())
    pr = tp2 / max(1, tp2 + fp2)
    rc = tp2 / max(1, tp2 + fn2)
    return thr, {"acc": float((pred == golds).mean()), "precision_pos": pr, "recall_pos": rc,
                 "f1_pos": 2 * pr * rc / max(1e-9, pr + rc), "pos_rate_pred": float(pred.mean()),
                 "pos_rate_true": float(golds.mean()), "n": int(len(golds))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--n-layer", type=int, default=None)
    ap.add_argument("--n-head", type=int, default=None)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str, default=str(ART_DIR / "model.pt"))
    ap.add_argument("--smoke", action="store_true", help="小样本冒烟测试")
    ap.add_argument("--eval-only", action="store_true", help="只评测已有 checkpoint 并写回指标")
    a = ap.parse_args()
    if a.eval_only:
        return eval_only(a)

    vocab = Vocab.load(ART_DIR / "vocab.json")
    mcfg = ModelConfig(vocab_size=len(vocab))
    tcfg = TrainConfig(device=a.device)
    for k, v in (("epochs", a.epochs), ("max_train_samples", a.max_train), ("max_val_samples", a.max_val),
                 ("batch_size", a.batch_size), ("lr", a.lr)):
        if v is not None:
            setattr(tcfg, k, v)
    for k, v in (("d_model", a.d_model), ("n_layer", a.n_layer), ("n_head", a.n_head)):
        if v is not None:
            setattr(mcfg, k, v)
    if a.smoke:
        tcfg.epochs, tcfg.max_train_samples, tcfg.max_val_samples, tcfg.batch_size = 1, 2048, 512, 64
        tcfg.log_every = 5

    device = pick_device(tcfg.device)
    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)
    model = GPT(mcfg, vocab.stoi).to(device)
    print(f"decoder-only GPT: {count_params(model)/1e6:.2f}M 参数 | d_model={mcfg.d_model} "
          f"n_layer={mcfg.n_layer} n_head={mcfg.n_head} d_ff={mcfg.d_ff} vocab={mcfg.vocab_size} | device={device}")

    tr_loader, va_loader, n_tr, n_va = build_loaders(tcfg, tcfg.batch_size)
    steps_per_epoch = max(1, len(tr_loader))
    total = steps_per_epoch * tcfg.epochs
    warmup = max(20, int(total * tcfg.warmup_ratio))
    print(f"训练样本 {n_tr} / 验证样本 {n_va} | steps/epoch={steps_per_epoch} 总 steps={total} warmup={warmup}")

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tcfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=tcfg.lr, betas=(0.9, 0.95))

    def lr_at(step):
        if step < warmup:
            return tcfg.lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return tcfg.lr * (tcfg.min_lr_ratio + (1 - tcfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))

    best = -1.0
    log = {"config": {"model": mcfg.to_dict(), "train": tcfg.to_dict(), "device": device,
                      "n_train": n_tr, "n_val": n_va}, "history": []}
    step = 0
    t0 = time.time()
    for ep in range(tcfg.epochs):
        model.train()
        run_loss, run_n = 0.0, 0
        for bi, (ids, labels) in enumerate(tr_loader):
            ids, labels = ids.to(device), labels.to(device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            loss, _, _ = forward_loss(model, ids, labels, tcfg.label_smoothing)
            if loss is None:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()
            run_loss += float(loss.detach()) * ids.size(0)
            run_n += ids.size(0)
            if step % 200 == 0 and hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()          # 释放分配器缓存，防止多 shape 场景下内存只涨不降
            if step % tcfg.log_every == 0:
                print(f"  ep{ep} step {step}/{total} loss {float(loss.detach()):.4f} "
                      f"lr {lr_at(step):.2e} {(time.time()-t0):.0f}s", flush=True)
            step += 1
        metrics = evaluate(model, va_loader, device)
        last_probs, last_golds = metrics.pop("_probs"), metrics.pop("_golds")
        thr_e, cal_e = best_threshold(last_probs, last_golds)
        metrics.update({"epoch": ep, "train_loss": run_loss / max(1, run_n), "elapsed_s": time.time() - t0,
                        "sent_threshold_cal": round(float(thr_e), 4),
                        "sent_acc_cal": cal_e.get("acc", metrics["sent_acc"]),
                        "sent_f1_cal": cal_e.get("f1_pos", metrics["sent_f1_pos"])})
        log["history"].append(metrics)
        print(f"[val] ep{ep} loss={metrics['train_loss']:.4f} sent_acc@0.5={metrics['sent_acc']:.4f} "
              f"sent_acc@cal={metrics['sent_acc_cal']:.4f}（阈值 {thr_e:.3f}）"
              f"rating_acc={metrics['rating_acc']:.4f} rating_mae={metrics['rating_mae']:.3f}", flush=True)
        # 用「校准阈值下的准确率+F1」选模型：P(好) 的绝对尺度会漂移，@0.5 的准确率不适合做选择标准
        score = metrics["sent_acc_cal"] + metrics["sent_f1_cal"] - metrics["rating_mae"] / 4
        if score > best:
            best = score
            torch.save({"model": mcfg.to_dict(), "state_dict": model.state_dict(),
                        "vocab_size": len(vocab), "metrics": metrics,
                        "pos_id": model.pos_id, "neg_id": model.neg_id, "rating_ids": model.rating_ids},
                       a.out)
            print(f"  -> 保存最佳模型 {a.out}（score={score:.4f}）", flush=True)

    log["best"] = best

    # ---- 阈值校准：用验证集最大化「好评」F1，修正正类偏多带来的概率漂移
    blob = torch.load(a.out, map_location="cpu", weights_only=False)
    best_model = GPT(mcfg, vocab.stoi)
    best_model.load_state_dict(blob["state_dict"])
    best_model.to(device).eval()
    m = evaluate(best_model, va_loader, device)
    thr, cal = best_threshold(m["_probs"], m["_golds"])
    blob["sent_threshold"] = thr
    blob["calibration"] = cal
    torch.save(blob, a.out)
    log["calibration"] = {"threshold": thr, **cal}
    log["val_at_0.5"] = {"sent_acc": m["sent_acc"], "sent_f1_pos": m["sent_f1_pos"]}
    (ART_DIR / "train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"阈值校准：P(好)>={thr:.4f} → val acc={cal['acc']:.4f} F1+={cal['f1_pos']:.4f} "
          f"（好评率 预测 {cal['pos_rate_pred']:.3f} vs 真实 {cal['pos_rate_true']:.3f}）")
    print(f"训练完成，用时 {(time.time()-t0)/60:.1f} min，最佳 score={best:.4f}")


def eval_only(a):
    """只对已有 checkpoint 做一次完整验证集评测 + 阈值校准，并清洗历史日志中失效的指标。"""
    vocab = Vocab.load(ART_DIR / "vocab.json")
    tcfg = TrainConfig()
    device = pick_device(a.device)
    _, va_loader, _, n_va = build_loaders(tcfg, tcfg.batch_size)
    blob = torch.load(a.out, map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**blob["model"])
    mcfg.vocab_size = len(vocab)
    model = GPT(mcfg, vocab.stoi)
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    met = evaluate(model, va_loader, device)
    thr, cal = best_threshold(met["_probs"], met["_golds"])
    model.sent_threshold = thr
    blob["sent_threshold"], blob["calibration"] = thr, cal
    torch.save(blob, a.out)
    p = ART_DIR / "train_log.json"
    log = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"config": {}}
    for h in log.get("history", []):
        for k in list(h):      # 旧版把「更倾向差」当成正类，语义与标签相反，指标无效，删除
            if k.startswith("sent_") and k not in ("sent_threshold_cal", "sent_acc_cal", "sent_f1_cal"):
                h.pop(k)
    log["final_eval"] = {k: (round(float(v), 4) if isinstance(v, float) else v)
                         for k, v in met.items() if not k.startswith("_")}
    log["final_eval"]["n_val"] = n_va
    log["calibration"] = {"threshold": thr, **cal}
    log["note"] = "final_eval/calibration 是在修复评估方向后，对最终 checkpoint 的完整验证集评测"
    p.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[eval-only] val: acc@0.5={met['sent_acc']:.4f} acc@cal={cal['acc']:.4f} "
          f"F1+={cal['f1_pos']:.4f}（阈值 {thr:.4f}）rating_acc={met['rating_acc']:.4f} "
          f"rating_mae={met['rating_mae']:.4f} n={met['n_sent_eval']}")


if __name__ == "__main__":
    main()
