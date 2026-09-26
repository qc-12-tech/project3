"""主动学习再训练：把人工复核回填的真实标签合入训练集，增量微调模型。

闭环：低置信样本 -> 人工复核（store 回填 true_label）-> 本脚本合入训练集微调 -> 更准。

核心设计：
1. 从样本库读取 `true_label` 非空的样本（人工标注的"困难样本"）；
2. 复用原始语料的磁盘编码缓存（mmap，不重建、不污染 1M 缓存）；
3. 用 WeightedRandomSampler 对困难样本加权采样，使其在每个 batch 中占
   `--human-ratio`（默认 30%），从而真正影响梯度（困难样本通常只有几十条）；
4. 小学习率、少量 epoch 微调，避免灾难性遗忘；
5. 微调前后在同一批困难样本 + 一小片原始样本上评估，量化闭环收益。

用法：
    python -m src.retrain                         # 默认 1 epoch，lr=3e-5，20 万原始样本
    python -m src.retrain --epochs 2 --lr 1e-5 --human-ratio 0.5
    python -m src.retrain --max-orig-samples 0    # 使用全部原始样本（更慢但更稳）
"""
import argparse
import json
import os
import shutil
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src import config as C
from src.dataset import ReviewDataset, build_or_load_cache, collate_fn
from src.model import TransformerClassifier
from src.store import ReviewStore
from src.train import build_scheduler, pick_device
from src.vocab import Vocab


def load_human_labeled(store_path=C.DB_PATH):
    """从样本库读取人工标注样本，返回 [(text, label_id), ...]（按文本去重，冲突取最新）。"""
    store = ReviewStore(store_path)
    labeled = []
    for r in store.human_labeled():
        lid = C.LABEL2ID.get(r["true_label"])
        if lid is None:
            continue
        labeled.append((r["text"], lid))
    # 按文本去重：同一文本保留最后一次标注（human_labeled 按 id 降序，靠前的更新）
    dedup = OrderedDict()
    for text, lid in labeled:
        dedup.setdefault(text, lid)
    return list(dedup.items())


@torch.no_grad()
def _eval(model, loader, device):
    model.eval()
    preds, trues = [], []
    for ids, mask, labels in loader:
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        logits = model(ids, mask)
        preds.extend(logits.argmax(-1).cpu().tolist())
        trues.extend(labels.cpu().tolist())
    if not trues:
        return 0.0, 0.0, [], []
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="macro", zero_division=0)
    return float(acc), float(f1), trues, preds


def _make_loader(ids, labels, batch_size):
    ds = ReviewDataset(ids, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=0, collate_fn=collate_fn)


def retrain(args):
    device = pick_device(args.device)
    print(f"设备: {device}")

    if not os.path.exists(C.VOCAB_PATH):
        raise SystemExit(f"缺少词表 {C.VOCAB_PATH}，请先运行 python -m src.train")
    if not os.path.exists(C.MODEL_PATH):
        raise SystemExit(f"缺少模型 {C.MODEL_PATH}，请先运行 python -m src.train")

    vocab = Vocab.load(C.VOCAB_PATH)
    human = load_human_labeled(args.db)
    print(f"人工标注样本（去重后）: {len(human)} 条")
    if not human:
        raise SystemExit("样本库中没有 true_label，无增量数据可训练，退出。")

    # ---- 1. 原始语料（复用 mmap 缓存，limit=None 命中已有 1M 缓存，不重新编码）----
    orig_ids, orig_labels = build_or_load_cache(
        args.data, vocab, max_len=args.max_len, verbose=True)
    n_orig_total = len(orig_labels)
    if args.max_orig_samples and 0 < args.max_orig_samples < n_orig_total:
        rng = np.random.default_rng(C.SEED)
        orig_idx = rng.choice(n_orig_total, size=args.max_orig_samples, replace=False)
    else:
        orig_idx = np.arange(n_orig_total)
    n_orig = len(orig_idx)
    print(f"原始样本参与微调: {n_orig} / {n_orig_total}")

    # ---- 2. 编码人工样本 ----
    human_ids = np.asarray([vocab.encode(t, args.max_len) for t, _ in human], dtype=np.int32)
    human_labels = np.asarray([l for _, l in human], dtype=np.int64)

    # ---- 3. 划分人工样本：留一部分做"困难样本"验证，量化闭环收益 ----
    if len(human) >= args.min_val_human:
        rng = np.random.default_rng(C.SEED)
        perm = rng.permutation(len(human))
        n_val = max(1, int(len(human) * args.val_frac))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
    else:
        val_idx, train_idx = np.array([], dtype=np.int64), np.arange(len(human))
        print(f"[warn] 人工样本 < {args.min_val_human} 条，全部用于训练，不做困难样本验证")

    human_train_ids, human_train_labels = human_ids[train_idx], human_labels[train_idx]
    human_val_ids, human_val_labels = human_ids[val_idx], human_labels[val_idx]
    n_human_train = len(human_train_labels)
    print(f"人工样本划分: 训练 {n_human_train} / 验证 {len(human_val_labels)}")

    # ---- 4. 构建训练 loader（加权采样上采样困难样本）----
    train_ds = ConcatDataset([
        ReviewDataset(orig_ids, orig_labels, indices=orig_idx),
        ReviewDataset(human_train_ids, human_train_labels),
    ])
    if n_human_train > 0 and args.human_ratio > 0:
        # 期望 human 占比 = human_ratio，解 orig 权重 1、human 权重 w：
        #   human_ratio = n_human*w / (n_orig + n_human*w)  =>  w = human_ratio*n_orig/(n_human*(1-human_ratio))
        w = args.human_ratio * n_orig / (n_human_train * (1 - args.human_ratio))
        weights = [1.0] * n_orig + [w] * n_human_train
        sampler = WeightedRandomSampler(
            weights, num_samples=n_orig + n_human_train, replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=0, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, collate_fn=collate_fn, drop_last=True)

    # ---- 5. 验证集：困难样本 + 一小片原始样本（回归检查）----
    val_loaders = {}
    if len(human_val_ids):
        bsz = max(1, min(64, len(human_val_ids)))
        val_loaders["困难样本"] = _make_loader(human_val_ids, human_val_labels, bsz)
    rng = np.random.default_rng(C.SEED + 1)
    n_check = min(args.overall_val_size, n_orig_total)
    check_idx = rng.choice(n_orig_total, size=n_check, replace=False)
    val_loaders["整体回归"] = _make_loader(
        orig_ids[check_idx], orig_labels[check_idx], args.batch_size)

    # ---- 6. 加载模型 ----
    ckpt = torch.load(C.MODEL_PATH, map_location=device, weights_only=True)
    cfg = ckpt["model_config"]
    model = TransformerClassifier(**cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    # ---- 7. 微调前基线 ----
    print("\n== 微调前基线 ==")
    before = {}
    for name, loader in val_loaders.items():
        acc, f1, _, _ = _eval(model, loader, device)
        before[name] = {"acc": round(acc, 4), "macro_f1": round(f1, 4)}
        print(f"  {name}: acc={acc:.4f}  macroF1={f1:.4f}")

    # ---- 8. 微调 ----
    print(f"\n== 微调: epochs={args.epochs} lr={args.lr} batch={args.batch_size} "
          f"human_ratio={args.human_ratio} ==")
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = build_scheduler(optimizer, total_steps,
                                int(total_steps * C.WARMUP_RATIO))
    model.train()
    for epoch in range(args.epochs):
        running, seen = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for ids, mask, labels in pbar:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            logits = model(ids, mask)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)
            pbar.set_postfix(loss=f"{running/seen:.4f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    # ---- 9. 微调后评估 ----
    print("\n== 微调后 ==")
    after = {}
    for name, loader in val_loaders.items():
        acc, f1, _, _ = _eval(model, loader, device)
        after[name] = {"acc": round(acc, 4), "macro_f1": round(f1, 4)}
        d = acc - before[name]["acc"]
        print(f"  {name}: acc={acc:.4f}  macroF1={f1:.4f}  (Δacc={d:+.4f})")

    # ---- 10. 保存（覆盖前先备份旧模型；仅当写回默认路径时）----
    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.abspath(out) == os.path.abspath(C.MODEL_PATH):
        backup = C.MODEL_PATH + ".before_retrain"
        if not os.path.exists(backup):
            shutil.copy2(C.MODEL_PATH, backup)
            print(f"已备份旧模型 -> {backup}")
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": args.epochs - 1,
        "best_val_f1": after.get("整体回归", after.get("困难样本", {"macro_f1": 0.0}))["macro_f1"],
        "model_config": cfg,
        "retrain": {
            "n_human": len(human), "n_human_train": n_human_train,
            "n_orig": n_orig, "epochs": args.epochs, "lr": args.lr,
            "human_ratio": args.human_ratio,
        },
    }, out)
    print(f"已保存微调模型 -> {out}")

    # ---- 11. 报告 ----
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_human_total": len(human),
        "n_human_train": n_human_train,
        "n_human_val": len(human_val_labels),
        "n_orig": n_orig,
        "config": {"epochs": args.epochs, "lr": args.lr,
                   "human_ratio": args.human_ratio, "batch_size": args.batch_size},
        "before": before,
        "after": after,
    }
    report_path = os.path.join(C.OUTPUT_DIR, "retrain_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"已保存报告 -> {report_path}")
    return model


def parse_args():
    p = argparse.ArgumentParser(description="主动学习再训练（增量微调）")
    p.add_argument("--data", default=C.RAW_DATA)
    p.add_argument("--db", default=C.DB_PATH, help="样本库路径（读 true_label 非空样本）")
    p.add_argument("--out", default=C.MODEL_PATH, help="输出模型路径，默认覆盖 model.pt")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=C.LR * 0.1, help="微调学习率（默认 1/10）")
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--weight-decay", type=float, default=C.WEIGHT_DECAY)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--max-len", type=int, default=C.MAX_LEN)
    p.add_argument("--max-orig-samples", type=int, default=200_000,
                   help="原始样本参与微调的数量，<=0 表示全部")
    p.add_argument("--human-ratio", type=float, default=0.30,
                   help="每个 batch 中困难样本的期望占比")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="困难样本中留作验证的比例")
    p.add_argument("--min-val-human", type=int, default=10,
                   help="困难样本少于该值时全部用于训练")
    p.add_argument("--overall-val-size", type=int, default=2000,
                   help="用于整体回归检查的原始样本数")
    p.add_argument("--device", default="auto")
    return p.parse_args()


if __name__ == "__main__":
    retrain(parse_args())
