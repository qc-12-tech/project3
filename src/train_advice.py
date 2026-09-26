"""训练「购买建议」Transformer 模型（推荐购买 / 谨慎购买 / 不建议购买）。

训练数据 data/advice.jsonl 由 src/label_advice.py 从已有评价的「具体内容」派生标签
（非原始情感标签）；复用情感模型词表 vocab.json，使用独立编码缓存。

用法：
    python -m src.label_advice                          # 先生成 data/advice.jsonl
    python -m src.train_advice                          # 默认 3 epoch，全量数据
    python -m src.train_advice --max-samples 100000     # 子集快速验证
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm import tqdm

from src import config as C
from src.dataset import make_dataloaders
from src.model import TransformerClassifier
from src.train import build_scheduler, evaluate, pick_device
from src.vocab import Vocab


def train(args):
    device = pick_device(args.device)
    print(f"设备: {device}")

    # ---- 1. 词表：复用情感模型的词表 ----
    if not os.path.exists(C.VOCAB_PATH):
        raise SystemExit(f"缺少词表 {C.VOCAB_PATH}，请先运行 python -m src.train")
    vocab = Vocab.load(C.VOCAB_PATH)
    print(f"加载词表: {len(vocab)} 个字符")

    # ---- 2. 数据：独立编码缓存，避免覆盖情感缓存 ----
    train_loader, val_loader = make_dataloaders(
        args.data, vocab, max_len=args.max_len, batch_size=args.batch_size,
        max_samples=args.max_samples, verbose=True,
        cache_prefix=C.ADVICE_CACHE_PREFIX)
    print(f"训练 batch 数/epoch: {len(train_loader)} | 验证 batch 数: {len(val_loader)}")

    # ---- 3. 模型 ----
    model = TransformerClassifier(
        vocab_size=len(vocab), d_model=args.d_model, num_heads=args.num_heads,
        num_layers=args.num_layers, d_ff=args.d_ff, dropout=args.dropout,
        num_classes=len(C.ADVICE_LABELS), max_len=args.max_len, pad_id=vocab.pad_id,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params/1e6:.3f} M")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = build_scheduler(optimizer, total_steps, int(total_steps * C.WARMUP_RATIO))

    # ---- 4. 训练循环 ----
    best_f1 = -1.0
    for epoch in range(args.epochs):
        model.train()
        running, seen = 0.0, 0
        t0 = time.time()
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

        val_loss, acc, f1, trues, preds = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch+1}] train_loss={running/seen:.4f} "
              f"val_loss={val_loss:.4f} acc={acc:.4f} macroF1={f1:.4f} "
              f"耗时={time.time()-t0:.1f}s")
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_f1": best_f1,
                "model_config": {
                    "vocab_size": len(vocab), "d_model": args.d_model,
                    "num_heads": args.num_heads, "num_layers": args.num_layers,
                    "d_ff": args.d_ff, "dropout": args.dropout,
                    "num_classes": len(C.ADVICE_LABELS), "max_len": args.max_len,
                    "pad_id": vocab.pad_id,
                },
            }, C.ADVICE_MODEL_PATH)
            print(f"  已保存最佳模型 -> {C.ADVICE_MODEL_PATH} (macroF1={best_f1:.4f})")

    # ---- 5. 最终报告 ----
    _, acc, f1, trues, preds = evaluate(model, val_loader, device)
    print("\n最终验证集报告（购买建议）：")
    print(classification_report(trues, preds, target_names=C.ADVICE_LABELS, digits=4))
    with open(os.path.join(C.OUTPUT_DIR, "train_advice_report.json"), "w",
              encoding="utf-8") as f:
        json.dump({"accuracy": acc, "macro_f1": f1, "best_val_f1": best_f1,
                   "params": n_params}, f, ensure_ascii=False, indent=2)
    return model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=C.ADVICE_DATA)
    p.add_argument("--epochs", type=int, default=C.EPOCHS)
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=C.LR)
    p.add_argument("--weight-decay", type=float, default=C.WEIGHT_DECAY)
    p.add_argument("--label-smoothing", type=float, default=C.LABEL_SMOOTHING)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-len", type=int, default=C.MAX_LEN)
    p.add_argument("--d-model", type=int, default=C.D_MODEL)
    p.add_argument("--num-heads", type=int, default=C.NUM_HEADS)
    p.add_argument("--num-layers", type=int, default=C.NUM_LAYERS)
    p.add_argument("--d-ff", type=int, default=C.D_FF)
    p.add_argument("--dropout", type=float, default=C.DROPOUT)
    p.add_argument("--device", default="auto")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
