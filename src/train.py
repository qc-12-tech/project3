"""训练 Transformer Encoder 外卖评价分类器。

用法：
    python -m src.train                          # 默认读 data/reviews.jsonl
    python -m src.train --epochs 3 --batch-size 256
    python -m src.train --max-samples 100000     # 小数据快速验证
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

from src import config as C
from src.dataset import make_dataloaders
from src.model import TransformerClassifier
from src.vocab import Vocab


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_scheduler(optimizer, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(min(1.0, progress) * math.pi))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues, losses = [], [], []
    criterion = nn.CrossEntropyLoss()
    for ids, mask, labels in loader:
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        logits = model(ids, mask)
        losses.append(criterion(logits, labels).item())
        preds.extend(logits.argmax(-1).cpu().tolist())
        trues.extend(labels.cpu().tolist())
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="macro")
    return np.mean(losses), acc, f1, trues, preds


def train(args):
    device = pick_device(args.device)
    print(f"设备: {device}")

    # ---- 1. 词表 ----
    if os.path.exists(C.VOCAB_PATH):
        vocab = Vocab.load(C.VOCAB_PATH)
        print(f"加载词表: {len(vocab)} 个字符")
    else:
        from src.generate_data import read_jsonl
        print("构建词表 ...")
        texts, _ = read_jsonl(args.data, limit=200_000)
        vocab = Vocab.build(texts)
        vocab.save(C.VOCAB_PATH)
        print(f"词表已保存: {C.VOCAB_PATH} ({len(vocab)} 字符)")

    # ---- 2. 数据 ----
    train_loader, val_loader = make_dataloaders(
        args.data, vocab, max_len=args.max_len, batch_size=args.batch_size,
        max_samples=args.max_samples, verbose=True)
    print(f"训练 batch 数/epoch: {len(train_loader)} | 验证 batch 数: {len(val_loader)}")

    # ---- 3. 模型 ----
    model = TransformerClassifier(
        vocab_size=len(vocab), d_model=args.d_model, num_heads=args.num_heads,
        num_layers=args.num_layers, d_ff=args.d_ff, dropout=args.dropout,
        num_classes=C.NUM_CLASSES, max_len=args.max_len, pad_id=vocab.pad_id,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params/1e6:.3f} M")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * C.WARMUP_RATIO)
    scheduler = build_scheduler(optimizer, total_steps, warmup_steps)

    start_epoch, best_f1 = 0, -1.0
    if args.resume and os.path.exists(C.MODEL_PATH):
        ckpt = torch.load(C.MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_val_f1", -1.0)
        print(f"从 checkpoint 恢复：epoch {start_epoch}, best_f1={best_f1:.4f}")

    # ---- 4. 训练循环 ----
    for epoch in range(start_epoch, args.epochs):
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
                    "num_classes": C.NUM_CLASSES, "max_len": args.max_len,
                    "pad_id": vocab.pad_id,
                },
            }, C.MODEL_PATH)
            print(f"  已保存最佳模型 -> {C.MODEL_PATH} (macroF1={best_f1:.4f})")

    # ---- 5. 最终报告 ----
    _, acc, f1, trues, preds = evaluate(model, val_loader, device)
    print("\n最终验证集报告：")
    print(classification_report(trues, preds, target_names=C.LABELS, digits=4))
    with open(os.path.join(C.OUTPUT_DIR, "train_report.json"), "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "macro_f1": f1, "best_val_f1": best_f1,
                   "params": n_params}, f, ensure_ascii=False, indent=2)
    return model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=C.RAW_DATA)
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
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
