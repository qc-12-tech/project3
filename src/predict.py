"""推理：三分类预测、情感打分、基于 MC Dropout 的“判断错误概率”。

核心输出（analyze）：
    label        预测标签（好评/中评/差评）
    probs        三类概率
    score        情感分 0~1（好评1 / 中评0.5 / 差评0），越低越严重
    error_prob   模型判断可能错误的概率 = 1 - 最大类概率（MC Dropout 平均）
    severity     严重程度：严重 / 警告 / 正常
    need_process 是否需立即处理（score < SCORE_CRITICAL）
    need_review  是否需人工复核（error_prob >= ERROR_ALERT）
"""
import argparse
import json
import os
import threading

import numpy as np
import torch
import torch.nn.functional as F

from src import config as C
from src.model import TransformerClassifier
from src.vocab import Vocab

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")


def _load_temperature():
    path = os.path.join(C.OUTPUT_DIR, "calibration.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return float(json.load(f).get("temperature", 1.0))
    return 1.0


class ReviewPredictor:
    def __init__(self, max_len=C.MAX_LEN, mc_samples=C.MC_SAMPLES, device=DEVICE):
        self.device = torch.device(device)
        self.max_len = max_len
        self.mc_samples = mc_samples
        self.temperature = _load_temperature()
        # 保护 MC Dropout 期间对共享模型 train()/eval() 的切换，避免并发请求互相污染
        self._mc_lock = threading.Lock()

        if not os.path.exists(C.MODEL_PATH):
            raise FileNotFoundError(
                f"未找到模型 {C.MODEL_PATH}，请先运行 python -m src.train")
        ckpt = torch.load(C.MODEL_PATH, map_location=self.device, weights_only=True)
        cfg = ckpt["model_config"]
        self.max_len = cfg.get("max_len", max_len)
        self.vocab = Vocab.load(C.VOCAB_PATH)
        self.model = TransformerClassifier(**cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    # ---------------- 基础前向 ----------------
    def _encode(self, texts):
        ids = [self.vocab.encode(t, self.max_len) for t in texts]
        ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        mask = ids != self.vocab.pad_id
        return ids, mask

    @torch.no_grad()
    def predict_probs(self, texts, mc_samples=None):
        """返回平均后的类别概率 [B, C]（MC Dropout + 温度缩放）。"""
        if isinstance(texts, str):
            texts = [texts]
        mc = mc_samples or self.mc_samples
        ids, mask = self._encode(texts)

        # MC Dropout：需要临时把模型切到 train 模式。共享模型在并发请求下会互相
        # 干扰，因此用锁串行化这段状态切换 + 采样前向，finally 确保恢复 eval。
        with self._mc_lock:
            self.model.train()  # 打开 dropout，用于 MC 采样
            try:
                probs_sum = torch.zeros(len(texts), C.NUM_CLASSES, device=self.device)
                for _ in range(max(1, mc)):
                    logits = self.model(ids, mask)
                    probs_sum += F.softmax(logits / self.temperature, dim=-1)
            finally:
                self.model.eval()
        return (probs_sum / max(1, mc)).cpu().numpy()

    def _probs_to_result(self, probs):
        pred = int(np.argmax(probs))
        error_prob = float(1.0 - probs[pred])
        score = float(np.dot(probs, np.asarray(C.LABEL_SCORE)))
        if score < C.SCORE_CRITICAL:
            severity = "严重"
        elif score < C.SCORE_WARNING:
            severity = "警告"
        else:
            severity = "正常"
        return {
            "label": C.ID2LABEL[pred],
            "label_id": pred,
            "probs": {C.LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)},
            "score": round(score, 4),
            "error_prob": round(error_prob, 4),
            "severity": severity,
            "need_process": score < C.SCORE_CRITICAL,
            "need_review": error_prob >= C.ERROR_ALERT,
        }

    def analyze(self, text):
        return self._probs_to_result(self.predict_probs([text])[0])

    def analyze_many(self, texts, mc_samples=None):
        probs = self.predict_probs(texts, mc_samples=mc_samples)
        return [self._probs_to_result(p) for p in probs]

    # ---------------- 温度校准 ----------------
    def calibrate(self, texts, labels, max_iter=200, lr=0.01):
        """在带标签的数据上学习温度 T，使 NLL 最小（改善错误概率的可靠性）。"""
        ids, mask = self._encode(texts)
        labels = torch.tensor(labels, dtype=torch.long, device=self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(ids, mask)
        log_T = torch.zeros(1, device=self.device, requires_grad=True)
        opt = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(logits / log_T.exp(), labels)
            loss.backward()
            return loss

        opt.step(closure)
        T = float(log_T.exp().item())
        self.temperature = T
        with open(os.path.join(C.OUTPUT_DIR, "calibration.json"), "w", encoding="utf-8") as f:
            json.dump({"temperature": T}, f)
        return T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, default=None, help="单条评价")
    p.add_argument("--calibrate", action="store_true", help="在验证集上校准温度")
    p.add_argument("--max-samples", type=int, default=20000)
    args = p.parse_args()

    predictor = ReviewPredictor()

    if args.calibrate:
        from src.generate_data import read_jsonl
        texts, labels = read_jsonl(C.RAW_DATA, limit=args.max_samples)
        T = predictor.calibrate(texts, labels)
        print(f"温度校准完成：T={T:.4f}")
        return

    demo = [
        "味道特别好，配送也快，下次还会再点！",
        "一般般吧，没什么惊喜。",
        "等了两个小时才送到，饭都凉了，差评！",
    ]
    for t in ([args.text] if args.text else demo):
        print(t)
        print(" ", predictor.analyze(t))


if __name__ == "__main__":
    main()
