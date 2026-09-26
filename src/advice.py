"""购买建议推理：加载 advice_model，输出 推荐购买/谨慎购买/不建议购买 + 置信度 + 理由。

输出字段：
    advice        建议标签（推荐购买/谨慎购买/不建议购买）
    advice_id     标签 id
    probs         三类概率
    confidence    置信度 = 最大类概率
    rating        1~5 星（由建议概率折合）
    reasons       从评价中提取的关注维度（口味/配送/包装/...）
    summary       一句自然语言购买建议
"""
import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from src import config as C
from src.keywords import ASPECT_WORDS, tokenize
from src.model import TransformerClassifier
from src.vocab import Vocab

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")

SUMMARY = {
    "推荐购买": "整体反馈正面，可以放心下单。",
    "谨慎购买": "评价褒贬不一，建议结合自身需求再决定。",
    "不建议购买": "负面反馈较多，建议谨慎考虑或更换商家。",
}


def extract_aspects(text, top_k=3):
    """从评价文本提取被提及的维度（用于展示“关注点/理由”）。"""
    counter = Counter()
    for w in tokenize(text):
        for aspect, keys in ASPECT_WORDS.items():
            if w in keys or any(len(k) >= 2 and k in w for k in keys):
                counter[aspect] += 1
                break
    return [a for a, _ in counter.most_common(top_k)]


class AdvicePredictor:
    def __init__(self, max_len=C.MAX_LEN, device=DEVICE):
        self.device = torch.device(device)
        self.max_len = max_len
        if not os.path.exists(C.ADVICE_MODEL_PATH):
            raise FileNotFoundError(
                f"未找到购买建议模型 {C.ADVICE_MODEL_PATH}，请先运行 python -m src.train_advice")
        ckpt = torch.load(C.ADVICE_MODEL_PATH, map_location=self.device, weights_only=True)
        cfg = ckpt["model_config"]
        self.max_len = cfg.get("max_len", max_len)
        self.vocab = Vocab.load(C.VOCAB_PATH)
        self.model = TransformerClassifier(**cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def _encode(self, texts):
        ids = [self.vocab.encode(t, self.max_len) for t in texts]
        ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        mask = ids != self.vocab.pad_id
        return ids, mask

    @torch.no_grad()
    def predict_probs(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        ids, mask = self._encode(texts)
        logits = self.model(ids, mask)
        return F.softmax(logits, dim=-1).cpu().numpy()

    def advise(self, text):
        probs = self.predict_probs([text])[0]
        pred = int(np.argmax(probs))
        advice = C.ADVICE_ID2LABEL[pred]
        confidence = float(probs[pred])
        rating = round(1.0 + 4.0 * float(np.dot(probs, np.asarray(C.ADVICE_SCORE))), 1)
        return {
            "advice": advice,
            "advice_id": pred,
            "probs": {C.ADVICE_LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)},
            "confidence": round(confidence, 4),
            "rating": rating,
            "reasons": extract_aspects(text),
            "summary": SUMMARY[advice],
        }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, default=None)
    args = p.parse_args()

    predictor = AdvicePredictor()
    demo = [
        "味道很好，配送也快，分量很足，值得回购！",
        "口味不错，但是配送太慢了，包装也洒了一点，谨慎下单。",
        "吃出头发了，食材也不新鲜，还有股怪味，劝退！",
    ]
    for t in ([args.text] if args.text else demo):
        r = predictor.advise(t)
        print(f"\n评价：{t}")
        print(f"  建议：{r['advice']}（置信度 {r['confidence']:.2f}，{r['rating']} 星）")
        print(f"  关注点：{r['reasons']}")
        print(f"  小结：{r['summary']}")


if __name__ == "__main__":
    main()
