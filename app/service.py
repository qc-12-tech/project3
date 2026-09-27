"""共享服务层：加载模型与聚合产物，供 FastAPI / Streamlit 复用。"""
import json
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ART_DIR, PRED_DIR                      # noqa: E402
from src.infer import load_model, score_texts                 # noqa: E402


@lru_cache(maxsize=1)
def _model():
    return load_model()


@lru_cache(maxsize=1)
def movies() -> dict:
    """movie_id -> 汇总信息（含优点/缺点）"""
    summary = json.loads((ART_DIR / "movie_summary.json").read_text(encoding="utf-8"))
    pc = {int(x["movie_id"]): x for x in json.loads((ART_DIR / "movie_pros_cons.json").read_text(encoding="utf-8"))}
    out = {}
    for m in summary["movies"]:
        mid = int(m["movie_id"])
        out[mid] = {**m, "pros_detail": pc.get(mid, {}).get("pros", []),
                    "cons_detail": pc.get(mid, {}).get("cons", []),
                    "weak_detail": pc.get(mid, {}).get("weak", []),
                    "keywords_pos": pc.get(mid, {}).get("keywords_pos", []),
                    "keywords_neg": pc.get(mid, {}).get("keywords_neg", [])}
    return out


@lru_cache(maxsize=1)
def aspects() -> dict:
    p = ART_DIR / "aspects.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@lru_cache(maxsize=1)
def spam() -> dict:
    p = ART_DIR / "spam_summary.json"
    if not p.exists():
        return {}
    return {int(x["movie_id"]): x for x in json.loads(p.read_text(encoding="utf-8"))}


@lru_cache(maxsize=32)
def sample_reviews(movie_id: int, polarity: str = "pos", limit: int = 20) -> list:
    """从预测分片里取该电影的高置信样本影评。"""
    frames = []
    for f in sorted(PRED_DIR.glob("preds_*.parquet")):
        try:
            frames.append(pd.read_parquet(f, columns=["row_id", "movie_id", "p_pos", "pred_sent", "exp_rating"],
                                          filters=[("movie_id", "==", movie_id)]))
        except Exception:
            continue
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    df = df[df["pred_sent"] == (1 if polarity == "pos" else 0)]
    if polarity == "pos":
        df = df.sort_values("p_pos", ascending=False)
    else:
        df = df.sort_values("p_pos", ascending=True)
    df = df.head(limit * 4)
    rev = pd.read_parquet(ART_DIR / "reviews.parquet", columns=["row_id", "comment", "likes", "star"],
                          filters=[("movie_id", "==", movie_id)])
    out = df.merge(rev, on="row_id", how="inner").head(limit)
    return [{"comment": r.comment, "star": int(r.star), "likes": int(r.likes),
             "p_pos": round(float(r.p_pos), 3), "pred_rating": round(float(r.exp_rating), 2)}
            for r in out.itertuples()]


def predict_text(text: str) -> dict:
    """单条文本 -> decoder-only 模型两步解码结果。"""
    model, vocab, cfg, dev = _model()
    r = score_texts(model, vocab, cfg, [text], device=dev, batch_size=1)
    p = float(r["p_pos"][0])
    thr = float(getattr(model, "sent_threshold", 0.5))
    return {"text": text, "sentiment": "好评" if p >= thr else "差评", "p_pos": round(p, 4),
            "threshold": round(thr, 4),
            "pred_rating": round(float(r["exp_rating"][0]), 2),
            "rating_argmax": int(r["arg_rating"][0]), "confidence": round(float(r["p_rating"][0]), 4)}


def movie_list() -> list:
    ms = movies()
    return sorted(ms.values(), key=lambda m: -float(m["final_score"]))
