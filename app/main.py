"""FastAPI 服务：电影评价分析结果接口。

启动：uvicorn app.main:app --reload --port 8000
"""
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.service import aspects, movie_list, movies, predict_text, sample_reviews, spam   # noqa: E402

app = FastAPI(title="电影评价分析 API（decoder-only Transformer）", version="1.0",
              description="基于从零实现的 decoder-only 因果语言模型：好评/差评统计、电影打分、优缺点挖掘")


class ReviewIn(BaseModel):
    text: str


@app.get("/")
def root():
    return {"service": "电影评价分析（decoder-only GPT）",
            "endpoints": ["/movies", "/movies/{movie_id}", "/movies/{movie_id}/reviews",
                          "/movies/{movie_id}/aspects", "/movies/{movie_id}/spam", "/predict"]}


@app.get("/movies")
def list_movies():
    sp = spam()
    return [{"movie_id": m["movie_id"], "movie_cn": m["movie_cn"], "movie_en": m["movie_en"],
             "n_reviews": m["n_reviews"], "star_pos": m["star_pos"], "star_neg": m["star_neg"],
             "star_mid": m["star_mid"], "model_pos": m["model_pos"], "model_neg": m["model_neg"],
             "model_pos_rate": round(float(m["model_pos_rate"]), 4),
             "bayes_rating": round(float(m["bayes_rating"]), 3),
             "final_score": float(m["final_score"]),
             "suspect_rate": sp.get(int(m["movie_id"]), {}).get("suspect_rate"),
             "pros": m.get("pros", "").split("、") if m.get("pros") else [],
             "cons": m.get("cons", "").split("、") if m.get("cons") else []}
            for m in movie_list()]


@app.get("/movies/{movie_id}")
def movie_detail(movie_id: int):
    ms = movies()
    if movie_id not in ms:
        raise HTTPException(404, "movie not found")
    m = ms[movie_id]
    return {"movie_id": movie_id, "movie_cn": m["movie_cn"], "movie_en": m["movie_en"],
            "stats": {"n_reviews": m["n_reviews"], "star_pos": m["star_pos"], "star_neg": m["star_neg"],
                      "star_mid": m["star_mid"], "star_pos_rate": round(float(m["star_pos_rate"]), 4),
                      "model_pos": m["model_pos"], "model_neg": m["model_neg"],
                      "model_pos_rate": round(float(m["model_pos_rate"]), 4),
                      "star_mean": round(float(m["star_mean"]), 3),
                      "pred_rating_mean": round(float(m.get("pred_rating_mean", 0)), 3),
                      "likes_sum": m.get("likes_sum", 0)},
            "score": {"bayes_rating": round(float(m["bayes_rating"]), 3),
                      "rating_score": round(float(m["rating_score"]), 2),
                      "sentiment_score": round(float(m["sentiment_score"]), 2),
                      "final_score": float(m["final_score"]), "confidence": float(m["confidence"])},
            "pros": m["pros_detail"], "cons": m["cons_detail"], "weak": m.get("weak_detail", []),
            "keywords": {"pos": m["keywords_pos"], "neg": m["keywords_neg"]},
            "spam": spam().get(movie_id)}


@app.get("/movies/{movie_id}/aspects")
def movie_aspects(movie_id: int):
    a = aspects().get(str(movie_id))
    if not a:
        raise HTTPException(404, "no aspect data (先运行 python -m src.aspects)")
    return a


@app.get("/movies/{movie_id}/reviews")
def movie_reviews(movie_id: int, polarity: str = Query("pos", pattern="^(pos|neg)$"), limit: int = 20):
    return sample_reviews(movie_id, polarity, limit)


@app.get("/movies/{movie_id}/spam")
def movie_spam(movie_id: int):
    """该电影的刷评论检测结果：疑似比例、各信号命中数、最可疑样本。"""
    s = spam().get(movie_id)
    if not s:
        raise HTTPException(404, "no spam data (先运行 python -m src.spam)")
    return s


@app.post("/predict")
def predict(item: ReviewIn):
    if not item.text.strip():
        raise HTTPException(400, "text is empty")
    return predict_text(item.text.strip())
