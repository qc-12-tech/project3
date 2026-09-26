"""FastAPI 服务：外卖评价识别与风险预警。

启动：
    uvicorn app.main:app --reload --port 8000
文档：
    http://127.0.0.1:8000/docs
"""
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config as C
from src import keywords as kw
from src.duplicate import check_duplicates
from src.predict import ReviewPredictor
from src.store import ReviewStore

STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["predictor"] = ReviewPredictor()
    STATE["store"] = ReviewStore()
    yield
    STATE.clear()


app = FastAPI(title="外卖评价识别系统", version="1.0.0", lifespan=lifespan)


# ---------------- Schemas ----------------
class PredictRequest(BaseModel):
    text: str = Field(..., description="一条评价文本", examples=["等了两个小时，饭都凉了"])
    save: bool = Field(True, description="是否存入样本库")


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., description="多条评价")
    save: bool = Field(False, description="是否批量入库")


class LabelRequest(BaseModel):
    true_label: str = Field(..., description="人工复核后的真实标签", examples=["差评"])


class DuplicateRequest(BaseModel):
    texts: List[str] = Field(..., description="同一商品的多条评价文本")


# ---------------- 元信息 ----------------
@app.get("/", tags=["元信息"])
def root():
    return {
        "name": "外卖评价识别系统",
        "labels": C.LABELS,
        "thresholds": {
            "score_critical": C.SCORE_CRITICAL,
            "score_warning": C.SCORE_WARNING,
            "error_alert": C.ERROR_ALERT,
        },
        "endpoints": ["/predict", "/predict/batch", "/negative/keywords",
                      "/reviews/pending", "/reviews/{id}/label", "/stats",
                      "/duplicate/check"],
    }


@app.get("/health", tags=["元信息"])
def health():
    return {"status": "ok", "model": os.path.basename(C.MODEL_PATH)}


# ---------------- 预测 ----------------
@app.post("/predict", tags=["预测"])
def predict(req: PredictRequest):
    predictor: ReviewPredictor = STATE["predictor"]
    result = predictor.analyze(req.text)
    result["text"] = req.text
    if req.save:
        result["review_id"] = STATE["store"].add(req.text, result, source="api")
    return result


@app.post("/predict/batch", tags=["预测"])
def predict_batch(req: BatchRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts 不能为空")
    predictor: ReviewPredictor = STATE["predictor"]
    results = predictor.analyze_many(req.texts)
    if req.save:
        STATE["store"].add_many(
            [(t, r, "batch") for t, r in zip(req.texts, results)])
    results_with_text = [dict(text=t, **r) for t, r in zip(req.texts, results)]
    alerts = [r for r in results_with_text if r["need_process"] or r["need_review"]]
    summary = {
        "total": len(results),
        "good": sum(r["label"] == "好评" for r in results),
        "neutral": sum(r["label"] == "中评" for r in results),
        "bad": sum(r["label"] == "差评" for r in results),
        "need_process": sum(r["need_process"] for r in results),
        "need_review": sum(r["need_review"] for r in results),
    }
    return {"summary": summary,
            "alerts": alerts,
            "results": results_with_text}


# ---------------- 查重 / 刷评论检测 ----------------
@app.post("/duplicate/check", tags=["查重检测"])
def duplicate_check(req: DuplicateRequest):
    if len(req.texts) < 2:
        raise HTTPException(status_code=400, detail="至少需要 2 条评价")
    return check_duplicates(req.texts)


# ---------------- 差评高频词 ----------------
@app.get("/negative/keywords", tags=["差评分析"])
def negative_keywords(top_k: int = C.KEYWORD_TOP_K):
    report = kw.load_report()
    if report is None:
        report = kw.build_from_dataset()
    report = dict(report)
    report["top_keywords"] = report["top_keywords"][:top_k]
    return report


@app.post("/negative/analyze", tags=["差评分析"])
def negative_analyze(req: BatchRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts 不能为空")
    return kw.analyze_negatives(req.texts)


# ---------------- 样本 / 复核 ----------------
@app.get("/reviews", tags=["样本管理"])
def list_reviews(limit: int = 50, offset: int = 0,
                 pred_label: Optional[str] = None, need_review: Optional[bool] = None,
                 need_process: Optional[bool] = None, processed: Optional[bool] = None):
    rows = STATE["store"].list(limit=limit, offset=offset, pred_label=pred_label,
                               need_review=need_review, need_process=need_process,
                               processed=processed)
    return {"count": len(rows), "items": rows}


@app.get("/reviews/pending", tags=["样本管理"])
def pending_reviews(limit: int = 50):
    rows = STATE["store"].list(limit=limit, need_review=True, processed=False)
    return {"count": len(rows), "items": rows}


@app.post("/reviews/{review_id}/label", tags=["样本管理"])
def label_review(review_id: int, req: LabelRequest):
    if req.true_label not in C.LABELS:
        raise HTTPException(status_code=400, detail=f"true_label 必须是 {C.LABELS}")
    row = STATE["store"].update_true_label(review_id, req.true_label)
    if row is None:
        raise HTTPException(status_code=404, detail="样本不存在")
    return row


@app.post("/reviews/{review_id}/process", tags=["样本管理"])
def process_review(review_id: int):
    row = STATE["store"].mark_processed(review_id)
    if row is None:
        raise HTTPException(status_code=404, detail="样本不存在")
    return row


@app.get("/stats", tags=["样本管理"])
def stats():
    return STATE["store"].stats()
