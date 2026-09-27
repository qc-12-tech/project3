"""接口层测试：直接用 FastAPI TestClient 打所有端点（含 404 与单条预测）。"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import json
import os

ROOT = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("PROJECT_ART_DIR", ROOT / "artifacts"))
N_MOVIES = len(json.loads((ART / "movies.json").read_text(encoding="utf-8"))["movies"])

pytest.importorskip("httpx")
needs_artifacts = pytest.mark.skipif(not (ART / "movie_summary.json").exists(),
                                     reason="还没有跑完整流程")
pytestmark = needs_artifacts


@pytest.fixture(scope="module")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def top_movie(client):
    return client.get("/movies").json()[0]


def test_root_lists_endpoints(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "/predict" in r.json()["endpoints"]


def test_movies_list(client, top_movie):
    data = client.get("/movies").json()
    assert len(data) == N_MOVIES
    assert data[0]["final_score"] >= data[-1]["final_score"]      # 按综合分降序
    assert set(top_movie) >= {"movie_id", "final_score", "pros", "cons", "suspect_rate"}


def test_movie_detail(client, top_movie):
    d = client.get(f"/movies/{top_movie['movie_id']}").json()
    assert d["movie_cn"] == top_movie["movie_cn"]
    assert 0 <= d["score"]["final_score"] <= 10
    assert d["pros"], "详情里应有优点"
    assert d["cons"] or d["weak"]
    assert d["stats"]["n_reviews"] > 1000


def test_movie_aspects(client, top_movie):
    a = client.get(f"/movies/{top_movie['movie_id']}/aspects").json()
    assert len(a["aspects"]) > 5
    assert all(0 <= r["pos_rate"] <= 1 for r in a["aspects"])


def test_movie_reviews_by_polarity(client, top_movie):
    for pol in ("pos", "neg"):
        rs = client.get(f"/movies/{top_movie['movie_id']}/reviews",
                        params={"polarity": pol, "limit": 5}).json()
        assert len(rs) == 5
        if pol == "pos":
            assert min(r["p_pos"] for r in rs) >= 0.5
        else:
            assert max(r["p_pos"] for r in rs) < 0.5
    assert client.get(f"/movies/{top_movie['movie_id']}/reviews",
                      params={"polarity": "bad"}).status_code == 422


def test_movie_spam(client, top_movie):
    s = client.get(f"/movies/{top_movie['movie_id']}/spam").json()
    assert 0 <= s["suspect_rate"] <= 1
    assert s["by_type"] and isinstance(s["top_suspects"], list)


def test_unknown_movie_returns_404(client):
    assert client.get("/movies/99999").status_code == 404


def test_predict_positive_vs_negative(client):
    """端到端：明显正/负面的影评，模型应当给出方向正确的判定。"""
    pos = client.post("/predict", json={"text": "剧情精彩，演员表演非常出色，结局也很感人，强烈推荐！"}).json()
    neg = client.post("/predict", json={"text": "剧情拖沓无聊，演技尴尬，完全浪费时间，烂片一部。"}).json()
    assert pos["p_pos"] > neg["p_pos"], "正面评论的 P(好) 应高于负面评论"
    assert pos["sentiment"] == "好评" and neg["sentiment"] == "差评"
    assert 1 <= pos["pred_rating"] <= 5 and 1 <= neg["pred_rating"] <= 5
    assert client.post("/predict", json={"text": "   "}).status_code == 400
