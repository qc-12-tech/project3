"""产物一致性验收测试：对全量跑出来的 artifacts 做交叉校验。

没有产物时整组跳过（例如只跑单元测试的场景）。
"""
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("PROJECT_ART_DIR", ROOT / "artifacts"))
N_MOVIES = len(json.loads((ART / "movies.json").read_text(encoding="utf-8"))["movies"]) \
    if (ART / "movies.json").exists() else 0
needs_artifacts = pytest.mark.skipif(not (ART / "movie_summary.csv").exists(),
                                     reason="还没有跑完整流程（缺 artifacts/movie_summary.csv）")
pytestmark = needs_artifacts


@pytest.fixture(scope="module")
def summary():
    return pd.read_csv(ART / "movie_summary.csv")


def test_movie_count_and_score_range(summary):
    assert len(summary) == N_MOVIES
    assert summary["final_score"].between(0, 10).all()
    assert summary["bayes_rating"].between(1, 5).all()
    assert summary["confidence"].between(0, 1).all()


def test_counts_are_internally_consistent(summary):
    assert (summary["star_pos"] + summary["star_neg"] + summary["star_mid"] == summary["n_reviews"]).all()
    assert (summary["model_pos"] + summary["model_neg"] == summary["n_reviews"]).all()
    assert (summary["n_suspect"] <= summary["n_reviews"]).all()
    assert summary["n_reviews"].min() > 0


def test_prediction_shards_cover_all_reviews_without_duplicates(summary):
    files = sorted((ART / "preds").glob("preds_*.parquet"))
    assert files, "没有预测分片"
    ids = np.concatenate([pd.read_parquet(f, columns=["row_id"])["row_id"].to_numpy() for f in files])
    assert len(ids) == int(summary["n_reviews"].sum()), "分片行数之和 != 汇总表影评总数"
    assert len(np.unique(ids)) == len(ids), "存在重复 row_id（分段续跑产生了重叠分片）"
    assert ids.min() == 0 and ids.max() == len(ids) - 1


def test_prediction_values_are_valid():
    for f in sorted((ART / "preds").glob("preds_*.parquet"))[:3]:
        df = pd.read_parquet(f, columns=["p_pos", "pred_sent", "exp_rating"])
        assert df["pred_sent"].isin([0, 1]).all()
        assert df["p_pos"].between(0, 1).all()
        assert df["exp_rating"].between(1, 5).all()


def test_aspects_are_well_formed():
    a = json.loads((ART / "aspects.json").read_text(encoding="utf-8"))
    assert len(a) == N_MOVIES
    for m in a.values():
        assert m["n_sampled"] > 0
        for r in m["aspects"]:
            assert 0 <= r["pos_rate"] <= 1 and r["mentions"] > 0
            assert isinstance(r["evidence_pos"], list) and isinstance(r["evidence_neg"], list)


def test_every_movie_has_actionable_pros_and_cons_or_weak():
    pc = json.loads((ART / "movie_pros_cons.json").read_text(encoding="utf-8"))
    assert len(pc) == N_MOVIES
    for x in pc:
        assert x["pros"], f"{x['movie_cn']} 没有优点"
        assert x["cons"] or x.get("weak"), f"{x['movie_cn']} 既无缺点也无相对短板"
        for e in x["pros"] + x["cons"] + x.get("weak", []):
            assert all(isinstance(s, str) and s for s in e["evidence"]) or e["evidence"] == []


def test_spam_summary_and_detail_ranges():
    sp = json.loads((ART / "spam_summary.json").read_text(encoding="utf-8"))
    assert len(sp) == N_MOVIES
    for x in sp:
        assert 0 <= x["suspect_rate"] <= 1
        assert x["n_suspect"] <= x["n_reviews"]
        assert set(x["by_type"]) >= {"dup_same_movie", "ad", "rating_mismatch"}
        for t in x["top_suspects"]:
            assert t["reasons"], "可疑样本必须带上命中的信号名"
    n = pq.ParquetFile(ART / "spam_reviews.parquet").metadata.num_rows
    assert n == sum(x["n_reviews"] for x in sp), "spam_reviews 行数 != 影评总数"


def test_train_log_reports_calibrated_metrics():
    log = json.loads((ART / "train_log.json").read_text(encoding="utf-8"))
    ev = log.get("final_eval")
    assert ev, "缺少 final_eval（请跑 python -m src.train --eval-only）"
    assert ev["sent_acc"] > 0.80 and ev["sent_f1_pos"] > 0.85 and ev["rating_mae"] < 0.8
    cal = log["calibration"]
    assert cal["acc"] > 0.80 and 0.05 < cal["threshold"] < 0.95


def test_reviews_parquet_row_count_matches_summary():
    n = pq.ParquetFile(ART / "reviews.parquet").metadata.num_rows
    s = pd.read_csv(ART / "movie_summary.csv")["n_reviews"].sum()
    assert n == int(s)
