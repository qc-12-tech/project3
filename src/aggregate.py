"""聚合与打分：
  1) 每部电影的 好评/差评/中评 统计（模型判定 + 真实星级两套口径）
  2) 贝叶斯加权评分（解决小样本电影虚高）→ 0~10 综合分
  3) 每部电影的 优点/缺点（来自 aspects.json 的方面词极性 + log-odds 关键词）
输出：artifacts/movie_summary.csv / movie_summary.json / movie_pros_cons.json / 分析报告.md
"""
import argparse
import json

import numpy as np
import pandas as pd

from .config import ART_DIR, PRED_DIR, STAR_POS_MIN, STAR_NEG_MAX, STAR_MID, AggregateConfig


def bayesian_rating(sum_star, n, prior_m, prior_mean):
    return (sum_star + prior_m * prior_mean) / (n + prior_m)


def _py(o):
    """numpy 标量 -> Python 原生类型（JSON 序列化用）。"""
    return o.item() if hasattr(o, "item") else str(o)


def load_preds(limit=None):
    files = sorted(PRED_DIR.glob("preds_*.parquet"))
    if not files:
        raise SystemExit("没有找到预测文件，请先运行 python -m src.infer")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.head(limit) if limit else df
    # float16/int8 -> 安全类型，避免 groupby 求和溢出与精度损失
    df["p_pos"] = df["p_pos"].astype("float32")
    df["exp_rating"] = df["exp_rating"].astype("float32")
    df["p_rating"] = df["p_rating"].astype("float32")
    df["pred_sent"] = df["pred_sent"].astype("int64")
    df["star"] = df["star"].astype("int64")
    df["likes"] = df["likes"].astype("int64")
    return df


def aggregate(acfg: AggregateConfig = None, verbose: bool = True):
    acfg = acfg or AggregateConfig()
    movies = {m["id"]: m for m in json.loads((ART_DIR / "movies.json").read_text(encoding="utf-8"))["movies"]}
    preds = load_preds()
    n = len(preds)
    prior_mean = float(preds["star"].mean())

    g = preds.groupby("movie_id")
    stat = pd.DataFrame({
        "n_reviews": g.size(),
        "model_pos": g["pred_sent"].sum(),                                   # 模型判好评数
        "model_neg": g["pred_sent"].apply(lambda s: int((s == 0).sum())),     # 模型判差评数
        "model_pos_rate": g["pred_sent"].mean(),
        "p_pos_mean": g["p_pos"].mean(),
        "pred_rating_mean": g["exp_rating"].mean(),
        "star_mean": g["star"].mean(),
        "star_sum": g["star"].sum(),
        "likes_sum": g["likes"].sum(),
        "likes_mean": g["likes"].mean(),
    })
    stat["star_pos"] = g["star"].apply(lambda s: int((s >= STAR_POS_MIN).sum()))
    stat["star_neg"] = g["star"].apply(lambda s: int((s <= STAR_NEG_MAX).sum()))
    stat["star_mid"] = g["star"].apply(lambda s: int((s == STAR_MID).sum()))
    stat["star_pos_rate"] = stat["star_pos"] / stat["n_reviews"]
    stat["bayes_rating"] = bayesian_rating(stat["star_sum"], stat["n_reviews"], acfg.prior_m, prior_mean)

    # ---- 综合评分（0~10）
    stat["rating_score"] = (stat["bayes_rating"] - 1) / 4 * 10              # 贝叶斯星级 -> 10 分制
    stat["sentiment_score"] = stat["model_pos_rate"] * 10                   # 模型好评率 -> 10 分制
    stat["final_score"] = (0.6 * stat["rating_score"] + 0.4 * stat["sentiment_score"]).round(2)
    stat["confidence"] = (stat["n_reviews"] / (stat["n_reviews"] + acfg.prior_m)).round(3)

    # ---- 优缺点
    aspects = json.loads((ART_DIR / "aspects.json").read_text(encoding="utf-8"))
    pros_cons = {}
    for mid, info in aspects.items():
        rows = info["aspects"]
        qualified = [r for r in rows if r["mentions"] >= acfg.aspect_min_mentions]
        pros = [r for r in qualified if r["pos_rate"] >= acfg.aspect_pos_threshold]
        cons = [r for r in qualified if r["pos_rate"] <= acfg.aspect_neg_threshold]
        pros.sort(key=lambda r: -(r["mentions"] * (r["pos_rate"] - acfg.aspect_pos_threshold)))
        cons.sort(key=lambda r: -(r["mentions"] * (acfg.aspect_neg_threshold - r["pos_rate"])))
        # 没有「显著缺点」的电影（好评率都高于阈值）也给出「相对短板」，保证每部片都有可改进点
        weak = [] if cons else sorted(qualified, key=lambda r: r["pos_rate"])[:2]
        kw = info.get("keywords", {"pos": [], "neg": []})
        pros_cons[int(mid)] = {
            "movie_id": int(mid), "movie_cn": info["movie_cn"], "movie_en": info["movie_en"],
            "n_sampled": info["n_sampled"],
            "pros": [{"aspect": r["aspect"], "mentions": r["mentions"], "pos_rate": r["pos_rate"],
                      "evidence": [e["clause"] for e in r["evidence_pos"]][:2]} for r in pros[:acfg.top_aspects]],
            "cons": [{"aspect": r["aspect"], "mentions": r["mentions"], "pos_rate": r["pos_rate"],
                      "evidence": [e["clause"] for e in r["evidence_neg"]][:2]} for r in cons[:acfg.top_aspects]],
            "weak": [{"aspect": r["aspect"], "mentions": r["mentions"], "pos_rate": r["pos_rate"],
                      "evidence": [e["clause"] for e in r["evidence_neg"]][:2]} for r in weak],
            "keywords_pos": [k["word"] for k in kw.get("pos", [])][:10],
            "keywords_neg": [k["word"] for k in kw.get("neg", [])][:10],
        }

    # ---- 汇总表
    out = stat.reset_index()
    out["movie_cn"] = out["movie_id"].map(lambda i: movies[int(i)]["cn"])
    out["movie_en"] = out["movie_id"].map(lambda i: movies[int(i)]["en"])
    out["pros"] = out["movie_id"].map(lambda i: "、".join(p["aspect"] for p in pros_cons.get(int(i), {}).get("pros", [])))
    out["cons"] = out["movie_id"].map(lambda i: "、".join(p["aspect"] for p in pros_cons.get(int(i), {}).get("cons", [])))
    out = out.sort_values("final_score", ascending=False).reset_index(drop=True)
    cols = ["movie_id", "movie_cn", "movie_en", "n_reviews", "star_pos", "star_neg", "star_mid",
            "star_pos_rate", "model_pos", "model_neg", "model_pos_rate", "pred_rating_mean",
            "star_mean", "bayes_rating",
            "rating_score", "sentiment_score", "final_score", "confidence", "likes_sum", "likes_mean",
            "pros", "cons"]
    spam_csv = ART_DIR / "spam_summary.csv"
    if spam_csv.exists():                                  # 刷评论检测结果（可选，先跑 src.spam）
        sp = pd.read_csv(spam_csv)
        out = out.merge(sp[["movie_id", "n_suspect", "suspect_rate"]], on="movie_id", how="left")
        out["n_suspect"] = out["n_suspect"].fillna(0).astype(int)
        out["suspect_rate"] = out["suspect_rate"].fillna(0.0)
        cols += ["n_suspect", "suspect_rate"]
    out = out[cols]
    out.to_csv(ART_DIR / "movie_summary.csv", index=False, encoding="utf-8-sig")
    (ART_DIR / "movie_pros_cons.json").write_text(
        json.dumps(list(pros_cons.values()), ensure_ascii=False, indent=1, default=_py), encoding="utf-8")
    (ART_DIR / "movie_summary.json").write_text(
        json.dumps({"prior_mean_star": prior_mean, "config": acfg.__dict__, "movies": out.to_dict("records")},
                   ensure_ascii=False, indent=1, default=_py), encoding="utf-8")
    if verbose:
        print(f"  [aggregate] {n} 条影评 / {len(out)} 部电影 -> movie_summary.csv")
        show = out.head(5)[["movie_cn", "n_reviews", "star_pos_rate", "final_score"]]
        print(show.to_string(index=False))
    write_report(out, pros_cons, prior_mean)
    return out, pros_cons


def write_report(out: pd.DataFrame, pros_cons: dict, prior_mean: float):
    lines = ["# 电影评价分析报告（decoder-only 因果语言模型）", "",
             f"- 影评总数：**{int(out['n_reviews'].sum()):,}** 条；电影：**{len(out)}** 部",
             f"- 评分口径：综合分 = 0.6 × 贝叶斯星级折算 + 0.4 × 模型好评率；全局平均星级 {prior_mean:.2f}", ""]
    for _, r in out.iterrows():
        pc = pros_cons.get(int(r["movie_id"]), {})
        lines.append(f"## {r['movie_cn']}（{r['movie_en']}）— **{r['final_score']:.2f} / 10**")
        lines.append("")
        lines.append(f"- 影评 {int(r['n_reviews']):,} 条｜真实好评 {int(r['star_pos']):,} / 差评 {int(r['star_neg']):,} / "
                     f"中评 {int(r['star_mid']):,}｜好评率 {r['star_pos_rate']:.1%}")
        lines.append(f"- 模型判定好评 {int(r['model_pos']):,} / 差评 {int(r['model_neg']):,}｜模型好评率 "
                     f"{r['model_pos_rate']:.1%}｜贝叶斯均分 {r['bayes_rating']:.2f}/5｜置信度 {r['confidence']:.2f}")
        if "suspect_rate" in out.columns:
            lines.append(f"- 疑似刷评 {int(r['n_suspect']):,} 条（{r['suspect_rate']:.2%}）")
        if pc.get("pros"):
            lines.append("- **优点**：" + "；".join(
                f"{p['aspect']}（提及 {p['mentions']}，好评率 {p['pos_rate']:.0%}）" for p in pc["pros"]))
            for p in pc["pros"][:2]:
                if p["evidence"]:
                    lines.append(f"    - “{p['evidence'][0][:60]}”")
        if pc.get("cons"):
            lines.append("- **缺点**：" + "；".join(
                f"{c['aspect']}（提及 {c['mentions']}，好评率 {c['pos_rate']:.0%}）" for c in pc["cons"]))
            for c in pc["cons"][:2]:
                if c["evidence"]:
                    lines.append(f"    - “{c['evidence'][0][:60]}”")
        elif pc.get("weak"):
            lines.append("- **相对短板**（无明显差评集中，但好评率最低）：" + "；".join(
                f"{w['aspect']}（提及 {w['mentions']}，好评率 {w['pos_rate']:.0%}）" for w in pc["weak"]))
        lines.append("")
    (ART_DIR / "分析报告.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior-m", type=int, default=None)
    a = ap.parse_args()
    cfg = AggregateConfig()
    if a.prior_m:
        cfg.prior_m = a.prior_m
    aggregate(cfg)
