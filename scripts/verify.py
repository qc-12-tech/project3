"""结果自检：对聚合产物做一致性校验，作为项目验收脚本。

用法：python scripts/verify.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("PROJECT_ART_DIR", ROOT / "artifacts"))
SMOKE = os.environ.get("PROJECT_SMOKE") == "1"          # 冒烟模式：样本少，放宽规模类断言
N_MOVIES = len(json.loads((ART / "movies.json").read_text(encoding="utf-8"))["movies"]) if (ART / "movies.json").exists() else 0
FAILS, WARNS = [], []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
    print(("  ✅ " if cond else "  ❌ ") + msg)
    return cond


def warn(cond, msg):
    if not cond:
        WARNS.append(msg)
        print("  ⚠️  " + msg)


def main():
    print("== 1. 文件存在性 ==")
    for f in ("movie_summary.csv", "movie_summary.json", "movie_pros_cons.json", "aspects.json",
              "train_log.json", "分析报告.md", "vocab.json", "movies.json"):
        check((ART / f).exists(), f"artifacts/{f} 存在")
    preds = sorted((ART / "preds").glob("preds_*.parquet"))
    check(len(preds) > 0, f"预测分片 {len(preds)} 个")

    summary = pd.read_csv(ART / "movie_summary.csv")
    print(f"\n== 2. 汇总表：{len(summary)} 部电影 ==")
    check(len(summary) == N_MOVIES, f"电影数 = {N_MOVIES}")
    check(summary["final_score"].between(0, 10).all(), "综合分全部落在 0~10")
    check((summary["star_pos"] + summary["star_neg"] + summary["star_mid"] == summary["n_reviews"]).all(),
          "星级三分类之和 == 影评数")
    check((summary["model_pos"] + summary["model_neg"] == summary["n_reviews"]).all(),
          "模型好评+差评 == 影评数")
    check(summary["bayes_rating"].between(1, 5).all(), "贝叶斯均分落在 1~5")
    if not SMOKE:
        check(summary["n_reviews"].min() >= 1000, "每部电影至少 1000 条影评")
    warn((summary["pros"].notna()).sum() >= 20, "至少 20 部电影挖掘出优点")
    warn((summary["cons"].notna()).sum() >= 20, "至少 20 部电影挖掘出缺点")

    print("\n== 3. 预测一致性 ==")
    n_rows = 0
    ppos_min, ppos_max = 1.0, 0.0
    thr = json.loads((ART / "train_log.json").read_text(encoding="utf-8")).get("calibration", {}).get("threshold", 0.5)
    for f in preds:
        df = pd.read_parquet(f, columns=["p_pos", "pred_sent", "exp_rating", "star"])
        n_rows += len(df)
        ppos_min = min(ppos_min, float(df["p_pos"].min()))
        ppos_max = max(ppos_max, float(df["p_pos"].max()))
        check(df["pred_sent"].isin([0, 1]).all(), f"{f.name}: pred_sent ∈ {{0,1}}")
        warn(df["exp_rating"].between(1, 5).all(), f"{f.name}: 期望评分落在 1~5")
        agree = ((df["p_pos"] >= thr).astype(int) == df["pred_sent"].astype(int)).mean()
        warn(agree > 0.99, f"{f.name}: pred_sent 与校准阈值 {thr:.3f} 判定一致率 {agree:.2%}")
    check(ppos_min >= 0 and ppos_max <= 1, f"P(好) ∈ [0,1]（{ppos_min:.3f}~{ppos_max:.3f}）")
    print(f"  预测总条数：{n_rows:,}")
    check(n_rows == int(summary["n_reviews"].sum()), "预测条数 == 汇总表影评总数")

    print("\n== 4. 模型与训练 ==")
    log = json.loads((ART / "train_log.json").read_text(encoding="utf-8"))
    ev = log.get("final_eval") or (log.get("history") or [{}])[-1]
    check(ev.get("sent_acc", 0) > 0.80, f"验证集情感准确率 > 0.80（{ev.get('sent_acc', 0):.4f}）")
    check(ev.get("sent_f1_pos", 0) > 0.85, f"好评 F1 > 0.85（{ev.get('sent_f1_pos', 0):.4f}）")
    check(ev.get("rating_mae", 9) < 0.8, f"评分 MAE < 0.8（{ev.get('rating_mae', 9):.4f}）")
    cal = log.get("calibration", {})
    if cal:
        check(cal.get("acc", 0) > 0.80, f"校准阈值下准确率 > 0.80（{cal.get('acc', 0):.4f}）")
        print(f"  阈值校准：{cal.get('threshold'):.4f} → acc={cal.get('acc', 0):.4f} "
              f"F1+={cal.get('f1_pos', 0):.4f}（判正率 {cal.get('pos_rate_pred', 0):.3f} vs "
              f"真实 {cal.get('pos_rate_true', 0):.3f}）")

    print("\n== 5. 优缺点挖掘 ==")
    pc = json.loads((ART / "movie_pros_cons.json").read_text(encoding="utf-8"))
    n_pros = sum(1 for x in pc if x["pros"])
    n_cons = sum(1 for x in pc if x["cons"])
    n_weak = sum(1 for x in pc if x.get("weak"))
    ev_ok = all(e["evidence"] == [] or all(isinstance(s, str) and s for s in e["evidence"])
                for x in pc for e in x["pros"] + x["cons"] + x.get("weak", []))
    check(ev_ok, "优点/缺点/短板证据句均为非空字符串")
    check(n_pros == N_MOVIES, f"{N_MOVIES} 部电影都挖掘出优点（实际 {n_pros}）")
    check(n_cons + n_weak == N_MOVIES, f"每部电影都有缺点或相对短板（显著缺点 {n_cons} + 相对短板 {n_weak}）")
    print(f"  显著缺点 {n_cons}/28，相对短板补齐 {n_weak}/28")

    print("\n== 6. 刷评论检测 ==")
    if (ART / "spam_summary.json").exists():
        sp = json.loads((ART / "spam_summary.json").read_text(encoding="utf-8"))
        check(len(sp) == N_MOVIES, f"刷评检测覆盖 {N_MOVIES} 部电影")
        check(all(0 <= x["suspect_rate"] <= 1 for x in sp), "疑似占比落在 0~1")
        check(all(x["n_suspect"] <= x["n_reviews"] for x in sp), "疑似条数 <= 总条数")
        check(all(x["suspect_rate"] > 0 for x in sp), "每部电影都检出了可疑样本")
        n_sp = sum(x["n_suspect"] for x in sp)
        n_all = sum(x["n_reviews"] for x in sp)
        print(f"  全站疑似刷评 {n_sp:,}/{n_all:,} = {n_sp/n_all:.2%}")
        top = sorted(sp, key=lambda x: -x["suspect_rate"])[:5]
        for x in top:
            print(f"    {x['movie_cn']:<10s} {x['suspect_rate']:.2%}（{x['n_suspect']}/{x['n_reviews']}）"
                  f" 主要信号: {sorted(x['by_type'], key=lambda k: -x['by_type'][k])[:3]}")
        check(all("reasons" in t for x in sp for t in x["top_suspects"]) or not any(x["top_suspects"] for x in sp),
              "可疑样本带命中信号说明")
    else:
        WARNS.append("缺少 spam_summary.json")
        print("  ⚠️  未找到刷评论检测结果（先运行 python -m src.spam）")

    print("\n" + "=" * 60)
    if FAILS:
        print(f"❌ 失败 {len(FAILS)} 项：")
        for m in FAILS:
            print("   -", m)
        return 1
    print(f"✅ 全部必需检查通过（警告 {len(WARNS)} 项）")
    for m in WARNS:
        print("   -", m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
