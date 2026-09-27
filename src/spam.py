"""刷评论（水军）检测：无监督可疑度打分 + 可解释证据。

数据集没有「刷评」标注，因此这里做的是**可疑度打分 + 证据链**，而不是有监督二分类。
信号分 5 类，全部向量化计算（200 万条不靠 Python 循环）：

1. 文本复用：完全重复（同片内 / 跨片复用同一段文字）、模板化开头（同片内前 12 字高度重复）
2. 广告营销：微信/VX/私聊/资源/福利/扫码等关键词
3. 用户行为：同一用户对同一电影反复评论、同一天在该电影集中刷多条
4. 时间聚集：某部电影单日评论量超过全局 p99.9（突然爆量）
5. 情感-评分矛盾：decoder-only 模型高置信判断的情感与星级相反（5 星配强负面文本 / 1 星配强正面文本）
6. 形态异常：高赞 + 极短文本（买赞嫌疑）、极短无信息文本

输出：
  artifacts/user_meta.parquet     每条影评的用户/日期行为特征
  artifacts/spam_reviews.parquet  每条影评的可疑度与命中信号
  artifacts/spam_summary.json/csv 每部电影的疑似刷评比例与最可疑样本
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import DATA_CSV, ART_DIR, PRED_DIR, STAR_POS_MIN, STAR_NEG_MAX
from .data import clean_text

# ------------------------------------------------------------------ 权重与规则
WEIGHTS = {
    "dup_same_movie": 0.55,    # 完全重复（同片）
    "dup_cross_movie": 0.45,   # 同一段文字出现在多部电影
    "template": 0.35,          # 模板化开头（同片 ≥5 条）
    "ad": 0.60,                # 广告/导流
    "user_burst": 0.45,        # 同一天在同一电影刷 ≥5 条
    "user_repeat": 0.35,       # 同一用户对同一电影评论 ≥3 次
    "movie_day_burst": 0.20,   # 该片当天评论量 > 全局 p99.9
    "rating_mismatch": 0.50,   # 模型高置信情感与星级相反
    "like_bomb": 0.40,         # 高赞 + 极短文本
    "low_info": 0.20,          # 极短无信息文本
}
SUSPECT_THRESHOLD = 0.50
AD_RE = re.compile(
    r"微信|微.?信|weixin|\bwx\b|\bvx\b|v信|加我|私聊|私信|QQ|公众号|扫码|二维码|链接|网址|资源|种子|免费看|"
    r"在线看|下载|福利|领取|代刷|返现|优惠|折扣|包场|团购|票务|一手|全网|独家|客服|咨询|联系我|"
    r"www\.|http|\.com|\.cn|@\w+", re.I)
_NON_WORD = re.compile(r"[^\u4e00-\u9fa5A-Za-z0-9]")
_REPEAT = re.compile(r"(.)\1{2,}")


def _sizes(keys: np.ndarray) -> np.ndarray:
    """每个元素所在分组的样本数。"""
    uniq, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    return cnt[inv]


def build_user_meta(force: bool = False, chunksize: int = 100_000, verbose: bool = True) -> Path:
    """流式复算与 data.prepare 完全一致的行序，导出每条影评的用户/日期行为特征。"""
    out = ART_DIR / "user_meta.parquet"
    if out.exists() and not force:
        if verbose:
            print(f"  [spam] 复用已有 {out}")
        return out

    seen, row_id = set(), 0
    rows_id, rows_mv, rows_uid, rows_day = [], [], [], []
    cols = ["Movie_Name_CN", "Star", "Username", "Date", "Comment"]
    for ci, ch in enumerate(pd.read_csv(DATA_CSV, chunksize=chunksize, usecols=cols)):
        ch["comment"] = ch["Comment"].map(clean_text)
        ch = ch[(ch["comment"].str.len() >= 4) & ch["Star"].between(1, 5)]
        key = (ch["Movie_Name_CN"].astype(str) + "|" + ch["Username"].astype(str) + "|" + ch["comment"])
        h = pd.util.hash_array(key.to_numpy(dtype=object))
        keep = np.array([x not in seen for x in h])
        for x in h[keep]:
            seen.add(x)
        ch = ch[keep]
        if not len(ch):
            continue
        n = len(ch)
        rows_id.append(np.arange(row_id, row_id + n, dtype=np.int64))
        rows_mv.append(pd.util.hash_array(ch["Movie_Name_CN"].astype(str).to_numpy(dtype=object)).astype(np.int64))
        rows_uid.append(pd.util.hash_array(ch["Username"].astype(str).to_numpy(dtype=object)).astype(np.int64))
        day = pd.to_datetime(ch["Date"], errors="coerce")
        days = (day - pd.Timestamp("1970-01-01")).dt.days.fillna(-1).to_numpy(dtype=np.int64)
        rows_day.append(days)
        row_id += n
        if verbose and ci % 5 == 0:
            print(f"  [spam] 用户特征 {row_id} 条", flush=True)

    df = pd.DataFrame({"row_id": np.concatenate(rows_id), "movie_key": np.concatenate(rows_mv),
                       "uid": np.concatenate(rows_uid), "day": np.concatenate(rows_day)})
    df["user_total"] = _sizes(df["uid"].to_numpy())
    df["user_movie_count"] = _sizes((df["uid"].to_numpy() << 20) ^ (df["movie_key"].to_numpy() & 0xFFFFF))
    df["user_movie_day_count"] = _sizes(
        (df["uid"].to_numpy() << 20) ^ (df["movie_key"].to_numpy() & 0xFFFFF) ^ (df["day"].to_numpy() << 40))
    df["movie_day_count"] = _sizes((df["movie_key"].to_numpy() << 20) ^ (df["day"].to_numpy() & 0xFFFFF))
    df[["row_id", "uid", "day", "user_total", "user_movie_count", "user_movie_day_count",
        "movie_day_count"]].to_parquet(out, index=False)
    if verbose:
        print(f"  [spam] -> {out}（{len(df)} 行）", flush=True)
    return out


def detect(threshold: float = SUSPECT_THRESHOLD, limit: int = None, verbose: bool = True):
    movies = {m["id"]: m for m in json.loads((ART_DIR / "movies.json").read_text(encoding="utf-8"))["movies"]}
    preds = pd.concat([pd.read_parquet(f, columns=["row_id", "movie_id", "star", "likes", "p_pos", "pred_sent"])
                       for f in sorted(PRED_DIR.glob("preds_*.parquet"))], ignore_index=True)
    rev = pd.read_parquet(ART_DIR / "reviews.parquet", columns=["row_id", "movie_id", "star", "likes", "comment"])
    um = pd.read_parquet(build_user_meta())
    df = rev.merge(preds[["row_id", "p_pos", "pred_sent"]], on="row_id", how="left")
    df = df.merge(um.drop(columns=["uid", "day"]), on="row_id", how="left")
    if limit:
        df = df.head(limit).copy()
    text = df["comment"].astype(str)
    norm1 = text.str.replace(_NON_WORD, "", regex=True)
    h1 = pd.util.hash_array(norm1.to_numpy(dtype=object)).astype(np.int64)
    mv = df["movie_id"].to_numpy(dtype=np.int64)
    likes = df["likes"].to_numpy(dtype=np.int64)
    star = df["star"].to_numpy(dtype=np.int64)
    pp = df["p_pos"].to_numpy(dtype=np.float64)

    if verbose:
        print(f"  [spam] 分析 {len(df)} 条影评", flush=True)
    # 1) 完全重复：同片 / 跨片
    pair = np.stack([h1, mv], axis=1)
    pu, pinv, pcnt = np.unique(pair, axis=0, return_inverse=True, return_counts=True)
    dup_same = pcnt[pinv]
    _, h1inv, h1cnt = np.unique(h1, return_inverse=True, return_counts=True)
    text_total = h1cnt[h1inv]
    uh, uhinv, uhcnt = np.unique(pu[:, 0], return_inverse=True, return_counts=True)
    pair_movies = uhcnt[uhinv]
    n_movies_with_text = pair_movies[pinv]
    # 2) 模板化开头（同片内前 12 字重复）
    tmpl = norm1.str.replace(_REPEAT, r"\1", regex=True).str[:12].where(norm1.str.len() >= 12, other="")
    h2 = pd.util.hash_array(tmpl.to_numpy(dtype=object)).astype(np.int64)
    tmpl_size = np.where(tmpl.str.len().to_numpy() > 0,
                         _sizes((h2 << 20) ^ (mv & 0xFFFFF)), 0)
    # 3) 广告 / 4) 形态
    ad = text.str.contains(AD_RE, regex=True, na=False).to_numpy()
    low_info = (norm1.str.len() <= 5).to_numpy()
    like_bomb = (likes >= 300) & (text.str.len() <= 15).to_numpy()
    # 5) 用户 / 时间聚集
    user_burst = (df["user_movie_day_count"].to_numpy() >= 5)
    user_repeat = (df["user_movie_count"].to_numpy() >= 3)
    mdc = df["movie_day_count"].to_numpy(dtype=np.float64)
    day_p999 = float(np.percentile(mdc, 99.9))
    movie_day_burst = mdc > day_p999
    # 6) 情感-评分矛盾（仅 1/2/4/5 星，且模型高置信）
    star_sent = np.where(star >= STAR_POS_MIN, 1, np.where(star <= STAR_NEG_MAX, 0, -1))
    mismatch = (star_sent >= 0) & (((pp >= 0.9) & (star_sent == 0)) | ((pp <= 0.1) & (star_sent == 1)))

    flags = {
        "dup_same_movie": dup_same >= 2,
        "dup_cross_movie": (n_movies_with_text >= 3) & (text_total >= 3),
        "template": tmpl_size >= 5,
        "ad": ad,
        "user_burst": user_burst,
        "user_repeat": user_repeat,
        "movie_day_burst": movie_day_burst,
        "rating_mismatch": mismatch,
        "like_bomb": like_bomb,
        "low_info": low_info,
    }
    score = np.zeros(len(df), dtype=np.float64)
    for k, v in flags.items():
        score += WEIGHTS[k] * v.astype(np.float64)
    score = np.clip(score, 0, 1)
    reasons = np.array(["|".join(k for k, v in flags.items() if v[i]) for i in range(len(df))], dtype=object)

    out = pd.DataFrame({
        "row_id": df["row_id"].to_numpy(dtype=np.int32),
        "movie_id": df["movie_id"].to_numpy(dtype=np.int32),
        "spam_score": score.astype(np.float32),
        "is_suspect": (score >= threshold),
        "reasons": reasons,
        "dup_same_movie_size": dup_same.astype(np.int32),
        "text_total_count": text_total.astype(np.int32),
        "text_movie_count": n_movies_with_text.astype(np.int32),
        "template_size": tmpl_size.astype(np.int32),
    })
    out.to_parquet(ART_DIR / "spam_reviews.parquet", index=False)

    df["spam_score"], df["reasons"] = score, reasons
    sus = df[score >= threshold]
    summary = {}
    for mid in sorted(df["movie_id"].unique()):
        m_all = df[df["movie_id"] == mid]
        m_sus = sus[sus["movie_id"] == mid]
        by_type = {k: int(v[df["movie_id"].to_numpy() == mid].sum()) for k, v in flags.items()}
        top = m_sus.sort_values(["spam_score", "likes"], ascending=False).head(8)
        summary[int(mid)] = {
            "movie_id": int(mid), "movie_cn": movies[int(mid)]["cn"], "movie_en": movies[int(mid)]["en"],
            "n_reviews": int(len(m_all)), "n_suspect": int(len(m_sus)),
            "suspect_rate": round(float(len(m_sus) / max(1, len(m_all))), 4),
            "mean_spam_score": round(float(m_all["spam_score"].mean()), 4),
            "by_type": by_type,
            "day_burst_p999": day_p999,
            "top_suspects": [{"comment": str(r.comment)[:80], "star": int(r.star), "likes": int(r.likes),
                              "spam_score": round(float(r.spam_score), 2), "reasons": r.reasons}
                             for r in top.itertuples()],
        }
    (ART_DIR / "spam_summary.json").write_text(json.dumps(list(summary.values()), ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    pd.DataFrame([{**{k: v for k, v in s.items() if k not in ("by_type", "top_suspects")},
                   **{f"spam_{k}": v for k, v in s["by_type"].items()}} for s in summary.values()]).to_csv(
        ART_DIR / "spam_summary.csv", index=False, encoding="utf-8-sig")
    if verbose:
        tot = sum(s["n_suspect"] for s in summary.values())
        print(f"  [spam] 疑似刷评 {tot}/{len(df)} = {tot/len(df):.2%} -> spam_summary.json / spam_reviews.parquet")
        for s in sorted(summary.values(), key=lambda x: -x["suspect_rate"])[:8]:
            print(f"    {s['movie_cn']:<10s} 可疑率 {s['suspect_rate']:.2%}（{s['n_suspect']}/{s['n_reviews']}）")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=SUSPECT_THRESHOLD)
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 条（调试用）")
    ap.add_argument("--force-user-meta", action="store_true")
    a = ap.parse_args()
    build_user_meta(force=a.force_user_meta)
    detect(a.threshold, limit=a.limit)
