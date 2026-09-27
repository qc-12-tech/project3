"""Streamlit 前端：电影评价分析看板。

启动：streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.service import aspects, movie_list, predict_text, sample_reviews, spam    # noqa: E402

st.set_page_config(page_title="电影评价分析（decoder-only Transformer）", page_icon="🎬", layout="wide")


@st.cache_data(show_spinner=False)
def _movies():
    return movie_list()


@st.cache_data(show_spinner=False)
def _aspects(mid):
    return aspects().get(str(mid))


try:
    movies = _movies()
except Exception as e:                                        # noqa: BLE001
    st.error(f"还没有聚合结果，请先运行完整流程：\n\n```\npython -m src.data && python -m src.train && "
             f"python -m src.infer && python -m src.aspects && python -m src.aggregate\n```\n\n错误：{e}")
    st.stop()

# ------------------------------------------------------------------ 侧边栏
st.sidebar.title("🎬 电影评价分析")
st.sidebar.caption("decoder-only（GPT 式）因果语言模型 · 从零训练")
names = {f"{m['movie_cn']}（{m['movie_en']}）": m for m in movies}
pick = st.sidebar.selectbox("选择电影", list(names.keys()))
m = names[pick]
mid = int(m["movie_id"])

st.sidebar.markdown("---")
st.sidebar.markdown("**全站排行（综合分）**")
rank = pd.DataFrame([{"电影": x["movie_cn"], "分数": float(x["final_score"])} for x in movies[:10]])
st.sidebar.dataframe(rank, hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ 头部
st.title(f"{m['movie_cn']} · {m['movie_en']}")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("综合评分", f"{float(m['final_score']):.2f} / 10")
c2.metric("贝叶斯均分", f"{float(m['bayes_rating']):.2f} / 5")
c3.metric("影评总数", f"{int(m['n_reviews']):,}")
c4.metric("好评 / 差评", f"{int(m['star_pos']):,} / {int(m['star_neg']):,}")
c5.metric("模型好评率", f"{float(m['model_pos_rate']):.1%}")
st.caption(f"模型判定：好评 {int(m['model_pos']):,} 条 · 差评 {int(m['model_neg']):,} 条 · "
           f"中评 {int(m['star_mid']):,} 条｜置信度 {float(m['confidence']):.2f}｜点赞合计 {int(m['likes_sum']):,}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 评价统计", "👍 优点 / 👎 缺点", "🗒 影评样本", "🚨 刷评论检测", "✍️ 单条预测"])

# ------------------------------------------------------------------ 统计
with tab1:
    left, right = st.columns(2)
    with left:
        st.subheader("真实星级分布 → 好评 / 中评 / 差评")
        dist = pd.DataFrame({"类别": ["好评(4-5星)", "中评(3星)", "差评(1-2星)"],
                             "条数": [int(m["star_pos"]), int(m["star_mid"]), int(m["star_neg"])]}).set_index("类别")
        st.bar_chart(dist)
        st.dataframe(dist.assign(占比=(dist["条数"] / dist["条数"].sum()).map("{:.2%}".format)))
    with right:
        st.subheader("模型判定（decoder-only 生成式情感）")
        dist2 = pd.DataFrame({"类别": ["好评", "差评"],
                              "条数": [int(m["model_pos"]), int(m["model_neg"])]}).set_index("类别")
        st.bar_chart(dist2)
        st.dataframe(pd.DataFrame({"指标": ["模型好评率", "真实好评率", "模型平均预测评分", "真实平均星级"],
                                   "数值": [f"{float(m['model_pos_rate']):.2%}", f"{float(m['star_pos_rate']):.2%}",
                                            f"{float(m['pred_rating_mean']):.2f}", f"{float(m['star_mean']):.2f}"]}))

# ------------------------------------------------------------------ 优缺点
with tab2:
    a = _aspects(mid)
    pc1, pc2 = st.columns(2)
    with pc1:
        st.subheader("👍 优点")
        if m["pros_detail"]:
            for p in m["pros_detail"]:
                st.markdown(f"**{p['aspect']}**（提及 {p['mentions']} 次，好评率 {p['pos_rate']:.0%}）")
                for ev in p["evidence"]:
                    st.caption(f"“{ev}”")
        else:
            st.info("没有达到阈值的优点方面词")
        if m["keywords_pos"]:
            st.markdown("**好评高频词**：" + " · ".join(m["keywords_pos"]))
    with pc2:
        st.subheader("👎 缺点")
        if m["cons_detail"]:
            for c in m["cons_detail"]:
                st.markdown(f"**{c['aspect']}**（提及 {c['mentions']} 次，好评率 {c['pos_rate']:.0%}）")
                for ev in c["evidence"]:
                    st.caption(f"“{ev}”")
        else:
            st.info("没有达到阈值的缺点方面词")
        if not m["cons_detail"] and m.get("weak_detail"):
            st.markdown("**相对短板**（无明显差评集中，但好评率最低）")
            for w in m["weak_detail"]:
                st.markdown(f"- {w['aspect']}（提及 {w['mentions']} 次，好评率 {w['pos_rate']:.0%}）")
                for ev in w["evidence"][:1]:
                    st.caption(f"“{ev}”")
        if m["keywords_neg"]:
            st.markdown("**差评高频词**：" + " · ".join(m["keywords_neg"]))
    if a:
        st.markdown("---")
        st.subheader("各方面词极性（按提及数排序）")
        rows = [{"方面": r["aspect"], "提及数": r["mentions"], "好评率": r["pos_rate"],
                 "点赞加权好评率": r["like_weighted_pos_rate"]} for r in a["aspects"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.bar_chart(pd.DataFrame(rows).set_index("方面")["好评率"])

# ------------------------------------------------------------------ 影评样本
with tab3:
    pol = st.radio("查看", ["好评", "差评"], horizontal=True)
    revs = sample_reviews(mid, "pos" if pol == "好评" else "neg", 20)
    if not revs:
        st.info("没有样本")
    for r in revs:
        with st.container(border=True):
            st.write(r["comment"])
            st.caption(f"真实星级 {r['star']} ★｜点赞 {r['likes']}｜模型 P(好)={r['p_pos']}｜"
                       f"模型预测评分 {r['pred_rating']}")

# ------------------------------------------------------------------ 刷评论检测
with tab4:
    s = spam().get(mid)
    if not s:
        st.info("还没有刷评论检测结果，请运行 `python -m src.spam`")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("疑似刷评", f"{s['n_suspect']:,} 条")
        c2.metric("疑似占比", f"{s['suspect_rate']:.2%}")
        c3.metric("平均可疑度", f"{s['mean_spam_score']:.3f}")
        st.caption(f"判定阈值 0.50；单日评论量爆量阈值 p99.9 = {s['day_burst_p999']:.0f} 条/天。"
                   f"刷评检测是**无监督可疑度**（数据集没有人工标注），不是确证。")
        zh = {"dup_same_movie": "完全重复(同片)", "dup_cross_movie": "跨片复用", "template": "模板化开头",
              "ad": "广告导流", "user_burst": "同日集中刷评", "user_repeat": "同用户反复评论",
              "movie_day_burst": "单日爆量", "rating_mismatch": "星级与文本情感矛盾",
              "like_bomb": "高赞短评", "low_info": "极短无信息"}
        hit = pd.DataFrame({"信号": [zh.get(k, k) for k in s["by_type"]],
                            "命中条数": list(s["by_type"].values())}).set_index("信号").sort_values("命中条数", ascending=False)
        st.bar_chart(hit[hit["命中条数"] > 0])
        st.dataframe(hit, use_container_width=True)
        st.subheader("最可疑样本")
        for r in s["top_suspects"]:
            with st.container(border=True):
                st.write(r["comment"])
                st.caption(f"可疑度 {r['spam_score']}｜星级 {r['star']} ★｜点赞 {r['likes']}｜命中 "
                           f"{'、'.join(zh.get(x, x) for x in r['reasons'].split('|'))}")

# ------------------------------------------------------------------ 单条预测
with tab5:
    text = st.text_area("输入一条影评，decoder-only 模型两步自回归解码给出好评/差评与评分",
                        "剧情拖沓，特效还行，演员表演在线，但结局太仓促了。", height=120)
    if st.button("预测", type="primary"):
        st.json(predict_text(text.strip()))
