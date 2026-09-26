"""美团外卖 · 评价管理中心（Streamlit 前端）。

视觉：美团黄主题（.streamlit/config.toml 提供主色）+ 少量安全 CSS；
布局：全部使用 Streamlit 原生控件（container(border=True) / columns / metric），
避免用自定义 HTML 包裹原生控件导致的错位。文案去 AI 化。

启动（两个终端）：
    # 1) 后端
    uvicorn app.main:app --port 8000
    # 2) 前端
    streamlit run app/streamlit_app.py
"""
import json

import pandas as pd
import requests
import streamlit as st

DEFAULT_API = "http://127.0.0.1:8000"

st.set_page_config(page_title="外卖评价管理中心", page_icon="🛵", layout="wide")

# ---------------- 少量安全 CSS（只做圆角/字重，不搞跨元素 div 包裹） ----------------
_CSS = """
<style>
.block-container { padding-top: 1.2rem; }
/* 主按钮：美团黄底 + 深色文字（保证黄色底上的可读性） */
.stButton > button {
    background: linear-gradient(180deg, #ffd43b, #ffb300);
    color: #1a1a1a;
    border: none;
    border-radius: 8px;
    font-weight: 700;
}
.stButton > button:hover, .stButton > button:active, .stButton > button:focus {
    background: linear-gradient(180deg, #ffd43b, #ffb000);
    color: #1a1a1a;
    border: none;
}
.stTextArea textarea, .stTextInput input { border-radius: 8px; }
[data-testid="stMetric"] { border-radius: 12px; }

/* 徽标（自包含、单行，安全） */
.mt-badge {
    display: inline-block; padding: 1px 10px; border-radius: 999px;
    font-size: 0.8rem; font-weight: 600; line-height: 1.6;
}
.mt-star { color: #ffb300; letter-spacing: 1px; }
.mt-star-empty { color: #e0e0e0; letter-spacing: 1px; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ---------------- 配色与文案 ----------------
LABEL_COLOR = {"好评": "#00b578", "中评": "#ff8a00", "差评": "#ff4d4f"}
SEV_COLOR = {"严重": "#ff4d4f", "警告": "#ff8a00", "正常": "#00b578"}
SEV_TEXT = {"严重": "严重", "警告": "关注", "正常": "正常"}
ADVICE_COLOR = {"推荐购买": "#00b578", "谨慎购买": "#ff8a00", "不建议购买": "#ff4d4f"}


def badge(text, color):
    return (f'<span class="mt-badge" style="background:{color}1A;color:{color};">'
            f'{text}</span>')


def label_badge(label):
    return badge(label, LABEL_COLOR.get(label, "#666"))


def sev_badge(sev):
    return badge(SEV_TEXT.get(sev, sev), SEV_COLOR.get(sev, "#666"))


def star_html(rating):
    full = max(1, min(5, int(round(rating))))
    return (f'<span class="mt-star">{"★" * full}</span>'
            f'<span class="mt-star-empty">{"☆" * (5 - full)}</span>')


def score_to_rating(score):
    """满意度 score 0~1 -> 1~5 星。"""
    return round(1.0 + 4.0 * score, 1)


def render_alerts(res):
    """用原生提示组件渲染预警（红/橙/绿，稳定不跑版）。"""
    if res.get("need_process"):
        st.error("⚠️ 差评预警：该评价情绪负面，建议尽快联系顾客处理")
    if res.get("need_review"):
        st.warning("🧐 系统存疑：该评价情绪较模糊，建议人工核实")
    if not res.get("need_process") and not res.get("need_review"):
        st.success("识别稳定，无需额外处理")


# ---------------- API 封装 ----------------
def api_get(base, path, params=None):
    try:
        r = requests.get(base + path, params=params, timeout=30)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def api_post(base, path, payload):
    try:
        r = requests.post(base + path, json=payload, timeout=120)
        r.raise_for_status()
        return r.json(), None
    except requests.HTTPError as e:
        return None, f"{e} - {r.text}"
    except Exception as e:
        return None, str(e)


# ---------------- 侧边栏 ----------------
st.sidebar.markdown("**🛵 评价管理中心**")
st.sidebar.caption("外卖商家运营台")
nav = st.sidebar.radio(
    "导航",
    ["评价识别", "批量识别", "差评分析", "购买建议", "待核实评价", "数据统计"],
    label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.caption("服务地址")
base = st.sidebar.text_input("FastAPI 地址", DEFAULT_API, label_visibility="collapsed")
if st.sidebar.button("检测连接"):
    info, err = api_get(base, "/")
    if err:
        st.sidebar.error(f"连接失败：{err}")
    else:
        st.sidebar.success("连接成功")

st.sidebar.markdown("---")
st.sidebar.caption(
    "满意度 1~5 星由系统按好评/中评/差评综合给出；"
    "低于 2.2 星建议尽快处理，存疑评价建议人工核实。阈值可在 src/config.py 调整。")

# ---------------- 顶部品牌区 ----------------
st.markdown(
    '<div style="height:4px;background:linear-gradient(90deg,#ffc300,#ff8a00);'
    'border-radius:3px;margin-bottom:10px;"></div>',
    unsafe_allow_html=True)
st.markdown("## 🛵 外卖评价管理中心")
st.caption("评价自动分类 · 差评预警 · 问题洞察 · 人工核实")
st.divider()

# ============================================================
# 1) 评价识别（单条）
# ============================================================
if nav == "评价识别":
    with st.container(border=True):
        st.markdown("**识别一条评价**")
        st.caption("粘贴顾客评价，系统自动分类并给出满意度与处理建议")
        text = st.text_area(
            "评价内容", "等了两个小时才送到，饭都凉了，包装还破了，差评！", height=110,
            label_visibility="collapsed")
        c1, c2, _ = st.columns([1, 1, 2])
        save = c1.checkbox("记录到后台", value=True)
        submit = c2.button("识别评价", type="primary")

    if submit:
        if not text.strip():
            st.info("请输入评价内容")
        else:
            res, err = api_post(base, "/predict", {"text": text, "save": save})
            if err:
                st.error(f"请求失败：{err}")
            else:
                rating = score_to_rating(res["score"])
                confidence = round((1 - res["error_prob"]) * 100, 1)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("评价分类", res["label"])
                c2.metric("满意度", f"{rating} 分")
                c3.metric("识别置信度", f"{confidence}%")
                c4.metric("处理等级", SEV_TEXT.get(res["severity"], res["severity"]))
                st.markdown(
                    f'{star_html(rating)}　<span style="color:#666;">{rating} / 5.0</span>　'
                    f'{label_badge(res["label"])}　{sev_badge(res["severity"])}',
                    unsafe_allow_html=True)
                render_alerts(res)
                with st.container(border=True):
                    st.markdown("**分类倾向**")
                    probs = pd.DataFrame(
                        {"类别": list(res["probs"].keys()),
                         "概率": list(res["probs"].values())}).set_index("类别")
                    st.bar_chart(probs, height=240)
                with st.expander("查看明细"):
                    st.json(res, expanded=False)

# ============================================================
# 2) 批量识别
# ============================================================
elif nav == "批量识别":
    with st.container(border=True):
        st.markdown("**批量识别**")
        st.caption("每行一条评价，一次性完成分类与预警汇总")
        default = "味道很好，还会再点\n一般般吧，没什么惊喜\n太难吃了，再也不来\n包装破了，汤洒了一袋子"
        bulk = st.text_area("评价列表", default, height=150, label_visibility="collapsed")
        c1, c2, _ = st.columns([1, 1, 2])
        save_b = c1.checkbox("记录到后台", value=True)
        submit_b = c2.button("批量识别", type="primary")

    if submit_b:
        texts = [t.strip() for t in bulk.splitlines() if t.strip()]
        if not texts:
            st.info("请输入至少一条评价")
        else:
            res, err = api_post(base, "/predict/batch", {"texts": texts, "save": save_b})
            if err:
                st.error(f"请求失败：{err}")
            else:
                s = res["summary"]
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("总数", s["total"])
                c2.metric("好评", s["good"])
                c3.metric("中评", s["neutral"])
                c4.metric("差评", s["bad"])
                c5.metric("待处理", s["need_process"])
                if res["alerts"]:
                    with st.container(border=True):
                        st.markdown("**预警列表**")
                        for a in res["alerts"]:
                            st.markdown(
                                f'{label_badge(a["label"])}　{sev_badge(a["severity"])}　'
                                f'{a["text"]}',
                                unsafe_allow_html=True)
                            render_alerts(a)
                with st.container(border=True):
                    st.markdown("**全部结果**")
                    df = pd.DataFrame([{
                        "评价": r["text"], "分类": r["label"],
                        "满意度": score_to_rating(r["score"]),
                        "置信度": f"{round((1 - r['error_prob']) * 100, 1)}%",
                        "处理等级": SEV_TEXT.get(r["severity"], r["severity"]),
                        "需处理": "是" if r["need_process"] else "否",
                        "需核实": "是" if r["need_review"] else "否",
                    } for r in res["results"]])
                    st.dataframe(df, use_container_width=True)

# ============================================================
# 3) 差评分析
# ============================================================
elif nav == "差评分析":
    top_k = st.slider("问题词数量", 5, 50, 20)
    report, err = api_get(base, "/negative/keywords", {"top_k": top_k})
    if err:
        st.error(f"请求失败：{err}")
    else:
        st.caption(f"基于 {report.get('n_negative')} 条差评 / {report.get('n_overall')} 条整体评价"
                   f" · 生成于 {report.get('generated_at')}")
        kws = report.get("top_keywords", [])
        if kws:
            with st.container(border=True):
                st.markdown("**高频问题词**")
                kw_df = pd.DataFrame(kws)
                st.bar_chart(kw_df.set_index("word")["count"], height=300)
                st.dataframe(kw_df, use_container_width=True)
        aspects = report.get("aspects", [])
        if aspects:
            with st.container(border=True):
                st.markdown("**问题维度分布**")
                a_df = pd.DataFrame([{"维度": a["aspect"], "占比": a["ratio"], "次数": a["count"]}
                                     for a in aspects]).set_index("维度")
                st.bar_chart(a_df["占比"], height=280)
                for a in aspects:
                    words = " / ".join(w["word"] for w in a["words"])
                    st.markdown(f"**{a['aspect']}（{a['ratio']*100:.1f}%）**：{words}")

# ============================================================
# 4) 购买建议
# ============================================================
elif nav == "购买建议":
    with st.container(border=True):
        st.markdown("**生成购买建议**")
        st.caption("输入一条评价，系统给出是否值得购买的建议")
        adv_text = st.text_area(
            "评价内容", "味道很赞，配送很快，分量很足，值得回购！", height=110,
            label_visibility="collapsed")
        adv_btn = st.button("生成建议", type="primary")
    if adv_btn:
        if not adv_text.strip():
            st.info("请输入评价内容")
        else:
            res, err = api_post(base, "/advice", {"text": adv_text})
            if err:
                st.error(f"请求失败：{err}")
            else:
                rating = res["rating"]
                confidence = round(res["confidence"] * 100, 1)
                c1, c2, c3 = st.columns(3)
                c1.metric("购买建议", res["advice"])
                c2.metric("置信度", f"{confidence}%")
                c3.metric("推荐星级", f"{rating} 分")
                st.markdown(
                    f'{star_html(rating)}　<span style="color:#666;">{rating} / 5.0</span>　'
                    f'{badge(res["advice"], ADVICE_COLOR.get(res["advice"], "#666"))}',
                    unsafe_allow_html=True)
                if res["reasons"]:
                    st.markdown(f"**关注点：** " + " · ".join(res["reasons"]))
                st.markdown(f"**结论：** {res['summary']}")
                with st.container(border=True):
                    st.markdown("**建议倾向**")
                    probs = pd.DataFrame(
                        {"建议": list(res["probs"].keys()),
                         "概率": list(res["probs"].values())}).set_index("建议")
                    st.bar_chart(probs, height=240)

# ============================================================
# 5) 待核实评价（人工复核）
# ============================================================
elif nav == "待核实评价":
    pending, err = api_get(base, "/reviews/pending", {"limit": 50})
    if err:
        st.error(f"请求失败：{err}")
    else:
        with st.container(border=True):
            st.markdown("**待核实评价**")
            st.caption(f"系统存疑的评价，请人工确认真实分类（共 {pending['count']} 条）")
            if not pending["count"]:
                st.success("当前没有待核实评价 🎉")
            else:
                options = {f"#{it['id']} · {it['text'][:32]}": it for it in pending["items"]}
                choice = st.radio("选择一条评价", list(options.keys()),
                                  label_visibility="collapsed")
                item = options[choice]
                try:
                    probs = json.loads(item["probs"])
                except Exception:
                    probs = item["probs"]
                rating = score_to_rating(item["score"])
                st.markdown(
                    f'{label_badge(item["pred_label"])}　{sev_badge(item["severity"])}　'
                    f'{star_html(rating)}　<span style="color:#666;">{rating} / 5.0</span>',
                    unsafe_allow_html=True)
                st.write(f"**评价内容：** {item['text']}")
                st.write(f"**系统分类：** {item['pred_label']}　|　满意度 {rating} 分　|　"
                         f"置信度 {round((1 - item['error_prob']) * 100, 1)}%")
                st.write(f"**分类倾向：** {probs}")
                c1, c2 = st.columns([1, 1])
                with c1:
                    label = st.selectbox("确认真实分类", ["好评", "中评", "差评"])
                    if st.button("提交核实结果", type="primary"):
                        _, e = api_post(base, f"/reviews/{item['id']}/label",
                                        {"true_label": label})
                        if e:
                            st.error(f"提交失败：{e}")
                        else:
                            st.success("已提交，结果将用于后续优化识别")
                            st.rerun()
                with c2:
                    if st.button("标记为已处理"):
                        _, e = api_post(base, f"/reviews/{item['id']}/process", {})
                        if e:
                            st.error(f"操作失败：{e}")
                        else:
                            st.success("已标记处理")
                            st.rerun()

# ============================================================
# 6) 数据统计
# ============================================================
else:
    stats, err = api_get(base, "/stats")
    if err:
        st.error(f"请求失败：{err}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("评价总数", stats["total"])
        c2.metric("待核实", stats["pending_review"])
        c3.metric("待处理", stats["pending_process"])
        avg = stats["avg_score"]
        c4.metric("平均满意度", f"{score_to_rating(avg)} 分" if avg is not None else "—")
        by_label = stats.get("by_label") or {}
        if by_label:
            with st.container(border=True):
                st.markdown("**分类分布**")
                df = pd.DataFrame({"类别": list(by_label.keys()),
                                   "数量": list(by_label.values())}).set_index("类别")
                st.bar_chart(df, height=300)
        with st.expander("查看明细"):
            st.json(stats, expanded=False)

# ---------------- 页脚 ----------------
st.markdown(
    '<div style="text-align:center;color:#c0c0c0;font-size:0.78rem;margin-top:20px;">'
    '本页面仅供商家运营参考，最终处理结果以实际情况为准</div>',
    unsafe_allow_html=True)
