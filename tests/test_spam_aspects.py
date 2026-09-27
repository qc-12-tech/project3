"""方面词挖掘与刷评论检测的工具函数测试（纯 CPU，不依赖已生成的产物）。"""
import numpy as np

from src.aspects import match_aspects, split_clauses
from src.lexicon import ASPECTS
from src.spam import AD_RE, WEIGHTS, SUSPECT_THRESHOLD, _sizes


def test_split_clauses_filters_short_pieces():
    cl = split_clauses("剧情很棒，但是节奏太拖沓了！特效呢？好。")
    assert "剧情很棒" in cl and "但是节奏太拖沓了" in cl
    assert all(len(c) >= 3 for c in cl)


def test_match_aspects_hits_expected_dimensions():
    assert "剧情故事" in match_aspects("剧本写得非常敷衍")
    assert "画面特效" in match_aspects("特效很震撼，画面精美")
    assert "配乐音效" in match_aspects("BGM 太好听了")
    assert match_aspects("今天天气不错") == []


def test_aspect_lexicon_has_no_duplicate_keywords_across_dimensions():
    seen = {}
    for name, kws in ASPECTS.items():
        for kw in kws:
            assert kw not in seen or seen[kw] == name, f"{kw} 同时属于 {seen.get(kw)} 与 {name}"


def test_group_sizes_counts_correctly():
    keys = np.array([1, 1, 1, 2, 2, 3])
    assert list(_sizes(keys)) == [3, 3, 3, 2, 2, 1]


def test_ad_regex_matches_spam_but_not_normal_review():
    assert AD_RE.search("加微信看资源")
    assert AD_RE.search("VX：abc123")
    assert AD_RE.search("私聊我领取福利")
    assert AD_RE.search("www.xxx.com")
    assert not AD_RE.search("剧情很棒，演员表演在线，结局有点仓促")


def test_spam_weights_are_bounded_and_threshold_reachable():
    assert all(0 < w <= 1 for w in WEIGHTS.values())
    assert 0 < SUSPECT_THRESHOLD <= 1
    # 单一强信号（完全重复 / 广告）应足以判为疑似
    assert WEIGHTS["dup_same_movie"] >= SUSPECT_THRESHOLD
    assert WEIGHTS["ad"] >= SUSPECT_THRESHOLD
