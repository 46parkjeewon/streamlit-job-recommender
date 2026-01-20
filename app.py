# -*- coding: utf-8 -*-
import os
import json
import ast

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =========================
# 경로 (현재 폴더 기준)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FINAL_CSV = os.path.join(BASE_DIR, "final_df_2.csv")
PRECOMP = os.path.join(BASE_DIR, "precomputed_job_map.parquet")
SCORES = os.path.join(BASE_DIR, "llm_6skills_cache.json")

SKILL_AXES = ["Python/R", "SQL/DB", "시각화", "ML/AI", "도메인지식", "협업"]

# =========================
# ✅ FINAL MAP 반영 (cluster -> name)
# =========================
CLUSTER_NAME_MAP = {
    0: "엔지니어링",
    1: "BI/시각화",
    2: "ML/AI",
    3: "리서치",
}

def cluster_label(cluster_id) -> str:
    try:
        c = int(cluster_id)
    except Exception:
        return f"Cluster {cluster_id}"
    nm = CLUSTER_NAME_MAP.get(c, "미분류")
    return f"Cluster {c} - {nm}"


def safe_list(x):
    if x is None:
        return []
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return [str(a).strip() for a in v if str(a).strip()]
        except Exception:
            return [s]
    return [s]


def clamp_score(v):
    try:
        v = int(v)
    except Exception:
        v = 3
    return max(1, min(5, v))


@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(PRECOMP):
        raise FileNotFoundError(f"missing: {PRECOMP}")
    if not os.path.exists(SCORES):
        raise FileNotFoundError(f"missing: {SCORES}")
    if not os.path.exists(FINAL_CSV):
        raise FileNotFoundError(f"missing: {FINAL_CSV}")

    pre_df = pd.read_parquet(PRECOMP).fillna("")
    final_df = pd.read_csv(FINAL_CSV).fillna("")

    with open(SCORES, "r", encoding="utf-8") as f:
        score_map = json.load(f)

    # 필수 컬럼 체크/보정
    for c in ["uid", "회사명", "세부직무명", "cluster", "x", "y"]:
        if c not in pre_df.columns:
            pre_df[c] = ""

    # cluster int 강제 (✅ 이름 맵 적용 안정)
    pre_df["cluster"] = pd.to_numeric(pre_df["cluster"], errors="coerce").fillna(0).astype(int)

    # x,y 숫자 강제(시각화 안전)
    pre_df["x"] = pd.to_numeric(pre_df["x"], errors="coerce")
    pre_df["y"] = pd.to_numeric(pre_df["y"], errors="coerce")

    # final_df_2 컬럼 보정
    for c in ["stem", "회사명", "세부직무명", "담당업무", "우대사항", "스킬"]:
        if c not in final_df.columns:
            final_df[c] = ""

    return pre_df, final_df, score_map


def build_scores_df(pre_df: pd.DataFrame, score_map: dict) -> pd.DataFrame:
    """
    pre_df(uid, cluster, 회사명, 세부직무명, x,y...) + score_map(uid -> 6역량) 결합
    결과: uid별 6역량 점수 + 메타(클러스터/세부직무/회사명)
    """
    rows = []
    for _, r in pre_df.iterrows():
        uid = str(r.get("uid", "")).strip()
        if not uid:
            continue
        s = score_map.get(uid)
        if not isinstance(s, dict):
            continue

        item = {
            "uid": uid,
            "cluster": int(r.get("cluster", 0)),
            "cluster_name": CLUSTER_NAME_MAP.get(int(r.get("cluster", 0)), "미분류"),
            "회사명": r.get("회사명", ""),
            "세부직무명": r.get("세부직무명", ""),
        }
        for k in SKILL_AXES:
            item[k] = clamp_score(s.get(k, 3))
        rows.append(item)

    sdf = pd.DataFrame(rows)
    if len(sdf) == 0:
        return pd.DataFrame(columns=["uid", "cluster", "cluster_name", "회사명", "세부직무명"] + SKILL_AXES)

    return sdf


def radar_from_mean(title: str, mean_scores: pd.Series) -> go.Figure:
    vals = [float(mean_scores.get(k, 0.0)) for k in SKILL_AXES]
    vals2 = vals + [vals[0]]
    axes2 = SKILL_AXES + [SKILL_AXES[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals2, theta=axes2, fill="toself"))
    fig.update_layout(
        template="plotly_white",
        height=420,
        title=title,
        polar=dict(radialaxis=dict(range=[0, 5], tickvals=[1, 2, 3, 4, 5])),
        showlegend=False,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def plot_umap(pre_df: pd.DataFrame) -> go.Figure:
    dfp = pre_df.dropna(subset=["x", "y"]).copy()
    # ✅ 클러스터명 컬럼 추가
    dfp["cluster_label"] = dfp["cluster"].apply(cluster_label)

    fig = px.scatter(
        dfp,
        x="x",
        y="y",
        color="cluster_label",  # ✅ 클러스터명으로 색상 분리
        hover_data={
            "회사명": True,
            "세부직무명": True,
            "cluster_label": True,
        },
        opacity=0.85,
    )
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(
        template="plotly_white",
        height=520,
        title="세부직무 시각화",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def topn_mean(sdf: pd.DataFrame, n: int) -> pd.Series:
    """
    Top-N 공고: 6역량 합이 높은 uid 상위 N개 평균
    """
    if len(sdf) == 0:
        return pd.Series({k: 0.0 for k in SKILL_AXES})
    tmp = sdf.copy()
    tmp["total"] = tmp[SKILL_AXES].sum(axis=1)
    tmp = tmp.sort_values("total", ascending=False).head(n)
    return tmp[SKILL_AXES].mean()


def main():
    st.set_page_config(page_title="채용공고 평균 역량 분석", layout="wide")
    st.title("채용공고 평균 역량 분석")

    pre_df, final_df, score_map = load_data()
    sdf = build_scores_df(pre_df, score_map)

    if len(sdf) == 0:
        st.error("llm_6skills_cache.json과 parquet(uid)이 매칭되지 않아 평균을 계산할 수 없어.")
        st.stop()

    # -------------------------
    # 0) 전체 시각화(보기용)
    # -------------------------
    with st.expander("세부직무 시각화(UMAP) 보기", expanded=True):
        st.plotly_chart(plot_umap(pre_df), use_container_width=True)

    # -------------------------
    # 1) Cluster 선택 (✅ 클러스터명 반영)
    # -------------------------
    st.subheader("클러스터 평균 역량")

    clusters = sorted(sdf["cluster"].dropna().unique().tolist())
    if len(clusters) == 0:
        st.error("cluster 컬럼을 찾을 수 없어.")
        st.stop()

    # ✅ 사용자에게 보여줄 옵션(label)과 실제 값(cluster id) 분리
    options = {cluster_label(c): int(c) for c in clusters}
    selected_label = st.selectbox("클러스터 선택", list(options.keys()), index=0)
    c = options[selected_label]  # 실제 cluster id
    cname = CLUSTER_NAME_MAP.get(int(c), "미분류")

    sdf_c = sdf[sdf["cluster"] == c]
    c_mean = sdf_c[SKILL_AXES].mean()

    left, right = st.columns([1.2, 1.0])
    with left:
        st.plotly_chart(radar_from_mean(f"{cluster_label(c)} 평균 역량", c_mean), use_container_width=True)
    with right:
        st.write("표본 수(공고):", len(sdf_c))
        st.dataframe(
            pd.DataFrame({"역량": SKILL_AXES, "평균점수": [round(float(c_mean[k]), 2) for k in SKILL_AXES]}),
            use_container_width=True,
            hide_index=True,
        )

    # -------------------------
    # 2) 세부직무 평균(선택한 cluster 내부)
    # -------------------------
    st.subheader("세부직무 평균 역량")
    subjobs = sorted(sdf_c["세부직무명"].dropna().unique().tolist())
    if len(subjobs) == 0:
        st.warning("이 클러스터 내 세부직무명이 비어있어.")
        st.stop()

    sj = st.selectbox("세부직무 선택(클러스터 내부)", subjobs, index=0)
    sdf_sj = sdf_c[sdf_c["세부직무명"] == sj]
    sj_mean = sdf_sj[SKILL_AXES].mean()

    left2, right2 = st.columns([1.2, 1.0])
    with left2:
        st.plotly_chart(radar_from_mean(f"{sj} 평균 역량 ({cluster_label(c)})", sj_mean), use_container_width=True)
    with right2:
        st.write("표본 수(공고):", len(sdf_sj))
        st.dataframe(
            pd.DataFrame({"역량": SKILL_AXES, "평균점수": [round(float(sj_mean[k]), 2) for k in SKILL_AXES]}),
            use_container_width=True,
            hide_index=True,
        )

    # -------------------------
    # 3) Top-N 공고 평균(선택한 cluster/세부직무 내부)
    # -------------------------
    st.subheader("Top-N 공고 평균 역량")
    n = st.slider("N 선택", min_value=3, max_value=30, value=10, step=1)

    tn_mean = topn_mean(sdf_sj, n)
    st.plotly_chart(
        radar_from_mean(f"Top-{n} 공고 평균 역량 ({sj} / {cluster_label(c)})", tn_mean),
        use_container_width=True
    )

    # Top-N 리스트
    tmp = sdf_sj.copy()
    tmp["total"] = tmp[SKILL_AXES].sum(axis=1)
    tmp = tmp.sort_values("total", ascending=False).head(n)

    st.caption("※ 아래 Top-N 표는 LLM 역량 합이 높은 순서입니다.")
    show_rows = []
    for _, r in tmp.iterrows():
        company = r["회사명"]
        subjob = r["세부직무명"]
        cand = final_df[(final_df["회사명"] == company) & (final_df["세부직무명"] == subjob)]
        stem = cand["stem"].iloc[0] if len(cand) > 0 else ""
        show_rows.append({
            "stem": stem,
            "회사명": company,
            "세부직무명": subjob,
            "클러스터": cluster_label(c),
            "총점": int(r["total"]),
            **{k: int(r[k]) for k in SKILL_AXES},
        })

    st.dataframe(pd.DataFrame(show_rows), use_container_width=True)

    st.divider()

    # -------------------------
    # (선택) 내 스펙 입력 → 평균 대비 비교
    # -------------------------
    st.subheader("내 스펙과 비교")
    st.write("내 역량 점수(1~5)를 직접 입력해서 평균과 비교할 수 있어.")
    cols = st.columns(6)
    my = {}
    for i, k in enumerate(SKILL_AXES):
        with cols[i]:
            my[k] = st.slider(k, 1, 5, 3)

    compare_target = st.radio("비교 대상", ["클러스터 평균", "세부직무 평균", "Top-N 평균"], horizontal=True)
    if compare_target == "클러스터 평균":
        base = c_mean
        title = f"내 스펙 vs {cluster_label(c)} 평균"
    elif compare_target == "세부직무 평균":
        base = sj_mean
        title = f"내 스펙 vs {sj} 평균 ({cluster_label(c)})"
    else:
        base = tn_mean
        title = f"내 스펙 vs Top-{n} 평균 ({sj} / {cluster_label(c)})"

    vals_me = [float(my[k]) for k in SKILL_AXES]
    vals_base = [float(base.get(k, 0.0)) for k in SKILL_AXES]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_base + [vals_base[0]],
        theta=SKILL_AXES + [SKILL_AXES[0]],
        fill="toself",
        name="평균"
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals_me + [vals_me[0]],
        theta=SKILL_AXES + [SKILL_AXES[0]],
        fill="toself",
        name="나"
    ))
    fig.update_layout(
        template="plotly_white",
        height=420,
        title=title,
        polar=dict(radialaxis=dict(range=[0, 5], tickvals=[1, 2, 3, 4, 5])),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()

