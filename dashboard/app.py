"""
app.py  —  Player Data 2026/27 · Football Analytics Dashboard
Streamlit web application — run with:  streamlit run dashboard/app.py
"""

import os, sys, shutil, glob as _glob
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    load_raw_data, clean_data, kpi_summary, league_summary,
    position_summary, top_players, nation_summary,
    gk_summary, normalise_radar, LEAGUE_COLORS,
)

# ── Paths (relative to project root, works from any cwd) ─────────────────────
_HERE      = os.path.dirname(__file__)
_ROOT      = os.path.join(_HERE, "..")
DATA_PATH  = os.path.join(_ROOT, "data",    "Player_data.csv")
MODEL_PATH = os.path.join(_ROOT, "outputs", "Player_data_model.joblib")

# ── Auto-update from Kaggle ───────────────────────────────────────────────────
def _try_kaggle_update() -> str:
    """
    Download the latest dataset from Kaggle using kagglehub.
    Returns a status string shown in the sidebar.
    Falls back silently if kagglehub is not installed or credentials
    are not configured.
    """
    try:
        import kagglehub
        dl_path = kagglehub.dataset_download(
            "hubertsidorowicz/football-players-stats-2026-2027"
        )
        # Find the light CSV in the downloaded folder
        candidates = _glob.glob(
            os.path.join(dl_path, "**", "*light*.csv"), recursive=True
        ) or _glob.glob(
            os.path.join(dl_path, "**", "*.csv"), recursive=True
        )
        if not candidates:
            return "kaggle: no CSV found in download"
        src = candidates[0]
        os.makedirs(os.path.join(_ROOT, "data"), exist_ok=True)
        shutil.copy2(src, DATA_PATH)
        return f"kaggle: updated from {os.path.basename(src)}"
    except ImportError:
        return "kagglehub not installed (pip install kagglehub)"
    except Exception as e:
        return f"kaggle update skipped: {e}"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Player Data Analytics 2026/27",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0e1117; }
  section[data-testid="stSidebar"] { background-color: #161b22; }
  section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

  div[data-testid="metric-container"] {
      background: linear-gradient(135deg,#1c2333 0%,#21262d 100%);
      border: 1px solid #30363d; border-radius: 12px; padding: 16px 20px;
  }
  div[data-testid="metric-container"] label {
      color:#8b949e !important; font-size:.78rem !important;
      letter-spacing:.06em; text-transform:uppercase;
  }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
      color:#f0f6fc !important; font-size:1.85rem !important; font-weight:700;
  }

  .section-hdr {
      font-size:1.05rem; font-weight:600; color:#58a6ff;
      border-left:4px solid #1f6feb; padding-left:10px;
      margin:18px 0 12px 0; letter-spacing:.03em;
  }
  .tab-title { font-size:1.3rem; font-weight:700; color:#f0f6fc; margin-bottom:2px; }
  .sub-txt   { color:#8b949e; font-size:.85rem; margin-bottom:14px; }

  div[data-testid="stDataFrame"] { border:1px solid #30363d; border-radius:10px; }
  footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & cleaning data …")
def get_data():
    raw = load_raw_data(DATA_PATH)
    return clean_data(raw)

df_full = get_data()

# ── Model loading (optional, cached) ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    if os.path.exists(MODEL_PATH):
        import joblib
        return joblib.load(MODEL_PATH)
    return None

model_bundle = get_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ Filters")
    st.markdown("---")

    all_leagues = sorted(df_full["League"].dropna().unique())
    sel_leagues = st.multiselect("🏆 League", all_leagues, default=all_leagues)

    all_pos = ["All", "FW", "MF", "DF", "GK"]
    sel_pos = st.selectbox("🎯 Position", all_pos)

    age_min, age_max = int(df_full["Age"].min()), int(df_full["Age"].max())
    age_range = st.slider("🎂 Age Range", age_min, age_max, (age_min, age_max))

    min_mins = st.slider("⏱ Min. Minutes Played", 0, 3000, 0, step=90)

    # ── Kaggle live-update section ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔄 Data Update")

    # Show last-modified time of the local CSV
    if os.path.exists(DATA_PATH):
        import datetime
        mtime = os.path.getmtime(DATA_PATH)
        last_updated = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        st.markdown(
            f"<div style='color:#8b949e;font-size:.74rem;'>Last updated: {last_updated}</div>",
            unsafe_allow_html=True,
        )

    if st.button("⬇️ Pull latest from Kaggle", use_container_width=True):
        with st.spinner("Downloading from Kaggle..."):
            status = _try_kaggle_update()
        if status.startswith("kaggle: updated"):
            st.success(status)
            st.cache_data.clear()   # force data reload
            st.rerun()
        else:
            st.warning(status)

    st.markdown("---")
    st.markdown(
        "<div style='color:#8b949e;font-size:.74rem;'>"
        "Data: FBref · Season 2026/27<br>Top 5 European Leagues</div>",
        unsafe_allow_html=True,
    )

# ── Apply filters ─────────────────────────────────────────────────────────────
df = df_full[df_full["League"].isin(sel_leagues)].copy()
if sel_pos != "All":
    df = df[df["PrimaryPos"] == sel_pos]
df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]
df = df[df["Min"].fillna(0) >= min_mins]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#f0f6fc;font-size:2rem;font-weight:800;margin-bottom:0;'>"
    "⚽ Player Data Analytics — 2026/27</h1>"
    "<p style='color:#8b949e;font-size:.88rem;margin-top:4px;'>"
    "Europe's Top 5 Leagues · FBref · Season in Progress</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── KPI Row ───────────────────────────────────────────────────────────────────
kpi = kpi_summary(df)
cols = st.columns(8)
labels = [
    ("👤 Players",   kpi["total_players"]),
    ("⚽ Goals",     kpi["total_goals"]),
    ("🎯 Assists",   kpi["total_assists"]),
    ("🎂 Avg Age",   kpi["avg_age"]),
    ("🌍 Nations",   kpi["total_nations"]),
    ("🏟 Clubs",     kpi["total_clubs"]),
    ("🏆 Leagues",   kpi["total_leagues"]),
    ("🎯 Avg SoT%",  f"{kpi['avg_sot_pct']}%"),
]
for col, (label, value) in zip(cols, labels):
    col.metric(label, f"{value:,}" if isinstance(value, int) else value)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 League Overview",
    "🔫 Attacking",
    "🛡 Defensive",
    "🧤 Goalkeeping",
    "🌍 Nations & Age",
    "🤖 ML Model",
    "🔍 Player Search",
])
(t_league, t_attack, t_defence, t_gk, t_nations, t_ml, t_search) = tabs

_DARK = dict(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
_AX   = dict(color="#8b949e", gridcolor="#21262d")
_L_COLORS = {
    "Premier League":"#58a6ff","La Liga":"#f85149",
    "Bundesliga":"#e3b341","Serie A":"#3fb950","Ligue 1":"#d2a8ff",
}

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · League Overview
# ─────────────────────────────────────────────────────────────────────────────
with t_league:
    st.markdown('<div class="tab-title">League Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-txt">Aggregated statistics per competition.</div>', unsafe_allow_html=True)

    lg = league_summary(df)

    # Goals & Assists grouped bar
    st.markdown('<div class="section-hdr">Goals & Assists</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_bar(x=lg["League"], y=lg["Goals"],   name="Goals",   marker_color="#58a6ff")
    fig.add_bar(x=lg["League"], y=lg["Assists"],  name="Assists", marker_color="#3fb950")
    fig.update_layout(**_DARK, barmode="group", height=340,
                      margin=dict(t=20,b=0),
                      xaxis=dict(title=None,**_AX),
                      yaxis=dict(title="Count",**_AX),
                      legend=dict(orientation="h",y=1.08))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-hdr">Player Distribution</div>', unsafe_allow_html=True)
        fig2 = px.pie(lg, names="League", values="Players",
                      color_discrete_sequence=["#58a6ff","#f85149","#e3b341","#3fb950","#d2a8ff"],
                      hole=0.42)
        fig2.update_layout(**_DARK, height=300, margin=dict(t=10,b=10),
                           legend=dict(font=dict(size=11)))
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<div class="section-hdr">Disciplinary Cards</div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_bar(x=lg["League"], y=lg["YellowCards"], name="Yellow", marker_color="#e3b341")
        fig3.add_bar(x=lg["League"], y=lg["RedCards"],    name="Red",    marker_color="#f85149")
        fig3.update_layout(**_DARK, barmode="group", height=300,
                           margin=dict(t=10,b=0),
                           xaxis=dict(title=None,**_AX),
                           yaxis=dict(title="Cards",**_AX),
                           legend=dict(orientation="h",y=1.08))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-hdr">Position Distribution Heatmap</div>', unsafe_allow_html=True)
    pos_h = (df.groupby(["League","PrimaryPos"], observed=True).size()
               .reset_index(name="Count")
               .pivot(index="League", columns="PrimaryPos", values="Count")
               .fillna(0))
    fig4 = px.imshow(pos_h, color_continuous_scale="Blues", text_auto=True, aspect="auto")
    fig4.update_layout(**_DARK, height=260, margin=dict(t=10,b=0),
                       coloraxis_showscale=False,
                       xaxis=dict(title=None,color="#8b949e"),
                       yaxis=dict(title=None,color="#8b949e"))
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="section-hdr">Avg Goals per 90 by League</div>', unsafe_allow_html=True)
    fig5 = px.bar(lg, x="League", y="AvgGoals90",
                  color="League", color_discrete_map=_L_COLORS,
                  text=lg["AvgGoals90"].round(3))
    fig5.update_layout(**_DARK, height=290, margin=dict(t=10,b=0),
                       showlegend=False,
                       xaxis=dict(title=None,**_AX),
                       yaxis=dict(title="Gls/90",**_AX))
    st.plotly_chart(fig5, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · Attacking
# ─────────────────────────────────────────────────────────────────────────────
with t_attack:
    st.markdown('<div class="tab-title">Attacking Statistics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-txt">Goals, assists, shots & efficiency — outfield players.</div>', unsafe_allow_html=True)

    df_att = df[df["PrimaryPos"] != "GK"].copy()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-hdr">Top 15 Scorers</div>', unsafe_allow_html=True)
        ts = top_players(df_att, "Gls", 15)
        fig = px.bar(ts, x="Gls", y="Player", orientation="h",
                     color="League", color_discrete_map=_L_COLORS,
                     text="Gls", hover_data=["Squad","Ast"])
        fig.update_layout(**_DARK, height=430, margin=dict(t=10,b=0),
                          yaxis=dict(autorange="reversed",title=None,color="#8b949e"),
                          xaxis=dict(title="Goals",**_AX), legend=dict(font=dict(size=10)))
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-hdr">Top 15 Assisters</div>', unsafe_allow_html=True)
        ta = top_players(df_att, "Ast", 15)
        fig = px.bar(ta, x="Ast", y="Player", orientation="h",
                     color="League", color_discrete_map=_L_COLORS,
                     text="Ast", hover_data=["Squad","Gls"])
        fig.update_layout(**_DARK, height=430, margin=dict(t=10,b=0),
                          yaxis=dict(autorange="reversed",title=None,color="#8b949e"),
                          xaxis=dict(title="Assists",**_AX), legend=dict(font=dict(size=10)))
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-hdr">Goals vs Shots (min. 5 shots) · Bubble = Minutes</div>', unsafe_allow_html=True)
    d_s = df_att[(df_att["Sh"] >= 5) & df_att["Gls"].notna()]
    fig = px.scatter(d_s, x="Sh", y="Gls", color="League", size="Min",
                     size_max=22, opacity=0.72, hover_name="Player",
                     color_discrete_map=_L_COLORS,
                     hover_data={"Squad":True,"SoT":True,"90s":True,"Min":False})
    fig.update_layout(**_DARK, height=400, margin=dict(t=10,b=0),
                      xaxis=dict(title="Total Shots",**_AX),
                      yaxis=dict(title="Goals",**_AX))
    st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-hdr">Avg SoT% by Position</div>', unsafe_allow_html=True)
        sp = (df_att[df_att["Sh"] >= 5].groupby("PrimaryPos", observed=True)["SoT%"]
              .mean().reset_index().round(1))
        fig = px.bar(sp, x="PrimaryPos", y="SoT%",
                     color="PrimaryPos",
                     color_discrete_sequence=["#58a6ff","#3fb950","#f78166","#d2a8ff"],
                     text="SoT%")
        fig.update_layout(**_DARK, height=280, margin=dict(t=10,b=0),
                          showlegend=False,
                          xaxis=dict(title="Position",**_AX),
                          yaxis=dict(title="SoT%",**_AX))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown('<div class="section-hdr">Goals per Shot — Top 20 (min. 10 shots)</div>', unsafe_allow_html=True)
        dg = df_att[(df_att["Sh"] >= 10) & df_att["G/Sh"].notna()].nlargest(20,"G/Sh")
        fig = px.bar(dg, x="G/Sh", y="Player", orientation="h",
                     color="G/Sh", color_continuous_scale="Blues", text=dg["G/Sh"].round(2))
        fig.update_layout(**_DARK, height=440, margin=dict(t=10,b=0),
                          yaxis=dict(autorange="reversed",title=None,color="#8b949e"),
                          xaxis=dict(title="G/Sh",**_AX), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Age-group breakdown
    st.markdown('<div class="section-hdr">Goals & Assists by Age Group</div>', unsafe_allow_html=True)
    from utils import age_group_stats
    ag = age_group_stats(df_att)
    fig = go.Figure()
    fig.add_bar(x=ag["AgeGroup"].astype(str), y=ag["Goals"],   name="Goals",   marker_color="#58a6ff")
    fig.add_bar(x=ag["AgeGroup"].astype(str), y=ag["Assists"],  name="Assists", marker_color="#3fb950")
    fig.update_layout(**_DARK, barmode="group", height=300, margin=dict(t=10,b=0),
                      xaxis=dict(title="Age Group",**_AX),
                      yaxis=dict(title="Count",**_AX),
                      legend=dict(orientation="h",y=1.08))
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · Defensive
# ─────────────────────────────────────────────────────────────────────────────
with t_defence:
    st.markdown('<div class="tab-title">Defensive Statistics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-txt">Tackles, interceptions, fouls and disciplinary data.</div>', unsafe_allow_html=True)

    df_def = df[df["PrimaryPos"] != "GK"].copy()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-hdr">Top 15 — Tackles Won</div>', unsafe_allow_html=True)
        tt = top_players(df_def, "TklW", 15)
        fig = px.bar(tt, x="TklW", y="Player", orientation="h",
                     color="League", color_discrete_map=_L_COLORS, text="TklW")
        fig.update_layout(**_DARK, height=420, margin=dict(t=10,b=0),
                          yaxis=dict(autorange="reversed",title=None,color="#8b949e"),
                          xaxis=dict(title="Tackles Won",**_AX), legend=dict(font=dict(size=10)))
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-hdr">Top 15 — Interceptions</div>', unsafe_allow_html=True)
        ti = top_players(df_def, "Int", 15)
        fig = px.bar(ti, x="Int", y="Player", orientation="h",
                     color="League", color_discrete_map=_L_COLORS, text="Int")
        fig.update_layout(**_DARK, height=420, margin=dict(t=10,b=0),
                          yaxis=dict(autorange="reversed",title=None,color="#8b949e"),
                          xaxis=dict(title="Interceptions",**_AX), legend=dict(font=dict(size=10)))
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-hdr">Tackles Won vs Interceptions (DF & MF · min. 3 games)</div>', unsafe_allow_html=True)
    dds = df_def[df_def["PrimaryPos"].isin(["DF","MF"]) & (df_def["Min"].fillna(0) >= 270)]
    fig = px.scatter(dds, x="TklW", y="Int", color="PrimaryPos",
                     hover_name="Player", opacity=0.7,
                     color_discrete_map={"DF":"#58a6ff","MF":"#3fb950"},
                     hover_data={"Squad":True,"League":True,"Min":True})
    fig.update_layout(**_DARK, height=370, margin=dict(t=10,b=0),
                      xaxis=dict(title="Tackles Won",**_AX),
                      yaxis=dict(title="Interceptions",**_AX))
    st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-hdr">Fouls Drawn vs Committed (avg) by League</div>', unsafe_allow_html=True)
        fl = (df_def.groupby("League", observed=True)
              .agg(Drawn=("Fld","mean"), Committed=("Fls","mean"))
              .reset_index().round(2))
        fig = go.Figure()
        fig.add_bar(x=fl["League"], y=fl["Drawn"],     name="Drawn",     marker_color="#3fb950")
        fig.add_bar(x=fl["League"], y=fl["Committed"], name="Committed", marker_color="#f85149")
        fig.update_layout(**_DARK, barmode="group", height=290, margin=dict(t=10,b=0),
                          xaxis=dict(title=None,**_AX),
                          yaxis=dict(title="Avg / Player",**_AX),
                          legend=dict(orientation="h",y=1.08))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown('<div class="section-hdr">Yellow Cards by Position</div>', unsafe_allow_html=True)
        yc = df_def.groupby("PrimaryPos", observed=True)["CrdY"].sum().reset_index()
        fig = px.pie(yc, names="PrimaryPos", values="CrdY", hole=0.42,
                     color_discrete_sequence=["#e3b341","#58a6ff","#3fb950","#f85149"])
        fig.update_layout(**_DARK, height=290, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · Goalkeeping
# ─────────────────────────────────────────────────────────────────────────────
with t_gk:
    st.markdown('<div class="tab-title">Goalkeeping Statistics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-txt">Save rates, clean sheets and GA/90 for qualified keepers (≥ 270 min).</div>', unsafe_allow_html=True)

    df_gk = gk_summary(df, min_mins=min_mins if min_mins > 270 else 270)

    if df_gk.empty:
        st.info("No goalkeepers match the current filters with sufficient minutes.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-hdr">Top 15 — Save %</div>', unsafe_allow_html=True)
            sv = df_gk.nlargest(15, "Save%")
            fig = px.bar(sv, x="Save%", y="Player", orientation="h",
                         color="Save%", color_continuous_scale="Blues",
                         text=sv["Save%"].round(1))
            fig.update_layout(**_DARK, height=420, margin=dict(t=10,b=0),
                              yaxis=dict(autorange="reversed",title=None,color="#8b949e"),
                              xaxis=dict(title="Save%",**_AX), coloraxis_showscale=False)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-hdr">Lowest GA/90 — Top 15</div>', unsafe_allow_html=True)
            ga90 = df_gk.nsmallest(15, "GA90")
            fig = px.bar(ga90, x="GA90", y="Player", orientation="h",
                         color="GA90", color_continuous_scale="RdYlGn_r",
                         text=ga90["GA90"].round(2))
            fig.update_layout(**_DARK, height=420, margin=dict(t=10,b=0),
                              yaxis=dict(autorange="reversed",title=None,color="#8b949e"),
                              xaxis=dict(title="GA/90",**_AX), coloraxis_showscale=False)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-hdr">Save% vs GA/90 Scatter</div>', unsafe_allow_html=True)
        dgs = df_gk.dropna(subset=["Save%","GA90"])
        fig = px.scatter(dgs, x="GA90", y="Save%", color="League",
                         hover_name="Player",
                         hover_data={"Squad":True,"CS":True,"Saves":True},
                         color_discrete_map=_L_COLORS, size_max=14)
        fig.update_layout(**_DARK, height=370, margin=dict(t=10,b=0),
                          xaxis=dict(title="GA/90",**_AX),
                          yaxis=dict(title="Save%",**_AX))
        st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<div class="section-hdr">Clean Sheet % by League</div>', unsafe_allow_html=True)
            csl = (df_gk.dropna(subset=["CS%"])
                   .groupby("League", observed=True)["CS%"].mean()
                   .reset_index().round(1))
            fig = px.bar(csl, x="League", y="CS%", color="League",
                         color_discrete_map=_L_COLORS, text="CS%")
            fig.update_layout(**_DARK, height=280, margin=dict(t=10,b=0),
                              showlegend=False,
                              xaxis=dict(title=None,**_AX),
                              yaxis=dict(title="CS%",**_AX))
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.markdown('<div class="section-hdr">GK Stats Table (sorted by Save%)</div>', unsafe_allow_html=True)
            st.dataframe(df_gk.head(12).reset_index(drop=True), use_container_width=True, height=280)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 · Nations & Age
# ─────────────────────────────────────────────────────────────────────────────
with t_nations:
    st.markdown('<div class="tab-title">Nationality & Demographics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-txt">Player origins, age profiles, and squad composition.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-hdr">Top 20 Nations by Player Count</div>', unsafe_allow_html=True)
        nc = nation_summary(df, 20)
        fig = px.bar(nc, x="Players", y="Nation", orientation="h",
                     color="Players", color_continuous_scale="Blues", text="Players")
        fig.update_layout(**_DARK, height=460, margin=dict(t=10,b=0),
                          yaxis=dict(autorange="reversed",title=None,color="#8b949e"),
                          xaxis=dict(title="Players",**_AX), coloraxis_showscale=False)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-hdr">Age Distribution by League</div>', unsafe_allow_html=True)
        fig = px.box(df.dropna(subset=["Age"]), x="League", y="Age",
                     color="League", points="outliers",
                     color_discrete_map=_L_COLORS)
        fig.update_layout(**_DARK, height=320, margin=dict(t=10,b=0),
                          showlegend=False,
                          xaxis=dict(title=None,**_AX),
                          yaxis=dict(title="Age",**_AX))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-hdr">Age Histogram</div>', unsafe_allow_html=True)
        fig = px.histogram(df.dropna(subset=["Age"]), x="Age", nbins=30,
                           color_discrete_sequence=["#58a6ff"])
        fig.update_layout(**_DARK, height=190, margin=dict(t=5,b=0),
                          xaxis=dict(title="Age",**_AX),
                          yaxis=dict(title="Players",**_AX), bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-hdr">Top 15 Nations — Goals + Assists (stacked)</div>', unsafe_allow_html=True)
    ng = nation_summary(df, 15)
    fig = go.Figure()
    fig.add_bar(x=ng["Nation"], y=ng["Goals"],   name="Goals",   marker_color="#58a6ff")
    fig.add_bar(x=ng["Nation"], y=ng["Assists"],  name="Assists", marker_color="#3fb950")
    fig.update_layout(**_DARK, barmode="stack", height=320, margin=dict(t=10,b=0),
                      xaxis=dict(title=None,**_AX),
                      yaxis=dict(title="G+A",**_AX),
                      legend=dict(orientation="h",y=1.08))
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 · ML Model
# ─────────────────────────────────────────────────────────────────────────────
with t_ml:
    st.markdown('<div class="tab-title">🤖 Predictive Model — High Performer Classifier</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-txt">Random Forest trained to predict whether an outfield player is a '
        'high-performer (top-25% G+A per 90). Run <code>python dashboard/modeling.py</code> from the '
        'project root to train & save the model.</div>',
        unsafe_allow_html=True,
    )

    chart_dir = os.path.join(_ROOT, "outputs", "charts")

    def _show_chart(name, caption):
        p = os.path.join(chart_dir, name)
        if os.path.exists(p):
            st.image(p, caption=caption, use_container_width=True)

    if model_bundle is None:
        st.warning("⚠️ Model not yet trained. Run `python dashboard/modeling.py` from the project root, then refresh.")
    else:
        mb = model_bundle
        c1, c2, c3 = st.columns(3)
        roc = mb.get("roc_auc", 0)
        rep = mb.get("report", {})
        c1.metric("🏆 Best Model",    mb.get("model_name", "—"))
        c2.metric("📈 ROC-AUC",       f"{roc:.4f}")
        acc = rep.get("accuracy", 0)
        c3.metric("✅ Accuracy",       f"{acc:.2%}" if acc else "—")

        st.markdown('<div class="section-hdr">Model Performance Charts</div>', unsafe_allow_html=True)
        r1c1, r1c2 = st.columns(2)
        with r1c1: _show_chart("feature_importance.png", "Feature Importance")
        with r1c2: _show_chart("confusion_matrix.png",   "Confusion Matrix")
        r2c1, r2c2 = st.columns(2)
        with r2c1: _show_chart("roc_curve.png",          "ROC Curve")
        with r2c2: _show_chart("cv_comparison.png",      "CV Model Comparison")

        st.markdown('<div class="section-hdr">Classification Report</div>', unsafe_allow_html=True)
        if rep:
            rep_df = pd.DataFrame(rep).T.drop(columns=["support"], errors="ignore").round(3)
            st.dataframe(rep_df, use_container_width=True)

        # Live prediction
        st.markdown('<div class="section-hdr">🔮 Live Player Prediction</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-txt">Select any player and get the model\'s high-performer probability.</div>', unsafe_allow_html=True)
        player_opts = sorted(df_full["Player"].dropna().unique())
        sel_player  = st.selectbox("Select player", player_opts)
        if sel_player:
            row = df_full[df_full["Player"] == sel_player].iloc[0]
            from modeling import predict_single
            pred_res = predict_single(mb, row)
            prob = pred_res["probability"]
            pred = pred_res["prediction"]
            col_a, col_b = st.columns([1, 3])
            with col_a:
                st.metric("Probability", f"{prob:.1%}")
                st.metric("Predicted", "⭐ High Performer" if pred == 1 else "Standard")
            with col_b:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob * 100,
                    number={"suffix": "%", "font": {"color": "#f0f6fc"}},
                    gauge={
                        "axis":  {"range": [0, 100], "tickcolor": "#8b949e"},
                        "bar":   {"color": "#58a6ff"},
                        "steps": [
                            {"range": [0,  50], "color": "#21262d"},
                            {"range": [50, 75], "color": "#1a3a5e"},
                            {"range": [75,100], "color": "#1f6feb"},
                        ],
                        "threshold": {"line": {"color": "#f85149","width": 3}, "value": 75},
                    },
                ))
                gauge.update_layout(**_DARK, height=220, margin=dict(t=20,b=0,l=30,r=30))
                st.plotly_chart(gauge, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 · Player Search
# ─────────────────────────────────────────────────────────────────────────────
with t_search:
    st.markdown('<div class="tab-title">Player Search & Radar Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-txt">Search any player, view full stats, and compare up to 3 players on a radar chart.</div>', unsafe_allow_html=True)

    sc, _ = st.columns([2, 1])
    with sc:
        query = st.text_input("🔍 Search name", placeholder="e.g. Salah, Lewandowski, Mbappé…")

    df_s = df.copy()
    if query:
        df_s = df_s[df_s["Player"].str.contains(query, case=False, na=False)]

    display = ["Player","Nation","Pos","Squad","League","Age","MP","Starts","Min",
               "Gls","Ast","G+A","Sh","SoT","SoT%","TklW","Int","Fld","Fls","CrdY","CrdR"]
    display = [c for c in display if c in df_s.columns]

    st.markdown(f'<div class="sub-txt">Showing {len(df_s):,} player(s)</div>', unsafe_allow_html=True)
    st.dataframe(df_s[display].sort_values("Gls", ascending=False).reset_index(drop=True),
                 use_container_width=True, height=320)

    st.markdown('<div class="section-hdr">⚡ Radar Comparison (up to 3 players)</div>', unsafe_allow_html=True)
    all_names = sorted(df_full["Player"].dropna().unique())
    sel = st.multiselect("Select players to compare", all_names,
                         default=all_names[:2] if len(all_names) >= 2 else all_names,
                         max_selections=3)

    RADAR_METRICS = ["Gls","Ast","Sh","SoT","TklW","Int","Fld","CrdY"]
    RADAR_LABELS  = ["Goals","Assists","Shots","Shots on Target",
                     "Tackles Won","Interceptions","Fouls Drawn","Yellow Cards"]

    if len(sel) >= 2:
        radar_vals = normalise_radar(df_full, sel, RADAR_METRICS)
        radar_colors = ["#58a6ff","#3fb950","#f78166"]
        fig_r = go.Figure()
        for i, name in enumerate(sel):
            if name not in radar_vals:
                continue
            v = radar_vals[name]
            fig_r.add_trace(go.Scatterpolar(
                r=v + [v[0]], theta=RADAR_LABELS + [RADAR_LABELS[0]],
                name=name, fill="toself",
                fillcolor=radar_colors[i % len(radar_colors)],
                opacity=0.35,
                line=dict(color=radar_colors[i % len(radar_colors)], width=2),
            ))
        fig_r.update_layout(
            polar=dict(
                bgcolor="#161b22",
                radialaxis=dict(visible=True, range=[0,100], color="#8b949e",
                                gridcolor="#30363d", tickfont=dict(size=9)),
                angularaxis=dict(color="#c9d1d9", gridcolor="#30363d"),
            ),
            **_DARK, height=450, margin=dict(t=40,b=20),
            legend=dict(font=dict(size=12), bgcolor="#161b22", bordercolor="#30363d"),
        )
        st.plotly_chart(fig_r, use_container_width=True)
    else:
        st.info("Select at least 2 players to see the radar chart.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#57606a;font-size:.76rem;'>"
    "Player Data 2026/27 · FBref via Kaggle · Top 5 European Leagues</p>",
    unsafe_allow_html=True,
)
