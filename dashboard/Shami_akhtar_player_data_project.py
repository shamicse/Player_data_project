"""
app.py  —  Player Data 2026/27 · Football Analytics Dashboard
Streamlit web application — run with:  streamlit run dashboard/app.py

All data utilities and ML modeling logic are inlined below in clearly
marked sections so the entire dashboard lives in a single file.
"""

import os, sys, shutil, glob as _glob, warnings, datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 · Path helpers
# ─────────────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))

def _find_file(*relative_paths: str) -> str:
    """Return the first existing path from candidates relative to the script,
    one level up, or the process cwd. Falls back to the first candidate."""
    candidates = []
    for rel in relative_paths:
        candidates.append(os.path.join(_HERE, rel))
        candidates.append(os.path.join(_HERE, "..", rel))
        candidates.append(os.path.join(os.getcwd(), rel))
    for c in candidates:
        if os.path.exists(c):
            return os.path.normpath(c)
    return os.path.normpath(candidates[0])

DATA_PATH  = _find_file("data/Player_data.csv")
MODEL_PATH = _find_file("outputs/Player_data_model.joblib")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 · Data utilities  (formerly utils.py)
# ─────────────────────────────────────────────────────────────────────────────

LEAGUE_MAP = {
    "eng Premier League": "Premier League",
    "es La Liga":         "La Liga",
    "de Bundesliga":      "Bundesliga",
    "it Serie A":         "Serie A",
    "fr Ligue 1":         "Ligue 1",
}

LEAGUE_COLORS = {
    "Premier League": "#3d0085",
    "La Liga":        "#d4001a",
    "Bundesliga":     "#d30000",
    "Serie A":        "#0b3f8a",
    "Ligue 1":        "#1a6fb5",
}

NUMERIC_COLS = [
    "Age", "Born", "MP", "Starts", "Min", "90s",
    "Gls", "Ast", "G+A", "G-PK", "PK", "PKatt",
    "CrdY", "CrdR", "G+A-PK",
    "Sh", "SoT", "SoT%", "Sh/90", "SoT/90", "G/Sh", "G/SoT",
    "PK_stats_shooting", "PKatt_stats_shooting",
    "Crs", "TklW", "Int", "Fld", "Fls", "2CrdY", "OG",
    "GA", "GA90", "SoTA", "Saves", "Save%",
    "W", "D", "L", "CS", "CS%",
    "PKatt_stats_keeper", "PKA", "PKsv", "PKm",
]


def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    if "Rk" in df.columns:
        df = df.drop_duplicates(subset="Rk", keep="first")
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Comp" in df.columns:
        df["League"] = df["Comp"].map(LEAGUE_MAP).fillna(df["Comp"])
    if "Pos" in df.columns:
        df["PrimaryPos"] = df["Pos"].str.split(",").str[0].str.strip()
    df["Gls_90"]    = (df["Gls"] / df["90s"].replace(0, np.nan)).round(3)
    df["Ast_90"]    = (df["Ast"] / df["90s"].replace(0, np.nan)).round(3)
    df["GA_90_off"] = (df["G+A"] / df["90s"].replace(0, np.nan)).round(3)
    df["InvolvementScore"] = df["GA_90_off"].fillna(0)
    df["DefWork_90"]  = ((df["TklW"] + df["Int"]) / df["90s"].replace(0, np.nan)).round(3)
    df["DisciplineIdx"] = ((df["CrdY"] + 3 * df["CrdR"]) / df["90s"].replace(0, np.nan)).round(3)
    df["ShotAcc"] = df["SoT%"].clip(0, 100)
    df["AgeGroup"] = pd.cut(
        df["Age"], bins=[0, 21, 25, 29, 33, 100],
        labels=["U21", "21-25", "26-29", "30-33", "34+"], right=True,
    )
    return df


def kpi_summary(df: pd.DataFrame) -> dict:
    return {
        "total_players":  int(df["Player"].nunique()),
        "total_goals":    int(df["Gls"].sum()),
        "total_assists":  int(df["Ast"].sum()),
        "avg_age":        round(float(df["Age"].mean()), 1),
        "total_nations":  int(df["Nation"].nunique()),
        "total_clubs":    int(df["Squad"].nunique()),
        "total_leagues":  int(df["League"].nunique()),
        "avg_sot_pct":    round(float(df.loc[df["Sh"] >= 5, "SoT%"].mean()), 1),
    }


def league_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("League", observed=True)
        .agg(
            Players    =("Player",  "count"),
            Goals      =("Gls",     "sum"),
            Assists    =("Ast",     "sum"),
            AvgAge     =("Age",     "mean"),
            TotalMins  =("Min",     "sum"),
            YellowCards=("CrdY",    "sum"),
            RedCards   =("CrdR",    "sum"),
            AvgGoals90 =("Gls_90",  "mean"),
        )
        .reset_index()
        .round({"AvgAge": 1, "AvgGoals90": 3})
    )


def position_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("PrimaryPos", observed=True)
        .agg(
            Players  =("Player", "count"),
            Goals    =("Gls",    "sum"),
            Assists  =("Ast",    "sum"),
            AvgTklW  =("TklW",   "mean"),
            AvgInt   =("Int",    "mean"),
            AvgSoTPct=("SoT%",   "mean"),
        )
        .reset_index()
        .round(2)
    )


def top_players(df: pd.DataFrame, metric: str, n: int = 15,
                min_90s: float = 3.0) -> pd.DataFrame:
    subset = df[df["90s"].fillna(0) >= min_90s].copy()
    base_cols = ["Player", "Nation", "Pos", "Squad", "League", "Age",
                 "Gls", "Ast", "G+A", "Sh", "SoT", "TklW", "Int", "Min", "90s"]
    cols = base_cols if metric in base_cols else base_cols + [metric]
    cols = [c for c in cols if c in subset.columns]
    return (
        subset.nlargest(n, metric)[cols]
        .drop_duplicates(subset="Player")
        .reset_index(drop=True)
    )


def age_group_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("AgeGroup", observed=True)
        .agg(Players=("Player", "count"), Goals=("Gls", "sum"), Assists=("Ast", "sum"))
        .reset_index()
    )


def nation_summary(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    ns = (
        df.groupby("Nation", observed=True)
        .agg(Players=("Player", "count"), Goals=("Gls", "sum"), Assists=("Ast", "sum"))
        .reset_index()
    )
    ns["G+A"] = ns["Goals"] + ns["Assists"]
    return ns.nlargest(top_n, "Players").reset_index(drop=True)


def gk_summary(df: pd.DataFrame, min_mins: int = 270) -> pd.DataFrame:
    gk = df[(df["PrimaryPos"] == "GK") & (df["Min"].fillna(0) >= min_mins)].copy()
    cols = ["Player", "Nation", "Squad", "League", "Min", "GA", "GA90",
            "SoTA", "Saves", "Save%", "W", "D", "L", "CS", "CS%"]
    return (
        gk[[c for c in cols if c in gk.columns]]
        .sort_values("Save%", ascending=False)
        .reset_index(drop=True)
    )


def normalise_radar(df: pd.DataFrame, player_names: list, metrics: list) -> dict:
    result = {}
    for name in player_names:
        row = df[df["Player"] == name]
        if row.empty:
            continue
        row = row.iloc[0]
        vals = []
        for m in metrics:
            col_max = df[m].max() if m in df.columns and df[m].max() > 0 else 1
            v = float(row[m]) if m in row.index and pd.notna(row[m]) else 0.0
            vals.append(round(v / col_max * 100, 1))
        result[name] = vals
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 · ML modeling  (formerly modeling.py)
# ─────────────────────────────────────────────────────────────────────────────

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

_TARGET_COL   = "HighPerformer"
_RANDOM_STATE = 42
_CHART_DIR    = _find_file("outputs/charts")

_FEATURE_COLS = [
    "Age", "MP", "Starts", "Min", "90s",
    "Gls", "Ast", "Sh", "SoT", "SoT%",
    "Sh/90", "SoT/90", "G/Sh",
    "TklW", "Int", "Fld", "Fls",
    "CrdY", "CrdR", "Crs",
    "PK", "PKatt",
]


def _build_features(df: pd.DataFrame):
    df = df[df["PrimaryPos"] != "GK"].copy()
    df = df[df["90s"].fillna(0) >= 3].copy()
    if "GA_90_off" not in df.columns:
        df["GA_90_off"] = df["G+A"] / df["90s"].replace(0, np.nan)
    threshold = df["GA_90_off"].quantile(0.75)
    df[_TARGET_COL] = (df["GA_90_off"] >= threshold).astype(int)
    le_pos    = LabelEncoder()
    le_league = LabelEncoder()
    df["PrimaryPos_enc"] = le_pos.fit_transform(df["PrimaryPos"].fillna("MF"))
    df["League_enc"]     = le_league.fit_transform(df["League"].fillna("Unknown"))
    numeric_feats = [c for c in _FEATURE_COLS + ["PrimaryPos_enc", "League_enc"]
                     if c in df.columns]
    return df, numeric_feats, le_pos, le_league, threshold


def train_model(df: pd.DataFrame) -> dict:
    df_model, feature_cols, le_pos, le_league, threshold = _build_features(df)
    X = df_model[feature_cols]
    y = df_model[_TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=_RANDOM_STATE, stratify=y
    )
    rf_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf",    RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            random_state=_RANDOM_STATE, class_weight="balanced")),
    ])
    gb_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf",    GradientBoostingClassifier(
            n_estimators=150, max_depth=4,
            learning_rate=0.08, random_state=_RANDOM_STATE)),
    ])
    lr_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("clf",    LogisticRegression(
            max_iter=500, random_state=_RANDOM_STATE, class_weight="balanced")),
    ])
    models = {
        "Random Forest":       rf_pipe,
        "Gradient Boosting":   gb_pipe,
        "Logistic Regression": lr_pipe,
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=_RANDOM_STATE)
    cv_results = {}
    for name, pipe in models.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=-1)
        cv_results[name] = scores
    best_name = max(cv_results, key=lambda k: cv_results[k].mean())
    best_pipe = models[best_name]
    best_pipe.fit(X_train, y_train)
    y_pred  = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    report  = classification_report(y_test, y_pred, output_dict=True)
    cm      = confusion_matrix(y_test, y_pred)
    feat_imp = None
    if hasattr(best_pipe.named_steps["clf"], "feature_importances_"):
        feat_imp = pd.Series(
            best_pipe.named_steps["clf"].feature_importances_,
            index=feature_cols,
        ).sort_values(ascending=False)
    return {
        "model":           best_pipe,
        "model_name":      best_name,
        "feature_cols":    feature_cols,
        "threshold":       threshold,
        "le_pos":          le_pos,
        "le_league":       le_league,
        "cv_results":      cv_results,
        "roc_auc":         roc_auc,
        "report":          report,
        "confusion_matrix": cm,
        "feat_importance": feat_imp,
        "X_test":  X_test,
        "y_test":  y_test,
        "y_proba": y_proba,
    }


def save_model(results: dict, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {k: v for k, v in results.items()
               if k not in ("X_test", "y_test", "y_proba")}
    joblib.dump(payload, path)


def _savefig(name: str) -> str:
    chart_dir = _find_file("outputs/charts")
    os.makedirs(chart_dir, exist_ok=True)
    fpath = os.path.join(chart_dir, name)
    plt.savefig(fpath, dpi=150, bbox_inches="tight", facecolor="#0e1117")
    plt.close()
    return fpath


def plot_feature_importance(results: dict):
    fi = results.get("feat_importance")
    if fi is None:
        return None
    top = fi.head(20)
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#161b22")
    ax.barh(top.index[::-1], top.values[::-1], color="#58a6ff")
    ax.set_xlabel("Importance", color="#c9d1d9")
    ax.set_title(f"Feature Importance — {results['model_name']}", color="#f0f6fc", pad=12)
    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values(): spine.set_edgecolor("#30363d")
    return _savefig("feature_importance.png")


def plot_confusion_matrix(results: dict):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#161b22")
    ConfusionMatrixDisplay(
        confusion_matrix=results["confusion_matrix"],
        display_labels=["Standard", "High Performer"],
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix", color="#f0f6fc", pad=10)
    ax.tick_params(colors="#c9d1d9")
    ax.set_xlabel("Predicted", color="#c9d1d9"); ax.set_ylabel("Actual", color="#c9d1d9")
    for spine in ax.spines.values(): spine.set_edgecolor("#30363d")
    return _savefig("confusion_matrix.png")


def plot_roc_curve(results: dict):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#161b22")
    RocCurveDisplay.from_predictions(
        results["y_test"], results["y_proba"],
        name=results["model_name"], ax=ax,
    )
    for line in ax.get_lines(): line.set_color("#58a6ff")
    ax.plot([0, 1], [0, 1], "k--", color="#8b949e")
    ax.set_title("ROC Curve", color="#f0f6fc", pad=10)
    ax.tick_params(colors="#c9d1d9")
    ax.set_xlabel("False Positive Rate", color="#c9d1d9")
    ax.set_ylabel("True Positive Rate", color="#c9d1d9")
    for spine in ax.spines.values(): spine.set_edgecolor("#30363d")
    return _savefig("roc_curve.png")


def plot_cv_comparison(results: dict):
    cv_data = results["cv_results"]
    names  = list(cv_data.keys())
    scores = [cv_data[n] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#161b22")
    bp = ax.boxplot(scores, tick_labels=names, patch_artist=True, notch=True)
    for patch, color in zip(bp["boxes"], ["#58a6ff", "#3fb950", "#f78166"]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    for element in ["whiskers", "caps", "medians", "fliers"]:
        plt.setp(bp[element], color="#c9d1d9")
    ax.set_ylabel("ROC-AUC", color="#c9d1d9")
    ax.set_title("5-Fold CV ROC-AUC Comparison", color="#f0f6fc", pad=10)
    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values(): spine.set_edgecolor("#30363d")
    return _savefig("cv_comparison.png")


def predict_single(model_bundle: dict, row: pd.Series) -> dict:
    feature_cols = model_bundle["feature_cols"]
    le_pos       = model_bundle["le_pos"]
    le_league    = model_bundle["le_league"]
    data = row.copy()
    if "PrimaryPos" in data.index:
        pos_val = data["PrimaryPos"] if data["PrimaryPos"] in le_pos.classes_ else le_pos.classes_[0]
        data["PrimaryPos_enc"] = le_pos.transform([pos_val])[0]
    if "League" in data.index:
        lg_val = data["League"] if data["League"] in le_league.classes_ else le_league.classes_[0]
        data["League_enc"] = le_league.transform([lg_val])[0]
    X = pd.DataFrame([data[feature_cols]])
    pred  = model_bundle["model"].predict(X)[0]
    proba = model_bundle["model"].predict_proba(X)[0][1]
    return {"prediction": int(pred), "probability": round(float(proba), 4)}


# CLI entry-point: python dashboard/app.py  (trains & saves the model)
if __name__ == "__main__":
    print("[app] Loading data …")
    df_raw   = load_raw_data(DATA_PATH)
    df_clean = clean_data(df_raw)
    print("[app] Training models …")
    results  = train_model(df_clean)
    print("[app] Saving model …")
    save_model(results, MODEL_PATH)
    print("[app] Exporting charts …")
    plot_feature_importance(results)
    plot_confusion_matrix(results)
    plot_roc_curve(results)
    plot_cv_comparison(results)
    print("[app] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 · Kaggle auto-update
# ─────────────────────────────────────────────────────────────────────────────

def _try_kaggle_update() -> str:
    """Pull latest CSV from Kaggle. Uses st.secrets for credentials on Cloud."""
    try:
        import kagglehub
        try:
            os.environ.setdefault("KAGGLE_USERNAME", st.secrets["kaggle"]["username"])
            os.environ.setdefault("KAGGLE_KEY",      st.secrets["kaggle"]["key"])
        except (KeyError, FileNotFoundError):
            pass
        dl_path = kagglehub.dataset_download(
            "hubertsidorowicz/football-players-stats-2026-2027"
        )
        candidates = _glob.glob(
            os.path.join(dl_path, "**", "*light*.csv"), recursive=True
        ) or _glob.glob(os.path.join(dl_path, "**", "*.csv"), recursive=True)
        if not candidates:
            return "kaggle: no CSV found in download"
        src = candidates[0]
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        shutil.copy2(src, DATA_PATH)
        return f"kaggle: updated from {os.path.basename(src)}"
    except ImportError:
        return "kagglehub not installed — add it to requirements.txt"
    except Exception as e:
        return f"kaggle update skipped: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 · Streamlit app
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Player Data Analytics 2026/27",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# ── Data loading (cached, auto-refreshes every 6 hours) ──────────────────────
_CACHE_TTL = 6 * 3600

@st.cache_data(show_spinner="Loading & cleaning data …", ttl=_CACHE_TTL)
def get_data():
    _try_kaggle_update()
    raw = load_raw_data(DATA_PATH)
    return clean_data(raw)

df_full = get_data()

# ── Model loading (optional, cached) ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model_bundle = get_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ Filters")
    st.markdown("---")

    all_leagues = sorted(df_full["League"].dropna().unique())
    sel_leagues = st.multiselect("🏆 League", all_leagues, default=all_leagues)

    all_pos  = ["All", "FW", "MF", "DF", "GK"]
    sel_pos  = st.selectbox("🎯 Position", all_pos)

    age_min, age_max = int(df_full["Age"].min()), int(df_full["Age"].max())
    age_range = st.slider("🎂 Age Range", age_min, age_max, (age_min, age_max))

    min_mins = st.slider("⏱ Min. Minutes Played", 0, 3000, 0, step=90)

    st.markdown("---")
    st.markdown("### 🔄 Data Update")
    if os.path.exists(DATA_PATH):
        mtime = os.path.getmtime(DATA_PATH)
        last_updated = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        st.markdown(
            f"<div style='color:#8b949e;font-size:.74rem;'>Last updated: {last_updated}</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<div style='color:#8b949e;font-size:.74rem;'>Auto-refreshes every 6 hours.</div>",
        unsafe_allow_html=True,
    )
    if st.button("⬇️ Force refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

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
    ("👤 Players",  kpi["total_players"]),
    ("⚽ Goals",    kpi["total_goals"]),
    ("🎯 Assists",  kpi["total_assists"]),
    ("🎂 Avg Age",  kpi["avg_age"]),
    ("🌍 Nations",  kpi["total_nations"]),
    ("🏟 Clubs",    kpi["total_clubs"]),
    ("🏆 Leagues",  kpi["total_leagues"]),
    ("🎯 Avg SoT%", f"{kpi['avg_sot_pct']}%"),
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

_DARK     = dict(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
_AX       = dict(color="#8b949e", gridcolor="#21262d")
_L_COLORS = {
    "Premier League": "#58a6ff", "La Liga": "#f85149",
    "Bundesliga": "#e3b341", "Serie A": "#3fb950", "Ligue 1": "#d2a8ff",
}

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · League Overview
# ─────────────────────────────────────────────────────────────────────────────
with t_league:
    st.markdown('<div class="tab-title">League Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-txt">Aggregated statistics per competition.</div>', unsafe_allow_html=True)

    lg = league_summary(df)

    st.markdown('<div class="section-hdr">Goals & Assists</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_bar(x=lg["League"], y=lg["Goals"],  name="Goals",   marker_color="#58a6ff")
    fig.add_bar(x=lg["League"], y=lg["Assists"], name="Assists", marker_color="#3fb950")
    fig.update_layout(**_DARK, barmode="group", height=340, margin=dict(t=20,b=0),
                      xaxis=dict(title=None,**_AX), yaxis=dict(title="Count",**_AX),
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
        fig3.update_layout(**_DARK, barmode="group", height=300, margin=dict(t=10,b=0),
                           xaxis=dict(title=None,**_AX), yaxis=dict(title="Cards",**_AX),
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
    fig5 = px.bar(lg, x="League", y="AvgGoals90", color="League",
                  color_discrete_map=_L_COLORS, text=lg["AvgGoals90"].round(3))
    fig5.update_layout(**_DARK, height=290, margin=dict(t=10,b=0), showlegend=False,
                       xaxis=dict(title=None,**_AX), yaxis=dict(title="Gls/90",**_AX))
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
        fig = px.bar(sp, x="PrimaryPos", y="SoT%", color="PrimaryPos",
                     color_discrete_sequence=["#58a6ff","#3fb950","#f78166","#d2a8ff"],
                     text="SoT%")
        fig.update_layout(**_DARK, height=280, margin=dict(t=10,b=0), showlegend=False,
                          xaxis=dict(title="Position",**_AX), yaxis=dict(title="SoT%",**_AX))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown('<div class="section-hdr">Goals per Shot — Top 20 (min. 10 shots)</div>', unsafe_allow_html=True)
        dg = df_att[(df_att["Sh"] >= 10) & df_att["G/Sh"].notna()].nlargest(20,"G/Sh")
        fig = px.bar(dg, x="G/Sh", y="Player", orientation="h",
                     color="G/Sh", color_continuous_scale="Blues",
                     text=dg["G/Sh"].round(2))
        fig.update_layout(**_DARK, height=440, margin=dict(t=10,b=0),
                          yaxis=dict(autorange="reversed",title=None,color="#8b949e"),
                          xaxis=dict(title="G/Sh",**_AX), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-hdr">Goals & Assists by Age Group</div>', unsafe_allow_html=True)
    ag = age_group_stats(df_att)
    fig = go.Figure()
    fig.add_bar(x=ag["AgeGroup"].astype(str), y=ag["Goals"],  name="Goals",   marker_color="#58a6ff")
    fig.add_bar(x=ag["AgeGroup"].astype(str), y=ag["Assists"], name="Assists", marker_color="#3fb950")
    fig.update_layout(**_DARK, barmode="group", height=300, margin=dict(t=10,b=0),
                      xaxis=dict(title="Age Group",**_AX), yaxis=dict(title="Count",**_AX),
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
                          xaxis=dict(title=None,**_AX), yaxis=dict(title="Avg / Player",**_AX),
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
                          xaxis=dict(title="GA/90",**_AX), yaxis=dict(title="Save%",**_AX))
        st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<div class="section-hdr">Clean Sheet % by League</div>', unsafe_allow_html=True)
            csl = (df_gk.dropna(subset=["CS%"])
                   .groupby("League", observed=True)["CS%"].mean()
                   .reset_index().round(1))
            fig = px.bar(csl, x="League", y="CS%", color="League",
                         color_discrete_map=_L_COLORS, text="CS%")
            fig.update_layout(**_DARK, height=280, margin=dict(t=10,b=0), showlegend=False,
                              xaxis=dict(title=None,**_AX), yaxis=dict(title="CS%",**_AX))
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
        fig.update_layout(**_DARK, height=320, margin=dict(t=10,b=0), showlegend=False,
                          xaxis=dict(title=None,**_AX), yaxis=dict(title="Age",**_AX))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-hdr">Age Histogram</div>', unsafe_allow_html=True)
        fig = px.histogram(df.dropna(subset=["Age"]), x="Age", nbins=30,
                           color_discrete_sequence=["#58a6ff"])
        fig.update_layout(**_DARK, height=190, margin=dict(t=5,b=0),
                          xaxis=dict(title="Age",**_AX), yaxis=dict(title="Players",**_AX),
                          bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-hdr">Top 15 Nations — Goals + Assists (stacked)</div>', unsafe_allow_html=True)
    ng = nation_summary(df, 15)
    fig = go.Figure()
    fig.add_bar(x=ng["Nation"], y=ng["Goals"],  name="Goals",   marker_color="#58a6ff")
    fig.add_bar(x=ng["Nation"], y=ng["Assists"], name="Assists", marker_color="#3fb950")
    fig.update_layout(**_DARK, barmode="stack", height=320, margin=dict(t=10,b=0),
                      xaxis=dict(title=None,**_AX), yaxis=dict(title="G+A",**_AX),
                      legend=dict(orientation="h",y=1.08))
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 · ML Model
# ─────────────────────────────────────────────────────────────────────────────
with t_ml:
    st.markdown('<div class="tab-title">🤖 Predictive Model — High Performer Classifier</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-txt">Random Forest trained to predict whether an outfield player is a '
        'high-performer (top-25% G+A per 90). Run <code>python dashboard/app.py</code> from the '
        'project root to train & save the model.</div>',
        unsafe_allow_html=True,
    )

    chart_dir = _find_file("outputs/charts")

    def _show_chart(name, caption):
        p = os.path.join(chart_dir, name)
        if os.path.exists(p):
            st.image(p, caption=caption, use_container_width=True)

    if model_bundle is None:
        st.warning("⚠️ Model not yet trained. Run `python dashboard/app.py` from the project root, then refresh.")
    else:
        mb = model_bundle
        c1, c2, c3 = st.columns(3)
        roc = mb.get("roc_auc", 0)
        rep = mb.get("report", {})
        c1.metric("🏆 Best Model",  mb.get("model_name", "—"))
        c2.metric("📈 ROC-AUC",     f"{roc:.4f}")
        acc = rep.get("accuracy", 0)
        c3.metric("✅ Accuracy",     f"{acc:.2%}" if acc else "—")

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

        st.markdown('<div class="section-hdr">🔮 Live Player Prediction</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-txt">Select any player and get the model\'s high-performer probability.</div>', unsafe_allow_html=True)
        player_opts = sorted(df_full["Player"].dropna().unique())
        sel_player  = st.selectbox("Select player", player_opts)
        if sel_player:
            row      = df_full[df_full["Player"] == sel_player].iloc[0]
            pred_res = predict_single(mb, row)
            prob     = pred_res["probability"]
            pred     = pred_res["prediction"]
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
        radar_vals   = normalise_radar(df_full, sel, RADAR_METRICS)
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
