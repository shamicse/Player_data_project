"""
modeling.py
Predictive modeling pipeline for the Player Data 2026/27 project.

Goal: Predict whether a player is a HIGH-PERFORMER (top-quartile Goals+Assists
      per 90 minutes) using a Random Forest classifier.

Exports:
  - outputs/Player_data_model.joblib   (trained model + metadata)
  - outputs/charts/feature_importance.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")          # headless rendering
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

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
TARGET_COL    = "HighPerformer"       # binary: top-25% G+A per 90 outfield players
MODEL_PATH    = os.path.join("outputs", "Player_data_model.joblib")
CHART_DIR     = os.path.join("outputs", "charts")
RANDOM_STATE  = 42

FEATURE_COLS = [
    "Age", "MP", "Starts", "Min", "90s",
    "Gls", "Ast", "Sh", "SoT", "SoT%",
    "Sh/90", "SoT/90", "G/Sh",
    "TklW", "Int", "Fld", "Fls",
    "CrdY", "CrdR", "Crs",
    "PK", "PKatt",
]

CATEGORICAL_FEATURES = ["PrimaryPos", "League"]


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare modelling DataFrame:
    - Filter to outfield players with >= 3 full games
    - Create binary target (top-25% G+A per 90)
    - Encode categoricals
    - Return feature matrix X and target y
    """
    df = df.copy()

    # Outfield players only with meaningful minutes
    df = df[df["PrimaryPos"] != "GK"].copy()
    df = df[df["90s"].fillna(0) >= 3].copy()

    # ── Target ────────────────────────────────────────────────────────────────
    if "GA_90_off" not in df.columns:
        df["GA_90_off"] = (df["G+A"] / df["90s"].replace(0, np.nan))
    threshold = df["GA_90_off"].quantile(0.75)
    df[TARGET_COL] = (df["GA_90_off"] >= threshold).astype(int)

    # ── Encode categoricals ───────────────────────────────────────────────────
    le_pos    = LabelEncoder()
    le_league = LabelEncoder()
    df["PrimaryPos_enc"] = le_pos.fit_transform(df["PrimaryPos"].fillna("MF"))
    df["League_enc"]     = le_league.fit_transform(df["League"].fillna("Unknown"))

    numeric_feats = FEATURE_COLS + ["PrimaryPos_enc", "League_enc"]
    numeric_feats = [c for c in numeric_feats if c in df.columns]

    return df, numeric_feats, le_pos, le_league, threshold


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame) -> dict:
    """
    Train a Random Forest + compare baselines.
    Returns a results dictionary with the best model and evaluation metrics.
    """
    df_model, feature_cols, le_pos, le_league, threshold = build_features(df)

    X = df_model[feature_cols]
    y = df_model[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # ── Pipelines ─────────────────────────────────────────────────────────────
    rf_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf",    RandomForestClassifier(
            n_estimators=200, max_depth=8,
            min_samples_leaf=3, random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ])

    gb_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf",    GradientBoostingClassifier(
            n_estimators=150, max_depth=4,
            learning_rate=0.08, random_state=RANDOM_STATE,
        )),
    ])

    lr_pipe = Pipeline([
        ("impute",  SimpleImputer(strategy="median")),
        ("scale",   StandardScaler()),
        ("clf",     LogisticRegression(max_iter=500, random_state=RANDOM_STATE,
                                        class_weight="balanced")),
    ])

    models = {
        "Random Forest":          rf_pipe,
        "Gradient Boosting":      gb_pipe,
        "Logistic Regression":    lr_pipe,
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    for name, pipe in models.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                  scoring="roc_auc", n_jobs=-1)
        cv_results[name] = scores
        print(f"  {name:<26}  CV ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    # Best model = highest mean CV AUC
    best_name = max(cv_results, key=lambda k: cv_results[k].mean())
    best_pipe  = models[best_name]
    best_pipe.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred  = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    report  = classification_report(y_test, y_pred, output_dict=True)
    cm      = confusion_matrix(y_test, y_pred)

    print(f"\n[modeling] Best model: {best_name}")
    print(f"[modeling] Test ROC-AUC: {roc_auc:.4f}")
    print(classification_report(y_test, y_pred))

    # ── Feature importance ────────────────────────────────────────────────────
    feat_imp = None
    if hasattr(best_pipe.named_steps["clf"], "feature_importances_"):
        feat_imp = pd.Series(
            best_pipe.named_steps["clf"].feature_importances_,
            index=feature_cols,
        ).sort_values(ascending=False)

    results = {
        "model":         best_pipe,
        "model_name":    best_name,
        "feature_cols":  feature_cols,
        "threshold":     threshold,
        "le_pos":        le_pos,
        "le_league":     le_league,
        "cv_results":    cv_results,
        "roc_auc":       roc_auc,
        "report":        report,
        "confusion_matrix": cm,
        "feat_importance": feat_imp,
        "X_test": X_test,
        "y_test": y_test,
        "y_proba": y_proba,
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Save & Load
# ─────────────────────────────────────────────────────────────────────────────

def save_model(results: dict, path: str = MODEL_PATH) -> None:
    """Persist model + metadata with joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {k: v for k, v in results.items()
               if k not in ("X_test", "y_test", "y_proba")}
    joblib.dump(payload, path)
    print(f"[modeling] Model saved -> {path}")


def load_model(path: str = MODEL_PATH) -> dict:
    """Load persisted model bundle."""
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# Chart exports
# ─────────────────────────────────────────────────────────────────────────────

def _savefig(name: str) -> str:
    os.makedirs(CHART_DIR, exist_ok=True)
    fpath = os.path.join(CHART_DIR, name)
    plt.savefig(fpath, dpi=150, bbox_inches="tight", facecolor="#0e1117")
    plt.close()
    return fpath


def plot_feature_importance(results: dict) -> str | None:
    """Bar chart of top-20 feature importances."""
    fi = results.get("feat_importance")
    if fi is None:
        return None
    top = fi.head(20)
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")
    bars = ax.barh(top.index[::-1], top.values[::-1], color="#58a6ff")
    ax.set_xlabel("Importance", color="#c9d1d9")
    ax.set_title(f"Feature Importance — {results['model_name']}", color="#f0f6fc", pad=12)
    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    return _savefig("feature_importance.png")


def plot_confusion_matrix(results: dict) -> str:
    """Confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")
    disp = ConfusionMatrixDisplay(
        confusion_matrix=results["confusion_matrix"],
        display_labels=["Standard", "High Performer"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix", color="#f0f6fc", pad=10)
    ax.tick_params(colors="#c9d1d9")
    ax.set_xlabel("Predicted", color="#c9d1d9")
    ax.set_ylabel("Actual", color="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    return _savefig("confusion_matrix.png")


def plot_roc_curve(results: dict) -> str:
    """ROC curve."""
    from sklearn.metrics import RocCurveDisplay
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")
    RocCurveDisplay.from_predictions(
        results["y_test"], results["y_proba"],
        name=results["model_name"], ax=ax,
    )
    for line in ax.get_lines():
        line.set_color("#58a6ff")
    ax.plot([0, 1], [0, 1], "k--", color="#8b949e")
    ax.set_title("ROC Curve", color="#f0f6fc", pad=10)
    ax.tick_params(colors="#c9d1d9")
    ax.set_xlabel("False Positive Rate", color="#c9d1d9")
    ax.set_ylabel("True Positive Rate", color="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    return _savefig("roc_curve.png")


def plot_cv_comparison(results: dict) -> str:
    """Cross-validation ROC-AUC boxplot for all models."""
    cv_data = results["cv_results"]
    names   = list(cv_data.keys())
    scores  = [cv_data[n] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")
    bp = ax.boxplot(scores, tick_labels=names, patch_artist=True, notch=True)
    colors = ["#58a6ff", "#3fb950", "#f78166"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ["whiskers", "caps", "medians", "fliers"]:
        plt.setp(bp[element], color="#c9d1d9")
    ax.set_ylabel("ROC-AUC", color="#c9d1d9")
    ax.set_title("5-Fold CV ROC-AUC Comparison", color="#f0f6fc", pad=10)
    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    return _savefig("cv_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Predict helper (for dashboard)
# ─────────────────────────────────────────────────────────────────────────────

def predict_single(model_bundle: dict, row: pd.Series) -> dict:
    """
    Given a loaded model bundle and a player row (Series),
    return {prediction: int, probability: float}.
    """
    feature_cols = model_bundle["feature_cols"]
    le_pos    = model_bundle["le_pos"]
    le_league = model_bundle["le_league"]

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


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from utils import load_raw_data, clean_data

    data_path = os.path.join("..", "data", "Player_data.csv")
    print("[modeling] Loading data …")
    df_raw  = load_raw_data(data_path)
    df_clean = clean_data(df_raw)

    print("[modeling] Training models …")
    results = train_model(df_clean)

    print("[modeling] Saving model …")
    save_model(results, os.path.join("..", MODEL_PATH))

    print("[modeling] Exporting charts ...")
    import modeling as _m
    _m.CHART_DIR = os.path.join("..", "outputs", "charts")
    plot_feature_importance(results)
    plot_confusion_matrix(results)
    plot_roc_curve(results)
    plot_cv_comparison(results)

    print("[modeling] Done.")
