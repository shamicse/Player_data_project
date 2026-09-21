"""
utils.py
Data cleaning, feature engineering, and KPI calculation functions
for the Player Data 2026/27 Football Analytics Project.
"""

import pandas as pd
import numpy as np

# ── League name mapping ───────────────────────────────────────────────────────
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

# Columns to coerce to numeric
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


# ─────────────────────────────────────────────────────────────────────────────
# Loading & Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw CSV and return a DataFrame."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
    - Coerce numeric columns
    - Map league names
    - Derive PrimaryPos
    - Derive engineered features
    - Drop duplicate Rk rows
    Returns cleaned DataFrame.
    """
    df = df.copy()

    # Drop unnamed index columns if present
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Deduplicate on Rk (keep first)
    if "Rk" in df.columns:
        df = df.drop_duplicates(subset="Rk", keep="first")

    # Coerce numerics
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # League short names
    if "Comp" in df.columns:
        df["League"] = df["Comp"].map(LEAGUE_MAP).fillna(df["Comp"])

    # Primary position (first listed)
    if "Pos" in df.columns:
        df["PrimaryPos"] = df["Pos"].str.split(",").str[0].str.strip()

    # ── Engineered features ───────────────────────────────────────────────────
    # Goals per 90
    df["Gls_90"] = (df["Gls"] / df["90s"].replace(0, np.nan)).round(3)

    # Assists per 90
    df["Ast_90"] = (df["Ast"] / df["90s"].replace(0, np.nan)).round(3)

    # G+A per 90
    df["GA_90_off"] = (df["G+A"] / df["90s"].replace(0, np.nan)).round(3)

    # Involvement score (normalised G+A per 90)
    df["InvolvementScore"] = df["GA_90_off"].fillna(0)

    # Defensive work rate: (TklW + Int) / 90s
    df["DefWork_90"] = ((df["TklW"] + df["Int"]) / df["90s"].replace(0, np.nan)).round(3)

    # Discipline index: (CrdY + 3*CrdR) / 90s
    df["DisciplineIdx"] = ((df["CrdY"] + 3 * df["CrdR"]) / df["90s"].replace(0, np.nan)).round(3)

    # Shot accuracy (capped)
    df["ShotAcc"] = df["SoT%"].clip(0, 100)

    # Age group
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 21, 25, 29, 33, 100],
        labels=["U21", "21-25", "26-29", "30-33", "34+"],
        right=True,
    )

    return df


def save_cleaned(df: pd.DataFrame, path: str) -> None:
    """Save cleaned DataFrame to CSV."""
    df.to_csv(path, index=False)
    print(f"[utils] Cleaned data saved -> {path}  ({len(df)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# KPI Calculations
# ─────────────────────────────────────────────────────────────────────────────

def kpi_summary(df: pd.DataFrame) -> dict:
    """Return top-level KPI dictionary for the dashboard header."""
    return {
        "total_players":   int(df["Player"].nunique()),
        "total_goals":     int(df["Gls"].sum()),
        "total_assists":   int(df["Ast"].sum()),
        "avg_age":         round(float(df["Age"].mean()), 1),
        "total_nations":   int(df["Nation"].nunique()),
        "total_clubs":     int(df["Squad"].nunique()),
        "total_leagues":   int(df["League"].nunique()),
        "avg_sot_pct":     round(float(df.loc[df["Sh"] >= 5, "SoT%"].mean()), 1),
    }


def league_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregated stats per league."""
    return (
        df.groupby("League", observed=True)
        .agg(
            Players    = ("Player",  "count"),
            Goals      = ("Gls",     "sum"),
            Assists    = ("Ast",     "sum"),
            AvgAge     = ("Age",     "mean"),
            TotalMins  = ("Min",     "sum"),
            YellowCards= ("CrdY",    "sum"),
            RedCards   = ("CrdR",    "sum"),
            AvgGoals90 = ("Gls_90",  "mean"),
        )
        .reset_index()
        .round({"AvgAge": 1, "AvgGoals90": 3})
    )


def position_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregated stats per primary position."""
    return (
        df.groupby("PrimaryPos", observed=True)
        .agg(
            Players   = ("Player", "count"),
            Goals     = ("Gls",    "sum"),
            Assists   = ("Ast",    "sum"),
            AvgTklW   = ("TklW",   "mean"),
            AvgInt    = ("Int",    "mean"),
            AvgSoTPct = ("SoT%",   "mean"),
        )
        .reset_index()
        .round(2)
    )


def top_players(df: pd.DataFrame, metric: str, n: int = 15,
                min_90s: float = 3.0) -> pd.DataFrame:
    """Return top-n players for a given metric with min playing time."""
    subset = df[df["90s"].fillna(0) >= min_90s].copy()
    base_cols = ["Player", "Nation", "Pos", "Squad", "League", "Age",
                 "Gls", "Ast", "G+A", "Sh", "SoT", "TklW", "Int",
                 "Min", "90s"]
    # Add metric only if it isn't already in base_cols (avoids duplicate columns)
    cols = base_cols if metric in base_cols else base_cols + [metric]
    cols = [c for c in cols if c in subset.columns]
    return (
        subset.nlargest(n, metric)[cols]
        .drop_duplicates(subset="Player")
        .reset_index(drop=True)
    )


def age_group_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Goals, assists, and player count by age group."""
    return (
        df.groupby("AgeGroup", observed=True)
        .agg(Players=("Player","count"),
             Goals  =("Gls",   "sum"),
             Assists=("Ast",   "sum"))
        .reset_index()
    )


def nation_summary(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Top nations by player count with goal/assist totals."""
    ns = (
        df.groupby("Nation", observed=True)
        .agg(Players=("Player","count"),
             Goals  =("Gls",   "sum"),
             Assists=("Ast",   "sum"))
        .reset_index()
    )
    ns["G+A"] = ns["Goals"] + ns["Assists"]
    return ns.nlargest(top_n, "Players").reset_index(drop=True)


def gk_summary(df: pd.DataFrame, min_mins: int = 270) -> pd.DataFrame:
    """Goalkeeper stats for qualified keepers (>= min_mins minutes)."""
    gk = df[(df["PrimaryPos"] == "GK") & (df["Min"].fillna(0) >= min_mins)].copy()
    cols = ["Player","Nation","Squad","League","Min","GA","GA90",
            "SoTA","Saves","Save%","W","D","L","CS","CS%"]
    return gk[[c for c in cols if c in gk.columns]].sort_values("Save%", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helper for radar charts
# ─────────────────────────────────────────────────────────────────────────────

def normalise_radar(df: pd.DataFrame, player_names: list,
                    metrics: list) -> dict:
    """
    Return {player_name: [0-100 normalised values]} for radar charts.
    Normalises each metric against the full df max.
    """
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
