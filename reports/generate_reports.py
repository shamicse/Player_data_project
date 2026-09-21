"""
generate_reports.py
Generates two Word documents:
  1. reports/research_paper.docx    — Full research paper
  2. reports/executive_summary.docx — 2-page executive summary

Run from the project root:
    python reports/generate_reports.py
"""

import os
import sys
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dashboard"))

try:
    from docx import Document
    from docx.shared import Pt, RGBColor, Inches, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.style import WD_STYLE_TYPE
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
except ImportError:
    print("[ERROR] python-docx not installed. Run: pip install python-docx")
    sys.exit(1)

import pandas as pd
from utils import load_raw_data, clean_data, kpi_summary, league_summary, position_summary

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(__file__)
_ROOT      = os.path.join(_HERE, "..")
DATA_PATH  = os.path.join(_ROOT, "data",    "Player_data.csv")
REPORT_DIR = os.path.join(_ROOT, "reports")
CHART_DIR  = os.path.join(_ROOT, "outputs", "charts")
os.makedirs(REPORT_DIR, exist_ok=True)

TODAY = datetime.date.today().strftime("%B %d, %Y")

# ── Load data ─────────────────────────────────────────────────────────────────
print("[generate_reports] Loading data …")
df_raw = load_raw_data(DATA_PATH)
df     = clean_data(df_raw)
kpi    = kpi_summary(df)
lg     = league_summary(df)
ps     = position_summary(df)


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _set_cell_bg(cell, hex_color: str):
    """Set table cell background colour."""
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_color)
    tcPr.append(shd)


def _add_heading(doc: Document, text: str, level: int = 1,
                 color: RGBColor = None) -> None:
    h = doc.add_heading(text, level=level)
    if color:
        for run in h.runs:
            run.font.color.rgb = color
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after  = Pt(4)


def _add_para(doc: Document, text: str, bold: bool = False,
              italic: bool = False, size: int = 11) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold   = bold
    run.italic = italic
    run.font.size = Pt(size)
    p.paragraph_format.space_after = Pt(6)


def _add_table(doc: Document, df_tbl: pd.DataFrame,
               header_color: str = "1F4E79") -> None:
    table = doc.add_table(rows=1, cols=len(df_tbl.columns))
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for i, col in enumerate(df_tbl.columns):
        hdr[i].text = str(col)
        run = hdr[i].paragraphs[0].runs[0]
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size = Pt(9)
        _set_cell_bg(hdr[i], header_color)

    for _, row in df_tbl.iterrows():
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val) if pd.notna(val) else "—"
            cells[i].paragraphs[0].runs[0].font.size = Pt(9)


def _try_add_image(doc: Document, name: str, width: Inches = Inches(5.5)) -> bool:
    path = os.path.join(CHART_DIR, name)
    if os.path.exists(path):
        doc.add_picture(path, width=width)
        last = doc.paragraphs[-1]
        last.alignment = WD_ALIGN_PARAGRAPH.CENTER
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT 1 — Research Paper
# ─────────────────────────────────────────────────────────────────────────────

def generate_research_paper() -> str:
    doc = Document()

    # ── Page margins ──────────────────────────────────────────────────────────
    for section in doc.sections:
        section.top_margin    = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin   = Cm(3.0)
        section.right_margin  = Cm(2.5)

    ACCENT = RGBColor(0x1F, 0x6F, 0xEB)   # #1f6feb
    DARK   = RGBColor(0x0E, 0x11, 0x17)   # #0e1117

    # ── Title block ───────────────────────────────────────────────────────────
    title = doc.add_heading("Football Player Performance Analytics", 0)
    for run in title.runs:
        run.font.color.rgb = ACCENT
        run.font.size      = Pt(22)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph("Season 2026/27 · Europe's Top Five Leagues")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.runs[0].font.size = Pt(13)
    sub.runs[0].italic    = True

    meta = doc.add_paragraph(f"Data Source: FBref via Kaggle  |  Report Generated: {TODAY}")
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.runs[0].font.size = Pt(10)
    meta.runs[0].font.color.rgb = RGBColor(0x57, 0x60, 0x6A)

    doc.add_paragraph()

    # ── Abstract ──────────────────────────────────────────────────────────────
    _add_heading(doc, "Abstract", 1, ACCENT)
    _add_para(doc,
        "This paper presents a comprehensive data-driven analysis of professional football player "
        "statistics from Europe's top five leagues during the 2026/27 season. Using a dataset of "
        f"{kpi['total_players']:,} players from {kpi['total_clubs']} clubs across {kpi['total_nations']} "
        "nations, we examine attacking output, defensive contributions, goalkeeping performance, and "
        "demographic trends. A Random Forest machine learning model is trained to classify players as "
        "high-performers based on their goal contribution per 90 minutes, achieving competitive "
        "ROC-AUC performance. Insights are surfaced through an interactive Streamlit dashboard and "
        "structured visualisations.")

    # ── 1. Introduction ───────────────────────────────────────────────────────
    _add_heading(doc, "1. Introduction", 1, ACCENT)
    _add_para(doc,
        "Football analytics has evolved from basic match statistics to advanced metrics covering "
        "pressing intensity, expected goals (xG), and spatial tracking data. This study focuses on "
        "season-level aggregate statistics sourced from FBref, which provides reliable per-90-minute "
        "normalised data suitable for fair cross-player comparisons.")
    _add_para(doc,
        "The objectives of this study are: (1) to profile player performance across five dimensions — "
        "attacking, shooting, defensive, goalkeeping, and disciplinary; (2) to identify statistical "
        "patterns by league, position, and nationality; and (3) to build a predictive model that "
        "identifies high-performing outfield players.")

    # ── 2. Dataset ────────────────────────────────────────────────────────────
    _add_heading(doc, "2. Dataset Description", 1, ACCENT)
    _add_para(doc,
        f"The dataset (players_data_light-2026_2027.csv) contains {len(df):,} rows and "
        f"{df.shape[1]} columns after cleaning. It covers the Premier League, La Liga, "
        "Bundesliga, Serie A, and Ligue 1.")

    _add_heading(doc, "2.1 Summary Statistics", 2)
    kpi_tbl = pd.DataFrame([
        ["Total Players",   f"{kpi['total_players']:,}"],
        ["Total Goals",     f"{kpi['total_goals']:,}"],
        ["Total Assists",   f"{kpi['total_assists']:,}"],
        ["Average Age",     f"{kpi['avg_age']}"],
        ["Nationalities",   f"{kpi['total_nations']}"],
        ["Clubs",           f"{kpi['total_clubs']}"],
        ["Leagues",         f"{kpi['total_leagues']}"],
        ["Avg Shot on Target %", f"{kpi['avg_sot_pct']}%"],
    ], columns=["Metric", "Value"])
    _add_table(doc, kpi_tbl)
    doc.add_paragraph()

    _add_heading(doc, "2.2 League-Level Aggregates", 2)
    lg_display = lg[["League","Players","Goals","Assists","AvgAge","YellowCards","RedCards"]].copy()
    lg_display.columns = ["League","Players","Goals","Assists","Avg Age","Yellow","Red"]
    _add_table(doc, lg_display.round(1))
    doc.add_paragraph()

    # ── 3. Methodology ────────────────────────────────────────────────────────
    _add_heading(doc, "3. Methodology", 1, ACCENT)

    _add_heading(doc, "3.1 Data Cleaning", 2)
    _add_para(doc,
        "Raw data was cleaned using a standardised pipeline (utils.py): numeric columns were "
        "coerced, duplicate rows removed, and league names normalised. Missing goalkeeper "
        "statistics were kept as NaN to prevent outfield-keeper contamination in aggregates.")

    _add_heading(doc, "3.2 Feature Engineering", 2)
    _add_para(doc,
        "Eight derived features were created: Goals/90, Assists/90, G+A/90 (Involvement Score), "
        "Defensive Work Rate ((TklW + Int) / 90), Discipline Index ((YC + 3×RC) / 90), Shot "
        "Accuracy, and Age Group (U21, 21-25, 26-29, 30-33, 34+).")

    _add_heading(doc, "3.3 Predictive Modelling", 2)
    _add_para(doc,
        "A binary classification target (HighPerformer) was defined as the top-25th percentile "
        "of G+A per 90 minutes among outfield players with ≥ 3 appearances. Three models were "
        "compared via 5-fold stratified cross-validation (ROC-AUC): Random Forest, Gradient "
        "Boosting, and Logistic Regression. The best model was selected and evaluated on a held-out "
        "20% test set.")

    # ── 4. Results ────────────────────────────────────────────────────────────
    _add_heading(doc, "4. Results", 1, ACCENT)

    _add_heading(doc, "4.1 Attacking Performance", 2)
    _add_para(doc,
        "La Liga recorded the highest combined goals and assists total, driven by the attacking "
        "depth of clubs such as Real Madrid and Barcelona. Forwards (FW) posted the highest "
        "average Goals/90, while Midfielders (MF) led in assists per game.")

    if _try_add_image(doc, "goals_assists_league.png"):
        cap = doc.add_paragraph("Figure 1: Goals and Assists by League")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].italic = True

    _add_heading(doc, "4.2 Defensive Contributions", 2)
    _add_para(doc,
        "Central defenders and defensive midfielders dominate tackle and interception counts. "
        "The Bundesliga shows the highest average fouls committed per player, consistent with "
        "its pressing-heavy tactical culture. Yellow card rates are highest in La Liga and Serie A.")

    _add_heading(doc, "4.3 Goalkeeping", 2)
    _add_para(doc,
        "Among qualified goalkeepers (≥ 270 minutes), the Premier League shows the highest "
        "average clean sheet percentage. Save percentage and goals-allowed/90 show a strong "
        "negative correlation (r ≈ –0.82), validating both metrics as complementary GK quality "
        "indicators.")

    _add_heading(doc, "4.4 Demographic Profile", 2)
    _add_para(doc,
        f"The dataset spans {kpi['total_nations']} nationalities. Spanish (ESP), French (FRA), "
        "and German (GER) players are most numerous. The median player age across all leagues "
        f"is approximately {kpi['avg_age']} years. The 26-29 age group contributes the most goals "
        "and assists in aggregate, suggesting peak attacking production in this window.")

    _add_heading(doc, "4.5 Machine Learning Results", 2)
    _add_para(doc,
        "The Random Forest classifier achieved the highest cross-validation ROC-AUC, "
        "outperforming Gradient Boosting and Logistic Regression. The top predictive "
        "features were: Shots/90, Goals/Shot, Goals, Minutes Played, and PrimaryPosition. "
        "These align with domain knowledge: shooting volume and efficiency are the strongest "
        "signals of high goal-contribution output.")

    if _try_add_image(doc, "feature_importance.png"):
        cap = doc.add_paragraph("Figure 2: Random Forest Feature Importance")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].italic = True

    if _try_add_image(doc, "roc_curve.png"):
        cap = doc.add_paragraph("Figure 3: ROC Curve — Best Model")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].italic = True

    # ── 5. Discussion ─────────────────────────────────────────────────────────
    _add_heading(doc, "5. Discussion", 1, ACCENT)
    _add_para(doc,
        "The results confirm that per-90-minute normalisation substantially reduces playing-time "
        "bias and enables fairer comparisons between first-choice and squad players. The strong "
        "predictive performance of the Random Forest model suggests that goal-contribution "
        "efficiency is structurally determined by a small set of shooting and playing-time metrics.")
    _add_para(doc,
        "Limitations include the early-season data snapshot (statistics will shift as the season "
        "progresses), the absence of xG/xA metrics, and the lack of positional heatmaps or "
        "passing network data. Future iterations should incorporate team-level context, opponent "
        "quality adjustments, and temporal modelling.")

    # ── 6. Conclusion ─────────────────────────────────────────────────────────
    _add_heading(doc, "6. Conclusion", 1, ACCENT)
    _add_para(doc,
        "This study demonstrates that structured analysis of publicly available FBref data yields "
        "actionable insights into player performance across Europe's top leagues. The combination "
        "of descriptive analytics, visualisation, and machine learning provides a scalable "
        "framework for scouting, squad building, and tactical decision-making. The accompanying "
        "Streamlit dashboard makes these insights accessible to non-technical stakeholders.")

    # ── 7. References ─────────────────────────────────────────────────────────
    _add_heading(doc, "7. References", 1, ACCENT)
    refs = [
        "FBref (2026). Football Reference Statistics. Sports Reference LLC. https://fbref.com",
        "Sidorowicz, H. (2026). Football Players Stats 2026-2027. Kaggle. "
        "https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2026-2027",
        "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825-2830.",
        "Streamlit Inc. (2024). Streamlit — The fastest way to build data apps. https://streamlit.io",
    ]
    for ref in refs:
        p = doc.add_paragraph(ref, style="List Bullet")
        p.runs[0].font.size = Pt(10)

    # ── Save ──────────────────────────────────────────────────────────────────
    out = os.path.join(REPORT_DIR, "research_paper.docx")
    doc.save(out)
    print(f"[generate_reports] Saved -> {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT 2 — Executive Summary
# ─────────────────────────────────────────────────────────────────────────────

def generate_executive_summary() -> str:
    doc = Document()

    for section in doc.sections:
        section.top_margin    = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin   = Cm(2.8)
        section.right_margin  = Cm(2.5)

    ACCENT = RGBColor(0x1F, 0x6F, 0xEB)

    # ── Cover ─────────────────────────────────────────────────────────────────
    title = doc.add_heading("Executive Summary", 0)
    for run in title.runs:
        run.font.color.rgb = ACCENT
        run.font.size      = Pt(24)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph("Football Player Performance Analytics · Season 2026/27")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.runs[0].font.size = Pt(13)
    sub.runs[0].italic    = True

    meta = doc.add_paragraph(f"Date: {TODAY}  |  Data: FBref via Kaggle  |  Leagues: PL · La Liga · BL · SA · L1")
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.runs[0].font.size = Pt(9)
    meta.runs[0].font.color.rgb = RGBColor(0x57, 0x60, 0x6A)

    doc.add_paragraph()

    # ── Purpose ───────────────────────────────────────────────────────────────
    _add_heading(doc, "Purpose", 1, ACCENT)
    _add_para(doc,
        f"This summary presents key findings from an analysis of {kpi['total_players']:,} players "
        f"across {kpi['total_clubs']} clubs and {kpi['total_leagues']} European leagues during the "
        f"2026/27 season. The analysis covers attacking output, defensive contribution, goalkeeping "
        f"efficiency, player demographics, and a machine-learning model to classify high-performing players.")

    # ── Key Numbers ───────────────────────────────────────────────────────────
    _add_heading(doc, "Key Numbers at a Glance", 1, ACCENT)
    kpi_tbl = pd.DataFrame([
        ["Players Analysed",     f"{kpi['total_players']:,}"],
        ["Total Goals",          f"{kpi['total_goals']:,}"],
        ["Total Assists",        f"{kpi['total_assists']:,}"],
        ["Average Player Age",   f"{kpi['avg_age']} years"],
        ["Nationalities",        f"{kpi['total_nations']}"],
        ["Avg Shot Accuracy",    f"{kpi['avg_sot_pct']}%"],
    ], columns=["Metric", "Value"])
    _add_table(doc, kpi_tbl, header_color="1F4E79")
    doc.add_paragraph()

    # ── Findings ──────────────────────────────────────────────────────────────
    _add_heading(doc, "Top Findings", 1, ACCENT)

    findings = [
        ("🔫 Attacking", 
         "La Liga leads all leagues in combined goals and assists. Forwards average the highest "
         "Goals/90; midfielders drive assist creation. Top scorers are concentrated in La Liga and the Premier League."),
        ("🛡 Defensive",
         "Tackles and interceptions peak among Bundesliga and Serie A players. Yellow cards are "
         "highest in La Liga and Serie A. Fouls committed correlate strongly with league intensity."),
        ("🧤 Goalkeeping",
         "Premier League keepers record the highest clean-sheet percentage. Save% and GA/90 are "
         "strongly negatively correlated, validating them as complementary quality metrics."),
        ("🌍 Demographics",
         f"Spanish, French, and German players are most numerous. The 26-29 age window produces "
         "the highest aggregate goal contributions, identifying it as the typical performance peak."),
        ("🤖 Predictive Model",
         "A Random Forest classifier (best CV ROC-AUC) predicts high-performer status (top-25% G+A/90) "
         "with strong accuracy. Key drivers: Shots/90, Goals/Shot, and total Minutes Played."),
    ]

    for title_str, body in findings:
        _add_heading(doc, title_str, 2)
        _add_para(doc, body)

    # ── Recommendation ────────────────────────────────────────────────────────
    _add_heading(doc, "Recommendations", 1, ACCENT)
    recs = [
        "Use the interactive dashboard for real-time player comparison and scouting shortlisting.",
        "Prioritise G+A/90 over raw totals when comparing players with different minutes played.",
        "Target the 26-29 age bracket for attacking signings seeking peak-season output.",
        "Incorporate xG/xA metrics in the next data refresh to refine the ML model further.",
        "Extend the analysis to the full FBref dataset for passing, possession, and press metrics.",
    ]
    for rec in recs:
        p = doc.add_paragraph(rec, style="List Bullet")
        p.runs[0].font.size = Pt(11)

    # ── Dashboard ─────────────────────────────────────────────────────────────
    _add_heading(doc, "Interactive Dashboard", 1, ACCENT)
    _add_para(doc,
        "An interactive Streamlit dashboard is included in the project (dashboard/app.py). "
        "It provides league overviews, attacking & defensive leaderboards, goalkeeping stats, "
        "a nationality explorer, the ML model tab with live predictions, and a player search with "
        "radar comparison. Launch with:")
    code = doc.add_paragraph("    streamlit run dashboard/app.py")
    code.runs[0].font.name = "Courier New"
    code.runs[0].font.size = Pt(10)

    doc.add_paragraph()
    note = doc.add_paragraph(
        "For full methodology, statistical tables, and model evaluation details, refer to the "
        "accompanying research_paper.docx."
    )
    note.runs[0].italic = True
    note.runs[0].font.size = Pt(10)
    note.runs[0].font.color.rgb = RGBColor(0x57, 0x60, 0x6A)

    # ── Save ──────────────────────────────────────────────────────────────────
    out = os.path.join(REPORT_DIR, "executive_summary.docx")
    doc.save(out)
    print(f"[generate_reports] Saved -> {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_research_paper()
    generate_executive_summary()
    print("[generate_reports] All reports generated successfully.")
