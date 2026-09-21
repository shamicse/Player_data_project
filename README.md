# ⚽ Player Data Analytics — 2026/27

**Comprehensive football player performance analytics for Europe's Top 5 Leagues.**  
Data sourced from [FBref via Kaggle](https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2026-2027) · Season 2026/27

---

## Project Structure

```
Player_data_project/
│
├── data/
│   └── Player_data.csv              # Raw dataset (2,034 players, 53 columns)
│
├── notebooks/
│   └── attrition_analysis.ipynb     # Full 7-step analytical workflow
│
├── dashboard/
│   ├── app.py                       # Streamlit web application (7 tabs)
│   ├── utils.py                     # Data cleaning & KPI functions
│   └── modeling.py                  # Predictive modeling pipeline
│
├── outputs/
│   ├── charts/                      # Exported visualisation PNGs
│   ├── cleaned_Player_data.csv      # Processed dataset
│   └── Player_data_model.joblib     # Trained Random Forest model
│
├── reports/
│   ├── research_paper.docx          # Full research paper
│   ├── executive_summary.docx       # Executive summary
│   └── generate_reports.py          # Script to regenerate .docx files
│
├── requirements.txt
├── README.md
├── SETUP.md
└── .gitignore
```

---

## Quick Start

```bash
# 1. Clone / navigate to project
cd Player_data_project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model & export charts
python dashboard/modeling.py

# 4. Generate Word reports
python reports/generate_reports.py

# 5. Launch the dashboard
streamlit run dashboard/app.py
```

---

## Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 📊 League Overview | Goals, assists, cards, player distribution per league |
| 🔫 Attacking | Top scorers/assisters, shot efficiency, age-group breakdown |
| 🛡 Defensive | Tackles, interceptions, fouls & discipline |
| 🧤 Goalkeeping | Save%, GA/90, clean sheets, qualified GK rankings |
| 🌍 Nations & Age | Nationality heatmap, age distributions |
| 🤖 ML Model | Feature importance, ROC curve, confusion matrix, live predictions |
| 🔍 Player Search | Full-text search + radar comparison for up to 3 players |

---

## Dataset Columns (53)

| Category | Columns |
|----------|---------|
| Identity | Player, Nation, Pos, Squad, Comp, Age, Born |
| Playing Time | MP, Starts, Min, 90s |
| Attacking | Gls, Ast, G+A, G-PK, PK, PKatt, G+A-PK |
| Shooting | Sh, SoT, SoT%, Sh/90, SoT/90, G/Sh, G/SoT |
| Defensive | TklW, Int, Crs, Fld, Fls, 2CrdY, CrdY, CrdR, OG |
| Goalkeeping | GA, GA90, SoTA, Saves, Save%, W, D, L, CS, CS% |

---

## Leagues Covered

- 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League
- 🇪🇸 La Liga
- 🇩🇪 Bundesliga
- 🇮🇹 Serie A
- 🇫🇷 Ligue 1

---

## ML Model

**Target:** `HighPerformer` — top-25th percentile G+A/90 among outfield players (≥ 3 appearances)  
**Algorithm:** Random Forest Classifier (best of RF / Gradient Boosting / Logistic Regression)  
**Validation:** 5-fold Stratified Cross-Validation · Metric: ROC-AUC  
**Top Features:** Shots/90, Goals/Shot, Total Goals, Minutes Played, Position

---

## License

Dataset © FBref / Sports Reference LLC.  
Code: MIT License.
