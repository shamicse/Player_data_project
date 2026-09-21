# Setup Guide — Player Data Analytics 2026/27

Complete instructions for cloning, setting up, and running the project locally or from GitHub.

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | ≥ 3.10 | https://python.org |
| pip | ≥ 23 | bundled with Python |
| Git | any | https://git-scm.com |

---

## Option A · Clone from GitHub

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Player_data_project.git
cd Player_data_project

# 2. (Recommended) Create a virtual environment
python -m venv .venv

# Activate — Windows
.venv\Scripts\activate

# Activate — macOS / Linux
source .venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt
```

---

## Option B · Run from existing folder

```bash
cd Player_data_project
pip install -r requirements.txt
```

---

## Step-by-Step: First Run

### 1. Verify data is present
```
data/Player_data.csv   ← must exist (2,034 rows)
```

### 2. Train the ML model & export charts
```bash
python dashboard/modeling.py
```
This will create:
- `outputs/Player_data_model.joblib`
- `outputs/charts/feature_importance.png`
- `outputs/charts/confusion_matrix.png`
- `outputs/charts/roc_curve.png`
- `outputs/charts/cv_comparison.png`

### 3. Generate Word reports
```bash
python reports/generate_reports.py
```
This will create:
- `reports/research_paper.docx`
- `reports/executive_summary.docx`

### 4. Launch the Streamlit dashboard
```bash
streamlit run dashboard/app.py
```
Opens at: **http://localhost:8501**

### 5. Run the Jupyter notebook (optional)
```bash
jupyter notebook notebooks/attrition_analysis.ipynb
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: Player_data.csv` | Ensure you run commands from `Player_data_project/` root |
| Model tab shows warning | Run `python dashboard/modeling.py` first |
| Reports missing images | Run `python dashboard/modeling.py` before `generate_reports.py` |
| Port 8501 in use | Run `streamlit run dashboard/app.py --server.port 8502` |

---

## Project Commands Reference

```bash
# Full pipeline (from project root)
pip install -r requirements.txt
python dashboard/modeling.py
python reports/generate_reports.py
streamlit run dashboard/app.py
```

---

## Publishing to GitHub

```bash
cd Player_data_project
git init
git add .
git commit -m "Initial commit — Player Data Analytics 2026/27"
git branch -M main
git remote add origin https://github.com/<your-username>/Player_data_project.git
git push -u origin main
```

> **Note:** The `.gitignore` excludes `__pycache__`, `.venv`, `*.joblib`, and `outputs/charts/*.png`  
> so only source code and data are tracked. Re-run the pipeline after cloning to regenerate outputs.
