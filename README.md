---
title: Fraud Detection
emoji: 🔍
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

# 🔍 Credit Card Fraud Detection — End-to-End ML System

---

## 🌐 Live Demo

**[https://huggingface.co/spaces/vishaalsai29/fraud-detection](https://huggingface.co/spaces/vishaalsai29/fraud-detection)**

Fully deployed fraud detection system — enter transaction details and get real-time predictions.

---

## 🎯 Business Problem

Credit card fraud costs the global economy over **$32 billion annually**. Building a model to detect it sounds straightforward — until you look at the data.

Only **0.17% of transactions are fraudulent**. This extreme class imbalance means a naive model that predicts "legitimate" for every transaction achieves 99.83% accuracy while catching **zero fraud**. Standard ML pipelines optimized for accuracy completely fail here.

This system takes a different approach: **business cost minimization**. Instead of tuning for accuracy or even F1-score, every modeling decision — from the choice of algorithm to the classification threshold — is driven by the real-world financial impact of each type of error. Catching fraud that would have been missed is worth far more than avoiding a few unnecessary alerts.

---

## 📊 Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions | 492 fraud cases
- **Fraud rate:** 0.17% (highly imbalanced)
- **Features:** `Time`, `Amount`, and `V1`–`V28` (PCA-transformed for cardholder privacy)
- **Target:** `Class` — 1 = fraud, 0 = legitimate

---

## 🧪 Experiment Tracking (MLflow)

Three experiments were tracked under the `fraud-detection` MLflow experiment group:

| Experiment | Recall | ROC-AUC |
|---|---|---|
| Logistic Regression (baseline) | 91.84% | 97.22% |
| XGBoost Default | 84.69% | 96.52% |
| **XGBoost Tuned ✅** | **88.78%** | **98.02%** |

**Why XGBoost Tuned was selected** despite having lower recall than Logistic Regression:

- **Precision:** LR precision was only 6% — for every genuine fraud alert, it generated 15 false alarms. XGBoost Tuned achieves 64% precision at the optimal threshold, making alerts far more actionable for fraud analysts.
- **ROC-AUC:** XGBoost Tuned achieves 98.02% vs LR's 97.22%, meaning it produces better-calibrated probability scores across all possible thresholds — critical when threshold selection is part of the optimization.
- **Business cost:** At their respective optimal thresholds, XGBoost Tuned produces a lower total business cost than Logistic Regression due to the precision difference reducing false positive investigation overhead.

All three experiments are logged with parameters, metrics, and artifacts in MLflow. Run `mlflow ui` in the project root to explore the experiment dashboard.

---

## 💰 Business-Meaningful Evaluation

### Cost Assumptions

| Error Type | Business Impact | Cost |
|---|---|---|
| **False Negative** | Fraud transaction approved — loss to bank | **$500** |
| **False Positive** | Legitimate transaction flagged — analyst time | **$10** |

The asymmetry is stark: **missing one fraud is 50× more costly than a false alert.** This directly drives the threshold optimization below.

### Threshold Analysis

Rather than using the default 0.5 classification threshold, thresholds from 0.10 to 0.90 were evaluated against total business cost:

| Threshold | False Negatives | False Positives | Total Cost |
|---|---|---|---|
| 0.50 (default) | 17 | 13 | $8,630 |
| **0.10 (optimal) ✅** | **11** | **48** | **$5,980** |

**Business savings: $2,650** by using the optimal threshold instead of the default.

**Key insight:** Lowering the threshold from 0.50 → 0.10 catches **6 additional fraud cases** (saves $3,000) at the cost of **35 more false alerts** (costs $350) — a net saving of **$2,650 per test period.**

This is the kind of analysis that separates a production ML system from a notebook experiment.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Docker Container                   │
│                                                     │
│  ┌──────────────────┐    ┌───────────────────────┐  │
│  │  FastAPI Backend  │    │   Streamlit Frontend  │  │
│  │   Port 8002       │◄───│     Port 7860         │  │
│  │                   │    │                       │  │
│  │  POST /predict    │    │  Transaction Input    │  │
│  │  GET  /health     │    │  Real-time Results    │  │
│  │  GET  /model-info │    │  Model Insights Tab   │  │
│  └────────┬──────────┘    └───────────────────────┘  │
│           │                                          │
│  ┌────────▼──────────┐                               │
│  │   Model Artifacts  │                               │
│  │  xgboost_tuned.pkl │                               │
│  │  scaler.pkl        │                               │
│  │  eval_results.json │                               │
│  └───────────────────┘                               │
└─────────────────────────────────────────────────────┘
```

- **FastAPI backend** (port 8002) — serves `/predict`, `/health`, and `/model-info` endpoints; loads all artifacts at startup via `lifespan`
- **Streamlit frontend** (port 7860) — interactive UI with a transaction analyser tab and a model insights tab
- **XGBoost model + StandardScaler** — loaded once at startup, predictions served in milliseconds
- **Docker container** — single image runs both services via `start.sh`; models downloaded from GitHub at build time

---

## 🚀 Running Locally

```bash
git clone https://github.com/vishaalsai/fraud-detection
cd fraud-detection
pip install -r requirements.txt
```

**Terminal 1 — API backend:**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8002 --reload
```

**Terminal 2 — Streamlit UI:**
```bash
streamlit run app/streamlit_app.py
```

Open **http://localhost:8501** in your browser.

**API docs (Swagger UI):** http://localhost:8002/docs

---

## 📡 Monitoring Plan

**Performance Degradation Detection**

In production, model performance is monitored by logging every prediction with its actual outcome (confirmed fraud/legitimate) once ground truth becomes available — typically 24–48 hours after the transaction. Key metrics tracked on a rolling 7-day window: precision, recall, F1, and false negative rate. If recall drops below 85% or the false negative rate increases by more than 3 percentage points from baseline, an automated alert is triggered for model review. Tools: Evidently AI for metric dashboards, custom logging middleware in FastAPI.

**Data Drift Detection**

Since V1–V28 are PCA-transformed features, drift is monitored at the distribution level using Population Stability Index (PSI) on each feature. A PSI > 0.2 on any feature triggers a drift alert. Additionally, the distribution of predicted probabilities is monitored daily — a significant shift in the score distribution (e.g., sudden spike in high-probability fraud scores) indicates potential data quality issues upstream. Evidently AI's `DataDriftPreset` would be used for automated drift reports generated weekly.

**Retraining Triggers**

Retraining is triggered by any of three conditions: (1) recall drops below 85% on the rolling 7-day window, (2) PSI > 0.2 detected on 3 or more features simultaneously, or (3) scheduled monthly retraining regardless of performance metrics to incorporate recent fraud patterns. The retraining pipeline uses the same `src/train.py` script with updated data, and MLflow experiment tracking ensures every retrained model is logged and compared against the current production model before deployment. A new model is only promoted to production if it achieves equal or better ROC-AUC and recall on a held-out validation set.

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Data Processing | Pandas, NumPy, Scikit-learn |
| Class Imbalance | imbalanced-learn |
| Model | XGBoost |
| Experiment Tracking | MLflow |
| API Backend | FastAPI + Uvicorn |
| Frontend UI | Streamlit |
| Containerization | Docker |
| Deployment | Hugging Face Spaces |
| Monitoring (planned) | Evidently AI |

---

## 📁 Project Structure

```
fraud-detection/
├── data/
│   └── raw/                   # creditcard.csv (gitignored)
├── notebooks/                 # exploratory notebooks
├── src/
│   ├── __init__.py
│   ├── train.py               # data loading, preprocessing, 3 MLflow experiments
│   ├── evaluate.py            # threshold analysis, business cost optimization, plots
│   └── predict.py             # inference utilities
├── api/
│   ├── __init__.py
│   └── main.py                # FastAPI app — /predict, /health, /model-info
├── app/
│   └── streamlit_app.py       # Streamlit UI — transaction analyser + model insights
├── models/                    # saved artifacts (gitignored except evaluation JSON)
│   ├── xgboost_tuned.pkl
│   ├── scaler.pkl
│   └── evaluation_results.json
├── mlruns/                    # MLflow tracking (gitignored)
├── Dockerfile                 # builds single container for HF Spaces
├── start.sh                   # starts uvicorn + streamlit inside container
├── requirements.txt
├── .env                       # secrets (gitignored)
└── README.md
```

---

## 👤 Author

**Vishaal Sai** | [github.com/vishaalsai](https://github.com/vishaalsai)
