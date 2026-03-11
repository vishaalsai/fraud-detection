import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_PATH   = "data/raw/creditcard.csv"
MODELS_DIR  = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

mlflow.set_tracking_uri("mlruns")          # use local file store, avoids URL-encoding issues
mlflow.set_experiment("fraud-detection")


def compute_metrics(y_true, y_pred, y_prob):
    """Return a dict of all classification metrics."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — LOADING DATA")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"\nClass distribution:")
print(df["Class"].value_counts())
print(f"\nFraud rate: {df['Class'].mean()*100:.4f}%")


# ─────────────────────────────────────────────
# 2. PREPROCESS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — PREPROCESSING")
print("=" * 60)

scaler = StandardScaler()

# Scale Amount and Time; drop originals
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
df["Time_scaled"]   = scaler.fit_transform(df[["Time"]])
df.drop(columns=["Amount", "Time"], inplace=True)

# Features and target
X = df.drop(columns=["Class"])
y = df["Class"]

print(f"Features: {X.shape[1]} columns")
print(f"Target: {y.value_counts().to_dict()}")

# Train/test split — stratified to preserve fraud ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# Class imbalance ratio (used for XGBoost)
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight = n_neg / n_pos
print(f"scale_pos_weight: {scale_pos_weight:.2f}  ({n_neg} negatives / {n_pos} positives)")

# Summary table accumulator
results = []


# ─────────────────────────────────────────────
# 3. EXPERIMENT 1 — Logistic Regression Baseline
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXPERIMENT 1 — logistic_regression_baseline")
print("=" * 60)

with mlflow.start_run(run_name="logistic_regression_baseline"):
    params_lr = {
        "model_type":   "LogisticRegression",
        "max_iter":     1000,
        "class_weight": "balanced",
    }
    mlflow.log_params(params_lr)

    model_lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    model_lr.fit(X_train, y_train)

    y_pred_lr = model_lr.predict(X_test)
    y_prob_lr = model_lr.predict_proba(X_test)[:, 1]

    metrics_lr = compute_metrics(y_test, y_pred_lr, y_prob_lr)
    mlflow.log_metrics(metrics_lr)

    print(classification_report(y_test, y_pred_lr, target_names=["Not Fraud", "Fraud"]))
    print(f"ROC-AUC: {metrics_lr['roc_auc']:.4f}")

    results.append({
        "experiment":  "logistic_regression_baseline",
        "recall":      metrics_lr["recall"],
        "roc_auc":     metrics_lr["roc_auc"],
    })


# ─────────────────────────────────────────────
# 3. EXPERIMENT 2 — XGBoost Default
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXPERIMENT 2 — xgboost_default")
print("=" * 60)

with mlflow.start_run(run_name="xgboost_default"):
    params_xgb_default = {
        "model_type":        "XGBClassifier",
        "scale_pos_weight":  round(scale_pos_weight, 4),
    }
    mlflow.log_params(params_xgb_default)

    model_xgb_default = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )
    model_xgb_default.fit(X_train, y_train)

    y_pred_xgb_d = model_xgb_default.predict(X_test)
    y_prob_xgb_d = model_xgb_default.predict_proba(X_test)[:, 1]

    metrics_xgb_d = compute_metrics(y_test, y_pred_xgb_d, y_prob_xgb_d)
    mlflow.log_metrics(metrics_xgb_d)

    print(classification_report(y_test, y_pred_xgb_d, target_names=["Not Fraud", "Fraud"]))
    print(f"ROC-AUC: {metrics_xgb_d['roc_auc']:.4f}")

    results.append({
        "experiment": "xgboost_default",
        "recall":     metrics_xgb_d["recall"],
        "roc_auc":    metrics_xgb_d["roc_auc"],
    })


# ─────────────────────────────────────────────
# 3. EXPERIMENT 3 — XGBoost Tuned  (saved model)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXPERIMENT 3 — xgboost_tuned")
print("=" * 60)

with mlflow.start_run(run_name="xgboost_tuned"):
    params_xgb_tuned = {
        "model_type":        "XGBClassifier",
        "n_estimators":      300,
        "max_depth":         6,
        "learning_rate":     0.05,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "scale_pos_weight":  round(scale_pos_weight, 4),
    }
    mlflow.log_params(params_xgb_tuned)

    model_xgb_tuned = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )
    model_xgb_tuned.fit(X_train, y_train)

    y_pred_xgb_t = model_xgb_tuned.predict(X_test)
    y_prob_xgb_t = model_xgb_tuned.predict_proba(X_test)[:, 1]

    metrics_xgb_t = compute_metrics(y_test, y_pred_xgb_t, y_prob_xgb_t)
    mlflow.log_metrics(metrics_xgb_t)

    print(classification_report(y_test, y_pred_xgb_t, target_names=["Not Fraud", "Fraud"]))
    print(f"ROC-AUC: {metrics_xgb_t['roc_auc']:.4f}")

    # ── Save model and scaler artifacts ──────
    model_path  = os.path.join(MODELS_DIR, "xgboost_tuned.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

    joblib.dump(model_xgb_tuned, model_path)
    joblib.dump(scaler, scaler_path)

    mlflow.log_artifact(model_path)
    mlflow.log_artifact(scaler_path)

    print(f"\nSaved model  -> {model_path}")
    print(f"Saved scaler -> {scaler_path}")

    results.append({
        "experiment": "xgboost_tuned",
        "recall":     metrics_xgb_t["recall"],
        "roc_auc":    metrics_xgb_t["roc_auc"],
    })


# ─────────────────────────────────────────────
# 4. SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY — All Experiments")
print("=" * 60)

summary = pd.DataFrame(results)
summary["recall"]  = summary["recall"].map("{:.4f}".format)
summary["roc_auc"] = summary["roc_auc"].map("{:.4f}".format)
summary.index = summary.index + 1  # 1-based index

print(summary.to_string(index=True))
print("\nDone. MLflow UI: run  mlflow ui  and open http://127.0.0.1:5000")
