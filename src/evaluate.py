import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_PATH   = "data/raw/creditcard.csv"
MODEL_PATH  = "models/xgboost_tuned.pkl"
SCALER_PATH = "models/scaler.pkl"
MODELS_DIR  = "models"

COST_FALSE_NEGATIVE = 500   # avg fraud amount missed  ($)
COST_FALSE_POSITIVE = 10    # cost to investigate alert ($)

THRESHOLD_MIN  = 0.10
THRESHOLD_MAX  = 0.91
THRESHOLD_STEP = 0.05


# ─────────────────────────────────────────────
# 1. LOAD MODEL AND DATA
# ─────────────────────────────────────────────
print("=" * 65)
print("STEP 1 -- LOAD MODEL AND DATA")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
print(f"Data loaded: {df.shape}")

# Recreate the exact same scaling used in train.py
scaler = joblib.load(SCALER_PATH)

# Scale Amount first (scaler was fit on Amount, then Time sequentially)
# We reproduce the same two-column scaling from train.py
scaler_amount = StandardScaler()
scaler_time   = StandardScaler()
df["Amount_scaled"] = scaler_amount.fit_transform(df[["Amount"]])
df["Time_scaled"]   = scaler_time.fit_transform(df[["Time"]])
df.drop(columns=["Amount", "Time"], inplace=True)

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Test set size: {X_test.shape[0]} samples  "
      f"({y_test.sum()} fraud, {(y_test == 0).sum()} legit)")

model = joblib.load(MODEL_PATH)
print(f"Model loaded: {MODEL_PATH}")

y_prob = model.predict_proba(X_test)[:, 1]
print(f"Predicted probabilities computed.")


# ─────────────────────────────────────────────
# 2. THRESHOLD ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2 -- THRESHOLD ANALYSIS")
print("=" * 65)

thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP)
threshold_rows = []

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    threshold_rows.append({
        "threshold":        round(t, 2),
        "precision":        round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":           round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":               round(f1_score(y_test, y_pred, zero_division=0), 4),
        "false_positives":  int(fp),
        "false_negatives":  int(fn),
    })

threshold_df = pd.DataFrame(threshold_rows)
print(threshold_df.to_string(index=False))


# ─────────────────────────────────────────────
# 3. BUSINESS COST ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3 -- BUSINESS COST ANALYSIS")
print(f"         FN cost = ${COST_FALSE_NEGATIVE}  |  FP cost = ${COST_FALSE_POSITIVE}")
print("=" * 65)

threshold_df["total_business_cost"] = (
    threshold_df["false_negatives"] * COST_FALSE_NEGATIVE +
    threshold_df["false_positives"] * COST_FALSE_POSITIVE
)

# Optimal threshold = lowest business cost
best_idx      = threshold_df["total_business_cost"].idxmin()
optimal_row   = threshold_df.loc[best_idx]
optimal_thr   = optimal_row["threshold"]

# Default threshold (0.5) metrics
default_pred = (y_prob >= 0.5).astype(int)
tn_d, fp_d, fn_d, tp_d = confusion_matrix(y_test, default_pred, labels=[0, 1]).ravel()
cost_default  = int(fn_d) * COST_FALSE_NEGATIVE + int(fp_d) * COST_FALSE_POSITIVE

cost_optimal  = int(optimal_row["total_business_cost"])
savings       = cost_default - cost_optimal

print(
    f"Default threshold (0.50): "
    f"FN={int(fn_d)}, FP={int(fp_d)}, Total Cost=${cost_default:,}"
)
print(
    f"Optimal threshold ({optimal_thr:.2f}): "
    f"FN={int(optimal_row['false_negatives'])}, "
    f"FP={int(optimal_row['false_positives'])}, "
    f"Total Cost=${cost_optimal:,}"
)
print(f"Business savings by using optimal threshold: ${savings:,}")


# ─────────────────────────────────────────────
# 4. FINAL EVALUATION AT OPTIMAL THRESHOLD
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"STEP 4 -- FINAL EVALUATION  (threshold = {optimal_thr:.2f})")
print("=" * 65)

y_pred_final = (y_prob >= optimal_thr).astype(int)
tn_f, fp_f, fn_f, tp_f = confusion_matrix(y_test, y_pred_final, labels=[0, 1]).ravel()

recall_final    = recall_score(y_test, y_pred_final, zero_division=0)
precision_final = precision_score(y_test, y_pred_final, zero_division=0)
f1_final        = f1_score(y_test, y_pred_final, zero_division=0)
roc_auc_final   = roc_auc_score(y_test, y_prob)
fpr_final       = fp_f / (fp_f + tn_f)   # FP / (FP + TN)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_final, target_names=["Not Fraud", "Fraud"]))

print("Confusion Matrix:")
cm_df = pd.DataFrame(
    [[tn_f, fp_f], [fn_f, tp_f]],
    index=["Actual: Not Fraud", "Actual: Fraud"],
    columns=["Predicted: Not Fraud", "Predicted: Fraud"]
)
print(cm_df)

print(f"\n  Fraud Detection Rate (Recall) : {recall_final:.4f}  ({recall_final*100:.2f}%)")
print(f"  False Positive Rate           : {fpr_final:.4f}  ({fpr_final*100:.2f}%)")
print(f"  Precision                     : {precision_final:.4f}")
print(f"  ROC-AUC Score                 : {roc_auc_final:.4f}")
print(f"  Total Business Cost           : ${int(optimal_row['total_business_cost']):,}")


# ─────────────────────────────────────────────
# 5. SAVE RESULTS TO JSON
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5 -- SAVE RESULTS")
print("=" * 65)

results = {
    "optimal_threshold":    float(optimal_thr),
    "recall":               round(recall_final, 6),
    "precision":            round(precision_final, 6),
    "f1":                   round(f1_final, 6),
    "roc_auc":              round(roc_auc_final, 6),
    "false_negatives":      int(fn_f),
    "false_positives":      int(fp_f),
    "total_business_cost":  int(optimal_row["total_business_cost"]),
    "cost_false_negative":  COST_FALSE_NEGATIVE,
    "cost_false_positive":  COST_FALSE_POSITIVE,
}

results_path = os.path.join(MODELS_DIR, "evaluation_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Saved evaluation results -> {results_path}")
print(json.dumps(results, indent=4))


# ─────────────────────────────────────────────
# 6. GENERATE AND SAVE PLOTS
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6 -- GENERATE PLOTS")
print("=" * 65)

# ── Plot 1: Precision & Recall vs Threshold ──
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(threshold_df["threshold"], threshold_df["precision"],
        marker="o", markersize=4, label="Precision", color="steelblue")
ax.plot(threshold_df["threshold"], threshold_df["recall"],
        marker="s", markersize=4, label="Recall",    color="darkorange")

ax.axvline(x=optimal_thr, color="red", linestyle="--", linewidth=1.5,
           label=f"Optimal Threshold ({optimal_thr:.2f})")

ax.set_title("Precision & Recall vs Classification Threshold", fontsize=13, fontweight="bold")
ax.set_xlabel("Threshold", fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_xlim(THRESHOLD_MIN - 0.02, THRESHOLD_MAX)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plot1_path = os.path.join(MODELS_DIR, "precision_recall_vs_threshold.png")
plt.tight_layout()
plt.savefig(plot1_path, dpi=150)
plt.close()
print(f"Saved -> {plot1_path}")

# ── Plot 2: Business Cost vs Threshold ──
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(threshold_df["threshold"], threshold_df["total_business_cost"],
        marker="o", markersize=4, color="crimson", label="Total Business Cost")

ax.axvline(x=optimal_thr, color="red", linestyle="--", linewidth=1.5,
           label=f"Min Cost Threshold ({optimal_thr:.2f})")

ax.set_title("Total Business Cost vs Classification Threshold", fontsize=13, fontweight="bold")
ax.set_xlabel("Threshold", fontsize=11)
ax.set_ylabel("Total Business Cost ($)", fontsize=11)
ax.set_xlim(THRESHOLD_MIN - 0.02, THRESHOLD_MAX)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plot2_path = os.path.join(MODELS_DIR, "business_cost_vs_threshold.png")
plt.tight_layout()
plt.savefig(plot2_path, dpi=150)
plt.close()
print(f"Saved -> {plot2_path}")

print("\nEvaluation complete.")
