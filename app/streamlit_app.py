import requests
import streamlit as st

API_BASE = "http://localhost:8002"

# ─────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔍",
    layout="wide",
)

# ─────────────────────────────────────────────
# 2. HEADER
# ─────────────────────────────────────────────
st.title("🔍 Credit Card Fraud Detection System")
st.markdown(
    "**Real-time transaction analysis powered by XGBoost (ROC-AUC: 0.98)**"
)
st.info(
    "This system analyzes credit card transactions and flags potentially fraudulent "
    "activity. The model uses a business-optimized threshold of **0.10** to minimize "
    "financial losses."
)

# ─────────────────────────────────────────────
# 3. SIDEBAR — Model Info
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Model Performance")
    st.metric("ROC-AUC",              "0.98")
    st.metric("Fraud Detection Rate", "88.78%")
    st.metric("False Positive Rate",  "0.08%")
    st.metric("Optimal Threshold",    "0.10")

    st.divider()

    st.header("Business Impact")
    st.metric("Cost per Missed Fraud",  "$500")
    st.metric("Cost per False Alert",   "$10")

# ─────────────────────────────────────────────
# 4. MAIN TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Analyze Transaction", "📊 Model Insights"])


# ══════════════════════════════════════════════
# TAB 1 — Analyze Transaction
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Enter Transaction Details")

    left, right = st.columns(2)

    with left:
        st.markdown("**Transaction Metadata**")
        time_val   = st.number_input("Time (seconds since first transaction)",
                                     min_value=0.0,     max_value=172800.0,
                                     value=50000.0,     step=1.0)
        amount_val = st.number_input("Amount ($)",
                                     min_value=0.01,    max_value=25000.0,
                                     value=100.0,       step=0.01)

        st.markdown("**PCA Components V1 – V14**")
        v_left = {}
        for i in range(1, 15):
            v_left[f"V{i}"] = st.number_input(
                f"V{i}", min_value=-10.0, max_value=10.0,
                value=0.0, step=0.1, key=f"v{i}"
            )

    with right:
        st.markdown("**PCA Components V15 – V28**")
        v_right = {}
        for i in range(15, 29):
            v_right[f"V{i}"] = st.number_input(
                f"V{i}", min_value=-10.0, max_value=10.0,
                value=0.0, step=0.1, key=f"v{i}"
            )

    st.divider()

    # ── Centered analyse button ──
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        analyse = st.button("🔍 Analyze Transaction", use_container_width=True, type="primary")

    if analyse:
        payload = {
            "Time":   time_val,
            "Amount": amount_val,
            **v_left,
            **v_right,
        }

        try:
            resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()

            prediction  = result["prediction"]
            probability = result["probability"]
            risk_level  = result["risk_level"]
            threshold   = result["threshold_used"]

            st.divider()
            st.subheader("Analysis Result")

            if prediction == "fraud":
                st.error(
                    "### 🚨 FRAUDULENT TRANSACTION DETECTED\n\n"
                    "**Recommended Action:** Block transaction and alert cardholder immediately."
                )
            else:
                st.success(
                    "### ✅ TRANSACTION APPEARS LEGITIMATE\n\n"
                    "**Recommended Action:** Approve transaction."
                )

            # Metrics row
            m1, m2, m3 = st.columns(3)
            m1.metric("Fraud Probability", f"{probability:.2%}")
            m2.metric("Risk Level",        risk_level)
            m3.metric("Threshold Used",    threshold)

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to API. Please start the FastAPI server first.\n\n"
                "```\nuvicorn api.main:app --host 0.0.0.0 --port 8002 --reload\n```"
            )
        except requests.exceptions.HTTPError as e:
            st.error(f"API returned an error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")


# ══════════════════════════════════════════════
# TAB 2 — Model Insights
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Model Performance & Business Analysis")

    try:
        info = requests.get(f"{API_BASE}/model-info", timeout=10).json()

        # ── Model metrics ──
        st.markdown("### Model Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC",           f"{info['roc_auc']:.4f}")
        c2.metric("Recall (Fraud)",     f"{info['recall']:.4f}")
        c3.metric("Precision",          f"{info['precision']:.4f}")
        c4.metric("Optimal Threshold",  info["optimal_threshold"])

        st.divider()

        # ── Business cost breakdown ──
        st.markdown("### Business Cost Breakdown")

        cost_fn     = info["false_negative_cost"]   # $500
        cost_fp     = info["false_positive_cost"]    # $10

        breakdown = {
            "Scenario":         ["Default Threshold (0.50)", "Optimal Threshold (0.10)"],
            "False Negatives":  [17,     11],
            "False Positives":  [13,     48],
            "Total Cost ($)":   [
                17 * cost_fn + 13 * cost_fp,   # $8,630
                11 * cost_fn + 48 * cost_fp,   # $5,980
            ],
        }

        import pandas as pd
        df_costs = pd.DataFrame(breakdown)
        st.dataframe(df_costs, use_container_width=True, hide_index=True)

        savings = breakdown["Total Cost ($)"][0] - breakdown["Total Cost ($)"][1]
        st.success(f"### Business savings by using optimal threshold: **${savings:,}**")

        st.divider()

        # ── Per-error cost reminder ──
        st.markdown("### Cost Assumptions")
        ca, cb = st.columns(2)
        ca.metric("Cost per Missed Fraud (FN)", f"${cost_fn}")
        cb.metric("Cost per False Alert (FP)",  f"${cost_fp}")

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to API. Please start the FastAPI server first.\n\n"
            "```\nuvicorn api.main:app --host 0.0.0.0 --port 8002 --reload\n```"
        )
    except Exception as e:
        st.error(f"Could not load model info: {e}")

# ─────────────────────────────────────────────
# 5. FOOTER
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:grey; font-size:0.85em;'>"
    "Built with XGBoost + FastAPI + Streamlit &nbsp;|&nbsp; "
    "Portfolio Project by Vishaal Sai"
    "</div>",
    unsafe_allow_html=True,
)
