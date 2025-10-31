#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telco Customer Churn Survival Analyzer
Author: Aniket
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
import os

# -------------------------------
# Page Config & Styling
# -------------------------------
st.set_page_config(
    page_title="Telco Churn Survival Analyzer",
    page_icon="chart_with_downwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {font-size: 2.5rem; color: #1E3A8A; font-weight: bold; text-align: center; margin-bottom: 0.5rem;}
    .sub-header {font-size: 1.3rem; color: #374151; text-align: center; margin-bottom: 2rem;}
    .metric-box {
        background-color: #F3F4F6; 
        padding: 1rem; 
        border-radius: 10px; 
        text-align: center !important; 
        color: black !important;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .metric-box h3 {
        color: #1E3A8A !important; 
        margin: 0.2rem 0;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
    }
    .metric-box p {
        color: #4B5563 !important; 
        margin: 0.1rem 0; 
        font-size: 0.9rem;
    }
    .stButton>button {background-color: #1E40AF; color: white; font-weight: bold;}
    .stButton>button:hover {background-color: #1E3A8A;}
    .footer {text-align: center; margin-top: 3rem; color: #6B7280; font-size: 0.9rem;}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------
# Feature engineering (avg_monthly_spend removed)
# -------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Only keep the necessary engineered feature: autopayment
    if "paymentmethod" in df.columns:
        df["autopayment"] = df["paymentmethod"]\
            .apply(lambda x: 1 if "automatic" in str(x).lower() else 0)
    
    # Removed: df["avg_monthly_spend"] = df["totalcharges"] / (df["tenure"].replace(0, 1))
    
    return df

# -------------------------------
# Load Model & Training Data
# -------------------------------
@st.cache_resource
def load_model():
    model_path = 'Project_Resources/project 2 cph_model.pkl'
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_train_data():
    train_path = "Project_Resources/project 2 df_train.parquet"
    if not os.path.exists(train_path):
        st.error(f"Training data not found: {train_path}")
        st.stop()
    
    return pd.read_parquet(train_path)

cph = load_model()
df_train = load_train_data()

# -------------------------------
# Model columns
# -------------------------------
try:
    MODEL_COLUMNS = list(cph.summary.index)
except Exception as e:
    st.error(f"Cannot extract model columns: {e}")
    st.stop()

# -------------------------------
# Align new customer (applies engineering + one-hot encoding)
# -------------------------------
def align_customer_input(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = engineer_features(raw_df) # Applies engineering to raw input
    encoded = pd.get_dummies(df, drop_first=False)
    if isinstance(encoded.columns, pd.MultiIndex):
        encoded.columns = ['_'.join(col).strip() \
                           for col in encoded.columns.values]
    else:
        encoded.columns = encoded.columns.astype(str)
    # Reindex to match the exact columns the model was trained on
    return encoded.reindex(columns=MODEL_COLUMNS, fill_value=0)

# -------------------------------
# Align training data (already encoded/engineered)
# -------------------------------
def align_training_data(df: pd.DataFrame) -> pd.DataFrame:
    # This aligns the features to the columns the model expects
    return df.reindex(columns=MODEL_COLUMNS, fill_value=0)

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("New Customer Profile")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        phone = st.selectbox("Phone Service", ["No", "Yes"])

    with col2:
        multiple_lines = st.selectbox("Multiple Lines",
                                      ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service",
                                ["DSL", "Fiber optic", "No"])
        online_sec = st.selectbox("Online Security",
                                  ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup",
                                     ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support",
                                    ["No", "Yes", "No internet service"])

    col3, col4 = st.columns(2)
    with col3:
        streaming_tv = st.selectbox("Streaming TV",
                                    ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies",
                                        ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month",
                                             "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing",
                                 ["No", "Yes"])

    with col4:
        payment = st.selectbox("Payment Method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0, 0.5)
        total = st.slider("Total Charges ($)", 18.0, 9000.0, 800.0, 50.0)
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    if st.button("Predict Churn Risk"):
        raw_input = pd.DataFrame({
            "gender": [gender],
            "seniorcitizen": [1 if senior == "Yes" else 0],
            "partner": [partner],
            "dependents": [dependents],
            "phoneservice": [phone],
            "multiplelines": [multiple_lines],
            "internetservice": [internet],
            "onlinesecurity": [online_sec],
            "onlinebackup": [online_backup],
            "techsupport": [tech_support],
            "streamingtv": [streaming_tv],
            "streamingmovies": [streaming_movies],
            "contract": [contract],
            "paperlessbilling": [paperless],
            "paymentmethod": [payment],
            "monthlycharges": [monthly],
            "totalcharges": [total],
            "tenure": [tenure],
        })

        try:
            X = align_customer_input(raw_input)
            if X.ndim == 1:
                X = X.values.reshape(1, -1)

            # Predictions
            risk_score = float(cph.predict_partial_hazard(X).iloc[0])
            times = np.arange(1, 73)
            surv_func = cph.predict_survival_function(X, times=times)
            prob_series = surv_func.iloc[:, 0]
            median_time = prob_series[prob_series <= 0.5]\
                .index.min() if (prob_series <= 0.5).any() else np.inf

            # ---- RISK GROUP – FINAL STRATEGY --------------------------------
            try:
                # 1. Drop the survival columns (duration/event)
                train_X_clean = df_train.drop(columns=["duration", "event"],
                                              errors="ignore").copy()

                # 2. **Ensure Engineered Feature Exists**: 
                # Apply engineer_features to the training set for 'autopayment'
                train_X_with_eng_features = engineer_features(train_X_clean)
                
                # Check if we need to one-hot-encode again (if df_train had
                # raw categoricals)
                # This logic is based on checking for a raw categorical column
                # (e.g., 'gender')
                if 'gender' in train_X_with_eng_features.columns:
                    train_X_with_eng_features = pd.get_dummies(
                        train_X_with_eng_features, drop_first=True)
                
                # 3. Align the engineered training data with the model columns
                train_X_aligned = align_training_data(
                    train_X_with_eng_features)

                # 4. Predict hazard on the whole training set → 1-D array
                train_risk = cph.predict_partial_hazard(
                    train_X_aligned).values.ravel()

                # 5. Quantiles (33 % / 66 %)
                q1 = np.quantile(train_risk, 1/3) \
                    if train_risk.sum() > 0 else 0
                q2 = np.quantile(train_risk, 2/3) \
                    if train_risk.sum() > 0 else 0
                
                # Safety check 
                if q1 == 0 and q2 == 0:
                     q1, q2 = 0.5, 1.5

                # 6. Classify the new customer
                risk_group = (
                    "Low"    if risk_score <= q1 else
                    "Medium" if risk_score <= q2 else
                    "High"
                )
            except Exception as e:
                st.warning(f"Risk group calculation failed: {e}")
                risk_group = "Unknown"
            # -----------------------------------------------------------------

            st.session_state.prediction = {
                "surv_func": surv_func,
                "median_time": median_time,
                "risk_score": risk_score,
                "risk_group": risk_group,
                "prob_24": prob_series.loc[24] \
                    if 24 in prob_series.index else prob_series.iloc[-1],
            }
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

# -------------------------------
# Dashboard
# -------------------------------
st.markdown("<h1 class='main-header'>Telco Churn Survival Analyzer</h1>",
            unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Predict customer lifetime using Cox\
 Proportional Hazards</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Model Overview",
                            "Customer Prediction",
                            "Global Insights"])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Model Summary")
        summary = cph.summary[["coef", "exp(coef)", "p"]].round(4)
        summary.columns = ["Coefficient", "Hazard Ratio", "p-value"]
        st.dataframe(summary.style.format({"Hazard Ratio": "{:.2f}",
                                           "p-value": "{:.4f}"}))

    with col2:
        st.markdown("### Top Risk Drivers")
        top = summary.copy()
        top["abs"] = top["Coefficient"].abs()
        top = top.sort_values("abs", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.barplot(data=top, y=top.index, x="Hazard Ratio",
                    palette="viridis", ax=ax)
        ax.axvline(1, color="red", linestyle="--")
        ax.set_title("Top Churn Risk Factors")
        for i, v in enumerate(top["Hazard Ratio"]):
            ax.text(v + 0.02, i, f"{v:.2f}", va="center")
        st.pyplot(fig)

with tab2:
    if "prediction" in st.session_state:
        p = st.session_state.prediction
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"<div class='metric-box'><h3>{p['risk_group']}\
</h3><p>Risk Group</p></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-box'><h3>{p['risk_score']:.3f}\
</h3><p>Risk Score</p></div>", unsafe_allow_html=True)
        with c3:
            lt = f"{p['median_time']:.0f}" \
                if np.isfinite(p['median_time']) else ">72"
            st.markdown(f"<div class='metric-box'><h3>{lt}</h3><p>Median\
 Lifetime (mo)</p></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div class='metric-box'><h3>{p['prob_24']:.1%}\
</h3><p>24-mo Survival</p></div>", unsafe_allow_html=True)

        st.markdown("### Predicted Survival Curve")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(p["surv_func"].index, p["surv_func"].iloc[:, 0],
                color="#1E40AF", linewidth=3)
        ax.fill_between(p["surv_func"].index, 0, p["surv_func"].iloc[:, 0],
                        alpha=0.1, color="#1E40AF")
        ax.axhline(0.5, color="red", linestyle="--", label="50% Survival")
        ax.set_xlabel("Time (Months)")
        ax.set_ylabel("Survival Probability")
        ax.set_title("Predicted Customer Lifetime")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("Enter customer details and click **Predict Churn Risk**")

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Global KM Curve")
        kmf = KaplanMeierFitter()
        kmf.fit(df_train["duration"], df_train["event"])
        fig, ax = plt.subplots()
        kmf.plot_survival_function(ax=ax, ci_show=True)
        ax.set_title("Overall Survival")
        st.pyplot(fig)

    with col2:
        st.markdown("### Survival by Contract")
        fig, ax = plt.subplots()
        for c in [col for col in df_train.columns \
                  if col.startswith("contract_")]:
            label = c.replace("contract_", "").replace("_", " ").title()
            mask = df_train.get(c, pd.Series(0, index=df_train.index)) == 1
            kmf.fit(df_train.loc[mask, "duration"],
                    df_train.loc[mask, "event"], label=label)
            kmf.plot_survival_function(ax=ax)
        ax.set_title("Survival by Contract Type")
        st.pyplot(fig)

st.markdown("<div class='footer'>Built using Lifelines + Streamlit | \
Model: Cox PH | Dataset: Telco Churn</div>", unsafe_allow_html=True)
st.markdown("<div class='footer'>Author: Aniket Walunj</div>",
            unsafe_allow_html=True)
