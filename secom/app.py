import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Semiconductor Yield Intelligence Dashboard",
    page_icon="🔬",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# --- 1. Load Artifacts Cached ---
@st.cache_resource
def load_artifacts():
    artifacts = {}
    artifacts['pipeline'] = joblib.load(os.path.join(ARTIFACTS_DIR, "model_pipeline.joblib"))
    artifacts['feature_names'] = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_names.joblib"))
    artifacts['pca'] = joblib.load(os.path.join(ARTIFACTS_DIR, "pca.joblib"))
    artifacts['kmeans'] = joblib.load(os.path.join(ARTIFACTS_DIR, "kmeans.joblib"))
    artifacts['cluster_summary'] = pd.read_csv(os.path.join(ARTIFACTS_DIR, "cluster_summary.csv"))
    artifacts['feature_importance'] = pd.read_csv(os.path.join(ARTIFACTS_DIR, "feature_importance.csv"))
    artifacts['metrics'] = pd.read_csv(os.path.join(ARTIFACTS_DIR, "metrics.csv"))
    artifacts['test_samples'] = pd.read_csv(os.path.join(ARTIFACTS_DIR, "test_samples.csv"))
    
    # Version-compatibility patch for SimpleImputer across different sklearn versions
    imputer_step = artifacts['pipeline'].named_steps.get("imputer", None)
    if imputer_step is not None and not hasattr(imputer_step, "_fill_dtype"):
        # Synthesize missing internal attribute expected by modern sklearn
        imputer_step._fill_dtype = np.float64
        
    return artifacts

try:
    art = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    load_err = str(e)

# Robust inference helper that bypasses internal sklearn version gaps
def robust_predict(pipeline, df_input):
    try:
        pred = pipeline.predict(df_input)[0]
        prob = pipeline.predict_proba(df_input)[0][1]
        return pred, prob
    except AttributeError:
        # Fallback if unpickled transformer internals conflict
        imputer = pipeline.named_steps["imputer"]
        scaler = pipeline.named_steps["scaler"]
        clf = pipeline.named_steps["classifier"]
        
        # Manual step-through using stored numpy statistics
        vals = np.array(df_input, dtype=np.float64)
        # Impute using saved medians
        nan_mask = np.isnan(vals)
        if hasattr(imputer, "statistics_"):
            vals[nan_mask] = np.take(imputer.statistics_, np.where(nan_mask)[1])
        # Scale
        vals_scaled = scaler.transform(vals)
        # Classify
        pred = clf.predict(vals_scaled)[0]
        prob = clf.predict_proba(vals_scaled)[0][1]
        return pred, prob

# --- Sidebar ---
with st.sidebar:
    st.header("🔬 System Metadata")
    st.markdown("**Project:** Predictive Quality Control")
    st.markdown("**Dataset:** UCI SECOM Benchmark")
    st.markdown("**Architecture:** End-to-End Scikit-Learn Pipeline")
    st.markdown("---")
    if artifacts_loaded:
        st.success("Artifacts Loaded: ✓")
        st.write(f"Active Features: **{len(art['feature_names'])}**")
        st.write(f"Holdout Samples: **{len(art['test_samples'])}**")
    else:
        st.error("Artifacts Missing!")
    st.markdown("---")
    st.caption("Decision-Support & Process Intelligence Prototype.")

# --- Header ---
st.title("🔬 AI-Based Semiconductor Manufacturing Defect Prediction")
st.subheader("Predictive Quality Control & Process Intelligence Prototype")

if not artifacts_loaded:
    st.error(f"Failed to load artifacts from '{ARTIFACTS_DIR}'. Details: {load_err}")
    st.stop()

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Process Overview & EDA",
    "⚡ In-Line Defect Prediction",
    "🧠 Process Intelligence",
    "📈 Model Evaluation"
])

# ==========================================
# TAB 1: OVERVIEW & EDA
# ==========================================
with tab1:
    st.markdown("### Manufacturing Process Line Status")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Production Runs", "1,567")
    col2.metric("Raw Sensor Signals", "590")
    col3.metric("Cleaned Active Features", str(len(art['feature_names'])))
    col4.metric("Baseline Failure Rate", "6.64% (Imbalanced)")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("#### Production Yield Distribution")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.barplot(x=['Pass (0)', 'Fail (1)'], y=[1463, 104], palette=['#2b8a3e', '#c92a2a'], ax=ax)
        ax.set_ylabel("Wafer Production Count")
        ax.set_title("Target Distribution: ~14 Pass per 1 Fail")
        st.pyplot(fig)
        
    with col_b:
        st.markdown("#### Data Hygiene & Preprocessing Architecture")
        st.info("""
        - **Raw Space (590 Variables):** Includes uninformative, constant, and sparse readings.
        - **Automated Cleaning:** Constant channels (zero variance) and sensors with >50% missingness are removed.
        - **Pipeline Encapsulation:** Median Imputation and Standardization are executed directly inside the serialized pipeline, preventing test leakage.
        """)

# ==========================================
# TAB 2: IN-LINE DEFECT PREDICTION
# ==========================================
with tab2:
    st.markdown("### Real-Time Wafer Quality Prediction")
    st.write("Perform quality control inference on an individual manufactured wafer:")

    source_option = st.radio("Choose Input Mode:", ["Select from Holdout Test Set", "Upload Custom Wafer Sensor CSV"], horizontal=True)

    if source_option == "Select from Holdout Test Set":
        test_df = art['test_samples']
        sample_idx = st.selectbox(
            "Select Wafer Sample:",
            options=range(len(test_df)),
            format_func=lambda i: f"Wafer Run #{i} — Ground Truth: {'FAIL' if test_df.iloc[i]['Actual_Fail'] == 1 else 'PASS'}"
        )
        sample_row = test_df.drop(columns=['Actual_Fail']).iloc[[sample_idx]]
        actual_val = int(test_df.iloc[sample_idx]['Actual_Fail'])
        
        if st.button("Evaluate Wafer Quality", type="primary"):
            pred, prob = robust_predict(art['pipeline'], sample_row)

            st.markdown("---")
            res1, res2, res3 = st.columns(3)
            with res1:
                st.markdown("#### In-House Verification")
                if actual_val == 1:
                    st.error("❌ Actual: FAIL (Excursion)")
                else:
                    st.success("✅ Actual: PASS (Nominal)")
            with res2:
                st.markdown("#### System Recommendation")
                if pred == 1:
                    st.error("🚨 HOLD WAFER (Predicted Fail)")
                else:
                    st.success("✅ PROCEED (Predicted Pass)")
            with res3:
                st.markdown("#### Estimated Failure Probability")
                st.metric("Defect Probability", f"{prob * 100:.2f}%")
                st.progress(float(prob))

    else:
        uploaded_file = st.file_uploader("Upload Sensor CSV (Expected format: 562 active features)", type=["csv"])
        if uploaded_file is not None:
            user_df = pd.read_csv(uploaded_file)
            missing_cols = [c for c in art['feature_names'] if c not in user_df.columns]
            if missing_cols:
                st.error(f"Validation Error: Uploaded CSV is missing {len(missing_cols)} required columns (e.g. {missing_cols[:3]}).")
            else:
                st.success("CSV Validated Successfully!")
                eval_row = user_df[art['feature_names']].iloc[[0]]
                if st.button("Predict Uploaded Wafer", type="primary"):
                    pred, prob = robust_predict(art['pipeline'], eval_row)
                    st.markdown("---")
                    c1, c2 = st.columns(2)
                    with c1:
                        if pred == 1:
                            st.error("🚨 Prediction: FAIL")
                        else:
                            st.success("✅ Prediction: PASS")
                    with c2:
                        st.metric("Estimated Defect Risk", f"{prob * 100:.2f}%")
                        st.progress(float(prob))

# ==========================================
# TAB 3: PROCESS INTELLIGENCE
# ==========================================
with tab3:
    st.markdown("### Process Intelligence & Yield Optimization")
    
    col_feat, col_clust = st.columns(2)
    
    with col_feat:
        st.markdown("#### Top 15 Informative Sensors (Gini Importance)")
        imp_df = art['feature_importance'].head(15)
        fig_imp, ax_imp = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=imp_df, x="Importance", y="Feature", palette="viridis", ax=ax_imp)
        ax_imp.set_xlabel("Feature Importance Score")
        ax_imp.set_ylabel("Sensor")
        st.pyplot(fig_imp)
        st.caption("Features indicate statistical association with yield excursions, not physical causation.")

    with col_clust:
        st.markdown("#### Unsupervised Operating Regime Audit (K-Means)")
        st.dataframe(art['cluster_summary'], use_container_width=True)
        st.info("""
        **Regime Interpretation:**  
        K-Means partitions runs based on PCA multivariate sensor projections. 
        Regimes exhibiting higher failure rates pinpoint process operating spaces that warrant closer engineering inspection.
        """)

# ==========================================
# TAB 4: MODEL EVALUATION
# ==========================================
with tab4:
    st.markdown("### Classifier Performance Benchmark")
    st.dataframe(art['metrics'], use_container_width=True)
    
    st.markdown("#### Critical Evaluation Factor: Minimizing False Negatives")
    st.warning("""
    In semiconductor manufacturing lines, **False Negatives** (failing to identify defective wafers) incur substantial waste by allowing flawed chips to undergo expensive downstream packaging. 
    Models configured with balanced cost weighting prioritize **Recall (Fail)** over raw classification accuracy.
    """)