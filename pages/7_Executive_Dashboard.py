import streamlit as st
from modules.data_quality import analyze_dataset_health
from modules.business_translator import translate_metrics_to_business

st.set_page_config(page_title="Executive Briefing Dashboard", layout="wide")
st.title("Executive Briefing Dashboard")

# Guardrail check
if 'df' not in st.session_state or 'ml_results' not in st.session_state:
    st.warning("Please complete data upload, configuration, and baseline model training before viewing the Executive Dashboard.")
    st.stop()

df = st.session_state['df']
metrics = st.session_state['ml_results']

# Pull original health score parameters
quality_results = analyze_dataset_health(df)
health_score = quality_results["health_score"]

st.markdown("""
### Strategic Data Assets Summary
This dashboard translates your backend technical metrics into high-level KPI cards and actionable business summaries.
""")

# Row 1: Executive KPI Scorecard Blocks
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Data Integrity Health Score", value=f"{health_score} / 100")

with col2:
    # Custom framework card with theme variables
    st.markdown(
        f"""
        <div style="
            background-color: var(--background-color);
            padding: 11px 16px; 
            border-radius: 0.5rem; 
            border: 1px solid rgba(128, 128, 128, 0.2);
            box-shadow: rgba(0, 0, 0, 0.05) 0px 1px 3px;
        ">
            <p style="margin: 0; font-size: 14px; color: rgba(128, 128, 128, 1.0); font-weight: 400; opacity: 0.8;">Active Framework</p>
            <p style="margin: 4px 0 0 0; font-size: 20px; font-weight: 600; white-space: nowrap;">{metrics["task"]}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    # Identify best model based on metrics
    models = [k for k in metrics.keys() if k != "task"]
    if metrics["task"] == "Classification":
        best_model = max(models, key=lambda m: metrics[m]["Accuracy"])
    elif metrics["task"] in ["Regression", "Time-Series Forecasting"]:
        best_model = max(models, key=lambda m: metrics[m]["R2 Score"])
    else:
        best_model = "K-Means Engine"
        
    # NEW FIX: Replaced standard st.metric with auto-scaling HTML card to prevent text truncation
    st.markdown(
        f"""
        <div style="
            background-color: var(--background-color);
            padding: 11px 16px; 
            border-radius: 0.5rem; 
            border: 1px solid rgba(128, 128, 128, 0.2);
            box-shadow: rgba(0, 0, 0, 0.05) 0px 1px 3px;
        ">
            <p style="margin: 0; font-size: 14px; color: rgba(128, 128, 128, 1.0); font-weight: 400; opacity: 0.8;">Optimized Architecture</p>
            <p style="margin: 4px 0 0 0; font-size: 18px; font-weight: 600; word-wrap: break-word;">{best_model}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# Row 2: Plain-English Impact Explanations
st.subheader("Business Impact Translations")
business_insights = translate_metrics_to_business(metrics)

if metrics["task"] == "Clustering":
    st.info(business_insights["Core Finding"])
else:
    for model_name, insights in business_insights.items():
        with st.expander(f"Analysis Breakdown: {model_name}", expanded=True):
            for insight in insights:
                st.write(f"• {insight}")

st.markdown("---")

# Row 3: Prescriptive Next Steps
st.subheader("Suggested Strategic Next Steps")
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("""
    **Short-Term Data Hygiene Priorities:**
    1. Apply the automated preparation prescriptions generated in the *Smart Recommendation Engine*.
    2. Isolate and investigate high-cardinality columns to ensure privacy and unique indexing standards.
    3. Treat or separate statistical outliers to prevent distortion during scaling.
    """)

with col_right:
    st.markdown("""
    **Long-Term Engineering Options:**
    1. Enrich the modeling framework by introducing categorical feature encoding tools.
    2. Scale the baseline architectures into production spaces via automated hyperparameter tuning.
    3. Export the performance logs to track structural model drift over operational periods.
    """)

# --- FULL EMBEDDED DELIVERABLE REPORT GENERATION ---
st.markdown("---")
st.subheader("Export Deliverables")

# Constructing a clean, structured plaintext download payload
report_content = f"""==================================================
DATA QUALITY & PREDICTIVE ANALYTICS WORKBENCH REPORT
==================================================
Dataset Health Score: {health_score} / 100
Active Analytics Framework: {metrics['task']}
Optimized Architecture Selected: {best_model}

--------------------------------------------------
CORE BUSINESS INSIGHTS GENERATED:
--------------------------------------------------
"""

if metrics["task"] == "Clustering":
    report_content += f"\n- {business_insights['Core Finding']}\n"
else:
    for model_name, insights in business_insights.items():
        report_content += f"\n[Architecture: {model_name}]\n"
        for insight in insights:
            report_content += f" - {insight}\n"

report_content += """
--------------------------------------------------
STRATEGIC NEXT STEPS & PRESCRIPTIONS:
--------------------------------------------------
1. Impute or adjust missing metrics based on smart prescription paths.
2. Isolate high-cardinality items to enforce database normalization rules.
3. Establish safety stock buffers or operational flags using the baseline model error margins.
=================================================="""

# Render the functional download button widget
st.download_button(
    label="Download Executive Briefing Report (.txt)",
    data=report_content,
    file_name="executive_data_audit_summary.txt",
    mime="text/plain",
    use_container_width=True
)