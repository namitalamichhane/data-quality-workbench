import streamlit as st

st.set_page_config(
    page_title="Data Quality and Predictive Analytics Workbench", 
    layout="wide"
)

st.title("Data Quality and Predictive Analytics Workbench")

st.markdown("""
### Welcome to your Analytics Production Environment

This platform functions as an automated diagnostic engine designed to run end-to-end dataset profiling, outlier detection, machine learning framework optimization, and business translation.

**To get started, follow the pipeline in the left sidebar navigation menu:**
1. **Upload:** Drop your target CSV into the framework memory cache.
2. **Data Quality and Recommendations:** Inspect anomalies and automated cleaning plans.
3. **EDA and Outliers:** Interact with your variables and isolate statistical variances.
4. **ML Engine:** Setup variables and train chronological prediction architectures.
5. **Executive Dashboard:** Review plain-English metrics and export summary reports.
""")