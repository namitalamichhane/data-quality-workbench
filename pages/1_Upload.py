import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

st.set_page_config(page_title="Upload Dataset", layout="wide")
st.title("Data Ingestion Center")

st.markdown("""
### Upload Your Target Asset
Provide an operational tracking dataset (.csv) or load a pre-configured baseline sample environment below.
""")

# --- MULTI-OPTION DEMO DATASET ACCELERATOR ---
st.markdown("##### Don't have a dataset ready? Load a sample pipeline:")

demo_choice = st.selectbox(
    "Choose a sample environment to inject into the workbench:",
    options=[
        "Select a sample dataset...",
        "Telecom Customer Churn Analytics",
        "Airlines Passenger Demand Logs",
        "Healthcare Patient Readmission Risk"
    ]
)

if demo_choice != "Select a sample dataset...":
    # Ensure a fresh state for every toggle choice to avoid caching collisions
    if 'ml_results' in st.session_state:
        del st.session_state['ml_results']
    if 'ml_predictors' in st.session_state:
        del st.session_state['ml_predictors']

    # Scenario 1: Telecom Customer Churn (Classification)
    if "Telecom Customer Churn" in demo_choice:
        # FIX 2: Added n_redundant=0 so scikit-learn doesn't force invisible math columns that break the total feature count
        X, y = make_classification(
            n_samples=500, 
            n_features=4, 
            n_informative=3, 
            n_redundant=0, 
            n_clusters_per_class=1, 
            weights=[0.5, 0.5], 
            random_state=42
        )
        demo_df = pd.DataFrame(X, columns=["Data_Usage_GB", "Customer_Service_Calls", "Contract_Tenure_Months", "Monthly_Billing_Amount"])
        demo_df["Churn_Status"] = y
        
    # Scenario 2: Airlines Passenger Demand (Time-Series Forecasting)
    elif "Airlines Passenger Demand" in demo_choice:
        n_periods = 400
        # FIX 3: Shifted regression features to act as high-variance time-indexed coordinates
        X, y = make_regression(n_samples=n_periods, n_features=3, noise=4.0, random_state=42)
        demo_df = pd.DataFrame(X, columns=["Fuel_Price_Index", "Average_Ticket_Cost", "Economic_Velocity_Score"])
        
        # We model a solid, stable linear growth pattern combined with an intentional cyclical flight wave
        time_trend = np.linspace(50, 200, n_periods)
        airline_seasonality = 35 * np.sin(np.linspace(0, 8 * np.pi, n_periods))
        demo_df["Passenger_Order_Demand"] = y + time_trend + airline_seasonality
        
        # Append explicit sequential date markers to trip has_time_col index rules
        demo_df["Flight_Date"] = pd.date_range(start="2025-01-01", periods=n_periods, freq="D").astype(str)
        
        cols = ["Flight_Date"] + [c for c in demo_df.columns if c != "Flight_Date"]
        demo_df = demo_df[cols]
        
    # Scenario 3: Healthcare Patient Readmission (Continuous Risk Regression Score)
    elif "Healthcare Patient" in demo_choice:
        # FIX 4: Bumped random noise up significantly to break perfect deterministic linearity and reflect authentic clinical chaos
        X, y = make_regression(n_samples=500, n_features=4, noise=32.5, random_state=42)
        demo_df = pd.DataFrame(X, columns=["Age_Index", "Comorbidity_Count", "Days_In_ICU", "Lab_Test_Frequency"])
        
        # Regularize onto a clean healthcare tracking risk score (0 - 100)
        y_scaled = (y - y.min()) / (y.max() - y.min()) * 100
        demo_df["Readmission_Risk_Cost_Val"] = y_scaled
    
    # Intentionally inject mild anomalies so the Data Quality Center has entries to report
    demo_df.iloc[4, 1] = None  
    demo_df = pd.concat([demo_df, demo_df.iloc[[10]]], ignore_index=True)
    
    st.session_state['df'] = demo_df
    st.success(f"🎉 Asset tracking parameters for '{demo_choice}' successfully synchronized!")

st.markdown("---")

# Main file uploader fallback frame
uploaded_file = st.file_uploader("Or, upload your own custom CSV file", type=["csv"])

if uploaded_file is not None:
    MAX_FILE_SIZE_MB = 10
    file_size_mb = uploaded_file.size / (1024 * 1024)
    
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File size variance exception: Upload exceeds the structural limit of {MAX_FILE_SIZE_MB}MB.")
        st.stop()
        
    df = pd.read_csv(uploaded_file)
    
    MAX_ROWS = 20000
    if len(df) > MAX_ROWS:
        st.error(f"Row ceiling boundary exception: Dataset length exceeds processing safety thresholds.")
        st.stop()
        
    st.session_state['df'] = df
    if 'ml_results' in st.session_state:
        del st.session_state['ml_results']
    st.success("Custom asset loaded successfully into cache.")

# --- DISPLAY METRICS ---
if 'df' in st.session_state:
    current_df = st.session_state['df']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Row Record Count", value=f"{len(current_df):,}")
    with col2:
        st.metric(label="Feature Attributes (Columns)", value=str(current_df.shape[1]))
    with col3:
        memory_usage_mb = current_df.memory_usage(deep=True).sum() / (1024 * 1024)
        st.metric(label="Memory Allocation Profile", value=f"{memory_usage_mb:.3f} MB")
        
    st.markdown("### Structural Data Preview")
    st.dataframe(current_df.head(10), use_container_width=True)