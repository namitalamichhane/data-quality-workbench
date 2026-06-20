import streamlit as st
from modules.ml_recommender import detect_target_and_task

st.set_page_config(page_title="ML Setup & Recommendation", layout="wide")
st.title("Machine Learning Framework Setup")

if 'df' not in st.session_state:
    st.warning("Please upload a dataset on the Upload page before configuring machine learning.")
    st.stop()

df = st.session_state['df']

st.markdown("""
### Smart Feature & Target Mapping
Review the engine's architectural audit. You can accept the automated baseline suggestions or manually override the target, framework, and specific predictor feature selections below.
""")

# 1. Run the automated detection module
ml_recommendations = detect_target_and_task(df)

# 2. Display the Engine Recommendation Alert
with st.container():
    st.subheader("Automated Architecture Prescriptions")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            f"""
            <div style="
                background-color: var(--background-color);
                padding: 11px 16px; 
                border-radius: 0.5rem; 
                border: 1px solid rgba(128, 128, 128, 0.2);
                box-shadow: rgba(0, 0, 0, 0.05) 0px 1px 3px;
                margin-bottom: 1rem;
            ">
                <p style="margin: 0; font-size: 14px; color: rgba(128, 128, 128, 1.0); font-weight: 400; opacity: 0.8;">Suggested ML Task</p>
                <p style="margin: 4px 0 0 0; font-size: 18px; font-weight: 600; word-wrap: break-word;">{ml_recommendations["recommended_task"]}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write(f"**Suggested Target Feature:** `{ml_recommendations['suggested_target']}`")
    with col2:
        st.markdown(f"**Engine Justification:**\n*{ml_recommendations['reasoning']}*")

st.markdown("---")

st.subheader("Configure Modeling Framework")

# 3. User Target Override Selection
all_columns = list(df.columns)
default_target_idx = all_columns.index(ml_recommendations["suggested_target"]) if ml_recommendations["suggested_target"] in all_columns else 0

chosen_target = st.selectbox(
    "Select Target Variable (Y):", 
    options=["None - Run Clustering"] + all_columns, 
    index=default_target_idx + 1 if ml_recommendations["suggested_target"] in all_columns else 0
)

# 4. Framework Dropdown Menu
available_tasks = ["Regression", "Classification", "Time-Series Forecasting", "Clustering"]
if chosen_target == "None - Run Clustering":
    default_task_idx = available_tasks.index("Clustering")
else:
    rec_task = ml_recommendations["recommended_task"]
    default_task_idx = available_tasks.index(rec_task) if rec_task in available_tasks else 0

chosen_task = st.selectbox(
    "Select Active Modeling Framework:",
    options=available_tasks,
    index=default_task_idx,
    help="Override the engine recommendation if your data structure requires a specific pipeline layout."
)

# 5. NEW: Interactive Predictor (IV) Column Selector
chosen_predictors = []
if chosen_task != "Clustering":
    # Filter available inputs down to valid numeric features, dropping the target feature automatically
    numeric_cols = list(df.select_dtypes(include=['int64', 'float64']).columns)
    available_ivs = [col for col in numeric_cols if col != chosen_target]
    
    chosen_predictors = st.multiselect(
        "Select Independent Variables (IV / Predictors to Include):",
        options=available_ivs,
        default=available_ivs,
        help="By default, all numeric attributes are selected. Uncheck columns to manually test custom feature combinations."
    )
    
    if not chosen_predictors:
        st.error("⚠️ Boundary Constraint: You must keep at least one predictor feature selected to train a baseline model.")

st.info(f"Initialization Guardrail: **{chosen_task} Pipeline** will be built for downstream training.")

# 6. Lock selections firmly into active session states
st.session_state['ml_task'] = chosen_task
st.session_state['ml_target'] = None if chosen_task == "Clustering" else chosen_target
st.session_state['ml_predictors'] = chosen_predictors

st.success("Configuration verified and locked. Proceed to the Model Training page to execute the pipeline.")