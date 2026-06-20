import streamlit as st
import pandas as pd
from modules.model_trainer import train_baseline_models

st.set_page_config(page_title="Baseline Model Training", layout="wide")
st.title("Baseline Model Training & Evaluation Lab")

# Guardrail: Check if dataset and configuration parameters are locked in session state
if 'df' not in st.session_state or 'ml_task' not in st.session_state:
    st.warning("Please upload a dataset and complete the ML Setup configurations before executing model training.")
    st.stop()

df = st.session_state['df']
task = st.session_state['ml_task']
target = st.session_state['ml_target']
# Pull the custom selected predictors from session state memory
predictors = st.session_state.get('ml_predictors', [])

st.write(f"Active Training Pipeline: **{task} Model Pipeline**")
if target:
    st.write(f"Target Feature (Y): `{target}`")

# NEW: Display selected independent variables so the user knows exactly what is entering the architecture
if task != "Clustering":
    st.write(f"Selected Predictor Features (IVs): `{', '.join(predictors) if predictors else 'All Numeric Columns'}`")

# Step 1: Run the Training Framework
if st.button("Execute Training Pipeline"):
    if task != "Clustering" and not predictors:
        st.error("⚠️ Pipeline Blocked: Go back to the ML Setup page and select at least one predictor feature.")
    else:
        with st.spinner("Partitioning variables and optimizing model paths..."):
            metrics = train_baseline_models(df, task, target)
            
        if "error" in metrics:
            st.error(metrics["error"])
        else:
            st.success("Training iteration completed successfully! Evaluation models are loaded below.")
            # Store results in memory so changing dropdowns doesn't wipe out the trained models
            st.session_state['ml_results'] = metrics

# Step 2: Render the Interactive Model Sandbox if models are trained
if 'ml_results' in st.session_state:
    metrics = st.session_state['ml_results']
    
    st.markdown("---")
    
    if metrics["task"] == "Clustering":
        st.subheader("K-Means Clustering Distribution")
        st.write(f"Identified Groupings: **{metrics['num_clusters']} Clusters Optimized**")
        st.metric(label="Silhouette Score", value=f"{metrics['Silhouette Score']:.4f}")
        
        for cluster, size in metrics["cluster_sizes"].items():
            st.metric(label=f"Cluster {cluster} Population", value=f"{size:,} rows")
            
    else:
        # Get the list of all trained supervised models
        available_models = [k for k in metrics.keys() if k != "task"]
        
        # Determine the smart recommendation default
        if metrics["task"] == "Classification":
            recommended_model = max(available_models, key=lambda m: metrics[m]["Accuracy"])
        else:
            recommended_model = max(available_models, key=lambda m: metrics[m]["R2 Score"])
            
        st.subheader("Interactive Performance Sandbox")
        st.info(f"**Engine Recommendation:** Our automated audit recommends the **{recommended_model}** architecture for this specific tracking structure based on initial variance testing.")
        
        # Create a dropdown selector for the user to explore models manually
        ordered_options = [recommended_model] + [m for m in available_models if m != recommended_model]
        
        selected_model = st.selectbox(
            "Select an evaluation architecture to inspect raw performance metrics:",
            options=ordered_options
        )
        
        # Display the metrics of ONLY the user's selected dropdown model
        st.markdown(f"### Performance Metrics: `{selected_model}`")
        
        model_metrics = metrics[selected_model]
        
        # Filter out the matrix array so it doesn't try to draw a standard numeric KPI card for it
        kpi_metrics = {k: v for k, v in model_metrics.items() if k != "Confusion Matrix"}
        metric_cols = st.columns(len(kpi_metrics))
        
        for col, (metric_name, val) in zip(metric_cols, kpi_metrics.items()):
            with col:
                st.metric(label=metric_name, value=f"{val:.4f}")
                
        # Render the Confusion Matrix visually if it exists for the model
        if "Confusion Matrix" in model_metrics:
            st.markdown("#### Confusion Matrix Breakdown")
            cm = model_metrics["Confusion Matrix"]
            
            # Map structural array values to readable row and column indices
            cm_df = pd.DataFrame(
                cm, 
                index=["Actual Negative", "Actual Positive"], 
                columns=["Predicted Negative", "Predicted Positive"]
            )
            st.dataframe(cm_df, use_container_width=True)
            
            st.caption(
                f"Interpretation: True Negatives: {cm[0][0]} | False Positives: {cm[0][1]} | "
                f"False Negatives: {cm[1][0]} | True Positives: {cm[1][1]}"
            )