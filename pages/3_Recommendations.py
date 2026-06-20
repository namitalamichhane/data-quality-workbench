import streamlit as st
from modules.data_quality import analyze_dataset_health
from modules.recommendations import generate_recommendations

st.set_page_config(page_title="Actionable Recommendations", layout="wide")
st.title(" Automated Data Preparation Prescriptions")

if 'df' not in st.session_state:
    st.warning("Please upload a dataset on the Upload page before viewing recommendations.")
    st.stop()

df = st.session_state['df']

# --- NEW ADDITION: FEATURE SELECTION TRANSPARENCY LOGS ---
st.markdown("###  Automated Pre-Modeling Feature Selection")

dropped_cols = []
for col in df.columns:
    # 1. >50% Missing Values check
    if df[col].isnull().mean() > 0.50:
        dropped_cols.append({"Column": col, "Reason": "Missing value rate exceeds the 50% threshold limit."})
    # 2. Zero Variance check
    elif df[col].nunique() <= 1:
        dropped_cols.append({"Column": col, "Reason": "Zero variance detected (column contains identical or empty values)."})

if dropped_cols:
    st.warning(" The engine has flagged the following features for exclusion from downstream training phases:")
    for item in dropped_cols:
        st.markdown(f"• **`{item['Column']}`** — *{item['Reason']}*")
else:
    st.success("Structural Feature Integrity Verified: All uploaded columns contain sufficient variance and density for modeling channels.")

st.markdown("---")

# --- CONTINUATION OF ORIGINAL RECOMMENDATIONS ---
st.markdown("""
### Target-Specific Data Action Items
Review the prescription details below to optimize your remaining features for predictive modeling structures.
""")

# Run the analyzer data backend
quality_results = analyze_dataset_health(df)
recs = generate_recommendations(quality_results, df)

if not recs:
    st.success("Excellent! No major structural anomalies were flagged for the remaining features.")
else:
    # Display recommendations as clean structured blocks
    for i, r in enumerate(recs, 1):
        with st.container():
            st.markdown(f"#### Recommendation {i}: Feature `{r['column']}`")
            
            # Setup columns for the structured prescription look
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**Detected Issue:** {r['issue']}")
                st.markdown(f"**Prescribed Action:** `{r['action']}`")
            with col2:
                st.markdown(f"**Analytics Justification:**\n*{r['reason']}*")
                
            st.markdown("---")