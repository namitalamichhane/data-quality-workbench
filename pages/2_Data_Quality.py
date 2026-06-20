import streamlit as st
from modules.data_quality import analyze_dataset_health

st.set_page_config(page_title="Data Quality Center", layout="wide")
st.title("🛡️ Data Quality Center")

# Check if a dataset exists in session state from the Upload page
if 'df' not in st.session_state:
    st.warning("Please upload a dataset on the Upload page before running the quality assessment.")
    st.stop()

df = st.session_state['df']

# Run backend evaluation function
with st.spinner("Analyzing dataset metrics..."):
    results = analyze_dataset_health(df)

# Display Overall Health Score Dashboard
st.subheader("Overall Dataset Health Score")
score = results["health_score"]

if score >= 80:
    st.success(f"Score: {score} / 100 - Good Health")
elif score >= 50:
    st.warning(f"Score: {score} / 100 - Moderate Issues Detected")
else:
    st.error(f"Score: {score} / 100 - Critical Action Required")

# Summary columns
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Missing Value Cells", value=f"{results['total_missing']:,} ({results['missing_percentage']:.2f}%)")
with col2:
    st.metric(label="Duplicate Rows", value=f"{results['duplicate_count']:,}")
with col3:
    st.metric(label="Identified Key Columns", value=str(len(results['potential_identifiers'])))

st.markdown("---")

# Breakdowns by issue type
st.subheader("Detailed Scan Findings")

# Section: Missing Values
if results['missing_by_column']:
    st.markdown("### Missing Value Breakdown")
    for col, data in results['missing_by_column'].items():
        st.write(f"- **{col}**: Missing {data['count']:,} values ({data['pct']:.2f}% of column)")
else:
    st.write("No missing values found in any column.")

# Section: Structuring issues
st.markdown("### Structural Issues Found")

if results['constant_columns']:
    st.write(f"- **Constant Columns (Zero Variance)**: {', '.join(results['constant_columns'])}")
else:
    st.write("- No constant columns found.")

if results['high_cardinality_columns']:
    st.write(f"- **High Cardinality Columns**: {', '.join(results['high_cardinality_columns'])}")
else:
    st.write("- No high-cardinality text columns found.")

if results['potential_identifiers']:
    st.write(f"- **Potential Unique Identifier/ID Columns**: {', '.join(results['potential_identifiers'])}")
else:
    st.write("- No pure unique identifier columns flagged.")