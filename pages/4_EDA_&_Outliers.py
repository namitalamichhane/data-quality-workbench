import streamlit as st
import plotly.express as px
from modules.eda import detect_outliers_iqr

st.set_page_config(page_title="EDA & Outlier Analysis", layout="wide")
st.title(" Exploratory Data Analysis & Outliers")

if 'df' not in st.session_state:
    st.warning("Please upload a dataset on the Upload page before running analysis.")
    st.stop()

df = st.session_state['df']

# Tab layout to separate the data overview from outlier deep dives
tab1, tab2 = st.tabs(["Overview & Distributions", "Outlier Diagnostic"])

with tab1:
    st.subheader("Statistical Summary Table")
    st.dataframe(df.describe(include='all'), use_container_width=True)
    
    # Dynamic chart rendering based on column type
    st.subheader("Feature Distribution Viewer")
    selected_col = st.selectbox("Select a column to visualize:", df.columns)
    
    if df[selected_col].dtype in ['int64', 'float64']:
        # FIX: Added marker_line_color and marker_line_width to separate the bars
        fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}", template="plotly_white")
        fig.update_traces(marker_line_color='white', marker_line_width=1.5)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Categorical distribution
        val_counts = df[selected_col].value_counts().reset_index()
        val_counts.columns = [selected_col, 'Count']
        # FIX: Added marker_line_color and marker_line_width to separate categorical bars
        fig = px.bar(val_counts, x=selected_col, y='Count', title=f"Frequency of {selected_col}", template="plotly_white")
        fig.update_traces(marker_line_color='white', marker_line_width=1.5)
        st.plotly_chart(fig, use_container_width=True)

    # Correlation Matrix Section
    st.subheader("Correlation Analysis")
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if numeric_df.shape[1] > 1:
        corr_matrix = numeric_df.corr()
        # FIX: Explicitly forced height, width, and aspect ratio to ensure it expands beautifully
        fig_corr = px.imshow(
            corr_matrix, 
            text_auto='.2f', 
            color_continuous_scale='RdBu_r', 
            title="Numeric Correlation Matrix",
            height=600,
            aspect="auto"
        )
        # Clean up layout padding so the text doesn't clip
        fig_corr.update_layout(margin=dict(l=50, r=50, t=50, b=50))
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough numeric columns available to compute a correlation matrix.")

with tab2:
    st.subheader("Automated IQR Outlier Scan")
    outlier_results = detect_outliers_iqr(df)
    
    if not outlier_results:
        st.success("No statistical outliers detected in any numeric columns.")
    else:
        # Display summary information
        for col, stats in outlier_results.items():
            st.markdown(f"#### Column: `{col}`")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric(label="Outlier Count", value=f"{stats['count']:,}")
                st.metric(label="Percentage of Data", value=f"{stats['percentage']:.2f}%")
            with col2:
                # Render boxplot to visually isolate outliers
                fig_box = px.box(df, y=col, title=f"Boxplot Analysis for {col}", template="plotly_white")
                st.plotly_chart(fig_box, use_container_width=True)
            st.markdown("---")