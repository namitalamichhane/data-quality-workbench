import pandas as pd
import numpy as np
import plotly.express as px

def detect_outliers_iqr(df):
    """
    Scans numeric columns using the Interquartile Range (IQR) method 
    to count outliers and calculate their percentage impact.
    """
    outlier_summary = {}
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numeric_cols:
        # Drop null values for accurate quartile calculation
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
            
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Count values outside the bounds
        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
        outlier_count = len(outliers)
        
        if outlier_count > 0:
            outlier_summary[col] = {
                "count": outlier_count,
                "percentage": (outlier_count / len(df)) * 100,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            }
            
    return outlier_summary