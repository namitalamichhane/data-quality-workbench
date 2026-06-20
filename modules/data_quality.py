import pandas as pd
import numpy as np

def analyze_dataset_health(df):
    """
    Scans a pandas DataFrame for missing data, duplicates, constant columns, 
    and high cardinality issues, returning a dictionary of metrics and an overall health score.
    """
    total_rows = len(df)
    total_cells = df.size
    
    # 1. Check Missing Values
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    missing_percentage = (total_missing / total_cells) * 100 if total_cells > 0 else 0
    
    missing_by_column = {}
    for col in df.columns:
        null_count = int(missing_counts[col])
        if null_count > 0:
            missing_by_column[col] = {
                "count": null_count,
                "pct": (null_count / total_rows) * 100
            }

    # 2. Check Duplicate Rows
    duplicate_count = int(df.duplicated().sum())
    duplicate_percentage = (duplicate_count / total_rows) * 100 if total_rows > 0 else 0

    # 3. Structural Column Auditing
    constant_columns = []
    high_cardinality_columns = []
    potential_identifiers = []
    
    for col in df.columns:
        unique_count = df[col].nunique()
        
        # Constant columns (Zero variance)
        if unique_count == 1:
            constant_columns.append(col)
            
        # High Cardinality (Text/Categorical columns with too many unique values)
        elif df[col].dtype == 'object' or isinstance(df[col].dtype, pd.CategoricalDtype):
            if unique_count > 50 and unique_count < total_rows:
                high_cardinality_columns.append(col)
                
        # Potential Primary Key / Identifier columns 
        # FIX: Only flag if the column is a text/string type, not numeric
        if unique_count == total_rows and df[col].dtype == 'object':
            potential_identifiers.append(col)

    # 4. Calculate Transparent Health Score
    # Start at 100 and apply structured penalties
    score = 100
    
    # Deduct up to 25 points based on overall missingness severity
    score -= min(25, missing_percentage * 2)
    
    # Deduct up to 25 points based on row duplication severity
    score -= min(25, duplicate_percentage * 2)
    
    # Deduct 10 points for every completely useless constant column
    score -= len(constant_columns) * 10
    
    # Ensure score stays bounded between 0 and 100
    final_score = int(max(0, min(100, score)))

    # Compile findings
    results = {
        "health_score": final_score,
        "total_missing": total_missing,
        "missing_percentage": missing_percentage,
        "missing_by_column": missing_by_column,
        "duplicate_count": duplicate_count,
        "constant_columns": constant_columns,
        "high_cardinality_columns": high_cardinality_columns,
        "potential_identifiers": potential_identifiers
    }
    
    return results