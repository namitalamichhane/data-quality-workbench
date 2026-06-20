import pandas as pd

def detect_target_and_task(df):
    """
    Scans column names for business keywords to automatically suggest 
    the most likely target variable and the correct machine learning task,
    using smart data-type fallbacks to avoid false clustering recommendations.
    """
    # 1. Strict time column detection
    time_keywords = ['date', 'time', 'timestamp', 'dt']
    has_time_col = False
    
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            has_time_col = True
            break
        col_lower = col.lower()
        if any(tk in col_lower for tk in time_keywords) or col_lower in ['year', 'month', 'day', 'week']:
            has_time_col = True
            break
    
    # Define keywords for target variable exploration
    classification_keywords = ['churn', 'target', 'label', 'fraud', 'default', 'status', 'subscribed', 'class']
    regression_keywords = ['sales', 'revenue', 'income', 'price', 'profit', 'amount', 'cost', 'orders', 'demand', 'val', 'value']
    
    # 2. Route to Time-Series Forecasting ONLY if true time marker AND numeric variable exist together
    if has_time_col:
        for col in df.columns:
            if any(kw in col.lower() for kw in regression_keywords):
                return {
                    "suggested_target": col,
                    "recommended_task": "Time-Series Forecasting",
                    "reasoning": f"Detected chronological time markers alongside a continuous metric '{col}'. The engine has prioritized a Time-Series Forecasting framework."
                }

    # 3. Standard Keyword Classification Check
    for col in df.columns:
        if any(kw in col.lower() for kw in classification_keywords):
            return {
                "suggested_target": col,
                "recommended_task": "Classification",
                "reasoning": f"The column '{col}' contains keywords associated with categorical outcomes."
            }
            
    # 4. Standard Keyword Regression Check 
    for col in df.columns:
        if any(kw in col.lower() for kw in regression_keywords):
            return {
                "suggested_target": col,
                "recommended_task": "Regression",
                "reasoning": f"The column '{col}' contains continuous numeric outcomes. A standard cross-validated regression model will forecast these quantitative values."
            }

    # 5. SMART FALLBACK: If no keywords match, look for the last numeric column as a fallback target
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        fallback_target = numeric_cols[-1] # Grabs a numeric column to evaluate
        if df[fallback_target].nunique() < 15:
            fallback_task = "Classification"
            reasoning = f"No keyword match found. Defaulting to the attribute '{fallback_target}' which acts as a discrete categorical class variance."
        else:
            fallback_task = "Regression"
            reasoning = f"No keyword match found. Defaulting to the continuous attribute '{fallback_target}' as a regression modeling target."
            
        return {
            "suggested_target": fallback_target,
            "recommended_task": fallback_task,
            "reasoning": reasoning
        }

    # 6. Absolute last resort if there is zero usable structure
    return {
        "suggested_target": "None",
        "recommended_task": "Clustering",
        "reasoning": "No obvious target columns or numeric metrics were detected. Initializing unsupervised clustering routines."
    }