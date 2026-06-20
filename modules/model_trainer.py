import pandas as pd
import numpy as np
import streamlit as st # Injected for reading session states dynamically
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

def train_baseline_models(df, task, target=None):
    # --- DYNAMIC INDEPENDENT VARIABLE (IV) FILTERING ---
    # Check if the user defined custom predictors in the session state menu
    if task != "Clustering" and 'ml_predictors' in st.session_state and st.session_state['ml_predictors']:
        chosen_features = st.session_state['ml_predictors']
        # Extract strictly what the user locked down in their selection box
        X_pure = df[chosen_features].copy()
    else:
        # Fallback to standard tracking setup if state is uninitialized or running Clustering
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        X_pure = df[numeric_cols].copy()
        if target in X_pure.columns:
            X_pure = X_pure.drop(columns=[target])
            
    # Handle missing value fill configurations safely for selected features
    X_pure = X_pure.fillna(X_pure.median())
    
    # --- FRAMEWORK 1: CLUSTERING ---
    if task == "Clustering":
        X_pure = X_pure.loc[:, X_pure.nunique() > 1]
        if X_pure.shape[1] == 0:
            return {"error": "Not enough numeric features with variation to run clustering."}
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_pure)
        
        from sklearn.metrics import silhouette_score
        if len(np.unique(clusters)) > 1:
            sil_score = float(silhouette_score(X_pure, clusters))
        else:
            sil_score = 0.0
            
        return {
            "task": "Clustering", 
            "cluster_sizes": pd.Series(clusters).value_counts().to_dict(), 
            "num_clusters": 3,
            "Silhouette Score": sil_score
        }

    # Guardrail check for supervised data paths
    if not target or target not in df.columns:
        return {"error": "Invalid target column specified for supervised learning."}
        
    y = df[target]

    # Double check to prevent target leakage if the user somehow added it to the features
    if target in X_pure.columns:
        X_pure = X_pure.drop(columns=[target])

    # Ensure we actually have variables left to build predictions on
    if X_pure.shape[1] == 0:
        return {"error": "No predictor columns remain. Select at least one independent variable column."}

    # --- FRAMEWORK 2: TIME-SERIES FORECASTING (CHRONOLOGICAL SPLIT) ---
    if task == "Time-Series Forecasting":
        y = y.astype(float)
        split_idx = int(len(df) * 0.8)
        
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        X_train, X_test = X_pure.iloc[:split_idx], X_pure.iloc[split_idx:]

        # Model A: Time-Indexed Linear Trend
        lin = LinearRegression()
        lin.fit(X_train, y_train)
        preds_lin = lin.predict(X_test)

        # Model B: Sequential Random Forest
        rf_ts = RandomForestRegressor(random_state=42)
        rf_ts.fit(X_train, y_train)
        preds_rf = rf_ts.predict(X_test)

        return {
            "task": "Time-Series Forecasting",
            "Time-Indexed Linear Trend": {
                "MAE": mean_absolute_error(y_test, preds_lin),
                "RMSE": np.sqrt(mean_squared_error(y_test, preds_lin)),
                "R2 Score": r2_score(y_test, preds_lin)
            },
            "Sequential Random Forest": {
                "MAE": mean_absolute_error(y_test, preds_rf),
                "RMSE": np.sqrt(mean_squared_error(y_test, preds_rf)),
                "R2 Score": r2_score(y_test, preds_rf)
            }
        }

    # --- FRAMEWORK 3: STANDARD SUPERVISED TRAIN/TEST (RANDOM SPLIT) ---
    X_train, X_test, y_train, y_test = train_test_split(X_pure, y, test_size=0.2, random_state=42)

    if task == "Classification":
        from sklearn.metrics import confusion_matrix
        from sklearn.utils.multiclass import type_of_target # NEW: Target data type analyzer
        
        # NEW GUARDRAIL: Intercept continuous target variables before classification models fit and crash
        if type_of_target(y_train) == 'continuous':
            return {
                "error": (
                    f"Structural Mismatch: The selected target feature `{target}` contains continuous numeric data. "
                    "A classification pipeline expects discrete classes or categories (e.g., Churn, Status). "
                    "Please go back to the ML Setup page and change the Active Modeling Framework to 'Regression'."
                )
            }
        
        # Proceed safely with classification mapping if the data passes inspection
        if y_train.dtype == 'object':
            y_train = y_train.astype('category').cat.codes
            y_test = y_test.astype('category').cat.codes
            
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        preds_lr = lr.predict(X_test)
        cm_lr = confusion_matrix(y_test, preds_lr)
        
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        preds_rf = rf.predict(X_test)
        cm_rf = confusion_matrix(y_test, preds_rf)

        return {
            "task": "Classification",
            "Logistic Regression": {
                "Accuracy": accuracy_score(y_test, preds_lr),
                "F1 Score": f1_score(y_test, preds_lr, average='weighted', zero_division=0),
                "Confusion Matrix": cm_lr.tolist()
            },
            "Random Forest": {
                "Accuracy": accuracy_score(y_test, preds_rf),
                "F1 Score": f1_score(y_test, preds_rf, average='weighted', zero_division=0),
                "Confusion Matrix": cm_rf.tolist()
            }
        }

    elif task == "Regression":
        y_train, y_test = y_train.astype(float), y_test.astype(float)
        lin = LinearRegression()
        lin.fit(X_train, y_train)
        preds_lin = lin.predict(X_test)

        rfr = RandomForestRegressor(random_state=42)
        rfr.fit(X_train, y_train)
        preds_rfr = rfr.predict(X_test)

        return {
            "task": "Regression",
            "Linear Regression": {
                "MAE": mean_absolute_error(y_test, preds_lin),
                "RMSE": np.sqrt(mean_squared_error(y_test, preds_lin)),
                "R2 Score": r2_score(y_test, preds_lin)
            },
            "Random Forest Regressor": {
                "MAE": mean_absolute_error(y_test, preds_rfr),
                "RMSE": np.sqrt(mean_squared_error(y_test, preds_rfr)),
                "R2 Score": r2_score(y_test, preds_rfr)
            }
        }