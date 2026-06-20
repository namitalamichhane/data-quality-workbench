---
title: Data Quality Workbench
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# Data Quality & Predictive Analytics Workbench

A self-service enterprise analytics platform built with Python and Streamlit. The application automatically profiles uploaded data assets, generates an executive-ready Health Score, extracts data quality issues with badged severity levels, recommends remediation transformations, and builds automated, crash-proof machine learning pipelines.

## 🔗 Live Application Demo
Explore the interactive SaaS platform live here: **[Launch Live Cloud Workbench](https://huggingface.co/spaces/namita8584/data-quality-workbench)**

*No dataset ready? Open the ingestion dropdown inside the app to synchronize real-time simulated business parameters for Telecom Churn, Airlines Demand, or Healthcare Risk profiles instantly.*

---

##  Core Platform Features

### 1. Ingestion Center & Cross-Domain Blueprints
* **Dynamic File Validator:** Enforces production resource constraints (Max 10MB size, Max 20,000 row matrices) to maintain server health and protect against application crashes.
* **Curated Business Demo Injectors:** Features one-click sample environment builders for immediate technical recruitment testing without an external file upload:
  * **Telecom Customer Churn Analytics:** Auto-configures a Classification pipeline.
  * **Airlines Passenger Demand Logs:** Auto-configures a Time-Series Forecasting pipeline with heavy seasonality variance curves.
  * **Healthcare Patient Readmission Risk:** Auto-configures a continuous quantitative Regression model.

### 2. Comprehensive Data Quality Diagnostics
* **Statistical Health Audit:** Scans variables for missing rows, absolute duplicates, zero-variance/constant attributes, high-cardinality flags, and primary key identifiers.
* **Algorithmic Scoring Matrix:** Translates diagnostic alerts into an objective **Dataset Health Score (0–100)** utilizing transparent multi-factor penalty rules.

### 3. Smart Architecture Recommender & Manual Override
* **Semantic Keyword Processor:** Parses data columns for business intent keywords (`churn`, `sales`, `status`, `val`) paired with an algorithmic data-type fallback analyzer to automatically prescribe the most effective ML task framework.
* **Interactive Feature Selection Panel:** Enables users to manually toggle independent predictor variables (IVs) to isolate features, automatically protecting against target variable data leakage.

### 4. Resilient Modeling Engine & Performance Sandbox
* **Supervised Learning Rails:** Trains competitive baseline pairs (`Linear Regression` vs. `Random Forest Regressor` for continuous tracks; `Logistic Regression` vs. `Random Forest Classifier` for categorical tracks).
* **Chronological Split Validation:** Features an $80/20$ historical train/test split mechanism for Time-Series forecasting to prevent lookahead bias and model validation corruption.
* **Memory-Safe Unsupervised Profiling:** Limits K-Means silhouette calculations to a fixed sample size to eliminate memory allocation exceptions (`_ArrayMemoryError`) on larger matrices.

### 5. Business Impact Translation Layer
* Bridges the gap between data engineering and executive decision-making by automatically mapping technical evaluation metrics ($R^2$, MAE, RMSE, Accuracy, F1-Score) into clear, plain-language business impact interpretations.

---

##  Technical Stack & Architecture

* **Interface & Deployment:** Streamlit, Hugging Face Spaces Cloud, VS Code
* **Data Processing & Engineering:** Pandas, NumPy
* **Machine Learning & Diagnostics:** Scikit-Learn (Linear Models, Ensemble Trees, K-Means Clustering, Metric Analytics Validation)
* **Interactive Visualizations:** Plotly Express (Dynamic Histograms, Boxplots, Correlation Heatmaps)

---

##  System Directory Layout

* 📁 **`data-quality-workbench/`** — *Root directory of the application workspace*
  * 📄 **`app.py`** — *Main platform landing hub and entry point*
  * 📄 **`requirements.txt`** — *Explicit version control dependencies for cloud compilation*
  * 📁 **`pages/`** — *Multi-page Streamlit application modules*
    * 📄 `1_Upload.py` — *Data Ingestion Center & multi-option demo injectors*
    * 📄 `2_Data_Quality.py` — *Statistical health evaluation & severity badging*
    * 📄 `3_EDA.py` — *Outlier analysis and distribution plots*
    * 📄 `4_ML_Setup.py` — *Feature selectors & semantic target override mapping*
    * 📄 `5_Model_Training.py` — *Performance sandbox pipelines & business insights*
  * 📁 **`modules/`** — *Core algorithmic utility logic*
    * 📄 `ml_recommender.py` — *Semantic task prescription rules*
    * 📄 `model_trainer.py` — *Supervised/Unsupervised statistical modeling engines*

---

## 📊 Verification & Local Execution

To review the workbench architecture locally on your computer, clone the repository and initialize the Streamlit server:

```bash
# Clone the repository
git clone [https://github.com/namitalamichhane/data-quality-workbench.git](https://github.com/namitalamichhane/data-quality-workbench.git)
cd data-quality-workbench

# Configure and launch the isolated virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows, utilize: venv\Scripts\activate

# Install component requirements
pip install -r requirements.txt

# Start the workbench application with a clean cache profile
streamlit run app.py --server.clearCache true