def generate_recommendations(quality_results, df):
    """
    Accepts the findings from the data quality scan and maps them to 
    actionable, business-justified data engineering recommendations.
    """
    recommendations = []
    
    # 1. Handle Missing Values
    if quality_results["total_missing"] > 0:
        for col, data in quality_results["missing_by_column"].items():
            # If a single column is missing more than half its data, recommend dropping it
            if data["pct"] > 50:
                recommendations.append({
                    "column": col,
                    "issue": f"Critical Missing Data ({data['pct']:.1f}% missing)",
                    "action": "Drop entire column",
                    "reason": "The missingness threshold exceeds 50%. Imputing this much data introduces significant bias and degrades predictive power."
                })
            else:
                # Differentiate handling by data type
                if df[col].dtype in ['int64', 'float64']:
                    recommendations.append({
                        "column": col,
                        "issue": f"Missing Numeric Values ({data['pct']:.1f}% missing)",
                        "action": "Impute with Median value",
                        "reason": "The median is robust against outliers and preserves the original distribution shape better than the mean for skewed features."
                    })
                else:
                    recommendations.append({
                        "column": col,
                        "issue": f"Missing Categorical Values ({data['pct']:.1f}% missing)",
                        "action": "Impute with Mode (Most frequent value) or 'Unknown' flag",
                        "reason": "Mode imputation maintains categorical frequency, while an explicit 'Unknown' category prevents loss of row data during modeling."
                    })

    # 2. Handle Duplicate Rows
    if quality_results["duplicate_count"] > 0:
        recommendations.append({
            "column": "Dataset Level",
            "issue": f"Found {quality_results['duplicate_count']:,} duplicate rows",
            "action": "Drop duplicate records completely",
            "reason": "Duplicate records cause data leakage. If split across train/test sets, they artificially inflate model validation metrics."
        })

    # 3. Handle Constant Columns
    if quality_results["constant_columns"]:
        for col in quality_results["constant_columns"]:
            recommendations.append({
                "column": col,
                "issue": "Constant Column (Zero Variance)",
                "action": "Remove column from modeling features",
                "reason": "A feature with zero variance provides zero predictive information to a machine learning algorithm and wastes computational memory."
            })

    # 4. Handle Potential Identifier Columns
    if quality_results["potential_identifiers"]:
        for col in quality_results["potential_identifiers"]:
            recommendations.append({
                "column": col,
                "issue": "High Cardinality / Unique Identifier Key",
                "action": "Exclude from predictive modeling features",
                "reason": "Primary keys or raw sequence numbers have perfect uniqueness but hold zero generalizable patterns for predictive algorithms."
            })

    return recommendations