def translate_metrics_to_business(metrics):
    """
    Accepts raw machine learning evaluation metrics and generates 
    plain-English interpretations for non-technical stakeholders.
    """
    task = metrics.get("task")
    interpretations = {}
    
    if task == "Clustering":
        interpretations["Core Finding"] = (
            f"The optimization engine identified {metrics['num_clusters']} distinct behavioral segments "
            f"within your customer data. These groupings represent natural operational variations "
            f"that require tailored, segment-specific strategies rather than a single approach."
        )
        return interpretations

    # Process Supervised & Time-Series Models
    models = [k for k in metrics.keys() if k != "task"]
    
    for model_name in models:
        model_metrics = metrics[model_name]
        model_notes = []
        
        # --- 1. CLASSIFICATION TRANSLATION ---
        if task == "Classification":
            accuracy = model_metrics.get("Accuracy", 0)
            f1 = model_metrics.get("F1 Score", 0)
            
            model_notes.append(
                f"Operational Reliability: The model achieves an overall accuracy of {accuracy*100:.1f}%, "
                f"meaning it successfully forecasts the correct categorical outcome for approximately "
                f"{int(accuracy*100)} out of every 100 business records processed."
            )
            if f1 < 0.70:
                model_notes.append(
                    "Risk Alert: The lower balance between precision and recall indicates the model may struggle "
                    "with minority classes. It is recommended to apply class-weight adjustments before deployment."
                )
            else:
                model_notes.append(
                    "Strategic Viability: The strong structural balance proves this framework is stable enough "
                    "to support automated sorting queues and target allocation pipelines."
                )
                
        # --- 2. REGRESSION TRANSLATION ---
        elif task == "Regression":
            r2 = model_metrics.get("R2 Score", 0)
            mae = model_metrics.get("MAE", 0)
            
            model_notes.append(
                f"Variance Explanation: This model accounts for {r2*100:.1f}% of the total historical "
                f"variation seen in your target KPI. The remaining variance is driven by external operational "
                f"factors outside this dataset's features."
            )
            model_notes.append(
                f"Expected Margin of Error: For any single prediction, the model's forecast will typically "
                f"deviate from the true physical outcome by an average value of +/- {mae:,.2f} units."
            )

        # --- 3. TIME-SERIES FORECASTING TRANSLATION (ADD THIS) ---
        elif task == "Time-Series Forecasting":
            r2 = model_metrics.get("R2 Score", 0)
            mae = model_metrics.get("MAE", 0)
            
            model_notes.append(
                f"Historical Pattern Capturing: This sequential model explains {r2*100:.1f}% of the up-and-down "
                f"chronological variance and seasonal trends present in your target metric timeline."
            )
            model_notes.append(
                f"Operational Safety Stock Buffer: For future chronological periods, expectations should factor in "
                f"an average margin of error of +/- {mae:,.2f} units. Logistics teams should use this metric "
                f"to directly establish warehouse safety stock thresholds and minimize out-of-stock events."
            )
            
        interpretations[model_name] = model_notes

    return interpretations