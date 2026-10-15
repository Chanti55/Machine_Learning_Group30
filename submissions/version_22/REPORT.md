# Submission Report: version_22

**Created At:** 2026-10-15 01:05:41  
**Model Family:** `GradientBoostingClassifier`  
**Decision Threshold (\tau):** `0.4200`  
**Features Used:** 44  
**Notes:** Tuned Gradient Boosting (150 trees) on 44 Vote features at F1-calibrated tau=0.42 (50 diffs vs v18, 74.3% repeats)  

---

## 📋 Fluent Kaggle Submission Description (Copy & Paste)
> Group 30 Version 22 predicts 90-day wholesale repeat purchases using a tuned GradientBoostingClassifier trained on 6,469 cleaned transactions across 44 curated features. Feature selection strictly eliminated identifier-like variables (customer_id and rare transport methods) and benchmarked all candidates against an empirical random_noise floor. Evaluated across 5 customer-grouped folds (GroupKFold), it achieves a mean ROC-AUC of 0.7903 and F1 of 0.7568, while maintaining a temporal generalization ROC-AUC of 0.7845 across 5 chronological splits. Predictions are calibrated at an operating decision threshold of tau = 0.42, projecting an expected repeat purchase rate of 76.0% across the 986 test purchase occasions. Context: Tuned Gradient Boosting (150 trees) on 44 Vote features at F1-calibrated tau=0.42 (50 diffs vs v18, 74.3% repeats).

---

## 1. Test Prediction Sanity Check
* **Total test rows:** 986
* **Predicted Repeat (1):** 749 (75.96%)
* **Predicted No-Repeat (0):** 237 (24.04%)
*(Reference train repeat rate: 55.97%)*

---

## 2. Local Cross-Validation Scores
```json
{
  "customer_val_roc_auc": 0.7903,
  "customer_val_f1": 0.7568,
  "temporal_val_roc_auc": 0.7845
}
```

---

## 3. Replication Artifacts in this Folder
* **`ML_Group30_Version22.csv`**: Verified Kaggle test predictions.
* **`description.txt`**: Fluent submission text ready to paste on Kaggle.
* **`config.json`**: Exact estimator parameters and threshold setting.
* **`selected_features.json`**: Exact 44 encoded features passed to the estimator.
* **`cleaning_recipe.json`**: The preprocessing and cleaning recipe used for this run.
* **`model.joblib`**: Fitted model artifact ready for single-line inference.
* **`cv_metrics.json`**: JSON record for programmatic comparisons across versions.
