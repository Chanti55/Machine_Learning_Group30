# Submission Report: version_21

**Created At:** 2026-10-15 01:05:41  
**Model Family:** `LogisticRegression`  
**Decision Threshold (\tau):** `0.5000`  
**Features Used:** 44  
**Notes:** Tuned L2 Logistic Regression (C=0.1) on 44 Vote features at standard tau=0.50 (113 diffs vs v18, 65.1% repeats)  

---

## 📋 Fluent Kaggle Submission Description (Copy & Paste)
> Group 30 Version 21 predicts 90-day wholesale repeat purchases using a tuned LogisticRegression trained on 6,469 cleaned transactions across 44 curated features. Feature selection strictly eliminated identifier-like variables (customer_id and rare transport methods) and benchmarked all candidates against an empirical random_noise floor. Evaluated across 5 customer-grouped folds (GroupKFold), it achieves a mean ROC-AUC of 0.7962 and F1 of 0.7497, while maintaining a temporal generalization ROC-AUC of 0.7931 across 5 chronological splits. Predictions are calibrated at an operating decision threshold of tau = 0.50, projecting an expected repeat purchase rate of 65.1% across the 986 test purchase occasions. Context: Tuned L2 Logistic Regression (C=0.1) on 44 Vote features at standard tau=0.50 (113 diffs vs v18, 65.1% repeats).

---

## 1. Test Prediction Sanity Check
* **Total test rows:** 986
* **Predicted Repeat (1):** 642 (65.11%)
* **Predicted No-Repeat (0):** 344 (34.89%)
*(Reference train repeat rate: 55.97%)*

---

## 2. Local Cross-Validation Scores
```json
{
  "customer_val_roc_auc": 0.7962,
  "customer_val_f1": 0.7497,
  "temporal_val_roc_auc": 0.7931
}
```

---

## 3. Replication Artifacts in this Folder
* **`ML_Group30_Version21.csv`**: Verified Kaggle test predictions.
* **`description.txt`**: Fluent submission text ready to paste on Kaggle.
* **`config.json`**: Exact estimator parameters and threshold setting.
* **`selected_features.json`**: Exact 44 encoded features passed to the estimator.
* **`cleaning_recipe.json`**: The preprocessing and cleaning recipe used for this run.
* **`model.joblib`**: Fitted model artifact ready for single-line inference.
* **`cv_metrics.json`**: JSON record for programmatic comparisons across versions.
