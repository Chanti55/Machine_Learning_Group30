# Submission Report: version_17

**Created At:** 2026-10-15 00:17:12  
**Model Family:** `LogisticRegression`  
**Decision Threshold (\tau):** `0.4100`  
**Features Used:** 45  
**Notes:** Group 30 Version 17: Tuned L2 Logistic Regression (C=0.1) on 2-of-3 Majority Vote features at optimal F1 threshold (tau=0.41).  

---


---

## ?? Fluent Kaggle Submission Description (Copy & Paste)
> Group 30 Version 17 predicts 90-day wholesale repeat purchases using a tuned LogisticRegression trained on 6,469 cleaned transactions across 45 curated features. Feature selection strictly eliminated identifier-like variables (customer_id and rare transport methods) and benchmarked all candidates against an empirical random_noise floor. Evaluated across 5 customer-grouped folds (GroupKFold), it achieves a mean ROC-AUC of 0.7903 and F1 of 0.7574, while maintaining a temporal generalization ROC-AUC of 0.7931 across 5 chronological splits. Predictions are calibrated at an operating decision threshold of tau = 0.41, projecting an expected repeat purchase rate of 76.6% across the 986 test purchase occasions. Context: Group 30 Version 17: Tuned L2 Logistic Regression (C=0.1) on 2-of-3 Majority Vote features at optimal F1 threshold (tau=0.41).

## 1. Test Prediction Sanity Check
* **Total test rows:** 986
* **Predicted Repeat (1):** 755 (76.57%)
* **Predicted No-Repeat (0):** 231 (23.43%)
*(Reference train repeat rate: 55.97%)*

---

## 2. Local Cross-Validation Scores
```json
{
  "customer_val_roc_auc": 0.7903,
  "customer_val_f1": 0.7574,
  "customer_val_balanced_acc": 0.7255,
  "temporal_val_roc_auc": 0.7931
}
```

---

## 3. Replication Artifacts in this Folder
* **`ML_Group30_Version17.csv`**: Verified Kaggle test predictions.
* **`config.json`**: Exact estimator parameters and threshold setting.
* **`selected_features.json`**: Exact 45 encoded features passed to the estimator.
* **`cleaning_recipe.json`**: The preprocessing and cleaning recipe used for this run.
* **`model.joblib`**: Fitted model artifact ready for single-line inference.
* **`cv_metrics.json`**: JSON record for programmatic comparisons across versions.
