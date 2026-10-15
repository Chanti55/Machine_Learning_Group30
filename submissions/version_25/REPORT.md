# Submission Report: version_25

**Created At:** 2026-10-15 01:36:37  
**Model Family:** `LogisticRegression`  
**Decision Threshold (\tau):** `0.5000`  
**Features Used:** 41  
**Notes:** LR_L1_tau50: LogisticRegression (C=0.01) on L1 features (41) at tau=0.50 | CV F1=0.7507 Acc=0.7245 AUC=0.7906  

---

## 📋 Fluent Kaggle Submission Description (Copy & Paste)
> Group 30 Version 25 predicts 90-day wholesale repeat purchases using a tuned LogisticRegression trained on 6,469 cleaned transactions across 41 curated features. Feature selection strictly eliminated identifier-like variables (customer_id and rare transport methods) and benchmarked all candidates against an empirical random_noise floor. Evaluated across 5 customer-grouped folds (GroupKFold), it achieves a mean ROC-AUC of 0.7906 and F1 of 0.7507. Predictions are calibrated at an operating decision threshold of tau = 0.50, projecting an expected repeat purchase rate of 67.1% across the 986 test purchase occasions. Context: LR_L1_tau50: LogisticRegression (C=0.01) on L1 features (41) at tau=0.50 | CV F1=0.7507 Acc=0.7245 AUC=0.7906.

---

## 1. Test Prediction Sanity Check
* **Total test rows:** 986
* **Predicted Repeat (1):** 662 (67.14%)
* **Predicted No-Repeat (0):** 324 (32.86%)
*(Reference train repeat rate: 55.97%)*

---

## 2. Local Cross-Validation Scores
```json
{
  "customer_val_roc_auc": 0.7906,
  "customer_val_f1": 0.7507,
  "cv_accuracy": 0.7245
}
```

---

## 3. Replication Artifacts in this Folder
* **`ML_Group30_Version25.csv`**: Verified Kaggle test predictions.
* **`description.txt`**: Fluent submission text ready to paste on Kaggle.
* **`config.json`**: Exact estimator parameters and threshold setting.
* **`selected_features.json`**: Exact 41 encoded features passed to the estimator.
* **`cleaning_recipe.json`**: The preprocessing and cleaning recipe used for this run.
* **`model.joblib`**: Fitted model artifact ready for single-line inference.
* **`cv_metrics.json`**: JSON record for programmatic comparisons across versions.
