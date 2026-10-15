# Submission Report: version_24

**Created At:** 2026-10-15 01:36:37  
**Model Family:** `LogisticRegression`  
**Decision Threshold (\tau):** `0.5100`  
**Features Used:** 44  
**Notes:** Best Accuracy: LogisticRegression (C=0.05) on Vote features (44) at tau=0.51 | CV F1=0.7492 Acc=0.7273 AUC=0.7945  

---

## 📋 Fluent Kaggle Submission Description (Copy & Paste)
> Group 30 Version 24 predicts 90-day wholesale repeat purchases using a tuned LogisticRegression trained on 6,469 cleaned transactions across 44 curated features. Feature selection strictly eliminated identifier-like variables (customer_id and rare transport methods) and benchmarked all candidates against an empirical random_noise floor. Evaluated across 5 customer-grouped folds (GroupKFold), it achieves a mean ROC-AUC of 0.7945 and F1 of 0.7492. Predictions are calibrated at an operating decision threshold of tau = 0.51, projecting an expected repeat purchase rate of 64.8% across the 986 test purchase occasions. Context: Best Accuracy: LogisticRegression (C=0.05) on Vote features (44) at tau=0.51 | CV F1=0.7492 Acc=0.7273 AUC=0.7945.

---

## 1. Test Prediction Sanity Check
* **Total test rows:** 986
* **Predicted Repeat (1):** 639 (64.81%)
* **Predicted No-Repeat (0):** 347 (35.19%)
*(Reference train repeat rate: 55.97%)*

---

## 2. Local Cross-Validation Scores
```json
{
  "customer_val_roc_auc": 0.7945,
  "customer_val_f1": 0.7492,
  "cv_accuracy": 0.7273
}
```

---

## 3. Replication Artifacts in this Folder
* **`ML_Group30_Version24.csv`**: Verified Kaggle test predictions.
* **`description.txt`**: Fluent submission text ready to paste on Kaggle.
* **`config.json`**: Exact estimator parameters and threshold setting.
* **`selected_features.json`**: Exact 44 encoded features passed to the estimator.
* **`cleaning_recipe.json`**: The preprocessing and cleaning recipe used for this run.
* **`model.joblib`**: Fitted model artifact ready for single-line inference.
* **`cv_metrics.json`**: JSON record for programmatic comparisons across versions.
