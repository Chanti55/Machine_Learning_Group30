# Submission Report: version_28

**Created At:** 2026-10-15 01:36:41  
**Model Family:** `SoftBlend`  
**Decision Threshold (\tau):** `0.4400`  
**Features Used:** 44  
**Notes:** Best Blend: SoftBlend(LR=80%+GB=15%+RF=5%) (C=0.01) on Vote features (44) at tau=0.44 | CV F1=0.7578 Acc=0.7159 AUC=0.7936  

---

## 📋 Fluent Kaggle Submission Description (Copy & Paste)
> Group 30 Version 28 predicts 90-day wholesale repeat purchases using a tuned SoftBlend trained on 6,469 cleaned transactions across 44 curated features. Feature selection strictly eliminated identifier-like variables (customer_id and rare transport methods) and benchmarked all candidates against an empirical random_noise floor. Evaluated across 5 customer-grouped folds (GroupKFold), it achieves a mean ROC-AUC of 0.7936 and F1 of 0.7578. Predictions are calibrated at an operating decision threshold of tau = 0.44, projecting an expected repeat purchase rate of 73.5% across the 986 test purchase occasions. Context: Best Blend: SoftBlend(LR=80%+GB=15%+RF=5%) (C=0.01) on Vote features (44) at tau=0.44 | CV F1=0.7578 Acc=0.7159 AUC=0.7936.

---

## 1. Test Prediction Sanity Check
* **Total test rows:** 986
* **Predicted Repeat (1):** 725 (73.53%)
* **Predicted No-Repeat (0):** 261 (26.47%)
*(Reference train repeat rate: 55.97%)*

---

## 2. Local Cross-Validation Scores
```json
{
  "customer_val_roc_auc": 0.7936,
  "customer_val_f1": 0.7578,
  "cv_accuracy": 0.7159
}
```

---

## 3. Replication Artifacts in this Folder
* **`ML_Group30_Version28.csv`**: Verified Kaggle test predictions.
* **`description.txt`**: Fluent submission text ready to paste on Kaggle.
* **`config.json`**: Exact estimator parameters and threshold setting.
* **`selected_features.json`**: Exact 44 encoded features passed to the estimator.
* **`cleaning_recipe.json`**: The preprocessing and cleaning recipe used for this run.
* **`model.joblib`**: Fitted model artifact ready for single-line inference.
* **`cv_metrics.json`**: JSON record for programmatic comparisons across versions.
