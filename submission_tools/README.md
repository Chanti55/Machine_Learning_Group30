# 🛠️ Submission & Reporting Tools (Group 30)

This folder contains the complete, reproducible tooling used to generate test predictions, package version bundles, auto-generate markdown reports, and maintain the Kaggle leaderboard.

---

## 📦 What's Inside

| File | Purpose |
| :--- | :--- |
| **`submission_tracker.py`** | Core engine that exports self-contained version bundles, generates individual `REPORT.md` files, formats fluent Kaggle descriptions, and refreshes `submissions/LEADERBOARD.md`. |
| **`record_score.py`** | Command-line tool to quickly record official Kaggle leaderboard scores and update all reports in one step. |
| **`batch_exporter.py`** | Reference batch execution script showing how to fit models, sweep thresholds, and export new versions. |

---

## ⚡ Zero Extra Dependencies
This toolkit is built using **100% pure Python** plus the standard packages already present in `requirements.txt`:
* Standard Library: `json`, `shutil`, `pathlib`, `datetime`
* Core Packages: `numpy`, `pandas`, `scikit-learn` (includes `joblib`)

**No extra packages (such as `tabulate`) are needed.**

---

## 🚀 Quick Usage

### 1. Recording a Kaggle Score
When you submit a prediction to Kaggle and receive a score, run:
```bash
python submission_tools/record_score.py version_21 0.7947
```
*(Or simply run `python submission_tools/record_score.py` and follow the interactive prompts).*

This automatically updates:
1. `submissions/version_XX/cv_metrics.json`
2. `submissions/version_XX/REPORT.md`
3. `submissions/LEADERBOARD.md`

### 2. Exporting a New Version Bundle from Python
In any notebook or script, you can package a submission with one call:

```python
import sys
sys.path.insert(0, "submission_tools")
from submission_tracker import export_submission_bundle

sub_path = export_submission_bundle(
    version_name="version_23",
    predictions=y_pred_binary,
    test_df=test,
    sample_df=sample,
    model=my_model,
    feature_names=all_feature_names,
    feature_mask=selected_mask,
    cv_metrics={"customer_val_roc_auc": 0.795, "customer_val_f1": 0.760},
    threshold=0.50,
    notes="Description of experiment",
    base_dir="submissions"
)
```

This automatically generates the entire bundle in `submissions/version_23/`:
* `ML_Group30_Version23.csv` *(validated 986-row test predictions)*
* `REPORT.md` *(full markdown report with distribution & metrics)*
* `description.txt` *(fluent copy-paste text for Kaggle)*
* `config.json` *(exact hyperparameters and settings)*
* `selected_features.json` *(retained feature names)*
* `cleaning_recipe.json` *(preprocessing recipe used)*
* `model.joblib` *(fitted estimator for instant inference)*
* Refreshes `submissions/LEADERBOARD.md`
