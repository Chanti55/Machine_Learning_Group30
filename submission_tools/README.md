# Submission & Reporting Tools (Group 30)

These are the scripts we use to turn model predictions into Kaggle submissions without
losing track of what we actually submitted. Every version gets its own folder with the
predictions, the config and a short report, so we can still reconstruct a run weeks later.

## Files

| File | What it does |
| :--- | :--- |
| `submission_tracker.py` | The main module. Exports a version bundle, writes its `REPORT.md`, builds the Kaggle description text and rebuilds `submissions/LEADERBOARD.md`. |
| `record_score.py` | Small CLI for entering the public leaderboard score once Kaggle has scored a submission. |
| `batch_exporter.py` | Example script: fits a few models, tries different thresholds and exports each result as a new version. Mostly there as a reference. |

## Dependencies

Nothing beyond `requirements.txt`. From the standard library we use `json`, `shutil`,
`pathlib` and `datetime`; otherwise it's `numpy`, `pandas` and `scikit-learn`
(`joblib` comes with it). We first built the leaderboard table with `tabulate`, then
dropped it because writing a Markdown table by hand is about ten lines.

## Usage

### Recording a Kaggle score

After Kaggle returns a score:

```bash
python submission_tools/record_score.py version_21 0.7947
```

Run it without arguments and it asks for the version and score. Either way, it updates
three files: `cv_metrics.json` and `REPORT.md` in the version folder, plus
`submissions/LEADERBOARD.md`.

### Exporting a new version

From a notebook or script:

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

That creates `submissions/version_23/` containing:

- `ML_Group30_Version23.csv`: the predictions. The function checks that all 986 test rows are present before writing.
- `REPORT.md`: metrics, threshold and the distribution of predicted classes
- `description.txt`: text you can paste straight into the Kaggle submission description
- `config.json`: hyperparameters and settings, so the run can be repeated

One thing to watch: `version_name` is not checked against existing folders. If you reuse
a name, the old bundle gets overwritten.` *(fitted estimator for instant inference)*
* Refreshes `submissions/LEADERBOARD.md`
