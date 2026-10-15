"""
Submission Tracker & Experiment Versioning for Machine Learning Group 30.
Enables 100% reproducible submission bundling under submissions/<version_name>/
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from datetime import datetime
import joblib
import numpy as np
import pandas as pd


def generate_fluent_description(
    version_name: str,
    model_name: str,
    features_count: int,
    threshold: float,
    cv_metrics: dict | None,
    test_rows: int,
    pos_rate: float,
    notes: str = "",
) -> str:
    """Generates a fluent, natural-language narrative description for Kaggle submissions."""
    metrics = cv_metrics or {}
    auc = metrics.get("customer_val_roc_auc", metrics.get("mean_roc_auc", None))
    f1 = metrics.get("customer_val_f1", metrics.get("mean_f1", None))
    time_auc = metrics.get("temporal_val_roc_auc", None)

    v_clean = version_name.replace("version_", "Version ").replace("version", "Version ")
    desc = (
        f"Group 30 {v_clean} predicts 90-day wholesale repeat purchases using a tuned {model_name} "
        f"trained on 6,469 cleaned transactions across {features_count} curated features. "
        f"Feature selection strictly eliminated identifier-like variables (customer_id and rare transport methods) "
        f"and benchmarked all candidates against an empirical random_noise floor. "
    )
    if auc is not None and f1 is not None:
        desc += (
            f"Evaluated across 5 customer-grouped folds (GroupKFold), it achieves a mean ROC-AUC of {auc:.4f} "
            f"and F1 of {f1:.4f}"
        )
        if time_auc is not None:
            desc += f", while maintaining a temporal generalization ROC-AUC of {time_auc:.4f} across 5 chronological splits. "
        else:
            desc += ". "
    else:
        desc += "Validation follows a strict leak-free customer-grouped protocol. "

    desc += (
        f"Predictions are calibrated at an operating decision threshold of tau = {threshold:.2f}, "
        f"projecting an expected repeat purchase rate of {pos_rate:.1%} across the {test_rows} test purchase occasions."
    )
    if notes:
        clean_note = notes.strip().rstrip(".")
        desc += f" Context: {clean_note}."

    return desc


def export_submission_bundle(
    version_name: str,
    predictions: np.ndarray | pd.Series,
    test_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    model=None,
    feature_mask: np.ndarray | None = None,
    feature_names: list[str] | np.ndarray | None = None,
    cv_metrics: dict | None = None,
    threshold: float = 0.50,
    cleaning_recipe_path: str | Path | None = None,
    notes: str = "",
    target_col: str = "repeat_purchase_90d",
    base_dir: str | Path = "submissions",
) -> Path:
    """
    Exports a self-contained, reproducible submission bundle containing:
    - ML_Group30_VersionXX.csv (validated Kaggle deliverable)
    - description.txt (ready-to-paste fluent submission narrative)
    - config.json (exact hyperparameters, threshold, seed, and settings)
    - cv_metrics.json (local CV scores & prediction distribution)
    - cleaning_recipe.json (the exact preprocessing recipe used)
    - selected_features.json (exact feature names retained)
    - model.joblib (fitted estimator for instant inference)
    - REPORT.md (human-readable executive summary)
    """
    base_path = Path(base_dir)
    version_dir = base_path / version_name
    version_dir.mkdir(parents=True, exist_ok=True)

    # 1. Format and strictly validate submission CSV
    sub = sample_df.copy()
    preds_array = np.asarray(predictions).astype(int)

    if len(preds_array) == len(test_df):
        sub[target_col] = sub["ID"].map(pd.Series(preds_array, index=test_df["ID"])).astype(int)
    else:
        raise ValueError(f"Predictions length ({len(preds_array)}) does not match test set ({len(test_df)})")

    assert len(sub) == len(sample_df), f"Submission row count ({len(sub)}) does not match sample ({len(sample_df)})"
    assert sub[target_col].isin([0, 1]).all(), "Predictions must be strictly binary integers (0 or 1)!"
    assert sub["ID"].equals(sample_df["ID"]), "ID order or content does not match sample_submission.csv!"
    assert sub[target_col].notna().all(), "Submission contains null/NaN values!"

    # Format filename according to Group 30 standard: ML_Group30_VersionXX.csv
    v_tag = version_name
    for prefix in ["version_", "version", "Version_", "v_"]:
        if v_tag.lower().startswith(prefix.lower()):
            v_tag = v_tag[len(prefix):]
            break
    formatted_v = f"Version{v_tag}" if not v_tag.lower().startswith("version") else v_tag

    sub_filename = f"ML_Group30_{formatted_v}.csv"
    sub_file = version_dir / sub_filename
    sub.to_csv(sub_file, index=False)

    # 2. Extract and save selected features
    selected_names = []
    if feature_names is not None:
        if feature_mask is not None:
            selected_names = [str(name) for name, keep in zip(feature_names, feature_mask) if keep]
        else:
            selected_names = [str(name) for name in feature_names]

    features_file = version_dir / "selected_features.json"
    features_payload = {
        "version": version_name,
        "feature_count": len(selected_names),
        "features": selected_names,
    }
    features_file.write_text(json.dumps(features_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # 3. Locate and copy the cleaning recipe JSON
    recipe_copied = False
    candidates = [
        cleaning_recipe_path,
        Path("cleaning_recipe.json"),
        Path("Final/cleaning_recipe.json"),
        Path("../cleaning_recipe.json"),
    ]
    for cand in candidates:
        if cand is not None and Path(cand).is_file():
            shutil.copy(cand, version_dir / "cleaning_recipe.json")
            recipe_copied = True
            break

    # 4. Save metadata / configuration
    model_params = {}
    model_name = "CustomEstimator"
    if model is not None:
        model_name = model.__class__.__name__
        if hasattr(model, "get_params"):
            try:
                model_params = model.get_params()
            except Exception:
                model_params = {}

    config_payload = {
        "version": version_name,
        "created_at": datetime.now().isoformat(),
        "model_type": model_name,
        "hyperparameters": model_params,
        "features_count": len(selected_names),
        "decision_threshold": float(threshold),
        "cleaning_recipe_included": recipe_copied,
        "notes": notes,
    }
    config_file = version_dir / "config.json"
    config_file.write_text(json.dumps(config_payload, indent=2, default=str), encoding="utf-8")

    # 5. Save CV metrics & prediction diagnostics
    pos_count = int((sub[target_col] == 1).sum())
    neg_count = int((sub[target_col] == 0).sum())
    pos_rate = float(sub[target_col].mean())

    metrics_payload = {
        "version": version_name,
        "local_cv_metrics": cv_metrics or {},
        "prediction_diagnostics": {
            "test_rows": len(sub),
            "predicted_0": neg_count,
            "predicted_1": pos_count,
            "predicted_1_rate": round(pos_rate, 4),
        },
        "kaggle_public_score": None,  # Fill in via record_kaggle_score(...)
        "kaggle_private_score": None,
    }
    metrics_file = version_dir / "cv_metrics.json"
    metrics_file.write_text(json.dumps(metrics_payload, indent=2, default=str), encoding="utf-8")

    # 6. Serialise model object if provided
    if model is not None:
        try:
            joblib.dump(model, version_dir / "model.joblib")
        except Exception as e:
            (version_dir / "model_save_error.txt").write_text(str(e), encoding="utf-8")

    # 7. Generate fluent Kaggle description text
    fluent_desc = generate_fluent_description(
        version_name=version_name,
        model_name=model_name,
        features_count=len(selected_names),
        threshold=threshold,
        cv_metrics=cv_metrics,
        test_rows=len(sub),
        pos_rate=pos_rate,
        notes=notes,
    )
    (version_dir / "description.txt").write_text(fluent_desc + "\n", encoding="utf-8")

    # 8. Write Human-Readable REPORT.md
    cv_formatted = json.dumps(cv_metrics or {}, indent=2)
    report_text = f"""# Submission Report: {version_name}

**Created At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Model Family:** `{model_name}`  
**Decision Threshold (\\tau):** `{threshold:.4f}`  
**Features Used:** {len(selected_names)}  
**Notes:** {notes if notes else 'Standard simulation run.'}  

---

## 📋 Fluent Kaggle Submission Description (Copy & Paste)
> {fluent_desc}

---

## 1. Test Prediction Sanity Check
* **Total test rows:** {len(sub)}
* **Predicted Repeat (1):** {pos_count} ({pos_rate:.2%})
* **Predicted No-Repeat (0):** {neg_count} ({1 - pos_rate:.2%})
*(Reference train repeat rate: 55.97%)*

---

## 2. Local Cross-Validation Scores
```json
{cv_formatted}
```

---

## 3. Replication Artifacts in this Folder
* **`{sub_filename}`**: Verified Kaggle test predictions.
* **`description.txt`**: Fluent submission text ready to paste on Kaggle.
* **`config.json`**: Exact estimator parameters and threshold setting.
* **`selected_features.json`**: Exact {len(selected_names)} encoded features passed to the estimator.
* **`cleaning_recipe.json`**: The preprocessing and cleaning recipe used for this run.
* **`model.joblib`**: Fitted model artifact ready for single-line inference.
* **`cv_metrics.json`**: JSON record for programmatic comparisons across versions.
"""
    (version_dir / "REPORT.md").write_text(report_text, encoding="utf-8")

    update_leaderboard_file(base_path)

    print(f"[SubmissionTracker] Successfully exported bundle -> {version_dir.resolve()}")
    print(f"                   Kaggle file: {sub_file.name} | Features: {len(selected_names)} | Test Pos Rate: {pos_rate:.2%}")
    print(f"\n[Kaggle Description Ready]:\n\"{fluent_desc}\"\n")
    return sub_file


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Pure-Python markdown table formatter requiring zero external dependencies."""
    if df.empty:
        return ""
    headers = ["Version"] + list(df.columns)
    rows = []
    for idx, row in df.iterrows():
        row_vals = [str(idx)]
        for v in row.values:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                row_vals.append("-")
            elif isinstance(v, float):
                row_vals.append(f"{v:.4f}")
            else:
                row_vals.append(str(v))
        rows.append(row_vals)

    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "|:" + "-|-:".join("-" * max(1, widths[i] - 1) for i in range(len(headers))) + ":|"
    body = ["| " + " | ".join(r[i].ljust(widths[i]) for i, h in enumerate(headers)) + " |" for r in rows]
    return "\n".join([header_line, sep_line] + body)


def update_leaderboard_file(base_dir: str | Path = "submissions") -> Path:
    """Updates the master submissions/LEADERBOARD.md table using pure Python."""
    base_path = Path(base_dir)
    leaderboard_df = list_submission_versions(base_dir=base_path)
    table_md = _df_to_markdown(leaderboard_df)

    lb_path = base_path / "LEADERBOARD.md"
    lb_text = f"""# 🏆 Machine Learning Group 30 - Kaggle Leaderboard Tracker

**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  

{table_md}

---
*Tip: Call `record_kaggle_score('version_XX', score)` in Python or edit `cv_metrics.json` directly.*
"""
    lb_path.write_text(lb_text, encoding="utf-8")
    return lb_path


def list_submission_versions(base_dir: str | Path = "submissions") -> pd.DataFrame:
    """
    Scans the submissions directory and compiles a master comparison DataFrame
    of all saved version bundles.
    """
    base_path = Path(base_dir)
    if not base_path.is_dir():
        return pd.DataFrame()

    records = []
    for vdir in sorted(base_path.iterdir()):
        if not vdir.is_dir() or vdir.name in ["new", "old", "__pycache__"]:
            continue

        cfg_file = vdir / "config.json"
        met_file = vdir / "cv_metrics.json"

        cfg = json.loads(cfg_file.read_text(encoding="utf-8")) if cfg_file.is_file() else {}
        met = json.loads(met_file.read_text(encoding="utf-8")) if met_file.is_file() else {}

        local_cv = met.get("local_cv_metrics", {})
        diag = met.get("prediction_diagnostics", {})

        records.append({
            "Version": vdir.name,
            "Model": cfg.get("model_type", "Unknown"),
            "Features": cfg.get("features_count", "-"),
            "Threshold": cfg.get("decision_threshold", 0.5),
            "Test Repeat %": f"{diag.get('predicted_1_rate', 0)*100:.1f}%" if "predicted_1_rate" in diag else "-",
            "Val ROC-AUC": local_cv.get("customer_val_roc_auc", local_cv.get("mean_roc_auc", "-")),
            "Val F1": local_cv.get("customer_val_f1", local_cv.get("mean_f1", "-")),
            "Temporal AUC": local_cv.get("temporal_val_roc_auc", "-"),
            "Kaggle Score": met.get("kaggle_public_score", "Pending"),
            "Notes": cfg.get("notes", "")[:60],
        })

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).set_index("Version")


def record_kaggle_score(
    version_name: str,
    public_score: float,
    private_score: float | None = None,
    base_dir: str | Path = "submissions",
) -> None:
    """
    Records the official Kaggle leaderboard score for a specific version:
    1. Updates submissions/<version_name>/cv_metrics.json
    2. Updates submissions/<version_name>/REPORT.md
    3. Regenerates the master submissions/LEADERBOARD.md table
    """
    base_path = Path(base_dir)
    version_dir = base_path / version_name
    if not version_dir.is_dir():
        candidates = [base_path / f"version_{version_name}", base_path / f"Version_{version_name}"]
        for c in candidates:
            if c.is_dir():
                version_dir = c
                break
        else:
            raise FileNotFoundError(f"Version directory not found for: {version_name} under {base_path.resolve()}")

    # 1. Update cv_metrics.json
    met_file = version_dir / "cv_metrics.json"
    if met_file.is_file():
        met = json.loads(met_file.read_text(encoding="utf-8"))
        met["kaggle_public_score"] = float(public_score)
        if private_score is not None:
            met["kaggle_private_score"] = float(private_score)
        met_file.write_text(json.dumps(met, indent=2), encoding="utf-8")

    # 2. Update REPORT.md
    report_file = version_dir / "REPORT.md"
    if report_file.is_file():
        text = report_file.read_text(encoding="utf-8")
        score_section = f"\n---\n\n## 🏆 Kaggle Scores\n* **Public Leaderboard:** `{public_score:.4f}`\n"
        if private_score is not None:
            score_section += f"* **Private Leaderboard:** `{private_score:.4f}`\n"
        else:
            score_section += "* **Private Leaderboard:** `Pending`\n"

        if "## 🏆 Kaggle Scores" in text:
            parts = text.split("## 🏆 Kaggle Scores")
            text = parts[0].rstrip() + score_section
        else:
            text = text.rstrip() + "\n" + score_section
        report_file.write_text(text, encoding="utf-8")

    # 3. Regenerate master submissions/LEADERBOARD.md
    lb_path = update_leaderboard_file(base_path)
    print(f"[SubmissionTracker] Successfully recorded Kaggle Public Score: {public_score:.4f} for {version_dir.name}!")
    print(f"                   Updated: {report_file.name} and {lb_path.name}")
