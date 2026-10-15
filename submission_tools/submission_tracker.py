from __future__ import annotations

import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

TRAINREPEATRATE = 0.5597   # repeat rate of the cleaned train rows, shown in REPORT.md
SKIPDIRS = ["new", "old", "__pycache__"]
RECIPEPATHS = [Path("cleaning_recipe.json"), Path("Final/cleaning_recipe.json"), Path("../cleaning_recipe.json")]
SCOREHEADER = "## 🏆 Kaggle Scores"


def readjson(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def writejson(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False), encoding="utf-8")


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# First key that exists in the metrics dict (customer folds first, then the mean over folds)
def getmetric(metrics: dict, *keys, default=None):
    return next((metrics[k] for k in keys if k in metrics), default)


# "version_03" -> "Version 03" for the description text
def versiontitle(version_name: str) -> str:
    return version_name.replace("version_", "Version ").replace("version", "Version ")


# "version_03" -> "ML_Group30_Version03.csv"
def submissionfilename(version_name: str) -> str:
    tag = version_name
    for prefix in ["version_", "version", "Version_", "v_"]:
        if tag.lower().startswith(prefix.lower()):
            tag = tag[len(prefix):]
            break
    if not tag.lower().startswith("version"):
        tag = f"Version{tag}"
    return f"ML_Group30_{tag}.csv"


def findversiondir(base_path: Path, version_name: str) -> Path:
    for folder in [version_name, f"version_{version_name}", f"Version_{version_name}"]:
        if (base_path / folder).is_dir():
            return base_path / folder
    raise FileNotFoundError(f"Version directory not found for: {version_name} under {base_path.resolve()}")



# Predictions in the order of sample_submission.csv, with the same checks as before
def buildsubmission(predictions, test_df: pd.DataFrame, sample_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    preds = np.asarray(predictions).astype(int)
    if len(preds) != len(test_df):
        raise ValueError(f"Predictions length ({len(preds)}) does not match test set ({len(test_df)})")

    sub = sample_df.copy()
    sub[target_col] = sub["ID"].map(pd.Series(preds, index=test_df["ID"])).astype(int)

    assert len(sub) == len(sample_df), f"Submission row count ({len(sub)}) does not match sample ({len(sample_df)})"
    assert sub[target_col].isin([0, 1]).all(), "Predictions must be strictly binary integers (0 or 1)!"
    assert sub["ID"].equals(sample_df["ID"]), "ID order or content does not match sample_submission.csv!"
    assert sub[target_col].notna().all(), "Submission contains null/NaN values!"
    return sub


def selectedfeatures(feature_names, feature_mask) -> list[str]:
    if feature_names is None:
        return []
    if feature_mask is None:
        return [str(name) for name in feature_names]
    return [str(name) for name, keep in zip(feature_names, feature_mask) if keep]


# Copies the first cleaning recipe that exists, returns whether one was found
def copyrecipe(cleaning_recipe_path, version_dir: Path) -> bool:
    for cand in [cleaning_recipe_path] + RECIPEPATHS:
        if cand is not None and Path(cand).is_file():
            shutil.copy(cand, version_dir / "cleaning_recipe.json")
            return True
    return False


def modelinfo(model) -> tuple[str, dict]:
    if model is None:
        return "CustomEstimator", {}
    params = model.get_params() if hasattr(model, "get_params") else {}
    return model.__class__.__name__, params


def savemodel(model, version_dir: Path) -> None:
    if model is None:
        return
    # A model that can not be pickled leaves a note instead of stopping the export
    try:
        joblib.dump(model, version_dir / "model.joblib")
    except (pickle.PicklingError, TypeError, AttributeError) as err:
        (version_dir / "model_save_error.txt").write_text(str(err), encoding="utf-8")




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
    metrics = cv_metrics or {}
    auc = getmetric(metrics, "customer_val_roc_auc", "mean_roc_auc")
    f1 = getmetric(metrics, "customer_val_f1", "mean_f1")
    timeauc = metrics.get("temporal_val_roc_auc")

    desc = (
        f"Group 30 {versiontitle(version_name)} predicts 90-day wholesale repeat purchases using a tuned {model_name} "
        f"trained on 6,469 cleaned transactions across {features_count} curated features. "
        f"Feature selection strictly eliminated identifier-like variables (customer_id and rare transport methods) "
        f"and benchmarked all candidates against an empirical random_noise floor. "
    )

    if auc is None or f1 is None:
        desc += "Validation follows a strict leak-free customer-grouped protocol. "
    else:
        desc += (f"Evaluated across 5 customer-grouped folds (GroupKFold), it achieves a mean ROC-AUC of {auc:.4f} "
                 f"and F1 of {f1:.4f}")
        if timeauc is None:
            desc += ". "
        else:
            desc += f", while maintaining a temporal generalization ROC-AUC of {timeauc:.4f} across 5 chronological splits. "

    desc += (
        f"Predictions are calibrated at an operating decision threshold of tau = {threshold:.2f}, "
        f"projecting an expected repeat purchase rate of {pos_rate:.1%} across the {test_rows} test purchase occasions."
    )
    if notes:
        desc += f" Context: {notes.strip().rstrip('.')}."
    return desc


def reporttext(version_name, model_name, threshold, featurecount, notes, desc, testrows, poscount, negcount,
               posrate, cv_metrics, subfilename) -> str:
    return f"""# Submission Report: {version_name}

**Created At:** {timestamp()}
**Model Family:** `{model_name}`
**Decision Threshold (\\tau):** `{threshold:.4f}`
**Features Used:** {featurecount}
**Notes:** {notes if notes else 'Standard simulation run.'}

def scoresection(public_score: float, private_score: float | None) -> str:
    private = f"`{private_score:.4f}`" if private_score is not None else "`Pending`"
    return (f"\n---\n\n{SCOREHEADER}\n* **Public Leaderboard:** `{public_score:.4f}`\n"
            f"* **Private Leaderboard:** {private}\n")

# main export
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
    base_path = Path(base_dir)
    version_dir = base_path / version_name
    version_dir.mkdir(parents=True, exist_ok=True)

    # 1. Submission csv
    sub = buildsubmission(predictions, test_df, sample_df, target_col)
    subfile = version_dir / submissionfilename(version_name)
    sub.to_csv(subfile, index=False)

    # 2. Selected features
    features = selectedfeatures(feature_names, feature_mask)
    writejson(version_dir / "selected_features.json",
              {"version": version_name, "feature_count": len(features), "features": features})

    # 3. Cleaning recipe
    recipecopied = copyrecipe(cleaning_recipe_path, version_dir)

    # 4. Config
    model_name, params = modelinfo(model)
    writejson(version_dir / "config.json", {
        "version": version_name,
        "created_at": datetime.now().isoformat(),
        "model_type": model_name,
        "hyperparameters": params,
        "features_count": len(features),
        "decision_threshold": float(threshold),
        "cleaning_recipe_included": recipecopied,
        "notes": notes,
    })

    # 5. CV metrics and prediction diagnostics
    poscount = int((sub[target_col] == 1).sum())
    negcount = int((sub[target_col] == 0).sum())
    posrate = float(sub[target_col].mean())
    writejson(version_dir / "cv_metrics.json", {
        "version": version_name,
        "local_cv_metrics": cv_metrics or {},
        "prediction_diagnostics": {"test_rows": len(sub), "predicted_0": negcount,
                                   "predicted_1": poscount, "predicted_1_rate": round(posrate, 4)},
        "kaggle_public_score": None,   # filled in by record_kaggle_score
        "kaggle_private_score": None,
    })

    # 6. Model
    savemodel(model, version_dir)

    # 7. Kaggle description and 8. REPORT.md
    desc = generate_fluent_description(version_name, model_name, len(features), threshold, cv_metrics,
                                       len(sub), posrate, notes)
    (version_dir / "description.txt").write_text(desc + "\n", encoding="utf-8")
    report = reporttext(version_name, model_name, threshold, len(features), notes, desc, len(sub),
                        poscount, negcount, posrate, cv_metrics, subfile.name)
    (version_dir / "REPORT.md").write_text(report, encoding="utf-8")

    update_leaderboard_file(base_path)

    print(f"[SubmissionTracker] Successfully exported bundle -> {version_dir.resolve()}")
    print(f"                   Kaggle file: {subfile.name} | Features: {len(features)} | Test Pos Rate: {posrate:.2%}")
    print(f"\n[Kaggle Description Ready]:\n\"{desc}\"\n")
    return subfile



def cellvalue(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


# Markdown table without extra packages (df.to_markdown needs tabulate)
def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    headers = ["Version"] + list(df.columns)
    rows = [[str(idx)] + [cellvalue(v) for v in row.values] for idx, row in df.iterrows()]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def line(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    sepline = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(headers), sepline] + [line(r) for r in rows])


def list_submission_versions(base_dir: str | Path = "submissions") -> pd.DataFrame:
    base_path = Path(base_dir)
    if not base_path.is_dir():
        return pd.DataFrame()

    records = []
    for vdir in sorted(base_path.iterdir()):
        if not vdir.is_dir() or vdir.name in SKIPDIRS:
            continue
        cfg = readjson(vdir / "config.json")
        met = readjson(vdir / "cv_metrics.json")
        localcv = met.get("local_cv_metrics", {})
        diag = met.get("prediction_diagnostics", {})

        records.append({
            "Version": vdir.name,
            "Model": cfg.get("model_type", "Unknown"),
            "Features": cfg.get("features_count", "-"),
            "Threshold": cfg.get("decision_threshold", 0.5),
            "Test Repeat %": f"{diag['predicted_1_rate'] * 100:.1f}%" if "predicted_1_rate" in diag else "-",
            "Val ROC-AUC": getmetric(localcv, "customer_val_roc_auc", "mean_roc_auc", default="-"),
            "Val F1": getmetric(localcv, "customer_val_f1", "mean_f1", default="-"),
            "Temporal AUC": localcv.get("temporal_val_roc_auc", "-"),
            "Kaggle Score": met.get("kaggle_public_score", "Pending"),
            "Notes": cfg.get("notes", "")[:60],
        })

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).set_index("Version")


def update_leaderboard_file(base_dir: str | Path = "submissions") -> Path:
    base_path = Path(base_dir)
    table = _df_to_markdown(list_submission_versions(base_path))
    lbpath = base_path / "LEADERBOARD.md"
    lbpath.write_text(f"""# Machine Learning Group 30 - Kaggle Leaderboard 

**Last updated:** {timestamp()}

{table}

---
*Tip: Call `record_kaggle_score('version_XX', score)` in Python or edit `cv_metrics.json` directly.*
""", encoding="utf-8")
    return lbpath


# kaggle
def record_kaggle_score(
    version_name: str,
    public_score: float,
    private_score: float | None = None,
    base_dir: str | Path = "submissions",
) -> None:
    base_path = Path(base_dir)
    version_dir = findversiondir(base_path, version_name)

    # 1. cv_metrics.json
    metfile = version_dir / "cv_metrics.json"
    if metfile.is_file():
        met = readjson(metfile)
        met["kaggle_public_score"] = float(public_score)
        if private_score is not None:
            met["kaggle_private_score"] = float(private_score)
        writejson(metfile, met)

    # 2. REPORT.md, an older score section is replaced
    reportfile = version_dir / "REPORT.md"
    if reportfile.is_file():
        text = reportfile.read_text(encoding="utf-8")
        text = text.split(SCOREHEADER)[0].rstrip().removesuffix("---").rstrip() + "\n"
        reportfile.write_text(text + scoresection(public_score, private_score), encoding="utf-8")

    # 3. LEADERBOARD.md
    lbpath = update_leaderboard_file(base_path)
    print(f"[SubmissionTracker] Successfully recorded Kaggle Public Score: {public_score:.4f} for {version_dir.name}!")
    print(f"                   Updated: {reportfile.name} and {lbpath.name}")

