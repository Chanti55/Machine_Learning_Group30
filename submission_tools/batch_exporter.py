
# Demonstrates how to export fully reproducible versions with REPORT.md into submissions/
# Usage: python submission_tools/batch_exporter.py

from pathlib import Path
import sys
import shutil
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.base import clone, BaseEstimator, ClassifierMixin

# Resolve project root
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "Final"))
sys.path.insert(0, str(ROOT_DIR / "Alternative_Version"))
sys.path.insert(0, str(ROOT_DIR / "submission_tools"))

try:
    from preprocessing import preprocessor
except ImportError:
    from Alternative_Version.preprocessing import preprocessor

from submission_tracker import export_submission_bundle, list_submission_versions

# Custom soft blend estimator for probability ensembling
class SoftBlendClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, gb_model, lr_model, rf_model, w_gb=0.45, w_lr=0.35, w_rf=0.20):
        self.gb = gb_model
        self.lr = lr_model
        self.rf = rf_model
        self.w_gb = w_gb
        self.w_lr = w_lr
        self.w_rf = w_rf

    def predict_proba(self, X):
        p1 = (self.w_gb * self.gb.predict_proba(X)[:, 1] +
              self.w_lr * self.lr.predict_proba(X)[:, 1] +
              self.w_rf * self.rf.predict_proba(X)[:, 1])
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def run_batch():
    trainraw = pd.read_csv(ROOT_DIR / "data/train.csv")
    test = pd.read_csv(ROOT_DIR / "data/test.csv")
    sample = pd.read_csv(ROOT_DIR / "data/sample_submission.csv")

    featurecols = [c for c in trainraw.columns if c not in ["ID", "customer_id", "repeat_purchase_90d"]]
    train = trainraw.loc[~trainraw[featurecols].isna().all(axis=1)].reset_index(drop=True)
    X, y = train.drop(columns="repeat_purchase_90d"), train["repeat_purchase_90d"]

    NUMERIC = [
        "Total Liquido", "Total Imp_", "Total Portes",
        "Total Desc_ Global", "Total Desc_ Linha", "Custo",
        "N_ Detalhes", "invoice_count_current_purchase",
        "days_since_previous_purchase", "customer_tenure_days",
        "prior_purchase_count", "prior_total_spend",
        "prior_mean_purchase_value", "prior_std_purchase_value",
        "prior_min_purchase_value", "prior_max_purchase_value",
        "prior_mean_gap_days", "prior_std_gap_days",
        "purchase_count_30d", "spend_30d",
        "purchase_count_90d", "spend_90d",
        "purchase_count_180d", "spend_180d",
        "purchase_count_365d", "spend_365d",
    ]
    CATEGORICAL = ["payment_type", "salesperson", "commercial_zone", "transport_method", "month"]
    flags = ["is_first_purchase", "has_gap_history", "typical_gap_le_90", "zero_cost", "has_shipping"]
    pipeline = preprocessor(NUMERIC + ["purchases_per_year", "discount_rate"], flags, CATEGORICAL)

    X_tr = pipeline.fit_transform(X)
    X_te = pipeline.transform(test)
    names = pipeline.named_steps["columns"].get_feature_names_out()
    n_half = int(np.ceil(len(names) * 0.50))

    # 44 Task 3 Majority Vote features
    l1_m = np.abs(LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=5000, random_state=42).fit(X_tr, y).coef_[0]) > 1e-8
    rfe_m = RFE(LogisticRegression(max_iter=2000, random_state=42), n_features_to_select=n_half, step=5).fit(X_tr, y).support_

    discrete = np.array([name.startswith(("flags__", "cat__")) for name in names])
    X_mi = X_tr.copy()
    for col_idx in np.flatnonzero(discrete):
        _, enc = np.unique(X_mi[:, col_idx], return_inverse=True)
        X_mi[:, col_idx] = enc
    mi_s = mutual_info_classif(X_mi, y.to_numpy(), discrete_features=discrete, random_state=42)
    mi_m = np.zeros(len(names), dtype=bool)
    mi_m[np.argsort(-mi_s)[:n_half]] = True
    vote_m = (mi_m.astype(int) + rfe_m.astype(int) + l1_m.astype(int)) >= 2

    # Fit estimators
    lr = LogisticRegression(C=0.1, max_iter=2000, random_state=42).fit(X_tr[:, vote_m], y)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=42).fit(X_tr[:, vote_m], y)
    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=42).fit(X_tr[:, vote_m], y)

    p_lr_test = lr.predict_proba(X_te[:, vote_m])[:, 1]
    p_rf_test = rf.predict_proba(X_te[:, vote_m])[:, 1]
    p_gb_test = gb.predict_proba(X_te[:, vote_m])[:, 1]
    p_blend_test = (0.45 * p_gb_test + 0.35 * p_lr_test + 0.20 * p_rf_test)

    blend_model = SoftBlendClassifier(gb, lr, rf)

    batch_configs = [
        {
            "version": "version_21",
            "model": lr,
            "probs": p_lr_test,
            "tau": 0.50,
            "cv": {"customer_val_roc_auc": 0.7962, "customer_val_f1": 0.7497, "temporal_val_roc_auc": 0.7931},
            "notes": "Tuned L2 Logistic Regression (C=0.1) on 44 Vote features at standard tau=0.50 (Highest Kaggle Score: 0.7947)",
        },
    ]

    for cfg in batch_configs:
        preds = (cfg["probs"] >= cfg["tau"]).astype(int)
        sub_path = export_submission_bundle(
            version_name=cfg["version"],
            predictions=preds,
            test_df=test,
            sample_df=sample,
            model=cfg["model"],
            feature_mask=vote_m,
            feature_names=names,
            cv_metrics=cfg["cv"],
            threshold=cfg["tau"],
            cleaning_recipe_path=ROOT_DIR / "cleaning_recipe.json",
            notes=cfg["notes"],
            base_dir=ROOT_DIR / "submissions",
        )
        shutil.copy(sub_path, ROOT_DIR / "submissions" / sub_path.name)
        print(f"Exported: {cfg['version']} -> {sub_path.name}")

if __name__ == "__main__":
    run_batch()
