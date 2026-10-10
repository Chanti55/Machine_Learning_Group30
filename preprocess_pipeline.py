"""Machine Learning Group 30 — Data Preprocessing & Pipeline Module

This module provides a leak-free, modular Scikit-Learn preprocessor for the
repeat purchase prediction project (Task 2 -> Task 3 & 4).

Usage:
    from preprocess_pipeline import get_clean_data, build_preprocessor

    # Quick extraction of clean matrices/dataframes:
    X_train_clean, X_test_clean, y_train, feature_names = get_clean_data(train_df, test_df)

    # Or plug the preprocessor into any Scikit-Learn Pipeline:
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    model = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('classifier', LogisticRegression(max_iter=2000))
    ])
    model.fit(X_train, y_train)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer

from course_helpers import CleaningLog


# -- Constants -----------------------------------------------------------------

TARGET = "repeat_purchase_90d"

BASE_NUMERIC = [
    "Total Liquido", "Total Imp_", "Total Portes", "Total Desc_ Global", "Total Desc_ Linha", "Custo",
    "N_ Detalhes", "invoice_count_current_purchase",
    "days_since_previous_purchase", "customer_tenure_days", "prior_purchase_count", "prior_total_spend",
    "prior_mean_purchase_value", "prior_std_purchase_value", "prior_min_purchase_value",
    "prior_max_purchase_value", "prior_mean_gap_days", "prior_std_gap_days",
    "purchase_count_30d", "spend_30d", "purchase_count_90d", "spend_90d",
    "purchase_count_180d", "spend_180d", "purchase_count_365d", "spend_365d"
]

ENGINEERED_NUMERIC = ["purchases_per_year"]
NON_LOG_NUMERIC = ["discount_rate"]
FLAGS = ["is_first_purchase", "has_gap_history", "typical_gap_le_90", "zero_cost", "has_shipping"]
CATEGORICAL = ["payment_type", "salesperson", "commercial_zone", "transport_method", "month"]

# Required prior purchases for history columns to exist
HISTORY_NEEDED = {
    "days_since_previous_purchase": 1,
    "prior_mean_purchase_value": 1,
    "prior_min_purchase_value": 1,
    "prior_max_purchase_value": 1,
    "prior_std_purchase_value": 2,
    "prior_mean_gap_days": 2,
    "prior_std_gap_days": 3,
}

SCALERS = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
    "none": None,
}


# -- Row-level Data Transformations --------------------------------------------

def repair(df: pd.DataFrame) -> pd.DataFrame:
    """Repair software placeholder codes and physical inconsistencies per row."""
    df = df.copy()
    k = df["prior_purchase_count"]
    total = df["prior_total_spend"]

    # Software code -4 represents missing history
    for col in ["days_since_previous_purchase", "prior_mean_gap_days"]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    # Mean purchase value is deterministic: total / count
    df["prior_mean_purchase_value"] = np.nan
    df.loc[k > 0, "prior_mean_purchase_value"] = total / k

    # Physical bound: maximum purchase cannot exceed total spend
    df.loc[df["prior_max_purchase_value"] > total + 0.01, "prior_max_purchase_value"] = np.nan

    # spend_365d: 1e9 placeholder to NaN, negative sign corrected, > total to NaN
    df.loc[df["spend_365d"] >= 1e8, "spend_365d"] = np.nan
    df["spend_365d"] = df["spend_365d"].abs()
    df.loc[df["spend_365d"] > total + 0.01, "spend_365d"] = np.nan
    return df


def fillstructural(df: pd.DataFrame) -> pd.DataFrame:
    """Impute structural missingness for first-time buyers and create indicator flags."""
    df = df.copy()
    k = df["prior_purchase_count"]
    for col, needed in HISTORY_NEEDED.items():
        if col in df.columns:
            df.loc[k < needed, col] = 0.0
    df.loc[k == 0, "spend_365d"] = 0.0
    df["is_first_purchase"] = (k == 0).astype(int)
    df["has_gap_history"] = (k >= 2).astype(int)
    return df


def addmonth(df: pd.DataFrame) -> pd.DataFrame:
    """Extract month string as seasonal categorical feature."""
    df = df.copy()
    df["month"] = pd.to_datetime(df["purchase_date"]).dt.strftime("%b")
    return df


def addfeatures(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer domain interaction ratios and flags strictly from row-level context."""
    df = df.copy()
    k = df["prior_purchase_count"]
    tenure = df["customer_tenure_days"]

    df["purchases_per_year"] = 0.0
    df.loc[tenure > 0, "purchases_per_year"] = k / (tenure / 365)

    meangap = tenure / k.replace(0, np.nan)
    df["typical_gap_le_90"] = (meangap <= 90).astype(int)

    gross = df["Total Bruto"].replace(0, np.nan)
    df["discount_rate"] = ((df["Total Desc_ Linha"] + df["Total Desc_ Global"]) / gross).fillna(0.0)
    df["zero_cost"] = (df["Custo"] == 0).astype(int)
    df["has_shipping"] = (df["Total Portes"] > 0).astype(int)
    return df


# -- Custom Scikit-Learn Transformers -----------------------------------------

class DomainPreprocessor(BaseEstimator, TransformerMixin):
    """Scikit-Learn Transformer for row-level repairs and domain feature engineering.
    
    Stateless transformer: operates identically during fit and transform.
    """

    def fit(self, X: pd.DataFrame, y=None) -> DomainPreprocessor:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = repair(X)
        X = fillstructural(X)
        X = addmonth(X)
        X = addfeatures(X)
        return X


def build_preprocessor(scaler: str = "standard", min_frequency: int = 30) -> Pipeline:
    """Build a complete, leak-free Scikit-Learn Pipeline.
    
    Args:
        scaler: 'standard', 'robust', 'minmax', or 'none'.
        min_frequency: Minimum level frequency for OneHotEncoder grouping.
        
    Returns:
        sklearn.pipeline.Pipeline
    """
    log_numeric = BASE_NUMERIC + ENGINEERED_NUMERIC
    
    # Numeric pipeline with log1p
    log_num_pipe = [
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(lambda x: np.log1p(np.clip(x, 0, None)), feature_names_out="one-to-one")),
    ]
    if SCALERS.get(scaler) is not None:
        log_num_pipe.append(("scaler", SCALERS[scaler]()))

    # Non-log numeric pipeline (e.g. discount_rate, bounded in [0, 1])
    plain_num_pipe = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if SCALERS.get(scaler) is not None:
        plain_num_pipe.append(("scaler", SCALERS[scaler]()))

    # Categorical pipeline with rare-level binning
    cat_pipe = [
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(min_frequency=min_frequency, handle_unknown="infrequent_if_exist", sparse_output=False)),
    ]

    col_transformer = ColumnTransformer(
        transformers=[
            ("log_numeric", Pipeline(log_num_pipe), log_numeric),
            ("plain_numeric", Pipeline(plain_num_pipe), NON_LOG_NUMERIC),
            ("flags", "passthrough", FLAGS),
            ("categorical", Pipeline(cat_pipe), CATEGORICAL),
        ],
        remainder="drop"
    )

    return Pipeline([
        ("domain", DomainPreprocessor()),
        ("columns", col_transformer),
    ])


# -- Cleaning Recipe Logger (Class Toolbox Integration) ------------------------

def build_cleaning_log() -> CleaningLog:
    """Construct a complete CleaningLog following course_helpers.py specifications."""
    log = CleaningLog("everything_and_then_some")

    log.record(
        column="All (empty rows)",
        action="drop_rows",
        reason="65 rows in train.csv have all feature values missing (uninformative); test rows are never dropped",
        rows_affected=65,
        carries={"drop_empty": True}
    )
    log.record(
        column="days_since_previous_purchase & prior_mean_gap_days",
        action="blank_to_nan",
        reason="Software sentinel code of -4 days represents unrecorded history, not genuine negative time",
        rows_affected=648,
        carries={"sentinel_threshold": 0}
    )
    log.record(
        column="prior_mean_purchase_value",
        action="recalculate",
        reason="Deterministic arithmetic quantity: recomputed as prior_total_spend / prior_purchase_count",
        rows_affected=6469,
        carries={"recompute_formula": "prior_total_spend / prior_purchase_count"}
    )
    log.record(
        column="prior_max_purchase_value",
        action="blank_to_nan",
        reason="Single purchase amount cannot exceed prior total spend (physical impossibility)",
        rows_affected=7,
        carries={"bound_column": "prior_total_spend"}
    )
    log.record(
        column="spend_365d",
        action="repair_sign_and_placeholders",
        reason="Values >= 1e8 are placeholders (set to NaN); negative amounts are sign errors (.abs()); > total spend set to NaN",
        rows_affected=23,
        carries={"placeholder_bound": 1e8}
    )
    log.record(
        column="prior_total_spend (extremes)",
        action="retain_and_log1p",
        reason="642 IQR outliers represent legitimate loyal high-value customers (84.1% repeat rate); variance compressed with log1p instead of row deletion",
        rows_affected=642,
        carries={"treatment": "log1p", "drop_outliers": False}
    )
    log.record(
        column="Structural gap features (days_since_previous_purchase, std, etc.)",
        action="structural_fill_zero_and_flag",
        reason="First-time buyers (k=0) have undefined purchase history by definition; fill with 0 and add is_first_purchase and has_gap_history flags",
        rows_affected=1413,
        carries={"structural_fill": 0.0, "flags": ["is_first_purchase", "has_gap_history"]}
    )
    log.record(
        column="warehouse & document_series",
        action="drop_columns",
        reason="warehouse is 99.7% constant (near-zero variance); document_series reflects historical invoice book numbering with no predictive signal",
        rows_affected=6469,
        carries={"dropped": ["warehouse", "document_series"]}
    )
    log.record(
        column="purchase_date",
        action="extract_month_and_drop_year",
        reason="Month captures strong, stable seasonality (r=0.81 across epochs); year is dropped to prevent distribution drift on unseen future test years",
        rows_affected=6469,
        carries={"extract": "month", "dropped": "year"}
    )
    log.record(
        column="clientes.csv",
        action="discard_auxiliary_table",
        reason="Empirical ablation across 5 customer-grouped CV folds showed no statistical gain (+0.0003 for LogReg, -0.0019 for Tree)",
        rows_affected=6469,
        carries={"use_clientes": False}
    )
    log.record(
        column="Engineered Interaction Features",
        action="add_domain_ratios",
        reason="Added purchases_per_year, typical_gap_le_90, discount_rate, zero_cost, has_shipping",
        rows_affected=6469,
        carries={"new_features": ["purchases_per_year", "typical_gap_le_90", "discount_rate", "zero_cost", "has_shipping"]}
    )
    log.record(
        column="Numeric columns (skewed amounts and counts)",
        action="log1p_and_standard_scaler",
        reason="log1p brings median absolute skew down from 3.7 to 0.99; StandardScaler ensures convergence for regularized linear models",
        rows_affected=6469,
        carries={"transform": "log1p", "scaler": "StandardScaler"}
    )
    return log


def save_cleaning_recipe(path: str | Path = "cleaning_recipe.json") -> Path:
    """Save the cleaning log as an audit recipe JSON file."""
    log = build_cleaning_log()
    target = Path(path)
    log.to_json(target)
    return target


# -- High-level Data Preparation Helper ---------------------------------------

def get_clean_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    scaler: str = "standard",
    min_frequency: int = 30,
    return_df: bool = True
) -> tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray | None, pd.Series, list[str]]:
    """Clean and preprocess train and test data with complete leakage prevention.
    
    Args:
        train_df: Raw train DataFrame (train.csv).
        test_df: Raw test DataFrame (test.csv) or None.
        scaler: 'standard', 'robust', 'minmax', or 'none'.
        min_frequency: Minimum level frequency for OneHotEncoder.
        return_df: If True, returns DataFrames with column names. If False, returns NumPy arrays.
        
    Returns:
        (X_train_clean, X_test_clean, y_train, feature_names)
    """
    train_df = train_df.copy()
    
    # 1. Filter empty rows from train
    feature_cols = [c for c in train_df.columns if c not in ["ID", "customer_id", TARGET]]
    empty = train_df[feature_cols].isna().sum(axis=1) == len(feature_cols)
    train_clean = train_df[~empty].reset_index(drop=True)

    y_train = train_clean[TARGET] if TARGET in train_clean.columns else None
    X_train = train_clean.drop(columns=[TARGET]) if TARGET in train_clean.columns else train_clean

    # 2. Build and fit pipeline strictly on train
    pipeline = build_preprocessor(scaler=scaler, min_frequency=min_frequency)
    X_train_arr = pipeline.fit_transform(X_train)

    # 3. Extract named features
    col_trans = pipeline.named_steps["columns"]
    feature_names = [f.split("__")[-1] for f in col_trans.get_feature_names_out()]

    X_test_arr = None
    if test_df is not None:
        test_clean = test_df.copy()
        if TARGET in test_clean.columns:
            test_clean = test_clean.drop(columns=[TARGET])
        X_test_arr = pipeline.transform(test_clean)

    if return_df:
        X_train_out = pd.DataFrame(X_train_arr, columns=feature_names, index=X_train.index)
        X_test_out = pd.DataFrame(X_test_arr, columns=feature_names, index=test_df.index) if test_df is not None else None
    else:
        X_train_out = X_train_arr
        X_test_out = X_test_arr

    return X_train_out, X_test_out, y_train, feature_names


if __name__ == "__main__":
    # Self-test when executed directly
    print("Testing preprocess_pipeline.py...")
    train_raw = pd.read_csv("data/train.csv")
    test_raw = pd.read_csv("data/test.csv")
    
    X_tr, X_te, y_tr, f_names = get_clean_data(train_raw, test_raw, return_df=True)
    print(f"X_train shape: {X_tr.shape}, X_test shape: {X_te.shape}, Features: {len(f_names)}")
    assert X_tr.shape[1] == len(f_names), "Feature names count mismatch!"
    assert not X_tr.isna().any().any(), "NaN values found in X_train!"
    assert not X_te.isna().any().any(), "NaN values found in X_test!"

    recipe_path = save_cleaning_recipe("cleaning_recipe.json")
    print(f"Cleaning recipe saved successfully to {recipe_path}!")
    print("All tests passed successfully!")
