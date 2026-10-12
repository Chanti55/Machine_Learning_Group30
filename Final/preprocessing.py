# Task 2 pre-processing for the repeat purchase data.
# Task 2 explains and tests each step, Task 3 and 4 import it from here.
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer


# number of earlier purchases needed so that the value exists
NEEDED = {"days_since_previous_purchase": 1, "prior_mean_purchase_value": 1,
          "prior_min_purchase_value": 1, "prior_max_purchase_value": 1,
          "prior_std_purchase_value": 2, "prior_mean_gap_days": 2, "prior_std_gap_days": 3}


def repair(df):
    df = df.copy()
    k = df["prior_purchase_count"]
    total = df["prior_total_spend"]

    # -4 days is a code, so it becomes missing
    for col in ["days_since_previous_purchase", "prior_mean_gap_days"]:
        df.loc[df[col] < 0, col] = np.nan

    # the mean can always be recomputed as total / count
    df["prior_mean_purchase_value"] = np.nan
    df.loc[k > 0, "prior_mean_purchase_value"] = total / k

    # a single purchase can not be bigger than the total spend
    df.loc[df["prior_max_purchase_value"] > total + 0.01, "prior_max_purchase_value"] = np.nan

    # spend_365d: placeholder to missing, remove the wrong sign, too big to missing
    df.loc[df["spend_365d"] >= 1e8, "spend_365d"] = np.nan
    df["spend_365d"] = df["spend_365d"].abs()
    df.loc[df["spend_365d"] > total + 0.01, "spend_365d"] = np.nan
    return df


def fillstructural(df):
    df = df.copy()
    k = df["prior_purchase_count"]
    for col, needed in NEEDED.items():
        df.loc[k < needed, col] = 0
    df.loc[k == 0, "spend_365d"] = 0   # no earlier purchase means no spend
    df["is_first_purchase"] = (k == 0).astype(int)
    df["has_gap_history"] = (k >= 2).astype(int)
    return df


def addmonth(df):
    df = df.copy()
    # month as text, so the encoder treats it as a category and not as a number
    df["month"] = pd.to_datetime(df["purchase_date"]).dt.strftime("%b")
    return df


def addfeatures(df):
    df = df.copy()
    k = df["prior_purchase_count"]
    tenure = df["customer_tenure_days"]

    df["purchases_per_year"] = 0.0
    df.loc[tenure > 0, "purchases_per_year"] = k / (tenure / 365)

    meangap = tenure / k.replace(0, np.nan)   # average days per earlier purchase, not the exact gap
    df["typical_gap_le_90"] = (meangap <= 90).astype(int)

    # gross total is never 0 in train, the guard keeps new data with a 0 from dividing by 0
    df["discount_rate"] = ((df["Total Desc_ Linha"] + df["Total Desc_ Global"]) / df["Total Bruto"].replace(0, np.nan)).fillna(0)
    df["zero_cost"] = (df["Custo"] == 0).astype(int)
    df["has_shipping"] = (df["Total Portes"] > 0).astype(int)
    return df


# row rules only, so fit has nothing to learn
class DomainPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return addfeatures(addmonth(fillstructural(repair(X))))


def preprocessor(numeric, flags, categorical):
    # discount_rate is a share between 0 and 1, so no log
    lognumeric = [c for c in numeric if c != "discount_rate"]
    plainnumeric = [c for c in numeric if c == "discount_rate"]
    columns = ColumnTransformer([
        ("log", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(lambda x: np.log1p(np.clip(x, 0, None)), feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]), lognumeric),
        ("share", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), plainnumeric),
        ("flags", "passthrough", flags),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(min_frequency=30, handle_unknown="infrequent_if_exist", sparse_output=False)),
        ]), categorical),
    ], remainder="drop")
    return Pipeline([("domain", DomainPreprocessor()), ("columns", columns)])
