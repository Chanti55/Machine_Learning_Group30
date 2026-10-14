# Comprehensive Analysis & Class Alignment: Task 2 (v3)

**Project:** Machine Learning Group 30 — Repeat Purchase Prediction (90 Days)  
**File Reviewed:** [`task2_v3.ipynb`](file:///C:/Users/tiago/OneDrive/Ambiente%20de%20Trabalho/NOVA-IMS/MachineLearning/Machine_Learning_Group30/task2_v3.ipynb)  
**Target Syllabus Deliverable:** Task 2 — *Clean and pre-process the data* (5 Points)

---

## 1. Executive Summary

`task2_v3.ipynb` implements a complete, rigorous data cleaning and preprocessing pipeline for the repeat purchase prediction problem (`repeat_purchase_90d`). The notebook takes raw transaction data (`train.csv`) and the optional auxiliary table (`clientes.csv`), diagnoses anomalies, handles structural and random missingness, engineers domain features, applies distribution-normalizing transformations, and verifies every step using **group-aware cross-validation** and **temporal out-of-time splits**.

The implementation is **exceptionally well-aligned with the methodologies taught in class** (Weeks 1 through 5), following core principles such as:
1. Strict avoidance of data leakage (transformers fit only on training folds).
2. Distinguishing physical/domain anomalies from legitimate statistical extremes.
3. Grouped customer validation (`GroupKFold` philosophy) to prevent repeated-measurement memorization.
4. Empirical ablation studies comparing progressive pipeline variants ($A \to B \to C \to D \to E$).

---

## 2. Detailed Walkthrough of `task2_v3.ipynb`

### Section 1: Setup & Import
- **Libraries**: Ingests standard scientific computing (`numpy`, `pandas`), visualization (`matplotlib`, `seaborn`), and scikit-learn transformers/models (`SimpleImputer`, `OneHotEncoder`, `StandardScaler`, `RobustScaler`, `MinMaxScaler`, `LogisticRegression`, `DecisionTreeClassifier`, `roc_auc_score`).
- **Reproducibility**: Sets a fixed `RANDOMSTATE = 42`.
- **Target Specification**: Explicitly sets `TARGET = "repeat_purchase_90d"`.

---

### Section 2: Missing Values & Implausible Observations

#### 2.1 Empty Rows & Row Drops
- **Finding**: Identifies **65 completely empty rows** in `train.csv` (all feature values are `NaN`).
- **Decision**: Drops these 65 rows from `train.csv` (leaves 6,469 rows).
- **Leakage Prevention**: Clarifies that rows in `test.csv` will **never be dropped**, as every submission ID requires a prediction. No entity duplicates were present outside of repeated customer IDs.

#### 2.2 Domain Rule Repairs (`repair(df)`)
Rather than blindly dropping rows or running generic imputers, a row-wise repair function fixes data recording defects:
1. **Sentinel Day Codes (`-4`)**: Negative day differences (`days_since_previous_purchase < 0` and `prior_mean_gap_days < 0`) represent software error codes rather than real time. They are blanked to `np.nan`.
2. **Mean Recomputation**: `prior_mean_purchase_value` is recomputed deterministically as `prior_total_spend / prior_purchase_count` when count $> 0$.
3. **Physical Impossibility Guard**: Single purchase maxima exceeding total spend (`prior_max_purchase_value > prior_total_spend + 0.01`) are set to `np.nan`.
4. **Placeholder `1e9` & Sign Errors**: In `spend_365d`, values $\ge 10^8$ are placeholders set to `np.nan`. Negative amounts are corrected with `.abs()` (the notebook proves that the absolute value aligns with earlier window totals), and amounts exceeding total spend are set to `np.nan`.
5. **Audit Table (`countproblems(df)`)**: Verifies that 100% of these recording issues are eliminated after repair.

#### 2.3 Extreme Values & Why $1.5 \times \text{IQR}$ Outlier Trimming Fails
- **The Boxplot Dilemma**: Applying the standard $1.5 \times \text{IQR}$ fence to `prior_total_spend` flags 642 rows (9.9% of the dataset) as outliers.
- **Critical Domain Insight**:
  - Customers in the outlier bracket have an **84.1% repeat purchase rate**, compared to **52.9%** for customers below the fence.
  - Deleting these rows would discard the most lucrative and loyal customers.
- **Decision**: Keep 100% of outlier rows and stabilize their variance using $\ln(1+x)$ (`log1p`) in Section 6.
- **Zero Cost Flags**: 486 rows have `Custo == 0` (repeat rate 67.7% vs 56.0% overall). These represent promotions or zero-cost lines; rather than treating them as errors, a binary flag `zero_cost` is engineered.

#### 2.4 Structural vs. Random Missingness
- **Structural Missingness (`fillstructural(df)`)**:
  - Columns such as `days_since_previous_purchase`, `prior_mean_gap_days`, and `prior_std_gap_days` require a minimum number of prior purchases ($k \ge 1, 2, 3$).
  - For first-time buyers ($k=0$), these columns are mathematically undefined, not randomly missing.
  - Imputes these structural gaps to `0` and creates two explicit indicators:
    - `is_first_purchase = (prior_purchase_count == 0).astype(int)`
    - `has_gap_history = (prior_purchase_count >= 2).astype(int)`
- **Random Missingness (`fillmissing()`)**:
  - Remaining numerical gaps (45 to 231 per column) are imputed using the **training set median**.
  - Categorical gaps receive an explicit `"missing"` category level.

#### 2.5 Formats & Consistency
- Verifies that all purchase dates parse cleanly to timestamps (`pd.to_datetime`).
- Verifies categorical string columns (`payment_type`, `salesperson`, `transport_method`, `warehouse`, `commercial_zone`) for casing or leading/trailing whitespace discrepancies.

---

### Section 3: Categorical Variables

- **One-Hot Encoding with Grouped Rare Levels**:
  - Uses `OneHotEncoder(min_frequency=30, handle_unknown="infrequent_if_exist", sparse_output=False)`.
  - Categories with fewer than 30 occurrences are collapsed into a single `infrequent` column. This compresses `transport_method` from 80 distinct levels down to 31 columns, avoiding the curse of dimensionality.
- **Uninformative Variables Dropped**:
  - `warehouse`: 99.7% constant (single level). Dropped due to near-zero variance.
  - `document_series`: Time-bound series with no predictive connection to customer loyalty. Dropped.
- **Temporal Feature: Seasonal Month vs. Year Drift**:
  - Discovers that repeat purchase rates vary from **0.44 in March** to **0.72 in May**.
  - Confirms seasonality is stable: the monthly pattern in 2015–2019 correlates at **$r = 0.81$** with 2020–2025.
  - Extracts `month` (e.g., `'Jan'`, `'Feb'`) as a one-hot categorical feature.
  - **Drops `year`**: Testing data from future periods contains unseen years; training on year would induce distribution drift.

---

### Section 4: Optional Auxiliary Table (`clientes.csv`)

- Explores merging customer demographic data via `customer_id`.
- Most columns in `clientes.csv` are redundant IDs or sparse text.
- Extracts two candidate features:
  1. `country_or_region`
  2. `island_known` (`1` if `island` is non-null, else `0`)
- Integrates via left-merge (`joinclientes()`), ensuring no purchase rows are dropped.
- **Empirical Validation (Section 7)** proves that adding `clientes.csv` improves ROC AUC by only $+0.0003$ on Logistic Regression and **degrades** Decision Tree performance by $-0.0019$. It is rightfully discarded.

---

### Section 5: Supplied Variables & Feature Engineering

#### 5.1 Supplied Variable Filtering
- Dropped: `ID` (submission identifier only), `customer_id` (used only for group splits), `random_noise` (synthetic noise column).
- Dropped: `Total a Pagar`, `Total Bruto` (perfect linear combinations of `Total Liquido`, discounts, and taxes).

#### 5.2 Engineered Domain Interaction Ratios
Five new features derived strictly from same-row observations:
1. `purchases_per_year`: `prior_purchase_count / (customer_tenure_days / 365)` ($\text{AUC} = 0.744$).
2. `typical_gap_le_90`: Indicator whether average tenure gap $\le 90$ days ($\text{AUC} = 0.692$).
3. `discount_rate`: `(Total Desc_ Linha + Total Desc_ Global) / Total Bruto` ($\text{AUC} = 0.624$).
4. `zero_cost`: Indicator whether `Custo == 0` ($\text{AUC} = 0.548$).
5. `has_shipping`: Indicator whether `Total Portes > 0` ($\text{AUC} = 0.544$).

---

### Section 6: Scaling & Transformations

- **`np.log1p` on Right-Skewed Quantities**:
  - Applied to 26 financial and count features (`Total Liquido`, `prior_total_spend`, `customer_tenure_days`, etc.).
  - Reduces median absolute skewness from **3.7 to 0.99**.
- **Scaler Benchmarking**:
  - Compares `StandardScaler`, `RobustScaler`, `MinMaxScaler`, and `None`.
  - Confirms that scaling and `log1p` are vital for gradient/regularized linear solvers (`lbfgs` reaches `max_iter=2000` without them).
  - Flags and one-hot binary columns are excluded from scaling to preserve boolean interpretability.

---

### Section 7: Master Preprocessing Function & Experimental Validation

#### 7.1 Leak-Free Pipeline Architecture (`foldmatrices`)
Encapsulates all logic into a unified function:
- Row-wise rules (`repair`, `fillstructural`, `addmonth`, `addfeatures`) execute independently per row.
- State-dependent transformers (`SimpleImputer`, `OneHotEncoder`, `StandardScaler`) **fit strictly on `train`** and transform `test`.

#### 7.2 Group-Aware Cross-Validation (`GroupKFold` Strategy)
- Customers make repeated purchases. Random row splitting would place purchases from the same customer into both training and validation folds, causing severe **target and identity leakage**.
- Implements a 5-fold partition grouping by `customer_id`.

#### 7.3 Progressive Variant Tournament
Evaluates models using ROC AUC across the 5 customer folds:

| Pipeline Variant | Logistic Regression AUC | Decision Tree AUC | Note |
|---|---|---|---|
| **A: Naive Baseline** | 0.8267 | 0.7725 | Raw median impute, uncleaned 1e9, unpruned OHE |
| **B: Cleaned** | 0.8335 (+0.0068) | 0.7743 (+0.0018) | Repaired, structural fill, log1p |
| **C: B + Month** | 0.8403 (+0.0068) | 0.7783 (+0.0040) | Seasonal month indicator added |
| **D: C + New Features** | 0.8409 (+0.0006) | 0.7773 (-0.0010) | Interaction ratios added |
| **E: D + Clientes** | 0.8412 (+0.0003) | 0.7754 (-0.0019) | Clientes table added |

- **Ablation Study**: Removing `repair`, `fillstructural`, or `log1p` from Variant B causes measurable performance drops across folds, justifying every single cleaning component.

#### 7.4 Temporal Generalization Audit
- Evaluates train years $< Y$ vs test year $Y$ (2018 through 2024).
- Confirms that the seasonal month feature provides consistent out-of-time gains (+0.0084 average gain across years).

---

## 3. Alignment Audit with Class Material

| Topic / Lecture Area | Class Concept (`classWork` Solutions & Summaries) | How `task2_v3.ipynb` Aligns | Compliance |
|---|---|---|:---:|
| **Week 1: Supervised ML Foundations** | - Define prediction moment & operational decision.<br>- Class imbalance awareness.<br>- Establish naive baseline. | - Predicts `repeat_purchase_90d` using prior history strictly up to purchase date.<br>- Measures repeat rate (56.0%).<br>- Compares all pipelines against a naive Variant A. | **Full** |
| **Week 2: 8-Stage ML Lifecycle** | - Sequential progression (EDA $\to$ Clean $\to$ Split $\to$ Model $\to$ Assess).<br>- The Golden Rule: validation data simulates the unseen future. | - Preprocessing parameters (medians, one-hot categories, scaler means) fit on train folds only. | **Full** |
| **Week 3: Data Quality & Anomalies** | - Domain rules vs statistical outliers.<br>- $1.5 \times \text{IQR}$ boxplot failure on skewed data.<br>- `log1p` over trimming.<br>- Structural vs random missingness.<br>- Entity duplicates and grouped structure. | - Identifies sentinel codes (`-4`), placeholders (`1e9`).<br>- Retains high-spend customers (84% repeat rate) and transforms with `log1p`.<br>- Structural fill for zero prior purchases.<br>- Groups validation splits by `customer_id`. | **Full** |
| **Week 4: Feature Work & Encoding** | - One-hot encoding with infrequent binning.<br>- Domain interaction ratios.<br>- Removing collinear / redundant columns. | - `OneHotEncoder(min_frequency=30, handle_unknown="infrequent_if_exist")`.<br>- Interaction ratios (`purchases_per_year`, `discount_rate`).<br>- Drops `Total a Pagar`, `Total Bruto`, `random_noise`. | **Full** |
| **Week 5: Validation & Assessment** | - `GroupKFold` for repeated subject observations.<br>- Temporal validation (`TimeSeriesSplit` logic).<br>- Metric: ROC AUC for probability ranking.<br>- Solver convergence on scaled features. | - Implements customer-grouped 5-fold CV.<br>- Implements expanding rolling-year temporal CV.<br>- Evaluates with ROC AUC.<br>- Solves convergence with `log1p` + `StandardScaler`. | **Full** |

---

## 4. Strengths & High-Value Recommendations

### Key Strengths of `task2_v3.ipynb`
1. **Flawless Leakage Prevention**: Transformations that learn parameters (`SimpleImputer`, `OneHotEncoder`, `StandardScaler`) are strictly isolated inside the training folds.
2. **Empirically Driven Decisions**: Every single design choice (whether to drop outliers, whether to keep `clientes.csv`, whether to keep `month`) is backed by cross-validation scores rather than guesswork.
3. **Domain Rigor**: Correctly recognizes that missing values in purchase gaps are structural characteristics of first-time buyers rather than missing data.

---

### What to Add / Recommendations for Improvement

#### 1. Encapsulate in a Formal Scikit-Learn `Pipeline` / `ColumnTransformer` (High Priority)
* **Current status**: `foldmatrices()` manually implements the transform logic with NumPy array slicing and concatenation.
* **Why improve**: In Task 3 (Feature Selection via RFE, Sequential Feature Selection) and Task 4 (Hyperparameter Tuning via `GridSearchCV`), Scikit-Learn tools expect a standard `Pipeline` or `ColumnTransformer` object implementing `.fit()` and `.transform()`.
* **Recommendation**: Create a custom transformer class or export a `preprocessor = ColumnTransformer(...)` wrapper. This will make Task 3 and Task 4 modular and concise.

#### 2. Export / Save the Preprocessing Recipe (Course Best Practice)
* **Class reference**: In Week 3 (`classWork/Solutions/week_03/course_helpers.py`), the class teaches creating a `CleaningLog` and saving the cleaning recipe to JSON (`recipe.json`).
* **Recommendation**: Save the final cleaning parameters or export a standalone module (`preprocess_pipeline.py`) so Task 3 and Task 4 can import the clean data pipeline in one line:
  ```python
  from preprocess_pipeline import get_clean_data
  X_train_clean, X_test_clean = get_clean_data(train_df, test_df)
  ```

#### 3. Deepen Domain Feature Interactions (Optional for Task 3/4)
* The 5 engineered features are strong. Consider adding 2 additional high-signal retail indicators:
  - **Recency Ratio**: `days_since_previous_purchase / (prior_mean_gap_days + 1)` (measures whether the customer is overdue relative to their normal purchasing cadence).
  - **Basket Depth**: `Total Liquido / (N_ Detalhes + 1)` (average spend per invoice line item).

#### 4. Document Why Feature Variant D Shows Slight Tree Degradation
* In Section 7.2, Variant D (new features) gives $+0.0006$ on Logistic Regression but $-0.0010$ on Decision Tree.
* Adding a 1-sentence markdown note explaining that tree models with fixed depth (`max_depth=5`) experience split competition when correlated ratio features are added will demonstrate deep theoretical understanding to the graders.

---

## 5. Summary Checklist for Deliverables

- [x] **Identify and handle missing values**: Addressed (empty rows dropped, structural zeros + flags, median/mode for random gaps).
- [x] **Identify and handle anomalous/implausible values**: Addressed (`-4` sentinels, `1e9` placeholders, negative amounts, mean recalculation).
- [x] **Deal with categorical variables**: Addressed (`OneHotEncoder` with `min_frequency=30`, constant columns dropped, seasonal month extracted).
- [x] **Evaluate `clientes.csv`**: Addressed (joined, benchmarked, proven unhelpful, justified exclusion).
- [x] **Create additional features**: Addressed (5 domain ratios/flags constructed and evaluated).
- [x] **Apply scaling and justify choices**: Addressed (`log1p` + `StandardScaler` compared against alternatives).
- [x] **Reproducible and zero test leakage**: Addressed (strictly isolated within training splits).
