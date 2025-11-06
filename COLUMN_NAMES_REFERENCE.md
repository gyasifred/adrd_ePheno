# ADRD ePhenotyping Pipeline - Column Names Reference

**Author**: Gyasi, Frederick
**For**: Dr. Paul
**Date**: 2025-11-06
**Purpose**: Comprehensive reference for all column names used throughout the pipeline

---

## Table of Contents

1. [Input Data Columns](#input-data-columns)
2. [Processed Data Columns](#processed-data-columns)
3. [Model Predictions Columns](#model-predictions-columns)
4. [Demographic Variables](#demographic-variables)
5. [Evaluation Metrics Columns](#evaluation-metrics-columns)
6. [Analysis Results Columns](#analysis-results-columns)

---

## Input Data Columns

### Raw CSV File: `data/raw/ptHx_sample_v2025-10-25.csv`

| Column Name | Data Type | Description | Example Values | Required |
|-------------|-----------|-------------|----------------|----------|
| `DE_ID` | String/Integer | De-identified patient unique identifier | "12345", "67890" | ✅ Yes |
| `content` | String (long text) | Raw clinical notes/text | "Patient presents with..." | ✅ Yes |
| `Label` | String | Original label as string | "ADRD", "NON-ADRD" | ✅ Yes |
| `GENDER` | String | Patient gender | "Male", "Female", "Other" | ⚠️ Optional |
| `RACE` | String | Patient race (verbatim from EHR) | "WHITE OR CAUCASIAN", "BLACK OR AFRICAN AMERICAN" | ⚠️ Optional |
| `HISPANIC` | String | Hispanic ethnicity indicator | "NO, NOT HISPANIC OR LATINO", "YES, HISPANIC OR LATINO" | ⚠️ Optional |
| `AGE` | Integer | Patient age (if available) | 65, 78, 82 | ⚠️ Optional |
| `INSURANCE` | String | Insurance type/financial class (if available) | "Medicare", "Medicaid", "Private" | ⚠️ Optional |

**Notes**:
- **Required columns**: `DE_ID`, `content`, `Label` must be present
- **Demographic columns**: Optional but needed for Aim 1 analysis
- **Text preprocessing**: The `content` column should contain preprocessed clinical notes

---

## Processed Data Columns

### After `01_prepare_data.R`: `data/train_set.rds` and `data/test_set.rds`

| Column Name | Data Type | Description | Transformation | Example Values |
|-------------|-----------|-------------|----------------|----------------|
| `DE_ID` | String/Integer | Patient identifier | (unchanged) | "12345" |
| **`txt`** | String (long text) | **Renamed from `content`** | `content` → `txt` | "Patient presents..." |
| `Label` | String | Original string label | (unchanged) | "ADRD", "NON-ADRD" |
| **`label`** | Integer | **Numeric label (NEW)** | `Label` converted to 0/1 | 0 (NON-ADRD), 1 (ADRD) |
| `GENDER` | String | Gender | (unchanged if present) | "Male", "Female" |
| `RACE` | String | Race | (unchanged if present) | "WHITE OR CAUCASIAN" |
| `HISPANIC` | String | Ethnicity | (unchanged if present) | "NO, NOT HISPANIC OR LATINO" |
| `AGE` | Integer | Age | (unchanged if present) | 75 |

**Key Transformations**:
1. **`content` → `txt`**: Renamed for compatibility with training scripts
2. **`Label` → `label`**: New numeric column created:
   - `"ADRD"` → `1` (positive class)
   - `"NON-ADRD"` → `0` (negative class)
3. **Both `Label` and `label` retained**: For traceability

### Split Information: `data/split_info.rds`

| Column Name | Data Type | Description | Example Values |
|-------------|-----------|-------------|----------------|
| `DE_ID` | String/Integer | Patient identifier | "12345" |
| `Label` | String | Original string label | "ADRD" |
| `label` | Integer | Numeric label | 0, 1 |
| `GENDER` | String | Gender (if available) | "Male" |
| `RACE` | String | Race (if available) | "WHITE OR CAUCASIAN" |
| `HISPANIC` | String | Ethnicity (if available) | "NO, NOT HISPANIC OR LATINO" |
| **`partition`** | String | Train/test assignment | **"train"**, **"test"** |

**Purpose**: Tracks which patients are in training vs test sets

---

## Model Predictions Columns

### After `03_evaluate_models.R`: `results/predictions_df.csv`

| Column Name | Data Type | Description | Range/Values | Notes |
|-------------|-----------|-------------|--------------|-------|
| `DE_ID` | String/Integer | Patient identifier | - | Links to original data |
| **`Label`** | Integer | True label (numeric) | 0, 1 | **Note**: Renamed from test set |
| `label_icd` | Integer | Alternative label column | 0, 1 | Same as `Label` |
| **`True_Class`** | String | True label as string | "ADRD", "CTRL" | Derived from `Label` |
| **`Predicted_Probability`** | Float | Model's predicted probability of ADRD | 0.0 to 1.0 | From best CNN model |
| **`Predicted_Class`** | String | Predicted class at threshold 0.5 | "ADRD", "CTRL" | Thresholded prediction |
| **`Correct`** | Boolean | Whether prediction was correct | TRUE, FALSE | TRUE if prediction matches truth |
| **`Risk_Category`** | String | Risk level based on probability | "Low", "Moderate", "High" | Binned probability |
| `GENDER` | String | Gender (if available) | "Male", "Female" | From test set |
| `RACE` | String | Race (if available) | "WHITE OR CAUCASIAN", etc. | From test set |
| `HISPANIC` | String | Ethnicity (if available) | "NO, NOT HISPANIC OR LATINO", etc. | From test set |

**Risk Category Bins**:
- **Low**: Predicted_Probability < 0.30
- **Moderate**: 0.30 ≤ Predicted_Probability < 0.70
- **High**: Predicted_Probability ≥ 0.70

**Important Notes**:
- **`Label`** in predictions file is **numeric (0/1)**, not string
- **`True_Class`** is the string representation ("ADRD" or "CTRL", not "NON-ADRD")
- **Demographics included only if present** in original test set

---

## Demographic Variables

### Gender Values

| Raw Value (EHR) | Simplified Value | Count (typical) | Notes |
|-----------------|------------------|-----------------|-------|
| `"Male"` | `"Male"` | ~40-50% | - |
| `"Female"` | `"Female"` | ~50-60% | - |
| `"Other"` | `"Other"` | <1% | Rare |
| `"Unknown"` | `"Unknown"` | <5% | Missing/refused |
| `""` (empty) | `"Unknown"` | Variable | Excluded from analysis |

### Race Values

| Raw Value (EHR) | Simplified Value | Typical % | Notes |
|-----------------|------------------|-----------|-------|
| `"WHITE OR CAUCASIAN"` | `"White"` | 60-80% | Most common |
| `"BLACK OR AFRICAN AMERICAN"` | `"Black"` | 15-30% | - |
| `"OTHER ASIAN"` | `"Asian"` | 2-5% | - |
| `"AMERICAN INDIAN OR ALASKA NATIVE"` | `"Am. Indian/AK Native"` | <1% | Very rare |
| `"NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER"` | `"Pacific Islander"` | <1% | Very rare |
| `"OTHER"` | `"Other"` | 5-10% | Heterogeneous |
| `"UNKNOWN"` | `"Unknown"` | <5% | Excluded from analysis |
| `"PATIENT REFUSED"` | `"Refused"` | <2% | Excluded from analysis |

**Simplification Logic** (in `04_demographic_analysis.R`):
- Applied via `simplify_category_name()` function
- Only for display in plots and reports
- **Raw values preserved in data files**

### Hispanic/Ethnicity Values

| Raw Value (EHR) | Simplified Value | Typical % | Notes |
|-----------------|------------------|-----------|-------|
| `"NO, NOT HISPANIC OR LATINO"` | `"Non-Hispanic"` | 85-95% | Majority |
| `"YES, HISPANIC OR LATINO"` | `"Hispanic"` | 5-15% | - |
| `"YES, ANOTHER HISPANIC OR LATINO"` | `"Hispanic"` | <5% | Collapsed with above |
| `"UNKNOWN"` | `"Unknown"` | <5% | Excluded from analysis |
| `"PATIENT REFUSED"` | `"Refused"` | <2% | Excluded from analysis |

### Age Values (if available)

| Variable | Data Type | Range | Description |
|----------|-----------|-------|-------------|
| `AGE` | Integer | 18-120 | Age in years at time of note |

**Typical Bins for Analysis**:
- **Young-old**: 65-74
- **Middle-old**: 75-84
- **Old-old**: 85+

---

## Evaluation Metrics Columns

### Model Performance Metrics: `results/evaluation_summary.csv`

| Column Name | Data Type | Description | Range | Interpretation |
|-------------|-----------|-------------|-------|----------------|
| `model` | String | Model name | "CNNr" | Random CNN |
| `cycle` | Integer | Training cycle number | 1-10 | Which of 10 training runs |
| **Confusion Matrix** |||||
| `tp` | Integer | True positives | ≥0 | ADRD correctly identified |
| `tn` | Integer | True negatives | ≥0 | CTRL correctly identified |
| `fp` | Integer | False positives | ≥0 | CTRL incorrectly called ADRD |
| `fn` | Integer | False negatives | ≥0 | ADRD incorrectly called CTRL |
| **Classification Metrics** |||||
| `auc` | Float | Area under ROC curve | 0.0-1.0 | Overall discriminative ability |
| `auc_ci_lower` | Float | AUC 95% CI lower bound | 0.0-1.0 | Confidence interval |
| `auc_ci_upper` | Float | AUC 95% CI upper bound | 0.0-1.0 | Confidence interval |
| `accuracy` | Float | Overall accuracy | 0.0-1.0 | (TP+TN)/(TP+TN+FP+FN) |
| `sensitivity` | Float | Sensitivity (recall) | 0.0-1.0 | TP/(TP+FN) - ADRD detection rate |
| `specificity` | Float | Specificity | 0.0-1.0 | TN/(TN+FP) - CTRL identification rate |
| `precision` | Float | Precision (PPV) | 0.0-1.0 | TP/(TP+FP) - ADRD prediction accuracy |
| `ppv` | Float | Positive predictive value | 0.0-1.0 | Same as precision |
| `npv` | Float | Negative predictive value | 0.0-1.0 | TN/(TN+FN) |
| `f1` | Float | F1 score | 0.0-1.0 | Harmonic mean of precision & recall |
| `f2` | Float | F2 score | 0.0-1.0 | Weighted toward recall |
| `fpr` | Float | False positive rate | 0.0-1.0 | FP/(FP+TN) = 1 - specificity |
| `fnr` | Float | False negative rate | 0.0-1.0 | FN/(FN+TP) = 1 - sensitivity |
| `mcc` | Float | Matthews correlation coefficient | -1.0 to 1.0 | Balanced measure |
| **Probability Metrics** |||||
| `brier_score` | Float | Brier score | 0.0-1.0 | Calibration metric (lower = better) |
| `log_loss` | Float | Log loss | ≥0 | Cross-entropy loss (lower = better) |
| **Optimal Threshold Metrics** |||||
| `youden_threshold` | Float | Youden's optimal threshold | 0.0-1.0 | Threshold maximizing sensitivity + specificity |
| `youden_index` | Float | Youden's index | 0.0-2.0 | Sensitivity + specificity - 1 |
| `youden_sensitivity` | Float | Sensitivity at optimal threshold | 0.0-1.0 | - |
| `youden_specificity` | Float | Specificity at optimal threshold | 0.0-1.0 | - |
| **Configuration** |||||
| `threshold_used` | Float | Classification threshold | 0.0-1.0 | Typically 0.5 |

### Metric Interpretation Guidelines

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| **AUC** | ≥0.90 | 0.80-0.89 | 0.70-0.79 | <0.70 |
| **Sensitivity** | ≥0.85 | 0.75-0.84 | 0.65-0.74 | <0.65 |
| **Specificity** | ≥0.85 | 0.75-0.84 | 0.65-0.74 | <0.65 |
| **F1 Score** | ≥0.85 | 0.75-0.84 | 0.65-0.74 | <0.65 |
| **Brier Score** | <0.10 | 0.10-0.15 | 0.15-0.20 | >0.20 |

---

## Analysis Results Columns

### Demographic Subgroup Analysis: `results/demographic/subgroup_performance.csv`

| Column Name | Data Type | Description | Example Values |
|-------------|-----------|-------------|----------------|
| `subgroup` | String | Demographic dimension | "Overall", "Gender", "Race", "Ethnicity" |
| `category` | String | Specific category within dimension | "Male", "Female", "White", "Black" |
| `category_short` | String | Simplified category name | "White" (from "WHITE OR CAUCASIAN") |
| **Sample Sizes** ||||
| `n` | Integer | Total samples in subgroup | 50, 100, 500 |
| `n_pos` | Integer | Number of ADRD cases | 10, 25, 125 |
| `n_neg` | Integer | Number of control cases | 40, 75, 375 |
| **Confusion Matrix** ||||
| `tp`, `tn`, `fp`, `fn` | Integer | Confusion matrix elements | ≥0 |
| **Performance Metrics** ||||
| `auc` | Float | AUC for this subgroup | 0.0-1.0 |
| `auc_ci_lower` | Float | AUC 95% CI lower | 0.0-1.0 |
| `auc_ci_upper` | Float | AUC 95% CI upper | 0.0-1.0 |
| `accuracy` | Float | Accuracy for subgroup | 0.0-1.0 |
| `sensitivity` | Float | Sensitivity for subgroup | 0.0-1.0 |
| `specificity` | Float | Specificity for subgroup | 0.0-1.0 |
| `precision` | Float | Precision for subgroup | 0.0-1.0 |
| `npv` | Float | NPV for subgroup | 0.0-1.0 |
| `f1` | Float | F1 score for subgroup | 0.0-1.0 |
| **Fairness Metric** ||||
| `ppr` | Float | Positive prediction rate | 0.0-1.0 | (TP+FP)/N - rate of positive predictions |

### Fairness Metrics Explained

| Metric | Formula | Interpretation | Fair Range |
|--------|---------|----------------|------------|
| **Positive Prediction Rate (PPR)** | (TP+FP)/N | Rate at which group is predicted positive | Should be similar across groups |
| **Demographic Parity** | max(PPR) - min(PPR) | Difference in PPR between groups | <0.05 is good |
| **Equalized Odds** | max(\|Sens_A - Sens_B\|, \|Spec_A - Spec_B\|) | Max difference in sensitivity or specificity | <0.10 is good |

---

## Data Flow Summary

```
INPUT (raw CSV)
    ├─ DE_ID
    ├─ content  → RENAMED TO → txt
    ├─ Label (string)  → CONVERTED TO → label (0/1)
    └─ Demographics (GENDER, RACE, HISPANIC, etc.)
         ↓
01_prepare_data.R
         ↓
PROCESSED DATA (train_set.rds, test_set.rds)
    ├─ DE_ID
    ├─ txt (renamed from content)
    ├─ Label (original string, kept for reference)
    ├─ label (NEW numeric: 0=CTRL, 1=ADRD)
    └─ Demographics
         ↓
02_train_cnnr.R (uses txt and label)
         ↓
MODELS (models/*.h5, tokenizer, etc.)
         ↓
03_evaluate_models.R
         ↓
PREDICTIONS (predictions_df.csv)
    ├─ DE_ID
    ├─ Label (numeric, from test set `label`)
    ├─ True_Class (string: "ADRD" or "CTRL")
    ├─ Predicted_Probability (0.0-1.0)
    ├─ Predicted_Class ("ADRD" or "CTRL")
    ├─ Correct (TRUE/FALSE)
    ├─ Risk_Category ("Low", "Moderate", "High")
    └─ Demographics
         ↓
04_demographic_analysis.R
         ↓
SUBGROUP RESULTS (demographic/subgroup_performance.csv)
    ├─ subgroup (Overall, Gender, Race, Ethnicity)
    ├─ category (Male, Female, White, Black, etc.)
    ├─ category_short (simplified names)
    ├─ Sample sizes (n, n_pos, n_neg)
    ├─ Performance metrics (auc, sensitivity, specificity, etc.)
    └─ Fairness metrics (ppr)
```

---

## Common Column Name Confusions ⚠️

### Confusion 1: `Label` vs `label`
- **`Label`** (capital L): Original string from CSV ("ADRD", "NON-ADRD")
- **`label`** (lowercase l): Numeric conversion (1, 0)
- **When**: `Label` retained for reference; `label` used for modeling
- **In predictions_df.csv**: `Label` is **numeric** (confusingly)

### Confusion 2: `content` vs `txt`
- **`content`**: Original column name in raw CSV
- **`txt`**: Renamed column in processed data for compatibility
- **Why**: Training scripts expect `txt` column name

### Confusion 3: True class naming
- **In Label column**: "ADRD", "NON-ADRD" (input)
- **In True_Class column**: "ADRD", "CTRL" (predictions) ← Note: "CTRL" not "NON-ADRD"

### Confusion 4: Demographics
- **Raw values**: All caps, verbose (e.g., "WHITE OR CAUCASIAN")
- **Simplified values**: Mixed case, short (e.g., "White")
- **In data files**: Raw values preserved
- **In plots**: Simplified values displayed

---

## Quick Reference Tables

### Label Conversions

| Stage | Column Name | Data Type | Positive Class | Negative Class |
|-------|-------------|-----------|----------------|----------------|
| **Input CSV** | `Label` | String | `"ADRD"` | `"NON-ADRD"` |
| **Processed RDS** | `Label` | String | `"ADRD"` | `"NON-ADRD"` |
| **Processed RDS** | `label` | Integer | `1` | `0` |
| **Predictions CSV** | `Label` | Integer | `1` | `0` |
| **Predictions CSV** | `True_Class` | String | `"ADRD"` | `"CTRL"` |
| **Predictions CSV** | `Predicted_Class` | String | `"ADRD"` | `"CTRL"` |

### File-Specific Column Lists

**`train_set.rds` / `test_set.rds`**:
- Essential: `DE_ID`, `txt`, `Label` (string), `label` (0/1)
- Optional: `GENDER`, `RACE`, `HISPANIC`, `AGE`, etc.

**`predictions_df.csv`**:
- Essential: `DE_ID`, `Label` (numeric!), `True_Class`, `Predicted_Probability`, `Predicted_Class`, `Correct`, `Risk_Category`
- Optional: `GENDER`, `RACE`, `HISPANIC`

**`subgroup_performance.csv`**:
- Essential: `subgroup`, `category`, `category_short`, `n`, `n_pos`, `n_neg`, `auc`, `accuracy`, `sensitivity`, `specificity`, `f1`, `ppr`
- Optional: `tp`, `tn`, `fp`, `fn`, confidence intervals

---

## Data Validation Checklist

When receiving new data, verify:

- [ ] **Required columns present**: `DE_ID`, `content`, `Label`
- [ ] **DE_ID uniqueness**: No duplicate patient IDs
- [ ] **Label values**: Only "ADRD" or "NON-ADRD" (or 0/1 if already processed)
- [ ] **Text completeness**: No empty `content` fields
- [ ] **Demographics completeness** (if available):
  - [ ] Valid GENDER values
  - [ ] Valid RACE values (check against expected list)
  - [ ] Valid HISPANIC values
- [ ] **Sample size**: At least 100 samples total, ideally 500+
- [ ] **Class balance**: At least 10% of each class (ADRD and NON-ADRD)

---

## Frequently Asked Questions

### Q1: Why is there both `Label` and `label`?
**A**: `Label` (string) is kept for traceability, `label` (numeric) is used for modeling. Numeric labels are required by TensorFlow/Keras.

### Q2: Why does `content` get renamed to `txt`?
**A**: For compatibility with Jihad Obeid's original code and Keras text processing functions which expect a column named `txt`.

### Q3: What if my demographic columns have different names?
**A**: Rename them before running the pipeline:
- Your column → Expected column
- "Sex" → "GENDER"
- "Ethnicity" → "RACE" (or "HISPANIC" depending on content)

### Q4: Can I add additional demographic variables?
**A**: Yes! Additional columns (e.g., `AGE`, `EDUCATION`, `INSURANCE`) will be preserved throughout the pipeline. You'll need to modify `04_demographic_analysis.R` to analyze them.

### Q5: Why are some demographic values all caps?
**A**: They come directly from the EHR system which often stores them in all caps. The pipeline preserves raw values and provides simplified versions for display.

---

## Contact

For questions about column names or data format, contact:
- **Dr. Paul** (Principal Investigator)
- **Research Team**

---

## Document Version

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-06 | Initial comprehensive reference |

---

**END OF DOCUMENT**
