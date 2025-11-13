# AIM 1: Demographic Performance Fairness Analysis

**Author**: Gyasi, Frederick
**Script**: `04_demographic_analysis.R`
**Version**: 2.0
**Purpose**: Comprehensive guide to demographic fairness analysis for ADRD ePhenotyping

---

## Table of Contents

1. [Research Objectives](#research-objectives)
2. [Research Questions](#research-questions)
3. [Implementation Overview](#implementation-overview)
4. [Statistical Methods Explained](#statistical-methods-explained)
5. [How to Interpret Results](#how-to-interpret-results)
6. [Example Interpretation](#example-interpretation)
7. [Fairness Assessment Framework](#fairness-assessment-framework)
8. [Output Files Reference](#output-files-reference)

---

## Research Objectives

### Primary Objective

**Evaluate whether the ADRD ePhenotyping CNN model performs equitably across demographic subgroups and social determinants of health (SDOH).**

### Secondary Objectives

1. **Quantify performance disparities** across patient demographics
2. **Identify vulnerable subgroups** with systematically worse performance
3. **Detect compound disparities** at intersectional levels (e.g., Gender × Race)
4. **Test statistical significance** of observed performance differences
5. **Provide actionable recommendations** for model recalibration or clinical deployment considerations

### Fairness Principles

This analysis operationalizes **equitable predictive performance**, defined as:

> *"The model should achieve similar diagnostic accuracy (AUC, sensitivity, specificity) across demographic groups, such that no patient population is systematically disadvantaged in ADRD detection."*

---

## Research Questions

### **Question 1: Demographic Performance**

**Q1a**: Does the CNN model's AUC differ significantly by gender?
**Q1b**: Does the CNN model's AUC differ significantly by race?
**Q1c**: Does the CNN model's AUC differ significantly by ethnicity?

**Hypothesis**:
- **H₀**: AUC is statistically equivalent across demographic subgroups
- **H₁**: AUC differs significantly, indicating demographic bias

---

### **Question 2: Social Determinants of Health (SDOH)**

**Q2a**: Does model performance differ by insurance type?
**Q2b**: Does model performance differ by education level?
**Q2c**: Does model performance differ by financial class?

**Hypothesis**:
- **H₀**: SDOH factors do not affect model performance
- **H₁**: SDOH factors significantly affect performance, indicating socioeconomic bias

**Clinical Relevance**: If model performance varies by insurance/education/financial status, this suggests the model may exacerbate existing healthcare disparities.

---

### **Question 3: Intersectionality**

**Q3**: Do certain intersectional groups (Gender × Race combinations) experience compound disparities?

**Hypothesis**:
- **H₀**: Intersectional performance equals or exceeds marginal performance
- **H₁**: Compound disparities exist (e.g., Black Females perform worse than either Black patients OR Female patients individually)

**Example Scenario**:
- Overall AUC: xx
- Female AUC: xx (acceptable)
- Black AUC: xx (acceptable)
- **Black Female AUC: xx (poor) ← Compound disparity**

---

### **Question 4: Statistical Significance**

**Q4**: Are observed performance differences statistically significant or due to random chance?

**Approach**:
- **Chi-squared test** for case/control distribution independence
- **Bootstrap confidence intervals** for AUC estimates
- **Effect size measures** to distinguish statistical vs. clinical significance

---

## Implementation Overview

### Analysis Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT: Predictions from 03_evaluate_models.R               │
│   - predictions_df.csv (model predictions on test set)     │
│   - test_set.rds (demographic variables)                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Data Normalization                                  │
│   - Standardize gender (FEMALE/F → Female)                  │
│   - Detect SDOH variables (Insurance/Education/Financial)   │
│   - Filter to valid demographic categories                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Overall Performance Baseline                        │
│   - Calculate overall AUC, sensitivity, specificity, F1     │
│   - Bootstrap 95% confidence intervals                      │
│   - Establish reference performance                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Demographic Subgroup Analysis                       │
│   - For each demographic variable (Gender, Race, etc.):     │
│     * Calculate subgroup-specific metrics                   │
│     * Bootstrap CIs                                          │
│     * Compare to overall baseline                           │
│     * Chi-squared test for independence                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: SDOH Subgroup Analysis                              │
│   - Same process for Insurance, Education, Financial Class  │
│   - Flag socioeconomic disparities                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: Intersectional Analysis                             │
│   - Gender × Race combinations (min N=30)                   │
│   - Detect compound disparities                             │
│   - Best vs worst performing groups                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: Visualization & Reporting                           │
│   - Enhanced AUC plots (Demographic/SDOH/Intersectional)    │
│   - Sensitivity-Specificity scatter                         │
│   - Intersectional heatmap                                  │
│   - Comprehensive text report                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ OUTPUT FILES                                                 │
│   - demographic_metrics.csv (all subgroup metrics)          │
│   - demographic_summary_report.txt (interpretation)         │
│   - Figures: AUC comparison, heatmaps, scatter plots        │
└─────────────────────────────────────────────────────────────┘
```

---

### Key Implementation Components

#### **1. Data Normalization (Lines 294-329, 422-470)**

**Purpose**: Standardize inconsistent demographic values across EHR systems

**Implementation**:
```r
# Gender normalization
GENDER = case_when(
  toupper(GENDER) %in% c("FEMALE", "F") ~ "Female",
  toupper(GENDER) %in% c("MALE", "M") ~ "Male",
  toupper(GENDER) == "UNKNOWN" ~ NA_character_,
  TRUE ~ GENDER
)
```

**Why This Matters**: Different hospitals may encode gender as "FEMALE", "F", "Female", "female". Without normalization, these would be treated as separate categories, fragmenting the analysis.

---

#### **2. SDOH Variable Detection (Lines 422-470)**

**Purpose**: Automatically detect social determinant variables regardless of institution-specific naming

**Implementation**:
```r
# Insurance detection - tries multiple column names
insurance_cols <- c("INSURANCE", "INSURANCE_TYPE", "PAYER", "INSURANCE_CLASS")
if (any(insurance_cols %in% names(analysis_data))) {
  found_col <- intersect(insurance_cols, names(analysis_data))[1]
  analysis_data <- rename(INSURANCE = !!found_col)
}
```

**Why This Matters**: Enables cross-institutional comparisons even when data dictionaries differ.

---

#### **3. Chi-Squared Independence Test (Lines 164-234)**

**Purpose**: Test whether ADRD vs Control case distribution is independent of demographic category

**Null Hypothesis (H₀)**: The proportion of ADRD cases is the same across all demographic subgroups

**Statistical Test Selection**:
- **Chi-squared test**: Used when all expected cell frequencies ≥ 5
- **Fisher's exact test**: Used when expected frequencies < 5 (small samples)

**Implementation**:
```r
perform_chi_squared_test <- function(data, group_var, label_var) {
  cont_table <- table(data[[label_var]], data[[group_var]])
  chi_test <- chisq.test(cont_table)

  # If expected frequencies < 5, use Fisher's exact
  if (any(chi_test$expected < 5)) {
    fisher.test(cont_table, simulate.p.value = TRUE, B = 10000)
  } else {
    chi_test
  }
}
```

**What This Tests**:
| Group    | ADRD Cases | Controls | Total |
|----------|------------|----------|-------|
| Female   | xx         | xx       | xx    |
| Male     | xx         | xx       | xx    |

Chi-squared tests: **"Are ADRD/Control proportions significantly different between Female and Male groups?"**

---

#### **4. Bootstrap Confidence Intervals (Lines 236-293)**

**Purpose**: Quantify uncertainty in performance metrics without parametric assumptions

**Method**: Stratified bootstrap with 10,000 resamples

**Implementation**:
```r
bootstrap_samples <- replicate(10000, {
  # Stratified sampling (preserve ADRD/Control ratio)
  boot_idx <- c(
    sample(which(y_true == 1), replace = TRUE),
    sample(which(y_true == 0), replace = TRUE)
  )
  # Calculate AUC on bootstrap sample
  auc(roc(y_true[boot_idx], y_pred_prob[boot_idx], quiet = TRUE))
})

# 95% CI from bootstrap distribution
ci_lower <- quantile(bootstrap_samples, 0.025)
ci_upper <- quantile(bootstrap_samples, 0.975)
```

**Why Stratified**: Preserves the ADRD/Control ratio in each bootstrap sample, ensuring valid resampling.

---

#### **5. Subgroup Metrics Calculation (Lines 236-293)**

**Purpose**: Calculate comprehensive performance metrics for each demographic subgroup

**Metrics Calculated**:

| Metric            | Formula                          | Interpretation                    |
|-------------------|----------------------------------|-----------------------------------|
| **AUC**           | Area under ROC curve             | Overall discrimination ability    |
| **Sensitivity**   | TP / (TP + FN)                   | True positive rate                |
| **Specificity**   | TN / (TN + FP)                   | True negative rate                |
| **PPV**           | TP / (TP + FP)                   | Positive predictive value         |
| **NPV**           | TN / (TN + FN)                   | Negative predictive value         |
| **F1 Score**      | 2 × (PPV × Sens) / (PPV + Sens) | Harmonic mean of precision/recall |
| **Accuracy**      | (TP + TN) / Total                | Overall correctness               |

**Bootstrap CIs**: Calculated for AUC to quantify uncertainty

---

#### **6. Intersectional Analysis (Lines 1101-1171)**

**Purpose**: Detect compound disparities at Gender × Race intersections

**Minimum Sample Size**: N ≥ 30 per intersectional group (to ensure statistical power)

**Implementation**:
```r
# Create intersection variable
intersection = paste(GENDER, "×", RACE)

# Analyze each intersection
for (group in unique(intersections)) {
  subset_data <- filter(data, intersection == group)
  if (nrow(subset_data) >= 30) {
    metrics <- calculate_subgroup_metrics(subset_data)
  }
}

# Identify best/worst performing groups
auc_range <- max(intersectional_aucs) - min(intersectional_aucs)
if (auc_range > 0.10) {
  cat("⚠️ WARNING: Compound disparities detected\n")
}
```

**Compound Disparity Detection**:
- Compare intersectional AUC to marginal AUCs
- Example: If `Female AUC = 0.92` and `Black AUC = 0.91`, but `Black Female AUC = 0.82`, this indicates a **compound disparity**

---

## Statistical Methods Explained

### Chi-Squared Test for Independence

**What It Does**: Tests whether the distribution of ADRD vs Control cases differs across demographic groups

**When to Use**:
- Categorical predictor (e.g., Gender, Race)
- Binary outcome (ADRD vs Control)
- Independent observations

**Assumptions**:
- ✅ Independent observations (patients)
- ✅ Expected cell frequencies ≥ 5 (otherwise use Fisher's exact)

**Formula**:
```
χ² = Σ [(Observed - Expected)² / Expected]

degrees of freedom = (rows - 1) × (columns - 1)
```

**Example Contingency Table**:

|           | Female | Male | Total |
|-----------|--------|------|-------|
| ADRD      | xx     | xx   | xx    |
| Control   | xx     | xx   | xx    |
| **Total** | xx     | xx   | xx    |

**Interpretation**:
- **p < 0.05**: ADRD prevalence differs significantly by group → suggests potential bias
- **p ≥ 0.05**: No significant difference in case distribution

---

### Bootstrap Confidence Intervals

**What It Does**: Provides robust uncertainty estimates for AUC without parametric assumptions

**Method**:
1. Resample with replacement from original data (stratified by ADRD/Control)
2. Calculate AUC on bootstrap sample
3. Repeat 10,000 times
4. CI = 2.5th and 97.5th percentiles of bootstrap AUC distribution

**Why Stratified**: Preserves the class imbalance ratio, ensuring each bootstrap sample is representative

**Interpretation**:
- **Narrow CI** (e.g., [0.91, 0.93]): High precision, reliable estimate
- **Wide CI** (e.g., [0.78, 0.95]): Low precision, possibly small sample size
- **Non-overlapping CIs**: Suggests statistically significant difference between groups

---

### Performance Disparity Thresholds

**AUC Disparity**:
- ✅ **<0.05**: Negligible disparity
- ⚠️ **0.05-0.10**: Moderate disparity (monitor)
- 🚨 **>0.10**: Severe disparity (requires intervention)

**Sensitivity Disparity**:
- ✅ **<0.10**: Acceptable
- ⚠️ **0.10-0.15**: Moderate (clinical review needed)
- 🚨 **>0.15**: Severe (unequal false negative rates)

**Clinical Significance**:
- A sensitivity difference of 0.15 means one group has **15% more missed ADRD cases** than another
- Example: If Male sensitivity = 0.85 and Female sensitivity = 0.70, then **15% more ADRD cases are missed in females**

---

## How to Interpret Results

### Step-by-Step Interpretation Guide

#### **1. Overall Performance Baseline**

**File**: `results/demographic/demographic_summary_report.txt`

**Look For**:
```
OVERALL MODEL PERFORMANCE
-------------------------
N: xx (ADRD: xx, Control: xx)
AUC: xx [95% CI: xx - xx]
Sensitivity: xx
Specificity: xx
F1 Score: xx
```

**Interpretation**:
- **AUC ≥ 0.90**: Excellent discrimination
- **AUC 0.80-0.89**: Good discrimination
- **AUC 0.70-0.79**: Acceptable discrimination
- **AUC < 0.70**: Poor discrimination

**Question**: *"Is the overall model performance acceptable for clinical deployment?"*

---

#### **2. Gender Analysis**

**File**: `results/demographic/demographic_summary_report.txt` → Section: "GENDER ANALYSIS"

**Look For**:
```
GENDER ANALYSIS
---------------
Female:
  N: xx (ADRD: xx, CTRL: xx)
  AUC: xx [xx - xx]
  Sensitivity: xx | Specificity: xx

Male:
  N: xx (ADRD: xx, CTRL: xx)
  AUC: xx [xx - xx]
  Sensitivity: xx | Specificity: xx

Statistical Test:
  Chi-squared p-value: xx
  Result: [SIGNIFICANT/NOT SIGNIFICANT]

Performance Comparison:
  AUC range: xx [⚠️ WARNING / ✓ OK]
```

**Interpretation Steps**:

1. **Compare AUCs**:
   - If Female AUC = xx and Male AUC = xx:
   - Difference = |xx - xx| = xx

2. **Check Statistical Significance**:
   - If p < 0.05 → **Significant difference** (gender affects performance)
   - If p ≥ 0.05 → No significant difference

3. **Assess Clinical Impact**:
   - AUC diff > 0.10 → **Clinically meaningful disparity**
   - Example: Female AUC = 0.92, Male AUC = 0.80 → **12% disparity → SEVERE**

4. **Check Sensitivity/Specificity**:
   - Sensitivity diff > 0.15 → **Unequal false negative rates**
   - Example: Female Sens = 0.70, Male Sens = 0.85 → Females have **15% more missed ADRD cases**

**Action Items**:
- ✅ **No disparity (diff < 0.05)**: Model is fair across gender
- ⚠️ **Moderate disparity (0.05-0.10)**: Document limitation, monitor in deployment
- 🚨 **Severe disparity (>0.10)**: Consider gender-stratified models or recalibration

---

#### **3. Race/Ethnicity Analysis**

**File**: `results/demographic/demographic_summary_report.txt` → "RACE ANALYSIS" / "ETHNICITY ANALYSIS"

**Same interpretation as Gender**, but additional considerations:

**Sample Size Warning**:
- Some racial groups may have small N (e.g., N < 30)
- Wide confidence intervals indicate low precision
- Document as limitation if specific racial groups are underrepresented

**Example**:
```
White: N = xx, AUC = xx [xx - xx]    ← Narrow CI (reliable)
Black: N = xx, AUC = xx [xx - xx]    ← Narrow CI (reliable)
Asian: N = 15, AUC = xx [xx - xx]    ← Wide CI (unreliable)
```

**Interpretation**:
- If Asian CI = [0.65, 0.98] → Too uncertain to conclude fairness/unfairness
- **Action**: Acknowledge as study limitation, recommend external validation in Asian cohorts

---

#### **4. SDOH Analysis (Insurance/Education/Financial)**

**File**: `results/demographic/demographic_summary_report.txt` → "SOCIAL DETERMINANTS OF HEALTH"

**Look For**:
```
INSURANCE ANALYSIS
------------------
Medicare:  AUC = xx [xx - xx]
Medicaid:  AUC = xx [xx - xx]
Commercial: AUC = xx [xx - xx]

Chi-squared p-value: xx
AUC range: xx [⚠️ WARNING / ✓ OK]
```

**Interpretation**:

**Question**: *"Does the model perform worse for patients with Medicaid (low socioeconomic status)?"*

**Red Flags**:
- Medicaid AUC << Commercial AUC → **Socioeconomic bias**
- Example: Medicaid AUC = 0.78, Commercial AUC = 0.93 → **15% disparity**
- **Implication**: Model may exacerbate healthcare disparities, disadvantaging underserved populations

**Clinical/Ethical Considerations**:
- If SDOH disparity exists → Raises ethical concerns about algorithmic fairness
- May require **policy interventions** (e.g., manual review for Medicaid patients)

---

#### **5. Intersectional Analysis**

**File**: `results/demographic/demographic_summary_report.txt` → "INTERSECTIONAL ANALYSIS"

**Look For**:
```
INTERSECTIONAL ANALYSIS (Gender × Race)
---------------------------------------
Best performing:  [Female × White]   AUC = xx
Worst performing: [Male × Black]     AUC = xx
Performance gap:  xx

⚠️ FINDING: Compound disparities detected
```

**Interpretation**:

**Compound Disparity**: When intersectional performance is worse than either marginal group

**Example**:
- Female AUC: 0.91
- Black AUC: 0.89
- **Black Female AUC: 0.79** ← Compound disparity!

**Why This Matters**:
- Simple gender/race analysis would miss this group's poor performance
- Intersectionality reveals **hidden vulnerable populations**

**Action Items**:
- Performance gap > 0.10 → **Recommend targeted recalibration** for worst-performing groups
- Document as key finding in manuscript

---

#### **6. Visualization Interpretation**

**File**: `figures/demographic/auc_by_subgroup_enhanced.png`

**What to Look For**:

1. **Dashed Red Line**: Overall AUC baseline
2. **Bar Heights**: Subgroup-specific AUCs
3. **Error Bars**: 95% confidence intervals
4. **Facets**: Separated by factor type (Demographic, SDOH, Intersectional)
5. **Faded Bars**: AUC < baseline - 0.05 (warning)

**Interpretation**:
- Bars **above baseline** → Better than overall performance
- Bars **below baseline** → Worse than overall performance
- **Faded bars** → Clinically significant underperformance
- **Wide error bars** → High uncertainty (small sample)

**Example**:
```
           |            ← Overall AUC = 0.90
-----------+------------
 Female    |████████████  AUC = 0.92 [0.89-0.95]  ✓
 Male      |███████      AUC = 0.82 [0.78-0.86]  ⚠️ (faded)
```
→ **Interpretation**: Male performance significantly below baseline

---

**File**: `figures/demographic/intersectional_heatmap.png`

**What to Look For**:
- **Color Scale**: Red (poor AUC) → Yellow (medium) → Green (excellent)
- **Midpoint**: Overall AUC (yellow)
- **Dark Red Cells**: Compound disparities

**Example**:
```
           White    Black    Asian
Female    🟢 0.93  🟡 0.88  🟢 0.91
Male      🟢 0.92  🔴 0.75  🟡 0.87
```
→ **Interpretation**: Black Male group has severe underperformance (AUC = 0.75)

---

## Example Interpretation

### Scenario: Complete Analysis Results

**Overall Performance**:
```
AUC: 0.90 [0.88 - 0.92]
Sensitivity: 0.85
Specificity: 0.88
```
→ **Interpretation**: Excellent overall performance

---

**Gender Analysis**:
```
Female: AUC = 0.92 [0.89 - 0.95], N = xx
Male:   AUC = 0.87 [0.83 - 0.91], N = xx

Chi-squared p = 0.032 (SIGNIFICANT)
AUC difference: 0.05 (MODERATE WARNING)
```

**Interpretation**:
1. ✅ Both genders have good AUC (>0.85)
2. ⚠️ Female AUC statistically significantly higher (p < 0.05)
3. ⚠️ Difference = 0.05 (moderate disparity, at threshold)
4. **Conclusion**: Gender bias exists but is moderate. Document as limitation. Consider stratified performance reporting in clinical deployment.

---

**Race Analysis**:
```
White: AUC = 0.91 [0.88 - 0.94], N = xx
Black: AUC = 0.88 [0.84 - 0.92], N = xx
Asian: AUC = 0.86 [0.75 - 0.97], N = 18 (small sample)

Chi-squared p = 0.18 (NOT SIGNIFICANT)
AUC range: 0.05 (OK)
```

**Interpretation**:
1. ✅ All racial groups have good AUC (>0.85)
2. ✅ No statistically significant difference (p = 0.18)
3. ⚠️ Asian CI is wide [0.75-0.97] due to small N → Low precision
4. **Conclusion**: No racial bias detected, but Asian group underpowered. External validation needed for Asian populations.

---

**Insurance Analysis (SDOH)**:
```
Commercial: AUC = 0.92 [0.89 - 0.95], N = xx
Medicare:   AUC = 0.89 [0.86 - 0.92], N = xx
Medicaid:   AUC = 0.80 [0.75 - 0.85], N = xx

Chi-squared p = 0.003 (SIGNIFICANT)
AUC range: 0.12 (⚠️ SEVERE DISPARITY)
```

**Interpretation**:
1. 🚨 **Severe socioeconomic bias**: Medicaid patients have 12% lower AUC
2. 🚨 Chi-squared p = 0.003 → Highly significant
3. **Clinical Impact**: Patients with lower socioeconomic status (Medicaid) are **systematically disadvantaged** by the algorithm
4. **Ethical Concern**: Model may exacerbate health disparities
5. **Action Items**:
   - Consider Medicaid-specific recalibration
   - Implement manual review process for Medicaid patients
   - Document as major limitation in manuscript
   - Discuss ethical implications in paper

---

**Intersectional Analysis**:
```
Best:  Female × White:  AUC = 0.94
Worst: Male × Black:    AUC = 0.82
Gap:   0.12 (⚠️ COMPOUND DISPARITY)

Marginal performance:
  Male AUC: 0.87
  Black AUC: 0.88
  Male × Black AUC: 0.82 ← Worse than either marginal group!
```

**Interpretation**:
1. 🚨 **Compound disparity detected**: Black Male group underperforms beyond additive effects
2. **Intersectionality insight**: Simple gender/race analysis (Male=0.87, Black=0.88) would miss this vulnerable group (0.82)
3. **Action Items**:
   - Targeted investigation of Black Male documentation patterns
   - Consider intersectional stratified models
   - Highlight in manuscript as key fairness finding

---

## Fairness Assessment Framework

### Decision Tree for Action Items

```
Is overall AUC ≥ 0.80?
│
├─ NO → Model not ready for clinical use (poor baseline performance)
│
└─ YES → Check demographic fairness:
    │
    ├─ AUC disparity < 0.05 for all groups?
    │   └─ YES → ✅ Fair model, ready for deployment
    │
    ├─ AUC disparity 0.05-0.10?
    │   └─ MODERATE DISPARITY:
    │       • Document in limitation section
    │       • Monitor in real-world deployment
    │       • Report stratified performance to clinicians
    │
    └─ AUC disparity > 0.10?
        └─ SEVERE DISPARITY:
            • Consider recalibration for disadvantaged groups
            • Implement manual review processes
            • Discuss ethical implications
            • May require algorithmic intervention before deployment
```

---

### Reporting Checklist for Manuscripts

**Table 1: Overall Performance**
- [ ] Overall AUC with 95% CI
- [ ] Sensitivity, Specificity, PPV, NPV, F1
- [ ] Sample size (N, ADRD, Control)

**Table 2: Demographic Subgroup Performance**
- [ ] AUC [95% CI] for each demographic category
- [ ] Chi-squared test results (statistic, df, p-value)
- [ ] Performance disparity ranges

**Table 3: SDOH Subgroup Performance**
- [ ] Insurance/Education/Financial Class metrics
- [ ] Statistical significance tests
- [ ] Socioeconomic bias flagged if present

**Table 4: Intersectional Analysis**
- [ ] Gender × Race combinations (N ≥ 30)
- [ ] Best/worst performing groups
- [ ] Compound disparity assessment

**Figure 1: AUC Comparison**
- [ ] Enhanced bar plot with factor types (Demographic/SDOH/Intersectional)
- [ ] Error bars (95% CI)
- [ ] Overall baseline (dashed line)

**Figure 2: Intersectional Heatmap**
- [ ] Gender × Race performance matrix
- [ ] Color-coded by AUC
- [ ] Sample sizes annotated

**Discussion Points**:
- [ ] Interpretation of statistically significant disparities
- [ ] Clinical implications of performance gaps
- [ ] Ethical considerations (especially SDOH disparities)
- [ ] Limitations (small sample sizes in specific groups)
- [ ] Recommendations for deployment (stratified reporting, manual review)

---

## Output Files Reference

### Results Directory: `results/demographic/`

| File                               | Contents                                  |
|------------------------------------|-------------------------------------------|
| `demographic_metrics.csv`          | All subgroup metrics (AUC, Sens, Spec, F1, CIs) |
| `demographic_summary_report.txt`   | Human-readable interpretation report      |
| `overall_performance.rds`          | R object with overall metrics             |
| `subgroup_metrics.rds`             | R object with all subgroup metrics        |

---

### Figures Directory: `figures/demographic/`

| File                                  | Visualization                           |
|---------------------------------------|-----------------------------------------|
| `auc_by_subgroup_enhanced.png`        | Enhanced AUC comparison (faceted by type) |
| `sensitivity_specificity.png`         | Scatter plot of Sens vs Spec            |
| `intersectional_heatmap.png`          | Gender × Race heatmap                   |
| `performance_comparison_matrix.png`   | Comprehensive metric matrix             |

---

## Additional Resources

**Related Documentation**:
- `STATISTICAL_SIGNIFICANCE_METHODOLOGY.md` - Statistical testing methods
- `README.md` - Pipeline overview
- `JIHAD_ARTIFACTS_COMPATIBILITY.md` - Model artifact reference

**Script**: `04_demographic_analysis.R`

**Dependencies**:
- `utils_statistical_tests.R` - Statistical testing utilities
- `results/predictions_df.csv` - Input predictions
- `data/test_set.rds` - Demographics

**Support**: Gyasi, Frederick

---

**Version History**:
- v2.0 (2025-11-13): Complete SDOH, intersectional, and statistical testing implementation
- v1.0 (2025-11-06): Initial demographic analysis

---

**End of AIM 1 Guide**
