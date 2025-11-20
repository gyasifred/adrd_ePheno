# Approximate Randomization Implementation for Demographic Fairness Testing

**Author**: Gyasi, Frederick
**Date**: 2025-11-20
**Purpose**: Explanation of permutation testing methodology for model performance comparisons

---

## Overview

Approximate randomization (permutation testing) is implemented to assess whether demographic factors significantly affect CNN model performance for ADRD classification. This replaces the bootstrap approach for hypothesis testing while retaining bootstrap for confidence interval estimation.

---

## Method Description

**Null Hypothesis (H₀)**: Demographic group membership does not affect model performance (AUC)

**Test Procedure**:

1. **Calculate observed statistic**: Compute AUC difference between two demographic groups (e.g., Female AUC - Male AUC)

2. **Generate null distribution**:
   - Pool all predictions and labels from both groups
   - Randomly shuffle group assignments (keeping predictions fixed)
   - Recalculate AUC difference
   - Repeat 10,000 times

3. **Calculate p-value**: Proportion of permuted differences ≥ |observed difference|

---

## Why Approximate Randomization Over Bootstrap

| Aspect | Bootstrap | Approximate Randomization |
|--------|-----------|---------------------------|
| **Purpose** | Estimate confidence intervals | Test null hypothesis |
| **Assumption** | Sample approximates population | Exchangeability under H₀ |
| **Interpretation** | "What's the uncertainty in my estimate?" | "Is this difference due to chance?" |
| **Distribution** | Distribution of statistic | Distribution under null |

**Key Advantage**: Approximate randomization provides exact p-values under the null hypothesis without distributional assumptions.

---

## Implementation Details

**Location**: `04_demographic_analysis.R`

**Configuration** (lines 78-82):
```r
N_PERMUTATIONS <- 10000
N_BOOTSTRAP <- 10000
FDR_ALPHA <- 0.05
RUN_STATISTICAL_TESTS <- TRUE
```

**Function**: `compare_groups_comprehensive()` from `utils_statistical_tests.R`

**Demographics Tested**:
- Gender (Female vs Male)
- Race (two largest groups)
- Ethnicity (Hispanic vs Non-Hispanic)
- Insurance (two largest types)
- Education (two largest levels)
- Financial Class (two largest categories)

---

## Output for Each Demographic

1. **Permutation p-value**: Proportion of permuted differences ≥ observed
2. **Cohen's d effect size**: Standardized measure of practical significance
3. **Null distribution plot**: Histogram with observed statistic marked

**Example Output**:
```
Permutation Test: Female vs Male
  Permutation test p-value: 0.0234 *** SIGNIFICANT
  Cohen's d: 0.42 (Small effect)
```

---

## Visualization

Each test generates a null distribution plot saved to `figures/demographic/`:

- **Blue histogram**: Distribution of 10,000 permuted AUC differences
- **Red vertical line**: Observed AUC difference
- **Orange dashed lines**: 95% critical values

This allows visual assessment of where the observed statistic falls within the null distribution.

---

## Statistical Interpretation

**P-value interpretation**:
- p < 0.05: Demographic factor significantly affects model performance
- p ≥ 0.05: No significant effect (difference could be due to chance)

**Effect size interpretation** (Cohen's d):
- |d| < 0.2: Negligible
- 0.2 ≤ |d| < 0.5: Small
- 0.5 ≤ |d| < 0.8: Medium
- |d| ≥ 0.8: Large

**Multiple testing**: Results should be interpreted with FDR correction when comparing across all demographics.

---

## Relationship to Chi-Squared Tests

The code implements **both** tests for complementary insights:

| Test | Question Answered |
|------|-------------------|
| **Chi-squared** | "Is ADRD/Control case distribution independent of demographics?" |
| **Permutation** | "Is model AUC significantly different between demographic groups?" |

Chi-squared tests case distribution; permutation tests model performance.

---

## Code References

- Permutation test function: `utils_statistical_tests.R`, lines 20-109
- Null distribution plotting: `04_demographic_analysis.R`, lines 167-235
- Gender test: lines 707-738
- Race test: lines 839-876
- Ethnicity test: lines 970-1006
- SDOH tests: lines 1091-1369

---

## Summary

The approximate randomization implementation provides a rigorous, non-parametric approach to testing whether demographic factors significantly affect ADRD classification performance. With 10,000 permutations per test, we obtain stable p-value estimates while making minimal distributional assumptions. The combination of permutation p-values, effect sizes, and visualizations provides a comprehensive fairness assessment framework.
