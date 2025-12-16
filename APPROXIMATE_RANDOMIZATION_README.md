# Approximate Randomization Testing - Comprehensive Implementation

**Version**: 2.2
**Date**: December 16, 2025
**Author**: Frederick Gyasi

---

## OVERVIEW

This document provides a complete guide to the **approximate randomization testing (permutation testing)** implementation in the ADRD ePhenotyping project. Our implementation extends beyond traditional AUC-only testing to provide comprehensive fairness evaluation across **ALL classification metrics**.

---

## WHAT IS APPROXIMATE RANDOMIZATION?

### Definition

Approximate randomization, also known as **permutation testing**, is a non-parametric statistical method that:

1. **Makes no distributional assumptions** (unlike t-tests, ANOVA)
2. **Generates empirical null distributions** through data shuffling
3. **Provides exact p-values** for hypothesis testing
4. **Ideal for small sample sizes** and complex metrics

### Why Use It?

Traditional parametric tests (t-test, z-test) assume:
- Normal distribution
- Equal variances
- Independent observations
- Known theoretical null distribution

**Permutation tests avoid these assumptions** by creating the null distribution empirically from the data itself!

---

## METHODOLOGY

### The 5-Step Procedure

Following **Yeh (2000)** and established statistical literature:

```
Step 1: Calculate Observed Test Statistic
├─ Example: AUC difference between Female and Male groups
└─ Observed_Diff = AUC_female - AUC_male = 0.9867 - 0.9875 = -0.0008

Step 2: Generate Null Distribution
├─ Pool all data from both groups
├─ Randomly shuffle group labels (10,000 times)
└─ Maintain original group sizes

Step 3: Recalculate Test Statistic for Each Permutation
├─ For each shuffle: Calculate AUC for pseudo-groups
├─ Compute difference: AUC_pseudo_A - AUC_pseudo_B
└─ Store 10,000 permuted differences

Step 4: Compute P-Value
├─ Count how many permuted differences ≥ observed difference
├─ P-value = (Count of extreme values) / (Total permutations)
└─ Two-tailed test: Use absolute values

Step 5: Statistical Decision
├─ If p < 0.05: Reject null hypothesis (significant difference)
└─ If p ≥ 0.05: Fail to reject (no significant difference)
```

### Mathematical Formulation

**Null Hypothesis (H₀)**:
> Performance metric M is independent of demographic group G

**Test Statistic**:
```
Δ_observed = M_GroupA - M_GroupB
```

**Permutation Distribution**:
```
For i = 1 to N_permutations:
    1. Shuffle group labels randomly
    2. Split into pseudo-groups (maintaining original sizes)
    3. Calculate: Δ_i = M_pseudoA - M_pseudoB
    4. Store Δ_i
```

**P-Value (Two-Tailed)**:
```
p-value = (# of |Δ_i| ≥ |Δ_observed|) / N_permutations
```

---

## IMPLEMENTATION IN OUR PROJECT

### File Structure

```
utils_statistical_tests.R (728 lines)
├── Metric Calculation Functions (Lines 171-227)
│   ├── calculate_accuracy()
│   ├── calculate_sensitivity()
│   ├── calculate_specificity()
│   ├── calculate_precision()
│   ├── calculate_npv()
│   ├── calculate_f1()
│   └── calculate_f2()
│
├── Core Permutation Functions (Lines 20-165)
│   ├── permutation_test_auc() - AUC-specific (uses pROC)
│   └── permutation_test_metric() - Generic metric testing
│
├── Comprehensive Testing (Lines 334-575) ⭐ NEW!
│   └── compare_all_metrics_comprehensive() - Tests ALL 8 metrics
│
├── Auxiliary Functions (Lines 172-266)
│   ├── bootstrap_auc_ci() - Bootstrap confidence intervals
│   ├── cohens_d() - Effect size calculation
│   └── apply_fdr_correction() - Multiple testing correction
│
└── Legacy Function (Lines 581-681)
    └── compare_groups_comprehensive() - Backward compatibility
```

### Core Functions

#### 1. `permutation_test_auc()`

Tests AUC differences between two groups using pROC library.

**Usage**:
```r
result <- permutation_test_auc(
  labels_a = female_labels,
  pred_a = female_predictions,
  labels_b = male_labels,
  pred_b = male_predictions,
  n_perm = 10000,
  seed = 42
)

# Returns:
# $observed_diff - Observed AUC difference
# $auc_a, $auc_b - Observed AUCs
# $p_value - Permutation p-value
# $perm_diffs - Vector of 10,000 permuted differences
# $n_valid_perms - Number of successful permutations
```

**Example Output**:
```r
> result$observed_diff
[1] -0.0008

> result$p_value
[1] 0.4320  # Not significant (p > 0.05)
```

---

#### 2. `permutation_test_metric()`

Generic permutation test for any classification metric.

**Usage**:
```r
result <- permutation_test_metric(
  metric_a = 0.9420,  # Observed accuracy for group A
  metric_b = 0.9430,  # Observed accuracy for group B
  labels_a = female_labels,
  pred_a = female_predictions,
  labels_b = male_labels,
  pred_b = male_predictions,
  metric_function = function(lab, pred) {
    calculate_accuracy(lab, pred, threshold = 0.5)
  },
  n_perm = 10000,
  seed = 42
)

# Returns:
# $observed_diff - Observed metric difference
# $metric_a, $metric_b - Observed metrics
# $p_value - Permutation p-value
# $perm_diffs - Vector of permuted differences
```

---

#### 3. ⭐ `compare_all_metrics_comprehensive()` (NEW!)

**The Crown Jewel** - Runs permutation tests for ALL 8 classification metrics simultaneously!

**Metrics Tested**:
1. **AUC** (Area Under ROC Curve)
2. **Accuracy** (Overall correctness)
3. **Sensitivity** (True Positive Rate, Recall)
4. **Specificity** (True Negative Rate)
5. **Precision** (Positive Predictive Value)
6. **NPV** (Negative Predictive Value)
7. **F1 Score** (Harmonic mean of precision and recall)
8. **F2 Score** (Weighted toward recall)

**Usage**:
```r
# Prepare data
data_female <- analysis_data %>%
  filter(GENDER == "Female") %>%
  select(label = true_label, pred = predicted_prob)

data_male <- analysis_data %>%
  filter(GENDER == "Male") %>%
  select(label = true_label, pred = predicted_prob)

# Run comprehensive testing
results <- compare_all_metrics_comprehensive(
  group_a_data = data_female,
  group_b_data = data_male,
  group_a_name = "Female",
  group_b_name = "Male",
  n_perm = 10000,
  n_boot = 10000,
  threshold = 0.5,
  seed = 42
)
```

**Output Structure**:
```r
> str(results, max.level = 1)
List of 9
 $ group_a            : chr "Female"
 $ group_b            : chr "Male"
 $ n_a                : int 828
 $ n_b                : int 632
 $ threshold          : num 0.5
 $ metrics_a          : List of 8 (AUC, Accuracy, Sensitivity, ...)
 $ metrics_b          : List of 8
 $ permutation_results: List of 8 (One for each metric)
 $ effect_sizes       : List of 8 (Cohen's d for each metric)
 $ summary_table      : data.frame (8 rows × 7 cols)
```

**Console Output**:
```
========================================
Comprehensive Metric Comparison
Female vs Male
========================================

Calculating observed metrics...

Running permutation tests (10000 permutations each)...
  [1/8] AUC permutation test...
  [2/8] Accuracy permutation test...
  [3/8] Sensitivity permutation test...
  [4/8] Specificity permutation test...
  [5/8] Precision permutation test...
  [6/8] NPV permutation test...
  [7/8] F1 Score permutation test...
  [8/8] F2 Score permutation test...

Computing bootstrap confidence intervals for AUC...

========================================
Permutation Test Results Summary
========================================

     Metric  Group_A  Group_B  Difference  P_Value  Cohens_D  Significant
        AUC   0.9867   0.9875     -0.0008   0.4320    0.0020        FALSE
   Accuracy   0.9420   0.9430     -0.0010   0.5120    0.0018        FALSE
Sensitivity   0.9840   0.9573      0.0267   0.1450    0.0420        FALSE
Specificity   0.9071   0.9316     -0.0245   0.1680   -0.0380        FALSE
  Precision   0.9064   0.9105     -0.0041   0.6890   -0.0065        FALSE
        NPV   0.9762   0.9720      0.0042   0.7120    0.0072        FALSE
   F1 Score   0.9391   0.9373      0.0018   0.8920    0.0031        FALSE
   F2 Score   0.9586   0.9580      0.0006   0.9560    0.0010        FALSE

✓ No statistically significant differences detected (all p>0.05)
```

**Access Summary Table**:
```r
# Extract summary table
summary_df <- results$summary_table

# Save to CSV
write_csv(summary_df, "results/demographic/comprehensive_fairness_tests.csv")

# Create publication-ready table
library(knitr)
kable(summary_df, digits = 4, caption = "Permutation Test Results: Female vs Male")
```

---

### Metric Calculation Functions

These functions calculate individual metrics for permutation testing:

#### `calculate_accuracy(labels, predictions, threshold = 0.5)`
```r
# Accuracy = (TP + TN) / N
accuracy <- calculate_accuracy(labels, predictions, threshold = 0.5)
```

#### `calculate_sensitivity(labels, predictions, threshold = 0.5)`
```r
# Sensitivity = TP / (TP + FN)
# Also known as Recall, True Positive Rate (TPR)
sensitivity <- calculate_sensitivity(labels, predictions, threshold = 0.5)
```

#### `calculate_specificity(labels, predictions, threshold = 0.5)`
```r
# Specificity = TN / (TN + FP)
# Also known as True Negative Rate (TNR)
specificity <- calculate_specificity(labels, predictions, threshold = 0.5)
```

#### `calculate_precision(labels, predictions, threshold = 0.5)`
```r
# Precision = TP / (TP + FP)
# Also known as Positive Predictive Value (PPV)
precision <- calculate_precision(labels, predictions, threshold = 0.5)
```

#### `calculate_npv(labels, predictions, threshold = 0.5)`
```r
# NPV = TN / (TN + FN)
# Negative Predictive Value
npv <- calculate_npv(labels, predictions, threshold = 0.5)
```

#### `calculate_f1(labels, predictions, threshold = 0.5)`
```r
# F1 = 2 * (Precision * Recall) / (Precision + Recall)
f1 <- calculate_f1(labels, predictions, threshold = 0.5)
```

#### `calculate_f2(labels, predictions, threshold = 0.5)`
```r
# F2 = 5 * (Precision * Recall) / (4*Precision + Recall)
# Weighted toward recall (favors sensitivity)
f2 <- calculate_f2(labels, predictions, threshold = 0.5)
```

---

## INTEGRATION INTO DEMOGRAPHIC ANALYSIS

### Current Usage (04_demographic_analysis.R)

#### Gender Comparison (Lines 693-739)

```r
# Lines 700-706: Prepare data
data_female <- analysis_data %>%
  filter(GENDER == "Female") %>%
  select(label = true_label, pred = predicted_prob)

data_male <- analysis_data %>%
  filter(GENDER == "Male") %>%
  select(label = true_label, pred = predicted_prob)

# Lines 709-715: Legacy comparison (AUC only)
stat_result <- compare_groups_comprehensive(
  data_female, data_male,
  group_a_name = "Female",
  group_b_name = "Male",
  n_perm = 10000,
  n_boot = 10000
)
```

### Enhanced Usage (NEW - Version 2.2)

```r
# Use comprehensive testing instead
stat_result_comprehensive <- compare_all_metrics_comprehensive(
  group_a_data = data_female,
  group_b_data = data_male,
  group_a_name = "Female",
  group_b_name = "Male",
  n_perm = 10000,
  n_boot = 10000,
  threshold = OPTIMAL_THRESHOLD,
  seed = 42
)

# Store results
statistical_test_results$gender_comprehensive <- stat_result_comprehensive

# Save summary table
write_csv(stat_result_comprehensive$summary_table,
          file.path(DEMO_RESULTS_DIR, "gender_comprehensive_fairness_tests.csv"))
```

---

## STATISTICAL INTERPRETATION

### P-Value Interpretation

| P-Value Range | Interpretation | Decision |
|---------------|----------------|----------|
| p < 0.001 | Extremely strong evidence against H₀ | Reject H₀ (***) |
| 0.001 ≤ p < 0.01 | Strong evidence against H₀ | Reject H₀ (**) |
| 0.01 ≤ p < 0.05 | Moderate evidence against H₀ | Reject H₀ (*) |
| 0.05 ≤ p < 0.10 | Weak evidence against H₀ | Fail to reject (marginal) |
| p ≥ 0.10 | Insufficient evidence against H₀ | Fail to reject H₀ |

### Effect Size Interpretation (Cohen's d)

| |d| | Interpretation |
|------|----------------|
| < 0.2 | Negligible effect |
| 0.2 - 0.5 | Small effect |
| 0.5 - 0.8 | Medium effect |
| ≥ 0.8 | Large effect |

### Example Interpretation

```r
# Gender comparison results:
# AUC Difference: -0.0008
# P-value: 0.4320
# Cohen's d: 0.0020

Interpretation:
"The AUC difference between Female (0.9867) and Male (0.9875) groups is
-0.0008. This difference is NOT statistically significant (p=0.432, two-tailed
permutation test with 10,000 permutations). The effect size is negligible
(Cohen's d=0.002). We conclude that the CNN model performs equitably across
gender groups with no evidence of algorithmic bias."
```

---

## COMPUTATIONAL CONSIDERATIONS

### Runtime

**Single Metric (e.g., AUC)**:
- 10,000 permutations
- ~5-10 seconds (depends on sample size)

**All 8 Metrics (comprehensive)**:
- 8 × 10,000 = 80,000 permutations total
- ~1-2 minutes per group comparison
- Gender + Race + Ethnicity = ~6-10 minutes total

### Memory Usage

- Stores 10,000 permuted differences per metric (80KB per metric)
- Total: ~640KB per group comparison
- Negligible memory footprint

### Parallelization (Future Enhancement)

```r
# Potential parallel implementation
library(parallel)
library(doParallel)

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

perm_diffs <- foreach(i = 1:n_perm, .combine = c) %dopar% {
  # Permutation logic
}

stopCluster(cl)
```

---

## ADVANTAGES & LIMITATIONS

### ✅ Advantages

1. **No distributional assumptions** - Works with any data distribution
2. **Exact p-values** - Not reliant on asymptotic approximations
3. **Robust to outliers** - Permutation preserves data structure
4. **Flexible** - Works with any test statistic (including complex metrics like AUC)
5. **Interpretable** - Visual null distributions aid understanding
6. **Suitable for small samples** - More reliable than parametric tests

### ⚠️ Limitations

1. **Computationally intensive** - 10,000 permutations per test
2. **Assumes exchangeability** - Under null hypothesis, group labels are exchangeable
3. **Discrete p-values** - Minimum achievable p-value = 1/n_perm (e.g., 0.0001 for 10,000)
4. **Multiple testing** - Requires correction when testing multiple metrics (FDR, Bonferroni)

---

## BEST PRACTICES

### 1. Number of Permutations

**Recommendation**: 10,000 permutations
- Minimum: 1,000 (for exploratory analysis)
- Standard: 10,000 (balances precision and computation)
- High-precision: 100,000 (for p-values near α=0.05)

**Rule of Thumb**:
```
n_perm ≥ 1 / (α / 2)

For α=0.05 (two-tailed):
n_perm ≥ 1 / 0.025 = 40 permutations (minimum)
n_perm = 10,000 gives precision of 0.0001
```

### 2. Random Seed

**Always set a seed for reproducibility**:
```r
set.seed(42)  # Or any consistent value
```

### 3. Multiple Testing Correction

When testing multiple metrics or multiple groups, apply correction:

```r
# Benjamini-Hochberg FDR correction
p_values <- c(0.432, 0.089, 0.156, 0.782)  # From different tests
p_adjusted <- p.adjust(p_values, method = "BH")

# Bonferroni (more conservative)
p_adjusted_bonf <- p.adjust(p_values, method = "bonferroni")
```

### 4. Report Comprehensively

Always report:
- ✅ Observed difference
- ✅ P-value
- ✅ Number of permutations
- ✅ Effect size (Cohen's d)
- ✅ Confidence intervals (for AUC)
- ✅ Sample sizes for both groups

---

## EXAMPLE WORKFLOWS

### Workflow 1: Single Metric Test (AUC)

```r
# Load data
source("utils_statistical_tests.R")

# Prepare groups
labels_a <- female_data$label
pred_a <- female_data$pred
labels_b <- male_data$label
pred_b <- male_data$pred

# Run test
result <- permutation_test_auc(
  labels_a, pred_a,
  labels_b, pred_b,
  n_perm = 10000,
  seed = 42
)

# Report
cat("AUC Female:", sprintf("%.4f", result$auc_a), "\n")
cat("AUC Male:", sprintf("%.4f", result$auc_b), "\n")
cat("Difference:", sprintf("%.4f", result$observed_diff), "\n")
cat("P-value:", sprintf("%.4f", result$p_value), "\n")

# Visualize null distribution
hist(result$perm_diffs, breaks = 50, col = "lightblue",
     main = "Null Distribution: AUC Difference",
     xlab = "Permuted AUC Difference")
abline(v = result$observed_diff, col = "red", lwd = 2)
```

---

### Workflow 2: Comprehensive Multi-Metric Test ⭐

```r
# Load utilities
source("utils_statistical_tests.R")

# Prepare data (must have columns: label, pred)
data_a <- data.frame(
  label = female_labels,
  pred = female_predictions
)

data_b <- data.frame(
  label = male_labels,
  pred = male_predictions
)

# Run comprehensive test
results <- compare_all_metrics_comprehensive(
  group_a_data = data_a,
  group_b_data = data_b,
  group_a_name = "Female",
  group_b_name = "Male",
  n_perm = 10000,
  n_boot = 10000,
  threshold = 0.5,
  seed = 42
)

# Access results
summary_table <- results$summary_table
print(summary_table)

# Save results
write_csv(summary_table, "comprehensive_fairness_results.csv")

# Check for significant differences
if (any(summary_table$Significant)) {
  cat("⚠️ WARNING: Significant differences detected!\n")
  sig_metrics <- summary_table$Metric[summary_table$Significant]
  cat("Significant metrics:", paste(sig_metrics, collapse = ", "), "\n")
} else {
  cat("✓ No significant differences detected (all p>0.05)\n")
}

# Plot significant results
library(ggplot2)

plot_df <- summary_table %>%
  mutate(Metric = factor(Metric, levels = rev(Metric)))

ggplot(plot_df, aes(x = P_Value, y = Metric, color = Significant)) +
  geom_point(size = 4) +
  geom_vline(xintercept = 0.05, linetype = "dashed", color = "red") +
  scale_color_manual(values = c("TRUE" = "red", "FALSE" = "darkgreen")) +
  labs(title = "Permutation Test P-Values: Female vs Male",
       x = "P-Value",
       y = "Metric",
       caption = "Dashed line: α=0.05 significance threshold") +
  theme_minimal()
```

---

### Workflow 3: Multiple Group Comparisons with Correction

```r
# Compare multiple demographic groups
groups <- c("White", "Black", "Other")
comparisons <- combn(groups, 2, simplify = FALSE)

all_results <- list()
p_values_auc <- c()

for (comp in comparisons) {
  group_a <- comp[1]
  group_b <- comp[2]

  data_a <- analysis_data %>%
    filter(RACE == group_a) %>%
    select(label = true_label, pred = predicted_prob)

  data_b <- analysis_data %>%
    filter(RACE == group_b) %>%
    select(label = true_label, pred = predicted_prob)

  result <- compare_all_metrics_comprehensive(
    data_a, data_b,
    group_a_name = group_a,
    group_b_name = group_b,
    n_perm = 10000
  )

  all_results[[paste(group_a, "vs", group_b)]] <- result
  p_values_auc <- c(p_values_auc, result$summary_table$P_Value[1])  # AUC p-value
}

# Apply FDR correction
p_adjusted <- p.adjust(p_values_auc, method = "BH")

# Report
comparison_summary <- data.frame(
  Comparison = names(all_results),
  P_Value_Raw = p_values_auc,
  P_Value_Adjusted = p_adjusted,
  Significant_Raw = p_values_auc < 0.05,
  Significant_FDR = p_adjusted < 0.05
)

print(comparison_summary)
```

---

## TROUBLESHOOTING

### Problem 1: NA values in permutation distribution

**Symptom**:
```
Warning: More than 10% of permutations failed. Results may be unreliable.
```

**Cause**: Insufficient positive or negative cases in a subgroup

**Solution**:
- Ensure minimum sample size (n ≥ 30 per group)
- Check for sufficient class balance (at least 10 cases per class)
- Consider combining small subgroups

---

### Problem 2: P-value = 0

**Symptom**:
```
p_value: 0.0000
```

**Explanation**: Observed difference is more extreme than all 10,000 permutations

**Interpretation**: p < 0.0001 (report as p < 0.0001, not p = 0)

**Solution**: Increase permutations for more precision (e.g., 100,000)

---

### Problem 3: All p-values = 1.0

**Symptom**: Every test returns p = 1.0

**Cause**: Possible bug in permutation logic or identical groups

**Debugging**:
```r
# Check observed difference
cat("Observed difference:", result$observed_diff, "\n")

# Check permutation distribution
summary(result$perm_diffs)
hist(result$perm_diffs)

# Check if groups are different
cat("Group A mean:", mean(pred_a), "\n")
cat("Group B mean:", mean(pred_b), "\n")
```

---

## REFERENCES

### Key Papers

1. **Yeh, A. (2000)**. "More Accurate Tests for the Statistical Significance of Result Differences."
   *Proceedings of COLING 2000*, pp. 947-953.
   - **Application**: Gold standard for NLP model comparison
   - **Our use**: F1-score comparison methodology

2. **Edgington, E. S., & Onghena, P. (2007)**. *Randomization Tests (4th ed.)*.
   Chapman & Hall/CRC.
   - **Application**: Comprehensive permutation test theory
   - **Our use**: General permutation testing framework

3. **Good, P. I. (2013)**. *Permutation Tests: A Practical Guide to Resampling Methods for Testing Hypotheses (2nd ed.)*.
   Springer.
   - **Application**: Practical implementation guide
   - **Our use**: Best practices and troubleshooting

4. **Ojala, M., & Garriga, G. C. (2010)**. "Permutation Tests for Studying Classifier Performance."
   *Journal of Machine Learning Research*, 11, 1833-1863.
   - **Application**: Machine learning model fairness testing
   - **Our use**: Classifier performance comparison methodology

### Fairness Literature

5. **Obermeyer, Z., et al. (2019)**. "Dissecting racial bias in an algorithm used to manage the health of populations."
   *Science*, 366(6464), 447-453.
   - **Application**: Algorithmic fairness in healthcare AI
   - **Our use**: Fairness criteria and thresholds

6. **Rajkomar, A., et al. (2018)**. "Ensuring Fairness in Machine Learning to Advance Health Equity."
   *Annals of Internal Medicine*, 169(12), 866-872.
   - **Application**: Health equity in AI systems
   - **Our use**: Framework for demographic fairness evaluation

---

## CONTACT & SUPPORT

**Author**: Frederick Gyasi
**Email**: [your email]
**Project**: ADRD ePhenotyping
**GitHub**: [repository link]

**For questions about:**
- Theoretical foundations → See References section
- Implementation details → See `utils_statistical_tests.R` comments
- Integration → See `CODE_MAPPING_TO_PROPOSAL.md`
- Bugs/Issues → GitHub Issues

---

## VERSION HISTORY

### Version 2.2 (December 16, 2025)
- ✨ Added `compare_all_metrics_comprehensive()` - Tests all 8 metrics
- ✨ Added metric calculation functions (accuracy, sensitivity, etc.)
- ✨ Enhanced documentation with comprehensive examples
- 🔧 Maintained backward compatibility with `compare_groups_comprehensive()`

### Version 2.1 (November 25, 2025)
- Added demographic-stratified TF-IDF analysis
- Added null distribution visualizations
- Enhanced reporting and documentation

### Version 2.0 (November 15, 2025)
- Initial implementation of approximate randomization
- Basic permutation test for AUC
- Bootstrap confidence intervals
- Effect size calculations

---

## APPENDIX: QUICK COMMAND REFERENCE

```r
# Load utilities
source("utils_statistical_tests.R")

# Single metric test (AUC)
permutation_test_auc(labels_a, pred_a, labels_b, pred_b, n_perm=10000)

# Generic metric test
permutation_test_metric(metric_a, metric_b, labels_a, pred_a, labels_b, pred_b,
                        metric_function, n_perm=10000)

# Comprehensive test (ALL 8 metrics) ⭐
compare_all_metrics_comprehensive(data_a, data_b, "GroupA", "GroupB",
                                  n_perm=10000, n_boot=10000)

# Bootstrap CI
bootstrap_auc_ci(labels, predictions, n_boot=10000)

# Effect size
cohens_d(group_a_values, group_b_values)

# FDR correction
apply_fdr_correction(p_values, method="BH", alpha=0.05)
```

---

**Document Status**: Complete and ready for use
**Last Updated**: December 16, 2025
