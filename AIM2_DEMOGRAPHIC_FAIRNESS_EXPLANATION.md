# Aim 2 Demographic Feature Fairness Analysis - Complete Explanation

**Author**: Gyasi, Frederick
**Version**: 2.0
**Purpose**: Extend demographic fairness from model performance (Aim 1) to feature analysis (Aim 2)

---

## 🎯 Why This Is Critical

### The Gap We're Addressing

**Aim 1** (04_demographic_analysis.R) answers:
> *"Does the model perform equitably across demographics?"*
> - Metrics: AUC, Sensitivity, Specificity, F1 by gender/race/ethnicity
> - Statistical tests: Chi-squared tests for performance differences

**Aim 2** (05_aim2_feature_analysis.R) MUST ALSO answer:
> *"Do the FEATURES driving predictions differ by demographics?"*
> - Which words are discriminative for Female vs Male ADRD patients?
> - Do LIME explanations rely on different features for different races?
> - Is behavioral testing sensitivity different by demographics?

### Why This Matters for Fairness

If the model uses **different linguistic features** to diagnose ADRD in different demographic groups, this indicates:

1. **Differential Clinical Documentation**
   - Physicians may describe symptoms differently for different demographics
   - Documentation bias: "aggressive" vs "assertive" behavior descriptions
   - Example: Memory complaints documented differently by patient gender

2. **Stereotypical Feature Associations**
   - Model learns demographic-specific patterns that may not generalize
   - Example: "wandering" associated more with male patients, "forgetful" with female

3. **Hidden Demographic Bias**
   - Predictions rely on features correlated with protected attributes
   - Violates fairness principles even if overall performance is equal

4. **Unequal Explainability**
   - Model decisions interpretable for majority group but opaque for minorities
   - Limits clinical trust and adoption in diverse settings

---

## 📋 What We Will Implement

### **Enhancement 1: Demographic-Stratified Chi-Squared Analysis**

**Current Behavior**:
```
Overall ADRD vs Control chi-squared:
  Top discriminative terms: dementia, alzheimer, memory, cognitive, ...
```

**NEW Behavior**:
```
Female ADRD vs Female Control:
  Top terms: memory (χ²=450), forgetful (χ²=380), confusion (χ²=350), ...

Male ADRD vs Male Control:
  Top terms: dementia (χ²=520), alzheimer (χ²=410), wandering (χ²=320), ...

COMPARISON:
  Common terms (top 10): 6/10 (60% overlap)
  Unique to Female: forgetful, misplacing
  Unique to Male: wandering, agitation

  ⚠️  FINDING: Different linguistic patterns suggest differential documentation
```

**Implementation**:
- For each demographic subgroup (Gender, Race, Ethnicity):
  - Filter corpus to that subgroup
  - Run chi-squared ADRD vs Control within subgroup
  - Get top 20 discriminative terms
  - Compare term overlap across subgroups
  - Calculate statistical significance of differences

**Clinical Interpretation**:
- **High overlap (>70%)**: Consistent terminology → Good generalization
- **Moderate overlap (40-70%)**: Some differences → Monitor for bias
- **Low overlap (<40%)**: Different features → Potential documentation bias

---

### **Enhancement 2: Demographic-Stratified LIME Analysis**

**Current Behavior**:
```
LIME explanations for 20 sample cases:
  Case 1: Important words: memory, cognitive, decline
  Case 2: Important words: dementia, alzheimer, forget
  ...
```

**NEW Behavior**:
```
Female ADRD Predictions (N=10 cases):
  Aggregate top features: memory (mean=0.45), cognitive (mean=0.38), confusion (mean=0.32)
  Feature overlap with overall: 8/10 (80%)

Male ADRD Predictions (N=10 cases):
  Aggregate top features: dementia (mean=0.52), alzheimer (mean=0.41), decline (mean=0.35)
  Feature overlap with overall: 7/10 (70%)

COMPARISON:
  Common features across genders: 5/10 (50%)

  ⚠️  FINDING: Model uses different features for gender subgroups
              May indicate differential prediction logic
```

**Implementation**:
- For each demographic subgroup:
  - Select 10 confident ADRD predictions from that subgroup
  - Generate LIME explanations for those cases
  - Aggregate feature importance (mean, median weights)
  - Compare top features across subgroups
  - Statistical test for importance score differences

**Clinical Interpretation**:
- **High overlap**: Model uses same logic across demographics → Fair explanations
- **Low overlap**: Model explanations differ by demographics → Potential bias concern

---

### **Enhancement 3: Demographic-Stratified Behavioral Testing**

**Current Behavior**:
```
Demonstration: Remove "alzheimer" from one ADRD case
  Original prediction: 0.85
  Modified prediction: 0.72
  Δ prediction: -0.13
```

**NEW Behavior**:
```
Remove "alzheimer" from all ADRD cases:

Female patients (N=50):
  Mean Δ prediction: -0.08 (SD=0.05)
  Sensitivity: LOW (predictions robust to term removal)

Male patients (N=50):
  Mean Δ prediction: -0.18 (SD=0.07)
  Sensitivity: HIGH (predictions fragile to term removal)

Statistical Test:
  Two-sample t-test: p = 0.003
  Result: Male predictions significantly more sensitive to "alzheimer" removal

  ⚠️  FINDING: Differential reliance on specific terms by gender
              May indicate over-reliance on stereotypical features for males
```

**Implementation**:
- For each discriminative term:
  - For each demographic subgroup:
    - Remove term from all ADRD cases in that subgroup
    - Measure prediction changes (Δ)
    - Calculate mean, median, SD of Δ
  - Compare sensitivity across subgroups with t-test or ANOVA
  - Identify terms with differential sensitivity

**Clinical Interpretation**:
- **Equal sensitivity**: Term equally important across demographics → Fair
- **Differential sensitivity**: Term more important for one group → Potential over-reliance

---

## 🔧 Implementation Details

### File Structure

**Main Changes to**: `05_aim2_feature_analysis.R`

```
CURRENT STRUCTURE:
├── PART 1: Corpus Analysis (chi-squared overall)           [EXISTS]
├── PART 2: TF-IDF Analysis (overall)                       [EXISTS]
├── PART 3-5: Visualizations                                [EXISTS]
├── PART 6B: LIME Explainability                            [EXISTS]
└── PART 7: Behavioral Testing                              [EXISTS]

NEW ADDITIONS:
├── PART 1B: Demographic-Stratified Chi-Squared Analysis    [NEW]
│   ├── For each demographic variable (Gender, Race, Ethnicity)
│   ├── Chi-squared within each subgroup
│   ├── Compare top terms across subgroups
│   └── Statistical testing of differences
│
├── PART 6C: Demographic-Stratified LIME Analysis           [NEW]
│   ├── LIME for each demographic subgroup
│   ├── Aggregate feature importance by subgroup
│   ├── Compare features across subgroups
│   └── Overlap analysis
│
├── PART 7B: Demographic-Stratified Behavioral Testing      [NEW]
│   ├── Term removal for each demographic
│   ├── Measure differential sensitivity
│   └── Statistical comparison (t-test, ANOVA)
│
└── PART 8: Demographic Feature Fairness Summary            [NEW]
    ├── Summary of findings
    ├── Fairness assessment
    └── Recommendations
```

### Code Placement

1. **PART 1B** - Insert after PART 1 (around line 260)
2. **PART 6C** - Insert after PART 6B LIME (around line 620)
3. **PART 7B** - Insert after PART 7 Behavioral Testing (around line 800)
4. **PART 8** - Insert at end before final completion message

### Prerequisites

**Data Requirements**:
- Demographics must be in `full_corpus` (merged from test_set)
- Variables: `GENDER`, `RACE`, `HISPANIC`
- Minimum N per subgroup: 20 ADRD + 20 Control for chi-squared
- Minimum N per subgroup: 5 ADRD cases for LIME

**Software Requirements**:
- All existing packages (quanteda, lime)
- No new package dependencies

---

## 📊 Statistical Framework

### 1. Chi-Squared Homogeneity Test

**Question**: Are discriminative term distributions significantly different across demographic groups?

**Method**:
```r
# For each of top 20 terms from overall analysis:
# Create contingency table: Subgroup × Term Presence
# H0: Term distribution homogeneous across subgroups
# Ha: Term distribution differs by subgroup

chisq.test(table(demographic_group, term_present))
```

**Interpretation**:
- p < 0.05: Term usage significantly differs by demographics → Feature bias concern
- p ≥ 0.05: Term usage consistent across demographics → Fair

### 2. Feature Overlap Analysis

**Metric**: Jaccard similarity
```
Overlap = |Top Terms A ∩ Top Terms B| / |Top Terms A ∪ Top Terms B|
```

**Thresholds**:
- Overlap > 0.7: High agreement → Good fairness
- Overlap 0.4-0.7: Moderate → Monitor
- Overlap < 0.4: Low → Bias concern

### 3. LIME Importance Score Comparison

**Method**: Two-sample t-test on feature weights
```r
# For features appearing in both subgroups:
t.test(weights_subgroup_A, weights_subgroup_B)
```

**Interpretation**:
- Significant difference (p < 0.05): Feature importance differs → Investigate why
- Non-significant: Consistent importance → Fair

### 4. Behavioral Testing Sensitivity

**Method**: ANOVA or Kruskal-Wallis test
```r
# Dependent variable: Δ prediction after term removal
# Independent variable: Demographic group
anova(lm(delta_prediction ~ demographic_group))
```

**Interpretation**:
- Significant effect: Differential sensitivity → Over-reliance concern
- Non-significant: Equal sensitivity → Fair

---

## 📈 Expected Outputs

### Files Generated

**Data Files**:
```
results/aim2/
├── demographic_chi2_stratified.rds          # Chi-squared results by demographics
├── demographic_chi2_comparison.csv          # Top terms comparison table
├── demographic_lime_stratified.rds          # LIME results by demographics
├── demographic_lime_comparison.csv          # Feature importance comparison
├── demographic_behavioral_sensitivity.rds   # Term removal sensitivity
└── demographic_behavioral_comparison.csv    # Sensitivity comparison table
```

**Visualizations** (if Phase 2 implemented):
```
figures/aim2/
├── chi2_terms_by_gender_heatmap.png        # Heatmap of term importance by gender
├── chi2_terms_by_race_heatmap.png          # Heatmap of term importance by race
├── lime_features_by_demographics.png       # Feature importance comparison plot
└── behavioral_sensitivity_by_demo.png      # Sensitivity comparison boxplots
```

### Console Output Example

```
================================================================================
PART 1B: Demographic-Stratified Chi-Squared Analysis
================================================================================

Investigating if discriminative features differ by demographics...

  Found demographic variable: GENDER
  Found demographic variable: RACE

Demographic variables available: GENDER, RACE

--------------------------------------------------------------------------------
Analyzing: GENDER
--------------------------------------------------------------------------------

Subgroups: Female, Male

  Analyzing Female (N = 523 : ADRD= 261 , CTRL= 262 )
    Top 5 discriminative terms: memory, cognitive, confusion, forgetful, recall
  Analyzing Male (N = 477 : ADRD= 238 , CTRL= 239 )
    Top 5 discriminative terms: dementia, alzheimer, decline, wandering, disorientation

  Comparing discriminative terms across GENDER subgroups:
    Common terms (top 10): 6 / 10
      cognitive, decline, memory, confusion, forget, impair
    Unique to Female: forgetful, recall, misplacing
    Unique to Male: wandering, agitation, alzheimer

    ⚠️  FINDING: Low term overlap (60%) suggests different linguistic patterns by GENDER

Demographic-stratified chi-squared results saved
```

---

## ⚙️ How to Integrate the Code

### Option 1: Manual Integration (Recommended for understanding)

1. **Open** `05_aim2_feature_analysis.R`
2. **Find** line ~260 (after PART 1 chi-squared analysis completes)
3. **Copy** PART 1B code from `05_aim2_DEMOGRAPHIC_ENHANCEMENTS.R`
4. **Paste** into `05_aim2_feature_analysis.R`
5. **Repeat** for PART 6C (after line ~620) and PART 8 (at end)

### Option 2: Source the Enhancement File (Quick testing)

Add at top of `05_aim2_feature_analysis.R`:
```r
# Load demographic fairness enhancements
if (file.exists("05_aim2_DEMOGRAPHIC_ENHANCEMENTS.R")) {
  source("05_aim2_DEMOGRAPHIC_ENHANCEMENTS.R")
}
```

**Note**: Option 1 is better for production; Option 2 for quick testing.

---

## ✅ Validation Checklist

After running enhanced `05_aim2_feature_analysis.R`:

- [ ] **Console shows demographic detection**:
      `Found demographic variable: GENDER, RACE`

- [ ] **Chi-squared stratified analysis runs**:
      Reports top terms for each demographic subgroup

- [ ] **Term overlap calculated**:
      Shows common terms and unique terms per subgroup

- [ ] **Files created**:
      - `demographic_chi2_stratified.rds`
      - `demographic_chi2_comparison.csv`

- [ ] **LIME stratified analysis runs** (if demographics available):
      Feature importance by subgroup calculated

- [ ] **Summary report generated**:
      PART 8 shows fairness assessment

---

## 🎓 Scientific Interpretation Guide

### Scenario 1: High Feature Overlap (>70%)

**Finding**:
```
Gender comparison:
  Common terms (top 10): 8/10 (80%)
  LIME feature overlap: 9/10 (90%)
```

**Interpretation**:
✅ **GOOD**: Model uses consistent features across demographics
- Indicates robust, generalizable patterns
- Low risk of demographic bias in features
- Clinical terminology consistent across groups

**Action**: No immediate concern, continue monitoring

---

### Scenario 2: Moderate Feature Overlap (40-70%)

**Finding**:
```
Race comparison:
  Common terms (top 10): 5/10 (50%)
  Unique to Black patients: hypertension, diabetes
  Unique to White patients: memory, cognitive
```

**Interpretation**:
⚠️ **CAUTION**: Some feature differences detected
- May reflect comorbidity patterns (hypertension in Black patients)
- May reflect differential clinical documentation
- Needs clinical expert review

**Action**:
1. Review unique terms with domain experts
2. Investigate if differences are clinically meaningful
3. Consider demographic-aware model calibration

---

### Scenario 3: Low Feature Overlap (<40%)

**Finding**:
```
Ethnicity comparison:
  Common terms (top 10): 3/10 (30%)
  Hispanic LIME features: español, familia, cultura
  Non-Hispanic LIME features: dementia, alzheimer, memory

  ⚠️  FINDING: Low overlap suggests different linguistic patterns
```

**Interpretation**:
🚨 **CONCERN**: Substantial feature differences
- Language differences (Spanish terms in Hispanic notes)
- Documentation bias (different symptom descriptions)
- Model may be less reliable for minority group

**Action**:
1. **Investigate root cause**: Documentation language? Cultural differences?
2. **Consider mitigation**:
   - Multilingual preprocessing
   - Demographic-specific models
   - Augment training data for minority groups
3. **Report limitation**: Note in publications/deployments

---

### Scenario 4: Differential Behavioral Sensitivity

**Finding**:
```
Remove "alzheimer":
  Female: Δ = -0.08 (robust)
  Male: Δ = -0.18 (sensitive)
  p = 0.003 (significant)
```

**Interpretation**:
⚠️ **CONCERN**: Model over-relies on "alzheimer" for male patients
- Male predictions more fragile to specific term removal
- May indicate stereotypical association
- Reduces robustness for male patients

**Action**:
1. Investigate why term more important for males
2. Check documentation patterns by gender
3. Consider ensemble with demographic-agnostic features

---

## 🔬 Research Implications

### Publication Value

This analysis enables you to answer critical research questions:

1. **Feature Fairness**:
   > "Do CNN models learn different linguistic patterns for diagnosing ADRD across demographic groups?"

2. **Explainability Equity**:
   > "Are model predictions equally interpretable (via LIME) for all demographics?"

3. **Robustness Fairness**:
   > "Is model sensitivity to term removal equal across demographic groups?"

4. **Documentation Bias Detection**:
   > "Do clinical notes contain systematic linguistic differences by patient demographics?"

### Regulatory Compliance

This analysis supports:
- **FDA AI/ML Guidance**: Demonstrate fairness across subgroups
- **EU AI Act**: Assess bias in high-risk medical AI
- **NIH INCLUDE Project**: Ensure inclusive research practices

### Clinical Deployment

Informs:
- **Which patient groups** need additional model validation
- **Which features** may be biased and need review
- **Whether separate models** are needed for demographic subgroups
- **Training data augmentation** needs for underrepresented groups

---

## 📌 Summary

### What This Code Does

1. ✅ **Stratifies all feature analyses by demographics**
   - Chi-squared discriminative terms
   - LIME feature importance
   - Behavioral testing sensitivity

2. ✅ **Compares features across demographic subgroups**
   - Calculates overlap/uniqueness
   - Statistical significance testing
   - Clinical interpretation

3. ✅ **Identifies feature-level bias**
   - Different linguistic patterns
   - Differential reliance on specific terms
   - Unequal explainability

4. ✅ **Provides actionable insights**
   - Which demographics show feature differences
   - Which terms are group-specific
   - Whether model is fair at feature level

### Integration with Aim 1

| Aim 1 (Performance Fairness) | Aim 2 (Feature Fairness) |
|-------------------------------|--------------------------|
| AUC differs by gender? | Top terms differ by gender? |
| Sensitivity differs by race? | LIME features differ by race? |
| Statistical test: χ² on outcomes | Statistical test: χ² on features |
| Interpretation: Performance bias | Interpretation: Feature bias |
| **Complementary analyses for comprehensive fairness assessment** |

---

## 🚀 Ready to Implement

**All code is ready in**: `05_aim2_DEMOGRAPHIC_ENHANCEMENTS.R`

**Steps**:
1. Review this explanation document
2. Integrate code sections into `05_aim2_feature_analysis.R`
3. Ensure demographics are in test_set/full_corpus
4. Run analysis
5. Review console output for fairness findings
6. Examine output CSV files for detailed comparisons

**Questions this answers**:
- ✅ Do features differ by demographics? → YES, via chi-squared stratification
- ✅ Do explanations differ by demographics? → YES, via LIME stratification
- ✅ Is sensitivity equal across demographics? → YES, via behavioral testing stratification
- ✅ Is this publication-ready? → YES, rigorous statistical framework

**Your Aim 2 feature fairness analysis will be complete!** 🎉
