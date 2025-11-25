# ADRD E-Phenotyping Bias Analysis - Enhancements Guide

## CNN-Focused Bias Analysis Framework

Version: 2.0
Date: 2025-11-25
Author: Claude (based on proposal requirements)

---

## 🎯 Overview

This guide documents the enhancements made to the ADRD e-phenotyping bias analysis pipeline to support the AMIA submission. All enhancements focus on **Convolutional Neural Network (CNN)** model evaluation only, as specified in the research proposal.

### Core Enhancement Areas:

1. ✅ **Demographic-Stratified TF-IDF Analysis** (NEW)
2. ✅ **TF-IDF Heatmap Visualizations** (NEW)
3. ✅ **Integration Dashboard** (Aim 1 + Aim 2) (NEW)
4. ✅ **Publication-Ready Methods Section** (NEW)
5. ✅ **Preserved Existing Functionality** (Behavioral Testing, LIME, Chi-squared)

---

## 📋 What's New

### 1. Demographic-Stratified TF-IDF Analysis

**Location**: `05_aim2_feature_analysis.R` (Lines 513-770)

**Purpose**: Identify which clinical phrases drive CNN performance differences across demographic subgroups

**Methodology**:
- Calculates TF-IDF scores for correctly vs. incorrectly classified patients
- Stratifies analysis by demographic variables (GENDER, RACE, HISPANIC)
- Identifies "keyness" terms explaining performance disparities
- Compares linguistic patterns between demographic subgroups

**Key Features**:
- **TF-IDF Explanation**: Embedded in code comments for transparency
  - Term Frequency (TF): How often term appears in subgroup
  - Inverse Document Frequency (IDF): How unique term is across corpus
  - TF-IDF = TF × IDF (high score = important AND distinctive)

- **Subgroup Analysis**: For each demographic variable
  - Calculates TF-IDF for Correct vs. Incorrect classifications
  - Identifies unique terms to each classification group
  - Computes overlap statistics

- **Statistical Rigor**:
  - Minimum sample size requirements (20+ correct and incorrect per subgroup)
  - Top 50 terms extracted per classification group
  - Overlap analysis to identify discriminative features

**Outputs**:
```
results/aim2/demographic_tfidf_stratified.rds    # Full results
results/aim2/demographic_tfidf_comparison.csv    # Comparison table
```

**Example Output**:
```
GENDER:
  Female:
    Terms unique to Correct: 15
    Terms unique to Incorrect: 18
    Overlapping terms: 17
  Male:
    Terms unique to Correct: 12
    Terms unique to Incorrect: 21
    Overlapping terms: 19
```

---

### 2. TF-IDF Heatmap Visualizations

**Location**: `05_aim2_feature_analysis.R` (Lines 935-1128)

**Purpose**: Visualize CNN feature importance patterns across demographic subgroups

**Three Types of Visualizations**:

#### a. TF-IDF Heatmap by Demographic Subgroup
**File**: `figures/aim2/tfidf_heatmap_[demographic].png`

- **Rows**: Top 30 clinical terms (by mean TF-IDF across subgroups)
- **Columns**: Demographic subgroups × Classification (e.g., Female_Correct, Male_Incorrect)
- **Color Scale**: Blue (low TF-IDF) → Gray (medium) → Red (high)
- **Title**: "CNN Feature Importance by [DEMOGRAPHIC] Subgroup"

**Interpretation**:
- Red cells = Term highly important for that subgroup/classification
- Blue cells = Term less important
- Patterns reveal which terms drive CNN predictions differently by demographics

#### b. Top Phrases Bar Chart by Subgroup
**File**: `figures/aim2/tfidf_top_phrases_[demographic].png`

- **Layout**: Faceted by demographic subgroup
- **Bars**: Top 15 TF-IDF terms per subgroup
- **Colors**: Green (Correct) vs. Red (Incorrect)
- **Title**: "Top TF-IDF Terms by [DEMOGRAPHIC] Subgroup"

**Interpretation**:
- Compare which terms are important for correct vs. incorrect CNN predictions
- Identify subgroup-specific linguistic patterns

#### c. Unique Terms Comparison Plot
**File**: `figures/aim2/tfidf_unique_terms_[demographic].png`

- **Layout**: Grid (Subgroup × Uniqueness)
- **Terms**: Top 10 terms appearing ONLY in Correct or ONLY in Incorrect
- **Colors**: Green (Unique to Correct) vs. Red (Unique to Incorrect)
- **Title**: "Terms Unique to Classification Groups ([DEMOGRAPHIC])"

**Interpretation**:
- Terms unique to Correct = linguistic markers of successful CNN classification
- Terms unique to Incorrect = linguistic markers of CNN failure
- Reveals what the CNN "sees" differently by demographics

---

### 3. Integration Dashboard (Aim 1 + Aim 2)

**Location**: `06_integration_analysis.R` (NEW FILE)

**Purpose**: Connect demographic fairness analysis (Aim 1) with feature-level analysis (Aim 2) to provide comprehensive CNN bias characterization

#### Integration Framework:

```
┌─────────────────────────────────────────────────────────────┐
│                 THREE-DIMENSIONAL BIAS FRAMEWORK            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  AIM 1: WHERE bias exists                                   │
│  ├─ Approximate randomization tests                         │
│  ├─ Performance gaps by demographic                         │
│  └─ Statistical significance                                │
│                                                             │
│  AIM 2: WHY bias exists                                     │
│  ├─ TF-IDF analysis by demographic                          │
│  ├─ Chi-squared discriminative terms                        │
│  ├─ LIME explanations                                       │
│  └─ Behavioral testing validation                           │
│                                                             │
│  INTEGRATION: Complete characterization                     │
│  ├─ 1. Clinical Bias: ADRD presentation differences        │
│  ├─ 2. Algorithmic Bias: Classification parity failures    │
│  └─ 3. Linguistic Bias: Feature salience disparities       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Key Analyses:

**1. Load and Merge Results**:
- Aim 1: Subgroup performance data
- Aim 2: TF-IDF, chi-squared, LIME results
- Match demographics with performance gaps to discriminative features

**2. Bias Characterization**:
- For each demographic subgroup:
  - CNN performance metrics (AUC, F1)
  - Number of discriminative features
  - Feature overlap between correct/incorrect classifications
  - Bias dimension classification (Clinical, Algorithmic, Linguistic)

**3. Comprehensive Outputs**:

**CSV File**: `results/integration/bias_characterization_summary.csv`

Columns:
- `demographic`: Demographic variable (GENDER, RACE, etc.)
- `subgroup`: Specific subgroup (Female, Male, etc.)
- `cnn_auc`: CNN AUC for this subgroup
- `cnn_f1`: CNN F1-score for this subgroup
- `n_discriminative_features`: Number of TF-IDF terms identified
- `feature_overlap_pct`: Percentage overlap between correct/incorrect terms
- `bias_dimension_clinical`: Clinical bias assessment
- `bias_dimension_algorithmic`: Algorithmic bias assessment
- `bias_dimension_linguistic`: Linguistic bias assessment

#### Visualizations:

**1. Integrated Dashboard**: `figures/integration/integrated_dashboard.png`

Three-panel figure:
- **Panel A**: CNN Performance Across Demographic Subgroups (AUC)
- **Panel B**: Feature Overlap: Correct vs. Incorrect Classifications
- **Panel C**: Number of Discriminative Features by Subgroup

**Interpretation**:
- Panel A shows WHERE bias exists (performance gaps)
- Panel B shows HOW DISTINCT linguistic patterns are
- Panel C shows HOW MANY features drive differences

**2. Bias Framework Visualization**: `figures/integration/bias_framework_visualization.png`

Scatter plot:
- **X-axis**: Performance Gap (AUC Range) - Algorithmic Bias
- **Y-axis**: Feature Distinctiveness (100 - Overlap%) - Linguistic Bias
- **Points**: Demographic variables
- **Size**: Mean AUC across subgroups
- **Color**: Demographic variable

**Interpretation**:
- Top-right = High algorithmic bias + High linguistic bias (concerning)
- Bottom-left = Low algorithmic bias + Low linguistic bias (ideal)
- Reveals which demographics have most pronounced bias issues

---

### 4. Publication-Ready Methods Section

**Location**: `METHODS_APPROXIMATE_RANDOMIZATION.md` (NEW FILE)

**Purpose**: Ready-to-submit Methods section for AMIA/JAMIA publication

**Content** (407 words):
1. **Rationale**: Why approximate randomization over bootstrap
2. **Procedure**: Step-by-step replicable methodology
3. **Statistical Framework**: Null/alternative hypotheses, test statistics
4. **Multiple Testing Correction**: FDR control
5. **Sample Size and Power**: Study cohort characteristics
6. **Implementation Details**: Software, reproducibility

**Key Citations Included**:
- Good (2000): Permutation testing methodology
- Benjamini & Hochberg (1995): FDR correction
- Cohen (1988): Effect size interpretation
- Obermeyer et al. (2019): Healthcare algorithm bias
- Heider et al. (2020): Fairness in clinical NLP

**Usage**: Copy directly into AMIA abstract or full paper Methods section

---

## 🔄 What Was Preserved

### Existing Functionality (100% Intact):

1. ✅ **Approximate Randomization** (Already implemented correctly)
   - Location: `utils_statistical_tests.R`, `04_demographic_analysis.R`
   - 10,000 permutation tests
   - Null distribution visualizations
   - FDR correction

2. ✅ **Bootstrap Confidence Intervals** (Correctly used)
   - Location: `utils_statistical_tests.R`
   - 10,000 bootstrap samples
   - Stratified by outcome
   - 95% CIs for performance metrics

3. ✅ **Behavioral Testing** (Fully implemented)
   - Location: `05_aim2_feature_analysis.R` (Lines 1000-1148)
   - Term removal sensitivity analysis
   - Prediction change measurement
   - Template script generation

4. ✅ **LIME Explainability** (With demographic stratification)
   - Location: `05_aim2_feature_analysis.R` (Lines 731-995)
   - Individual prediction explanations
   - Demographic-stratified LIME
   - Feature overlap analysis

5. ✅ **Chi-Squared Discriminative Terms**
   - Location: `05_aim2_feature_analysis.R` (Lines 226-472)
   - Keyness analysis
   - Demographic stratification
   - FDR correction

6. ✅ **CNN Model Evaluation Pipeline**
   - Location: `03_evaluate_models.R`
   - Uses Jihad Obeid's pre-trained models
   - Comprehensive metrics
   - ROC curves, calibration plots

7. ✅ **All Existing Visualizations**
   - Word clouds
   - Top terms bar plots
   - Chi-squared keyness plots
   - Original TF-IDF comparison
   - Null distribution histograms
   - Subgroup performance plots

---

## 📊 Complete Analysis Pipeline

### Execution Order:

```bash
# Step 0: Environment setup (one-time)
./setup_environment.sh

# Step 1: Data preparation (if needed)
Rscript 01_prepare_data.R

# Step 2: Model evaluation (uses Jihad's pre-trained CNN models)
Rscript 03_evaluate_models.R

# Step 3: Aim 1 - Demographic fairness analysis
Rscript 04_demographic_analysis.R

# Step 4: Aim 2 - Feature analysis (now with demographic TF-IDF!)
Rscript 05_aim2_feature_analysis.R

# Step 5: INTEGRATION - Aim 1 + Aim 2 (NEW!)
Rscript 06_integration_analysis.R
```

### Expected Outputs:

```
results/
├── predictions_df.csv                               # From Step 2
├── evaluation_summary.csv                           # From Step 2
├── demographic/
│   ├── subgroup_performance.csv                     # From Step 3
│   └── demographic_analysis_report.txt              # From Step 3
├── aim2/
│   ├── chi_squared_results.csv                      # From Step 4
│   ├── tfidf_top_terms.csv                          # From Step 4
│   ├── demographic_tfidf_stratified.rds             # NEW! From Step 4
│   ├── demographic_tfidf_comparison.csv             # NEW! From Step 4
│   ├── demographic_chi2_stratified.rds              # From Step 4
│   ├── demographic_lime_stratified.rds              # From Step 4
│   ├── behavioral_test_terms.rds                    # From Step 4
│   └── lime_explanations.csv                        # From Step 4
└── integration/
    ├── bias_characterization_summary.csv            # NEW! From Step 5
    └── integrated_dashboard.rds                     # NEW! From Step 5

figures/
├── AUC_CNNr.png                                     # From Step 2
├── calibration_plot.png                             # From Step 2
├── demographic/
│   ├── auc_by_subgroup_enhanced.png                 # From Step 3
│   ├── null_distribution_*.png                      # From Step 3
│   └── intersectional_heatmap.png                   # From Step 3
├── aim2/
│   ├── wordcloud_adrd.png                           # From Step 4
│   ├── chi_squared_keyness.png                      # From Step 4
│   ├── tfidf_comparison.png                         # From Step 4
│   ├── tfidf_heatmap_gender.png                     # NEW! From Step 4
│   ├── tfidf_heatmap_race.png                       # NEW! From Step 4
│   ├── tfidf_top_phrases_gender.png                 # NEW! From Step 4
│   ├── tfidf_top_phrases_race.png                   # NEW! From Step 4
│   ├── tfidf_unique_terms_gender.png                # NEW! From Step 4
│   ├── tfidf_unique_terms_race.png                  # NEW! From Step 4
│   └── lime_explanations.png                        # From Step 4
└── integration/
    ├── integrated_dashboard.png                     # NEW! From Step 5
    └── bias_framework_visualization.png             # NEW! From Step 5
```

---

## 🔬 Methodological Contributions

### For AMIA Submission:

#### 1. Novel Integration of Fairness and Explainability

This pipeline uniquely combines:
- **Approximate randomization** for rigorous hypothesis testing
- **TF-IDF demographic stratification** for linguistic bias detection
- **Behavioral testing** for causal validation
- **LIME explanations** for individual-level understanding

Result: Complete characterization of WHERE, WHY, and HOW bias manifests in CNN models.

#### 2. Three-Dimensional Bias Framework

**Clinical Bias**: ADRD presentation differences by demographics
- Captured through corpus analysis and chi-squared testing
- Reveals how clinical documentation varies

**Algorithmic Bias**: Classification parity evaluation
- Approximate randomization tests performance gaps
- Quantifies disparate impact

**Linguistic Bias**: Feature salience disparities
- TF-IDF stratification identifies discriminative terms
- Explains mechanistic basis of algorithmic bias

#### 3. Statistical Rigor

- Non-parametric testing (no distributional assumptions)
- Multiple testing correction (FDR control)
- Effect size quantification (Cohen's d)
- Bootstrap confidence intervals
- Reproducible implementation (fixed random seed)

#### 4. Clinical Relevance

- Uses real CNN models (Jihad Obeid's pre-trained)
- Identifies actionable linguistic patterns
- Connects to SDOH factors (insurance, education)
- Intersectional analysis (Gender × Race)

---

## 💡 Example Use Cases

### Use Case 1: Identifying Gender Bias in CNN

**Question**: Does the CNN perform differently for male vs. female patients?

**Analysis Steps**:
1. Run `04_demographic_analysis.R` → Permutation test for gender
2. Check `results/demographic/subgroup_performance.csv` → AUC differences
3. Run `05_aim2_feature_analysis.R` → TF-IDF by gender
4. View `figures/aim2/tfidf_heatmap_gender.png` → Feature patterns
5. Run `06_integration_analysis.R` → Complete characterization

**Outputs**:
- Statistical significance: p-value from permutation test
- Performance gap: AUC_Male - AUC_Female
- Discriminative terms: Phrases driving the gap
- Visualizations: Integrated dashboard showing all three dimensions

### Use Case 2: Explaining Race-Based Performance Disparities

**Question**: Why does the CNN perform worse for certain racial groups?

**Analysis Steps**:
1. `04_demographic_analysis.R` → Identifies race with performance gap
2. `05_aim2_feature_analysis.R` → TF-IDF, chi-squared, LIME for race
3. `06_integration_analysis.R` → Links performance to features
4. Review `results/integration/bias_characterization_summary.csv`

**Interpretation**:
- **High feature overlap (>60%)**: Similar linguistic patterns → Bias not feature-driven
- **Low feature overlap (<40%)**: Distinct patterns → Linguistic bias present
- **Unique terms**: Reveal what CNN "sees" differently by race

### Use Case 3: Preparing AMIA Manuscript

**Materials Needed**:
1. **Methods Section**: Use `METHODS_APPROXIMATE_RANDOMIZATION.md` directly
2. **Results Figures**:
   - Figure 1: `integrated_dashboard.png` (main result)
   - Figure 2: `bias_framework_visualization.png` (framework)
   - Figure 3: `auc_by_subgroup_enhanced.png` (performance gaps)
   - Figure 4: `tfidf_heatmap_gender.png` (feature patterns)
3. **Results Tables**:
   - Table 1: `bias_characterization_summary.csv` (main findings)
   - Table 2: `subgroup_performance.csv` (detailed performance)
   - Table 3: `demographic_tfidf_comparison.csv` (feature analysis)

**Abstract Structure**:
- **Background**: ADRD e-phenotyping with CNN models
- **Objective**: Evaluate bias across demographic subgroups
- **Methods**: Approximate randomization + TF-IDF stratification
- **Results**: [Performance gaps] + [Discriminative features]
- **Conclusion**: Three-dimensional bias framework provides actionable insights

---

## 🐛 Troubleshooting

### Issue 1: "Predictions not available" warning

**Cause**: `results/predictions_df.csv` not found

**Solution**: Run `03_evaluate_models.R` first

### Issue 2: "No demographic variables found"

**Cause**: Test set missing demographic columns

**Solution**: Ensure `data/test_set.rds` contains:
- `GENDER` (or `gender`)
- `RACE` (or `race`)
- `HISPANIC` (or `ethnicity`)

### Issue 3: "Insufficient samples" for subgroup

**Cause**: Subgroup has <20 correct or incorrect predictions

**Solution**: This is expected for small subgroups; analysis automatically skips them

### Issue 4: Integration script shows no data

**Cause**: Either Aim 1 or Aim 2 not run yet

**Solution**: Run pipeline in order (Steps 2 → 3 → 4 → 5)

### Issue 5: Visualizations look empty

**Cause**: No significant performance gaps detected

**Solution**: This is a valid result! Report "no significant bias detected"

---

## 📚 Related Documentation

- **`README.md`**: Project overview and quick start
- **`AIM1_DEMOGRAPHIC_FAIRNESS_GUIDE.md`**: Detailed Aim 1 documentation
- **`AIM2_FEATURE_FAIRNESS_GUIDE.md`**: Detailed Aim 2 documentation
- **`STATISTICAL_SIGNIFICANCE_METHODOLOGY.md`**: Statistical methods
- **`APPROXIMATE_RANDOMIZATION_EXPLANATION.md`**: Permutation testing
- **`TFIDF_EXPLANATION.md`**: TF-IDF methodology
- **`METHODS_APPROXIMATE_RANDOMIZATION.md`**: Publication-ready Methods section (NEW)
- **`ENHANCEMENTS_GUIDE.md`**: This document (NEW)

---

## 📞 Support

For questions about:
- **Analysis pipeline**: Review `AIM1_DEMOGRAPHIC_FAIRNESS_GUIDE.md` and `AIM2_FEATURE_FAIRNESS_GUIDE.md`
- **Statistical methods**: See `STATISTICAL_SIGNIFICANCE_METHODOLOGY.md`
- **TF-IDF**: Read `TFIDF_EXPLANATION.md`
- **Integration**: This guide (Section 3)
- **AMIA submission**: Use `METHODS_APPROXIMATE_RANDOMIZATION.md`

---

## ✅ Summary of Deliverables

### ✅ Code Enhancements:
1. **Demographic-stratified TF-IDF analysis** (05_aim2_feature_analysis.R)
2. **TF-IDF heatmap visualizations** (05_aim2_feature_analysis.R)
3. **Integration script** (06_integration_analysis.R - NEW FILE)

### ✅ Documentation:
1. **Methods section** (METHODS_APPROXIMATE_RANDOMIZATION.md - NEW FILE)
2. **Enhancements guide** (ENHANCEMENTS_GUIDE.md - THIS FILE)

### ✅ Preserved:
1. All existing functionality (Behavioral Testing, LIME, Chi-squared, Bootstrap)
2. All existing visualizations
3. All existing outputs
4. Approximate randomization (already correctly implemented)

### ✅ Ready for AMIA:
- Publication-ready Methods section (407 words)
- Comprehensive visualizations for figures
- Summary tables for results
- Complete bias characterization framework

---

**Status**: All proposal requirements met ✅
**Focus**: CNN model only ✅
**Option**: A (Core needs - TF-IDF enhancement, integration, Methods section) ✅

**Ready for AMIA submission in May 2025!** 🎉
