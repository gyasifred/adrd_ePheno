# ADRD ePhenotyping Project - Proposal Analysis & Implementation Roadmap

**Author**: Gyasi, Frederick
**Date**: 2025-11-06
**Project**: ADRD e-Phenotyping with focus on model fairness and explainability

---

## Executive Summary

This document provides a comprehensive analysis of the research proposal and implementation roadmap for the ADRD ePhenotyping project. The project aims to evaluate AI model performance across demographic groups (Aim 1) and identify cohort-specific features using explainability methods (Aim 2).

---

## Proposal Overview

### Background

- **Problem**: AI models for ADRD detection show potential gender bias
  - F1 scores lower for male notes than female notes
  - CCW algorithm performed significantly worse for male patients
  - Random Forest (RF) model showed numerical (but not statistically significant) worse performance for males

### Primary Objective

Develop and deploy **generalizable and equitable models** for e-phenotyping ADRD by:
1. Analyzing relative model performance across demographic cohorts
2. Identifying feature salience and documenting model sensitivity to different medical factors

---

## Specific Aims

### Aim 1: Evaluate Model Performance Differences Between Demographic Groups and Social Drivers

**Objective**: The objective for this aim is to expand the analysis of relative performance differences that we have evaluated our set of trained ADRD e-phenotyping models on.

**Key Activities**:
1. Use **PRE-TRAINED MODELS** from Jihad Obeid (no new training)
2. Focus from **evaluation onwards** with test set
3. Expand beyond binary gender comparison to:
   - **Demographic factors**: Gender, Race, Ethnicity, Age
   - **Social determinants of health (SDOH)**: Insurance type, Education level, Financial class
   - **Intersectional groups**: e.g., Gender × Race

**Methodology**:
- Use pre-trained models to process all notes
- Compare model performance at e-phenotyping level (patient-level predictions)
- Use **approximate randomization** for statistical significance testing
- Evaluate across a wide range of demographic factors and SDOH categories

**Hypothesis**:
- For some categories (like gender), we expect uniform numerical differences
- For other categories, different top-performing cohorts for different model types
- The approach will evaluate models for statistically significant differences across demographic factors

**Expected Outcome**:
- Identify where current models fail certain subgroups
- Provide evidence for targeted improvements

---

### Aim 2: Evaluate Ablation Studies and Explainable AI Approaches to Identify Cohort-Specific Feature Differences

**Objective**: Understand particular phrases in unstructured clinical notes that each model considers most important for e-phenotyping a patient, and how models are particularly sensitive to specific clinical notes.

**Key Question**: What are the salient features/words/phrases that most significantly help classify notes OR are missing in incorrectly classified notes?

**Methodology**:

1. **Identify Overrepresented Terms** (χ² Testing)
   - Use χ²-squared testing to identify terms/phrases overrepresented in certain groups
   - Term frequency analysis to find differential documentation patterns

2. **Behavioral Testing**
   - Create parallel test corpora that differ only in terms of specific targeted text examples
   - Downstream changes in model predictions directly attributable to presence/absence of phrases
   - Use domain knowledge + subject matter experts to classify terms as:
     - Clinically relevant vs. incidental language
     - Important distinctions

3. **Explainability Methods**
   - **LIME** (Local Interpretable Model-agnostic Explanations)
   - **SHAP** (SHapley Additive exPlanations)
   - Identify contribution of different phrases to model performance
   - Understand clinical identification of ADRD/comorbidities
   - Determine if phrases are overrepresented or underrepresented

**Working Hypothesis**:
The most salient phrases contributing to a model's performance will help us understand the clinical and artifactual evidence used by models to be important.

**Expected Outcome**:
- Corpus analysis showing overrepresented words
- Feature importance rankings
- Behavioral test results
- LIME/SHAP explanations for model decisions

---

## Current Implementation Status

### Completed Components

1. **Data Preparation Pipeline** (`01_prepare_data.R`)
   - ✓ Load and validate clinical notes
   - ✓ Convert string labels to numeric (ADRD→1, NON-ADRD→0)
   - ✓ Rename 'content' to 'txt' for compatibility
   - ✓ Stratified train/test split (80/20)
   - ✓ Comprehensive data quality checks

2. **CNN Training Pipeline** (`02_train_cnnr.R`)
   - ✓ Random CNN (CNNr) implementation following Jihad's methodology
   - ✓ 10 training cycles for statistical robustness
   - ✓ Median AUC selection with max F1
   - ✓ All artifacts saved (tokenizer, models, histories)

3. **Evaluation Pipeline** (`03_evaluate_models.R`)
   - ✓ Comprehensive metrics calculation
   - ✓ ROC curve analysis
   - ✓ Calibration plots
   - ✓ Probability distributions
   - ✓ Confusion matrices

4. **Demographic Analysis** (`04_demographic_analysis.R`)
   - ✓ Gender-stratified analysis
   - ✓ Race-stratified analysis
   - ✓ Ethnicity-stratified analysis
   - ✓ Performance comparisons across subgroups
   - ✓ Fairness metrics

---

## Key Issues and Improvements Needed

### Issue 1: Artifact Naming Standardization ⚠️

**Problem**: Inconsistent naming between Jihad's code and current pipeline

| Artifact | Jihad's Naming | Current Pipeline | Recommendation |
|----------|----------------|------------------|----------------|
| Tokenizer | `CL07_tokenizer_ref2` | `tokenizer_cnnr` | Keep current (simpler) |
| Models | `CL07_model_CNNr01.h5` | `model_CNNr01.h5` | Keep current |
| Results | Date suffixes | No date suffixes | Keep current (cleaner) |
| Word counts | `notes_clean_wc.rds` | Not created | Add if needed |

**Resolution**: Document both conventions; current pipeline is cleaner and more maintainable

---

### Issue 2: Demographic Names Need Cleaning ⚠️

**Problem**: Raw demographic values are too verbose for plots and reports

**Current Examples**:
- `"WHITE OR CAUCASIAN"` → Should be `"White"`
- `"BLACK OR AFRICAN AMERICAN"` → Should be `"Black"`
- `"NO, NOT HISPANIC OR LATINO"` → Should be `"Non-Hispanic"`

**Solution**: Already implemented in `04_demographic_analysis.R` via:
- `simplify_category_name()` function
- `wrap_text()` function for plot labels

**Enhancement Needed**: Verify all categories are properly simplified

---

### Issue 3: AIM 1 Visualizations Need Improvement 🔧

**Current Plots** (Good):
- ✓ AUC by subgroup
- ✓ Sensitivity vs Specificity scatter
- ✓ Metrics comparison faceted plot

**Improvements Needed**:
1. Add confidence interval error bars consistently
2. Add sample size annotations
3. Create heatmap for intersectional analysis
4. Add fairness metrics visualization (PPR, demographic parity)
5. Create performance gap analysis plot

---

### Issue 4: AIM 2 Script Missing ❌

**Required**: New script `05_aim2_feature_analysis.R` to include:

1. **Corpus Analysis**
   - Word frequency analysis by class (ADRD vs CTRL)
   - TF-IDF analysis
   - Chi-squared test for overrepresented terms
   - Visualization of top discriminative words

2. **Model Explainability**
   - LIME implementation for CNN
   - SHAP implementation (if feasible for Keras)
   - Feature importance extraction
   - Visualization of important n-grams

3. **Behavioral Testing Framework**
   - Create test cases with targeted phrase substitutions
   - Measure model sensitivity to specific terms
   - Document clinical relevance

---

### Issue 5: Statistical Significance Documentation Needed 📄

**Required**: Markdown document explaining statistical methods

**Content Needed**:
1. **Approximate Randomization Testing**
   - Methodology
   - Implementation approach
   - Interpretation guidelines
   - R package recommendations (`coin`, `lmPerm`, `perm`)

2. **Multiple Testing Correction**
   - Bonferroni correction
   - Benjamini-Hochberg FDR
   - When to use each method

3. **Effect Size Measures**
   - Cohen's d
   - AUC differences
   - Practical significance thresholds

4. **Bootstrap Confidence Intervals**
   - For AUC comparisons
   - For sensitivity/specificity differences

---

### Issue 6: Column Names Documentation Needed 📋

**Required**: Document for Dr. Paul listing all column names

**Sections Needed**:
1. **Input Data Columns** (raw CSV)
2. **Processed Data Columns** (after 01_prepare_data.R)
3. **Prediction Columns** (after 03_evaluate_models.R)
4. **Demographic Variables** with value mappings
5. **Model Output Columns**

---

## Implementation Priority

### Phase 1: Documentation & Analysis (High Priority) ✅
1. ✅ Create this roadmap document
2. 🔄 Create statistical significance methodology document
3. 🔄 Create column names reference document
4. 🔄 Create artifact naming conventions document

### Phase 2: AIM 1 Enhancements (High Priority)
1. Verify and enhance demographic name cleaning
2. Improve visualization quality
3. Add intersectional analysis
4. Add fairness metrics calculations

### Phase 3: AIM 2 Implementation (Critical) ⚠️
1. Create `05_aim2_feature_analysis.R` script
2. Implement corpus analysis (TF-IDF, chi-squared)
3. Implement LIME/SHAP explainability
4. Create behavioral testing framework
5. Generate comprehensive visualizations

### Phase 4: Integration & Testing
1. Test complete pipeline end-to-end
2. Verify all artifacts are properly named
3. Generate example outputs
4. Create comprehensive user documentation

---

## Technical Specifications

### Focus Model: Random CNN (CNNr)

**Architecture** (from Jihad's implementation):
- **Embedding**: 200-dimensional random embeddings (learned)
- **CNN Filters**: 200 filters per kernel size
- **Kernel Sizes**: 3, 4, 5 (captures 3-grams, 4-grams, 5-grams)
- **Pooling**: Global max pooling per branch
- **Dense Layer**: 200 hidden units
- **Dropout**: 0.2 throughout
- **Output**: Sigmoid activation for binary classification

**Training** (for reference - we use pre-trained):
- 10 training cycles
- Median AUC selection with max F1
- Adam optimizer, lr=0.0004
- Batch size 32
- Early stopping with patience=7

### Statistical Methods

**For Aim 1** (Performance Comparison):
- **Primary**: Approximate randomization test
- **Secondary**: Bootstrap confidence intervals for AUC
- **Multiple testing**: Benjamini-Hochberg FDR correction
- **Effect size**: AUC differences, Cohen's d for continuous metrics
- **Minimum group size**: N ≥ 10 for subgroup analysis

**For Aim 2** (Feature Analysis):
- **Term importance**: χ² test with FDR correction
- **Explainability**: LIME with 1000 samples, SHAP if compatible
- **Behavioral tests**: Paired comparisons with Wilcoxon signed-rank

---

## Data Requirements

### Input Data Format

**Raw CSV** (`ptHx_sample_v2025-10-25.csv`):
```
Required columns:
- DE_ID: Patient identifier
- content: Clinical notes text
- Label: String ('ADRD' or 'NON-ADRD')

Optional demographic columns:
- GENDER: Patient gender
- RACE: Patient race
- HISPANIC: Hispanic ethnicity indicator
- AGE: Patient age (if available)
- INSURANCE/FINANCIAL_CLASS: SDOH indicators (if available)
```

### Test Set Requirements

**For fair evaluation**:
- Stratified by label + demographics
- No patient overlap with training
- Representative of deployment population
- Sufficient samples per demographic group (N ≥ 10)

---

## Success Criteria

### Aim 1 Success Metrics:
- ✓ Performance metrics calculated for all demographic subgroups with N ≥ 10
- ✓ Statistical significance tests completed for all comparisons
- ✓ Fairness metrics (demographic parity, equalized odds) computed
- ✓ Visualization suite generated
- ✓ Disparities identified and documented

### Aim 2 Success Metrics:
- ⚠️ Top 100 discriminative terms identified with statistical significance
- ⚠️ LIME/SHAP explanations generated for sample of predictions
- ⚠️ Behavioral test suite created and executed
- ⚠️ Clinical relevance assessment completed
- ⚠️ Feature importance visualizations created

---

## Next Steps

### Immediate Actions:
1. Create statistical significance methodology document
2. Create column names reference document
3. Enhance demographic name cleaning (verify completeness)
4. **Start AIM 2 script development** (highest priority)

### This Week:
1. Complete all documentation
2. Implement AIM 2 corpus analysis
3. Implement LIME explainability
4. Test complete pipeline

### Next Week:
1. Implement SHAP (if feasible)
2. Create behavioral testing framework
3. Generate all visualizations
4. Prepare results for Dr. Paul

---

## References

### Key Papers Mentioned in Proposal:
- Behavioral Testing methodology
- LIME: Ribeiro et al.
- SHAP: Lundberg et al.
- Term frequency-inverse document frequency (TF-IDF)
- Chi-squared testing for feature selection

### R Packages Needed:
**Current**:
- tidyverse, keras, tensorflow, pROC, ggplot2

**Additional for Aim 2**:
- `quanteda` (text analysis - already installed)
- `quanteda.textstats` (text statistics - already installed)
- `lime` (explainability)
- `fastshap` or `shapviz` (SHAP values)
- `wordcloud` (visualization - already installed)
- `coin` (permutation tests)

---

## Contact & Collaboration

**Key Stakeholders**:
- Dr. Paul (Principal Investigator)
- Jihad Obeid (Original model developer)
- Research team

**Code Repositories**:
- Current pipeline: `/home/user/adrd_ePheno/`
- Branch: `claude/standardize-pipeline-artifacts-011CUrqALuYizhgYftC1NEk3`

---

## Appendices

### A. Jihad's Original Code Analysis

**File**: `adrd_classif_v1_jihad_obeid.r`

**Key Components**:
1. **Preprocessing**: BOW for traditional ML, sequences for CNN
2. **Models**: RF, SVM, CNNr (random embeddings), CNNw (Word2Vec)
3. **Evaluation**: 10-cycle training with median AUC selection
4. **Outputs**: ROC curves, metrics Excel, predictions

**Differences from Current Pipeline**:
- Jihad's code is a monolithic RMarkdown notebook
- Current pipeline is modular (5 scripts)
- Current pipeline has cleaner naming conventions
- Current pipeline focuses on CNNr only
- Current pipeline adds demographic analysis

### B. Artifact Inventory

**Models Directory** (`models/`):
- `model_CNNr01.h5` through `model_CNNr10.h5`
- `tokenizer_cnnr` (Keras format)
- `word_index.rds`
- `vocab_size.rds`
- `maxlen.rds`
- `model_architecture.txt`
- `history_CNNr01.rds` through `history_CNNr10.rds`

**Results Directory** (`results/`):
- `metrics_df_CNNr.rds`
- `roc_df_rows_CNNr.rds`
- `test_labels.rds`
- `training_histories.rds`
- `best_model_info.rds`
- `best_model_evaluation.rds`
- `evaluation_summary.csv`
- `evaluation_summary.rds`
- `Summary_Metrics_CNNr.xlsx`
- `roc_df.rds`
- `predictions_df.csv`
- `training_report.txt`
- `evaluation_report.txt`

**Figures Directory** (`figures/`):
- `AUC_CNNr.png`
- `AUC_CNNr_zoom.png`
- `metrics_boxplot.png`
- `calibration_plot.png`
- `probability_distribution.png`
- `confusion_matrix.png`

**Demographic Subdirectories**:
- `results/demographic/subgroup_performance.*`
- `figures/demographic/*.png`

### C. Pipeline Workflow Diagram

```
01_prepare_data.R
    ↓
    data/train_set.rds, data/test_set.rds, data/split_info.rds
    ↓
02_train_cnnr.R
    ↓
    models/*.h5, models/tokenizer_cnnr, results/metrics_df_CNNr.rds
    ↓
03_evaluate_models.R
    ↓
    results/predictions_df.csv, figures/*.png
    ↓
04_demographic_analysis.R
    ↓
    results/demographic/*, figures/demographic/*
    ↓
05_aim2_feature_analysis.R  ← TO BE CREATED
    ↓
    results/aim2/*, figures/aim2/*
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-06 | Claude | Initial comprehensive analysis |

---

**END OF DOCUMENT**
