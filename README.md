# ADRD ePhenotyping Pipeline

**Version**: 2.1 (Enhanced)
**Author**: Gyasi, Frederick
**Date**: 2025-11-25
**Project**: ADRD ePhenotyping with Model Fairness and Explainability

---

## 🎯 Overview

This pipeline evaluates **pre-trained CNN models** (from Jihad Obeid) for ADRD classification with focus on:

- **Aim 1**: Demographic fairness analysis with approximate randomization testing
- **Aim 2**: Feature analysis with TF-IDF, LIME, chi-squared, and behavioral testing
- **Integration**: Complete bias characterization connecting Aim 1 + Aim 2 ✨ NEW!

### 🆕 What's New in v2.1 (AMIA Submission Ready):

1. ✅ **Demographic-Stratified TF-IDF Analysis** - Identifies clinical phrases driving CNN performance differences across demographic subgroups
2. ✅ **TF-IDF Heatmap Visualizations** - Visual representation of feature importance by demographics
3. ✅ **Integration Dashboard** - Comprehensive bias characterization linking WHERE bias exists (Aim 1) to WHY it exists (Aim 2)
4. ✅ **Publication-Ready Methods Section** - 407-word Methods section for AMIA/JAMIA submission

**KEY**: We use **Jihad's trained CNN models** - NO new training required!

---

## 📋 Quick Start

### 1. Setup Environment (One Time)
```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

### 2. Copy Jihad's Trained Models
```bash
mkdir -p models

# Required artifacts (CNNr models only - we focus on Random CNN)
cp /path/to/jihad/CL07_tokenizer_ref2 models/
cp /path/to/jihad/CL07_model_CNNr*.h5 models/         # 10 models: CNNr01-CNNr10
cp /path/to/jihad/CL07_model_CNNr*_hx.rds models/     # Training histories
cp /path/to/jihad/maxlen.rds models/                   # Sequence length

# Note: Pipeline auto-detects CL07_* naming convention
# CNNw (Word2Vec) models are NOT needed - we focus on CNNr (Random) only
```

### 3. Copy Your Test Data
```bash
mkdir -p data
cp /path/to/test_set.rds data/
cp /path/to/train_set.rds data/  # Needed for Aim 2
```

### 4. Run Complete Analysis Pipeline
```bash
./activate_adrd.sh

# Evaluate Jihad's models (NO training!)
Rscript 03_evaluate_models.R          # Model evaluation
Rscript 04_demographic_analysis.R     # Aim 1: Fairness
Rscript 05_aim2_feature_analysis.R    # Aim 2: Features (enhanced!)
Rscript 06_integration_analysis.R     # Integration (NEW!)
```

---

## 📂 Main Scripts (Execution Order)

### ⭐ Core Workflow (Use These)

1. **`03_evaluate_models.R`** - Model Evaluation
   - Evaluates all trained CNN models on test set
   - Auto-detects Jihad's naming convention
   - Generates predictions and comprehensive metrics
   - **Output**: `results/predictions_df.csv`

2. **`04_demographic_analysis.R`** - Aim 1: Fairness Analysis
   - Performance across demographic groups
   - Approximate randomization testing (10,000 permutations)
   - Bootstrap confidence intervals
   - Effect sizes and FDR correction
   - **Output**: `results/demographic/subgroup_performance.csv`

3. **`05_aim2_feature_analysis.R`** - Aim 2: Feature Analysis (ENHANCED!)
   - Chi-squared discriminative terms
   - **TF-IDF analysis by demographic subgroups** ✨ NEW!
   - LIME explainability with demographic stratification
   - Behavioral testing framework
   - **New Outputs**:
     - `results/aim2/demographic_tfidf_stratified.rds`
     - `results/aim2/demographic_tfidf_comparison.csv`
     - `figures/aim2/tfidf_heatmap_*.png`

4. **`06_integration_analysis.R`** - Integration Dashboard ✨ NEW!
   - Connects Aim 1 + Aim 2 findings
   - Complete bias characterization (Clinical, Algorithmic, Linguistic)
   - Identifies WHERE bias exists and WHY
   - **New Outputs**:
     - `results/integration/bias_characterization_summary.csv`
     - `figures/integration/integrated_dashboard.png`
     - `figures/integration/bias_framework_visualization.png`

### 📝 Optional Scripts

- **`01_prepare_data.R`** - Only if you need to create train/test split
- **`02_train_cnnr.R`** - Reference only (use Jihad's models instead)

---

## 📊 Key Output Files

### ⭐ Must Review (AMIA Submission)
- `results/integration/bias_characterization_summary.csv` ⭐⭐ - Complete bias analysis (NEW!)
- `results/demographic/subgroup_performance.csv` ⭐ - Performance by demographics
- `results/aim2/demographic_tfidf_comparison.csv` ⭐ - Features by demographics (NEW!)
- `results/aim2/discriminative_terms.xlsx` - Top ADRD vs CTRL terms

### 🎨 Key Visualizations (For Publication)
- `figures/integration/integrated_dashboard.png` ⭐⭐ - Main figure (NEW!)
- `figures/integration/bias_framework_visualization.png` ⭐⭐ - Framework (NEW!)
- `figures/aim2/tfidf_heatmap_gender.png` ⭐ - Feature importance (NEW!)
- `figures/demographic/auc_by_subgroup_enhanced.png` - Performance gaps
- `figures/demographic/null_distribution_*.png` - Permutation tests
- `figures/aim2/chi_squared_keyness.png` - Discriminative terms

---

## 🛠️ Utility Scripts

- **`utils_model_loader.R`** - Auto-detects model naming conventions
- **`utils_statistical_tests.R`** - Permutation tests, bootstrap CIs

---

## 📚 Documentation

### 🆕 New Documentation (v2.1)
- **`METHODS_APPROXIMATE_RANDOMIZATION.md`** ⭐⭐ - Publication-ready Methods section (407 words) - **Use for AMIA submission!**
- **`ENHANCEMENTS_GUIDE.md`** ⭐ - Complete guide to new features (TF-IDF, integration, visualizations)

### Core Documentation
- `AIM1_DEMOGRAPHIC_FAIRNESS_GUIDE.md` - Aim 1 detailed guide
- `AIM2_FEATURE_FAIRNESS_GUIDE.md` - Aim 2 detailed guide
- `STATISTICAL_SIGNIFICANCE_METHODOLOGY.md` - Statistical methods
- `APPROXIMATE_RANDOMIZATION_EXPLANATION.md` - Permutation testing explained
- `TFIDF_EXPLANATION.md` - TF-IDF methodology
- `COLUMN_NAMES_REFERENCE.md` - Data formats
- `JIHAD_ARTIFACTS_COMPATIBILITY.md` - Model naming conventions

---

## ⚙️ Configuration

### Statistical Testing (in 04_demographic_analysis.R)
```r
RUN_STATISTICAL_TESTS <- TRUE   # Enable/disable
N_PERMUTATIONS <- 10000         # Permutations for significance
N_BOOTSTRAP <- 10000            # Bootstrap samples
```

### Feature Analysis (in 05_aim2_feature_analysis.R)
```r
TOP_N_FEATURES <- 100           # Top discriminative terms
MIN_TERM_FREQ <- 10             # Minimum term frequency
```

---

## 🎓 Research Aims

**Aim 1**: Evaluate CNN fairness across demographic groups
- Approximate randomization testing (10,000 permutations)
- Classification parity evaluation
- Performance gap identification

**Aim 2**: Identify discriminative features and explain CNN predictions
- Demographic-stratified TF-IDF analysis ✨ NEW!
- Chi-squared discriminative terms
- LIME explainability
- Behavioral testing framework

**Integration**: Complete bias characterization ✨ NEW!
- Links WHERE bias exists (Aim 1) to WHY it exists (Aim 2)
- Three-dimensional framework: Clinical, Algorithmic, Linguistic bias
- Publication-ready visualizations and summary tables

---

## 🎯 Three-Dimensional Bias Framework

This pipeline uniquely evaluates CNN bias across three dimensions:

1. **Clinical Bias**: How ADRD presentation differs by demographics
   - Corpus analysis and chi-squared testing
   - Identifies clinical documentation variation

2. **Algorithmic Bias**: CNN classification parity evaluation
   - Approximate randomization tests performance gaps
   - Quantifies disparate impact across subgroups

3. **Linguistic Bias**: Feature salience disparities
   - TF-IDF stratification by demographics
   - Explains mechanistic basis of algorithmic bias
   - Identifies which terms drive performance differences

**Result**: Complete understanding of WHERE, WHY, and HOW bias manifests in CNN models.

---

## 📞 Support

### For Analysis Questions:
- **TF-IDF enhancements**: See `ENHANCEMENTS_GUIDE.md`
- **AMIA submission**: Use `METHODS_APPROXIMATE_RANDOMIZATION.md`
- **Aim 1 details**: See `AIM1_DEMOGRAPHIC_FAIRNESS_GUIDE.md`
- **Aim 2 details**: See `AIM2_FEATURE_FAIRNESS_GUIDE.md`
- **Statistical methods**: See `STATISTICAL_SIGNIFICANCE_METHODOLOGY.md`

---

## 📝 Version History

**Version 2.1 (2025-11-25)**: AMIA Submission Ready
- ✅ Demographic-stratified TF-IDF analysis
- ✅ TF-IDF heatmap visualizations (3 types)
- ✅ Integration dashboard (Aim 1 + Aim 2)
- ✅ Three-dimensional bias framework
- ✅ Publication-ready Methods section (407 words)
- ✅ Comprehensive documentation and usage examples

**Version 2.0 (2025-11-06)**: Statistical Rigor
- Approximate randomization testing
- Bootstrap confidence intervals
- LIME explainability
- Behavioral testing framework
- Jihad model compatibility

---

**🎉 Ready for AMIA Submission - May 2025!**
