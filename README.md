# ADRD ePhenotyping Pipeline

**Version**: 2.0
**Date**: 2025-11-06
**Project**: ADRD ePhenotyping with Model Fairness and Explainability

---

## 🎯 Overview

This pipeline evaluates **pre-trained CNN models** (from Jihad Obeid) for ADRD classification with focus on:

- **Aim 1**: Demographic fairness analysis with statistical significance testing
- **Aim 2**: Feature analysis with LIME explainability and behavioral testing

**KEY**: We use **Jihad's trained models** - NO new training required!

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
cp /path/to/jihad/CL07_tokenizer_ref2 models/
cp /path/to/jihad/CL07_model_CNNr*.h5 models/
cp /path/to/jihad/CL07_*.rds models/  # If available
```

### 3. Copy Your Test Data
```bash
mkdir -p data
cp /path/to/test_set.rds data/
cp /path/to/train_set.rds data/  # Needed for Aim 2
```

### 4. Run Evaluation Pipeline
```bash
./activate_adrd.sh

# Evaluate Jihad's models (NO training!)
Rscript 03_evaluate_models.R
Rscript 04_demographic_analysis.R
Rscript 05_aim2_feature_analysis.R
```

---

## 📂 Main Scripts (Execution Order)

### ⭐ Core Workflow (Use These)

1. **`03_evaluate_models.R`** - Model Evaluation
   - Evaluates all trained models on test set
   - Auto-detects Jihad's naming convention
   - Generates predictions and metrics

2. **`04_demographic_analysis.R`** - Aim 1: Fairness Analysis
   - Performance across demographic groups
   - Statistical significance testing
   - Effect sizes and confidence intervals

3. **`05_aim2_feature_analysis.R`** - Aim 2: Feature Analysis
   - Chi-squared discriminative terms
   - LIME explainability
   - Behavioral testing framework

### 📝 Optional Scripts

- **`01_prepare_data.R`** - Only if you need to create train/test split
- **`02_train_cnnr.R`** - Reference only (use Jihad's models instead)

---

## 📊 Key Output Files

### Must Review
- `results/aim2/discriminative_terms.xlsx` ⭐ - Top ADRD vs CTRL terms
- `results/demographic/subgroup_performance.csv` ⭐ - Fairness metrics
- `results/evaluation_summary.csv` - Overall performance

### Visualizations
- `figures/demographic/auc_by_subgroup.png` - AUC by group
- `figures/aim2/chi_squared_keyness.png` - Discriminative terms
- `figures/aim2/wordcloud_adrd.png` - ADRD term cloud

---

## 🛠️ Utility Scripts

- **`utils_model_loader.R`** - Auto-detects model naming conventions
- **`utils_statistical_tests.R`** - Permutation tests, bootstrap CIs

---

## 📚 Documentation

- `PROPOSAL_ANALYSIS_AND_ROADMAP.md` - Project overview
- `STATISTICAL_SIGNIFICANCE_METHODOLOGY.md` - Statistical methods
- `COLUMN_NAMES_REFERENCE.md` - Data formats
- `IMPLEMENTATION_SUMMARY.md` - Technical details

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

**Aim 1**: Evaluate model fairness across demographic groups
**Aim 2**: Identify discriminative features and explain predictions

---

## 📞 Support

See documentation in `DOCUMENTATION/` folder or contact research team.

---

**Version 2.0**: Statistical testing, LIME explainability, behavioral testing, Jihad model compatibility
