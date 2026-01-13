# ADRD ePhenotyping with Fairness Analysis

---

## Project Information

**Lead/Mentor:** Jihad Obeid, Paul Heider

**Contributors:** Frederick Gyasi

**Current Funding:** N/A

**IRB #:** N/A

**RMID:** N/A

**SPARCRequest:** N/A

---

## Project Summary

This project evaluates a pre-trained Convolutional Neural Network (CNN) model for automated detection of Alzheimer's Disease and Related Dementias (ADRD) from clinical Electronic Health Record (EHR) notes, with comprehensive fairness analysis across demographic groups. The CNN model was originally developed by Knox and Obeid using MUSC clinical data.

### Research Questions

1. **Does the CNN model perform equitably across demographic groups (gender, race, ethnicity)?**

2. **Do the discriminative linguistic features driving CNN predictions differ across demographic subgroups?**

---

## Project Experiments

### Aim 1: Demographic Fairness Analysis

**Objective:** Evaluate CNN classification parity across demographic subgroups using approximate randomization testing (10,000 permutations).

**Subgroups Analyzed:** Gender (2), Race (4), Ethnicity (2), Intersectional Gender x Race (4)

### Aim 2: Feature-Level Fairness Analysis

**Objective:** Identify discriminative features and assess consistency across demographics using Chi-squared testing, TF-IDF analysis, and behavioral testing.

**Features Tested:** 13,889 clinical terms from 1,460 patient documents

---

## Experimental Results

### Dataset Characteristics

| Characteristic | Value |
|----------------|-------|
| Total Samples | 1,460 |
| ADRD Cases | 657 (45.0%) |
| Control Cases | 803 (55.0%) |
| Data Source | MUSC EHR Clinical Notes |

### Demographic Distribution

| Demographic | Category | N | % |
|-------------|----------|---|---|
| Gender | Female | 828 | 56.7% |
| Gender | Male | 632 | 43.3% |
| Race | White | 1,013 | 69.4% |
| Race | Black | 407 | 27.9% |
| Race | Asian | 10 | 0.7% |
| Race | Other | 21 | 1.4% |
| Ethnicity | Non-Hispanic | 1,441 | 98.7% |
| Ethnicity | Hispanic | 14 | 1.0% |

### Overall Model Performance (Best Model - Cycle 9)

| Metric | Value | 95% CI |
|--------|-------|--------|
| AUC | 0.9867 | 0.9818-0.9916 |
| Accuracy | 94.25% | - |
| Sensitivity | 97.26% | - |
| Specificity | 91.78% | - |
| Precision (PPV) | 90.64% | - |
| NPV | 97.62% | - |
| F1 Score | 0.9383 | - |
| Brier Score | 0.0440 | - |

### Confusion Matrix (n=1,460)

|  | Predicted Control | Predicted ADRD |
|--|-------------------|----------------|
| Actual Control | 737 (TN) | 66 (FP) |
| Actual ADRD | 18 (FN) | 639 (TP) |

### Aim 1 Results: Demographic Fairness

#### Gender Performance

| Metric | Female | Male | Difference |
|--------|--------|------|------------|
| AUC | 0.9867 | 0.9875 | 0.0008 |
| Sensitivity | 98.40% | 95.73% | 2.67% |
| Specificity | 90.71% | 93.16% | 2.45% |

**Finding:** No statistically significant disparities (p=0.432)

#### Racial Performance

| Race | N | AUC | Sensitivity | Specificity |
|------|---|-----|-------------|-------------|
| White | 1,013 | 0.9855 | 97.04% | 91.43% |
| Black | 407 | 0.9893 | 97.83% | 93.22% |
| Asian | 10 | 1.0000 | 100% | 100% |
| Other | 21 | 0.9727 | 90.00% | 81.82% |

**Finding:** AUC difference (White-Black): 0.0038 (within fairness threshold)

#### Intersectional Performance (Gender x Race)

| Intersection | N | AUC | Sensitivity | Specificity |
|--------------|---|-----|-------------|-------------|
| Female x White | 546 | 0.9839 | 98.23% | 89.69% |
| Male x White | 467 | 0.9880 | 95.56% | 93.38% |
| Female x Black | 260 | 0.9913 | 98.54% | 92.68% |
| Male x Black | 147 | 0.9853 | 96.77% | 94.44% |

**Finding:** AUC range: 0.0074 (within acceptable limits)

### Aim 2 Results: Feature Analysis

#### Chi-Squared Test Results

| Measure | Value |
|---------|-------|
| Features Tested | 13,889 |
| Significant Features (FDR < 0.05) | 3,790 |
| ADRD-Overrepresented | 3,790 |
| Control-Overrepresented | 0 |

#### Top 10 Discriminative Terms for ADRD

| Rank | Term | Chi-Squared | Clinical Category |
|------|------|-------------|-------------------|
| 1 | goal | 4,623 | Care planning |
| 2 | outcome | 4,401 | Care planning |
| 3 | ongoing | 3,717 | Disease management |
| 4 | progressing | 2,753 | Disease progression |
| 5 | discharge | 2,129 | Care transitions |
| 6 | oral | 2,114 | Medication admin |
| 7 | pt | 2,059 | Patient reference |
| 8 | dementia | 1,858 | Diagnosis |
| 9 | admission | 1,841 | Hospital encounters |
| 10 | care | 1,725 | General care |

#### Feature Consistency Across Demographics

| Comparison | Overlap | Interpretation |
|------------|---------|----------------|
| Female vs. Male | 90% | Highly Consistent |
| Black vs. White | 70% | Good Consistency |

**Finding:** 70-90% term overlap indicates robust, generalizable discriminative features

---

## Key Findings

1. **Model Performance:** CNN achieves AUC=0.9867 with 97.26% sensitivity for ADRD detection

2. **Demographic Fairness:** No statistically significant performance disparities across gender, race, or ethnicity (all p-values > 0.05)

3. **Intersectional Fairness:** AUC variance < 1% across Gender x Race intersections

4. **Feature Generalizability:** 70-90% discriminative term overlap across demographic subgroups indicates model captures universal ADRD phenotypes

---

## Resources

### Code Repository
- **GitHub:** https://github.com/gyasifred/adrd_ePheno

### Datasets Used
- MUSC Electronic Health Records (de-identified clinical notes)
- Pre-trained CNN models (Knox/Obeid, 10 cycles)

### Key Analysis Scripts
- `03_evaluate_models.R` - Model evaluation
- `04_demographic_analysis.R` - Aim 1: Fairness analysis
- `05_aim2_feature_analysis.R` - Aim 2: Feature analysis
- `06_integration_analysis.R` - Integration analysis

---

## Acknowledgements

**Lead/Mentor:** Jihad Obeid, Paul Heider

**Contributors:** Frederick Gyasi

**Data Source:** Medical University of South Carolina (MUSC) Research Data Warehouse

---
