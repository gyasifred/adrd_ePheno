# AMIA 2025 Submission Package - README

## Overview

This package contains all materials for AMIA 2025 conference submission on **"Deep Learning-Based ADRD ePhenotyping with Fairness Analysis"**.

## Files Created

### 1. PowerPoint Presentation
**File**: `AMIA_2025_ADRD_ePhenotyping_Presentation.pptx` (702 KB, 21 slides)

**Contents**:
- Title slide
- Background & Motivation
- Study Objectives (Aim 1 & 2)
- Dataset Description
- Methodology (Data Preprocessing, CNN Architecture, Fairness Framework)
- Results - Overall Performance (AUC=0.987)
- Results - Visualizations (ROC, Confusion Matrix, Calibration)
- Results - Demographic Fairness Analysis
- Results - Intersectional Analysis
- Results - Feature Analysis (Chi-squared, TF-IDF)
- Discussion (Key Findings, Clinical Implications, Comparison with Literature)
- Limitations & Future Work
- Conclusions
- Acknowledgments

**Usage**:
- Open in Microsoft PowerPoint, Google Slides, or LibreOffice Impress
- Edit speaker notes, acknowledgments, and funding information
- Add your institution logo to title slide
- Customizepage transitions and animations as needed

---

### 2. Detailed Paper Sections
**File**: `AMIA_Paper_Methodology_Results_Discussion.md` (34 KB, ~12,000 words)

**Contents**:

#### METHODS (~3,500 words)
1. **Study Design and Data Source**
   - IRB approval, HIPAA compliance
   - Cohort definition (N=1,460)
   - Inclusion/exclusion criteria

2. **Data Collection and Cohort Definition**
   - ICD-10 codes for ADRD identification
   - Demographic characteristics (Tables)
   - Matched control selection

3. **Text Preprocessing Pipeline**
   - De-identification (PHI removal)
   - Tokenization (quanteda)
   - Stopword removal
   - Artifact filtering
   - Vocabulary construction (38,079 → 13,890 features)

4. **CNN Model Architecture**
   - Embedding layer (300-dim)
   - Convolutional layers (filter sizes 3,4,5)
   - Pooling and regularization
   - Training configuration (Adam, early stopping)
   - Model selection criteria

5. **Evaluation Metrics**
   - Discrimination (AUC, sensitivity, specificity, F1)
   - Calibration (Brier score, log loss)
   - Optimal threshold (Youden's Index)

6. **Fairness Analysis**
   - Demographic stratification
   - Fairness criteria (equalized odds, AUC parity)
   - Statistical significance testing (approximate randomization)
   - Minimum sample size requirements

7. **Feature Analysis and Interpretability**
   - Chi-squared (χ²) test for discriminative terms
   - TF-IDF weighting
   - Demographic-stratified feature patterns
   - Clinical validation

8. **Software and Statistical Analysis**
   - R packages, Python libraries
   - Statistical significance thresholds

#### RESULTS (~4,000 words)
1. **Cohort Characteristics**
   - Table 1: Baseline demographics stratified by ADRD status
   - Statistical comparisons (p-values)

2. **Overall Model Performance**
   - Table 2: Performance metrics with 95% CIs
   - Confusion matrix (TP=639, FP=66, FN=18, TN=737)
   - Calibration metrics
   - Optimal threshold analysis
   - Cross-cycle stability

3. **Demographic Fairness Analysis**
   - Table 3: Performance across gender, race, ethnicity
   - Statistical significance tests (all p>0.05)
   - Key findings (no disparities detected)

4. **Intersectional Fairness Analysis**
   - Table 4: Gender × Race performance
   - AUC range: 0.9839 - 0.9913 (Δ=0.0074)
   - Best: Female × Black (0.9913)

5. **Feature Analysis and Model Interpretability**
   - Table 5: Top 20 discriminative terms (χ² statistics)
   - Clinical interpretation categories
   - TF-IDF analysis
   - Table 6: Top TF-IDF terms (clinically relevant)
   - Control-associated terms

6. **Demographic-Stratified Feature Analysis**
   - Table 7: Feature overlap across subgroups
   - 70-90% consistency
   - Common vs. unique terms
   - Clinical implications

#### DISCUSSION (~4,500 words)
1. **Principal Findings**
   - Three key advances summarized
   - Contextualization of results

2. **Comparison with Literature**
   - Performance benchmarking vs. published models
   - Fairness contrast with documented algorithmic bias
   - Explanation for equitable performance

3. **Clinical Implications**
   - Scalable screening (calculation example)
   - Early detection potential
   - Reducing disparities
   - Integration into clinical workflow

4. **Methodological Strengths** (5 points)
   - Comprehensive fairness evaluation
   - Transparent feature analysis
   - Demographic-stratified features
   - Rigorous calibration assessment
   - Cross-cycle stability

5. **Limitations** (8 detailed points)
   - Single-site data
   - Small demographic subgroups
   - Pre-trained model
   - Retrospective design
   - Temporal validation gap
   - No external validation
   - Feature analysis limitations
   - No longitudinal analysis

6. **Future Directions** (10 research directions)
   - Multi-site external validation
   - Prospective clinical trial
   - Temporal validation
   - Explainability enhancements
   - Multi-modal integration
   - Longitudinal progression modeling
   - Subgroup robustness
   - Behavioral testing
   - EHR integration
   - Health equity research

7. **Ethical Considerations** (5 topics)
   - Bias mitigation
   - Clinical validation
   - Informed consent
   - Data privacy
   - Transparency and reproducibility

8. **Conclusions**
   - Summary of achievements
   - Framework for responsible AI
   - Future impact statement

#### REFERENCES
13 peer-reviewed citations from:
- JAMA, Science, New England Journal of Medicine
- Journal of the American Medical Informatics Association
- JMIR Medical Informatics
- Nature/NPJ Digital Medicine

---

### 3. Python Script for Presentation Generation
**File**: `create_amia_presentation.py` (15 KB)

**Purpose**: Programmatic PowerPoint generation using python-pptx library

**Features**:
- Automated slide creation with consistent styling
- Functions for title slides, content slides, image slides
- Color scheme: Blue headers (RGB: 0,51,102)
- Professional formatting (font sizes, spacing, alignment)

**How to Use**:
```bash
python3 create_amia_presentation.py
```

**Customization**:
- Edit slide content in the script
- Modify color scheme (RGBColor values)
- Add/remove slides
- Change image paths for figures
- Adjust font sizes and spacing

---

## Key Results Summary

### Overall Performance (N=1,460)
| Metric | Value | 95% CI |
|--------|-------|--------|
| **AUC** | **0.9867** | 0.9818 - 0.9916 |
| **Accuracy** | **94.25%** | 92.92% - 95.42% |
| **Sensitivity** | **97.26%** | 95.78% - 98.38% |
| **Specificity** | **91.78%** | 89.67% - 93.60% |
| **F1 Score** | **0.9383** | 0.9261 - 0.9494 |

### Fairness Analysis
✅ **Gender**: No significant difference (p=0.432)
✅ **Race**: AUC variance 0.0038 (within ±0.05 threshold)
✅ **Ethnicity**: Equitable performance maintained
✅ **Intersectional**: Range <0.01 across Gender × Race

### Top Discriminative Features
1. goal (χ²=4,596)
2. outcome (χ²=4,377)
3. ongoing (χ²=3,696)
4. progressing (χ²=2,738)
5. dementia (χ²=1,850)

---

## How to Use This Package

### For Oral Presentation
1. **Open PowerPoint**: `AMIA_2025_ADRD_ePhenotyping_Presentation.pptx`
2. **Customize**:
   - Add your institution logo (Insert → Picture)
   - Update acknowledgments slide with funding sources
   - Add contact information
   - Adjust speaker notes
3. **Practice**: 10-minute presentation (21 slides = ~30 sec/slide)
4. **Export**: Save as PDF for backup

### For Paper Submission
1. **Open Markdown File**: `AMIA_Paper_Methodology_Results_Discussion.md`
2. **Copy Sections** into AMIA submission template:
   - Methods → Methods section
   - Results → Results section (with tables)
   - Discussion → Discussion section
3. **Add**:
   - Abstract (250 words)
   - Introduction (referencing IMG-20251106-WA000*.jpg proposal)
   - References (expand from provided 13)
4. **Format Tables**: Convert markdown tables to Word/LaTeX format
5. **Insert Figures**: From `figures/` and `figures/demographic/` directories
6. **Proofread**: Check word limit (4 pages for AMIA)

### For Poster Presentation
1. **Extract Key Results** from paper sections
2. **Use Visualizations**:
   - figures/AUC_CNNr.png (ROC curve)
   - figures/confusion_matrix.png
   - figures/calibration_plot.png
   - figures/demographic/auc_by_subgroup_enhanced.png
   - figures/demographic/intersectional_heatmap.png
3. **Create Sections**:
   - Background (from Discussion)
   - Methods (condensed from Methods section)
   - Results (Tables 1-7, visualizations)
   - Conclusions (from Discussion/Conclusions)

---

## Figures Available

### Main Figures (`figures/`)
- `AUC_CNNr.png` - ROC curves (10 model cycles)
- `AUC_CNNr_zoom.png` - Zoomed ROC
- `confusion_matrix.png` - Confusion matrix heatmap
- `calibration_plot.png` - Calibration curve
- `probability_distribution.png` - Predicted probability distributions
- `metrics_boxplot.png` - Performance metrics across cycles

### Demographic Figures (`figures/demographic/`)
- `auc_by_subgroup_enhanced.png` - AUC across demographic groups
- `intersectional_heatmap.png` - Gender × Race performance
- `metrics_comparison.png` - Metrics by demographic
- `sensitivity_specificity.png` - Sen/Spec by group
- `null_distribution_gender.png` - Permutation test (gender)
- `null_distribution_race.png` - Permutation test (race)
- `null_distribution_ethnicity.png` - Permutation test (ethnicity)

---

## AMIA Submission Guidelines

### Paper Track
- **Length**: 4 pages (excluding references)
- **Format**: 12pt Times New Roman, 1-inch margins
- **Sections**: Abstract, Introduction, Methods, Results, Discussion, References
- **Tables**: Max 3-4 tables
- **Figures**: Max 3-4 figures
- **Deadline**: [Check AMIA website]

### Podium Presentation
- **Duration**: 10 minutes presentation + 2 minutes Q&A
- **Slides**: 15-25 slides recommended
- **Practice**: Rehearse timing
- **Backup**: Bring PDF version

### Poster Track
- **Size**: 4 ft (height) × 6 ft (width) typical
- **Sections**: Background, Methods, Results, Discussion, Conclusions
- **Font**: Readable from 6 feet away (≥28pt body text)
- **Visuals**: Emphasize figures over text

---

## Citation Information

### Suggested Citation
```
Gyasi F, Obeid J, et al. Deep Learning-Based ADRD ePhenotyping with Comprehensive
Fairness Analysis. AMIA 2025 Annual Symposium, [City, Date]. 2025.
```

### BibTeX
```bibtex
@inproceedings{gyasi2025adrd,
  title={Deep Learning-Based ADRD ePhenotyping with Comprehensive Fairness Analysis},
  author={Gyasi, Frederick and Obeid, Jihad and others},
  booktitle={AMIA 2025 Annual Symposium},
  year={2025}
}
```

---

## Contact Information

**Principal Investigator**: Frederick Gyasi
- **Email**: [your-email@institution.edu]
- **GitHub**: https://github.com/gyasifred/adrd_ePheno
- **Institution**: [Your Institution]

**Model Development**: Jihad Obeid
- **Email**: [jihad-email@institution.edu]

---

## Repository Structure

```
adrd_ePheno/
├── AMIA_2025_ADRD_ePhenotyping_Presentation.pptx  # PowerPoint (21 slides)
├── AMIA_Paper_Methodology_Results_Discussion.md    # Paper sections (12K words)
├── create_amia_presentation.py                     # Python script for PPTX
├── figures/                                        # Main visualizations
│   ├── AUC_CNNr.png
│   ├── confusion_matrix.png
│   ├── calibration_plot.png
│   └── ...
├── figures/demographic/                            # Fairness visualizations
│   ├── auc_by_subgroup_enhanced.png
│   ├── intersectional_heatmap.png
│   └── ...
├── results/                                        # Analysis results
│   ├── evaluation_report.txt
│   ├── predictions_df.csv
│   └── demographic/
│       ├── demographic_analysis_report.txt
│       └── subgroup_performance.xlsx
├── data/                                           # Dataset (not in repo)
├── models/                                         # Trained models
└── *.R                                             # Analysis scripts
```

---

## License

This research code and materials are open source under MIT License.
See LICENSE file for details.

---

## Acknowledgments

- **Funding**: [Your funding source]
- **Data**: [Your institution] Electronic Health Records
- **IRB**: Protocol #[number]
- **Compute**: [HPC/GPU resources used]

---

## Version History

- **v1.0** (2025-11-25): Initial AMIA submission package
  - 21-slide PowerPoint presentation
  - 12,000-word paper sections (Methods, Results, Discussion)
  - Python script for programmatic PPTX generation
  - Complete fairness analysis
  - Feature interpretability analysis

---

## Next Steps

1. **Review Materials**: Read through paper sections and presentation
2. **Customize**: Add institution-specific information
3. **Proofread**: Check for typos, formatting
4. **Practice**: Rehearse 10-minute presentation
5. **Submit**: Follow AMIA submission portal instructions
6. **Prepare Q&A**: Anticipate questions on fairness, generalizability, clinical implementation

**Good luck with your AMIA submission! 🎯**
