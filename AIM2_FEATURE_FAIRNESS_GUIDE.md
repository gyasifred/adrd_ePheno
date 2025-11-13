# AIM 2: Feature-Level Fairness Analysis

**Author**: Gyasi, Frederick
**Script**: `05_aim2_feature_analysis.R`
**Version**: 2.0
**Purpose**: Comprehensive guide to feature-level fairness for ADRD ePhenotyping

---

## Table of Contents

1. [Research Objectives](#research-objectives)
2. [Research Questions](#research-questions)
3. [Implementation Overview](#implementation-overview)
4. [Statistical Methods Explained](#statistical-methods-explained)
5. [How to Interpret Results](#how-to-interpret-results)
6. [Example Interpretation](#example-interpretation)
7. [Feature Fairness Framework](#feature-fairness-framework)
8. [Output Files Reference](#output-files-reference)

---

## Research Objectives

### Primary Objective

**Identify cohort-specific linguistic features that drive ADRD predictions and evaluate whether these features differ across demographic subgroups.**

### Why This Matters

**Aim 1** (04_demographic_analysis.R) answers:
> *"Does the model PERFORM equitably?"* (Output fairness)

**Aim 2** (05_aim2_feature_analysis.R) answers:
> *"Do the FEATURES driving predictions differ by demographics?"* (Input fairness)

### The Fairness Gap

**Scenario**: Aim 1 shows equal AUC (0.90) for both Female and Male patients. ✅ Fair performance?

**BUT** Aim 2 reveals:
- **Female ADRD** detected via: "memory loss", "forgetful", "misplacing items"
- **Male ADRD** detected via: "dementia", "alzheimer", "wandering"

**Implication**: Model learns **demographic-specific linguistic patterns**, indicating:
1. **Differential clinical documentation** (physicians describe symptoms differently)
2. **Stereotypical associations** (gender-correlated symptom documentation)
3. **Hidden bias** (model relies on demographic proxies even if demographics aren't input features)
4. **Unequal explainability** (model decisions interpretable for majority but opaque for minorities)

---

### Secondary Objectives

1. **Identify discriminative linguistic features** for ADRD vs Control
2. **Test whether features differ by demographics** (Gender, Race, Ethnicity)
3. **Quantify model explanation consistency** across subgroups (LIME analysis)
4. **Assess behavioral sensitivity** to term removal
5. **Detect documentation bias** in clinical notes

---

## Research Questions

### **Question 1: Overall Feature Identification**

**Q1**: Which linguistic features (words/terms) are most discriminative of ADRD vs Control?

**Methods**:
- **Chi-squared (χ²) testing**: Statistical keyness
- **TF-IDF weighting**: Term importance

**Expected Outputs**:
- Top 100 discriminative terms (e.g., "dementia", "alzheimer", "memory", "cognitive")
- p-values for statistical significance
- Effect sizes (chi-squared statistic)

---

### **Question 2: Demographic-Stratified Feature Analysis**

**Q2a**: Do discriminative features differ by gender?
**Q2b**: Do discriminative features differ by race?
**Q2c**: Do discriminative features differ by ethnicity?

**Hypothesis**:
- **H₀**: Top discriminative terms are consistent across demographic subgroups
- **H₁**: Discriminative terms differ significantly, indicating differential documentation patterns

**Operationalization**:
- Calculate **term overlap** across subgroups
- **Low overlap (<50%)** → Different linguistic patterns by demographics
- **High overlap (>70%)** → Consistent documentation

**Example**:
```
Top 10 terms for Female ADRD:  [memory, forgetful, confusion, ...]
Top 10 terms for Male ADRD:    [dementia, alzheimer, wandering, ...]

Overlap: 4/10 (40%) → LOW → ⚠️ Differential documentation
```

---

### **Question 3: Model Explainability Consistency**

**Q3**: Are LIME explanations consistent across demographic subgroups?

**What LIME Does**: For individual predictions, LIME identifies which words contributed most to the ADRD classification

**Fairness Question**: *"Does the model use different features to explain predictions for different demographic groups?"*

**Hypothesis**:
- **H₀**: LIME feature importance is consistent across demographics
- **H₁**: LIME features differ, indicating the model "explains itself differently" by demographics

**Why This Matters**:
- If LIME shows different features for different groups → **Unequal explainability**
- Example: Model clearly explains ADRD predictions for White patients ("alzheimer" + "memory") but provides opaque explanations for Black patients (many low-weight features)
- **Clinical Impact**: Limits physician trust and adoption in diverse settings

---

### **Question 4: Behavioral Sensitivity**

**Q4**: Is model sensitivity to term removal consistent across demographics?

**Method**: Remove discriminative terms and measure prediction change (Δ probability)

**Fairness Concern**: If certain demographic groups rely on fewer "fragile" features, the model may be less robust for those groups

---

## Implementation Overview

### Analysis Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT: Clinical Notes + Model Predictions                   │
│   - train_set.rds / test_set.rds (text + demographics)     │
│   - predictions_df.csv (ADRD predictions)                   │
│   - Best CNN model from models/ directory                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ PART 1: Overall Chi-Squared Feature Analysis                │
│   - Corpus creation (ADRD vs Control)                       │
│   - Tokenization, stopword removal                          │
│   - Chi-squared keyness testing                             │
│   - Top 100 discriminative terms                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ PART 1B: Demographic-Stratified Chi-Squared (NEW v2.0)      │
│   - For each demographic (Gender, Race, Ethnicity):         │
│     * Subset corpus to demographic subgroup                 │
│     * Run chi-squared ADRD vs Control                       │
│     * Extract top 20 terms                                  │
│     * Calculate term overlap across subgroups               │
│   - Interpretation: Low overlap → Differential patterns     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ PART 2-5: TF-IDF & Visualization                            │
│   - TF-IDF weighting                                        │
│   - Word clouds (ADRD vs Control)                           │
│   - Frequency plots                                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ PART 6: LIME Explainability                                 │
│   - Load trained CNN model                                  │
│   - Generate LIME explanations for top ADRD predictions     │
│   - Feature importance weights                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ PART 6C: Demographic-Stratified LIME (NEW v2.0)             │
│   - For each demographic subgroup:                          │
│     * Select top 10 confident ADRD predictions              │
│     * Generate LIME explanations                            │
│     * Aggregate feature importance                          │
│     * Calculate overlap across subgroups                    │
│   - Interpretation: Low overlap → Different explanation patterns │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ PART 7: Behavioral Testing Framework                        │
│   - Identify discriminative terms for removal               │
│   - Template script for term removal testing                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ PART 8: Demographic Feature Fairness Summary (NEW v2.0)     │
│   - Consolidate chi-squared term overlap findings           │
│   - Consolidate LIME feature importance findings            │
│   - Fairness assessment report                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ OUTPUT FILES                                                 │
│   - chi_squared_results.csv                                 │
│   - demographic_chi2_stratified.rds                         │
│   - demographic_chi2_comparison.csv                         │
│   - demographic_lime_stratified.rds                         │
│   - demographic_lime_comparison.csv                         │
│   - Figures: word clouds, frequency plots, LIME plots       │
└─────────────────────────────────────────────────────────────┘
```

---

## Statistical Methods Explained

### 1. Chi-Squared Keyness Testing

**What It Does**: Identifies which words are statistically overrepresented in ADRD notes compared to Control notes

**Formula**:
```
χ² = Σ [(Observed - Expected)² / Expected]

For each term:
  Observed_ADRD = frequency in ADRD corpus
  Observed_Control = frequency in Control corpus
  Expected = frequency if term was equally distributed
```

**Contingency Table for Term "memory"**:

|         | ADRD Notes | Control Notes | Total |
|---------|------------|---------------|-------|
| "memory"| xx         | xx            | xx    |
| Other   | xx         | xx            | xx    |
| **Total** | xx       | xx            | xx    |

**Chi-squared test**: *"Is 'memory' more frequent in ADRD than Control beyond chance?"*

**Output**:
- **χ² statistic**: Higher = stronger association
- **p-value**: p < 0.05 → statistically significant overrepresentation

**Interpretation**:
```
Term: "dementia"
  χ² = 450.2
  p < 0.001
  → HIGHLY discriminative of ADRD (appears much more in ADRD notes)

Term: "the"
  χ² = 0.3
  p = 0.58
  → NOT discriminative (common in both ADRD and Control)
```

---

### 2. Demographic-Stratified Chi-Squared

**Purpose**: Test whether discriminative terms differ by demographic subgroup

**Method**:
1. **Subset data** by demographic (e.g., Female patients only)
2. **Run chi-squared** on Female ADRD vs Female Control
3. **Extract top 20 terms** for Female subgroup
4. **Repeat** for Male subgroup
5. **Compare term overlap**:
   ```
   Overlap = |Top_Female ∩ Top_Male| / 10 × 100%
   ```

**Interpretation**:
- **Overlap ≥ 70%**: Consistent documentation (✅ Fair)
- **Overlap 50-69%**: Moderate differences (⚠️ Monitor)
- **Overlap < 50%**: Different linguistic patterns (🚨 Differential documentation)

**Example**:
```
Top 10 Female ADRD terms:   [memory, forgetful, confusion, dementia, ...]
Top 10 Male ADRD terms:     [dementia, alzheimer, wandering, agitation, ...]

Common terms: [dementia, alzheimer, memory] → 3/10 = 30%

⚠️ FINDING: Low overlap (30%) suggests differential documentation
   Female notes emphasize subjective memory complaints
   Male notes emphasize clinical diagnoses
```

---

### 3. TF-IDF Weighting

**What It Does**: Quantifies term importance by balancing frequency and specificity

**Formula**:
```
TF-IDF(term, document) = TF(term, document) × IDF(term)

TF = Term Frequency in document
IDF = log(Total documents / Documents containing term)
```

**Intuition**:
- Terms that appear **frequently** in a document (high TF) but **rarely** across all documents (high IDF) get high weights
- Example: "dementia" in ADRD note → High TF-IDF (frequent in this note, rare overall)
- Example: "the" → Low TF-IDF (frequent everywhere)

**Use**: Identify terms that uniquely characterize ADRD vs Control

---

### 4. LIME (Local Interpretable Model-agnostic Explanations)

**What It Does**: Explains individual predictions by identifying which words contributed most to the classification

**Method** (simplified):
1. **Select a prediction** to explain (e.g., Patient X predicted as ADRD with 92% probability)
2. **Create perturbations**: Generate similar texts by removing words
3. **Get predictions** on perturbed texts
4. **Fit local model**: Linear model relating word presence to prediction change
5. **Extract weights**: Which words increased/decreased ADRD probability

**Output** (for Patient X):
```
Feature Weights:
  "alzheimer"  +0.35  (strongly increases ADRD probability)
  "memory"     +0.28
  "dementia"   +0.22
  "loss"       +0.15
  "cognitive"  +0.12
  ...
  "normal"     -0.10  (decreases ADRD probability)
```

**Interpretation**: This patient was classified as ADRD primarily due to words: "alzheimer", "memory", "dementia"

---

### 5. Demographic-Stratified LIME

**Purpose**: Test whether LIME explanations differ by demographic subgroup

**Method**:
1. **Subset predictions** by demographic (e.g., Female ADRD patients)
2. **Select top 10 confident** Female ADRD predictions
3. **Generate LIME explanations** for each
4. **Aggregate feature importance**:
   ```
   Mean weight of "memory" = mean across 10 Female explanations
   ```
5. **Repeat** for Male ADRD patients
6. **Compare feature overlap**:
   ```
   Top 10 Female LIME features: [alzheimer, memory, dementia, ...]
   Top 10 Male LIME features:   [alzheimer, dementia, wandering, ...]

   Overlap = 6/10 = 60%
   ```

**Interpretation**:
- **Overlap < 40%**: Model uses **different features** to explain predictions for different demographics
- **Implication**: Unequal explainability → Physicians may trust model less for certain groups

**Example Finding**:
```
Female LIME features: memory (weight=0.30), forgetful (0.25), confusion (0.22)
Male LIME features:   dementia (0.35), alzheimer (0.28), wandering (0.20)

Overlap: 2/10 = 20% ⚠️ LOW

INTERPRETATION: Model "thinks differently" about Female vs Male ADRD patients
  → Female: Relies on subjective memory complaints
  → Male: Relies on clinical diagnoses
  → Potential bias in feature learning
```

---

## How to Interpret Results

### Step-by-Step Interpretation Guide

---

#### **PART 1: Overall Chi-Squared Results**

**File**: `results/aim2/chi_squared_results.csv`

**Columns**:
- `feature`: Word/term
- `chi2`: Chi-squared statistic (higher = more discriminative)
- `p`: p-value (< 0.05 = statistically significant)
- `n_target`: Frequency in ADRD notes
- `n_reference`: Frequency in Control notes

**Look For**:
```
Top 20 ADRD-associated terms (sorted by chi2):
  1. dementia       χ²=xx, p<0.001, n_ADRD=xx, n_CTRL=xx
  2. alzheimer      χ²=xx, p<0.001, n_ADRD=xx, n_CTRL=xx
  3. memory         χ²=xx, p<0.001, n_ADRD=xx, n_CTRL=xx
  ...
```

**Interpretation**:
1. **High χ² (>100)** + **p < 0.001**: Strong ADRD marker
2. **Frequency ratio**: If n_ADRD >> n_CTRL → Term is ADRD-specific
3. **Clinical validation**: Do top terms align with clinical understanding of ADRD?
   - ✅ "dementia", "alzheimer", "memory", "cognitive" → Clinically valid
   - ⚠️ Unexpected terms (e.g., "patient", "history") → May be artifacts

**Questions to Ask**:
- Are top discriminative terms clinically meaningful?
- Are there spurious correlations (e.g., administrative terms)?
- Do terms reflect actual ADRD symptomatology?

---

#### **PART 1B: Demographic-Stratified Chi-Squared**

**Files**:
- `results/aim2/demographic_chi2_stratified.rds` (R object)
- `results/aim2/demographic_chi2_comparison.csv` (table)

**Look For** (in console output):
```
GENDER ANALYSIS
---------------
Analyzing: GENDER
Subgroups: Female, Male

  Analyzing Female (N = xx : ADRD=xx, CTRL=xx)
    Top 5 discriminative terms: memory, forgetful, confusion, dementia, loss

  Analyzing Male (N = xx : ADRD=xx, CTRL=xx)
    Top 5 discriminative terms: dementia, alzheimer, wandering, cognitive, decline

  Comparing discriminative terms across GENDER subgroups:
    Common terms (top 10): 6 / 10
      dementia, alzheimer, memory, cognitive, loss, patient

    Unique to Female: forgetful, misplacing, confusion
    Unique to Male: wandering, agitation, behavior

    ✓ Good term overlap (60%) - consistent patterns across GENDER
```

**Interpretation**:

**Overlap ≥ 70%**: ✅ **Consistent Documentation**
- Model learns similar features for both groups
- Fair feature-level representation
- **Example**: Male and Female both rely on "dementia", "memory", "cognitive"

**Overlap 50-69%**: ⚠️ **Moderate Differences**
- Some linguistic divergence
- **Action**: Document as limitation, investigate unique terms
- **Example**: Female=60% overlap → Some gender-specific language but mostly consistent

**Overlap < 50%**: 🚨 **Differential Documentation**
- Substantially different linguistic patterns
- **Potential causes**:
  1. **Stereotypical documentation**: Physicians describe symptoms differently by gender
  2. **Symptom presentation differences**: True clinical differences in ADRD manifestation
  3. **Documentation bias**: Gendered language in medical notes
- **Action**: Qualitative review of notes, consider separate models or recalibration

**Example Red Flag**:
```
Overlap: 30%

Unique to Female: forgetful, misplacing, memory loss, repeating
Unique to Male: dementia, alzheimer, diagnosis, cognitive impairment

INTERPRETATION:
  Female notes → Subjective complaints ("forgetful", "misplacing")
  Male notes → Clinical diagnoses ("dementia", "alzheimer")

POTENTIAL BIAS:
  Physicians may document female memory complaints as subjective symptoms
  but document male complaints as clinical diagnoses
  → Gender bias in documentation practices
```

---

#### **PART 6C: Demographic-Stratified LIME**

**Files**:
- `results/aim2/demographic_lime_stratified.rds`
- `results/aim2/demographic_lime_comparison.csv`

**Look For** (in console output):
```
LIME ANALYSIS BY: GENDER
------------------------
  Generating LIME explanations for Female (10 cases)
    Top 5 important features: memory, loss, forgetful, dementia, confusion
    Mean weights: 0.28, 0.22, 0.20, 0.18, 0.15

  Generating LIME explanations for Male (10 cases)
    Top 5 important features: alzheimer, dementia, cognitive, decline, impairment
    Mean weights: 0.32, 0.28, 0.24, 0.19, 0.16

  Comparing LIME feature importance across GENDER:
    Common important features: 4 / 10
      dementia, memory, cognitive, loss

    ✓ Good overlap (40%) - consistent features across GENDER
```

**Interpretation**:

**Overlap ≥ 60%**: ✅ **Consistent Explanations**
- Model uses similar features to explain predictions for both groups
- Equal explainability

**Overlap 40-59%**: ⚠️ **Moderate Explanation Differences**
- Some divergence in how model explains itself
- **Action**: Review unique features, document as limitation

**Overlap < 40%**: 🚨 **Unequal Explainability**
- Model "thinks differently" about different demographic groups
- **Clinical Impact**: Physicians may trust/understand model predictions differently for different patient populations
- **Example**:
  ```
  Female LIME: memory (0.30), forgetful (0.25), confusion (0.22)
    → Focuses on subjective symptoms

  Male LIME: alzheimer (0.35), dementia (0.32), diagnosis (0.28)
    → Focuses on diagnostic terms

  Overlap: 20% (only "dementia", "memory" in common)

  ⚠️ FINDING: Model relies on different feature sets by gender
     → Potential hidden gender bias in feature learning
  ```

**Clinical Consideration**:
- If LIME shows high weights for **subjective terms** (forgetful, confusion) in one group but **diagnostic terms** (alzheimer, dementia) in another:
  - **Risk**: Model may be less robust for the subjective-term group (fewer "hard" diagnostic signals)
  - **Action**: Validate with additional clinical review for affected group

---

#### **PART 8: Demographic Feature Fairness Summary**

**File**: Console output at end of script

**Look For**:
```
================================================================================
PART 8: Demographic Feature Fairness Summary
================================================================================

DISCRIMINATIVE TERMS BY DEMOGRAPHICS
----------------------------------------

GENDER:
  Female: memory, forgetful, confusion, dementia, loss
  Male: dementia, alzheimer, wandering, cognitive, decline

RACE:
  White: alzheimer, dementia, memory, cognitive, decline
  Black: dementia, memory, cognitive, loss, impairment

LIME FEATURE IMPORTANCE BY DEMOGRAPHICS
----------------------------------------

GENDER:
  Female: memory, loss, forgetful, confusion, cognitive
  Male: alzheimer, dementia, cognitive, decline, impairment

FAIRNESS ASSESSMENT:
----------------------------------------
✓ Demographic-stratified chi-squared analysis complete
✓ Demographic-stratified LIME analysis complete

KEY QUESTIONS ANSWERED:
1. Do discriminative features differ by demographics? → See chi2 comparison
2. Do LIME explanations differ by demographics? → See LIME comparison
3. Are certain groups explained differently? → Check overlap percentages

OUTPUT FILES:
  - demographic_chi2_stratified.rds
  - demographic_chi2_comparison.csv
  - demographic_lime_stratified.rds
  - demographic_lime_comparison.csv

NEXT STEPS:
1. Review chi-squared term overlap across demographic groups
2. Examine LIME feature importance differences
3. If low overlap (<50%), investigate differential documentation patterns
4. Consider demographic-specific model recalibration if needed
```

**Interpretation**:

**Review Checklist**:
1. **Chi-squared term overlap**:
   - Compare top 5 terms across demographics
   - Identify unique terms per group
   - Question: *"Why are these terms unique? Clinical reality or documentation bias?"*

2. **LIME feature overlap**:
   - Compare top 5 LIME features
   - Question: *"Does the model explain predictions similarly across groups?"*

3. **Combined assessment**:
   - If **both** chi-squared and LIME show low overlap → Strong evidence of differential patterns
   - If **only one** shows low overlap → Investigate further

---

## Example Interpretation

### Scenario: Complete Analysis Results

---

**OVERALL CHI-SQUARED RESULTS**:
```
Top 10 ADRD-discriminative terms:
  1. dementia       (χ²=520, p<0.001)
  2. alzheimer      (χ²=480, p<0.001)
  3. memory         (χ²=450, p<0.001)
  4. cognitive      (χ²=380, p<0.001)
  5. loss           (χ²=350, p<0.001)
  ...
```
→ **Interpretation**: Clinically valid terms, strong statistical significance

---

**DEMOGRAPHIC-STRATIFIED CHI-SQUARED (Gender)**:
```
Female Top 10: memory, forgetful, confusion, misplacing, dementia, loss, cognitive, repeating, decline, problems
Male Top 10:   dementia, alzheimer, wandering, cognitive, agitation, memory, loss, behavior, decline, impairment

Common terms: dementia, memory, cognitive, loss, decline → 5/10 = 50%
Unique to Female: forgetful, confusion, misplacing, repeating, problems
Unique to Male: alzheimer, wandering, agitation, behavior, impairment
```

**Interpretation**:
1. ⚠️ **Moderate overlap (50%)** - at threshold for concern
2. **Female-unique terms**: Subjective complaints ("forgetful", "confusion", "misplacing")
3. **Male-unique terms**: Diagnostic labels ("alzheimer"), behavioral symptoms ("wandering", "agitation")
4. **Hypothesis**: Differential documentation → Female symptoms documented as subjective complaints, Male symptoms documented as objective signs

**Action Items**:
- Qualitative review of random sample of Female vs Male ADRD notes
- Investigate: Is this true clinical difference or documentation bias?
- Document as potential limitation in manuscript
- Consider: *"Does this affect model robustness?"* (Subjective terms may be less reliable signals)

---

**DEMOGRAPHIC-STRATIFIED LIME (Gender)**:
```
Female LIME top 10:  memory (0.28), loss (0.22), forgetful (0.20), confusion (0.18), dementia (0.16), ...
Male LIME top 10:    alzheimer (0.32), dementia (0.28), cognitive (0.24), wandering (0.20), decline (0.18), ...

Common features: dementia, memory, loss, cognitive → 4/10 = 40%
```

**Interpretation**:
1. ⚠️ **Low overlap (40%)** - model uses different features by gender
2. **Female LIME**: Heavy reliance on "memory", "loss", "forgetful" (subjective terms)
3. **Male LIME**: Heavy reliance on "alzheimer", "dementia" (diagnostic labels)
4. **Concern**: Model may explain Female predictions via subjective symptoms, Male predictions via hard diagnoses
   - **Risk**: Unequal physician trust (diagnostic terms are more "credible" than subjective complaints)

**Action Items**:
- Highlight as key fairness finding in manuscript
- Discussion point: *"Model learns gendered documentation patterns, potentially reflecting societal biases in medical documentation"*
- Recommendation: External validation with diverse datasets to confirm/refute pattern

---

**COMBINED ASSESSMENT**:

| Analysis         | Overlap | Interpretation                    |
|------------------|---------|-----------------------------------|
| Chi-squared      | 50%     | Moderate differential documentation |
| LIME             | 40%     | Different explanation patterns    |
| **Overall**      | **45%** | 🚨 **Feature-level bias detected** |

**Manuscript Text** (suggested):
> *"Demographic-stratified feature analysis revealed significant differences in linguistic patterns by gender. Female ADRD notes were characterized by subjective memory complaints (e.g., 'forgetful', 'confusion'), while male notes emphasized clinical diagnoses (e.g., 'alzheimer', 'dementia'). Chi-squared term overlap was 50%, and LIME explanation overlap was 40%, indicating the model learns gender-specific documentation patterns. This suggests potential gender bias in clinical documentation practices, which the model subsequently encodes. Future work should investigate whether these patterns reflect true clinical differences or stereotypical documentation biases."*

---

## Feature Fairness Framework

### Decision Tree for Action Items

```
Are top chi-squared terms clinically valid?
│
├─ NO → Investigate spurious correlations, data quality issues
│
└─ YES → Check demographic-stratified chi-squared:
    │
    ├─ Term overlap ≥ 70%?
    │   └─ YES → ✅ Consistent features across demographics
    │
    ├─ Term overlap 50-69%?
    │   └─ MODERATE DIFFERENCES:
    │       • Qualitative review of unique terms
    │       • Document as limitation
    │       • Consider external validation
    │
    └─ Term overlap < 50%?
        └─ DIFFERENTIAL DOCUMENTATION:
            • Qualitative chart review (random sample)
            • Investigate: Clinical reality vs documentation bias?
            • Check LIME overlap:
                │
                ├─ LIME overlap < 40%?
                │   └─ 🚨 SEVERE FEATURE BIAS:
                │       • Model learns demographic-specific patterns
                │       • Major manuscript finding
                │       • Ethical discussion required
                │       • Consider demographic-stratified models
                │
                └─ LIME overlap ≥ 40%?
                    └─ ⚠️ MODERATE CONCERN:
                        • Document in limitation section
                        • Discuss documentation practices
```

---

### Fairness Red Flags

**🚨 Severe Concerns** (require intervention):
1. **Chi-squared overlap < 40%** + **LIME overlap < 30%**
   - Model fundamentally learns different features by demographics
   - Strong evidence of hidden bias

2. **Subjective vs Diagnostic terms**:
   - One group: "forgetful", "confusion" (subjective)
   - Other group: "alzheimer", "dementia" (diagnostic)
   - **Risk**: Model less robust for subjective-term group

3. **Stereotypical associations**:
   - Female: "emotional", "anxious" (gendered language)
   - Male: "aggressive", "wandering" (gendered behavior descriptions)
   - **Risk**: Model encodes societal stereotypes

---

### Reporting Checklist for Manuscripts

**Table 1: Overall Feature Analysis**
- [ ] Top 20 chi-squared terms (ADRD vs Control)
- [ ] Chi-squared statistics, p-values
- [ ] Clinical validation of top terms

**Table 2: Demographic-Stratified Chi-Squared**
- [ ] Top 10 terms per demographic subgroup
- [ ] Term overlap percentages
- [ ] Unique terms per group

**Table 3: LIME Feature Importance**
- [ ] Top 10 LIME features overall
- [ ] Demographic-stratified LIME features
- [ ] Feature overlap percentages

**Figure 1: Word Clouds**
- [ ] ADRD word cloud
- [ ] Control word cloud
- [ ] Visual comparison of term prominence

**Figure 2: Chi-Squared Term Frequency**
- [ ] Bar plot of top discriminative terms
- [ ] Sorted by chi-squared statistic

**Figure 3: LIME Explanations**
- [ ] Feature weight visualization
- [ ] Example individual explanations

**Discussion Points**:
- [ ] Clinical validity of discriminative features
- [ ] Interpretation of demographic differences (clinical vs documentation bias?)
- [ ] Implications for model fairness
- [ ] Ethical considerations (stereotypical associations?)
- [ ] Limitations (sample size, single institution)
- [ ] Recommendations (external validation, qualitative review)

---

## Output Files Reference

### Results Directory: `results/aim2/`

| File                                  | Contents                                      |
|---------------------------------------|-----------------------------------------------|
| `chi_squared_results.csv`             | Overall chi-squared keyness results           |
| `discriminative_terms.xlsx`           | Top ADRD/Control terms (multi-sheet)         |
| `demographic_chi2_stratified.rds`     | Demographic-stratified chi-squared (R object) |
| `demographic_chi2_comparison.csv`     | Demographic chi-squared comparison table      |
| `demographic_lime_stratified.rds`     | Demographic-stratified LIME (R object)        |
| `demographic_lime_comparison.csv`     | Demographic LIME comparison table             |
| `lime_explanations.csv`               | Individual LIME explanations                  |
| `behavioral_test_terms.rds`           | Terms for behavioral testing                  |

---

### Figures Directory: `figures/aim2/`

| File                          | Visualization                       |
|-------------------------------|-------------------------------------|
| `wordcloud_adrd.png`          | ADRD-specific word cloud            |
| `wordcloud_control.png`       | Control word cloud                  |
| `chi_squared_top_terms.png`   | Top discriminative terms bar plot   |
| `lime_explanations.png`       | LIME feature weights visualization  |
| `tfidf_comparison.png`        | TF-IDF weighted terms               |

---

## Integration with Aim 1

### Combined Fairness Assessment

**Aim 1** (Performance Fairness) + **Aim 2** (Feature Fairness) = **Comprehensive Fairness Evaluation**

**Scenario 1: Fair Performance + Fair Features** ✅
- Aim 1: AUC equal across demographics
- Aim 2: Feature overlap ≥ 70%
- **Conclusion**: Model is fair at both output and input levels

---

**Scenario 2: Fair Performance + Unfair Features** ⚠️
- Aim 1: AUC equal across demographics
- Aim 2: Feature overlap < 50%
- **Interpretation**: Model achieves equal performance but uses different reasoning for different groups
- **Risk**: "Right answer, wrong reasons" → May not generalize, unequal explainability
- **Example**: Female AUC = Male AUC = 0.90, but model uses "forgetful" for Females, "alzheimer" for Males

---

**Scenario 3: Unfair Performance + Fair Features** ⚠️
- Aim 1: AUC differs significantly by demographics
- Aim 2: Feature overlap ≥ 70%
- **Interpretation**: Model uses consistent features but performs differently
- **Possible causes**: Class imbalance in demographics, different base rates
- **Action**: Investigate performance disparities (Aim 1 analysis)

---

**Scenario 4: Unfair Performance + Unfair Features** 🚨
- Aim 1: AUC differs significantly
- Aim 2: Feature overlap < 50%
- **Interpretation**: Severe fairness issues at both levels
- **Action**: Major intervention needed (recalibration, stratified models, external validation)

---

## Additional Resources

**Related Documentation**:
- `AIM1_DEMOGRAPHIC_FAIRNESS_GUIDE.md` - Performance fairness
- `STATISTICAL_SIGNIFICANCE_METHODOLOGY.md` - Statistical methods
- `README.md` - Pipeline overview

**Script**: `05_aim2_feature_analysis.R`

**Dependencies**:
- `utils_model_loader.R` - Model loading utilities
- `data/train_set.rds`, `data/test_set.rds`
- `results/predictions_df.csv`
- Best CNN model in `models/`

**Support**: Gyasi, Frederick

---

**Version History**:
- v2.0 (2025-11-13): Added demographic-stratified chi-squared and LIME analysis
- v1.0 (2025-11-06): Initial feature analysis implementation

---

**End of AIM 2 Guide**
