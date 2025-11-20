# TF-IDF Explanation for ADRD Feature Analysis (Aim 2)

**Author**: Gyasi, Frederick
**Date**: 2025-11-20
**Purpose**: Explanation of TF-IDF methodology for identifying discriminative clinical terms

---

## What is TF-IDF?

**TF-IDF (Term Frequency-Inverse Document Frequency)** is a numerical statistic that reflects how important a word is to a document in a collection.

**Analogy for Epidemiology Students**: Think of TF-IDF as an "exposure metric" - it measures how strongly a word is "exposed" to a particular document class (ADRD vs Control) while accounting for how common it is overall.

---

## The Formula

```
TF-IDF(term, document) = TF(term, document) × IDF(term)
```

### 1. Term Frequency (TF)
How often a word appears in a document:
```
TF = (Count of term in document) / (Total words in document)
```

### 2. Inverse Document Frequency (IDF)
How rare/specific a word is across all documents:
```
IDF = log(Total documents / Documents containing the term)
```

---

## Why TF-IDF Matters for ADRD Classification

**Problem**: Some words are very common but not discriminative
- "the", "is", "patient" appear frequently in BOTH ADRD and Control notes
- These get HIGH TF but LOW IDF → Low TF-IDF

**Solution**: TF-IDF upweights rare, specific terms
- "dementia", "alzheimer" appear frequently in ADRD but rarely in Control
- These get HIGH TF and HIGH IDF → High TF-IDF

---

## Implementation in Your Code

**Location**: `05_aim2_feature_analysis.R`, lines 482-511

```r
# Calculate TF-IDF
adrd_tfidf <- dfm_tfidf(adrd_dfm_grouped)

# Get TF-IDF weights for ADRD documents
tfidf_weights <- convert(adrd_tfidf, "data.frame")
```

**The `dfm_tfidf()` function** from the `quanteda` package:
1. Takes your document-feature matrix (DFM)
2. Calculates TF-IDF for each term
3. Returns weighted matrix

---

## Relationship to Chi-Squared Testing

**Your code uses BOTH** TF-IDF and Chi-squared - they answer different questions:

| Method | Question | Location |
|--------|----------|----------|
| **Chi-squared** | "Is this term statistically overrepresented in ADRD vs Control?" | Lines 226-276 |
| **TF-IDF** | "How important is this term within ADRD documents specifically?" | Lines 482-511 |

**Combined Use**:
1. Chi-squared identifies **statistically significant** discriminative terms
2. TF-IDF ranks them by **importance within documents**

---

## Chi-Squared Implementation Details

**Lines 226-231**: Uses `textstat_keyness()` which performs chi-squared test:
```r
chi2_results <- textstat_keyness(adrd_dfm_grouped,
                                  target = "ADRD",
                                  measure = "chi2")
```

This creates a contingency table for each term:

|          | ADRD Notes | Control Notes |
|----------|------------|---------------|
| Has term | a          | b             |
| No term  | c          | d             |

Then calculates: `χ² = Σ [(Observed - Expected)² / Expected]`

---

## Using TF-IDF for Demographic Comparisons

**Question**: "How can I compare phrase importance between demographic subgroups?"

**Implementation**: PART 1B in `05_aim2_feature_analysis.R` (lines 308-450):

```r
# For each demographic subgroup
for (demo_val in demo_values) {

  # Filter to subgroup
  subgroup_corpus <- full_corpus %>%
    filter(.data[[demo_var]] == demo_val)

  # Run chi-squared within subgroup
  chi2_sub <- textstat_keyness(sub_dfm_grouped,
                                target = "ADRD",
                                measure = "chi2")

  # Get top discriminative terms for this subgroup
  top_terms <- chi2_sub %>% head(20)
}

# Compare terms across subgroups
overlap <- intersect(female_terms, male_terms)
```

---

## Relationship to Behavioral Testing

**Behavioral Testing** tests model sensitivity to term removal:
> "What happens to the prediction when I remove a discriminative term?"

**TF-IDF + Chi-squared identify WHICH terms to test**:

1. **TF-IDF**: Ranks terms by importance
2. **Chi-squared**: Confirms statistical significance
3. **Behavioral Testing**: Measures model sensitivity to term removal

**Workflow**:
```
Chi-squared → Top discriminative terms
     ↓
TF-IDF → Rank by importance
     ↓
Select top 10-20 terms
     ↓
Behavioral Testing → Remove each term, measure Δ prediction
```

---

## Clinical Interpretation Example

**Say TF-IDF identifies these top terms for ADRD**:
1. "dementia" (TF-IDF = 0.85)
2. "memory" (TF-IDF = 0.72)
3. "alzheimer" (TF-IDF = 0.68)
4. "cognitive" (TF-IDF = 0.61)

**And Chi-squared confirms significance**:
- dementia: χ² = 520, p < 0.001
- memory: χ² = 450, p < 0.001

**Interpretation**:
> "The terms 'dementia' and 'memory' are both highly important within ADRD notes (high TF-IDF) and statistically significantly overrepresented compared to Control notes (high χ², p < 0.001). These are strong candidates for behavioral testing to assess model sensitivity."

---

## Summary Table

| Concept | What It Does | Code Location | Output |
|---------|--------------|---------------|--------|
| **Chi-squared** | Tests term overrepresentation | `textstat_keyness()`, line 229 | χ² statistic, p-value |
| **TF-IDF** | Weights terms by importance | `dfm_tfidf()`, line 482 | Weight 0-1 |
| **Demographic Stratification** | Compares terms across groups | Lines 308-450 | Term overlap % |
| **LIME** | Explains individual predictions | Lines 798-804 | Feature weights |
| **Behavioral Testing** | Tests sensitivity to term removal | Lines 1030+ | Δ probability |

---

## Output Files

Your existing code produces these key outputs:

- `results/aim2/chi_squared_results.csv` - All terms with χ² statistics
- `results/aim2/discriminative_terms.xlsx` - Top terms for ADRD/Control
- `results/aim2/demographic_chi2_comparison.csv` - Terms by demographic subgroup

---

## Key Takeaways

1. **TF-IDF** measures term importance within documents
2. **Chi-squared** tests statistical significance of term associations
3. **Use both together** for robust feature identification
4. **Demographic stratification** reveals if different terms drive predictions for different groups
5. **Behavioral testing** validates which terms actually affect model predictions
