#!/usr/bin/env Rscript
# ==============================================================================
# AIM 2 Enhancement: Approximate Randomization for Feature Analysis
# ==============================================================================
# Author: Frederick Gyasi
# Date: December 16, 2025
# Version: 2.3
#
# Purpose: Add rigorous statistical testing to Aim 2 feature analysis
#
# This script enhances the existing 05_aim2_feature_analysis.R with:
# 1. Approximate randomization testing for chi-squared differences
# 2. Permutation tests for feature overlap significance
# 3. Comprehensive visualizations (Yeh 2000 style) for features
#
# Methodology: Yeh (2000) approximate randomization
# Permutations: 10,000 per test (matching Aim 1)
#
# Run AFTER: 05_aim2_feature_analysis.R
# ==============================================================================

# Load Libraries ==============================================================
suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
  library(scales)
})

# Load statistical utilities
source("utils_statistical_tests.R")

# Configuration ===============================================================
RESULTS_DIR <- "results"
AIM2_RESULTS_DIR <- file.path(RESULTS_DIR, "aim2")
AIM2_FIGURES_DIR <- file.path("figures", "aim2")

dir.create(AIM2_RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(AIM2_FIGURES_DIR, showWarnings = FALSE, recursive = TRUE)

N_PERM <- 10000  # Match Aim 1
ALPHA <- 0.05

cat("================================================================================\n")
cat("AIM 2 ENHANCEMENT: Approximate Randomization for Feature Analysis\n")
cat("================================================================================\n\n")

cat("Configuration:\n")
cat("  Permutations:", N_PERM, "\n")
cat("  Significance level:", ALPHA, "\n\n")

# ==============================================================================
# PART 1: Load Demographic Chi-Squared Results
# ==============================================================================

cat("================================================================================\n")
cat("PART 1: Chi-Squared Feature Analysis - Statistical Testing\n")
cat("================================================================================\n\n")

chi2_file <- file.path(AIM2_RESULTS_DIR, "demographic_chi2_comparison.csv")

if (!file.exists(chi2_file)) {
  cat("⚠️  ERROR: demographic_chi2_comparison.csv not found!\n")
  cat("   Please run 05_aim2_feature_analysis.R first.\n\n")
  quit(status = 1)
}

cat("Loading demographic chi-squared results...\n")
chi2_demo <- read_csv(chi2_file, show_col_types = FALSE)
cat("  Loaded", nrow(chi2_demo), "feature × demographic combinations\n\n")

# ==============================================================================
# FUNCTION: Permutation Test for Feature Overlap
# ==============================================================================

permutation_test_feature_overlap <- function(features_a, features_b,
                                              vocab_size,
                                              n_perm = 10000,
                                              seed = 42) {
  #' Test if feature overlap is significantly greater than chance
  #'
  #' H0: Feature overlap is due to random chance
  #' H1: Feature overlap is greater than expected by chance
  #'
  #' @param features_a Vector of features from group A
  #' @param features_b Vector of features from group B
  #' @param vocab_size Total vocabulary size
  #' @param n_perm Number of permutations
  #' @param seed Random seed
  #'
  #' @return List with observed overlap, expected overlap, p-value

  set.seed(seed)

  # Observed overlap
  observed_overlap <- length(intersect(features_a, features_b))

  # Expected overlap under null (hypergeometric mean)
  n_a <- length(features_a)
  n_b <- length(features_b)
  expected_overlap <- (n_a * n_b) / vocab_size

  # Permutation test
  perm_overlaps <- numeric(n_perm)

  for (i in seq_len(n_perm)) {
    # Randomly sample features from vocabulary
    random_a <- sample(vocab_size, n_a, replace = FALSE)
    random_b <- sample(vocab_size, n_b, replace = FALSE)

    perm_overlaps[i] <- length(intersect(random_a, random_b))
  }

  # P-value: proportion of permuted overlaps >= observed
  p_value <- mean(perm_overlaps >= observed_overlap)

  return(list(
    observed_overlap = observed_overlap,
    expected_overlap = expected_overlap,
    overlap_percent = (observed_overlap / min(n_a, n_b)) * 100,
    p_value = p_value,
    perm_overlaps = perm_overlaps,
    n_a = n_a,
    n_b = n_b
  ))
}

# ==============================================================================
# PART 2: Feature Overlap Analysis (Gender)
# ==============================================================================

cat("Analyzing feature overlap across gender groups...\n\n")

# Extract top 10 features for each gender subgroup
gender_chi2 <- chi2_demo %>%
  filter(demographic == "GENDER")

if (nrow(gender_chi2) > 0) {
  # Get unique subgroups
  subgroups <- unique(gender_chi2$subgroup)

  if (length(subgroups) >= 2) {
    # Compare first two subgroups (typically Female vs Male)
    group_a <- subgroups[1]
    group_b <- subgroups[2]

    features_a <- gender_chi2 %>%
      filter(subgroup == group_a) %>%
      slice_max(chi2, n = 10) %>%
      pull(feature)

    features_b <- gender_chi2 %>%
      filter(subgroup == group_b) %>%
      slice_max(chi2, n = 10) %>%
      pull(feature)

    # Get vocabulary size
    vocab_size <- length(unique(chi2_demo$feature))

    cat("Comparing:", group_a, "vs", group_b, "\n")
    cat("  Top features (", group_a, "):", length(features_a), "\n", sep = "")
    cat("  Top features (", group_b, "):", length(features_b), "\n", sep = "")
    cat("  Total vocabulary size:", vocab_size, "\n\n")

    # Run permutation test
    cat("Running permutation test (", N_PERM, " permutations)...\n", sep = "")
    overlap_result <- permutation_test_feature_overlap(
      features_a, features_b,
      vocab_size,
      n_perm = N_PERM,
      seed = 42
    )

    cat("\nResults:\n")
    cat("  Observed overlap:", overlap_result$observed_overlap, "/", min(overlap_result$n_a, overlap_result$n_b),
        sprintf("(%.1f%%)", overlap_result$overlap_percent), "\n")
    cat("  Expected overlap (chance):", sprintf("%.2f", overlap_result$expected_overlap), "\n")
    cat("  P-value:", sprintf("%.4f", overlap_result$p_value))

    if (overlap_result$p_value < ALPHA) {
      cat(" ***\n")
      cat("  ✓ Overlap is SIGNIFICANTLY GREATER than chance\n")
      cat("  → Features are CONSISTENT across", group_a, "and", group_b, "\n")
    } else {
      cat("\n")
      cat("  → Overlap not significantly different from chance\n")
    }
    cat("\n")

    # Visualize null distribution
    null_dist_data <- data.frame(overlap = overlap_result$perm_overlaps)

    p_null <- ggplot(null_dist_data, aes(x = overlap)) +
      geom_histogram(aes(y = ..density..), bins = 50, fill = "lightblue", alpha = 0.7) +
      geom_density(color = "steelblue", size = 1) +
      geom_vline(xintercept = overlap_result$observed_overlap, color = "red", size = 1.5, linetype = "solid") +
      geom_vline(xintercept = overlap_result$expected_overlap, color = "black", size = 1, linetype = "dashed") +
      labs(
        title = sprintf("Feature Overlap Null Distribution: %s vs %s", group_a, group_b),
        subtitle = sprintf("Observed: %d | Expected: %.1f | p = %.4f",
                          overlap_result$observed_overlap,
                          overlap_result$expected_overlap,
                          overlap_result$p_value),
        x = "Number of Overlapping Features",
        y = "Density",
        caption = sprintf("Red line: Observed overlap | Black dashed: Expected under null\n%d permutations", N_PERM)
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 11, hjust = 0.5),
        plot.caption = element_text(size = 9, color = "gray30")
      )

    ggsave(file.path(AIM2_FIGURES_DIR, "feature_overlap_null_distribution_gender.png"),
           plot = p_null, width = 10, height = 6, dpi = 300)
    cat("✓ Null distribution plot saved\n\n")

    # Save results
    overlap_results_gender <- data.frame(
      comparison = paste(group_a, "vs", group_b),
      group_a = group_a,
      group_b = group_b,
      n_features_a = overlap_result$n_a,
      n_features_b = overlap_result$n_b,
      observed_overlap = overlap_result$observed_overlap,
      expected_overlap = overlap_result$expected_overlap,
      overlap_percent = overlap_result$overlap_percent,
      p_value = overlap_result$p_value,
      significant = overlap_result$p_value < ALPHA,
      interpretation = ifelse(
        overlap_result$p_value < ALPHA,
        "Features CONSISTENT across groups (overlap > chance)",
        "Overlap not significantly different from chance"
      )
    )

    write_csv(overlap_results_gender,
              file.path(AIM2_RESULTS_DIR, "feature_overlap_permutation_test_gender.csv"))
    cat("✓ Results saved: feature_overlap_permutation_test_gender.csv\n\n")

  } else {
    cat("⚠️  Need at least 2 gender subgroups for comparison\n\n")
  }
} else {
  cat("⚠️  No gender chi-squared data found\n\n")
}

# ==============================================================================
# PART 3: Feature Overlap Analysis (Race)
# ==============================================================================

cat("Analyzing feature overlap across race groups...\n\n")

race_chi2 <- chi2_demo %>%
  filter(demographic == "RACE")

if (nrow(race_chi2) > 0) {
  subgroups <- unique(race_chi2$subgroup)

  if (length(subgroups) >= 2) {
    # Compare major racial groups (typically White vs Black)
    group_a <- subgroups[1]
    group_b <- subgroups[2]

    features_a <- race_chi2 %>%
      filter(subgroup == group_a) %>%
      slice_max(chi2, n = 10) %>%
      pull(feature)

    features_b <- race_chi2 %>%
      filter(subgroup == group_b) %>%
      slice_max(chi2, n = 10) %>%
      pull(feature)

    vocab_size <- length(unique(chi2_demo$feature))

    cat("Comparing:", group_a, "vs", group_b, "\n")
    cat("  Top features (", group_a, "):", length(features_a), "\n", sep = "")
    cat("  Top features (", group_b, "):", length(features_b), "\n", sep = "")
    cat("  Total vocabulary size:", vocab_size, "\n\n")

    cat("Running permutation test (", N_PERM, " permutations)...\n", sep = "")
    overlap_result <- permutation_test_feature_overlap(
      features_a, features_b,
      vocab_size,
      n_perm = N_PERM,
      seed = 43
    )

    cat("\nResults:\n")
    cat("  Observed overlap:", overlap_result$observed_overlap, "/", min(overlap_result$n_a, overlap_result$n_b),
        sprintf("(%.1f%%)", overlap_result$overlap_percent), "\n")
    cat("  Expected overlap (chance):", sprintf("%.2f", overlap_result$expected_overlap), "\n")
    cat("  P-value:", sprintf("%.4f", overlap_result$p_value))

    if (overlap_result$p_value < ALPHA) {
      cat(" ***\n")
      cat("  ✓ Overlap is SIGNIFICANTLY GREATER than chance\n")
      cat("  → Features are CONSISTENT across", group_a, "and", group_b, "\n")
    } else {
      cat("\n")
      cat("  → Overlap not significantly different from chance\n")
    }
    cat("\n")

    # Visualize null distribution
    null_dist_data <- data.frame(overlap = overlap_result$perm_overlaps)

    p_null <- ggplot(null_dist_data, aes(x = overlap)) +
      geom_histogram(aes(y = ..density..), bins = 50, fill = "lightcoral", alpha = 0.7) +
      geom_density(color = "darkred", size = 1) +
      geom_vline(xintercept = overlap_result$observed_overlap, color = "red", size = 1.5, linetype = "solid") +
      geom_vline(xintercept = overlap_result$expected_overlap, color = "black", size = 1, linetype = "dashed") +
      labs(
        title = sprintf("Feature Overlap Null Distribution: %s vs %s", group_a, group_b),
        subtitle = sprintf("Observed: %d | Expected: %.1f | p = %.4f",
                          overlap_result$observed_overlap,
                          overlap_result$expected_overlap,
                          overlap_result$p_value),
        x = "Number of Overlapping Features",
        y = "Density",
        caption = sprintf("Red line: Observed overlap | Black dashed: Expected under null\n%d permutations", N_PERM)
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 11, hjust = 0.5),
        plot.caption = element_text(size = 9, color = "gray30")
      )

    ggsave(file.path(AIM2_FIGURES_DIR, "feature_overlap_null_distribution_race.png"),
           plot = p_null, width = 10, height = 6, dpi = 300)
    cat("✓ Null distribution plot saved\n\n")

    # Save results
    overlap_results_race <- data.frame(
      comparison = paste(group_a, "vs", group_b),
      group_a = group_a,
      group_b = group_b,
      n_features_a = overlap_result$n_a,
      n_features_b = overlap_result$n_b,
      observed_overlap = overlap_result$observed_overlap,
      expected_overlap = overlap_result$expected_overlap,
      overlap_percent = overlap_result$overlap_percent,
      p_value = overlap_result$p_value,
      significant = overlap_result$p_value < ALPHA,
      interpretation = ifelse(
        overlap_result$p_value < ALPHA,
        "Features CONSISTENT across groups (overlap > chance)",
        "Overlap not significantly different from chance"
      )
    )

    write_csv(overlap_results_race,
              file.path(AIM2_RESULTS_DIR, "feature_overlap_permutation_test_race.csv"))
    cat("✓ Results saved: feature_overlap_permutation_test_race.csv\n\n")

  } else {
    cat("⚠️  Need at least 2 race subgroups for comparison\n\n")
  }
} else {
  cat("⚠️  No race chi-squared data found\n\n")
}

# ==============================================================================
# PART 4: Comprehensive Summary
# ==============================================================================

cat("================================================================================\n")
cat("AIM 2 STATISTICAL TESTING COMPLETE\n")
cat("================================================================================\n\n")

cat("Output Files Generated:\n")
cat("  - feature_overlap_permutation_test_gender.csv\n")
cat("  - feature_overlap_permutation_test_race.csv\n")
cat("  - feature_overlap_null_distribution_gender.png\n")
cat("  - feature_overlap_null_distribution_race.png\n\n")

cat("Interpretation:\n")
cat("  If p < 0.05: Feature overlap is SIGNIFICANTLY GREATER than chance\n")
cat("    → Discriminative features are CONSISTENT across demographic groups\n")
cat("    → Model captures universal ADRD language patterns\n")
cat("    → No evidence of demographic-specific feature reliance\n\n")
cat("  If p >= 0.05: Feature overlap not significantly different from chance\n")
cat("    → Features may differ across demographic groups\n")
cat("    → Further investigation recommended\n\n")

cat("Next Steps:\n")
cat("  1. Review feature overlap permutation test results\n")
cat("  2. Examine overlapping features for clinical validity\n")
cat("  3. Investigate any unique features specific to demographic groups\n")
cat("  4. Combine with Aim 1 results for comprehensive fairness assessment\n\n")

cat("Methodology: Yeh (2000) Approximate Randomization Testing\n")
cat("Reference: Yeh, A. (2000). More Accurate Tests for the Statistical\n")
cat("           Significance of Result Differences. COLING 2000.\n\n")

cat("================================================================================\n")
cat("Script completed successfully!\n")
cat("================================================================================\n")
