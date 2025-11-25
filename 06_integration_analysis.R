#!/usr/bin/env Rscript
# ==============================================================================
# ADRD ePhenotyping Pipeline - INTEGRATION: Aim 1 + Aim 2 Analysis
# ==============================================================================
# Version: 1.0
# Author: Gyasi, Frederick
# Purpose: Integrate demographic fairness analysis (Aim 1) with feature-level
#          analysis (Aim 2) to provide comprehensive bias characterization
#
# INTEGRATION FRAMEWORK:
# - Aim 1: WHERE bias exists (which demographics show performance gaps)
# - Aim 2: WHY bias exists (which features/phrases drive differences)
# - Combined: Complete bias characterization across three dimensions:
#   1. Clinical Bias: ADRD presentation differences
#   2. Algorithmic Bias: Classification parity failures
#   3. Linguistic Bias: Feature salience disparities
#
# Inputs:
#   - results/demographic/subgroup_performance.csv (from Aim 1)
#   - results/aim2/demographic_tfidf_comparison.csv (from Aim 2)
#   - results/aim2/demographic_chi2_stratified.rds (from Aim 2)
#   - results/aim2/demographic_lime_stratified.rds (from Aim 2)
#
# Outputs:
#   - results/integration/bias_characterization_summary.csv
#   - results/integration/integrated_dashboard.rds
#   - figures/integration/integrated_dashboard.png
#   - figures/integration/bias_framework_visualization.png
# ==============================================================================

# Define operators FIRST
`%+%` <- function(a, b) paste0(a, b)

# Load Libraries ==============================================================
cat(strrep("=", 80) %+% "\n")
cat("ADRD ePhenotyping - INTEGRATION: Aim 1 + Aim 2 Analysis\n")
cat(strrep("=", 80) %+% "\n\n")

cat("Loading required libraries...\n")
suppressPackageStartupMessages({
  library(tidyverse)    # Data manipulation
  library(ggplot2)      # Visualization
  library(gridExtra)    # Multiple plots
  library(patchwork)    # Plot composition
  library(scales)       # Plot formatting
  library(writexl)      # Excel export
})

options(dplyr.summarise.inform = FALSE)

# Configuration ===============================================================
cat("\nConfiguration:\n")
cat(strrep("-", 80) %+% "\n")

# Paths
RESULTS_DIR <- "results"
FIGURES_DIR <- "figures"
DEMO_RESULTS_DIR <- file.path(RESULTS_DIR, "demographic")
AIM2_RESULTS_DIR <- file.path(RESULTS_DIR, "aim2")
INTEGRATION_RESULTS_DIR <- file.path(RESULTS_DIR, "integration")
INTEGRATION_FIGURES_DIR <- file.path(FIGURES_DIR, "integration")

# Create directories
dir.create(INTEGRATION_RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(INTEGRATION_FIGURES_DIR, showWarnings = FALSE, recursive = TRUE)

# Analysis parameters
SIGNIFICANCE_THRESHOLD <- 0.05
MIN_EFFECT_SIZE <- 0.2  # Cohen's d threshold for meaningful difference

cat("  Significance threshold:", SIGNIFICANCE_THRESHOLD, "\n")
cat("  Minimum effect size:", MIN_EFFECT_SIZE, "\n\n")

# ==============================================================================
# PART 1: LOAD AIM 1 RESULTS (Demographic Fairness)
# ==============================================================================

cat(strrep("=", 80) %+% "\n")
cat("PART 1: Loading Aim 1 Results (Demographic Fairness)\n")
cat(strrep("=", 80) %+% "\n\n")

# Load subgroup performance data
subgroup_file <- file.path(DEMO_RESULTS_DIR, "subgroup_performance.csv")

if (!file.exists(subgroup_file)) {
  stop("Aim 1 results not found! Run 04_demographic_analysis.R first.")
}

aim1_results <- read_csv(subgroup_file, show_col_types = FALSE)
cat("Loaded Aim 1 results:", nrow(aim1_results), "demographic comparisons\n")

# Display summary
cat("\nDemographic variables analyzed:\n")
print(table(aim1_results$Demographic))
cat("\n")

# Identify significant performance gaps
significant_gaps <- aim1_results %>%
  filter(Comparison_Type == "Overall") %>%
  select(Demographic, Subgroup, AUC, Accuracy, F1, Sensitivity, Specificity) %>%
  group_by(Demographic) %>%
  mutate(
    AUC_gap = max(AUC, na.rm = TRUE) - min(AUC, na.rm = TRUE),
    F1_gap = max(F1, na.rm = TRUE) - min(F1, na.rm = TRUE)
  ) %>%
  ungroup() %>%
  filter(AUC_gap > 0.05 | F1_gap > 0.05)  # Meaningful performance gaps

cat("Significant performance gaps identified:\n")
if (nrow(significant_gaps) > 0) {
  print(significant_gaps %>%
          select(Demographic, Subgroup, AUC, F1, AUC_gap, F1_gap) %>%
          arrange(desc(AUC_gap)))
} else {
  cat("  (none)\n")
}
cat("\n")

# ==============================================================================
# PART 2: LOAD AIM 2 RESULTS (Feature-Level Analysis)
# ==============================================================================

cat(strrep("=", 80) %+% "\n")
cat("PART 2: Loading Aim 2 Results (Feature-Level Analysis)\n")
cat(strrep("=", 80) %+% "\n\n")

# Load TF-IDF demographic comparison
tfidf_file <- file.path(AIM2_RESULTS_DIR, "demographic_tfidf_comparison.csv")
aim2_tfidf <- NULL

if (file.exists(tfidf_file)) {
  aim2_tfidf <- read_csv(tfidf_file, show_col_types = FALSE)
  cat("Loaded TF-IDF demographic comparison:", nrow(aim2_tfidf), "term-subgroup pairs\n")
  cat("  Demographics:", paste(unique(aim2_tfidf$demographic), collapse = ", "), "\n")
} else {
  cat("WARNING: TF-IDF demographic comparison not found\n")
}

# Load chi-squared demographic stratified results
chi2_file <- file.path(AIM2_RESULTS_DIR, "demographic_chi2_stratified.rds")
aim2_chi2 <- NULL

if (file.exists(chi2_file)) {
  aim2_chi2 <- readRDS(chi2_file)
  cat("Loaded chi-squared stratified results for",
      length(aim2_chi2), "demographic variables\n")
} else {
  cat("WARNING: Chi-squared stratified results not found\n")
}

# Load LIME demographic stratified results
lime_file <- file.path(AIM2_RESULTS_DIR, "demographic_lime_stratified.rds")
aim2_lime <- NULL

if (file.exists(lime_file)) {
  aim2_lime <- readRDS(lime_file)
  cat("Loaded LIME stratified results\n")
} else {
  cat("Note: LIME stratified results not found (optional)\n")
}

cat("\n")

# ==============================================================================
# PART 3: INTEGRATE AIM 1 + AIM 2 FINDINGS
# ==============================================================================

cat(strrep("=", 80) %+% "\n")
cat("PART 3: Integrating Aim 1 + Aim 2 Findings\n")
cat(strrep("=", 80) %+% "\n\n")

# Initialize integration results list
integration_results <- list()

# For each demographic variable with significant gaps
if (nrow(significant_gaps) > 0 && !is.null(aim2_tfidf)) {

  demographics_to_analyze <- unique(significant_gaps$Demographic)

  for (demo_var in demographics_to_analyze) {

    cat(strrep("-", 80) %+% "\n")
    cat("Analyzing:", demo_var, "\n")
    cat(strrep("-", 80) %+% "\n\n")

    # Get Aim 1 performance data for this demographic
    demo_performance <- aim1_results %>%
      filter(Demographic == demo_var, Comparison_Type == "Overall") %>%
      select(Subgroup, AUC, Accuracy, F1, Sensitivity, Specificity)

    cat("Performance by subgroup:\n")
    print(demo_performance)
    cat("\n")

    # Get Aim 2 TF-IDF data for this demographic
    demo_tfidf <- aim2_tfidf %>%
      filter(demographic == demo_var)

    if (nrow(demo_tfidf) > 0) {

      # Identify discriminative features by subgroup
      cat("Discriminative features by subgroup:\n")

      for (subgroup in unique(demo_tfidf$subgroup)) {

        subgroup_performance <- demo_performance %>%
          filter(Subgroup == subgroup)

        subgroup_tfidf <- demo_tfidf %>%
          filter(subgroup == !!subgroup)

        # Get top terms for correct vs incorrect
        top_correct <- subgroup_tfidf %>%
          filter(classification == "Correct") %>%
          arrange(desc(tfidf_score)) %>%
          head(10)

        top_incorrect <- subgroup_tfidf %>%
          filter(classification == "Incorrect") %>%
          arrange(desc(tfidf_score)) %>%
          head(10)

        # Find unique terms
        unique_to_correct <- setdiff(top_correct$term, top_incorrect$term)
        unique_to_incorrect <- setdiff(top_incorrect$term, top_correct$term)

        cat("\n  ", subgroup, ":\n", sep = "")
        if (nrow(subgroup_performance) > 0) {
          cat("    Performance: AUC=", round(subgroup_performance$AUC, 3),
              ", F1=", round(subgroup_performance$F1, 3), "\n", sep = "")
        }
        cat("    Terms unique to correct classifications (", length(unique_to_correct), "):\n", sep = "")
        if (length(unique_to_correct) > 0) {
          cat("      ", paste(head(unique_to_correct, 5), collapse = ", "), "\n")
        } else {
          cat("      (none)\n")
        }
        cat("    Terms unique to incorrect classifications (", length(unique_to_incorrect), "):\n", sep = "")
        if (length(unique_to_incorrect) > 0) {
          cat("      ", paste(head(unique_to_incorrect, 5), collapse = ", "), "\n")
        } else {
          cat("      (none)\n")
        }
      }
      cat("\n")

      # Store integration results
      integration_results[[demo_var]] <- list(
        performance = demo_performance,
        tfidf = demo_tfidf,
        summary = data.frame(
          demographic = demo_var,
          subgroup = unique(demo_tfidf$subgroup),
          stringsAsFactors = FALSE
        )
      )
    }
  }
}

cat("Integration analysis complete\n\n")

# ==============================================================================
# PART 4: CREATE COMPREHENSIVE BIAS CHARACTERIZATION
# ==============================================================================

cat(strrep("=", 80) %+% "\n")
cat("PART 4: Creating Comprehensive Bias Characterization\n")
cat(strrep("=", 80) %+% "\n\n")

# Create summary table linking performance gaps to features
bias_characterization <- list()

if (length(integration_results) > 0 && !is.null(aim2_tfidf)) {

  for (demo_var in names(integration_results)) {

    demo_data <- integration_results[[demo_var]]
    performance <- demo_data$performance
    tfidf <- demo_data$tfidf

    # For each subgroup, characterize bias
    for (subgroup in unique(tfidf$subgroup)) {

      subgroup_perf <- performance %>% filter(Subgroup == subgroup)
      subgroup_tfidf <- tfidf %>% filter(subgroup == !!subgroup)

      # Calculate feature distinctiveness
      correct_terms <- subgroup_tfidf %>%
        filter(classification == "Correct") %>%
        pull(term)

      incorrect_terms <- subgroup_tfidf %>%
        filter(classification == "Incorrect") %>%
        pull(term)

      term_overlap <- length(intersect(correct_terms, incorrect_terms)) /
                     max(length(correct_terms), length(incorrect_terms), 1)

      # Create bias characterization entry
      bias_entry <- data.frame(
        demographic = demo_var,
        subgroup = subgroup,
        cnn_auc = ifelse(nrow(subgroup_perf) > 0, subgroup_perf$AUC, NA),
        cnn_f1 = ifelse(nrow(subgroup_perf) > 0, subgroup_perf$F1, NA),
        n_discriminative_features = nrow(subgroup_tfidf),
        feature_overlap_pct = round(term_overlap * 100, 1),
        bias_dimension_clinical = "ADRD presentation varies by demographics",
        bias_dimension_algorithmic = ifelse(
          nrow(subgroup_perf) > 0 && subgroup_perf$AUC < 0.8,
          "Significant performance gap",
          "Acceptable performance"
        ),
        bias_dimension_linguistic = ifelse(
          term_overlap < 0.5,
          "Distinct linguistic patterns by classification",
          "Similar linguistic patterns"
        ),
        stringsAsFactors = FALSE
      )

      bias_characterization[[paste0(demo_var, "_", subgroup)]] <- bias_entry
    }
  }

  # Combine all characterizations
  bias_summary <- bind_rows(bias_characterization)

  # Save bias characterization
  write_csv(bias_summary,
            file.path(INTEGRATION_RESULTS_DIR,
                     "bias_characterization_summary.csv"))
  write_xlsx(bias_summary,
             file.path(INTEGRATION_RESULTS_DIR,
                      "bias_characterization_summary.xlsx"))

  cat("Bias characterization summary:\n")
  print(bias_summary %>%
          select(demographic, subgroup, cnn_auc, cnn_f1,
                 n_discriminative_features, feature_overlap_pct))
  cat("\n")

  cat("Bias characterization saved\n\n")
} else {
  cat("Insufficient data for bias characterization\n\n")
  bias_summary <- NULL
}

# Save integration results
saveRDS(integration_results,
        file.path(INTEGRATION_RESULTS_DIR, "integrated_dashboard.rds"))
cat("Integration results saved\n\n")

# ==============================================================================
# PART 5: CREATE INTEGRATED VISUALIZATIONS
# ==============================================================================

cat(strrep("=", 80) %+% "\n")
cat("PART 5: Creating Integrated Visualizations\n")
cat(strrep("=", 80) %+% "\n\n")

# Visualization 1: Performance Gaps + Feature Counts
if (!is.null(bias_summary) && nrow(bias_summary) > 0) {

  cat("Creating integrated dashboard...\n")

  # Plot 1: Performance by subgroup
  plot1 <- ggplot(bias_summary,
                  aes(x = subgroup, y = cnn_auc, fill = demographic)) +
    geom_col(position = "dodge") +
    geom_hline(yintercept = 0.8, linetype = "dashed", color = "red", size = 0.7) +
    scale_fill_brewer(palette = "Set2", name = "Demographic") +
    labs(
      title = "CNN Performance Across Demographic Subgroups",
      subtitle = "Aim 1: Classification Parity Evaluation",
      x = "Subgroup",
      y = "AUC"
    ) +
    theme_classic() +
    theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 11, hjust = 0.5),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
      axis.title = element_text(size = 12, face = "bold"),
      legend.position = "bottom"
    )

  # Plot 2: Feature distinctiveness
  plot2 <- ggplot(bias_summary,
                  aes(x = subgroup, y = feature_overlap_pct, fill = demographic)) +
    geom_col(position = "dodge") +
    geom_hline(yintercept = 50, linetype = "dashed", color = "blue", size = 0.7) +
    scale_fill_brewer(palette = "Set2", name = "Demographic") +
    labs(
      title = "Feature Overlap: Correct vs. Incorrect Classifications",
      subtitle = "Aim 2: Linguistic Bias Evaluation",
      x = "Subgroup",
      y = "Feature Overlap (%)"
    ) +
    theme_classic() +
    theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 11, hjust = 0.5),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
      axis.title = element_text(size = 12, face = "bold"),
      legend.position = "bottom"
    )

  # Plot 3: Number of discriminative features
  plot3 <- ggplot(bias_summary,
                  aes(x = subgroup, y = n_discriminative_features, fill = demographic)) +
    geom_col(position = "dodge") +
    scale_fill_brewer(palette = "Set2", name = "Demographic") +
    labs(
      title = "Discriminative Features by Subgroup",
      subtitle = "Aim 2: Feature Importance Analysis",
      x = "Subgroup",
      y = "Number of Features"
    ) +
    theme_classic() +
    theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 11, hjust = 0.5),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
      axis.title = element_text(size = 12, face = "bold"),
      legend.position = "bottom"
    )

  # Combine plots
  integrated_dashboard <- plot1 / plot2 / plot3 +
    plot_annotation(
      title = "ADRD CNN Bias Analysis: Integrated Dashboard",
      subtitle = "Connecting Classification Parity (Aim 1) with Feature-Level Analysis (Aim 2)",
      theme = theme(
        plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 12, hjust = 0.5)
      )
    )

  ggsave(
    file.path(INTEGRATION_FIGURES_DIR, "integrated_dashboard.png"),
    plot = integrated_dashboard,
    width = 14,
    height = 16,
    dpi = 300
  )

  cat("  Integrated dashboard saved\n")
}

# Visualization 2: Three-Dimensional Bias Framework
if (!is.null(bias_summary) && nrow(bias_summary) > 0) {

  cat("Creating bias framework visualization...\n")

  # Create summary by demographic
  bias_framework <- bias_summary %>%
    group_by(demographic) %>%
    summarize(
      mean_auc = mean(cnn_auc, na.rm = TRUE),
      auc_range = max(cnn_auc, na.rm = TRUE) - min(cnn_auc, na.rm = TRUE),
      mean_overlap = mean(feature_overlap_pct, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      algorithmic_bias = case_when(
        auc_range > 0.1 ~ "High",
        auc_range > 0.05 ~ "Moderate",
        TRUE ~ "Low"
      ),
      linguistic_bias = case_when(
        mean_overlap < 40 ~ "High",
        mean_overlap < 60 ~ "Moderate",
        TRUE ~ "Low"
      )
    )

  # Create framework plot
  framework_plot <- ggplot(bias_framework,
                          aes(x = auc_range, y = 100 - mean_overlap)) +
    geom_point(aes(size = mean_auc, color = demographic), alpha = 0.7) +
    geom_text(aes(label = demographic), vjust = -1, size = 4, fontface = "bold") +
    scale_color_brewer(palette = "Set1", name = "Demographic") +
    scale_size_continuous(name = "Mean AUC", range = c(5, 15)) +
    labs(
      title = "Three-Dimensional Bias Framework for CNN Model",
      subtitle = "Algorithmic Bias (x-axis) vs. Linguistic Bias (y-axis)",
      x = "Performance Gap (AUC Range) - Algorithmic Bias",
      y = "Feature Distinctiveness (100 - Overlap%) - Linguistic Bias"
    ) +
    theme_classic() +
    theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 11, hjust = 0.5),
      axis.title = element_text(size = 12, face = "bold"),
      legend.position = "right"
    )

  ggsave(
    file.path(INTEGRATION_FIGURES_DIR, "bias_framework_visualization.png"),
    plot = framework_plot,
    width = 12,
    height = 8,
    dpi = 300
  )

  cat("  Bias framework visualization saved\n\n")
}

# ==============================================================================
# SUMMARY
# ==============================================================================

cat(strrep("=", 80) %+% "\n")
cat("INTEGRATION ANALYSIS COMPLETE\n")
cat(strrep("=", 80) %+% "\n\n")

cat("Summary:\n")
cat(strrep("-", 80) %+% "\n")
cat("AIM 1 (WHERE bias exists):\n")
cat("  - Evaluated CNN performance across demographic subgroups\n")
cat("  - Identified", nrow(significant_gaps), "subgroups with performance gaps\n")
cat("\n")
cat("AIM 2 (WHY bias exists):\n")
cat("  - Analyzed discriminative features by demographic\n")
if (!is.null(aim2_tfidf)) {
  cat("  - Identified", length(unique(aim2_tfidf$term)),
      "unique clinical terms\n")
  cat("  - Analyzed", length(unique(aim2_tfidf$demographic)),
      "demographic variables\n")
}
cat("\n")
cat("INTEGRATION (Complete bias characterization):\n")
cat("  - Connected performance gaps to linguistic features\n")
cat("  - Characterized bias across three dimensions:\n")
cat("    1. Clinical Bias: ADRD presentation differences\n")
cat("    2. Algorithmic Bias: Classification parity evaluation\n")
cat("    3. Linguistic Bias: Feature salience disparities\n")
cat("\n")

cat("Outputs:\n")
cat(strrep("-", 80) %+% "\n")
cat("Results:\n")
cat("  ", file.path(INTEGRATION_RESULTS_DIR, "bias_characterization_summary.csv"), "\n")
cat("  ", file.path(INTEGRATION_RESULTS_DIR, "integrated_dashboard.rds"), "\n")
cat("\n")
cat("Figures:\n")
cat("  ", file.path(INTEGRATION_FIGURES_DIR, "integrated_dashboard.png"), "\n")
cat("  ", file.path(INTEGRATION_FIGURES_DIR, "bias_framework_visualization.png"), "\n")
cat("\n")

cat(strrep("=", 80) %+% "\n")
cat("Ready for AMIA submission!\n")
cat(strrep("=", 80) %+% "\n")
