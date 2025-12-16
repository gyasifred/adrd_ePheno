# ==============================================================================
# Statistical Testing Utilities
# ==============================================================================
# Author: Gyasi, Frederick
# Purpose: Implement statistical significance testing for demographic comparisons
#
# Methods:
# - Approximate randomization (permutation) testing
# - Bootstrap confidence intervals
# - Multiple testing correction (FDR)
# - Effect size calculation
# ==============================================================================

library(pROC)

# ==============================================================================
# PERMUTATION TEST FOR AUC DIFFERENCE
# ==============================================================================

permutation_test_auc <- function(labels_a, pred_a, labels_b, pred_b,
                                  n_perm = 10000, seed = 42) {
  #' Perform permutation test for AUC difference between two groups
  #'
  #' @param labels_a True labels for group A
  #' @param pred_a Predicted probabilities for group A
  #' @param labels_b True labels for group B
  #' @param pred_b Predicted probabilities for group B
  #' @param n_perm Number of permutations (default 10000)
  #' @param seed Random seed for reproducibility
  #'
  #' @return List with observed difference, p-value, and permutation distribution

  set.seed(seed)

  # Calculate observed AUCs
  auc_a_obs <- tryCatch({
    auc(roc(labels_a, pred_a, quiet = TRUE))
  }, error = function(e) NA)

  auc_b_obs <- tryCatch({
    auc(roc(labels_b, pred_b, quiet = TRUE))
  }, error = function(e) NA)

  if (is.na(auc_a_obs) || is.na(auc_b_obs)) {
    return(list(
      observed_diff = NA,
      auc_a = auc_a_obs,
      auc_b = auc_b_obs,
      p_value = NA,
      perm_diffs = NA,
      error = "Failed to calculate AUC"
    ))
  }

  observed_diff <- auc_a_obs - auc_b_obs

  # Pool data
  labels_pooled <- c(labels_a, labels_b)
  pred_pooled <- c(pred_a, pred_b)
  n_a <- length(labels_a)
  n_b <- length(labels_b)
  n_total <- n_a + n_b

  # Permutation distribution
  perm_diffs <- numeric(n_perm)

  for (i in seq_len(n_perm)) {
    # Shuffle indices
    shuffled_idx <- sample(n_total)

    # Split into groups maintaining original sizes
    idx_a <- shuffled_idx[1:n_a]
    idx_b <- shuffled_idx[(n_a + 1):n_total]

    # Calculate AUCs
    auc_a_perm <- tryCatch({
      auc(roc(labels_pooled[idx_a], pred_pooled[idx_a], quiet = TRUE))
    }, error = function(e) NA)

    auc_b_perm <- tryCatch({
      auc(roc(labels_pooled[idx_b], pred_pooled[idx_b], quiet = TRUE))
    }, error = function(e) NA)

    if (!is.na(auc_a_perm) && !is.na(auc_b_perm)) {
      perm_diffs[i] <- auc_a_perm - auc_b_perm
    } else {
      perm_diffs[i] <- NA
    }
  }

  # Remove NAs
  perm_diffs <- perm_diffs[!is.na(perm_diffs)]

  if (length(perm_diffs) < n_perm * 0.9) {
    warning("More than 10% of permutations failed. Results may be unreliable.")
  }

  # Calculate p-value (two-sided)
  p_value <- mean(abs(perm_diffs) >= abs(observed_diff))

  return(list(
    observed_diff = observed_diff,
    auc_a = auc_a_obs,
    auc_b = auc_b_obs,
    p_value = p_value,
    perm_diffs = perm_diffs,
    n_valid_perms = length(perm_diffs)
  ))
}

# ==============================================================================
# PERMUTATION TEST FOR GENERIC METRIC
# ==============================================================================

permutation_test_metric <- function(metric_a, metric_b,
                                     labels_a, pred_a,
                                     labels_b, pred_b,
                                     metric_function,
                                     n_perm = 10000, seed = 42) {
  #' Perform permutation test for any metric difference
  #'
  #' @param metric_a Observed metric for group A
  #' @param metric_b Observed metric for group B
  #' @param labels_a, pred_a, labels_b, pred_b Data for recalculation
  #' @param metric_function Function to calculate metric(labels, predictions)
  #' @param n_perm Number of permutations
  #' @param seed Random seed
  #'
  #' @return List with p-value and permutation distribution

  set.seed(seed)

  observed_diff <- metric_a - metric_b

  # Pool data
  labels_pooled <- c(labels_a, labels_b)
  pred_pooled <- c(pred_a, pred_b)
  n_a <- length(labels_a)
  n_b <- length(labels_b)
  n_total <- n_a + n_b

  # Permutation distribution
  perm_diffs <- numeric(n_perm)

  for (i in seq_len(n_perm)) {
    shuffled_idx <- sample(n_total)
    idx_a <- shuffled_idx[1:n_a]
    idx_b <- shuffled_idx[(n_a + 1):n_total]

    metric_a_perm <- metric_function(labels_pooled[idx_a], pred_pooled[idx_a])
    metric_b_perm <- metric_function(labels_pooled[idx_b], pred_pooled[idx_b])

    perm_diffs[i] <- metric_a_perm - metric_b_perm
  }

  p_value <- mean(abs(perm_diffs) >= abs(observed_diff))

  return(list(
    observed_diff = observed_diff,
    metric_a = metric_a,
    metric_b = metric_b,
    p_value = p_value,
    perm_diffs = perm_diffs
  ))
}

# ==============================================================================
# METRIC CALCULATION FUNCTIONS (for permutation testing)
# ==============================================================================

calculate_accuracy <- function(labels, predictions, threshold = 0.5) {
  #' Calculate accuracy from labels and predicted probabilities
  pred_class <- ifelse(predictions >= threshold, 1, 0)
  return(mean(pred_class == labels))
}

calculate_sensitivity <- function(labels, predictions, threshold = 0.5) {
  #' Calculate sensitivity (recall, TPR) from labels and predicted probabilities
  pred_class <- ifelse(predictions >= threshold, 1, 0)
  tp <- sum(labels == 1 & pred_class == 1)
  fn <- sum(labels == 1 & pred_class == 0)
  return(ifelse((tp + fn) > 0, tp / (tp + fn), NA))
}

calculate_specificity <- function(labels, predictions, threshold = 0.5) {
  #' Calculate specificity (TNR) from labels and predicted probabilities
  pred_class <- ifelse(predictions >= threshold, 1, 0)
  tn <- sum(labels == 0 & pred_class == 0)
  fp <- sum(labels == 0 & pred_class == 1)
  return(ifelse((tn + fp) > 0, tn / (tn + fp), NA))
}

calculate_precision <- function(labels, predictions, threshold = 0.5) {
  #' Calculate precision (PPV) from labels and predicted probabilities
  pred_class <- ifelse(predictions >= threshold, 1, 0)
  tp <- sum(labels == 1 & pred_class == 1)
  fp <- sum(labels == 0 & pred_class == 1)
  return(ifelse((tp + fp) > 0, tp / (tp + fp), NA))
}

calculate_npv <- function(labels, predictions, threshold = 0.5) {
  #' Calculate Negative Predictive Value (NPV)
  pred_class <- ifelse(predictions >= threshold, 1, 0)
  tn <- sum(labels == 0 & pred_class == 0)
  fn <- sum(labels == 1 & pred_class == 0)
  return(ifelse((tn + fn) > 0, tn / (tn + fn), NA))
}

calculate_f1 <- function(labels, predictions, threshold = 0.5) {
  #' Calculate F1 Score from labels and predicted probabilities
  precision <- calculate_precision(labels, predictions, threshold)
  sensitivity <- calculate_sensitivity(labels, predictions, threshold)
  if (is.na(precision) || is.na(sensitivity) || (precision + sensitivity) == 0) {
    return(NA)
  }
  return(2 * (precision * sensitivity) / (precision + sensitivity))
}

calculate_f2 <- function(labels, predictions, threshold = 0.5) {
  #' Calculate F2 Score (weighted toward recall) from labels and predicted probabilities
  precision <- calculate_precision(labels, predictions, threshold)
  sensitivity <- calculate_sensitivity(labels, predictions, threshold)
  if (is.na(precision) || is.na(sensitivity) || (4 * precision + sensitivity) == 0) {
    return(NA)
  }
  return(5 * (precision * sensitivity) / (4 * precision + sensitivity))
}

# ==============================================================================
# BOOTSTRAP CONFIDENCE INTERVAL FOR AUC
# ==============================================================================

bootstrap_auc_ci <- function(labels, predictions,
                              n_boot = 10000,
                              conf_level = 0.95,
                              seed = 42) {
  #' Bootstrap confidence interval for AUC
  #'
  #' @param labels True labels
  #' @param predictions Predicted probabilities
  #' @param n_boot Number of bootstrap samples
  #' @param conf_level Confidence level (default 0.95)
  #' @param seed Random seed
  #'
  #' @return List with CI and bootstrap distribution

  set.seed(seed)

  # Observed AUC
  auc_obs <- tryCatch({
    auc(roc(labels, predictions, quiet = TRUE))
  }, error = function(e) NA)

  if (is.na(auc_obs)) {
    return(list(
      observed = NA,
      ci_lower = NA,
      ci_upper = NA,
      error = "Failed to calculate AUC"
    ))
  }

  # Stratified bootstrap
  pos_idx <- which(labels == 1)
  neg_idx <- which(labels == 0)
  n_pos <- length(pos_idx)
  n_neg <- length(neg_idx)

  boot_aucs <- numeric(n_boot)

  for (i in seq_len(n_boot)) {
    # Sample with replacement from each class
    boot_pos <- sample(pos_idx, n_pos, replace = TRUE)
    boot_neg <- sample(neg_idx, n_neg, replace = TRUE)
    boot_idx <- c(boot_pos, boot_neg)

    boot_auc <- tryCatch({
      auc(roc(labels[boot_idx], predictions[boot_idx], quiet = TRUE))
    }, error = function(e) NA)

    boot_aucs[i] <- boot_auc
  }

  # Remove NAs
  boot_aucs <- boot_aucs[!is.na(boot_aucs)]

  # Calculate CI (percentile method)
  alpha <- 1 - conf_level
  ci_lower <- quantile(boot_aucs, alpha / 2)
  ci_upper <- quantile(boot_aucs, 1 - alpha / 2)

  return(list(
    observed = auc_obs,
    ci_lower = ci_lower,
    ci_upper = ci_upper,
    boot_dist = boot_aucs,
    n_valid = length(boot_aucs)
  ))
}

# ==============================================================================
# EFFECT SIZE (COHEN'S D)
# ==============================================================================

cohens_d <- function(x, y) {
  #' Calculate Cohen's d effect size
  #'
  #' @param x Vector of values for group 1
  #' @param y Vector of values for group 2
  #'
  #' @return Cohen's d value

  n1 <- length(x)
  n2 <- length(y)

  mean1 <- mean(x, na.rm = TRUE)
  mean2 <- mean(y, na.rm = TRUE)

  sd1 <- sd(x, na.rm = TRUE)
  sd2 <- sd(y, na.rm = TRUE)

  # Pooled standard deviation
  pooled_sd <- sqrt(((n1 - 1) * sd1^2 + (n2 - 1) * sd2^2) / (n1 + n2 - 2))

  d <- (mean1 - mean2) / pooled_sd

  return(d)
}

# ==============================================================================
# COMPREHENSIVE METRIC PERMUTATION TESTING
# ==============================================================================

compare_all_metrics_comprehensive <- function(group_a_data, group_b_data,
                                               group_a_name = "Group A",
                                               group_b_name = "Group B",
                                               n_perm = 10000,
                                               n_boot = 10000,
                                               threshold = 0.5,
                                               seed = 42) {
  #' Comprehensive permutation testing for ALL classification metrics
  #'
  #' Implements approximate randomization testing for:
  #'   - AUC (Area Under ROC Curve)
  #'   - Accuracy
  #'   - Sensitivity (Recall, TPR)
  #'   - Specificity (TNR)
  #'   - Precision (PPV)
  #'   - NPV (Negative Predictive Value)
  #'   - F1 Score
  #'   - F2 Score
  #'
  #' @param group_a_data, group_b_data Data frames with columns: label, pred
  #' @param group_a_name, group_b_name Names for reporting
  #' @param n_perm Number of permutations
  #' @param n_boot Number of bootstrap samples (for AUC CI only)
  #' @param threshold Classification threshold (default 0.5)
  #' @param seed Random seed
  #'
  #' @return List with permutation test results for all metrics

  cat("========================================\n")
  cat("Comprehensive Metric Comparison\n")
  cat(group_a_name, "vs", group_b_name, "\n")
  cat("========================================\n\n")

  # Extract data
  labels_a <- group_a_data$label
  pred_a <- group_a_data$pred
  labels_b <- group_b_data$label
  pred_b <- group_b_data$pred

  # Calculate observed metrics for both groups
  cat("Calculating observed metrics...\n")

  metrics_a <- list(
    auc = tryCatch(auc(roc(labels_a, pred_a, quiet = TRUE)), error = function(e) NA),
    accuracy = calculate_accuracy(labels_a, pred_a, threshold),
    sensitivity = calculate_sensitivity(labels_a, pred_a, threshold),
    specificity = calculate_specificity(labels_a, pred_a, threshold),
    precision = calculate_precision(labels_a, pred_a, threshold),
    npv = calculate_npv(labels_a, pred_a, threshold),
    f1 = calculate_f1(labels_a, pred_a, threshold),
    f2 = calculate_f2(labels_a, pred_a, threshold)
  )

  metrics_b <- list(
    auc = tryCatch(auc(roc(labels_b, pred_b, quiet = TRUE)), error = function(e) NA),
    accuracy = calculate_accuracy(labels_b, pred_b, threshold),
    sensitivity = calculate_sensitivity(labels_b, pred_b, threshold),
    specificity = calculate_specificity(labels_b, pred_b, threshold),
    precision = calculate_precision(labels_b, pred_b, threshold),
    npv = calculate_npv(labels_b, pred_b, threshold),
    f1 = calculate_f1(labels_b, pred_b, threshold),
    f2 = calculate_f2(labels_b, pred_b, threshold)
  )

  # Run permutation tests for each metric
  cat("\nRunning permutation tests (", n_perm, " permutations each)...\n", sep = "")

  perm_results <- list()

  # AUC (special handling with pROC)
  cat("  [1/8] AUC permutation test...\n")
  perm_results$auc <- permutation_test_auc(labels_a, pred_a, labels_b, pred_b,
                                            n_perm = n_perm, seed = seed)

  # Accuracy
  cat("  [2/8] Accuracy permutation test...\n")
  perm_results$accuracy <- permutation_test_metric(
    metrics_a$accuracy, metrics_b$accuracy,
    labels_a, pred_a, labels_b, pred_b,
    metric_function = function(lab, pred) calculate_accuracy(lab, pred, threshold),
    n_perm = n_perm, seed = seed + 1
  )

  # Sensitivity
  cat("  [3/8] Sensitivity permutation test...\n")
  perm_results$sensitivity <- permutation_test_metric(
    metrics_a$sensitivity, metrics_b$sensitivity,
    labels_a, pred_a, labels_b, pred_b,
    metric_function = function(lab, pred) calculate_sensitivity(lab, pred, threshold),
    n_perm = n_perm, seed = seed + 2
  )

  # Specificity
  cat("  [4/8] Specificity permutation test...\n")
  perm_results$specificity <- permutation_test_metric(
    metrics_a$specificity, metrics_b$specificity,
    labels_a, pred_a, labels_b, pred_b,
    metric_function = function(lab, pred) calculate_specificity(lab, pred, threshold),
    n_perm = n_perm, seed = seed + 3
  )

  # Precision
  cat("  [5/8] Precision permutation test...\n")
  perm_results$precision <- permutation_test_metric(
    metrics_a$precision, metrics_b$precision,
    labels_a, pred_a, labels_b, pred_b,
    metric_function = function(lab, pred) calculate_precision(lab, pred, threshold),
    n_perm = n_perm, seed = seed + 4
  )

  # NPV
  cat("  [6/8] NPV permutation test...\n")
  perm_results$npv <- permutation_test_metric(
    metrics_a$npv, metrics_b$npv,
    labels_a, pred_a, labels_b, pred_b,
    metric_function = function(lab, pred) calculate_npv(lab, pred, threshold),
    n_perm = n_perm, seed = seed + 5
  )

  # F1 Score
  cat("  [7/8] F1 Score permutation test...\n")
  perm_results$f1 <- permutation_test_metric(
    metrics_a$f1, metrics_b$f1,
    labels_a, pred_a, labels_b, pred_b,
    metric_function = function(lab, pred) calculate_f1(lab, pred, threshold),
    n_perm = n_perm, seed = seed + 6
  )

  # F2 Score
  cat("  [8/8] F2 Score permutation test...\n")
  perm_results$f2 <- permutation_test_metric(
    metrics_a$f2, metrics_b$f2,
    labels_a, pred_a, labels_b, pred_b,
    metric_function = function(lab, pred) calculate_f2(lab, pred, threshold),
    n_perm = n_perm, seed = seed + 7
  )

  # Bootstrap CIs for AUC
  cat("\nComputing bootstrap confidence intervals for AUC...\n")
  boot_a <- bootstrap_auc_ci(labels_a, pred_a, n_boot = n_boot, seed = seed + 100)
  boot_b <- bootstrap_auc_ci(labels_b, pred_b, n_boot = n_boot, seed = seed + 101)

  # Effect size (Cohen's d) for all metrics
  effect_sizes <- list(
    auc = cohens_d(rep(metrics_a$auc, length(labels_a)), rep(metrics_b$auc, length(labels_b))),
    accuracy = cohens_d(rep(metrics_a$accuracy, length(labels_a)), rep(metrics_b$accuracy, length(labels_b))),
    sensitivity = cohens_d(rep(metrics_a$sensitivity, length(labels_a)), rep(metrics_b$sensitivity, length(labels_b))),
    specificity = cohens_d(rep(metrics_a$specificity, length(labels_a)), rep(metrics_b$specificity, length(labels_b))),
    precision = cohens_d(rep(metrics_a$precision, length(labels_a)), rep(metrics_b$precision, length(labels_b))),
    npv = cohens_d(rep(metrics_a$npv, length(labels_a)), rep(metrics_b$npv, length(labels_b))),
    f1 = cohens_d(rep(metrics_a$f1, length(labels_a)), rep(metrics_b$f1, length(labels_b))),
    f2 = cohens_d(rep(metrics_a$f2, length(labels_a)), rep(metrics_b$f2, length(labels_b)))
  )

  # Compile summary table
  cat("\n========================================\n")
  cat("Permutation Test Results Summary\n")
  cat("========================================\n\n")

  summary_df <- data.frame(
    Metric = c("AUC", "Accuracy", "Sensitivity", "Specificity", "Precision", "NPV", "F1 Score", "F2 Score"),
    Group_A = c(metrics_a$auc, metrics_a$accuracy, metrics_a$sensitivity, metrics_a$specificity,
                metrics_a$precision, metrics_a$npv, metrics_a$f1, metrics_a$f2),
    Group_B = c(metrics_b$auc, metrics_b$accuracy, metrics_b$sensitivity, metrics_b$specificity,
                metrics_b$precision, metrics_b$npv, metrics_b$f1, metrics_b$f2),
    Difference = c(
      perm_results$auc$observed_diff,
      perm_results$accuracy$observed_diff,
      perm_results$sensitivity$observed_diff,
      perm_results$specificity$observed_diff,
      perm_results$precision$observed_diff,
      perm_results$npv$observed_diff,
      perm_results$f1$observed_diff,
      perm_results$f2$observed_diff
    ),
    P_Value = c(
      perm_results$auc$p_value,
      perm_results$accuracy$p_value,
      perm_results$sensitivity$p_value,
      perm_results$specificity$p_value,
      perm_results$precision$p_value,
      perm_results$npv$p_value,
      perm_results$f1$p_value,
      perm_results$f2$p_value
    ),
    Cohens_D = c(
      effect_sizes$auc, effect_sizes$accuracy, effect_sizes$sensitivity, effect_sizes$specificity,
      effect_sizes$precision, effect_sizes$npv, effect_sizes$f1, effect_sizes$f2
    ),
    Significant = c(
      perm_results$auc$p_value < 0.05,
      perm_results$accuracy$p_value < 0.05,
      perm_results$sensitivity$p_value < 0.05,
      perm_results$specificity$p_value < 0.05,
      perm_results$precision$p_value < 0.05,
      perm_results$npv$p_value < 0.05,
      perm_results$f1$p_value < 0.05,
      perm_results$f2$p_value < 0.05
    )
  )

  print(summary_df, row.names = FALSE)
  cat("\n")

  # Count significant differences
  n_significant <- sum(summary_df$Significant, na.rm = TRUE)
  if (n_significant > 0) {
    cat("⚠️  ", n_significant, " metric(s) show statistically significant differences (p<0.05)\n", sep = "")
  } else {
    cat("✓ No statistically significant differences detected (all p>0.05)\n")
  }
  cat("\n")

  # Return comprehensive results
  results <- list(
    # Group information
    group_a = group_a_name,
    group_b = group_b_name,
    n_a = nrow(group_a_data),
    n_b = nrow(group_b_data),
    threshold = threshold,

    # Observed metrics
    metrics_a = metrics_a,
    metrics_b = metrics_b,

    # Permutation test results for all metrics
    permutation_results = perm_results,

    # Bootstrap CIs (AUC only)
    bootstrap_a = boot_a,
    bootstrap_b = boot_b,

    # Effect sizes
    effect_sizes = effect_sizes,

    # Summary table
    summary_table = summary_df
  )

  return(results)
}

# ==============================================================================
# COMPREHENSIVE GROUP COMPARISON (LEGACY - now replaced by compare_all_metrics_comprehensive)
# ==============================================================================

compare_groups_comprehensive <- function(group_a_data, group_b_data,
                                          group_a_name = "Group A",
                                          group_b_name = "Group B",
                                          n_perm = 10000,
                                          n_boot = 10000,
                                          seed = 42) {
  #' Comprehensive statistical comparison between two groups
  #'
  #' NOTE: This function is retained for backward compatibility.
  #'       Use compare_all_metrics_comprehensive() for complete metric testing.
  #'
  #' @param group_a_data, group_b_data Data frames with columns: label, pred
  #' @param group_a_name, group_b_name Names for reporting
  #' @param n_perm Number of permutations
  #' @param n_boot Number of bootstrap samples
  #' @param seed Random seed
  #'
  #' @return List with all test results

  cat("Comparing", group_a_name, "vs", group_b_name, "...\n")

  # Extract data
  labels_a <- group_a_data$label
  pred_a <- group_a_data$pred
  labels_b <- group_b_data$label
  pred_b <- group_b_data$pred

  # Calculate metrics
  auc_a <- tryCatch({
    auc(roc(labels_a, pred_a, quiet = TRUE))
  }, error = function(e) NA)

  auc_b <- tryCatch({
    auc(roc(labels_b, pred_b, quiet = TRUE))
  }, error = function(e) NA)

  # Sensitivity and specificity
  sens_a <- sum(labels_a == 1 & pred_a >= 0.5) / sum(labels_a == 1)
  sens_b <- sum(labels_b == 1 & pred_b >= 0.5) / sum(labels_b == 1)

  spec_a <- sum(labels_a == 0 & pred_a < 0.5) / sum(labels_a == 0)
  spec_b <- sum(labels_b == 0 & pred_b < 0.5) / sum(labels_b == 0)

  # Permutation test for AUC
  cat("  Running permutation test (", n_perm, " permutations)...\n", sep = "")
  perm_result <- permutation_test_auc(labels_a, pred_a, labels_b, pred_b,
                                       n_perm = n_perm, seed = seed)

  # Bootstrap CIs
  cat("  Computing bootstrap CIs...\n")
  boot_a <- bootstrap_auc_ci(labels_a, pred_a, n_boot = n_boot, seed = seed)
  boot_b <- bootstrap_auc_ci(labels_b, pred_b, n_boot = n_boot, seed = seed + 1)

  # Effect size
  effect_size <- cohens_d(
    rep(auc_a, length(labels_a)),  # Approximate
    rep(auc_b, length(labels_b))
  )

  # Compile results
  results <- list(
    group_a = group_a_name,
    group_b = group_b_name,
    n_a = nrow(group_a_data),
    n_b = nrow(group_b_data),

    # AUC
    auc_a = auc_a,
    auc_b = auc_b,
    auc_diff = auc_a - auc_b,
    auc_a_ci = c(boot_a$ci_lower, boot_a$ci_upper),
    auc_b_ci = c(boot_b$ci_lower, boot_b$ci_upper),

    # Sensitivity
    sens_a = sens_a,
    sens_b = sens_b,
    sens_diff = sens_a - sens_b,

    # Specificity
    spec_a = spec_a,
    spec_b = spec_b,
    spec_diff = spec_a - spec_b,

    # Statistical tests
    perm_p_value = perm_result$p_value,
    n_valid_perms = perm_result$n_valid_perms,

    # Effect size
    cohens_d = effect_size,

    # Full results
    permutation_result = perm_result,
    bootstrap_a = boot_a,
    bootstrap_b = boot_b
  )

  cat("  AUC difference:", sprintf("%.4f", results$auc_diff),
      "p =", sprintf("%.4f", results$perm_p_value), "\n")

  return(results)
}

# ==============================================================================
# MULTIPLE TESTING CORRECTION
# ==============================================================================

apply_fdr_correction <- function(p_values, method = "BH", alpha = 0.05) {
  #' Apply FDR correction to p-values
  #'
  #' @param p_values Vector of p-values
  #' @param method Correction method (default "BH" for Benjamini-Hochberg)
  #' @param alpha Significance level
  #'
  #' @return Data frame with original and adjusted p-values

  p_adjusted <- p.adjust(p_values, method = method)

  results <- data.frame(
    original_p = p_values,
    adjusted_p = p_adjusted,
    significant_raw = p_values < alpha,
    significant_adj = p_adjusted < alpha
  )

  return(results)
}

cat("Statistical testing utilities loaded successfully!\n")
cat(strrep("=", 80), "\n")
cat("CORE FUNCTIONS:\n")
cat("  permutation_test_auc() - Permutation test for AUC difference\n")
cat("  permutation_test_metric() - Generic permutation test for any metric\n")
cat("  bootstrap_auc_ci() - Bootstrap confidence intervals for AUC\n")
cat("  cohens_d() - Cohen's d effect size calculation\n")
cat("  apply_fdr_correction() - Multiple testing correction (FDR)\n\n")
cat("METRIC CALCULATION FUNCTIONS:\n")
cat("  calculate_accuracy() - Accuracy calculation\n")
cat("  calculate_sensitivity() - Sensitivity (TPR, Recall)\n")
cat("  calculate_specificity() - Specificity (TNR)\n")
cat("  calculate_precision() - Precision (PPV)\n")
cat("  calculate_npv() - Negative Predictive Value\n")
cat("  calculate_f1() - F1 Score\n")
cat("  calculate_f2() - F2 Score (weighted toward recall)\n\n")
cat("COMPREHENSIVE TESTING FUNCTIONS:\n")
cat("  ⭐ compare_all_metrics_comprehensive() - ALL metrics permutation testing (NEW!)\n")
cat("     Tests: AUC, Accuracy, Sensitivity, Specificity, Precision, NPV, F1, F2\n")
cat("  compare_groups_comprehensive() - Legacy AUC-only testing (backward compat)\n")
cat(strrep("=", 80), "\n")
