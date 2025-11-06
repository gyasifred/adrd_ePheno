# ==============================================================================
# Statistical Testing Utilities
# ==============================================================================
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
# COMPREHENSIVE GROUP COMPARISON
# ==============================================================================

compare_groups_comprehensive <- function(group_a_data, group_b_data,
                                          group_a_name = "Group A",
                                          group_b_name = "Group B",
                                          n_perm = 10000,
                                          n_boot = 10000,
                                          seed = 42) {
  #' Comprehensive statistical comparison between two groups
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

cat("Statistical testing utilities loaded\n")
cat("  permutation_test_auc() - Permutation test for AUC difference\n")
cat("  bootstrap_auc_ci() - Bootstrap confidence intervals\n")
cat("  cohens_d() - Cohen's d effect size\n")
cat("  compare_groups_comprehensive() - Full comparison suite\n")
cat("  apply_fdr_correction() - Multiple testing correction\n")
