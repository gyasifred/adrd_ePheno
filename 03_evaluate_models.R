#!/usr/bin/env Rscript
# ==============================================================================
# ADRD Classification Pipeline - Model Evaluation
# ==============================================================================
# Purpose: Comprehensive evaluation of trained CNN models
# Code follows methodology from Jihad Obeid's original implementation
#
# This script performs:
# 1. Loads all trained models and artifacts
# 2. Generates predictions on test set
# 3. Calculates comprehensive metrics (following Jihad's approach)
# 4. Selects best model using median AUC with max F1
# 5. Creates visualizations (ROC, calibration, distributions)
#
# Inputs:  models/*, data/test_set.rds, results/*
# Outputs: results/evaluation_*, figures/*
# ==============================================================================

# Define operators FIRST (before any usage)
`%+%` <- function(a, b) paste0(a, b)

# Load Libraries ==============================================================
cat(strrep("=", 80) %+% "\n")
cat("ADRD CNN Evaluation Pipeline\n")
cat(strrep("=", 80) %+% "\n\n")

cat("Loading required libraries...\n")
suppressPackageStartupMessages({
  library(reticulate)   # Python/TensorFlow
  library(keras)        # Neural networks (Keras 2)
  library(tensorflow)   # TensorFlow backend
  library(tidyverse)    # Data manipulation
  library(pROC)         # ROC analysis
  library(ggplot2)      # Visualization
  library(gridExtra)    # Multiple plots
  library(scales)       # Plot formatting
  library(writexl)      # Excel export
  library(magrittr)     # Pipe operators
  library(dplyr)        # Data manipulation
})

# Configure Environment =======================================================
cat("\nConfiguring environment...\n")

# Use conda environment with fallback
tryCatch({
  use_condaenv("adrd-pipeline", required = FALSE)
  cat("  Using conda environment: adrd-pipeline\n")
}, error = function(e) {
  cat("  WARNING: conda env 'adrd-pipeline' not found. Using default Python.\n")
})

# TensorFlow configuration
tf$keras$backend$set_floatx('float32')

# GPU configuration
physical_devices <- tf$config$list_physical_devices('GPU')
if (length(physical_devices) > 0) {
  cat("  GPU detected:", length(physical_devices), "device(s)\n")
  for (gpu in physical_devices) {
    tf$config$experimental$set_memory_growth(gpu, TRUE)
  }
} else {
  cat("  No GPU - using CPU\n")
}

options(dplyr.summarise.inform = FALSE)

# Configuration ===============================================================
cat("\nConfiguration:\n")
cat(strrep("-", 80) %+% "\n")

# Paths
DATA_DIR <- "data"
MODEL_DIR <- "models"
RESULTS_DIR <- "results"
FIGURES_DIR <- "figures"

# Create directories if needed
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIGURES_DIR, showWarnings = FALSE, recursive = TRUE)

# Evaluation parameters (following Jihad's approach)
OPTIMAL_THRESHOLD <- 0.5        # Classification threshold (adjustable)
CONFIDENCE_LEVEL <- 0.95        # For confidence intervals

cat("  Optimal threshold:", OPTIMAL_THRESHOLD, "\n")
cat("  Confidence level:", CONFIDENCE_LEVEL, "\n\n")

# Helper Functions ============================================================

# Comprehensive metrics calculation (following Jihad's approach)
calculate_comprehensive_metrics <- function(y_true, y_pred_prob, 
                                           threshold = 0.5,
                                           conf_level = 0.95) {
  # Binary classification
  y_pred_class <- ifelse(y_pred_prob >= threshold, 1, 0)
  
  # Confusion matrix
  tp <- sum(y_true == 1 & y_pred_class == 1)
  tn <- sum(y_true == 0 & y_pred_class == 0)
  fp <- sum(y_true == 0 & y_pred_class == 1)
  fn <- sum(y_true == 1 & y_pred_class == 0)
  
  # Basic metrics
  n <- length(y_true)
  accuracy <- (tp + tn) / n
  sensitivity <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
  specificity <- ifelse(tn + fp > 0, tn / (tn + fp), 0)
  precision <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
  npv <- ifelse(tn + fn > 0, tn / (tn + fn), 0)
  f1 <- ifelse(precision + sensitivity > 0,
               2 * (precision * sensitivity) / (precision + sensitivity), 0)
  f2 <- ifelse(precision + sensitivity > 0,
               5 * (precision * sensitivity) / (4 * precision + sensitivity), 0)
  
  # Additional metrics
  fpr <- ifelse(fp + tn > 0, fp / (fp + tn), 0)
  fnr <- ifelse(fn + tp > 0, fn / (fn + tp), 0)
  ppv <- precision
  
  # Matthews Correlation Coefficient
  mcc_num <- (tp * tn) - (fp * fn)
  mcc_den <- sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  mcc <- ifelse(mcc_den == 0, 0, mcc_num / mcc_den)
  
  # ROC and AUC
  roc_obj <- NULL
  auc <- NA
  auc_ci_lower <- NA
  auc_ci_upper <- NA
  youden_threshold <- NA
  youden_index <- NA
  youden_sensitivity <- NA
  youden_specificity <- NA
  
  tryCatch({
    roc_obj <- pROC::roc(y_true, y_pred_prob, 
                         levels = c(0, 1), 
                         direction = "<", 
                         quiet = TRUE)
    
    auc <- as.numeric(roc_obj$auc)
    auc_ci <- ci.auc(roc_obj, conf.level = conf_level)
    auc_ci_lower <- auc_ci[1]
    auc_ci_upper <- auc_ci[3]
    
    # Youden's Index
    coords_youden <- coords(roc_obj, "best", 
                           ret = c("threshold", "specificity", "sensitivity"),
                           best.method = "youden")
    
    youden_threshold <- coords_youden$threshold
    youden_index <- coords_youden$sensitivity + coords_youden$specificity - 1
    youden_sensitivity <- coords_youden$sensitivity
    youden_specificity <- coords_youden$specificity
  }, error = function(e) {
    warning("Could not calculate ROC metrics: ", e$message)
  })
  
  # Brier score (calibration)
  brier_score <- mean((y_pred_prob - y_true)^2)
  
  # Log loss
  epsilon <- 1e-15  # Avoid log(0)
  y_pred_clipped <- pmax(pmin(y_pred_prob, 1 - epsilon), epsilon)
  log_loss <- -mean(y_true * log(y_pred_clipped) + 
                    (1 - y_true) * log(1 - y_pred_clipped))
  
  return(list(
    # Confusion matrix
    tp = tp, tn = tn, fp = fp, fn = fn,
    
    # Classification metrics
    accuracy = accuracy,
    sensitivity = sensitivity,
    specificity = specificity,
    precision = precision,
    ppv = ppv,
    npv = npv,
    f1 = f1,
    f2 = f2,
    fpr = fpr,
    fnr = fnr,
    mcc = mcc,
    
    # Probability metrics
    auc = auc,
    auc_ci_lower = auc_ci_lower,
    auc_ci_upper = auc_ci_upper,
    brier_score = brier_score,
    log_loss = log_loss,
    
    # Optimal threshold metrics
    youden_threshold = youden_threshold,
    youden_index = youden_index,
    youden_sensitivity = youden_sensitivity,
    youden_specificity = youden_specificity,
    
    # ROC object for plotting
    roc_obj = roc_obj
  ))
}

# Load Test Data ==============================================================
cat(strrep("=", 80) %+% "\n")
cat("Loading Test Data\n")
cat(strrep("=", 80) %+% "\n\n")

test_file <- file.path(DATA_DIR, "test_set.rds")

if (!file.exists(test_file)) {
  stop("Test data not found: ", test_file, 
       "\nRun 01_prepare_data.R first!")
}

test_set <- readRDS(test_file)
cat("Test set loaded:", nrow(test_set), "samples\n")

# Extract test labels
y_test <- as.numeric(test_set$label)

cat("  ADRD cases:", sum(y_test), 
    sprintf("(%.1f%%)", mean(y_test) * 100), "\n")
cat("  Control cases:", sum(y_test == 0), 
    sprintf("(%.1f%%)", mean(y_test == 0) * 100), "\n\n")

# Load Model Utilities ========================================================
cat("Loading model utilities...\n")
source("utils_model_loader.R")
cat("\n")

# Load Artifacts ==============================================================
cat(strrep("=", 80) %+% "\n")
cat("Loading Tokenizer and Artifacts (Auto-detection)\n")
cat(strrep("=", 80) %+% "\n\n")

# Use automatic artifact loading (compatible with both naming conventions)
artifacts <- load_all_artifacts(MODEL_DIR)

# Extract for compatibility with rest of script
tokenizer <- artifacts$tokenizer
maxlen <- artifacts$maxlen
vocab_size <- artifacts$vocab_size

cat("\n")

# Prepare Test Sequences ======================================================
cat("Preparing test sequences...\n")

# Convert text to sequences (following Jihad's approach)
seq_test <- texts_to_sequences(tokenizer, test_set$txt)

# Pad sequences
x_test <- pad_sequences(seq_test, maxlen = maxlen, padding = "pre")

cat("  Test tensor shape:", paste(dim(x_test), collapse = " x "), "\n\n")

# Find Available Models =======================================================
cat(strrep("=", 80) %+% "\n")
cat("Finding Available Models (Auto-detection)\n")
cat(strrep("=", 80) %+% "\n\n")

# Use utility function to find models (compatible with both naming conventions)
model_files <- artifacts$model_files
cycles <- artifacts$cycles

cat("Models detected using automatic naming convention detection\n\n")

# Evaluate All Models =========================================================
cat(strrep("=", 80) %+% "\n")
cat("Evaluating All Models\n")
cat(strrep("=", 80) %+% "\n\n")

# Initialize storage (following Jihad's approach)
all_eval_results <- NULL
roc_df <- data.frame(Label = y_test)  # Start with labels
all_predictions <- matrix(nrow = length(cycles), ncol = length(y_test))
all_roc_objects <- list()

for (i in seq_along(cycles)) {
  cycle <- cycles[i]
  model_file <- model_files[i]
  
  cat(sprintf("Evaluating Cycle %d/%d...\n", cycle, length(cycles)))
  cat(strrep("-", 40), "\n")
  
  # Load model
  cat("  Loading model...\n")
  model <- load_model_hdf5(model_file)
  
  # Generate predictions
  cat("  Generating predictions...\n")
  predictions <- model %>% predict(x_test, verbose = 0)
  predictions_vec <- predictions[, 1]
  
  # Store predictions
  all_predictions[i, ] <- predictions_vec
  
  # Add to roc_df (following Jihad's approach)
  model_col_name <- paste0("CNNr_", sprintf("%02d", cycle))
  roc_df <- cbind(roc_df, predictions_vec)
  colnames(roc_df)[ncol(roc_df)] <- model_col_name
  
  # Calculate metrics
  cat("  Calculating metrics...\n")
  metrics <- calculate_comprehensive_metrics(
    y_test, 
    predictions_vec,
    threshold = OPTIMAL_THRESHOLD,
    conf_level = CONFIDENCE_LEVEL
  )
  
  # Store ROC object
  all_roc_objects[[i]] <- metrics$roc_obj
  
  # Create results dataframe (following Jihad's structure)
  eval_df <- data.frame(
    model = "CNNr",
    cycle = cycle,
    
    # Confusion matrix
    tp = metrics$tp,
    tn = metrics$tn,
    fp = metrics$fp,
    fn = metrics$fn,
    
    # Classification metrics
    auc = metrics$auc,
    auc_ci_lower = metrics$auc_ci_lower,
    auc_ci_upper = metrics$auc_ci_upper,
    accuracy = metrics$accuracy,
    sensitivity = metrics$sensitivity,
    specificity = metrics$specificity,
    precision = metrics$precision,
    ppv = metrics$ppv,
    npv = metrics$npv,
    f1 = metrics$f1,
    f2 = metrics$f2,
    fpr = metrics$fpr,
    fnr = metrics$fnr,
    mcc = metrics$mcc,
    
    # Probability metrics
    brier_score = metrics$brier_score,
    log_loss = metrics$log_loss,
    
    # Youden metrics
    youden_threshold = metrics$youden_threshold,
    youden_index = metrics$youden_index,
    youden_sensitivity = metrics$youden_sensitivity,
    youden_specificity = metrics$youden_specificity,
    
    threshold_used = OPTIMAL_THRESHOLD,
    stringsAsFactors = FALSE
  )
  
  # Accumulate results
  if (is.null(all_eval_results)) {
    all_eval_results <- eval_df
  } else {
    all_eval_results <- rbind(all_eval_results, eval_df)
  }
  
  # Print key metrics
  cat(sprintf("  AUC: %.4f", metrics$auc))
  if (!is.na(metrics$auc_ci_lower)) {
    cat(sprintf(" (95%% CI: %.4f-%.4f)", 
                metrics$auc_ci_lower, metrics$auc_ci_upper))
  }
  cat("\n")
  cat(sprintf("  Accuracy: %.4f\n", metrics$accuracy))
  cat(sprintf("  Sensitivity: %.4f\n", metrics$sensitivity))
  cat(sprintf("  Specificity: %.4f\n", metrics$specificity))
  cat(sprintf("  F1 Score: %.4f\n\n", metrics$f1))
  
  # Clean up
  rm(model)
  gc()
}

cat("All models evaluated successfully\n\n")

# Select Best Model ===========================================================
cat(strrep("=", 80) %+% "\n")
cat("Selecting Best Model (Median AUC with Max F1)\n")
cat(strrep("=", 80) %+% "\n\n")

# Find median AUC model (following Jihad's exact approach)
median_auc <- median(all_eval_results$auc, na.rm = TRUE)

best_row <- all_eval_results %>%
  mutate(auc_diff = abs(auc - median_auc)) %>%
  filter(auc_diff == min(auc_diff, na.rm = TRUE)) %>%
  filter(f1 == max(f1, na.rm = TRUE)) %>%
  slice(1)

best_cycle <- best_row$cycle
best_idx <- which(cycles == best_cycle)
best_metrics <- best_row

cat("Selection criteria: Median AUC with Max F1 (Jihad's method)\n")
cat("  Median AUC across all models:", sprintf("%.4f", median_auc), "\n")
cat("  Selected cycle:", best_cycle, "\n")
cat("  Selected AUC:", sprintf("%.4f", best_metrics$auc), "\n")
cat("  Selected F1:", sprintf("%.4f", best_metrics$f1), "\n\n")

# Get best model predictions
best_pred <- all_predictions[best_idx, ]

# Save best model info
best_model_info <- list(
  best_cycle = best_cycle,
  best_idx = best_idx,
  median_auc = median_auc,
  best_model_file = model_files[best_idx],
  best_metrics = best_metrics,
  best_predictions = best_pred
)

saveRDS(best_model_info, 
        file.path(RESULTS_DIR, "best_model_evaluation.rds"))
cat("Best model info saved\n\n")

# Save All Results ============================================================
cat(strrep("=", 80) %+% "\n")
cat("Saving Evaluation Results\n")
cat(strrep("=", 80) %+% "\n\n")

# Save comprehensive results

eval_summary_file <- file.path(RESULTS_DIR, 
                               paste0("evaluation_summary.csv"))
write_csv(all_eval_results, eval_summary_file)
cat("Evaluation summary saved:", eval_summary_file, "\n")

saveRDS(all_eval_results, 
        file.path(RESULTS_DIR, paste0("evaluation_summary.rds")))

# Save as Excel (following Jihad's approach)
write_xlsx(all_eval_results, 
           file.path(RESULTS_DIR, paste0("Summary_Metrics_CNNr.xlsx")))
cat("Excel summary saved\n")

# Save ROC dataframe (following Jihad's format)
roc_df_file <- file.path(RESULTS_DIR, paste0("roc_df.rds"))
saveRDS(roc_df, roc_df_file)
cat("ROC dataframe saved:", roc_df_file, "\n")

# Save predictions matrix
predictions_file <- file.path(RESULTS_DIR, "evaluation_predictions.rds")
saveRDS(all_predictions, predictions_file)
cat("Predictions matrix saved:", predictions_file, "\n")

# Detailed predictions for best model (following Jihad's format)
detailed_predictions <- data.frame(
  DE_ID = test_set$DE_ID,
  label_icd = y_test,
  Label = y_test,
  True_Class = ifelse(y_test == 1, "ADRD", "CTRL"),
  Predicted_Probability = round(best_pred, 4),
  Predicted_Class = ifelse(best_pred >= OPTIMAL_THRESHOLD, "ADRD", "CTRL"),
  Correct = (y_test == 1 & best_pred >= OPTIMAL_THRESHOLD) |
            (y_test == 0 & best_pred < OPTIMAL_THRESHOLD),
  Risk_Category = cut(best_pred,
                      breaks = c(0, 0.3, 0.7, 1.0),
                      labels = c("Low", "Moderate", "High"),
                      include.lowest = TRUE),
  stringsAsFactors = FALSE
)

# Add demographics if available
if ("GENDER" %in% names(test_set)) {
  detailed_predictions$GENDER <- test_set$GENDER
}
if ("RACE" %in% names(test_set)) {
  detailed_predictions$RACE <- test_set$RACE
}
if ("HISPANIC" %in% names(test_set)) {
  detailed_predictions$HISPANIC <- test_set$HISPANIC
}

write_csv(detailed_predictions, 
          file.path(RESULTS_DIR, paste0("predictions_df.csv")))
cat("Detailed predictions saved\n\n")

# Summary Statistics ==========================================================
cat(strrep("=", 80) %+% "\n")
cat("Summary Statistics\n")
cat(strrep("=", 80) %+% "\n\n")

summary_stats <- all_eval_results %>%
  summarise(
    N_Models = n(),
    Mean_AUC = mean(auc, na.rm = TRUE),
    SD_AUC = sd(auc, na.rm = TRUE),
    Min_AUC = min(auc, na.rm = TRUE),
    Max_AUC = max(auc, na.rm = TRUE),
    Median_AUC = median(auc, na.rm = TRUE),
    Mean_Accuracy = mean(accuracy, na.rm = TRUE),
    Mean_Sensitivity = mean(sensitivity, na.rm = TRUE),
    Mean_Specificity = mean(specificity, na.rm = TRUE),
    Mean_F1 = mean(f1, na.rm = TRUE)
  )

cat("Performance Across All Models:\n")
cat(strrep("-", 40), "\n")
print(summary_stats)
cat("\n")

cat("Best Model Performance:\n")
cat(strrep("-", 40), "\n")
cat("  Cycle:", best_metrics$cycle, "\n")
cat("  AUC:", sprintf("%.4f", best_metrics$auc))
if (!is.na(best_metrics$auc_ci_lower)) {
  cat(sprintf(" (95%% CI: %.4f-%.4f)", 
              best_metrics$auc_ci_lower, best_metrics$auc_ci_upper))
}
cat("\n")
cat("  Accuracy:", sprintf("%.4f", best_metrics$accuracy), "\n")
cat("  Sensitivity:", sprintf("%.4f", best_metrics$sensitivity), "\n")
cat("  Specificity:", sprintf("%.4f", best_metrics$specificity), "\n")
cat("  Precision:", sprintf("%.4f", best_metrics$precision), "\n")
cat("  F1 Score:", sprintf("%.4f", best_metrics$f1), "\n")
cat("  MCC:", sprintf("%.4f", best_metrics$mcc), "\n\n")

# Create Visualizations =======================================================
cat(strrep("=", 80) %+% "\n")
cat("Creating Visualizations\n")
cat(strrep("=", 80) %+% "\n\n")

# 1. ROC Curve (following Jihad's ggplot approach) ===========================
cat("Creating ROC curve...\n")

# Function to create ROC plot (following Jihad's implementation)
ggplot_roc_df <- function(roc_df, zoom = FALSE) {
  n <- ncol(roc_df)  # Number of columns (models + Label)
  
  # Define colors
  mod_cols <- c("#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6")
  mod_nms <- names(roc_df)[2:n]  # Skip Label column
  mod_nms_auc <- character()
  
  # Create ROC objects
  ROC <- lapply(roc_df[, 2:n], function(x) {
    pROC::roc(roc_df$Label, x, levels = c(0, 1), direction = "<", quiet = TRUE)
  })
  
  # Initialize dataframe
  df <- data.frame(
    x = numeric(),
    y = numeric(),
    mdl = character(),
    stringsAsFactors = FALSE
  )
  
  # Create model names with AUC
  for (i in seq_along(ROC)) {
    auc_val <- as.numeric(ROC[[i]]$auc)
    mod_nms_auc <- c(mod_nms_auc, 
                     sprintf("%s (AUC=%.3f)", mod_nms[i], auc_val))
  }
  
  # Fill dataframe with ROC data
  for (i in seq_along(ROC)) {
    df <- rbind(df, data.frame(
      x = ROC[[i]]$specificities,
      y = ROC[[i]]$sensitivities,
      mdl = mod_nms_auc[i],
      stringsAsFactors = FALSE
    ))
  }
  
  # Sort data
  df <- df %>%
    arrange(mdl, desc(x), y) %>%
    mutate(mdl = factor(mdl, levels = mod_nms_auc))
  
  # Create plot
  g1 <- ggplot(df, aes(x = x, y = y, group = mdl, color = mdl)) +
    geom_line(size = 0.8) +
    scale_color_manual(values = rep(mod_cols, length.out = length(mod_nms_auc))) +
    labs(x = "Specificity", y = "Sensitivity", col = "")
  
  # Conditional formatting
  if (zoom) {
    g1 <- g1 +
      scale_y_continuous(breaks = seq(0, 1, 0.1)) +
      scale_x_reverse(breaks = seq(0, 1, 0.1)) +
      coord_cartesian(ylim = c(0.5, 1), xlim = c(1, 0.5))
  } else {
    g1 <- g1 +
      geom_abline(intercept = 1, slope = 1, col = "grey", linetype = "dashed") +
      scale_y_continuous(breaks = seq(0, 1, 0.2)) +
      scale_x_reverse(breaks = seq(0, 1, 0.2))
  }
  
  # Theme (following Jihad's style)
  g1 <- g1 +
    theme_classic() +
    theme(
      panel.border = element_rect(color = "black", fill = NA),
      legend.justification = c(1, 0),
      legend.title = element_blank(),
      legend.text = element_text(size = rel(1.4), hjust = 1),
      legend.background = element_blank(),
      legend.key.width = unit(rel(2), "line"),
      legend.box.background = element_rect(color = "black"),
      axis.text = element_text(size = rel(1.4)),
      axis.title = element_text(size = rel(1.6), face = "bold"),
      aspect.ratio = 1,
      legend.position = c(1.0, 0)
    )
  
  return(g1)
}

# Create ROC plot
roc_plot <- ggplot_roc_df(roc_df, zoom = FALSE)

# Save ROC plot (following Jihad's naming)
roc_file <- file.path(FIGURES_DIR, paste0("AUC_CNNr.png"))
ggsave(roc_file, plot = roc_plot,
       scale = 1, width = 6, height = 6, units = "in", dpi = 300)
cat("  ROC curve saved:", roc_file, "\n")

# Zoomed ROC
roc_plot_zoom <- ggplot_roc_df(roc_df, zoom = TRUE)
roc_zoom_file <- file.path(FIGURES_DIR, paste0("AUC_CNNr_zoom.png"))
ggsave(roc_zoom_file, plot = roc_plot_zoom,
       scale = 1, width = 6, height = 6, units = "in", dpi = 300)
cat("  Zoomed ROC curve saved:", roc_zoom_file, "\n")

# 2. Metrics Distribution =====================================================
cat("Creating metrics distribution plots...\n")

# Prepare metrics data
metrics_long <- all_eval_results %>%
  select(cycle, auc, accuracy, sensitivity, specificity, f1) %>%
  pivot_longer(cols = -cycle, names_to = "Metric", values_to = "Value") %>%
  mutate(Metric = factor(Metric, 
                         levels = c("auc", "accuracy", "sensitivity", 
                                   "specificity", "f1"),
                         labels = c("AUC", "Accuracy", "Sensitivity", 
                                   "Specificity", "F1 Score")))
    

# Box plot
metrics_boxplot <- ggplot(metrics_long, aes(x = Metric, y = Value, fill = Metric)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.5, size = 2) +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "Performance Metrics Distribution",
    subtitle = sprintf("Across %d training cycles", length(cycles)),
    x = "Metric",
    y = "Value"
  ) +
  theme_classic() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 12, face = "bold"),
    legend.position = "none"
  )

ggsave(file.path(FIGURES_DIR, "metrics_boxplot.png"),
       plot = metrics_boxplot, width = 10, height = 6, dpi = 300)
cat("  Metrics boxplot saved\n")

# 3. Calibration Plot =========================================================
cat("Creating calibration plot...\n")

# Use best model predictions
best_pred <- all_predictions[best_idx, ]

# Create calibration bins
n_bins <- 10
calibration_data <- data.frame(
  Predicted = best_pred,
  Actual = y_test
) %>%
  mutate(Bin = cut(Predicted, breaks = seq(0, 1, length.out = n_bins + 1),
                   include.lowest = TRUE)) %>%
  group_by(Bin) %>%
  summarise(
    Mean_Predicted = mean(Predicted),
    Mean_Actual = mean(Actual),
    N = n(),
    .groups = "drop"
  )

calibration_plot <- ggplot(calibration_data, 
                           aes(x = Mean_Predicted, y = Mean_Actual)) +
  geom_point(aes(size = N), alpha = 0.7, color = "#3498DB") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", 
              color = "#E74C3C") +
  geom_smooth(method = "loess", se = TRUE, color = "#2ECC71") +
  scale_size_continuous(name = "Sample Size") +
  labs(
    title = "Calibration Plot - Best Model",
    subtitle = sprintf("Cycle %d", best_cycle),
    x = "Predicted Probability",
    y = "Observed Frequency"
  ) +
  theme_classic() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 12, face = "bold"),
    legend.position = "bottom",
    aspect.ratio = 1
  )

ggsave(file.path(FIGURES_DIR, "calibration_plot.png"),
       plot = calibration_plot, width = 8, height = 8, dpi = 300)
cat("  Calibration plot saved\n")

# 4. Probability Distribution =================================================
cat("Creating probability distribution plot...\n")

dist_data <- data.frame(
  Probability = best_pred,
  True_Class = ifelse(y_test == 1, "ADRD", "NON-ADRD")
)

prob_dist_plot <- ggplot(dist_data, aes(x = Probability, fill = True_Class)) +
  geom_histogram(alpha = 0.6, position = "identity", bins = 30) +
  geom_vline(xintercept = OPTIMAL_THRESHOLD, linetype = "dashed", 
             color = "black", size = 1) +
  annotate("text", x = OPTIMAL_THRESHOLD + 0.05, y = Inf, 
           label = sprintf("Threshold: %.3f", OPTIMAL_THRESHOLD),
           vjust = 1.5, hjust = 0) +
  scale_fill_manual(values = c("ADRD" = "#E74C3C", "NON-ADRD" = "#3498DB")) +
  labs(
    title = "Predicted Probability Distribution",
    subtitle = sprintf("Best Model: Cycle %d", best_cycle),
    x = "ADRD Probability",
    y = "Count",
    fill = "True Class"
  ) +
  theme_classic() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 12, face = "bold"),
    legend.position = "bottom"
  )

ggsave(file.path(FIGURES_DIR, "probability_distribution.png"),
       plot = prob_dist_plot, width = 10, height = 6, dpi = 300)
cat("  Probability distribution saved\n")

# 5. Confusion Matrix Heatmap =================================================
cat("Creating confusion matrix heatmap...\n")

conf_matrix_data <- data.frame(
  Predicted = c("NON-ADRD", "ADRD", "NON-ADRD", "ADRD"),
  Actual = c("NON-ADRD", "NON-ADRD", "ADRD", "ADRD"),
  Count = c(best_metrics$tn, best_metrics$fp, 
            best_metrics$fn, best_metrics$tp)
) %>%
  mutate(
    Label = sprintf("%d", Count),
    Predicted = factor(Predicted, levels = c("ADRD", "NON-ADRD")),
    Actual = factor(Actual, levels = c("NON-ADRD", "ADRD"))
  )

confusion_plot <- ggplot(conf_matrix_data, 
                        aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile(color = "white", size = 2) +
  geom_text(aes(label = Label), size = 12, fontface = "bold") +
  scale_fill_gradient(low = "#F0F8FF", high = "#3498DB") +
  labs(
    title = "Confusion Matrix - Best Model",
    subtitle = sprintf("Cycle %d (Threshold: %.3f)", 
                      best_cycle, OPTIMAL_THRESHOLD),
    x = "Predicted Class",
    y = "Actual Class"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5),
    axis.text = element_text(size = 12, face = "bold"),
    axis.title = element_text(size = 13, face = "bold"),
    legend.position = "right",
    panel.grid = element_blank(),
    aspect.ratio = 1
  )

ggsave(file.path(FIGURES_DIR, "confusion_matrix.png"),
       plot = confusion_plot, width = 8, height = 8, dpi = 300)
cat("  Confusion matrix saved\n\n")

# Create Evaluation Report ====================================================
cat(strrep("=", 80) %+% "\n")
cat("Creating Evaluation Report\n")
cat(strrep("=", 80) %+% "\n\n")

report_file <- file.path(RESULTS_DIR, "evaluation_report.txt")
sink(report_file)

cat("ADRD CNN Evaluation Report\n")
cat(strrep("=", 80), "\n\n")

cat("Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

cat("Test Set Information:\n")
cat(strrep("-", 40), "\n")
cat("Total samples:", length(y_test), "\n")
cat("ADRD cases:", sum(y_test), sprintf("(%.1f%%)", mean(y_test) * 100), "\n")
cat("Control cases:", sum(y_test == 0), 
    sprintf("(%.1f%%)", mean(y_test == 0) * 100), "\n\n")

cat("Models Evaluated:\n")
cat(strrep("-", 40), "\n")
cat("Number of models:", length(cycles), "\n")
cat("Cycles:", paste(cycles, collapse = ", "), "\n\n")

cat("Performance Summary (All Models):\n")
cat(strrep("-", 40), "\n")
cat("Mean AUC:", sprintf("%.4f", summary_stats$Mean_AUC),
    "±", sprintf("%.4f", summary_stats$SD_AUC), "\n")
cat("Range:", sprintf("%.4f - %.4f", 
                     summary_stats$Min_AUC, summary_stats$Max_AUC), "\n")
cat("Mean Accuracy:", sprintf("%.4f", summary_stats$Mean_Accuracy), "\n")
cat("Mean Sensitivity:", sprintf("%.4f", summary_stats$Mean_Sensitivity), "\n")
cat("Mean Specificity:", sprintf("%.4f", summary_stats$Mean_Specificity), "\n")
cat("Mean F1 Score:", sprintf("%.4f", summary_stats$Mean_F1), "\n\n")

cat("Best Model Selection:\n")
cat(strrep("-", 40), "\n")
cat("Selection criteria: Median AUC\n")
cat("Median AUC:", sprintf("%.4f", median_auc), "\n")
cat("Selected cycle:", best_cycle, "\n\n")

cat("Best Model Performance (Threshold:", OPTIMAL_THRESHOLD, "):\n")
cat(strrep("-", 40), "\n")
cat("AUC:", sprintf("%.4f", best_metrics$auc),
    sprintf("(95%% CI: %.4f-%.4f)\n", 
            best_metrics$auc_ci_lower, best_metrics$auc_ci_upper))
cat("Accuracy:", sprintf("%.4f", best_metrics$accuracy), "\n")
cat("Sensitivity (Recall):", sprintf("%.4f", best_metrics$sensitivity), "\n")
cat("Specificity:", sprintf("%.4f", best_metrics$specificity), "\n")
cat("Precision (PPV):", sprintf("%.4f", best_metrics$precision), "\n")
cat("NPV:", sprintf("%.4f", best_metrics$npv), "\n")
cat("F1 Score:", sprintf("%.4f", best_metrics$f1), "\n")
cat("F2 Score:", sprintf("%.4f", best_metrics$f2), "\n")
cat("MCC:", sprintf("%.4f", best_metrics$mcc), "\n\n")

cat("Confusion Matrix:\n")
cat(strrep("-", 40), "\n")
cat(sprintf("%-15s %-10s %-10s\n", "", "Pred: CTRL", "Pred: ADRD"))
cat(sprintf("%-15s %-10d %-10d\n", "True: CTRL", best_metrics$tn, best_metrics$fp))
cat(sprintf("%-15s %-10d %-10d\n", "True: ADRD", best_metrics$fn, best_metrics$tp))
cat("\n")

cat("Error Rates:\n")
cat(strrep("-", 40), "\n")
cat("False Positive Rate:", sprintf("%.4f", best_metrics$fpr), "\n")
cat("False Negative Rate:", sprintf("%.4f", best_metrics$fnr), "\n\n")

cat("Calibration Metrics:\n")
cat(strrep("-", 40), "\n")
cat("Brier Score:", sprintf("%.4f", best_metrics$brier_score), "\n")
cat("Log Loss:", sprintf("%.4f", best_metrics$log_loss), "\n\n")

cat("Youden's Index Analysis:\n")
cat(strrep("-", 40), "\n")
cat("Optimal threshold:", sprintf("%.4f", best_metrics$youden_threshold), "\n")
cat("Youden's Index:", sprintf("%.4f", best_metrics$youden_index), "\n")
cat("Sensitivity at optimal:", 
    sprintf("%.4f", best_metrics$youden_sensitivity), "\n")
cat("Specificity at optimal:", 
    sprintf("%.4f", best_metrics$youden_specificity), "\n\n")

cat("Per-Cycle Results:\n")
cat(strrep("-", 40), "\n")
print(all_eval_results %>%
      select(cycle, auc, accuracy, sensitivity, specificity, f1, mcc))

sink()

cat("Evaluation report saved:", report_file, "\n\n")

# Final Summary ===============================================================
cat(strrep("=", 80) %+% "\n")
cat("EVALUATION COMPLETE\n")
cat(strrep("=", 80) %+% "\n\n")

cat("Summary:\n")
cat("  ", length(cycles), "models evaluated\n")
cat("  Best model: Cycle", best_cycle, 
    sprintf("(AUC: %.4f)\n", best_metrics$auc))
cat("  Mean AUC:", sprintf("%.4f ± %.4f\n", 
                          summary_stats$Mean_AUC, summary_stats$SD_AUC))

cat("\nOutput Files:\n")
cat("  Evaluation summary:", eval_summary_file, "\n")
cat("  Detailed predictions:", 
    file.path(RESULTS_DIR, "detailed_predictions.csv"), "\n")
cat("  Report:", report_file, "\n")
cat("  Figures:", FIGURES_DIR, "/\n\n")

cat("Generated Figures:\n")
cat("  - roc_curve.png\n")
cat("  - roc_curve_zoom.png\n")
cat("  - metrics_boxplot.png\n")
cat("  - calibration_plot.png\n")
cat("  - probability_distribution.png\n")
cat("  - confusion_matrix.png\n\n")

cat("Next Steps:\n")
cat("  1. Review evaluation report and figures\n")
cat("  2. Run 04_demographic_analysis.R for subgroup analysis\n")
cat("  3. Run 05_inference.R to apply model to new data\n\n")

cat(strrep("=", 80) %+% "\n")
cat("Evaluation script completed successfully!\n")
cat(strrep("=", 80) %+% "\n")