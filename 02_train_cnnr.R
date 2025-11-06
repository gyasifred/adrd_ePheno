#!/usr/bin/env Rscript
# ==============================================================================
# ADRD Classification Pipeline - CNN Training (CNNr) [REFERENCE ONLY]
# ==============================================================================
# Version: 2.0
#
# ⚠️  WARNING: This script is for REFERENCE ONLY
# ⚠️  Use Jihad Obeid's pre-trained models instead - NO training required!
# ⚠️  See README.md for instructions on using pre-trained models
# ⚠️  Copy Jihad's models to models/ directory and run 03_evaluate_models.R
#
# Purpose: Documents CNN training process with random embeddings
# Code follows methodology from Jihad Obeid's original implementation
#
# Inputs:  data/train_set.rds, data/test_set.rds
# Outputs: models/*, results/*, All inference artifacts
# ==============================================================================

# Load Libraries --------------------------------------------------------------
cat(paste0(strrep("=", 80), "\n"))
cat("ADRD CNN Training Pipeline - CNNr (Random Embeddings)\n")
cat(paste0(strrep("=", 80), "\n\n"))

cat("Loading required libraries...\n")
suppressPackageStartupMessages({
  library(reticulate)   # Python/TensorFlow integration
  library(keras)        # Neural networks (Keras 2)
  library(tensorflow)   # TensorFlow backend
  library(tidyverse)    # Data manipulation
  library(magrittr)     # Pipe operators
  library(tictoc)       # Timing
  library(pROC)         # ROC analysis
})

# Define operators for compatibility
`%+%` <- function(a, b) paste0(a, b)

options(dplyr.summarise.inform = FALSE)
set.seed(42)  # Reproducibility

# Configure Environment -------------------------------------------------------
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

# GPU memory growth configuration
physical_devices <- tf$config$list_physical_devices('GPU')
if (length(physical_devices) > 0) {
  cat("  GPU detected:", length(physical_devices), "device(s)\n")
  for (gpu in physical_devices) {
    tf$config$experimental$set_memory_growth(gpu, TRUE)
  }
} else {
  cat("  No GPU detected - using CPU\n")
}

# Configuration ---------------------------------------------------------------
cat("\nConfiguration Parameters:\n")
cat(strrep("-", 80) %+% "\n")

# Directory paths
DATA_DIR    <- "data"
MODEL_DIR   <- "models"
RESULTS_DIR <- "results"
FIGURES_DIR <- "figures"

# Create directories
dirs <- c(MODEL_DIR, RESULTS_DIR, FIGURES_DIR)
for (d in dirs) dir.create(d, showWarnings = FALSE, recursive = TRUE)

# Model hyperparameters (following Jihad's configuration)
MAX_VOCAB       <- 40000      # Maximum vocabulary size
EMBEDDING_DIM   <- 200        # Embedding vector dimension
NUM_FILTERS     <- 200        # Number of CNN filters per kernel
FILTER_SIZES    <- c(3, 4, 5) # Kernel sizes for multi-scale feature extraction
HIDDEN_DIMS     <- 200        # Hidden layer size
DROPOUT_RATE    <- 0.2        # Dropout rate for regularization

# Training parameters (following Jihad's configuration)
N_CYCLES        <- 10         # Number of training cycles for robustness
EPOCHS          <- 30         # Maximum epochs per cycle
BATCH_SIZE      <- 32         # Mini-batch size
LEARNING_RATE   <- 0.0004     # Initial learning rate
VALIDATION_SPLIT <- 0.15      # Validation split ratio
EARLY_STOP_PATIENCE <- 7      # Early stopping patience

cat("Model Architecture:\n")
cat("  Max vocab:", MAX_VOCAB, "\n")
cat("  Embedding dim:", EMBEDDING_DIM, "\n")
cat("  Filters per kernel:", NUM_FILTERS, "\n")
cat("  Kernel sizes:", paste(FILTER_SIZES, collapse=", "), "\n")
cat("  Hidden units:", HIDDEN_DIMS, "\n")
cat("  Dropout:", DROPOUT_RATE, "\n\n")

cat("Training Configuration:\n")
cat("  Cycles:", N_CYCLES, "\n")
cat("  Epochs/cycle:", EPOCHS, "\n")
cat("  Batch size:", BATCH_SIZE, "\n")
cat("  Learning rate:", LEARNING_RATE, "\n")
cat("  Validation split:", VALIDATION_SPLIT, "\n")
cat("  Early stop patience:", EARLY_STOP_PATIENCE, "\n\n")

# Load Data -------------------------------------------------------------------
cat(paste0(strrep("=", 80), "\n"))
cat("Loading Data\n")
cat(paste0(strrep("=", 80), "\n\n"))

tic("Data loading")

train_file <- file.path(DATA_DIR, "train_set.rds")
test_file  <- file.path(DATA_DIR, "test_set.rds")

# Check files exist
for (f in c(train_file, test_file)) {
  if (!file.exists(f)) {
    stop("File not found: ", f, "\nRun 01_prepare_data.R first!")
  }
}

# Load datasets
train_set <- readRDS(train_file)
test_set  <- readRDS(test_file)

cat("Training set:", nrow(train_set), "samples\n")
cat("Test set:    ", nrow(test_set),  "samples\n\n")

# Validate columns
required_cols <- c("DE_ID", "txt", "label")
for (nm in c("train_set", "test_set")) {
  missing <- setdiff(required_cols, names(get(nm)))
  if (length(missing) > 0) {
    stop(nm, " missing columns: ", paste(missing, collapse=", "))
  }
}

# Display class distribution
cat("Class distribution:\n")
cat("Train:\n"); print(table(train_set$label))
cat("  ADRD %:", sprintf("%.1f%%", mean(train_set$label == 1) * 100), "\n\n")
cat("Test:\n");  print(table(test_set$label))
cat("  ADRD %:", sprintf("%.1f%%", mean(test_set$label == 1) * 100), "\n\n")

toc()

# Text Tokenization -----------------------------------------------------------
cat(paste0(strrep("=", 80), "\n"))
cat("Text Tokenization & Sequence Preparation\n")
cat(paste0(strrep("=", 80), "\n\n"))

tic("Tokenization")

cat("Creating tokenizer...\n")
# Create tokenizer following Jihad's approach
tokenizer <- text_tokenizer(
  num_words = MAX_VOCAB,
  filters = "",           # No filtering (text already preprocessed)
  lower = TRUE,          # Convert to lowercase
  split = " "            # Split on spaces
)

cat("Fitting tokenizer on combined text...\n")
# Fit on all available text (train + test) to capture full vocabulary
combined_text <- c(train_set$txt, test_set$txt)
tokenizer %>% fit_text_tokenizer(combined_text)

# Extract word index and calculate vocabulary size
word_index <- tokenizer$word_index
vocab_size <- min(length(word_index), MAX_VOCAB) + 1  # +1 for padding token

cat("  Unique tokens found:", length(word_index), "\n")
cat("  Vocab size (with padding):", vocab_size, "\n\n")

# ==============================================================================
# Save Tokenizer - Keras 2 Method (following Jihad's approach)
# ==============================================================================
cat("Saving tokenizer and artifacts...\n")

# Save tokenizer using Keras 2 method
tokenizer_file <- file.path(MODEL_DIR, "tokenizer_cnnr")
tokenizer %>% save_text_tokenizer(tokenizer_file)
cat("  Tokenizer saved to:", tokenizer_file, "\n")

# Save supporting artifacts (critical for inference)
saveRDS(word_index, file.path(MODEL_DIR, "word_index.rds"))
cat("  word_index.rds saved\n")

saveRDS(vocab_size, file.path(MODEL_DIR, "vocab_size.rds"))
cat("  vocab_size.rds saved\n")

# Analyze sequence lengths
cat("\nConverting to sequences and analyzing lengths...\n")
all_seq <- texts_to_sequences(tokenizer, combined_text)
word_counts <- sapply(all_seq, length)

cat("  Sequence length statistics:\n")
print(summary(word_counts))
cat("\n")

# Determine maximum length (+1 buffer, following Jihad's approach)
maxlen <- max(word_counts) + 1
cat("  Maximum sequence length (maxlen):", maxlen, "\n")

# Save maxlen (critical for inference)
saveRDS(maxlen, file.path(MODEL_DIR, "maxlen.rds"))
cat("  maxlen.rds saved\n\n")

# Prepare training and test sequences
cat("Preparing padded sequences...\n")
seq_train <- texts_to_sequences(tokenizer, train_set$txt)
seq_test  <- texts_to_sequences(tokenizer, test_set$txt)

# Pad sequences to uniform length (pre-padding, following Jihad's approach)
x_train <- pad_sequences(seq_train, maxlen = maxlen, padding = "pre")
x_test  <- pad_sequences(seq_test,  maxlen = maxlen, padding = "pre")

# Extract labels as numeric (0 and 1, following Jihad's approach)
y_train <- as.numeric(train_set$label)
y_test  <- as.numeric(test_set$label)

cat("  x_train shape:", paste(dim(x_train), collapse = " x "), "\n")
cat("  x_test  shape:", paste(dim(x_test),  collapse = " x "), "\n")
cat("  y_train length:", length(y_train), "\n")
cat("  y_test  length:", length(y_test), "\n\n")

toc()

# CNN Model Architecture ------------------------------------------------------
cat(paste0(strrep("=", 80), "\n"))
cat("CNN Model Architecture Definition\n")
cat(paste0(strrep("=", 80), "\n\n"))

# Build CNN model function (following Jihad's exact architecture)
build_cnn_model <- function(vocab_size, 
                            maxlen,
                            embedding_dim = 200,
                            num_filters = 200,
                            filter_sizes = c(3, 4, 5),
                            hidden_dims = 200,
                            dropout_rate = 0.2) {
  
  # Input layer: accepts integer sequences
  inputs <- layer_input(shape = maxlen, name = "input")
  
  # Embedding layer: converts word indices to dense vectors
  # Random initialization, learned during training
  embedding <- inputs %>%
    layer_embedding(
      input_dim = vocab_size,
      output_dim = embedding_dim,
      input_length = maxlen,
      name = "embedding"
    ) %>%
    layer_dropout(dropout_rate, name = "drop_embed")
  
  # Parallel CNN branches with different kernel sizes
  # This captures different n-gram patterns (3-grams, 4-grams, 5-grams)
  conv_branches <- list()
  
  for (filt_sz in filter_sizes) {
    conv_branch <- embedding %>%
      layer_conv_1d(
        filters = num_filters,
        kernel_size = filt_sz,
        activation = "relu",
        name = paste0("conv_", filt_sz)
      ) %>%
      layer_global_max_pooling_1d(name = paste0("pool_", filt_sz))
    
    conv_branches[[length(conv_branches) + 1]] <- conv_branch
  }
  
  # Concatenate features from all CNN branches
  if (length(conv_branches) > 1) {
    merged <- layer_concatenate(conv_branches, name = "concat")
  } else {
    merged <- conv_branches[[1]]
  }
  
  # Fully connected layers for classification
  output <- merged %>%
    layer_dense(hidden_dims, activation = "relu", name = "dense") %>%
    layer_dropout(dropout_rate, name = "drop_dense") %>%
    layer_dense(1, activation = "sigmoid", name = "output")
  
  # Create and compile model
  model <- keras_model(inputs, output)
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = LEARNING_RATE),
    loss = "binary_crossentropy",
    metrics = c("accuracy")
  )
  
  return(model)
}

# Build sample model for architecture display
cat("Building sample model for architecture summary...\n")
sample_model <- build_cnn_model(
  vocab_size, 
  maxlen,
  EMBEDDING_DIM, 
  NUM_FILTERS,
  FILTER_SIZES, 
  HIDDEN_DIMS, 
  DROPOUT_RATE
)

cat("\nModel Summary:\n")
cat(strrep("-", 80) %+% "\n")
print(summary(sample_model))

# Save architecture to file
arch_file <- file.path(MODEL_DIR, "model_architecture.txt")
sink(arch_file)
cat("CNN Model Architecture for ADRD Classification\n")
cat(strrep("=", 80), "\n\n")
print(summary(sample_model))
sink()
cat("\nArchitecture saved to:", arch_file, "\n\n")

# Clean up sample model
rm(sample_model)
gc()

# Metrics Calculation Function ------------------------------------------------
cat("Defining metrics calculation function...\n")

# Calculate comprehensive metrics (following Jihad's approach)
calculate_metrics <- function(y_true, y_pred_prob, threshold = 0.5) {
  y_pred_class <- ifelse(y_pred_prob >= threshold, 1, 0)
  
  # Confusion matrix
  tp <- sum(y_true == 1 & y_pred_class == 1)
  tn <- sum(y_true == 0 & y_pred_class == 0)
  fp <- sum(y_true == 0 & y_pred_class == 1)
  fn <- sum(y_true == 1 & y_pred_class == 0)
  
  # Calculate metrics
  accuracy     <- (tp + tn) / length(y_true)
  sensitivity  <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
  specificity  <- ifelse(tn + fp > 0, tn / (tn + fp), 0)
  precision    <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
  f1           <- ifelse(precision + sensitivity > 0,
                        2 * precision * sensitivity / (precision + sensitivity), 0)
  
  # Calculate AUC
  auc <- NA
  tryCatch({
    roc_obj <- pROC::roc(y_true, y_pred_prob, 
                        levels = c(0, 1), 
                        direction = "<", 
                        quiet = TRUE)
    auc <- as.numeric(roc_obj$auc)
  }, error = function(e) {
    warning("Could not calculate AUC: ", e$message)
  })
  
  list(
    tp = tp, tn = tn, fp = fp, fn = fn,
    accuracy = accuracy, 
    sensitivity = sensitivity,
    specificity = specificity, 
    precision = precision,
    f1 = f1, 
    auc = auc
  )
}

cat("Metrics function ready\n\n")

# Training Callbacks ----------------------------------------------------------
cat("Configuring training callbacks...\n")

# Early stopping callback (following Jihad's configuration)
early_stopping <- callback_early_stopping(
  monitor = "val_loss",
  patience = EARLY_STOP_PATIENCE,
  restore_best_weights = TRUE,
  verbose = 1
)

# Learning rate reduction callback (following Jihad's configuration)
reduce_lr <- callback_reduce_lr_on_plateau(
  monitor = "val_loss",
  factor = 0.5,
  patience = 1,
  min_lr = 0.0001,
  verbose = 0
)

cat("  Early stopping: patience =", EARLY_STOP_PATIENCE, "\n")
cat("  LR reduction: factor = 0.5, patience = 1\n\n")

# Training Loop ---------------------------------------------------------------
cat(paste0(strrep("=", 80), "\n"))
cat("Training CNN Models - ", N_CYCLES, " Cycles for Statistical Robustness\n")
cat(paste0(strrep("=", 80), "\n\n"))

# Initialize storage for results (following Jihad's approach)
roc_df_rows <- data.frame(matrix(y_test, nrow = 1))  # First row = test labels
metrics_df <- NULL  # Will accumulate metrics from each cycle
all_histories <- vector("list", N_CYCLES)

tic("Total training time")

# Main training loop (following Jihad's exact approach)
for (cycle in seq_len(N_CYCLES)) {
  cat(paste0(strrep("=", 80), "\n"))
  cat(sprintf("CYCLE %d/%d\n", cycle, N_CYCLES))
  cat(paste0(strrep("=", 80), "\n\n"))
  
  cycle_start <- Sys.time()
  
  # Generate model identifiers
  model_id <- sprintf("CNNr%02d", cycle)
  model_file <- file.path(MODEL_DIR, paste0("model_", model_id, ".h5"))
  history_file <- file.path(MODEL_DIR, paste0("history_", model_id, ".rds"))
  
  # Build fresh model for this cycle
  cat("Building model...\n")
  model <- build_cnn_model(
    vocab_size, 
    maxlen,
    EMBEDDING_DIM, 
    NUM_FILTERS,
    FILTER_SIZES, 
    HIDDEN_DIMS, 
    DROPOUT_RATE
  )
  
  # Train model
  cat("Training...\n")
  history <- model %>% fit(
    x = x_train, 
    y = y_train,
    batch_size = BATCH_SIZE,
    epochs = EPOCHS,
    validation_split = VALIDATION_SPLIT,
    callbacks = list(early_stopping, reduce_lr),
    verbose = 0  # Silent training (following Jihad's approach)
  )
  
  # Calculate training time and actual epochs
  cycle_end <- Sys.time()
  t_elapsed <- as.numeric(difftime(cycle_end, cycle_start, units = "secs"))
  actual_epochs <- length(history$metrics$loss)
  
  cat("\nTraining complete:\n")
  cat("  Epochs trained:", actual_epochs, "\n")
  cat("  Time elapsed:  ", round(t_elapsed, 1), "seconds\n")
  cat("  Final val_loss:", round(tail(history$metrics$val_loss, 1), 4), "\n")
  cat("  Final val_acc: ", round(tail(history$metrics$val_accuracy, 1), 4), "\n\n")
  
  # Generate predictions on test set
  cat("Predicting on test set...\n")
  pred <- model %>% predict(x_test, verbose = 0)
  pred_prob <- pred[, 1]  # Extract probability column
  
  # Store predictions (following Jihad's approach)
  roc_df_rows <- rbind(roc_df_rows, pred_prob)
  
  # Calculate metrics
  cat("Calculating metrics...\n")
  mets <- calculate_metrics(y_test, pred_prob)
  
  # Create metrics row
  metrics_row <- data.frame(
    model = "CNNr",
    i = cycle,
    epochs = actual_epochs,
    time_secs = t_elapsed,
    auc = mets$auc,
    accuracy = mets$accuracy,
    sensitivity = mets$sensitivity,
    specificity = mets$specificity,
    precision = mets$precision,
    F1 = mets$f1,
    tp = mets$tp,
    tn = mets$tn,
    fp = mets$fp,
    fn = mets$fn,
    stringsAsFactors = FALSE
  )
  
  # Accumulate metrics (following Jihad's approach)
  if (is.null(metrics_df)) {
    metrics_df <- metrics_row
  } else {
    metrics_df <- rbind(metrics_df, metrics_row)
  }
  
  cat("Test Performance:\n")
  cat("  AUC:", sprintf("%.4f", mets$auc), "\n")
  cat("  Accuracy:", sprintf("%.4f", mets$accuracy), "\n")
  cat("  F1 Score:", sprintf("%.4f", mets$f1), "\n\n")
  
  # Save model and history
  cat("Saving model and history...\n")
  model %>% save_model_hdf5(model_file)
  saveRDS(history, history_file)
  cat("  Model saved:  ", model_file, "\n")
  cat("  History saved:", history_file, "\n\n")
  
  # Store history
  all_histories[[cycle]] <- history
  
  # Clean up
  rm(model)
  gc()
  
  cat("Cycle", cycle, "complete\n")
  cat(strrep("-", 80), "\n\n")
}

toc()

# Save Results ----------------------------------------------------------------
cat(paste0(strrep("=", 80), "\n"))
cat("Saving Training Results\n")
cat(paste0(strrep("=", 80), "\n\n"))

# Save metrics (NO DATE SUFFIX)
metrics_file <- file.path(RESULTS_DIR, "metrics_df_CNNr.rds")
saveRDS(metrics_df, metrics_file)
cat("Metrics saved:", metrics_file, "\n")

# Save predictions matrix (NO DATE SUFFIX)
roc_file <- file.path(RESULTS_DIR, "roc_df_rows_CNNr.rds")
saveRDS(roc_df_rows, roc_file)
cat("Predictions saved:", roc_file, "\n")

# Save test labels
test_labels_file <- file.path(RESULTS_DIR, "test_labels.rds")
saveRDS(y_test, test_labels_file)
cat("Test labels saved:", test_labels_file, "\n")

# Save all histories
histories_file <- file.path(RESULTS_DIR, "training_histories.rds")
saveRDS(all_histories, histories_file)
cat("Training histories saved:", histories_file, "\n\n")

# Summary Statistics ----------------------------------------------------------
cat(paste0(strrep("=", 80), "\n"))
cat("Summary Statistics Across All Cycles\n")
cat(paste0(strrep("=", 80), "\n\n"))

# Calculate summary statistics
summary_stats <- data.frame(
  Metric = c("AUC", "Accuracy", "Sensitivity", "Specificity", "F1"),
  Mean = c(
    mean(metrics_df$auc, na.rm = TRUE),
    mean(metrics_df$accuracy, na.rm = TRUE),
    mean(metrics_df$sensitivity, na.rm = TRUE),
    mean(metrics_df$specificity, na.rm = TRUE),
    mean(metrics_df$F1, na.rm = TRUE)
  ),
  SD = c(
    sd(metrics_df$auc, na.rm = TRUE),
    sd(metrics_df$accuracy, na.rm = TRUE),
    sd(metrics_df$sensitivity, na.rm = TRUE),
    sd(metrics_df$specificity, na.rm = TRUE),
    sd(metrics_df$F1, na.rm = TRUE)
  ),
  Min = c(
    min(metrics_df$auc, na.rm = TRUE),
    min(metrics_df$accuracy, na.rm = TRUE),
    min(metrics_df$sensitivity, na.rm = TRUE),
    min(metrics_df$specificity, na.rm = TRUE),
    min(metrics_df$F1, na.rm = TRUE)
  ),
  Max = c(
    max(metrics_df$auc, na.rm = TRUE),
    max(metrics_df$accuracy, na.rm = TRUE),
    max(metrics_df$sensitivity, na.rm = TRUE),
    max(metrics_df$specificity, na.rm = TRUE),
    max(metrics_df$F1, na.rm = TRUE)
  ),
  stringsAsFactors = FALSE
)

print(summary_stats)
cat("\n")

# Select best model (following Jihad's median AUC approach)
median_auc <- median(metrics_df$auc, na.rm = TRUE)
best_row <- metrics_df %>%
  mutate(auc_diff = abs(auc - median_auc)) %>%
  filter(auc_diff == min(auc_diff)) %>%
  filter(F1 == max(F1)) %>%
  slice(1)

best_cycle <- best_row$i
best_auc <- best_row$auc

cat("Best Model Selection (Median AUC with Max F1):\n")
cat(strrep("-", 40), "\n")
cat("  Median AUC:", sprintf("%.4f", median_auc), "\n")
cat("  Selected cycle:", best_cycle, "\n")
cat("  Selected AUC:", sprintf("%.4f", best_auc), "\n")
cat("  Selected F1:", sprintf("%.4f", best_row$F1), "\n\n")

# Save best model info
best_info <- list(
  best_cycle = best_cycle,
  median_auc = median_auc,
  best_model_file = file.path(MODEL_DIR, sprintf("model_CNNr%02d.h5", best_cycle)),
  best_metrics = best_row
)
saveRDS(best_info, file.path(RESULTS_DIR, "best_model_info.rds"))
cat("Best model info saved\n\n")

# Create Training Report ------------------------------------------------------
cat(paste0(strrep("=", 80), "\n"))
cat("Creating Training Report\n")
cat(paste0(strrep("=", 80), "\n\n"))

report_file <- file.path(RESULTS_DIR, "training_report.txt")
sink(report_file)

cat("ADRD CNN Training Report - CNNr (Random Embeddings)\n")
cat(strrep("=", 80), "\n\n")
cat("Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

cat("Configuration:\n")
cat(strrep("-", 40), "\n")
cat("Vocabulary size:", MAX_VOCAB, "\n")
cat("Embedding dimension:", EMBEDDING_DIM, "\n")
cat("Filters per kernel:", NUM_FILTERS, "\n")
cat("Kernel sizes:", paste(FILTER_SIZES, collapse = ", "), "\n")
cat("Hidden units:", HIDDEN_DIMS, "\n")
cat("Dropout rate:", DROPOUT_RATE, "\n")
cat("Training cycles:", N_CYCLES, "\n")
cat("Epochs per cycle:", EPOCHS, "\n")
cat("Batch size:", BATCH_SIZE, "\n")
cat("Learning rate:", LEARNING_RATE, "\n\n")

cat("Data:\n")
cat(strrep("-", 40), "\n")
cat("Training samples:", nrow(train_set), "\n")
cat("Test samples:", nrow(test_set), "\n")
cat("Actual vocab size:", vocab_size, "\n")
cat("Maximum sequence length:", maxlen, "\n\n")

cat("Training Results:\n")
cat(strrep("-", 40), "\n")
cat("Mean AUC:", sprintf("%.4f ± %.4f", 
    mean(metrics_df$auc), sd(metrics_df$auc)), "\n")
cat("Mean Accuracy:", sprintf("%.4f ± %.4f", 
    mean(metrics_df$accuracy), sd(metrics_df$accuracy)), "\n")
cat("Mean Sensitivity:", sprintf("%.4f ± %.4f", 
    mean(metrics_df$sensitivity), sd(metrics_df$sensitivity)), "\n")
cat("Mean Specificity:", sprintf("%.4f ± %.4f", 
    mean(metrics_df$specificity), sd(metrics_df$specificity)), "\n")
cat("Mean F1 Score:", sprintf("%.4f ± %.4f", 
    mean(metrics_df$F1), sd(metrics_df$F1)), "\n\n")

cat("Best Model:\n")
cat(strrep("-", 40), "\n")
cat("Selection method: Median AUC with Max F1\n")
cat("Median AUC:", sprintf("%.4f", median_auc), "\n")
cat("Selected cycle:", best_cycle, "\n")
cat("Selected AUC:", sprintf("%.4f", best_auc), "\n")
cat("Selected F1:", sprintf("%.4f", best_row$F1), "\n\n")

cat("Per-Cycle Results:\n")
cat(strrep("-", 40), "\n")
print(metrics_df)

sink()
cat("Report saved to:", report_file, "\n\n")

# Final Summary ---------------------------------------------------------------
cat(paste0(strrep("=", 80), "\n"))
cat("TRAINING COMPLETE\n")
cat(paste0(strrep("=", 80), "\n\n"))

cat("Summary:\n")
cat("  Trained", N_CYCLES, "CNN models\n")
cat("  Mean AUC:", sprintf("%.4f ± %.4f", 
    mean(metrics_df$auc), sd(metrics_df$auc)), "\n")
cat("  Best model: Cycle", best_cycle, 
    sprintf("(AUC: %.4f)\n", best_auc))
cat("\n")

cat("Critical Inference Artifacts Saved:\n")
cat("  Tokenizer:  ", file.path(MODEL_DIR, "tokenizer_cnnr"), "\n")
cat("  Word index: ", file.path(MODEL_DIR, "word_index.rds"), "\n")
cat("  Vocab size: ", file.path(MODEL_DIR, "vocab_size.rds"), "\n")
cat("  Maxlen:     ", file.path(MODEL_DIR, "maxlen.rds"), "\n\n")

cat("Models saved in:", MODEL_DIR, "/\n")
cat("Results saved in:", RESULTS_DIR, "/\n\n")

cat("Next Steps:\n")
cat("  1. Run 03_evaluate_models.R for comprehensive evaluation\n")
cat("  2. Review training report:", report_file, "\n")
cat("  3. Examine training histories for convergence\n\n")

cat(paste0(strrep("=", 80), "\n"))
cat("CNN training completed successfully!\n")