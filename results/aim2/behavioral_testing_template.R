# Behavioral Testing Script Template
# ==============================================================================
# Purpose: Systematically test model sensitivity to specific terms
#
# Usage:
#   1. Select terms from behavioral_test_terms.rds
#   2. For each term, create modified versions of test cases
#   3. Compare predictions on original vs modified texts
#   4. Measure sensitivity (Δ prediction)
# ==============================================================================

# Load required libraries
suppressPackageStartupMessages({
  library(tidyverse)  # For data manipulation (filter, %>%, etc.)
  library(keras)      # For model prediction functions
})

# Load terms
behavioral_test_terms <- readRDS("results/aim2/behavioral_test_terms.rds")

# Load model and artifacts
source("utils_model_loader.R")
artifacts <- load_all_artifacts("models")
best_cycle <- 1  # Update with your best model cycle
model <- load_model_auto(best_cycle, "models")

# Load test set
test_set <- readRDS("data/test_set.rds")

# Function to test term removal
test_term_removal <- function(de_id, term, model, artifacts, test_set) {
  # Get original text
  original_text <- test_set %>%
    filter(DE_ID == de_id) %>%
    pull(txt)

  # Check if term present
  term_present <- grepl(paste0("\\b", term, "\\b"), original_text, ignore.case = TRUE)

  if (!term_present) {
    return(list(term_present = FALSE, pred_change = NA))
  }

  # Remove term
  modified_text <- gsub(paste0("\\b", term, "\\b"), "", original_text, ignore.case = TRUE)
  modified_text <- gsub("\\s+", " ", trimws(modified_text))

  # Get predictions
  seq_orig <- texts_to_sequences(artifacts$tokenizer, original_text)
  seq_mod <- texts_to_sequences(artifacts$tokenizer, modified_text)

  pad_orig <- pad_sequences(seq_orig, maxlen = artifacts$maxlen, padding = "pre")
  pad_mod <- pad_sequences(seq_mod, maxlen = artifacts$maxlen, padding = "pre")

  pred_orig <- model %>% predict(pad_orig, verbose = 0)
  pred_mod <- model %>% predict(pad_mod, verbose = 0)

  pred_change <- pred_mod[1, 1] - pred_orig[1, 1]

  return(list(
    term_present = TRUE,
    pred_orig = pred_orig[1, 1],
    pred_mod = pred_mod[1, 1],
    pred_change = pred_change
  ))
}

# Run behavioral tests
# Example: Test all ADRD terms on all ADRD cases
cat("\n")
cat("================================================================================\n")
cat("Running Behavioral Tests\n")
cat("================================================================================\n\n")

results <- data.frame()

# Get ADRD cases
adrd_cases <- test_set %>% filter(label == 1)
n_cases_to_test <- min(50, nrow(adrd_cases))

cat("Testing", length(behavioral_test_terms$adrd), "ADRD terms on", n_cases_to_test, "ADRD cases\n\n")

for (i in seq_along(behavioral_test_terms$adrd)) {
  term <- behavioral_test_terms$adrd[i]
  cat("Testing term", i, "of", length(behavioral_test_terms$adrd), ":", term, "\n")

  term_results <- 0

  for (de_id in adrd_cases$DE_ID[1:n_cases_to_test]) {
    result <- test_term_removal(de_id, term, model, artifacts, test_set)

    if (result$term_present) {
      results <- rbind(results, data.frame(
        DE_ID = de_id,
        term = term,
        pred_orig = result$pred_orig,
        pred_mod = result$pred_mod,
        pred_change = result$pred_change,
        stringsAsFactors = FALSE
      ))
      term_results <- term_results + 1
    }
  }

  cat("  Found term in", term_results, "cases\n")
}

# Analyze and save results
cat("\n")
cat("================================================================================\n")
cat("Results Summary\n")
cat("================================================================================\n\n")

if (nrow(results) > 0) {
  cat("Total tests with term present:", nrow(results), "\n")
  cat("Prediction change summary:\n")
  print(summary(results$pred_change))
  cat("\n")

  # Aggregate by term
  term_summary <- results %>%
    group_by(term) %>%
    summarize(
      n_cases = n(),
      mean_change = mean(pred_change),
      sd_change = sd(pred_change),
      median_change = median(pred_change),
      .groups = "drop"
    ) %>%
    arrange(desc(abs(mean_change)))

  cat("\nMean prediction change by term:\n")
  print(term_summary)

  # Save results
  write_csv(results, "results/aim2/behavioral_test_results.csv")
  write_csv(term_summary, "results/aim2/behavioral_test_summary.csv")
  cat("\nResults saved to results/aim2/behavioral_test_results.csv\n")
  cat("Summary saved to results/aim2/behavioral_test_summary.csv\n")
} else {
  cat("No results - terms not found in any test cases\n")
}

