#!/usr/bin/env Rscript
# ==============================================================================
# ADRD ePhenotyping Pipeline - Data Preparation [REQUIRED FOR EVALUATION]
# ==============================================================================
# Version: 2.1
# Author: Gyasi, Frederick
# Project: adrd_ephenotyping
#
# ⚠️  UPDATED APPROACH: Full dataset used for evaluation (no train/test split)
# ⚠️  Uses pre-trained models from Jihad Obeid - NO NEW TRAINING
#
# Purpose: Load, validate, and prepare FULL dataset for CNN evaluation
#
# Input:  data/raw/ptHx_sample_v2025-11-24.csv
# Output: data/test_set.rds (full dataset), data/train_set.rds (small reference)
#
# Key Operations:
# 1. Load preprocessed clinical notes with STRING labels ('ADRD', 'NON-ADRD')
# 2. Convert string labels to numeric (ADRD→1, NON-ADRD→0)
# 3. Rename 'content' column to 'txt' for downstream compatibility
# 4. Use ENTIRE dataset as test set for evaluation
# 5. Create small train_set.rds reference file for compatibility
# 6. Validate data quality and class balance
# ==============================================================================

# Load Libraries --------------------------------------------------------------
cat(strrep("=", 80), "\n", sep = "")
cat("ADRD ePhenotyping - Data Preparation\n")
cat(strrep("=", 80), "\n\n")

cat("Loading required libraries...\n")
suppressPackageStartupMessages({
  library(tidyverse)   # Data manipulation
  library(magrittr)    # Pipe operators (%>% and %+%)
  library(rsample)     # Stratified splitting
})

# Define %+% (paste0) if magrittr didn't export it
`%+%` <- function(a, b) paste0(a, b)

options(dplyr.summarise.inform = FALSE)

# Configuration ---------------------------------------------------------------
cat("\nConfiguration:\n")
cat(strrep("-", 80), "\n", sep = "")

RAW_DATA_DIR <- "data/raw"
OUTPUT_DIR   <- "data"
INPUT_FILE   <- "ptHx_sample_v2025-11-24.csv"  # UPDATED filename

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# NO TRAIN/TEST SPLIT - Using full dataset for evaluation
RANDOM_SEED      <- 42

# Required columns in input CSV
REQUIRED_COLS    <- c("DE_ID", "content", "Label")
# Available demographic columns
DEMOGRAPHIC_COLS <- c("GENDER", "RACE", "HISPANIC")

cat("  Input file:", INPUT_FILE, "\n")
cat("  Approach: FULL DATASET for evaluation (no train/test split)\n")
cat("  Using pre-trained CNN models from Jihad Obeid\n")
cat("  Random seed:", RANDOM_SEED, "\n")
cat("  Column transformations:\n")
cat("    - 'content' to 'txt' (for downstream pipeline)\n")
cat("    - 'Label' (string) to 'label' (numeric: 'ADRD' to 1, 'NON-ADRD' to 0)\n\n")

# Load Data -------------------------------------------------------------------
cat(strrep("=", 80), "\n", sep = "")
cat("Loading Data\n")
cat(strrep("=", 80), "\n\n")

input_path <- file.path(RAW_DATA_DIR, INPUT_FILE)
if (!file.exists(input_path)) {
  stop("Input file not found: ", input_path,
       "\nPlease place ", INPUT_FILE, " in ", RAW_DATA_DIR)
}

cat("Reading CSV file...\n")
raw_data <- read_csv(input_path, show_col_types = FALSE)

cat("  Loaded:", nrow(raw_data), "rows\n")
cat("  Columns:", paste(names(raw_data), collapse = ", "), "\n\n")

# Validate Required Columns ---------------------------------------------------
cat("Validating required columns...\n")
missing_cols <- setdiff(REQUIRED_COLS, names(raw_data))
if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "),
       "\nRequired: ", paste(REQUIRED_COLS, collapse = ", "))
}
cat("  All required columns present\n\n")

# Inspect Label Column Format -------------------------------------------------
cat("Inspecting Label column format...\n")
label_preview <- head(unique(raw_data$Label), 10)
cat("  Unique Label values:", paste(label_preview, collapse = ", "), "\n")
cat("  Label data type:", class(raw_data$Label), "\n")

if (is.character(raw_data$Label) || is.factor(raw_data$Label)) {
  cat("  Labels are string/factor format (as expected)\n")
  expected_labels <- c("ADRD", "NON-ADRD")
  actual_labels <- unique(na.omit(raw_data$Label))
  unexpected <- setdiff(actual_labels, expected_labels)
  if (length(unexpected) > 0) {
    cat("  WARNING: Unexpected label values found:", 
        paste(unexpected, collapse = ", "), "\n")
  }
} else if (is.numeric(raw_data$Label)) {
  cat("  Labels are already numeric (will use as-is)\n")
} else {
  cat("  WARNING: Unknown label format\n")
}
cat("\n")

# Rename content to txt ---------------------------------------------------------
cat("Renaming 'content' to 'txt'...\n")
raw_data <- raw_data %>% rename(txt = content)
cat("  Column renamed: content to txt\n")
cat("    (Required for compatibility with scripts 02-05)\n\n")

# Demographic Columns ---------------------------------------------------------
available_demos <- intersect(DEMOGRAPHIC_COLS, names(raw_data))
cat("Available demographic columns:\n")
if (length(available_demos) > 0) {
  for (demo_col in available_demos) {
    unique_vals <- unique(na.omit(raw_data[[demo_col]]))
    n_unique <- length(unique_vals)
    cat("  - ", demo_col, ": ", n_unique, " unique values\n", sep = "")
    if (n_unique <= 10) {
      cat("      Values: ", paste(head(unique_vals, 10), collapse = ", "), "\n", sep = "")
    }
  }
} else {
  cat("  WARNING: No demographic columns found - stage 4 analysis will be limited\n")
}
cat("\n")

# Data Quality Checks ---------------------------------------------------------
cat(strrep("=", 80), "\n", sep = "")
cat("Data Quality Assessment\n")
cat(strrep("=", 80), "\n\n")

# 1. Check for Duplicate Patients
cat("Checking for duplicate patients...\n")
n_dup <- sum(duplicated(raw_data$DE_ID))
if (n_dup > 0) {
  cat("  WARNING:", n_dup, "duplicate DE_IDs found\n")
  cat("    Action: Keeping first occurrence of each patient\n")
  raw_data <- raw_data %>% 
    arrange(DE_ID) %>% 
    distinct(DE_ID, .keep_all = TRUE)
  cat("    Result:", nrow(raw_data), "unique patients\n")
} else {
  cat("  No duplicates found\n")
}
cat("\n")

# 2. Check Text Completeness
cat("Checking text completeness...\n")
miss_txt <- sum(is.na(raw_data$txt) | raw_data$txt == "")
miss_pct <- miss_txt / nrow(raw_data) * 100
cat("  Missing/empty txt:", miss_txt, sprintf("(%.2f%%)\n", miss_pct))

if (miss_pct > 5) {
  cat("  WARNING: >5% missing text - consider data quality review\n")
} else if (miss_pct > 0) {
  cat("  Some records will be excluded from analysis\n")
} else {
  cat("  All records have text content\n")
}
cat("\n")

# 3. Calculate Text Length Statistics
cat("Calculating text length statistics...\n")
raw_data <- raw_data %>% mutate(text_len = nchar(txt))

txt_stats <- raw_data %>%
  filter(!is.na(txt), txt != "") %>%
  summarise(
    Min    = min(text_len, na.rm = TRUE),
    Q1     = quantile(text_len, 0.25, na.rm = TRUE),
    Median = median(text_len, na.rm = TRUE),
    Mean   = mean(text_len, na.rm = TRUE),
    Q3     = quantile(text_len, 0.75, na.rm = TRUE),
    Max    = max(text_len, na.rm = TRUE)
  )

cat("  Text length (characters):\n")
print(txt_stats, width = 80)
cat("\n  Approximate word count (chars/5):\n")
cat("    Mean:  ", round(txt_stats$Mean / 5, 0), " words\n", sep = "")
cat("    Median:", round(txt_stats$Median / 5, 0), " words\n", sep = "")

if (txt_stats$Mean < 1000) {
  cat("  WARNING: Short average text (<200 words) - may limit model performance\n")
}
cat("\n")

# Remove temporary text_len column
raw_data <- raw_data %>% select(-text_len)

# 4. Label Distribution
cat("Checking label distribution...\n")

if (is.character(raw_data$Label) || is.factor(raw_data$Label)) {
  cat("  Label format: String/factor\n")
  label_tbl <- table(raw_data$Label, useNA = "ifany")
  print(label_tbl)
  cat("\n")
  
  if ("ADRD" %in% names(label_tbl)) {
    adrd_pct <- label_tbl["ADRD"] / sum(label_tbl) * 100
    cat("  ADRD prevalence:", sprintf("%.2f%%", adrd_pct), "\n")
  } else {
    stop("ERROR: 'ADRD' label not found in data")
  }
  
} else if (is.numeric(raw_data$Label)) {
  cat("  Label format: Numeric\n")
  label_tbl <- table(raw_data$Label, useNA = "ifany")
  print(label_tbl)
  cat("\n")
  
  if (1 %in% names(label_tbl)) {
    adrd_pct <- prop.table(label_tbl)["1"] * 100
  } else if ("1" %in% names(label_tbl)) {
    adrd_pct <- prop.table(label_tbl)["1"] * 100
  } else {
    stop("ERROR: Label value '1' not found in data")
  }
  cat("  ADRD prevalence:", sprintf("%.2f%%", adrd_pct), "\n")
}

# Assess class balance
if (adrd_pct < 10) {
  cat("  WARNING: Low ADRD prevalence (<10%) - consider class weights in training\n")
} else if (adrd_pct > 50) {
  cat("  WARNING: High ADRD prevalence (>50%) - verify population representativeness\n")
} else if (adrd_pct >= 15 && adrd_pct <= 40) {
  cat("  Good class balance for binary classification\n")
}
cat("\n")

# 5. Demographic Distribution
if (length(available_demos) > 0) {
  cat("Demographic distribution summary:\n")
  for (demo_col in available_demos) {
    cat("\n", demo_col, ":\n", sep = "")
    demo_tbl <- raw_data %>%
      filter(!is.na(!!sym(demo_col)), !!sym(demo_col) != "") %>%
      count(!!sym(demo_col), name = "N") %>%
      arrange(desc(N)) %>%
      mutate(Percent = sprintf("%.1f%%", N / sum(N) * 100))
    
    print(demo_tbl, n = 15)
    
    small_groups <- demo_tbl %>% filter(N < 10)
    if (nrow(small_groups) > 0) {
      cat("  WARNING: ", nrow(small_groups), " groups with <10 patients\n", sep = "")
    }
  }
  cat("\n")
}

# Clean Dataset ---------------------------------------------------------------
cat(strrep("=", 80), "\n", sep = "")
cat("Preparing Clean Dataset\n")
cat(strrep("=", 80), "\n\n")

# Filter out invalid records
if (is.character(raw_data$Label) || is.factor(raw_data$Label)) {
  clean_data <- raw_data %>%
    filter(
      !is.na(txt), txt != "",
      Label %in% c("ADRD", "NON-ADRD")
    )
} else {
  clean_data <- raw_data %>%
    filter(
      !is.na(txt), txt != "",
      Label %in% c(0, 1)
    )
}

n_excl <- nrow(raw_data) - nrow(clean_data)
cat("Excluded records (missing/invalid data):", n_excl, "\n")
cat("Clean dataset size:", nrow(clean_data), "patients\n\n")

if (nrow(clean_data) == 0) {
  stop("ERROR: No valid records remaining after filtering!")
}

# Convert Labels to Numeric ---------------------------------------------------
cat("Creating numeric 'label' column...\n")

if (is.character(clean_data$Label) || is.factor(clean_data$Label)) {
  cat("  Converting string labels to numeric:\n")
  cat("    'ADRD'     to 1 (positive class)\n")
  cat("    'NON-ADRD' to 0 (negative class)\n")
  
  clean_data <- clean_data %>%
    mutate(
      label = case_when(
        Label == "ADRD"     ~ 1L,
        Label == "NON-ADRD" ~ 0L,
        TRUE ~ NA_integer_
      )
    )
  
  n_na_labels <- sum(is.na(clean_data$label))
  if (n_na_labels > 0) {
    cat("  WARNING:", n_na_labels, "labels could not be converted\n")
    clean_data <- clean_data %>% filter(!is.na(label))
  }
  
} else if (is.numeric(clean_data$Label)) {
  cat("  Labels already numeric - copying to 'label' column\n")
  clean_data <- clean_data %>%
    mutate(label = as.integer(Label))
}

# Verify conversion
cat("\nLabel conversion verification:\n")
cat("  Numeric label distribution:\n")
label_numeric_tbl <- table(clean_data$label, useNA = "ifany")
print(label_numeric_tbl)

if ("Label" %in% names(clean_data)) {
  cat("\n  Cross-tabulation (original vs numeric):\n")
  print(table(Original = clean_data$Label, Numeric = clean_data$label))
}

cat("\n  Label conversion complete\n")
cat("    Final dataset: ", sum(clean_data$label == 1), " ADRD, ",
    sum(clean_data$label == 0), " NON-ADRD\n\n", sep = "")

# Dataset Assignment ---------------------------------------------------------
cat(strrep("=", 80), "\n", sep = "")
cat("Dataset Assignment for Evaluation\n")
cat(strrep("=", 80), "\n\n")

set.seed(RANDOM_SEED)

cat("APPROACH: Full dataset used for evaluation\n")
cat("  - Using pre-trained CNN models (Jihad Obeid)\n")
cat("  - NO new training required\n")
cat("  - ALL data assigned to test set\n")
cat("  - Minimal train_set created for pipeline compatibility\n\n")

# Use ENTIRE dataset as test set
test_data <- clean_data

# Create minimal train set (10 samples) for pipeline compatibility only
# Some scripts may reference train_set.rds, but it won't be used
cat("Creating datasets...\n")
train_data <- clean_data %>%
  group_by(label) %>%
  slice_head(n = 5) %>%  # 5 ADRD + 5 NON-ADRD
  ungroup()

cat("Dataset assignments:\n")
cat("  Test set (FULL DATASET):", nrow(test_data), "patients\n")
cat("  Train set (reference):   ", nrow(train_data), "patients (NOT USED)\n")
cat("  Note: Train set is minimal reference only - evaluation uses test set\n\n")

# Verify Dataset Quality -----------------------------------------------------
cat(strrep("=", 80), "\n", sep = "")
cat("Verifying Test Dataset Quality\n")
cat(strrep("=", 80), "\n\n")

# 1. Check test set label distribution
cat("Label distribution in test set (FULL DATASET):\n")
test_adrd_pct  <- mean(test_data$label) * 100
cat("  ADRD:     ", sprintf("%.2f%%", test_adrd_pct),
    sprintf("(%d/%d)\n", sum(test_data$label), nrow(test_data)))
cat("  NON-ADRD: ", sprintf("%.2f%%", 100 - test_adrd_pct),
    sprintf("(%d/%d)\n\n", sum(test_data$label == 0), nrow(test_data)))

# 2. Check demographic distribution in test set
if (length(available_demos) > 0) {
  cat("Demographic distribution in test set:\n")
  for (demo_col in available_demos) {
    cat("\n", demo_col, ":\n", sep = "")

    demo_tbl <- test_data %>%
      filter(!is.na(!!sym(demo_col)), !!sym(demo_col) != "") %>%
      count(!!sym(demo_col), name = "N") %>%
      mutate(Percent = sprintf("%.1f%%", N / sum(N) * 100)) %>%
      arrange(desc(N))

    print(demo_tbl, n = 15)

    # Check for adequate subgroup sizes
    small_groups <- demo_tbl %>% filter(N < 20)
    if (nrow(small_groups) > 0) {
      cat("  Note: ", nrow(small_groups),
          " groups have <20 patients (may limit stratified analysis)\n", sep = "")
    }
  }
  cat("\n")
}

# Save Dataset Metadata ------------------------------------------------------
cat(strrep("=", 80), "\n", sep = "")
cat("Saving Dataset Information\n")
cat(strrep("=", 80), "\n\n")

# Create dataset metadata (all data is test set)
dataset_info <- test_data %>%
  select(DE_ID, Label, label, all_of(available_demos)) %>%
  mutate(partition = "test")  # All data is test

dataset_info_file <- file.path(OUTPUT_DIR, "split_info.rds")
saveRDS(dataset_info, dataset_info_file)
cat("Dataset metadata saved:", dataset_info_file, "\n")
cat("  Contains: patient IDs, labels, demographics\n")
cat("  Partition: ALL data assigned to 'test' (full dataset evaluation)\n\n")

# Save Datasets ---------------------------------------------------------------
train_file <- file.path(OUTPUT_DIR, "train_set.rds")
test_file  <- file.path(OUTPUT_DIR, "test_set.rds")

saveRDS(train_data, train_file)
saveRDS(test_data,  test_file)

cat("Training dataset saved:", train_file, "\n")
cat("  Rows:", nrow(train_data), "\n")
cat("  Columns:", paste(names(train_data), collapse = ", "), "\n\n")

cat("Test dataset saved:", test_file, "\n")
cat("  Rows:", nrow(test_data), "\n")
cat("  Columns:", paste(names(test_data), collapse = ", "), "\n\n")

# Generate Summary Report -----------------------------------------------------
cat(strrep("=", 80), "\n", sep = "")
cat("Generating Summary Report\n")
cat(strrep("=", 80), "\n\n")

summary_file <- file.path(OUTPUT_DIR, "data_summary.txt")
sink(summary_file)

cat("ADRD ePhenotyping - Data Preparation Summary\n")
cat(strrep("=", 80), "\n\n")
cat("Generated:", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"), "\n\n")

cat("INPUT DATA\n")
cat(strrep("-", 40), "\n")
cat("File:              ", INPUT_FILE, "\n")
cat("Original records:  ", nrow(raw_data), "\n")
cat("Excluded records:  ", n_excl, "\n")
cat("Clean records:     ", nrow(clean_data), "\n\n")

cat("LABEL CONVERSION\n")
cat(strrep("-", 40), "\n")
cat("Original format:   ", class(raw_data$Label)[1], "\n")
cat("Conversion:        'ADRD' to 1, 'NON-ADRD' to 0\n")
cat("ADRD cases:        ", sum(clean_data$label == 1), 
    sprintf(" (%.2f%%)\n", mean(clean_data$label) * 100))
cat("Control cases:     ", sum(clean_data$label == 0), 
    sprintf(" (%.2f%%)\n", mean(clean_data$label == 0) * 100), "\n")

cat("COLUMN TRANSFORMATIONS\n")
cat(strrep("-", 40), "\n")
cat("'content' to 'txt'      (for pipeline compatibility)\n")
cat("'Label'   to 'label'    (string to numeric)\n\n")

cat("TRAIN/TEST SPLIT\n")
cat(strrep("-", 40), "\n")
cat("Random seed:       ", RANDOM_SEED, "\n")
cat("Target ratio:      ", sprintf("%.2f / %.2f", TRAIN_PROPORTION, 1 - TRAIN_PROPORTION), "\n")
cat("Actual ratio:      ", sprintf("%.4f / %.4f", actual_prop, 1 - actual_prop), "\n")
cat("Stratified by:     ", paste(strat_vars, collapse = ", "), "\n\n")

cat("TRAINING SET\n")
cat(strrep("-", 40), "\n")
cat("Total patients:    ", nrow(train_data), "\n")
cat("ADRD cases:        ", sum(train_data$label == 1), 
    sprintf(" (%.2f%%)\n", mean(train_data$label) * 100))
cat("Control cases:     ", sum(train_data$label == 0), 
    sprintf(" (%.2f%%)\n", mean(train_data$label == 0) * 100), "\n")

cat("TEST SET\n")
cat(strrep("-", 40), "\n")
cat("Total patients:    ", nrow(test_data), "\n")
cat("ADRD cases:        ", sum(test_data$label == 1), 
    sprintf(" (%.2f%%)\n", mean(test_data$label) * 100))
cat("Control cases:     ", sum(test_data$label == 0), 
    sprintf(" (%.2f%%)\n", mean(test_data$label == 0) * 100), "\n")

cat("QUALITY CHECKS\n")
cat(strrep("-", 40), "\n")
cat("Patient overlap:   ", length(overlap), " (should be 0)\n")
cat("Label balance diff:", sprintf("%.2f percentage points\n", prop_diff))
cat("Status:            ", ifelse(length(overlap) == 0 && prop_diff < 5, 
                                 "PASSED", "CHECK WARNINGS"), "\n\n")

cat("OUTPUT FILES\n")
cat(strrep("-", 40), "\n")
cat("Training set:      ", train_file, "\n")
cat("Test set:          ", test_file, "\n")
cat("Split metadata:    ", split_info_file, "\n")
cat("This report:       ", summary_file, "\n\n")

cat("NEXT STEPS\n")
cat(strrep("-", 40), "\n")
cat("1. Review this summary report\n")
cat("2. Verify label distributions are acceptable\n")
cat("3. Run 02_train_cnnr.R to train CNN models\n")

sink()

cat("Summary report saved:", summary_file, "\n\n")

# Final Console Summary -------------------------------------------------------
cat(strrep("=", 80), "\n", sep = "")
cat("DATA PREPARATION COMPLETE\n")
cat(strrep("=", 80), "\n\n")

cat("Processing Summary:\n")
cat("  Input:       ", nrow(raw_data), " records\n", sep = "")
cat("  Clean:       ", nrow(clean_data), " records\n", sep = "")
cat("  Training:    ", nrow(train_data), 
    sprintf(" (%.1f%%, %d ADRD)\n", actual_prop * 100, sum(train_data$label)), sep = "")
cat("  Test:        ", nrow(test_data), 
    sprintf(" (%.1f%%, %d ADRD)\n", (1 - actual_prop) * 100, sum(test_data$label)), sep = "")

cat("\nLabel Conversion:\n")
cat("  'ADRD'     to 1 (", sum(clean_data$label == 1), " cases)\n", sep = "")
cat("  'NON-ADRD' to 0 (", sum(clean_data$label == 0), " cases)\n", sep = "")

cat("\nColumn Transformations:\n")
cat("  'content' to 'txt' (downstream compatibility)\n")
cat("  'Label'   to 'label' (numeric format)\n")

cat("\nOutput Files:\n")
cat("  ", train_file, "\n", sep = "")
cat("  ", test_file, "\n", sep = "")
cat("  ", split_info_file, "\n", sep = "")
cat("  ", summary_file, "\n", sep = "")

cat("\nNext Steps:\n")
cat("  1. Review ", basename(summary_file), " for detailed statistics\n", sep = "")
cat("  2. Verify demographic distributions if needed\n")
cat("  3. Run: Rscript 03_evaluate_models.R (using Jihad's pre-trained models)\n")
cat("  4. Skip training - models are pre-trained!\n")

cat("\n", strrep("=", 80), "\n", sep = "")
cat("Data preparation completed successfully!\n")
cat("Full dataset ready for CNN evaluation.\n")
cat(strrep("=", 80), "\n", sep = "")