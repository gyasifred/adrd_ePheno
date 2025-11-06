#!/usr/bin/env Rscript
# ==============================================================================
# ADRD ePhenotyping Pipeline - Data Preparation
# ==============================================================================
# Project: adrd_ephenotyping
# Purpose: Load, validate, and split data for CNN training
#
# Input:  data/raw/ptHx_sample_v2025-10-25.csv
# Output: data/train_set.rds, data/test_set.rds, data/split_info.rds
#
# Key Operations:
# 1. Load preprocessed clinical notes with STRING labels ('ADRD', 'NON-ADRD')
# 2. Convert string labels to numeric (ADRD→1, NON-ADRD→0)
# 3. Rename 'content' column to 'txt' for downstream compatibility
# 4. Create stratified 80/20 train/test split
# 5. Validate data quality and class balance
# 6. Save processed datasets with both original and numeric labels
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
INPUT_FILE   <- "ptHx_sample_v2025-10-25.csv"

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

TRAIN_PROPORTION <- 0.80
RANDOM_SEED      <- 42

# Required columns in input CSV
REQUIRED_COLS    <- c("DE_ID", "content", "Label")
# Available demographic columns
DEMOGRAPHIC_COLS <- c("GENDER", "RACE", "HISPANIC")

cat("  Input file:", INPUT_FILE, "\n")
cat("  Train/test split:", TRAIN_PROPORTION, "/", 1 - TRAIN_PROPORTION, "\n")
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

# Stratified Train/Test Split -------------------------------------------------
cat(strrep("=", 80), "\n", sep = "")
cat("Creating Stratified Train/Test Split\n")
cat(strrep("=", 80), "\n\n")

set.seed(RANDOM_SEED)

# Determine stratification variables
strat_vars <- c("label")
use_gender <- FALSE
use_race <- FALSE

if ("GENDER" %in% available_demos && n_distinct(clean_data$GENDER, na.rm = TRUE) > 1) {
  use_gender <- TRUE
  strat_vars <- c(strat_vars, "GENDER")
}

if ("RACE" %in% available_demos && n_distinct(clean_data$RACE, na.rm = TRUE) > 1) {
  race_counts <- clean_data %>% count(RACE, name = "n") %>% filter(n >= 20)
  if (nrow(race_counts) >= 2) {
    use_race <- TRUE
    strat_vars <- c(strat_vars, "RACE")
  }
}

cat("Stratification variables:", paste(strat_vars, collapse = ", "), "\n")
cat("  (Ensures balanced representation across groups)\n\n")

# Create composite stratification variable
# rsample's initial_split only accepts a single column for strata
if (length(strat_vars) > 1) {
  cat("Creating composite stratification variable...\n")
  
  # Build composite strata column
  clean_data <- clean_data %>%
    mutate(
      strata_group = case_when(
        use_gender & use_race ~ paste(label, GENDER, RACE, sep = "_"),
        use_gender & !use_race ~ paste(label, GENDER, sep = "_"),
        !use_gender & use_race ~ paste(label, RACE, sep = "_"),
        TRUE ~ as.character(label)
      )
    )
  
  # Check strata group sizes
  strata_summary <- clean_data %>%
    count(strata_group, name = "n") %>%
    arrange(n)
  
  cat("  Created", nrow(strata_summary), "strata groups\n")
  
  # Check for small groups
  small_groups <- strata_summary %>% filter(n < 5)
  if (nrow(small_groups) > 0) {
    cat("  WARNING:", nrow(small_groups), "groups have <5 patients\n")
    cat("  These may cause stratification issues\n")
    if (nrow(small_groups) <= 5) {
      cat("  Small groups:\n")
      print(small_groups)
    }
  }
  
  strata_col <- "strata_group"
} else {
  strata_col <- "label"
}

cat("\n")

# Perform stratified split
cat("Performing stratified split...\n")
split_obj <- initial_split(
  clean_data,
  prop = TRAIN_PROPORTION,
  strata = strata_col
)

train_data <- training(split_obj)
test_data  <- testing(split_obj)

# Remove temporary strata column if created
if ("strata_group" %in% names(train_data)) {
  train_data <- train_data %>% select(-strata_group)
  test_data <- test_data %>% select(-strata_group)
}

cat("Split results:\n")
cat("  Training set:", nrow(train_data), "patients\n")
cat("  Test set:    ", nrow(test_data),  "patients\n\n")

# Verify Split Quality --------------------------------------------------------
cat(strrep("=", 80), "\n", sep = "")
cat("Verifying Split Quality\n")
cat(strrep("=", 80), "\n\n")

# 1. Check for patient overlap
overlap <- intersect(train_data$DE_ID, test_data$DE_ID)
if (length(overlap) > 0) {
  stop("ERROR: Patient overlap detected - ", length(overlap), " IDs in both sets!")
} else {
  cat("  No patient overlap between train and test sets\n")
}

# 2. Check split proportion
actual_prop <- nrow(train_data) / (nrow(train_data) + nrow(test_data))
cat("  Actual split proportion:", sprintf("%.4f / %.4f", 
    actual_prop, 1 - actual_prop), "\n\n")

# 3. Check label balance
cat("Label balance verification:\n")
train_adrd_pct <- mean(train_data$label) * 100
test_adrd_pct  <- mean(test_data$label) * 100
cat("  Training set ADRD:", sprintf("%.2f%%", train_adrd_pct), 
    sprintf("(%d/%d)\n", sum(train_data$label), nrow(train_data)))
cat("  Test set ADRD:    ", sprintf("%.2f%%", test_adrd_pct), 
    sprintf("(%d/%d)\n", sum(test_data$label), nrow(test_data)))

prop_diff <- abs(train_adrd_pct - test_adrd_pct)
cat("  Difference:", sprintf("%.2f percentage points\n", prop_diff))

if (prop_diff > 5) {
  cat("  WARNING: >5% difference in class distribution\n")
} else {
  cat("  Well-balanced split\n")
}
cat("\n")

# 4. Check demographic balance
if (length(available_demos) > 0) {
  cat("Demographic balance across splits:\n")
  for (demo_col in available_demos) {
    cat("\n", demo_col, ":\n", sep = "")
    
    train_tbl <- train_data %>% count(!!sym(demo_col), name = "Train_N")
    test_tbl  <- test_data  %>% count(!!sym(demo_col), name = "Test_N")
    
    balance_tbl <- full_join(train_tbl, test_tbl, by = demo_col) %>%
      replace_na(list(Train_N = 0, Test_N = 0)) %>%
      mutate(
        Train_Pct = Train_N / sum(Train_N) * 100,
        Test_Pct  = Test_N  / sum(Test_N)  * 100,
        Diff_Pct  = abs(Train_Pct - Test_Pct)
      )
    
    print(balance_tbl %>% 
          select(!!sym(demo_col), Train_N, Test_N, Train_Pct, Test_Pct, Diff_Pct), 
          n = 15)
    
    max_diff <- max(balance_tbl$Diff_Pct, na.rm = TRUE)
    if (max_diff > 10) {
      cat("  WARNING: Some groups differ by >10 percentage points\n")
    }
  }
  cat("\n")
}

# Save Split Metadata ---------------------------------------------------------
cat(strrep("=", 80), "\n", sep = "")
cat("Saving Split Information\n")
cat(strrep("=", 80), "\n\n")

# Create comprehensive split metadata
split_info <- bind_rows(
  train_data %>% 
    select(DE_ID, Label, label, all_of(available_demos)) %>% 
    mutate(partition = "train"),
  test_data  %>% 
    select(DE_ID, Label, label, all_of(available_demos)) %>% 
    mutate(partition = "test")
)

split_info_file <- file.path(OUTPUT_DIR, "split_info.rds")
saveRDS(split_info, split_info_file)
cat("Split metadata saved:", split_info_file, "\n")
cat("  Contains: patient IDs, labels, demographics, partition assignment\n\n")

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
cat("  3. Run: Rscript scripts/02_train_cnnr.R\n")

cat("\n", strrep("=", 80), "\n", sep = "")
cat("Data preparation completed successfully!\n")
cat(strrep("=", 80), "\n", sep = "")