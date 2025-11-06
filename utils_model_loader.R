# ==============================================================================
# Model and Artifact Loading Utilities
# ==============================================================================
# Purpose: Load trained models and artifacts with compatibility for both
#          current naming conventions and Jihad Obeid's original naming
#
# Supports:
# - Current naming: tokenizer_cnnr, model_CNNr01.h5, etc.
# - Jihad's naming: CL07_tokenizer_ref2, CL07_model_CNNr01.h5, etc.
# ==============================================================================

library(keras)
library(reticulate)

# Detect naming convention and find artifacts
find_artifact <- function(artifact_type, model_dir = "models", cycle = NULL) {

  patterns <- list(
    tokenizer = c(
      "tokenizer_cnnr",           # Current convention
      "CL07_tokenizer_ref2"       # Jihad's convention
    ),
    model = c(
      sprintf("model_CNNr%02d.h5", cycle),           # Current
      sprintf("CL07_model_CNNr%02d.h5", cycle)       # Jihad's
    ),
    history = c(
      sprintf("history_CNNr%02d.rds", cycle),        # Current
      sprintf("CL07_model_CNNr%02d_hx.rds", cycle)   # Jihad's
    ),
    word_index = c(
      "word_index.rds",           # Current
      "CL07_word_index.rds"       # Jihad's (if exists)
    ),
    vocab_size = c(
      "vocab_size.rds",           # Current
      "CL07_vocab_size.rds"       # Jihad's (if exists)
    ),
    maxlen = c(
      "maxlen.rds",               # Current
      "CL07_maxlen.rds"           # Jihad's (if exists)
    )
  )

  if (!artifact_type %in% names(patterns)) {
    stop("Unknown artifact type: ", artifact_type)
  }

  # Try each pattern
  for (pattern in patterns[[artifact_type]]) {
    path <- file.path(model_dir, pattern)
    if (file.exists(path)) {
      cat("  Found", artifact_type, ":", basename(path), "\n")
      return(path)
    }
  }

  # Not found
  return(NULL)
}

# Load tokenizer (handles both Keras formats)
load_tokenizer_auto <- function(model_dir = "models") {
  cat("Loading tokenizer...\n")

  tokenizer_path <- find_artifact("tokenizer", model_dir)

  if (is.null(tokenizer_path)) {
    stop("Tokenizer not found! Tried:\n",
         "  - tokenizer_cnnr\n",
         "  - CL07_tokenizer_ref2")
  }

  tryCatch({
    tokenizer <- load_text_tokenizer(tokenizer_path)
    cat("  Tokenizer loaded successfully\n")
    return(tokenizer)
  }, error = function(e) {
    stop("Failed to load tokenizer from ", tokenizer_path, "\n",
         "Error: ", e$message)
  })
}

# Load model by cycle
load_model_auto <- function(cycle, model_dir = "models") {
  cat("Loading model for cycle", cycle, "...\n")

  model_path <- find_artifact("model", model_dir, cycle)

  if (is.null(model_path)) {
    stop("Model for cycle ", cycle, " not found! Tried:\n",
         "  - model_CNNr", sprintf("%02d", cycle), ".h5\n",
         "  - CL07_model_CNNr", sprintf("%02d", cycle), ".h5")
  }

  tryCatch({
    model <- load_model_hdf5(model_path)
    cat("  Model loaded successfully\n")
    return(model)
  }, error = function(e) {
    stop("Failed to load model from ", model_path, "\n",
         "Error: ", e$message)
  })
}

# Load maxlen
load_maxlen_auto <- function(model_dir = "models") {
  cat("Loading maxlen...\n")

  maxlen_path <- find_artifact("maxlen", model_dir)

  if (is.null(maxlen_path)) {
    stop("maxlen.rds not found!")
  }

  maxlen <- readRDS(maxlen_path)
  cat("  Maxlen:", maxlen, "\n")
  return(maxlen)
}

# Load vocab size
load_vocab_size_auto <- function(model_dir = "models") {
  cat("Loading vocab size...\n")

  vocab_path <- find_artifact("vocab_size", model_dir)

  if (is.null(vocab_path)) {
    cat("  vocab_size.rds not found (optional)\n")
    return(NULL)
  }

  vocab_size <- readRDS(vocab_path)
  cat("  Vocab size:", vocab_size, "\n")
  return(vocab_size)
}

# Load word index
load_word_index_auto <- function(model_dir = "models") {
  cat("Loading word index...\n")

  windex_path <- find_artifact("word_index", model_dir)

  if (is.null(windex_path)) {
    cat("  word_index.rds not found (optional)\n")
    return(NULL)
  }

  word_index <- readRDS(windex_path)
  cat("  Word index size:", length(word_index), "\n")
  return(word_index)
}

# Find all available model files
find_all_models <- function(model_dir = "models") {
  cat("Scanning for trained models...\n")

  # Try both naming conventions
  patterns <- c(
    "^model_CNNr\\d+\\.h5$",           # Current
    "^CL07_model_CNNr\\d+\\.h5$"       # Jihad's
  )

  all_files <- list.files(model_dir, full.names = FALSE)
  model_files <- c()

  for (pattern in patterns) {
    matches <- grep(pattern, all_files, value = TRUE)
    if (length(matches) > 0) {
      model_files <- c(model_files, file.path(model_dir, matches))
    }
  }

  if (length(model_files) == 0) {
    stop("No trained models found in ", model_dir, "\n",
         "Tried patterns:\n",
         "  - model_CNNr##.h5\n",
         "  - CL07_model_CNNr##.h5")
  }

  # Extract cycle numbers
  cycles <- as.numeric(gsub(".*CNNr(\\d+)\\.h5", "\\1", basename(model_files)))

  # Sort by cycle
  order_idx <- order(cycles)
  model_files <- model_files[order_idx]
  cycles <- cycles[order_idx]

  cat("  Found", length(model_files), "models (cycles:",
      paste(cycles, collapse = ", "), ")\n")

  return(list(
    files = model_files,
    cycles = cycles
  ))
}

# Load all artifacts needed for evaluation
load_all_artifacts <- function(model_dir = "models") {
  cat(strrep("=", 80), "\n")
  cat("Loading All Required Artifacts\n")
  cat(strrep("=", 80), "\n\n")

  artifacts <- list()

  # Load tokenizer
  artifacts$tokenizer <- load_tokenizer_auto(model_dir)

  # Load maxlen (required)
  artifacts$maxlen <- load_maxlen_auto(model_dir)

  # Load optional artifacts
  artifacts$vocab_size <- load_vocab_size_auto(model_dir)
  artifacts$word_index <- load_word_index_auto(model_dir)

  # Find all models
  models_info <- find_all_models(model_dir)
  artifacts$model_files <- models_info$files
  artifacts$cycles <- models_info$cycles

  cat("\n")
  cat("Artifacts loaded successfully!\n")
  cat("  Tokenizer: ✓\n")
  cat("  Maxlen: ✓\n")
  cat("  Models:", length(artifacts$cycles), "found\n\n")

  return(artifacts)
}

# Load results files with compatibility
find_results_file <- function(file_type, results_dir = "results") {

  patterns <- list(
    metrics = c(
      "metrics_df_CNNr.rds",
      "metrics_df_CNNr_MUSC_*.rds"
    ),
    roc = c(
      "roc_df_rows_CNNr.rds",
      "roc_df_rows_CNNr_MUSC_*.rds"
    ),
    predictions = c(
      "predictions_df.csv",
      "predictions_CNNr_MUSC_*.csv"
    )
  )

  if (!file_type %in% names(patterns)) {
    stop("Unknown results file type: ", file_type)
  }

  # Try exact matches first
  for (pattern in patterns[[file_type]]) {
    if (!grepl("\\*", pattern)) {
      # Exact pattern
      path <- file.path(results_dir, pattern)
      if (file.exists(path)) {
        return(path)
      }
    }
  }

  # Try glob patterns
  for (pattern in patterns[[file_type]]) {
    if (grepl("\\*", pattern)) {
      # Glob pattern
      files <- Sys.glob(file.path(results_dir, pattern))
      if (length(files) > 0) {
        # Return most recent if multiple
        return(files[length(files)])
      }
    }
  }

  return(NULL)
}

cat("Model loading utilities loaded\n")
cat("  find_artifact() - Find artifact with both naming conventions\n")
cat("  load_tokenizer_auto() - Load tokenizer automatically\n")
cat("  load_model_auto() - Load model by cycle\n")
cat("  load_all_artifacts() - Load all artifacts at once\n")
cat("  find_all_models() - Find all trained models\n")
cat("  find_results_file() - Find results files\n")
