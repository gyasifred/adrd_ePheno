---
title: "Classification"
output:
  html_document:
    df_print: paged
    toc: yes
    toc_depth: '2'
  html_notebook:
    theme: cerulean
    toc: yes
    toc_depth: 2
editor_options:
  chunk_output_type: inline
---

# ADRD Classification Pipeline
# Code Author: Jihad Obeid

# Purpose

# ADRD Classification Pipeline
# This script implements a comprehensive machine learning pipeline for classifying
# clinical text notes for ADRD (Alzheimer's Disease and Related Dementias) detection.
# It uses both traditional ML (Random Forest, SVM) and deep learning (CNN) approaches.
# 
# Binary Classification Task:
# - Label 0: Control group (CTRL) - patients without ADRD
# - Label 1: ADRD group - patients with Alzheimer's Disease and Related Dementias

Classify for ADRD (label)

# Load packages

```{r message=FALSE, warning=FALSE}
# Load necessary packages for data manipulation, text processing, and machine learning
suppressPackageStartupMessages({
library(dplyr)     # Data manipulation and transformation
library(tidyr)     # Data tidying and reshaping  
library(magrittr)  # Pipe operators (%>%, %<>%)
library(stringr)   # String manipulation and regex operations
library(ggplot2)   # Data visualization and plotting
library(scales)    # Scale functions for plots
library(janitor)   # Data cleaning utilities
library(quanteda) # Comprehensive text analytics framework
library(quanteda.textplots)  # Text visualization tools
library(quanteda.textstats)  # Text statistics functions
library(quanteda.textmodels) # Text modeling functions
library(caret)     # Classification and Regression Training - ML workflows
library(pROC)      # ROC curve analysis and AUC calculation
library(tictoc)    # Simple timing functions for performance monitoring
library(keras)     # High-level neural networks API (TensorFlow backend)
})

# Configure global options
options(dplyr.summarise.inform = FALSE)  # Suppress dplyr grouping messages
quanteda_options(threads = 10)           # Use 10 threads for parallel processing
```

```{r}
# Load custom utility functions
suppressPackageStartupMessages({
# source("utils3.R")      # Additional utility functions (commented out)
source("utils4.R")        # Text processing utilities (includes text2dfm function)
source("7z_funs.R")       # Functions for reading 7z compressed files
})
```

# Read Data

```{r}
# Load the main clinical notes dataset from compressed file
# notes_pheno_clean.7z contains:
# - DE_ID: De-identified patient ID
# - txt: Preprocessed clinical text with special tokens
# - label: Binary classification target (0=CTRL, 1=ADRD)  
# - exclude: Quality control flag (0=include, 1=exclude)
notes_concat_clean = read7zR("../data/notes_pheno_clean.7z")

# Load pre-determined patient train/test splits
# This ensures consistent evaluation and prevents data leakage
# pts_shuffle_train-test.rds contains:
# - DE_ID: Patient identifier matching notes dataset
# - partition: "train" or "test" assignment
# - label: Patient-level label for verification
pts_train_test = readRDS("../data/pts_shuffle_train-test.rds")

# Set institution identifier for output file naming
institution = "MUSC"  # Change this to your institution abbreviation
```

# Select Notes

```{r}
# Start timing for data selection process
tic()

# Create the main machine learning dataset
# Select only essential columns and optionally filter excluded records
notes_ml = notes_concat_clean %>% 
  # filter(exclude==0) %>%  # Uncomment to exclude flagged records
  select(DE_ID, txt, label)   # Keep only ID, text, and label columns
#  mutate(label = factor(label, levels=c(0,1), labels=c("CTRL", "ADRD"))) %>%  # Convert to factor labels if needed

# Stop timing and display duration
toc()
```

```{r}
# Display total number of records in the dataset
notes_ml %>% nrow
```

```{r}
# Show class distribution - important for understanding data balance
# This helps identify if we have class imbalance issues
notes_ml %>% 
  group_by(label) %>%
  summarise(n=n())
```

## Partition data

```{r}
# Alternative approach: Random sampling within R (currently commented out)
# This would create random train/test splits instead of using pre-determined splits
# set.seed(123)
# idx_train <- createDataPartition(notes_ml$label, times = 1,
#                                p = 0.75, list = FALSE)
# 
# # deep learning train/test
# notes_train <- notes_ml[idx_train,]
# notes_test <- notes_ml[-idx_train,]
```

```{r}
# Use pre-determined patient-level train/test splits
# This approach ensures all notes from the same patient stay in the same split,
# preventing data leakage and providing more realistic performance estimates

# Create training set by joining with partition data and filtering
notes_train <- notes_ml %>% 
  inner_join(pts_train_test %>% select(-label), join_by(DE_ID)) %>%  # Join by patient ID
  filter(partition=="train")  # Keep only training patients

# Create test set using same approach
notes_test <- notes_ml %>% 
  inner_join(pts_train_test %>% select(-label), join_by(DE_ID)) %>%  # Join by patient ID  
  filter(partition=="test")   # Keep only test patients
```

verify proportions

```{r}
# Verify that class proportions are similar between train and test sets
# This is crucial for fair evaluation
prop.table(table(notes_train$label))
```

```{r}
prop.table(table(notes_test$label))
```

## Y_test

```{r}
# Extract test labels for evaluation purposes
# This will be used throughout the pipeline for calculating metrics
y_test <- notes_test$label
```

# BOW Data Set ====

# Traditional Machine Learning: Bag-of-Words (BOW) Approach
# For traditional ML models (Random Forest, SVM), we need to remove special tokens
# that were added during preprocessing, as they don't provide meaningful information
# for bag-of-words representations

Clean out standard tokens.

```{r}
tic()  # Start timing

# Remove special preprocessing tokens from text
# These tokens were useful for neural networks but not for BOW approaches:
# - </s>: Sentence separator token
# - _decnum_: Decimal number placeholder  
# - _lgnum_: Large number placeholder
# - _date_: Date placeholder
# - _time_: Time placeholder
notes_bow = notes_ml %>% 
  mutate(txt = str_remove_all(txt, "</s>|_decnum_|_lgnum_|_date_|_time_"))

toc()  # Display processing time
```

Create BOW partitions

```{r}
# Create train/test splits for BOW models using the cleaned text
# These splits maintain the same patient-level partitioning as the original data

# BOW training set with cleaned text (no special tokens)
notes_train_bow <- notes_bow %>% 
  inner_join(pts_train_test %>% select(-label), join_by(DE_ID)) %>% 
  filter(partition=="train")

# BOW test set with cleaned text (no special tokens)  
notes_test_bow <- notes_bow %>% 
  inner_join(pts_train_test %>% select(-label), join_by(DE_ID)) %>% 
  filter(partition=="test")
```

# Init roc_df

```{r}
# Initialize data structures for collecting model results
# roc_df will store test labels and predictions from all models for ROC analysis
roc_df <- data.frame(Label=y_test)  # Start with test labels as first column

# summary_df will store performance metrics (AUC, accuracy, etc.) for all models
summary_df <- NULL  # Will be populated using custom functions
```

# BOW MODELS======

## Utils

```{r}
# Load additional utility functions for model evaluation
suppressPackageStartupMessages({
# source("utils3.R")                    # Additional utilities (commented out)
source("metrics_collection_dl.R")       # Functions for collecting performance metrics
})
# tr = reticulate::import('transformers')  # For transformer models (if needed)
```

## Text-Matrices

# Feature Engineering: Convert Text to Numerical Representations
# This section creates document-feature matrices (DFM) and TF-IDF representations
# that traditional ML algorithms can work with

Takes ~8 secs

```{r}
# ==== NLP pre-processing ====
# Convert text documents to document-feature matrices (sparse matrices)
# text2dfm() is a custom function that handles:
# - Tokenization (splitting text into words)
# - Lowercasing
# - Removing punctuation and stop words
# - Creating document-term matrix

tic()  # Start timing

# Create DFM for training data
# Each row = document, each column = unique word, values = word frequencies
train_tokens_dfm <- text2dfm(notes_train_bow$txt)

# Create DFM for test data using same preprocessing
test_tokens_dfm <- text2dfm(notes_test_bow$txt)

toc()  # Display processing time (~8 seconds expected)
```

```{r}
# Display matrix dimensions to understand feature space
# Format: [number_of_documents, number_of_unique_features]
dim(train_tokens_dfm)  # Training matrix dimensions
dim(test_tokens_dfm)   # Test matrix dimensions
```

## reduce feature space

```{r}
# Feature selection to reduce dimensionality and improve model performance
# Remove rare words that appear infrequently across the corpus
# This reduces noise and computational complexity

# Apply frequency-based feature selection:
# - min_termfreq = 50: Word must appear at least 50 times total
# - min_docfreq = 0.01: Word must appear in at least 1% of documents
train_tokens_dfm <- dfm_trim(train_tokens_dfm, min_termfreq = 50, min_docfreq = 0.01)

# Display reduced dimensions
dim(train_tokens_dfm)  # Should show fewer features than before
```

## tfidf

```{r}
# Create TF-IDF (Term Frequency-Inverse Document Frequency) representations
# TF-IDF weights words by:
# - Term Frequency: How often word appears in document
# - Inverse Document Frequency: How rare the word is across all documents
# This helps identify words that are distinctive for each document

tic()  # Start timing

# Convert frequency matrices to TF-IDF weighted matrices
train_tokens_tfidf <- train_tokens_dfm %>% dfm_tfidf()
test_tokens_tfidf <- test_tokens_dfm %>% dfm_tfidf()

# Ensure test set features exactly match training set features
# This prevents dimensionality mismatches and ensures consistent feature space
test_tokens_dfm <- dfm_match(test_tokens_dfm, 
                             features = featnames(train_tokens_dfm))
test_tokens_tfidf <- dfm_match(test_tokens_tfidf, 
                               features = featnames(train_tokens_tfidf))

toc()  # Display processing time

# Verify final dimensions match between train and test
dim(train_tokens_dfm)   # Training matrix final dimensions
dim(test_tokens_dfm)    # Test matrix final dimensions (should match feature count)
```

Note: the labels have to be 0's and 1's. Not factors. 
So you may need to subtract -1 from the factored labels.

Takes ~42 secs

```{r}
tic()  # Start timing (~42 seconds expected)

# Convert sparse quanteda matrices to standard R data frames
# Remove first column which contains document names (not needed for modeling)
train_tokens_df <- convert(train_tokens_tfidf, to = "data.frame")[, -1]
test_tokens_df <-  convert(test_tokens_tfidf, to = "data.frame")[, -1]

toc()  # Display conversion time
```

```{r}
tic()  # Start timing

# Prepare data in multiple formats for different model requirements

# Create matrices for models that expect matrix input (e.g., neural networks)
x_train <- as.matrix(train_tokens_df)  # Training feature matrix
x_test <- as.matrix(test_tokens_df)    # Test feature matrix

# Extract label vectors (must be numeric, not factors)
y_train <- notes_train_bow$label  # Training labels (0s and 1s)
y_test <- notes_test_bow$label    # Test labels (0s and 1s)

# Create data frames with labels for models that expect data.frame input
# Convert labels to factors for algorithms like Random Forest and SVM
train_tokens_df1 <- cbind(label=as.factor(y_train), train_tokens_df)
test_tokens_df1 <- cbind(label=as.factor(y_test), test_tokens_df)

toc()  # Display processing time
```

```{r}
# Display final matrix dimensions for verification
# These matrices will be used by neural networks
dim(x_train)  # Training matrix: [n_train_docs, n_features]
dim(x_test)   # Test matrix: [n_test_docs, n_features]
```

## Random Forest

# Random Forest Model Training
# Random Forest is an ensemble method that builds multiple decision trees
# and combines their predictions. It's robust to overfitting and provides
# feature importance measures.

Est run time: 1 sec.

```{r message=FALSE, warning=FALSE}
# Load Random Forest implementation
suppressPackageStartupMessages({
library(ranger)  # Fast implementation of Random Forest
})
```

```{r}
tic()  # Start timing

# Set Random Forest hyperparameters
mtry_min <- 150  # Number of variables to randomly sample at each split
                # This controls the randomness vs accuracy tradeoff

set.seed(1234)   # Set seed for reproducible results

# Train Random Forest model using ranger (faster than randomForest)
mod <- ranger(
  label~.,                    # Formula: predict label from all features  
  data = train_tokens_df1,   # Training data with factor labels
  num.trees=201,             # Number of trees (odd number prevents ties in voting)
  mtry=mtry_min,             # Variables to consider at each split
  importance="permutation",   # Calculate variable importance via permutation
  probability = TRUE         # Return class probabilities (needed for ROC)
)

toc()  # Display training time (~1 second expected)
```

```{r}
# Generate predictions on test set
# Random Forest returns probabilities for each class
pred <- predict(mod, test_tokens_df)

# Extract probability of ADRD class (class 1, second column)
# pred$predictions is a matrix: [n_samples, n_classes]
prob_rf <- pred$predictions[,2]  # Probability of positive class (ADRD)
```

```{r}
# Store Random Forest predictions in results data frame
# This accumulates predictions from all models for ROC comparison
roc_df <- cbind(roc_df, RF= prob_rf)
```

```{r}
# Calculate and display summary metrics for Random Forest
# summary_metrics() is a custom function that calculates:
# - AUC (Area Under ROC Curve)
# - Accuracy, Precision, Recall, F1-score
# - Optimal threshold via Youden's index
summary_df %<>% summary_metrics("RF", test_labels = y_test, pred_prob = prob_rf)
```

### Var Importance

```{r}
# Analyze and visualize feature importance from Random Forest
# Variable importance shows which words/features are most predictive

# Extract variable importance scores from trained model
df <- data.frame("Feature" = names(mod$variable.importance),
                 "MDA" = mod$variable.importance)  # MDA = Mean Decrease in Accuracy

# Create variable importance plot
df %>% 
  arrange(desc(MDA)) %>%           # Sort by importance (descending)
  head(15) %>%                     # Show top 15 most important features
  ggplot(aes(x=reorder(Feature,MDA), y=MDA, fill=MDA))+ 
      geom_bar(stat="identity", position="dodge") +  # Horizontal bar chart
      coord_flip() +                               # Flip coordinates for readability
      ylab("Mean Decrease in Accuracy (MDA)") +    # Y-axis label
      xlab("") +                                   # No X-axis label needed
      # ggtitle("Variable Importance") +           # Optional title
      guides(fill="none") +                        # Remove legend
      scale_fill_gradient(low="red", high="blue") + # Color gradient
      theme(axis.text.x = element_text(size=12),   # Customize text sizes
            axis.text.y = element_text(size=12),
            axis.title.x = element_text(size=14))
```

## Save Var imp

```{r}
# Save variable importance plot with timestamp
date_sfx <- format(Sys.time(), "_%Y-%m-%d")  # Create date suffix
fn <- paste0("RF_MDA_ADRD01",date_sfx,".png")  # Generate filename

# Save plot as high-resolution PNG
ggsave(fn, plot = last_plot(),
       scale = 1, width = 6, height = 4, units ="in",
       dpi = 300)  # 300 DPI for publication quality
```

## SVM

# Support Vector Machine (SVM) Model Training
# SVM finds optimal hyperplane to separate classes by maximizing margin
# Uses RBF (Radial Basis Function) kernel for non-linear classification

Est run time: 40 secs.

```{r message=FALSE, warning=FALSE}
# Load SVM implementation
library(kernlab)  # Kernel-based machine learning methods
# Note: cross and alpha parameters will be overridden by our settings
```

```{r}
tic()  # Start timing (~40 seconds expected)

# Train SVM with RBF kernel and optimized hyperparameters
SVM <- kernlab::ksvm(
  label ~ .,                        # Formula: predict label from all features
  data = train_tokens_df1,         # Training data with factor labels
  scaled = FALSE,                  # No feature scaling (TF-IDF already normalized)
  kernel = "rbfdot",               # Radial Basis Function kernel for non-linear patterns
  kpar = "automatic",              # Automatic kernel parameter selection
  C = 60,                          # Regularization parameter (higher = less regularization)
                                   # C=30-60 provides good middle ground
  cross = 5,                       # 5-fold cross-validation for model selection
  type = "C-svc",                  # C-classification (standard SVM classification)
  prob.model = TRUE                # Enable probability estimates (needed for ROC)
)

toc()  # Display training time
```

```{r}
# Display SVM model summary including:
# - Training error and cross-validation error
# - Number of support vectors
# - Kernel parameters
SVM
```

```{r}
# Generate probability predictions on test set
# Extract probability of ADRD class (positive class, second column)
prob_svm <- kernlab::predict(SVM, test_tokens_df, type = 'prob')[,2]

# Store SVM predictions in results data frame
roc_df <- cbind(roc_df, SVM= prob_svm)
```

```{r}
# Calculate and display summary metrics for SVM
# Compare performance with Random Forest
summary_df %<>% summary_metrics("SVM", test_labels = y_test, pred_prob = prob_svm)
```

## Save checkpoint

```{r}
# Save traditional ML results as checkpoint for recovery
# This allows resuming from this point if deep learning training fails
save(mod, SVM, roc_df, summary_df, file = "rf_svm_checkpoint.rda")
```

## Retrieve checkpoin

# DL MODELS======

# Deep Learning Models Section
# This section implements Convolutional Neural Networks (CNNs) for text classification
# Two variants: CNN with random embeddings (CNNr) and CNN with Word2Vec embeddings (CNNw)

```{r}
# Load saved traditional ML results to continue pipeline
load("rf_svm_checkpoint.rda")
```

## DL Libraries

```{r}
# Load deep learning libraries
suppressPackageStartupMessages({
library(keras)        # High-level neural networks API
library(tensorflow)   # TensorFlow backend for deep learning
library(dplyr)        # Data manipulation (reload to ensure availability)
})

# Configure dplyr to suppress grouping messages
options(dplyr.summarise.inform = FALSE)
```

## Utils

```{r}
# Load utility functions for deep learning
suppressPackageStartupMessages({
source("utils3.R")                    # Additional utility functions
source("metrics_collection_dl.R")     # Deep learning specific metric collection
})
# tr = reticulate::import('transformers')  # For transformer models (if needed)
```

## DL PREP

# Deep Learning Preparation
# Configure TensorFlow settings and prepare text data for neural networks

### Memory Issues

# GPU Memory Configuration
# Configure TensorFlow to avoid GPU memory issues by allowing memory growth
# This prevents TensorFlow from allocating all GPU memory at once

Note that if running TensorFlow on GPU one could specify the following parameters in order to avoid memory issues. (Not sure if needed, since it worked without this call.)

```{r}
# Configure GPU memory growth for each available GPU
# This prevents out-of-memory errors on systems with limited GPU RAM
physical_devices = tf$config$list_physical_devices('GPU')
for(gpu in physical_devices){
  cat("1-", gpu$name, "\n")  # Print GPU name
  tf$config$experimental$set_memory_growth(gpu,TRUE)  # Enable memory growth
}
```

Set default float type to float32.

```{r}
# Set default floating point precision to float32
# This is more memory efficient than float64 and sufficient for most NLP tasks
tf$keras$backend$set_floatx('float32')
```

## tokenize

# Text Tokenization for Neural Networks
# Convert raw text to sequences of integers that neural networks can process

```{r}
# Set vocabulary size limit for neural networks
# Larger vocabularies capture more nuance but require more memory
max_vocab <- 40000

# Create and fit tokenizer on all available text data
# filters = "" means we keep all characters (preprocessing already done)
tokenizer <- text_tokenizer(num_words = max_vocab, filters = "") %>% 
  fit_text_tokenizer(notes_ml$txt)  # Fit on complete dataset for comprehensive vocabulary

# Extract word-to-index mapping from fitted tokenizer
word_index = tokenizer$word_index

# Calculate actual vocabulary size (+1 for padding/unknown tokens)
vocab_size <- length(word_index)+1

cat("Found", vocab_size, "unique tokens.\n")

# Convert all texts to sequences of integers
# Each word is replaced by its index in the vocabulary
my_seq <- texts_to_sequences(tokenizer, notes_ml$txt)

# Count words per document after tokenization
word_counts <- sapply(my_seq, length)
```

## save tokenizer

```{r}
# Save trained tokenizer for future use and reproducibility
# This ensures consistent text preprocessing across different runs
tokenizer %>% save_text_tokenizer("../models/CL07_tokenizer_ref2")
```

```{r}
# Determine maximum sequence length for padding
# +1 provides small buffer for edge cases
maxlen <- max(word_counts) + 1

cat("Max len+1:", maxlen, " words.\n")
```

Set limit if needed.

```{r}
# Optional: Set a reasonable limit for sequence length to manage memory
# Uncomment and adjust if you encounter memory issues
# maxlen <- 1000
```

```{r}
# Create dataset with word counts for analysis
# This helps understand document length distribution
notes_wc = notes_ml %>% 
  select(DE_ID, label, txt)

# Add word count column
notes_wc$wc = word_counts
```

```{r}
# Visualize word count distribution by class
# This helps understand if document length differs between CTRL and ADRD groups
notes_wc %>% 
  mutate(Group=factor(label, labels = c("CTRL", "ADRD"))) %>% 
ggplot(aes(x=wc, fill=Group)) + 
  geom_density(alpha=0.25) +  # Semi-transparent density plots
  labs(title="Word Count Distribution by Class",
       x="Word Count", y="Density")
```

Wc vs wc_raw

# The following code blocks analyze word count differences
# They are currently commented out but can be used for additional analysis

```{r}
# Compare tokenized word counts with raw word counts (if available)
# notes_wc %>% 
# ggplot(aes(x=wc)) + 
#   geom_density(alpha=0.25) + 
#   geom_density(aes(x=wc_raw), color="red", linetype="dashed")
```

```{r}
# Analyze difference between raw and processed word counts
# notes_wc %>% 
#   mutate(delta=wc_raw - wc) %>% 
#   summarise(n=n(), min(delta), max(delta), mean(delta))
```

### Save Wordcounts

```{r}
# Save word count analysis for future reference
saveRDS(notes_wc, "../data/notes_clean_wc.rds")
```

## set maxlen

based on graph above

```{r}
# Alternative: Set fixed maximum length based on data analysis
# This can help manage memory usage for very long documents
# maxlen = 800
```

## create sequences

```{r}
tic()  # Start timing

# Convert training and test text to integer sequences
# Each word is replaced by its vocabulary index
seq_train <- texts_to_sequences(tokenizer, notes_train$txt)
seq_test <- texts_to_sequences(tokenizer, notes_test$txt)

toc()  # Display conversion time
```

## convert to matrix

```{r}
# Convert variable-length sequences to fixed-length matrices
# Pad sequences to uniform length (required for batch processing)

# Pad training sequences to maxlen (shorter sequences padded with 0s)
x_train <- pad_sequences(seq_train, maxlen = maxlen) #, padding = "post"

# Pad test sequences to same length
x_test <- pad_sequences(seq_test, maxlen = maxlen)   #, padding = "post"

# Convert labels to numeric format (required by Keras)
y_train <- as.numeric(notes_train$label)  # Training labels
y_test <- as.numeric(notes_test$label)    # Test labels

# Display tensor shapes for verification
cat("Shape of data tensor:", dim(x_train), "\n")     # [n_samples, sequence_length]
cat('Shape of label tensor:', length(y_train), "\n") # [n_samples]
```

## CNNr

# CNN with Random Embeddings (CNNr)
# This model learns word embeddings from scratch during training

### build CNN function

```{r}
# CNN Model Architecture Definition
# This function creates a CNN model for text classification with flexible embedding options

build_cnn <- function(EMBEDDING_DIM = 200,      # Embedding vector dimension
                      num_filters = 200,        # Number of CNN filters per kernel size
                      maxlen = 800,             # Maximum input sequence length  
                      vocab_size = 10000,       # Vocabulary size for embedding layer
                      pre_trained = F,          # Use pre-trained embeddings?
                      embedding_matrix = NULL,  # Pre-trained embedding weights
                      trainable_embed=T) {      # Allow embedding layer training?
  
  # Model hyperparameters
  hidden_dims <- 200      # Hidden layer size
  filt_sz <- c(3, 4, 5)   # CNN kernel sizes (captures 3,4,5-grams)
  drop_rate <- 0.2        # Dropout rate for regularization
  
  # Input layer: accepts sequences of integers (word indices)
  inputs <- layer_input(shape = maxlen, name = "input")
  
  # Embedding layer: converts word indices to dense vectors
  if(pre_trained == FALSE) {
    # Random embeddings: initialized randomly, learned during training
    embedding_lyr <- inputs %>% 
      layer_embedding(input_dim = vocab_size,      # Size of vocabulary
                     output_dim = EMBEDDING_DIM,   # Embedding dimension
                     input_length = maxlen,        # Input sequence length
                     name = "embed") %>% 
      layer_dropout(drop_rate, name = "drop1")    # Dropout for regularization
  } else {
    # Pre-trained embeddings: loaded from external source (e.g., Word2Vec)
    embedding_lyr <- inputs %>% 
      layer_embedding(input_dim = vocab_size, 
                     output_dim = EMBEDDING_DIM, 
                     input_length = maxlen, 
                     trainable = trainable_embed,          # Allow fine-tuning?
                     weights = list(embedding_matrix),     # Load pre-trained weights
                     name = "embed") %>% 
      layer_dropout(drop_rate, name = "drop1")
  }
  
  # Parallel CNN branches with different kernel sizes
  # This captures different n-gram patterns simultaneously:
  # - conv_1: captures 3-gram patterns
  # - conv_2: captures 4-gram patterns  
  # - conv_3: captures 5-gram patterns
  
  conv_1 <- embedding_lyr %>% 
    layer_conv_1d(filters= num_filters, filt_sz[1], activation = "relu", name = "conv1") %>% 
    layer_global_max_pooling_1d(name = "maxp1_glob")  # Extract most important features
  
  conv_2 <- embedding_lyr %>% 
    layer_conv_1d(filters= num_filters, filt_sz[2], activation = "relu", name = "conv2") %>% 
    layer_global_max_pooling_1d(name = "maxp2_glob")
  
  conv_3 <- embedding_lyr %>% 
    layer_conv_1d(filters= num_filters, filt_sz[3], activation = "relu", name = "conv3") %>% 
    layer_global_max_pooling_1d(name = "maxp3_glob")
  
  # Concatenate features from all CNN branches
  merged_tensor <- layer_concatenate(c(conv_1, conv_2, conv_3), axis=1, name = "merged")
  
  # Fully connected layers for final classification
  hidden <- merged_tensor %>% 
    layer_dense(hidden_dims, activation = "relu", name = "hidden")
  
  # Dropout and activation for regularization
  dropout <- hidden %>% 
    layer_dropout(drop_rate, name = "drop2") %>% 
    layer_activation("relu", name = "activ_solo")
  
  # Output layer: single neuron with sigmoid for binary classification
  output <- dropout %>% 
    layer_dense(1, activation = "sigmoid", name = "sigm")  # Outputs probability [0,1]
  
  # Create Keras model connecting inputs to outputs
  model <- keras_model(inputs, output)
  
  # Compile model with loss function, optimizer, and metrics
  model %>% compile(
    loss = "binary_crossentropy",  # Standard loss for binary classification
    # loss = tf$losses$BinaryCrossentropy(from_logits=F),  # Alternative syntax
    optimizer = "adam",            # Adam optimizer (adaptive learning rate)
    metrics = "accuracy"          # Track accuracy during training
    # metrics = tf$metrics$Accuracy()  # Alternative syntax
  )
  
  return(model)  # Return compiled model
}
```

### build model

```{r}
# Create CNN model instance with random embeddings
model <- build_cnn(EMBEDDING_DIM = 200,    # 200-dimensional embeddings
                   num_filters = 200,      # 200 filters per kernel size
                   maxlen = maxlen,        # Use calculated max length
                   vocab_size = vocab_size) # Use actual vocabulary size

# Display model architecture summary
summary(model)
```

### train

# Model Training with Statistical Robustness
# Train multiple instances of the same model to get stable performance estimates

Consider running 10x. Save rocs and metrics from each. Take median auc with max F.

~ 80 secs for one cyle (if ~8k tokens)

```{r}
# Training configuration
mdl_nm = "CNNr"        # Model name for file identification
n_cyles = 10           # Number of training runs for statistical robustness

# Hyperparameters
lr <- .0004            # Learning rate
epochs <- 30           # Maximum epochs per run
batch_size <- 32       # Mini-batch size

# Set learning rate in optimizer
k_set_value(model$optimizer$lr, lr)

# Training callbacks for adaptive training
early_stopping <- callback_early_stopping(monitor = 'val_loss', patience = 7)  # Stop if no improvement
reduce_lr <- callback_reduce_lr_on_plateau(monitor = 'val_loss', factor=0.5,    # Reduce LR if plateau
                                          patience = 1, min_lr = 0.0001,
                                          verbose = 0)

# Initialize result storage
roc_df_rows <- data.frame(matrix(y_test, nrow=1))  # First row contains test labels
metrics_df <- NULL  # Will store metrics from each training run

# Main training loop for statistical robustness
tic()  # Start timing entire training process
for(i in 1:n_cyles){
  # Generate unique identifiers for each run
  mdl_id <- paste0(mdl_nm, str_pad(i,2,pad = "0"))  # e.g., "CNNr01", "CNNr02"
  mdl_file_nm <- paste0("../models/CL07_model_", mdl_id, ".h5")    # Model save path
  hx_file_nm <- paste0("../models/CL07_model_", mdl_id, "_hx.rds") # History save path
  
  # Build fresh model for each run (prevents weight inheritance)
  model <- build_cnn(EMBEDDING_DIM = 200, num_filters = 200, 
                     maxlen = maxlen, vocab_size = vocab_size)
  
  # Train model with validation split
  t_begin=Sys.time()  # Start timing this run
  history <- model %>%
    keras::fit(
      x_train, y_train,                              # Training data and labels
      batch_size = batch_size,                       # Mini-batch size
      epochs = epochs,                               # Maximum epochs
      callbacks = list(early_stopping, reduce_lr),  # Adaptive training
      verbose=0,                                     # Silent training (no output)
      validation_split = 0.15                        # 15% for validation monitoring
    )
  
  # Calculate training time and actual epochs
  t_secs=as.numeric(difftime(Sys.time(), t_begin, units="secs"))
  eps=length(history$metrics[[1]])  # Actual epochs trained (may be less due to early stopping)
  
  # Generate predictions on test set
  pred <- model %>% predict(x_test)
  p <- pred[,1]  # Extract probabilities (single output neuron)
  
  # Store results
  roc_df_rows <- rbind(roc_df_rows, p)  # Append predictions to matrix
  
  # Collect detailed metrics for this run
  metrics_df %<>% summary_metrics_dl(model_name = mdl_nm, i=i,
                                     test_labels = y_test, 
                                     pred_prob = p,
                                     elapsed_time = t_secs,
                                     epochs = eps)

  # Save model and training history
  model %>% save_model_hdf5(mdl_file_nm)   # Save trained model
  history %>% saveRDS(hx_file_nm)          # Save training history
  
  cat("i:",i)  # Progress indicator
}
cat("\n")
toc()  # Display total training time
```

```{r}
# Display metrics from all training runs
# This shows variability across different random initializations
metrics_df
```

### save local metrics

```{r}
# Save detailed results with timestamp for future analysis
date_sfx <- format(Sys.time(), "_%Y-%m-%d")

# Save prediction matrix (rows = runs, columns = test samples)
saveRDS(roc_df_rows, paste0("../results/roc_df_rows_", mdl_nm, "_", institution, date_sfx, ".rds"))

# Save performance metrics for all runs
saveRDS(metrics_df, paste0("../results/metrics_df_", mdl_nm, "_", institution, date_sfx, ".rds"))
```

```{r}
# Plot training history from last run
# Shows loss and accuracy curves during training
plot(history)
```

### evaluate

Get median.

```{r}
# Select best model based on statistical robustness approach:
# 1. Find median AUC across all runs (reduces impact of outliers)
# 2. Among models closest to median AUC, pick highest F1 score

# Calculate median AUC across all runs
med_auc=median(metrics_df$auc)

# Find model closest to median AUC with best F1 score
med_mdl=metrics_df %>% 
  mutate(auc_d=abs(auc-med_auc)) %>%     # Distance from median AUC
  filter(auc_d==min(auc_d)) %>%          # Minimum distance (closest to median)
  filter(F1==max(F1))                    # Best F1 score among ties

# Get run index of selected model
med_i=med_mdl$i[1]

# Extract predictions from selected run (+1 because first row contains labels)
p = roc_df_rows[med_i+1,] %>% unlist() %>% unname()
```

### Summary Metrics

```{r}
# Add CNNr predictions to main results dataframe
roc_df <- bind_cols(roc_df, !!mdl_nm:= p)

# Calculate and display summary metrics for selected CNNr model
summary_df %<>% summary_metrics(mdl_nm, test_labels = y_test, pred_prob = p)
```

### Save check point

```{r}
# Save checkpoint including CNN results
save(institution, summary_df, roc_df, file = "tmp_summary_metrics.rda")
```

### Clear progress markers

```{r}
# Clean up temporary progress files
file.remove(dir(pattern = "tmp_progress_marker"))
```

## CNNw

# CNN with Word2Vec Embeddings (CNNw)
# This model uses pre-trained word embeddings instead of learning from scratch

### Retrieve checkpoin

```{r}
# Load checkpoint to continue from CNNr results
load("tmp_summary_metrics.rda")
```

### read saved w2v_df200

# Load Pre-trained Word Embeddings
# Word2Vec embeddings capture semantic relationships between words
# trained on large external corpora

~3.5secs

```{r}
tic()  # Start timing

# Load pre-trained 200-dimensional word embeddings
# This file contains word vectors trained on a large medical corpus
w2v_df <- readRDS("../../CIRR/CIRR_explor/cirr_w2v_df_d200.rds")

# Function to merge vocabulary with word vectors
# Creates embedding matrix matching our tokenizer's vocabulary
merge_vocab_wv2 <- function(vocab, wv, EMBEDDING_DIM = 200){
  cols = ncol(wv) - 1  # Number of embedding dimensions
  n = nrow(wv)         # Number of word vectors
  
  # Find words in our vocabulary that don't have pre-trained embeddings
  unmatched_words <- vocab %>% 
    anti_join(wv, by = "word") %>% 
    select(word)
  
  # Create random embeddings for unmatched words
  # Uses uniform distribution [-0.05, 0.05] matching Keras defaults
  rn <- rand_unif_df(nrow(unmatched_words), cols = EMBEDDING_DIM)
  unmatched_words <- bind_cols(unmatched_words, rn)
  
  # Combine pre-trained and random embeddings
  wv <- wv[1:n,] %>% dplyr::union(unmatched_words)
  
  # Create embedding matrix in correct order for our vocabulary
  we <- vocab %>% 
    inner_join(wv, by = "word") %>% 
    select(-c(1,2)) %>%                           # Remove word and index columns
    as.matrix() %>% 
    scales::rescale(to= c(-0.05, 0.05))          # Rescale to Keras default range
  
  return(we)
}

# Helper function to generate random uniform embeddings
rand_unif_df <- function(rows, cols){
  m <- array(0, c(rows, cols))
  # Fill matrix with random values in Keras default range
  for(i in 1:rows) m[i,] <- runif(cols, min = -0.05, max = 0.05)
  return(as.data.frame(m))
}

# Set embedding dimension
EMBEDDING_DIM = 200

# Create vocabulary dataframe from tokenizer word index
vocab <- data.frame(idx = unlist(word_index)) %>% 
  tibble::rownames_to_column(var="word")

# Find words without pre-trained embeddings
unmatched_words <- vocab %>% 
  anti_join(w2v_df, by = "word") %>% 
  select(word)

# Set seed for reproducible random embeddings
set.seed(123)

# Create complete embedding matrix
w2v_emb200 <- merge_vocab_wv2(vocab, wv = w2v_df)

cat("\nUnmatched words: ", nrow(unmatched_words), "\n")

# Add row of zeros for padding token (vocab_size = word_index + 1)
w2v_emb200 <- rbind(w2v_emb200, rep(0,200))

# Clean up large objects to free memory
rm(w2v_df, vocab, unmatched_words)

# Display final embedding matrix dimensions
dim(w2v_emb200)  # [vocab_size, embedding_dim]

toc()  # Display processing time (~3.5 seconds)
```

### build model

```{r}
# Create CNN model with pre-trained Word2Vec embeddings
model <- build_cnn(EMBEDDING_DIM = 200,        # Match embedding dimension
                   num_filters = 200,          # CNN filter count
                   maxlen = maxlen,            # Sequence length
                   vocab_size = vocab_size,    # Vocabulary size
                   pre_trained = T,            # Use pre-trained embeddings
                   embedding_matrix = w2v_emb200,  # Load Word2Vec weights
                   trainable_embed = T)        # Allow fine-tuning of embeddings

# Display model architecture
summary(model)
```

### train

# Training CNN with Word2Vec embeddings
# Same training procedure as CNNr but with pre-trained word representations

Consider running 10x. Save rocs and metrics from each. Take median auc with max F.

~ 80 secs for one cyle (if ~8k tokens)

```{r}
# Training configuration for CNNw
mdl_nm = "CNNw"        # Model name identifier
n_cyles = 10           # Number of training runs

# Training hyperparameters (same as CNNr for fair comparison)
lr <- .0004
epochs <- 30
batch_size <- 32

# Set learning rate
k_set_value(model$optimizer$lr, lr)

# Training callbacks
early_stopping <- callback_early_stopping(monitor = 'val_loss', patience = 7)
reduce_lr <- callback_reduce_lr_on_plateau(monitor = 'val_loss', factor=0.5, 
                                          patience = 1, min_lr = 0.0001,
                                          verbose = 0)

# Initialize result storage
roc_df_rows <- data.frame(matrix(y_test, nrow=1))  # Test labels in first row
metrics_df <- NULL

# Training loop
tic()
for(i in 1:n_cyles){
  # Generate file names for this run
  mdl_id <- paste0(mdl_nm, str_pad(i,2,pad = "0"))
  mdl_file_nm <- paste0("../models/CL07_model_", mdl_id, ".h5")
  hx_file_nm <- paste0("../models/CL07_model_", mdl_id, "_hx.rds")
  
  # Build model with Word2Vec embeddings
  model <- build_cnn(EMBEDDING_DIM = 200, num_filters = 200, 
                     maxlen = maxlen, vocab_size = vocab_size,
                     pre_trained = T,                    # Enable pre-trained embeddings
                     embedding_matrix = w2v_emb200,      # Load Word2Vec weights
                     trainable_embed = T)                # Allow fine-tuning
  
  # Train model
  t_begin=Sys.time()
  history <- model %>%
    keras::fit(
      x_train, y_train,
      batch_size = batch_size,
      epochs = epochs,
      callbacks = list(early_stopping, reduce_lr),
      verbose=0,
      validation_split = 0.15
    )
  
  # Calculate training metrics
  t_secs=as.numeric(difftime(Sys.time(), t_begin, units="secs"))
  eps=length(history$metrics[[1]])
  
  # Generate and store predictions
  pred <- model %>% predict(x_test)
  p <- pred[,1]
  roc_df_rows <- rbind(roc_df_rows, p)
  
  # Collect performance metrics
  metrics_df %<>% summary_metrics_dl(model_name = mdl_nm, i=i,
                                     test_labels = y_test, 
                                     pred_prob = p,
                                     elapsed_time = t_secs,
                                     epochs = eps)

  # Save model and history
  model %>% save_model_hdf5(mdl_file_nm)
  history %>% saveRDS(hx_file_nm)

  cat("i:",i)  # Progress indicator
}
cat("\n")
toc()
```

```{r}
# Display CNNw training metrics
metrics_df
```

### save local metrics

```{r}
# Save CNNw results with timestamp
date_sfx <- format(Sys.time(), "_%Y-%m-%d")

saveRDS(roc_df_rows, paste0("../results/roc_df_rows_", mdl_nm, "_", institution, date_sfx, ".rds"))
saveRDS(metrics_df, paste0("../results/metrics_df_", mdl_nm, "_", institution, date_sfx, ".rds"))
```

```{r}
# Plot training history
plot(history)
```

### evaluate

Get median.

```{r}
# Select best CNNw model using same approach as CNNr
med_auc=median(metrics_df$auc)
med_mdl=metrics_df %>% 
  mutate(auc_d=abs(auc-med_auc)) %>% 
  filter(auc_d==min(auc_d)) %>% 
  filter(F1==max(F1))

med_i=med_mdl$i[1]
# Extract predictions from selected model (+1 for label row)
p = roc_df_rows[med_i+1,] %>% unlist() %>% unname()
```

### Summary Metrics

```{r}
# Add CNNw predictions to results
roc_df <- bind_cols(roc_df, !!mdl_nm:= p)

# Calculate summary metrics
summary_df %<>% summary_metrics(mdl_nm, test_labels = y_test, pred_prob = p)
```

### Save check point

```{r}
# Save final checkpoint with all model results
save(institution, summary_df, roc_df, file = "tmp_summary_metrics2.rda")
```

### Save predictions

```{r}
# Create comprehensive predictions dataset
# Includes patient IDs, true labels, and predictions from all models
predictions_df=notes_test %>% 
  select(DE_ID, label_icd=label) %>%  # Patient ID and true label
  bind_cols(roc_df)                   # Add all model predictions
```

```{r}
# Save final predictions with timestamp
date_sfx <- format(Sys.time(), "_%Y-%m-%d")
saveRDS(predictions_df, paste0("../results/predictions_df_", date_sfx, ".rds"))
```

### Clear progress markers

```{r}
# Clean up temporary files
file.remove(dir(pattern = "tmp_progress_marker"))
```

# Summary DF ========

# Final Results and Visualization
# This section creates summary tables and ROC plots comparing all models

```{r}
# Display final performance summary for all models
# Shows AUC, accuracy, precision, recall, F1-score for RF, SVM, CNNr, CNNw
summary_df
```

Save to excel

```{r}
# Export results to Excel file for sharing and further analysis
date_sfx <- format(Sys.time(), "_%Y-%m-%d")
fn <- paste0("../results/Summary_Metrics_ADRD01",date_sfx,".xlsx")

writexl::write_xlsx(summary_df, fn)
```

# Plot ROC ========

# ROC Curve Visualization
# Compare performance of all models using ROC curves

```{r}
# Create ROC curves for all models
n <- ncol(roc_df)  # Number of models + label column

# Function to create ROC objects
roc2 <- function(X, response) roc(X, response = response, levels=c(0,1), direction="<")

# Generate ROC objects for each model (columns 2 to n, skipping Label column)
ROC <- lapply(roc_df[,2:n], roc2, response = roc_df$Label)

# Create legend text with AUC values
nms <- names(ROC)
leg_concat <- function(X, nm) paste0(nm, " (AUC=", format(X$auc, digits = 3, nsmall = 3), ")")
leg_txt = mapply(leg_concat, ROC, nm = nms[seq_along(ROC)])
leg_txt_placeholder = rep(" ", length(leg_txt))  # Placeholder for positioning

# Create ROC plot
par(pty = "s")  # Square plotting region
plot(ROC[[1]], col=1)  # Plot first ROC curve

# Add remaining ROC curves in different colors
x <- mapply(lines, ROC, col=seq_along(ROC))

# Add legend with AUC values
temp <- legend("bottomright", leg_txt_placeholder, 
               text.width = .45,     # Adjust width as needed
               lty = 1, xjust = 1, lwd = 2, col=1:n)

# Add AUC text to legend
text(temp$rect$left + temp$rect$w, temp$text$y,
     leg_txt, pos = 2)
```

```{r}
# Optional: Save ROC plot as PNG
# dev.copy(png, '../results/AUC_plot.png')
# dev.off()
```

# Plot ROC Fancy

# Enhanced ROC Visualization using ggplot2
# Creates publication-quality ROC curves with better formatting

See: file:///D:/Projects/MUSC/Grants/R01_Sdetect/R56_app/data_USF/roc_rds/rds_explor_jo/rds_final.nb.html

```{r}
# Function to create enhanced ROC plot with ggplot2
ggplot_roc_df <- function(roc_df, zoom=FALSE){
  n <- ncol(roc_df)  # Number of models + label
  
  # Define colors for each model
  mod_cols <- c("black","red","Lime Green","blue") #"green","orange","purple","brown","pink","grey","yellow")
  mod_nms <- names(roc_df)[2:n]  # Model names (skip Label column)
  mod_nms_auc <- as.character()  # Will store model names with AUC values

  # Create ROC objects for each model
  ROC <- lapply(roc_df[,2:n], roc, response = roc_df$Label, 
                levels = c(0,1), direction="<")
  
  # Initialize empty dataframe for plotting
  df <- data.frame(x=as.integer(), y=as.integer(), mdl=as.character(), stringsAsFactors = F)
  
  # Create model names with AUC values for legend
  for(i in 1:(n-1)){ 
    mod_nms_auc <- c(mod_nms_auc, paste0(mod_nms[i]," (AUC=",format(ROC[[i]]$auc, digits=3,nsmall=3),")"))
  }
  
  # Fill dataframe with ROC curve data
  for(i in 1:(n-1)){ 
    df %<>% bind_rows(data.frame(x=ROC[[i]]$specificities, y=ROC[[i]]$sensitivities, 
                                 mdl=paste0(mod_nms[i]," (AUC=",format(ROC[[i]]$auc, digits=3,nsmall=3),")"), 
                                 stringsAsFactors = F))
  }
  
  # Sort data to avoid plotting artifacts
  df %<>% arrange(mdl, desc(x), y) %>% 
    mutate(mdl=factor(mdl,levels=mod_nms_auc))

  # Create base ggplot
  g1 <- ggplot(df, aes(x=x, y=y, group=mdl, color=mdl)) +
    geom_line(linewidth=0.8) +                      # ROC curves
    scale_color_manual(values=mod_cols) +           # Custom colors
    labs(x="Specificity", y="Sensitivity", col="") # Axis labels
  
  # Conditional formatting for zoomed or full view
  if(zoom){
    # Zoomed view focusing on high-performance region
    g1 <- g1+
      scale_y_continuous(breaks=seq(0,1, 0.1)) +
      scale_x_reverse(breaks=seq(0,1, 0.1)) +
      coord_cartesian(ylim=c(0.5,1), xlim=c(1, 0.5)) 
  }else{
    # Full ROC space with diagonal reference line
    g1 <- g1+
      geom_abline(intercept = 1, slope = 1, col = "grey") +  # Random classifier line
      scale_y_continuous(breaks=seq(0,1, 0.2)) +
      scale_x_reverse(breaks=seq(0,1, 0.2)) 
  }
  
  # Apply consistent theme and formatting
  g1 <- g1+
    theme_classic()+
    theme(panel.border = element_rect(color = "black", fill=NA),
          legend.justification=c(1,0),               # Position legend
          legend.title=element_blank(),              # No legend title
          legend.text=element_text(size=rel(1.4), hjust=1), 
          legend.background = element_blank(),       # Transparent legend background
          legend.key.width = unit(rel(2), "line"),   # Legend line width
          legend.box.background = element_rect(color = "black")) + # Legend border
    theme(axis.text=element_text(size = rel(1.4)),           # Axis text size
          axis.title=element_text(size = rel(1.6),face="bold"), # Axis title formatting
          aspect.ratio = 1) +                                # Square aspect ratio
    theme(legend.position = "inside", legend.position.inside = c(1.0,0))  # Legend positioning
  
  return(g1)
}
```

```{r}
# Create enhanced ROC plot
ggplot_roc_df(roc_df)
```

## Save ROC

```{r}
# Save ROC plot as high-resolution PNG
date_sfx <- format(Sys.time(), "_%Y-%m-%d")
fn <- paste0("../results/AUC_ADRD01",date_sfx,".png")

ggsave(fn, plot = last_plot(),
       scale = 1, width = 6, height = 6, units ="in",
       dpi = 300)  # Publication quality resolution
```

## AUC zoom

```{r}
# Create zoomed ROC plot focusing on high-performance region
ggplot_roc_df(roc_df, zoom = T)
```

```{r}
# Save zoomed ROC plot
date_sfx <- format(Sys.time(), "_%Y-%m-%d")
fn <- paste0("../results/AUC_ADRD01",date_sfx,"_zoom.png")

ggsave(fn, plot = last_plot(),
       scale = 1, width = 6, height = 6, units ="in",
       dpi = 300)
```

# ====END====

# PIPELINE SUMMARY
# This pipeline successfully implements a comprehensive ADRD classification system:
#
# 1. Data Loading: Loads clinical notes and patient splits
# 2. Traditional ML: Random Forest and SVM with TF-IDF features
# 3. Deep Learning: CNN with random and Word2Vec embeddings  
# 4. Evaluation: Statistical robustness through multiple runs
# 5. Visualization: ROC curves and performance metrics
#
# Key outputs:
# - Performance metrics for all models
# - ROC curve visualizations
# - Saved trained models for future use
# - Comprehensive predictions dataset
#
# Expected performance: CNNw (Word2Vec) typically performs best,
# followed by CNNr, SVM, and Random Forest.
