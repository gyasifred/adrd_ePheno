# ============================================================================
# AIM 2 DEMOGRAPHIC FAIRNESS ENHANCEMENTS
# ============================================================================
# Author: Gyasi, Frederick
# Purpose: Add demographic stratification to feature analysis (Aim 2)
#
# This code adds demographic fairness analysis to 05_aim2_feature_analysis.R
# to investigate whether discriminative features differ by demographic groups
# ============================================================================

# ==============================================================================
# PART 1B: DEMOGRAPHIC-STRATIFIED CHI-SQUARED ANALYSIS
# ==============================================================================
# INSERT AFTER: PART 1 Chi-Squared Analysis (around line 260)
# PURPOSE: Identify if discriminative terms differ by demographic subgroups

cat(strrep("=", 80) %+% "\n")
cat("PART 1B: Demographic-Stratified Chi-Squared Analysis\n")
cat(strrep("=", 80) %+% "\n\n")

cat("Investigating if discriminative features differ by demographics...\n\n")

# Check for demographic variables in full_corpus
demo_vars_available <- c()
for (demo in c("GENDER", "RACE", "HISPANIC")) {
  if (demo %in% names(full_corpus)) {
    demo_vars_available <- c(demo_vars_available, demo)
    cat("  Found demographic variable:", demo, "\n")
  }
}

if (length(demo_vars_available) == 0) {
  cat("\n⚠️  No demographic variables found in corpus.\n")
  cat("   Demographic feature analysis will be skipped.\n\n")
} else {
  cat("\nDemographic variables available:", paste(demo_vars_available, collapse = ", "), "\n\n")

  # Storage for demographic-stratified results
  demo_chi2_results <- list()

  # Analyze for each demographic variable
  for (demo_var in demo_vars_available) {

    cat(strrep("-", 80) %+% "\n")
    cat("Analyzing:", demo_var, "\n")
    cat(strrep("-", 80) %+% "\n\n")

    # Get unique demographic values (exclude NA, empty, UNKNOWN)
    demo_values <- full_corpus %>%
      filter(!is.na(.data[[demo_var]]),
             .data[[demo_var]] != "",
             .data[[demo_var]] != "UNKNOWN") %>%
      pull(.data[[demo_var]]) %>%
      unique()

    # Normalize gender values if needed
    if (demo_var == "GENDER") {
      full_corpus <- full_corpus %>%
        mutate(GENDER = case_when(
          toupper(GENDER) %in% c("FEMALE", "F") ~ "Female",
          toupper(GENDER) %in% c("MALE", "M") ~ "Male",
          TRUE ~ GENDER
        ))
      demo_values <- c("Female", "Male")
    }

    cat("Subgroups:", paste(demo_values, collapse = ", "), "\n\n")

    # For each demographic subgroup, calculate chi-squared terms
    demo_var_results <- list()

    for (demo_val in demo_values) {

      # Filter corpus to this demographic subgroup
      subgroup_corpus <- full_corpus %>%
        filter(.data[[demo_var]] == demo_val)

      # Check minimum sample sizes
      n_adrd <- sum(subgroup_corpus$label == 1, na.rm = TRUE)
      n_ctrl <- sum(subgroup_corpus$label == 0, na.rm = TRUE)

      if (n_adrd < 20 || n_ctrl < 20) {
        cat("  Skipping", demo_val, "(insufficient samples: ADRD=", n_adrd, ", CTRL=", n_ctrl, ")\n")
        next
      }

      cat("  Analyzing", demo_val, "(N =", nrow(subgroup_corpus),
          ": ADRD=", n_adrd, ", CTRL=", n_ctrl, ")\n")

      # Create corpus for this subgroup
      sub_corpus <- corpus(subgroup_corpus, text_field = "txt", docid_field = "DE_ID")
      docvars(sub_corpus, "label_str") <- ifelse(subgroup_corpus$label == 1, "ADRD", "CTRL")

      # Tokenize and create DFM
      sub_tokens <- tokens(sub_corpus,
                           remove_punct = TRUE,
                           remove_symbols = TRUE,
                           remove_numbers = TRUE) %>%
        tokens_tolower() %>%
        tokens_remove(pattern = stopwords("en"))

      sub_dfm <- dfm(sub_tokens) %>%
        dfm_trim(min_termfreq = 5, min_docfreq = 3)

      # Group by label
      sub_dfm_grouped <- dfm_group(sub_dfm, groups = label_str)

      # Calculate chi-squared
      tryCatch({
        chi2_sub <- textstat_keyness(sub_dfm_grouped,
                                     target = "ADRD",
                                     measure = "chi2")

        # Get top 20 terms
        top_terms <- chi2_sub %>%
          arrange(desc(chi2)) %>%
          head(20) %>%
          mutate(
            demographic = demo_var,
            subgroup = demo_val,
            p_value = pchisq(chi2, df = 1, lower.tail = FALSE)
          )

        demo_var_results[[demo_val]] <- top_terms

        cat("    Top 5 discriminative terms:",
            paste(head(top_terms$feature, 5), collapse = ", "), "\n")

      }, error = function(e) {
        cat("    Error in chi-squared calculation:", e$message, "\n")
      })
    }

    # Store results for this demographic variable
    if (length(demo_var_results) > 0) {
      demo_chi2_results[[demo_var]] <- demo_var_results

      # Compare top terms across subgroups
      if (length(demo_var_results) >= 2) {
        cat("\n  Comparing discriminative terms across", demo_var, "subgroups:\n")

        # Get top 10 terms for each subgroup
        all_top_terms <- lapply(names(demo_var_results), function(sg) {
          demo_var_results[[sg]] %>%
            head(10) %>%
            pull(feature)
        })
        names(all_top_terms) <- names(demo_var_results)

        # Calculate overlap
        if (length(all_top_terms) == 2) {
          overlap <- intersect(all_top_terms[[1]], all_top_terms[[2]])
          cat("    Common terms (top 10):", length(overlap), "/", 10, "\n")
          if (length(overlap) > 0) {
            cat("      ", paste(overlap, collapse = ", "), "\n")
          }

          # Unique to each group
          unique_1 <- setdiff(all_top_terms[[1]], all_top_terms[[2]])
          unique_2 <- setdiff(all_top_terms[[2]], all_top_terms[[1]])

          if (length(unique_1) > 0) {
            cat("    Unique to", names(all_top_terms)[1], ":",
                paste(unique_1, collapse = ", "), "\n")
          }
          if (length(unique_2) > 0) {
            cat("    Unique to", names(all_top_terms)[2], ":",
                paste(unique_2, collapse = ", "), "\n")
          }

          # Interpretation
          overlap_pct <- length(overlap) / 10 * 100
          if (overlap_pct < 50) {
            cat("\n    ⚠️  FINDING: Low term overlap (", sprintf("%.0f%%", overlap_pct),
                ") suggests different linguistic patterns by ", demo_var, "\n", sep = "")
          } else {
            cat("\n    ✓ Good term overlap (", sprintf("%.0f%%", overlap_pct),
                ") - consistent patterns across ", demo_var, "\n", sep = "")
          }
        }
      }
    }
    cat("\n")
  }

  # Save demographic-stratified chi-squared results
  if (length(demo_chi2_results) > 0) {
    saveRDS(demo_chi2_results,
            file.path(AIM2_RESULTS_DIR, "demographic_chi2_stratified.rds"))

    # Create comparison table
    comparison_df <- bind_rows(lapply(names(demo_chi2_results), function(demo) {
      bind_rows(lapply(names(demo_chi2_results[[demo]]), function(sg) {
        demo_chi2_results[[demo]][[sg]] %>%
          head(20) %>%
          select(feature, chi2, p_value) %>%
          mutate(demographic = demo, subgroup = sg)
      }))
    }))

    write_csv(comparison_df,
              file.path(AIM2_RESULTS_DIR, "demographic_chi2_comparison.csv"))

    cat("Demographic-stratified chi-squared results saved\n\n")
  }
}

# ==============================================================================
# PART 6C: DEMOGRAPHIC-STRATIFIED LIME ANALYSIS
# ==============================================================================
# INSERT AFTER: PART 6B LIME Explainability (around line 620)
# PURPOSE: Analyze if LIME explanations differ by demographics

if (!is.null(predictions) && requireNamespace("lime", quietly = TRUE) &&
    exists("demo_vars_available") && length(demo_vars_available) > 0) {

  cat(strrep("=", 80) %+% "\n")
  cat("PART 6C: Demographic-Stratified LIME Analysis\n")
  cat(strrep("=", 80) %+% "\n\n")

  cat("Analyzing if LIME feature importance differs by demographics...\n\n")

  # Ensure we have the model and explainer
  if (!exists("model") || !exists("text_model") || !exists("explainer")) {
    cat("⚠️  Model or explainer not available. Skipping demographic LIME analysis.\n\n")
  } else {

    # Merge predictions with test_set to get demographics
    test_data_with_demo <- test_set %>%
      left_join(predictions %>% select(DE_ID, Predicted_Probability, Label),
                by = "DE_ID")

    # For each demographic variable
    demo_lime_results <- list()

    for (demo_var in demo_vars_available) {

      if (!demo_var %in% names(test_data_with_demo)) next

      cat(strrep("-", 80) %+% "\n")
      cat("LIME Analysis by:", demo_var, "\n")
      cat(strrep("-", 80) %+% "\n\n")

      # Normalize gender if needed
      if (demo_var == "GENDER") {
        test_data_with_demo <- test_data_with_demo %>%
          mutate(GENDER = case_when(
            toupper(GENDER) %in% c("FEMALE", "F") ~ "Female",
            toupper(GENDER) %in% c("MALE", "M") ~ "Male",
            TRUE ~ GENDER
          ))
      }

      # Get unique subgroups
      demo_values <- test_data_with_demo %>%
        filter(!is.na(.data[[demo_var]]),
               .data[[demo_var]] != "",
               .data[[demo_var]] != "UNKNOWN") %>%
        pull(.data[[demo_var]]) %>%
        unique()

      demo_var_lime <- list()

      for (demo_val in demo_values) {

        # Select ADRD cases from this demographic subgroup
        subgroup_cases <- test_data_with_demo %>%
          filter(.data[[demo_var]] == demo_val, label == 1) %>%
          arrange(desc(Predicted_Probability)) %>%
          head(10)  # Top 10 confident ADRD predictions

        if (nrow(subgroup_cases) < 5) {
          cat("  Skipping", demo_val, "(insufficient ADRD cases)\n")
          next
        }

        cat("  Generating LIME explanations for", demo_val,
            "(", nrow(subgroup_cases), "cases)\n")

        # Generate LIME explanations
        tryCatch({
          explanations_sub <- explain(
            subgroup_cases$txt,
            explainer,
            n_labels = 1,
            n_features = 10,
            n_permutations = 500  # Reduced for speed
          )

          # Aggregate feature importance
          feature_importance <- explanations_sub %>%
            filter(label == "1") %>%  # ADRD class
            group_by(feature) %>%
            summarise(
              mean_weight = mean(feature_weight),
              median_weight = median(feature_weight),
              n_appearances = n(),
              .groups = "drop"
            ) %>%
            arrange(desc(abs(mean_weight))) %>%
            head(20) %>%
            mutate(
              demographic = demo_var,
              subgroup = demo_val
            )

          demo_var_lime[[demo_val]] <- feature_importance

          cat("    Top 5 important features:",
              paste(head(feature_importance$feature, 5), collapse = ", "), "\n")
          cat("    Mean weights:",
              paste(sprintf("%.3f", head(feature_importance$mean_weight, 5)), collapse = ", "), "\n")

        }, error = function(e) {
          cat("    Error generating LIME explanations:", e$message, "\n")
        })
      }

      # Store and compare
      if (length(demo_var_lime) > 0) {
        demo_lime_results[[demo_var]] <- demo_var_lime

        # Compare feature importance across subgroups
        if (length(demo_var_lime) >= 2) {
          cat("\n  Comparing LIME feature importance across", demo_var, ":\n")

          # Get top features for each subgroup
          top_features <- lapply(names(demo_var_lime), function(sg) {
            demo_var_lime[[sg]] %>%
              head(10) %>%
              pull(feature)
          })
          names(top_features) <- names(demo_var_lime)

          # Calculate overlap
          if (length(top_features) == 2) {
            overlap <- intersect(top_features[[1]], top_features[[2]])
            cat("    Common important features:", length(overlap), "/", 10, "\n")
            if (length(overlap) > 0) {
              cat("      ", paste(overlap, collapse = ", "), "\n")
            }

            overlap_pct <- length(overlap) / 10 * 100
            if (overlap_pct < 40) {
              cat("\n    ⚠️  FINDING: Low overlap (", sprintf("%.0f%%", overlap_pct),
                  ") - model uses different features for ", demo_var, " subgroups\n", sep = "")
              cat("    This may indicate differential prediction patterns by demographics\n")
            } else {
              cat("\n    ✓ Good overlap (", sprintf("%.0f%%", overlap_pct),
                  ") - consistent features across ", demo_var, "\n", sep = "")
            }
          }
        }
      }
      cat("\n")
    }

    # Save demographic LIME results
    if (length(demo_lime_results) > 0) {
      saveRDS(demo_lime_results,
              file.path(AIM2_RESULTS_DIR, "demographic_lime_stratified.rds"))

      # Create comparison table
      lime_comparison_df <- bind_rows(lapply(names(demo_lime_results), function(demo) {
        bind_rows(lapply(names(demo_lime_results[[demo]]), function(sg) {
          demo_lime_results[[demo]][[sg]]
        }))
      }))

      write_csv(lime_comparison_df,
                file.path(AIM2_RESULTS_DIR, "demographic_lime_comparison.csv"))

      cat("Demographic-stratified LIME results saved\n\n")
    }
  }
}

# ==============================================================================
# PART 8: DEMOGRAPHIC FEATURE FAIRNESS SUMMARY
# ==============================================================================
# INSERT BEFORE: Final completion message (at end of script)

cat(strrep("=", 80) %+% "\n")
cat("PART 8: Demographic Feature Fairness Summary\n")
cat(strrep("=", 80) %+% "\n\n")

if (exists("demo_chi2_results") && length(demo_chi2_results) > 0) {
  cat("DISCRIMINATIVE TERMS BY DEMOGRAPHICS\n")
  cat(strrep("-", 40), "\n")

  for (demo_var in names(demo_chi2_results)) {
    cat("\n", demo_var, ":\n", sep = "")
    for (sg in names(demo_chi2_results[[demo_var]])) {
      top_5 <- demo_chi2_results[[demo_var]][[sg]] %>%
        head(5) %>%
        pull(feature)
      cat("  ", sg, ": ", paste(top_5, collapse = ", "), "\n", sep = "")
    }
  }
  cat("\n")
}

if (exists("demo_lime_results") && length(demo_lime_results) > 0) {
  cat("LIME FEATURE IMPORTANCE BY DEMOGRAPHICS\n")
  cat(strrep("-", 40), "\n")

  for (demo_var in names(demo_lime_results)) {
    cat("\n", demo_var, ":\n", sep = "")
    for (sg in names(demo_lime_results[[demo_var]])) {
      top_5 <- demo_lime_results[[demo_var]][[sg]] %>%
        head(5) %>%
        pull(feature)
      cat("  ", sg, ": ", paste(top_5, collapse = ", "), "\n", sep = "")
    }
  }
  cat("\n")
}

cat("FAIRNESS ASSESSMENT:\n")
cat(strrep("-", 40), "\n")
cat("✓ Demographic-stratified chi-squared analysis complete\n")
cat("✓ Demographic-stratified LIME analysis complete\n")
cat("\nKEY QUESTIONS ANSWERED:\n")
cat("1. Do discriminative features differ by demographics? → See chi2 comparison\n")
cat("2. Do LIME explanations differ by demographics? → See LIME comparison\n")
cat("3. Are certain groups explained differently? → Check overlap percentages\n\n")

cat("OUTPUT FILES:\n")
cat("  - demographic_chi2_stratified.rds\n")
cat("  - demographic_chi2_comparison.csv\n")
cat("  - demographic_lime_stratified.rds\n")
cat("  - demographic_lime_comparison.csv\n\n")

cat("NEXT STEPS:\n")
cat("1. Review chi-squared term overlap across demographic groups\n")
cat("2. Examine LIME feature importance differences\n")
cat("3. If low overlap (<50%), investigate differential documentation patterns\n")
cat("4. Consider demographic-specific model recalibration if needed\n\n")
