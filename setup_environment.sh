#!/bin/bash
# Setup script for ADRD ePhenotyping Pipeline on H100 cluster
# No sudo access required - all installations in user space
# Follows TensorFlow pip installation guide for GPU support
# UPDATED: Complete package list for ADRD ePhenotyping project
# Maintains Keras 2 (legacy) support
# VERSION: 2025-11-06 - Added LIME and statistical testing packages
set -e # Exit on error
echo "=========================================="
echo "ADRD ePhenotyping Pipeline Environment Setup"
echo "Project: adrd_ephenotyping"
echo "Version: 2.0 (with LIME & Statistical Testing)"
echo "=========================================="
# ============================================
# 1. ACTIVATE MINICONDA
# ============================================
echo ""
echo "[Step 1/7] Activating Miniconda..."
source ~/miniconda3/bin/activate
# Verify conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: Conda not found. Please ensure Miniconda is installed."
    exit 1
fi
echo "✓ Conda version: $(conda --version)"
# Verify NVIDIA GPU driver
echo "Checking NVIDIA driver..."
nvidia-smi || { echo "ERROR: NVIDIA driver not found. Ensure H100 drivers are installed."; exit 1; }
# ============================================
# 2. CREATE CONDA ENVIRONMENT
# ============================================
echo ""
echo "[Step 2/7] Creating conda environment 'adrd-pipeline'..."
# Remove existing environment if it exists
conda env remove -n adrd-pipeline -y 2>/dev/null || true
# Create environment with Python 3.9, R 4.3, and essential packages
conda create -n adrd-pipeline -y \
    python=3.9 \
    r-base=4.3 \
    r-essentials \
    r-devtools \
    r-tidyverse \
    r-data.table
echo "✓ Environment created successfully"
# ============================================
# 3. ACTIVATE ENVIRONMENT
# ============================================
echo ""
echo "[Step 3/7] Activating adrd-pipeline environment..."
conda activate adrd-pipeline
# ============================================
# 4. INSTALL PYTHON PACKAGES
# ============================================
echo ""
echo "[Step 4/7] Installing Python packages..."
# Upgrade pip
pip install --upgrade pip setuptools wheel
echo ""
echo "Installing TensorFlow with CUDA support..."
pip install tensorflow[and-cuda]==2.16.1
echo ""
echo "Installing tf-keras for Keras 2 compatibility..."
pip install tf-keras==2.16.0
echo ""
echo "Creating symbolic links for NVIDIA libraries..."
TF_DIR=$(python -c 'import tensorflow; print(tensorflow.__file__)' 2>/dev/null)
if [ -z "$TF_DIR" ]; then
    echo "ERROR: TensorFlow not found. Installation may have failed."
    exit 1
fi
pushd $(dirname "$TF_DIR") || { echo "ERROR: Failed to change to TensorFlow directory."; exit 1; }
ln -svf ../nvidia/*/lib/*.so* . || echo "Warning: Some NVIDIA library links may have failed."
popd
echo "Creating symbolic link for ptxas..."
NVCC_DIR=$(python -c "import nvidia.cuda_nvcc; print(nvidia.cuda_nvcc.__file__)" 2>/dev/null)
if [ -z "$NVCC_DIR" ]; then
    echo "Warning: nvidia.cuda_nvcc not found. Skipping ptxas link."
else
    PTXAS_PATH=$(find $(dirname $(dirname "$NVCC_DIR"))/*/bin/ -name ptxas -print -quit 2>/dev/null)
    if [ -n "$PTXAS_PATH" ]; then
        ln -sf "$PTXAS_PATH" "$CONDA_PREFIX/bin/ptxas" || echo "Warning: Failed to create ptxas link."
    else
        echo "Warning: ptxas not found. GPU functionality may be limited."
    fi
fi
echo ""
echo "Installing core Python packages..."
pip install \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scikit-learn==1.3.0
echo ""
echo "Installing visualization packages..."
pip install \
    matplotlib==3.7.2 \
    seaborn==0.12.2
echo ""
echo "Installing additional utilities..."
pip install \
    tqdm \
    h5py
echo ""
echo "Installing TensorRT (optional, for optimization)..."
pip install --extra-index-url https://pypi.ngc.nvidia.com tensorrt-cu12 || echo "Warning: TensorRT installation failed (optional)"
echo "✓ Python packages installed"
echo ""
echo "Configuring Keras 2 (legacy) support..."
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh" << 'EOF'
#!/bin/bash
export TF_USE_LEGACY_KERAS=1
EOF
chmod +x "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh" << 'EOF'
#!/bin/bash
unset TF_USE_LEGACY_KERAS
EOF
chmod +x "$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh"
export TF_USE_LEGACY_KERAS=1
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', tf.config.list_physical_devices('GPU'))"
echo ""
python -c "import tf_keras; print('tf-keras version:', tf_keras.__version__)"
echo ""
echo "[Step 5/7] Installing R packages..."
cat > /tmp/install_r_packages.R << 'EOF'
options(repos = c(CRAN = "https://cloud.r-project.org/"))
install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(paste0("Installing ", pkg, "...\n"))
    install.packages(pkg, dependencies = TRUE)
  } else {
    cat(paste0("✓ ", pkg, " already installed\n"))
  }
}
cat("\n=== Installing Core tidyverse Packages ===\n")
core_packages <- c("dplyr","tidyr","magrittr","stringr","tibble","readr","purrr","tidyverse")
sapply(core_packages, install_if_missing)
cat("\n=== Installing Visualization Packages ===\n")
viz_packages <- c("ggplot2","scales","gridExtra")
sapply(viz_packages, install_if_missing)
cat("\n=== Installing Statistical & ML Packages ===\n")
stat_packages <- c("pROC","broom")
sapply(stat_packages, install_if_missing)
cat("\n=== Installing Utility Packages ===\n")
util_packages <- c("janitor","tictoc","writexl")
sapply(util_packages, install_if_missing)
cat("\n=== Installing Deep Learning Packages ===\n")
install_if_missing("reticulate")
library(reticulate)
use_condaenv("adrd-pipeline", required = TRUE)
Sys.setenv(TF_USE_LEGACY_KERAS = "1")
py_config()
install_if_missing("keras")
install_if_missing("tensorflow")
library(keras)
library(tensorflow)
tryCatch({
  tf_keras <- reticulate::import("tf_keras")
  cat("✓ tf_keras version:", tf_keras$`__version__`, "\n")
  cat("✓ Keras 2 (legacy) is available\n")
}, error = function(e) {
  cat("⚠ Warning: tf_keras import failed\n")
  cat("Error message:", conditionMessage(e), "\n")
})
cat("\nTensorFlow version:", as.character(tf$version$VERSION), "\n")
gpus <- tf$config$list_physical_devices('GPU')
if (length(gpus) > 0) {
  cat("✓ GPU(s) detected:\n")
  for (gpu in gpus) cat(" -", gpu$name, "\n")
} else {
  cat("⚠ No GPU detected - will use CPU\n")
}
# ===========================
# TEXT ANALYSIS PACKAGES
# ===========================
cat("\n=== Installing Text Analysis Packages ===\n")
install_if_missing("RColorBrewer") # For color palettes
install_if_missing("ggrepel") # For better plot labels
install_if_missing("tidytext") # For text analysis
install_if_missing("quanteda") # For text processing
install_if_missing("quanteda.textstats") # For text statistics
install_if_missing("quanteda.textplots") # For text plots
install_if_missing("tm") # For text mining
install_if_missing("wordcloud") # For word clouds
# ===========================
# EXPLAINABILITY PACKAGES (NEW)
# ===========================
cat("\n=== Installing Explainability Packages ===\n")
install_if_missing("lime") # For LIME explainability (Aim 2)
cat("\n=== Installation Complete ===\n")
EOF
Rscript /tmp/install_r_packages.R
rm /tmp/install_r_packages.R

# ============================================
# 6. INSTALL JUPYTER AND R KERNEL
# ============================================
echo ""
echo "[Step 6/7] Installing Jupyter and R kernel..."
pip install jupyter notebook jupyterlab
R -e "install.packages('IRkernel', repos='https://cloud.r-project.org/'); IRkernel::installspec(user = TRUE)"

echo "✓ Jupyter and R kernel installed"

# ============================================
# 7. CONFIGURE R ENVIRONMENT
# ============================================
echo ""
echo "[Step 7/7] Configuring R environment..."

# Add configuration to .Rprofile
cat >> ~/.Rprofile << EOF

# ========================================
# ADRD ePhenotyping Pipeline Configuration
# ========================================
# Automatically set Python path and Keras 2 support
Sys.setenv(RETICULATE_PYTHON="$(which python)")
Sys.setenv(TF_USE_LEGACY_KERAS="1")

# Optional: Suppress dplyr summarise messages
options(dplyr.summarise.inform = FALSE)
EOF

echo "✓ .Rprofile configured"

# ============================================
# 8. SAVE ENVIRONMENT CONFIGURATION
# ============================================
echo ""
echo "=========================================="
echo "Saving environment configuration..."
echo "=========================================="

# Export conda environment
conda env export > adrd_environment.yml
echo "✓ Environment saved to adrd_environment.yml"

# Create quick reference document
cat > ENVIRONMENT_INFO.md << 'EOF'
# ADRD ePhenotyping Pipeline - Environment Information

## Python Packages (via pip)

### Deep Learning
- tensorflow[and-cuda]==2.16.1
- tf-keras==2.16.0 (Keras 2 legacy support)

### Scientific Computing
- numpy==1.24.3
- pandas==2.0.3
- scikit-learn==1.3.0

### Visualization
- matplotlib==3.7.2
- seaborn==0.12.2

### Utilities
- tqdm
- h5py

### Optional
- tensorrt-cu12 (for optimization)

## R Packages (via CRAN)

### Core Data Manipulation
- tidyverse (meta-package including:)
  - dplyr (data manipulation)
  - tidyr (data tidying)
  - stringr (string operations)
  - readr (fast CSV reading)
  - purrr (functional programming)
  - tibble (modern data frames)
- magrittr (pipe operators)

### Visualization
- ggplot2 (graphics)
- scales (scale functions)
- gridExtra (multiple plots)

### Statistical Analysis
- pROC (ROC curve analysis - CRITICAL)
- broom (tidy statistical outputs)

### Utilities
- janitor (data cleaning)
- tictoc (timing)
- writexl (Excel export)

### Deep Learning
- reticulate (R-Python interface)
- keras (Keras 2 via tf-keras)
- tensorflow (TensorFlow interface)

### Text Analysis (Aim 2)
- quanteda (text processing)
- quanteda.textstats (text statistics)
- quanteda.textplots (text plots)
- tidytext (tidy text mining)
- tm (text mining framework)
- wordcloud (word clouds)
- RColorBrewer (color palettes)
- ggrepel (better labels)

### Explainability (Aim 2 - NEW)
- lime (Local Interpretable Model-agnostic Explanations)

## Environment Variables

### Keras 2 Support
```bash
export TF_USE_LEGACY_KERAS=1
```
This is automatically set when activating the environment.

### Python Path for R
```r
Sys.setenv(RETICULATE_PYTHON="/path/to/conda/envs/adrd-pipeline/bin/python")
```
This is automatically set in .Rprofile

## GPU Configuration

### Check GPU in Python
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### Check GPU in R
```r
library(tensorflow)
tf$config$list_physical_devices('GPU')
```

## Package Verification

### Verify Keras 2 in Python
```python
import tf_keras
print(tf_keras.__version__)  # Should be 2.16.0
```

### Verify Keras 2 in R
```r
Sys.setenv(TF_USE_LEGACY_KERAS = "1")
library(keras)
keras:::keras_version()  # Should show Keras 2.x
```

## Activation

### Quick activation
```bash
source ~/miniconda3/bin/activate
conda activate adrd-pipeline
```

### Verify environment
```bash
echo $TF_USE_LEGACY_KERAS  # Should output: 1
which python               # Should point to conda env
which R                    # Should point to conda env
```

## Notes

- Keras 2 (legacy) is maintained via tf-keras package
- TF_USE_LEGACY_KERAS=1 must be set for Keras 2 to work
- All R scripts automatically get this environment variable via .Rprofile
- GPU support is enabled by default with TensorFlow 2.16.1
- LIME package added for Aim 2 explainability analysis
EOF

echo "✓ Environment info saved to ENVIRONMENT_INFO.md"

# Create activation script for easy future use
cat > activate_adrd.sh << 'EOF'
#!/bin/bash
# Quick activation script for ADRD ePhenotyping pipeline

source ~/miniconda3/bin/activate
conda activate adrd-pipeline

echo "=========================================="
echo "ADRD ePhenotyping Pipeline Environment"
echo "=========================================="
echo ""
echo "Python: $(which python)"
echo "  TensorFlow: $(python -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || echo 'Not loaded')"
echo "  tf-keras: $(python -c 'import tf_keras; print(tf_keras.__version__)' 2>/dev/null || echo 'Not loaded')"
echo ""
echo "R: $(which R)"
echo ""
echo "Environment Variables:"
echo "  TF_USE_LEGACY_KERAS: ${TF_USE_LEGACY_KERAS:-not set}"
echo ""
echo "=========================================="
echo "Quick Tests:"
echo "=========================================="
echo ""
echo "Test GPU in Python:"
echo "  python -c 'import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))'"
echo ""
echo "Test Keras 2 in Python:"
echo "  python -c 'import tf_keras; print(tf_keras.__version__)'"
echo ""
echo "Test GPU in R:"
echo "  R -e 'library(tensorflow); tf\$config\$list_physical_devices(\"GPU\")'"
echo ""
echo "Test Keras 2 in R:"
echo "  R -e 'Sys.setenv(TF_USE_LEGACY_KERAS=\"1\"); library(keras); keras:::keras_version()'"
echo ""
echo "=========================================="
echo "Start Working:"
echo "=========================================="
echo ""
echo "Run pipeline scripts (with Jihad's trained models):"
echo "  Rscript 03_evaluate_models.R"
echo "  Rscript 04_demographic_analysis.R"
echo "  Rscript 05_aim2_feature_analysis.R"
echo ""
echo "Or train your own models:"
echo "  Rscript 01_prepare_data.R"
echo "  Rscript 02_train_cnnr.R"
echo "  Rscript 03_evaluate_models.R"
echo "  Rscript 04_demographic_analysis.R"
echo "  Rscript 05_aim2_feature_analysis.R"
echo ""
echo "Or start Jupyter:"
echo "  jupyter notebook --no-browser --port=8888"
echo ""
EOF

chmod +x activate_adrd.sh
echo "✓ Activation script saved to activate_adrd.sh"

# ============================================
# 9. FINAL VERIFICATION
# ============================================
echo ""
echo "=========================================="
echo "Running Final Verification..."
echo "=========================================="

echo ""
echo "Testing Python TensorFlow + GPU:"
python << 'PYEOF'
import tensorflow as tf
print(f"  TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"  GPUs available: {len(gpus)}")
for gpu in gpus:
    print(f"    - {gpu.name}")
PYEOF

echo ""
echo "Testing Python tf-keras:"
python << 'PYEOF'
import tf_keras
print(f"  tf-keras version: {tf_keras.__version__}")
print("  ✓ Keras 2 (legacy) is available")
PYEOF

echo ""
echo "Testing R TensorFlow:"
R --quiet --no-save << 'REOF'
Sys.setenv(TF_USE_LEGACY_KERAS = "1")
library(tensorflow)
cat("  TensorFlow version:", as.character(tf$version$VERSION), "\n")
gpus <- tf$config$list_physical_devices('GPU')
cat("  GPUs available:", length(gpus), "\n")
REOF

# ============================================
# 10. COMPLETION MESSAGE
# ============================================
echo ""
echo "=========================================="
echo "✓ SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Environment: adrd-pipeline"
echo "Location: $(conda env list | grep adrd-pipeline | awk '{print $2}')"
echo ""
echo "=========================================="
echo "Installed Packages:"
echo "=========================================="
echo ""
echo "Python:"
echo "  • TensorFlow 2.16.1 (with GPU support)"
echo "  • tf-keras 2.16.0 (Keras 2 legacy)"
echo "  • NumPy, Pandas, scikit-learn"
echo "  • Matplotlib, Seaborn"
echo ""
echo "R:"
echo "  • tidyverse (dplyr, tidyr, ggplot2, etc.)"
echo "  • keras, tensorflow (with Keras 2 support)"
echo "  • pROC (for ROC analysis)"
echo "  • quanteda, tidytext, tm (for text analysis)"
echo "  • lime (for explainability - Aim 2)"
echo "  • writexl, janitor, tictoc"
echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Activate environment:"
echo "   ./activate_adrd.sh"
echo "   # OR"
echo "   conda activate adrd-pipeline"
echo ""
echo "2. Verify installation:"
echo "   python -c 'import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))'"
echo "   R -e 'library(keras); keras:::keras_version()'"
echo ""
echo "3. Copy Jihad's trained models (if using pre-trained):"
echo "   mkdir -p models"
echo "   cp /path/to/jihad/CL07_* models/"
echo "   # Models will be auto-detected!"
echo ""
echo "4. Place your data file:"
echo "   mkdir -p data/raw"
echo "   # Copy ptHx_sample_v2025-10-25.csv to data/raw/"
echo ""
echo "5. Run pipeline (with Jihad's models - NO training needed):"
echo "   Rscript 03_evaluate_models.R"
echo "   Rscript 04_demographic_analysis.R"
echo "   Rscript 05_aim2_feature_analysis.R"
echo ""
echo "   OR train your own models:"
echo "   Rscript 01_prepare_data.R"
echo "   Rscript 02_train_cnnr.R"
echo "   Rscript 03_evaluate_models.R"
echo "   Rscript 04_demographic_analysis.R"
echo "   Rscript 05_aim2_feature_analysis.R"
echo ""
echo "=========================================="
echo "Important Files Created:"
echo "=========================================="
echo "  • adrd_environment.yml - Full environment spec"
echo "  • ENVIRONMENT_INFO.md - Package documentation"
echo "  • activate_adrd.sh - Quick activation script"
echo ""
echo "=========================================="
echo "CRITICAL: Keras 2 Configuration"
echo "=========================================="
echo "  TF_USE_LEGACY_KERAS=1 is automatically set"
echo "  This ensures Keras 2 (not Keras 3) is used"
echo "  No manual configuration needed!"
echo "=========================================="
echo ""
echo "=========================================="
echo "NEW FEATURES (v2.0):"
echo "=========================================="
echo "  ✓ Statistical significance testing"
echo "  ✓ LIME explainability (lime package)"
echo "  ✓ Behavioral testing framework"
echo "  ✓ Model compatibility (auto-detects Jihad's models)"
echo "  ✓ Utility scripts (utils_model_loader.R, utils_statistical_tests.R)"
echo ""
echo "For more information, see:"
echo "  - PROPOSAL_ANALYSIS_AND_ROADMAP.md"
echo "  - STATISTICAL_SIGNIFICANCE_METHODOLOGY.md"
echo "  - COLUMN_NAMES_REFERENCE.md"
echo "  - IMPLEMENTATION_SUMMARY.md"
echo "=========================================="
echo ""
