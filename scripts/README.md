# Scripts Directory

This directory contains utility scripts for analysis, data export, and testing.

## Analysis Scripts

### `analyze_classifier_performance.py`
Analyzes the trained classifier's prediction patterns and distribution.

**Usage:**
```bash
python scripts/analyze_classifier_performance.py
```

**Outputs:**
- Prediction distribution by class
- Confidence statistics
- Temporal patterns in predictions
- Summary statistics printed to console

### `filter_fault_windows.py`
Filters prediction data to include only fault windows (excludes Normal class).

**Usage:**
```bash
python scripts/filter_fault_windows.py
```

**Inputs:**
- `outputs/exp_full_dataset/predictions_for_colab.parquet`

**Outputs:**
- `outputs/exp_full_dataset/predictions_faults_only.parquet`

### `filter_high_confidence_faults.py`
Filters prediction data to include only high-confidence fault windows.

**Usage:**
```bash
python scripts/filter_high_confidence_faults.py
```

**Configuration:**
- Default confidence threshold: 0.7
- Excludes Normal class predictions

**Outputs:**
- `outputs/exp_full_dataset/predictions_high_conf_faults.parquet`
- Summary statistics

## Data Export Scripts

### `export_for_colab.py`
Prepares trained model predictions and features for LLM explanation generation on Google Colab.

**Usage:**
```bash
python scripts/export_for_colab.py \
    --model outputs/my_experiment/model.pkl \
    --data data/raw/your_data.csv \
    --output outputs/my_experiment/predictions_for_colab.parquet
```

**What it does:**
1. Loads trained classifier
2. Processes raw CSV data
3. Generates predictions with confidence scores
4. Extracts features for each window
5. Exports to Parquet format for Colab

**Outputs:**
- Parquet file with: window_id, prediction, confidence, features

## Testing Scripts

### `verify_setup.py`
Verifies that all dependencies are correctly installed and the environment is properly configured.

**Usage:**
```bash
python scripts/verify_setup.py
```

**Checks:**
- Python version
- Required packages (aeon, sktime, torch, transformers, etc.)
- GPU availability (if applicable)
- Import statements for all core modules

### `example_usage.py`
Demonstrates end-to-end usage of the FD-LLM system.

**Usage:**
```bash
python scripts/example_usage.py
```

**Demonstrates:**
- Loading CSV data
- Creating windows
- Training a classifier
- Generating predictions
- Basic evaluation

**Note:** This is a simplified example for learning purposes. For production use, see `training/train_rocket.py`.

## Common Workflows

### 1. Train and Export for Colab
```bash
# Train classifier locally
python training/train_rocket.py --data data.csv --run_name exp_001

# Export for Colab
python scripts/export_for_colab.py \
    --model outputs/exp_001/model.pkl \
    --data data.csv

# Upload predictions_for_colab.parquet to Google Drive
```

### 2. Analyze Classifier Performance
```bash
# After training, analyze prediction patterns
python scripts/analyze_classifier_performance.py
```

### 3. Filter for Fault Analysis
```bash
# Filter to only fault windows
python scripts/filter_fault_windows.py

# Or filter to high-confidence faults only
python scripts/filter_high_confidence_faults.py
```

## Notes

- All scripts assume they are run from the project root directory (`fd-llm/`)
- Default paths point to `outputs/exp_full_dataset/` - modify as needed
- Scripts include error handling and progress indicators
- For detailed documentation, see comments within each script

---

**Directory**: `/scripts/`  
**Last Updated**: 2025-10-13

