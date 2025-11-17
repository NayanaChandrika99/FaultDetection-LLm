# FD-LLM Interactive Demo Guide

## Overview

The FD-LLM demo showcases a complete hybrid fault detection system that combines:
- **Fast time-series classification** (MultiROCKET) for real-time pattern recognition
- **LLM explanations** (Mistral-7B) for natural language fault diagnosis
- **Self-consistency voting** for reliable explanations

## Running the Demo

### Prerequisites

1. **Conda environment** with dependencies installed:
   ```bash
   conda activate rm
   ```

2. **Pre-trained model and data** (already included):
   - `outputs/exp_full_dataset/model.pkl` - Trained classifier
   - `outputs/exp_full_dataset/predictions_for_colab.parquet` - 5,236 predictions
   - `outputs/exp_full_dataset/explanations.jsonl` - LLM-generated explanations

### Run the Demo

```bash
python demo.py
```

**Duration:** ~5 seconds  
**Output:** Colorized terminal display with system analysis

## What the Demo Shows

### 1. Model Loading
- ✅ Loads pre-trained MultiROCKET classifier
- ✅ Displays model type and configuration

### 2. Classifier Predictions Analysis
- **Total predictions:** 5,236 windows (60-second windows from sensor data)
- **Prediction distribution:**
  - Dilution: 50.1% (2,625 windows)
  - Normal: 34.4% (1,800 windows)
  - Settling/Segregation: 15.5% (811 windows)
- **Confidence statistics:**
  - Mean: 71.7%
  - Median: 74.9%
  - Range: 34.4% - 100%

### 3. LLM Explanations Quality
- **Total explanations:** 6 (sample set for demo)
- **Voting agreement:** 100% (all 5 votes agreed on diagnosis)
- **Diagnoses:**
  - Settling/Segregation: 66.7%
  - Dilution: 33.3%

### 4. Example Fault Explanation

The demo shows a detailed breakdown of one fault detection:

**Classifier Output:**
- Predicted class (e.g., "Dilution")
- Confidence score (e.g., 74.6%)

**LLM Explanation:**
- Final diagnosis with confidence
- **Evidence** (5+ specific numeric observations):
  - "Density mean at 1009.6 kg/m³ is below normal range"
  - "SG at 1.010 is 0.015 below target"
  - "Density trend at -65.00 kg/m³ per 5min indicates progressive dilution"
- **Cross-checks** (physical validation):
  - Mass balance verification
  - Density-SG correlation checks
- **Recommended actions:**
  - "Increase feed rate of solids"
  - "Monitor density and SG closely"

### 5. Classifier vs LLM Agreement
- Shows how often the classifier and LLM agree on diagnosis
- In the demo: **100% agreement** (6/6 windows)

### 6. System Capabilities
Full feature list:
- ✓ Time-Series Classification (MultiROCKET + Ridge)
- ✓ Feature Engineering (Flow, Density, Pressure)
- ✓ Physical Validation (Mass balance, Density-SG)
- ✓ LLM Explanations (Mistral-7B with QLoRA)
- ✓ Self-Consistency (5-vote ensemble)
- ✓ 10 Fault Classes
- ✓ Real-time Capable (<1ms classifier, ~3s LLM)
- ✓ Colab GPU Integration

### 7. Performance Metrics
- Displays model accuracy and F1 scores
- Notes demo limitations (75% accuracy vs 90% production target)

### 8. Next Steps
- Instructions for training on your own data
- Colab integration workflow
- Documentation references

## Demo Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FD-LLM Demo Flow                        │
└─────────────────────────────────────────────────────────────┘

1. Load Model
   └─> outputs/exp_full_dataset/model.pkl
   
2. Load Predictions
   └─> 5,236 windows with classifier predictions
   
3. Load Explanations
   └─> 6 LLM-generated explanations (sample)
   
4. Analyze & Display
   ├─> Prediction distribution
   ├─> Confidence statistics
   ├─> Example fault with full explanation
   ├─> Agreement analysis
   └─> System capabilities

5. Show Next Steps
   └─> Guide for using the system
```

## Understanding the Output

### Color Coding

The demo uses colors for clarity:
- **Green (✓):** Success messages, positive results
- **Blue (▶):** Section headers
- **Cyan:** Informational messages
- **Yellow (⚠):** Warnings
- **Red (✗):** Errors

### Key Metrics Explained

**Confidence Score:**
- Classifier confidence: Probability assigned to predicted class
- LLM confidence: Average confidence from self-consistency voting
- Higher is better (>70% is good, >85% is excellent)

**Voting Agreement:**
- Percentage of LLM votes that agreed on the same diagnosis
- 100% = all 5 explanations agreed (most reliable)
- <80% = mixed opinions (less reliable)

**Evidence Quality:**
- Each explanation includes 5+ specific numeric observations
- Evidence is validated to ensure it contains actual numbers and comparisons
- Cross-checks verify physical consistency

## Customizing the Demo

### Show Different Examples

Edit `demo.py` to change which example is shown:

```python
# Line ~165: Change the filter criteria
for exp in explanations:
    if exp['confidence'] > 0.8:  # Change threshold
        if exp['final_diagnosis'] == 'Pipeline Blockage':  # Specific fault
            example = exp
            break
```

### Analyze More Explanations

The demo only shows 6 explanations for speed. To analyze all:

```python
# Generate more explanations on Colab
# See docs/COLAB_SETUP.md for instructions
```

### Add Custom Metrics

Add your own analysis in `demo.py`:

```python
def show_custom_analysis(predictions_df, explanations):
    """Your custom analysis here"""
    print_section("9. Custom Analysis")
    # Your code here
    
# Call it in main():
show_custom_analysis(predictions_df, explanations)
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'pandas'"
**Solution:** Activate the conda environment:
```bash
conda activate rm
```

### "Model not found at outputs/exp_full_dataset/model.pkl"
**Solution:** Train a model first:
```bash
python training/train_rocket.py --data data/raw/your_data.csv
```

### "No explanations found"
**Solution:** Generate explanations on Colab:
1. Upload `FD_LLM_Colab_Explainer.ipynb` to Colab
2. Select GPU runtime
3. Run all cells
4. Download `explanations.jsonl` to `outputs/exp_full_dataset/`

### Demo runs but shows no colors
**Solution:** Your terminal may not support ANSI colors. The demo will still work, just without colors.

## What Makes This Demo Unique

### 1. Hybrid Architecture
Unlike pure ML or pure LLM systems, FD-LLM combines:
- **Fast classifier** for real-time detection (milliseconds)
- **LLM explainer** for detailed diagnosis (seconds)
- Best of both worlds: speed + interpretability

### 2. Self-Consistency
The LLM generates 5 independent explanations and votes:
- Reduces hallucinations
- Increases reliability
- Provides confidence metrics

### 3. Physical Validation
All explanations are grounded in:
- Actual sensor readings
- Physical laws (mass balance, density-SG)
- Domain knowledge (process engineering)

### 4. Production-Ready Design
- Modular architecture (easy to extend)
- Configuration-driven (YAML files)
- Scalable (classifier local, LLM on cloud GPU)
- Well-documented (comprehensive guides)

## Demo vs Production

| Aspect | Demo | Production |
|--------|------|------------|
| **Accuracy** | 75% | 90%+ required |
| **Training Data** | Abnormal period | Representative data |
| **Labels** | Heuristic | Expert-validated |
| **Testing** | Basic | Comprehensive |
| **Robustness** | Not tested | Validated |
| **Deployment** | Local | Cloud/Edge |

## Next Steps After Demo

1. **Understand the system:**
   - Read `README.md` for architecture
   - Review `docs/original_spec.md` for design decisions
   - Check `DEMO_PROJECT_SUMMARY.md` for current status

2. **Train on your data:**
   - Prepare CSV with sensor readings
   - Run training script
   - Evaluate performance

3. **Generate explanations:**
   - Export predictions for Colab
   - Use GPU to generate explanations
   - Analyze results

4. **Customize for your use case:**
   - Modify fault classes
   - Adjust thresholds
   - Add domain-specific features

## Support

For questions or issues:
- **Demo issues:** Check this guide's troubleshooting section
- **System usage:** See `README.md`
- **Colab setup:** See `docs/COLAB_SETUP.md`
- **Architecture:** See `docs/ARCHITECTURE.md`

---

**Demo Version:** 1.0  
**Last Updated:** 2025-10-13  
**Estimated Demo Time:** 5 seconds

