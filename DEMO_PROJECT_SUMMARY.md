# FD-LLM Demo Project - Complete Summary

## Project Overview

**FD-LLM** is a hybrid fault detection and explanation system for slurry pipeline monitoring that combines:
- **MultiROCKET Classifier** (fast time-series pattern recognition)
- **Mistral-7B LLM** (natural language explanations with self-consistency)

**Status:** ✅ **Complete Demo/Prototype**

### 🎬 Quick Demo

```bash
conda activate rm
python demo.py
```

See the system in action with pre-trained models and LLM explanations!

---

## What We Built

### 1. Data Pipeline ✅
- CSV loading with flexible timestamp parsing
- 60-second windowing with 15-second stride
- Feature extraction (flow, density, pressure statistics)
- Physical validation (mass balance, density-SG checks)

**Files:**
- `data/loaders/slurry_loader.py`
- `models/encoders/feature_extractor.py`
- `utils/physical_checks.py`

### 2. MultiROCKET Classifier ✅
- Time-series classifier trained on sensor data
- **Current Performance:** 75% accuracy
- Detects: Dilution, Normal, Settling/Segregation
- Fast inference (~milliseconds per window)

**Files:**
- `models/rocket_heads.py`
- `training/train_rocket.py`
- `outputs/exp_full_dataset/model.pkl`

**Results:**
```
Overall Accuracy: 75%
Macro F1: 0.66

Prediction Distribution (5,236 windows):
  Dilution: 2,625 (50.1%)
  Normal: 1,800 (34.4%)
  Settling/Segregation: 811 (15.5%)
```

### 3. LLM Explainer ✅
- Mistral-7B-Instruct with 4-bit quantization
- Generates structured JSON explanations with:
  - Fault diagnosis
  - Numeric evidence from sensor data
  - Physical consistency cross-checks
  - Recommended actions
- Self-consistency voting (k=5 explanations, majority vote)

**Files:**
- `explainer/llm_setup.py`
- `explainer/prompt_templates.py`
- `explainer/self_consistency.py`
- `FD_LLM_Colab_Explainer.ipynb`

**Example Output:**
```json
{
  "final_diagnosis": "Dilution",
  "confidence": 0.856,
  "evidence": [
    "Density Mean is 1009.6 kg/m³, below normal range of 1015 kg/m³",
    "Density trending downward at -65.00 kg/m³ per 5min",
    "SG at 1.010 is 0.020 below target, confirming water addition"
  ],
  "cross_checks": [
    "Check for potential upstream influencing factors on flow and density",
    "Inspect slurry composition and potential dilution sources"
  ],
  "recommended_actions": [
    "Monitor and adjust slurry composition to maintain process variables within normal range",
    "Investigate upstream water injection systems"
  ]
}
```

### 4. Analysis & Utilities ✅
- Filter scripts for fault-only and high-confidence predictions
- Performance analysis tools
- Colab notebook for GPU-accelerated explanation generation

**Files:**
- `filter_fault_windows.py`
- `filter_high_confidence_faults.py`
- `analyze_classifier_performance.py`
- `export_for_colab.py`

---

## Current Dataset Statistics

**Input Data:** `data_4b0c_250926-0000_250926-2251.csv`

**Characteristics:**
- Time period: 09/26/2025, 00:00 - 22:51 (22 hours 51 minutes)
- Total windows: 5,236
- Fault rate: 65.6% (abnormally high - likely stress test or maintenance period)
- Temporal pattern: Mostly faults except windows 1000-1500 (normal period)

**Filtered Datasets Available:**
1. **All faults:** 3,436 windows (removes Normal)
2. **High-confidence faults:** 2,403 windows (confidence ≥0.7)

---

## Demo Capabilities

### What This System Can Do:

✅ **Real-time fault detection** (classifier runs in milliseconds)  
✅ **Natural language explanations** for detected faults  
✅ **Evidence-based reasoning** with numeric claims from actual sensor data  
✅ **Self-consistency validation** (multiple explanation attempts with voting)  
✅ **Actionable recommendations** for operators  
✅ **Scalable architecture** (classifier local, LLM on cloud GPU)  

### Current Limitations:

⚠️ **Classifier accuracy is 75%** (production systems need ≥90%)  
⚠️ **Training data from abnormal period** (65% fault rate vs expected 5-15%)  
⚠️ **LLM explanations not fine-tuned** (using pre-trained Mistral, not domain-adapted)  
⚠️ **Label quality uncertain** (heuristic-based, not expert-verified)  

---

## Key Files & Outputs

### Trained Models
```
outputs/exp_full_dataset/
├── model.pkl              # Trained MultiROCKET classifier
├── results.json           # Training metrics
├── confusion_matrix.png   # Performance visualization
└── classification_report.txt
```

### Prediction Datasets
```
outputs/exp_full_dataset/
├── predictions_for_colab.parquet           # All 5,236 windows
├── predictions_faults_only.parquet         # 3,436 fault windows
└── predictions_high_conf_faults.parquet    # 2,403 high-confidence faults
```

### Explanations (if generated)
```
outputs/exp_full_dataset/
└── explanations.jsonl     # LLM-generated explanations
```

### Notebooks
```
FD_LLM_Colab_Explainer.ipynb   # GPU-accelerated explanation generation
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RAW SENSOR DATA                         │
│              (CSV with timestamps + 11 sensors)             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  DATA PREPROCESSING                         │
│  • Parse timestamps (MM:SS.s format)                        │
│  • Resample to 1 Hz                                         │
│  • Interpolate gaps ≤3 seconds                              │
│  • Create 60s windows (15s stride)                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               FEATURE EXTRACTION                            │
│  • Flow: mean, std, CV, zeros, rate_of_change              │
│  • Density: mean, std, trend, spikes, SG deviation         │
│  • Pressure: mean, variability, correlation with flow      │
│  • Physical validation (mass balance, density-SG)          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            MultiROCKET CLASSIFIER (Local/Fast)              │
│  • Processes windows in milliseconds                        │
│  • Output: {fault_type, confidence}                         │
│  • 75% accuracy (demo quality)                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         MISTRAL-7B LLM EXPLAINER (Colab GPU)               │
│  • Input: classifier prediction + extracted features        │
│  • Self-consistency: Generate 5 explanations, vote          │
│  • Output: Structured JSON with evidence & actions          │
│  • ~90s per window (with k=5)                               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    FINAL OUTPUT                             │
│  • Fault Diagnosis: "Dilution"                             │
│  • Confidence: 0.856                                        │
│  • Evidence: [3-5 numeric claims]                          │
│  • Recommended Actions: [2-3 specific steps]               │
└─────────────────────────────────────────────────────────────┘
```

---

## How to Use (Demo Workflow)

### 1. Run Classifier on New Data
```bash
conda activate rm

python training/train_rocket.py \
    --data data/raw/YOUR_DATA.csv \
    --config experiments/configs/baseline.yaml \
    --output_dir outputs/YOUR_EXPERIMENT
```

### 2. Export Predictions for LLM
```bash
python export_for_colab.py \
    --model outputs/YOUR_EXPERIMENT/model.pkl \
    --data data/raw/YOUR_DATA.csv \
    --output outputs/YOUR_EXPERIMENT/predictions_for_colab.parquet
```

### 3. (Optional) Filter to High-Confidence Faults
```bash
python filter_high_confidence_faults.py \
    --input outputs/YOUR_EXPERIMENT/predictions_for_colab.parquet \
    --output outputs/YOUR_EXPERIMENT/predictions_high_conf_faults.parquet \
    --threshold 0.7
```

### 4. Generate Explanations (Google Colab)
1. Upload `FD_LLM_Colab_Explainer.ipynb` to Colab
2. Upload predictions file to Google Drive
3. Select GPU runtime (T4/V100/A100)
4. Run all cells
5. Download `explanations.jsonl`

---

## Production Readiness Checklist

To upgrade this demo to a production system:

### Must-Have Improvements:
- [ ] **Retrain classifier to ≥90% accuracy**
  - Get representative training data (85-95% Normal operations)
  - Expert-validate labels (not heuristic-based)
  - Balance classes properly
  
- [ ] **Validate on multiple time periods**
  - Test on data from different dates/conditions
  - Ensure generalization

- [ ] **Reduce false positive rate**
  - Current: ~25% (based on 75% accuracy)
  - Target: <5%

### Nice-to-Have Improvements:
- [ ] Fine-tune LLM on domain-specific examples
- [ ] Add more fault types (blockage, cavitation, etc.)
- [ ] Real-time deployment infrastructure
- [ ] Operator feedback loop for continuous improvement
- [ ] Dashboard for monitoring and visualization

---

## Key Learnings & Insights

### What Worked Well:
✅ **Hybrid architecture is sound** - combining fast classifier + LLM explainer  
✅ **Self-consistency voting improves reliability** - majority vote reduces hallucinations  
✅ **Mistral-7B follows JSON format well** - better than DeepSeek-R1 for structured output  
✅ **Colab GPU integration** - cost-effective for LLM inference  
✅ **Feature extraction is solid** - basic stats capture key patterns  

### What Needs Improvement:
⚠️ **Label quality is critical** - heuristic labels limit accuracy  
⚠️ **Representative data matters** - abnormal data (65% faults) hurts generalization  
⚠️ **LLM speed is a bottleneck** - 90s/window with k=5 is too slow for production  

### Design Decisions:
- **Why MultiROCKET?** Fast, robust baseline for time-series (no deep learning complexity)
- **Why Mistral over DeepSeek-R1?** Better at following JSON format (DeepSeek shows reasoning)
- **Why self-consistency?** LLMs can hallucinate; voting improves faithfulness
- **Why separate classifier + LLM?** Classifier is fast for real-time, LLM for explanation quality

---

## Demo Use Cases

### 1. Research & Development
- Demonstrate hybrid ML + LLM approach
- Benchmark different LLM models
- Test explanation quality metrics

### 2. Stakeholder Presentations
- Show end-to-end fault detection + explanation
- Demonstrate natural language interface
- Prove concept feasibility

### 3. Pilot Deployment
- Run on historical data to identify patterns
- Get operator feedback on explanation quality
- Inform production system requirements

---

## Repository Structure

```
fd-llm/
├── data/
│   ├── loaders/           # CSV parsing, windowing
│   └── raw/               # Input CSV files
├── models/
│   ├── encoders/          # Feature extraction
│   ├── rocket_heads.py    # MultiROCKET classifier
│   └── fusion.py          # Late fusion (optional)
├── explainer/
│   ├── llm_setup.py       # LLM loading (QLoRA)
│   ├── prompt_templates.py # Prompts & validation
│   └── self_consistency.py # Voting mechanism
├── training/
│   └── train_rocket.py    # Main training script
├── evaluation/
│   └── metrics.py         # Performance metrics
├── utils/
│   └── physical_checks.py # Mass balance validation
├── outputs/
│   └── exp_full_dataset/  # Results & models
├── experiments/
│   └── configs/           # YAML configurations
├── tests/                 # Unit tests
├── filter_*.py            # Utility scripts
├── analyze_*.py           # Analysis scripts
├── export_for_colab.py    # Prepare data for LLM
├── FD_LLM_Colab_Explainer.ipynb
├── requirements.txt
├── README.md
└── DEMO_PROJECT_SUMMARY.md (this file)
```

---

## Contact & Next Steps

**For Production Deployment:**
1. Collect representative operational data
2. Expert-label a training set (500-1000 windows)
3. Retrain classifier to ≥90% accuracy
4. Fine-tune LLM on domain examples
5. Deploy with monitoring infrastructure

**For Research:**
- Experiment with different LLM models
- Test explanation faithfulness metrics
- Compare with other time-series classifiers
- Publish results on benchmark datasets

---

**This is a complete, functional demo that proves the hybrid fault detection + LLM explanation concept works. It's ready for demonstrations, research, and as a foundation for a production system.**

**Status:** ✅ Demo Complete | ⚠️ Not Production-Ready (needs 90%+ accuracy)

