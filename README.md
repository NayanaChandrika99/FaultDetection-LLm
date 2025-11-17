# FD-LLM: Multi-Sensor Slurry Fault Diagnosis

Hybrid system combining time-series classification (MultiROCKET) with LLM explanations for slurry pipeline fault monitoring.

## Overview
  
**Architecture**: Time-Series Classifier → LLM Explainer

- **Classifier**: MultiROCKET + Ridge (fast, accurate baseline)
- **Explainer**: LLaMA-3-8B + QLoRA with self-consistency voting
- **Core Principle**: Use TS models for numeric classification, LLMs for natural language explanation

## Quick Start

### Try the Demo

See the system in action with pre-trained models:

```bash
# Activate environment
conda activate rm

# Run interactive demo
python demo.py
```

The demo showcases:
- ✓ Trained MultiROCKET classifier with 5,236 predictions
- ✓ LLM-generated explanations with evidence and recommendations
- ✓ Classifier vs LLM agreement analysis
- ✓ System capabilities and performance metrics

### Installation

```bash
# Clone and install
git clone <repository>
cd fd-llm
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

### Basic Usage

**Option 1: All Local (CPU-based explanations)**
```bash
# 1. Train classifier
python training/train_rocket.py \
    --config experiments/configs/baseline.yaml \
    --data data/raw/your_data.csv \
    --output_dir outputs \
    --run_name my_experiment

# 2. Generate explanations (requires GPU or slow on CPU)
python explainer/run_explainer.py \
    --pred_file outputs/my_experiment/predictions.parquet \
    --output explanations.jsonl \
    --k 5

# 3. Evaluate
python evaluation/metrics.py \
    --run_id my_experiment \
    --robustness
```

**Option 2: Local Training + Colab GPU (Recommended)**
```bash
# 1. Train classifier locally (fast on CPU)
python training/train_rocket.py \
    --config experiments/configs/baseline.yaml \
    --data data/raw/your_data.csv \
    --run_name my_experiment

# 2. Export for Colab
python scripts/export_for_colab.py \
    --model outputs/my_experiment/model.pkl \
    --data data/raw/your_data.csv

# 3. Upload to Google Drive and run LLM on Colab GPU
#    (See docs/COLAB_SETUP.md for detailed instructions)

# 4. Download results and evaluate locally
python evaluation/metrics.py --run_id my_experiment
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_loaders.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## Performance Snapshot

Numbers below reference the canonical run stored in `outputs/exp_full_dataset/results.json`.

- **Coverage**: 3,697 labeled 60 s windows (11 sensors, 10 fault classes) with a 2,957/740 train–test split after physics-based filtering.
- **Classifier**: MultiROCKET + Ridge reaches **0.66 macro F1** and 0.63 mean confidence on the hold-out set; inference remains <1 ms per window on CPU.
- **Explainer**: Self-consistent Llama‑3 (4-bit) averages 2–5 s per window on a Colab T4 GPU (~10× faster than CPU) while enforcing ≥3 numeric evidence claims.
- **Artifacts**: Saved model (`model.pkl`), predictions, explanations, and metrics JSON keep every experiment reproducible.

## Documentation Guide

- `docs/ARCHITECTURE.md` – end-to-end data/ML/LLM architecture, deployment modes, and scaling considerations.
- `docs/IMPLEMENTATION_SUMMARY.md` – feature-by-feature delivery checklist with current performance numbers.
- `docs/COLAB_SETUP.md` – step-by-step instructions for exporting predictions, running the explainer on Colab, and pulling results back.
- `DEMO_PROJECT_SUMMARY.md` & `EXECUTIVE_BRIEFING.md` – stakeholder-ready summaries of scope, risks, and demo storylines.
- `FD_LLM_Colab_Explainer.ipynb` – ready-to-run Colab companion notebook referenced throughout the docs.

## Project Structure

```
fd-llm/
├── README.md                   # Main documentation (you are here)
├── DEMO_PROJECT_SUMMARY.md     # Project status and demo scope
├── requirements.txt
├── setup.py
├── data/
│   ├── raw/                    # CSV files (gitignored)
│   └── loaders/                # Data loading utilities
│       ├── slurry_loader.py    # CSV parsing, windowing
│       └── window_creator.py   # Labeling, validation
├── models/
│   ├── encoders/               # Feature extraction
│   │   ├── flow_features.py
│   │   ├── density_features.py
│   │   ├── pressure_features.py
│   │   └── feature_extractor.py
│   ├── rocket_heads.py         # MultiROCKET classifier
│   └── fusion.py               # Late fusion
├── training/
│   └── train_rocket.py         # Main training script
├── evaluation/
│   ├── metrics.py              # Performance metrics
│   └── robustness_tests.py     # Robustness suite
├── explainer/
│   ├── prompt_templates.py     # LLM prompts
│   ├── llm_setup.py           # Model loading
│   ├── self_consistency.py    # Explanation generation
│   └── run_explainer.py       # CLI tool
├── utils/
│   └── physical_checks.py     # Physics validation
├── experiments/configs/
│   └── baseline.yaml          # Configuration
├── tests/                     # Unit & integration tests
├── scripts/                   # Analysis & utility scripts
│   ├── analyze_classifier_performance.py
│   ├── filter_fault_windows.py
│   ├── filter_high_confidence_faults.py
│   ├── export_for_colab.py
│   ├── verify_setup.py
│   └── example_usage.py
├── docs/                      # Documentation
│   ├── original_spec.md       # Original project specification
│   ├── ARCHITECTURE.md        # System architecture diagram
│   ├── COLAB_SETUP.md        # Google Colab integration guide
│   └── IMPLEMENTATION_SUMMARY.md  # Implementation notes
├── outputs/                   # Training outputs (gitignored)
│   └── exp_full_dataset/     # Final experiment results
└── FD_LLM_Colab_Explainer.ipynb  # Ready-to-run Colab notebook
```

## Data Format

### Expected CSV Structure

```csv
Timestamp,Slurry Flow (m3/s),Slurry Density (kg/m3),SG,Pressure (kPa),...
00:00.0,2.1,1205.3,1.205,485.2,...
00:01.0,2.0,1203.8,1.204,487.1,...
```

**Key columns:**
- `Timestamp`: MM:SS.s format
- Flow sensors: `Slurry Flow (m3/s)`, `Mass Flow (kg/s)`, etc.
- Density sensors: `Slurry Density (kg/m3)`, `SG`, `Percent Solids`
- Process: `Pressure (kPa)`, `Temperature (C)`, `DV (um)`

## Configuration

Edit `experiments/configs/baseline.yaml`:

```yaml
data:
  window_sec: 60        # Window size
  stride_sec: 15        # Overlap stride (75% overlap)
  sample_rate: 1.0      # Hz

model:
  family: "multirocket"
  fusion: "late"        # Separate transforms per sensor group
  
  multirocket:
    n_kernels: 6250
    kernels_per_group:
      flow: 3000
      density: 2000
      process: 1250

explainer:
  backbone: "meta-llama/Llama-3-8B-Instruct"
  self_consistency_k: 5
  temperature: 0.8
```

## Features

### Automatic Feature Extraction

**Flow Features:**
- Mean, std, coefficient of variation
- Rate of change, zero-flow events
- Stability score

**Density Features:**
- Mean, std, trend (kg/m³ per 5min)
- SG deviation from target
- Spike detection

**Process Features:**
- Pressure-flow correlation
- Temperature state classification
- Particle size drift

### Physical Validation

Automatically validates:
- **Mass Balance**: `mass_flow ≈ volumetric_flow × density` (error <15%)
- **Density-SG**: `SG ≈ Density / 1000` (deviation <0.05)
- **Solids Consistency**: Solids concentration checks

Invalid windows are automatically filtered before training.

## Fault Classes

1. **Normal** - Stable operation
2. **Pump Cavitation** - Low pressure, flow fluctuations
3. **Pipeline Blockage** - High pressure, low flow
4. **Settling/Segregation** - Particle size drift, density stratification
5. **Air Entrainment** - Low density, flow oscillations
6. **Dilution** - Low solids concentration
7. **Concentration** - High solids concentration
8. **Valve Transient** - Sudden flow changes
9. **Instrument Fault** - Physical inconsistencies
10. **Process Upset** - Multiple parameters out of range

## Model Training

### Training Pipeline

```python
from training.train_rocket import prepare_data, train_model

# Load and prepare data
windows, labels, metadata = prepare_data(
    csv_path="data.csv",
    config=config
)

# Train model
model, X_train, X_test, y_train, y_test = train_model(
    windows=windows,
    labels=labels,
    config=config
)
```

### Smoke Test

Quick test with minimal data:

```bash
python training/train_rocket.py \
    --data data/raw/sample.csv \
    --max_samples 100 \
    --run_name smoke_test
```

## LLM Explanations

### Self-Consistency Process

1. Generate k=5 explanations with temperature=0.8
2. Vote on final diagnosis (most common)
3. Aggregate evidence from agreeing explanations
4. Return consolidated result with confidence

### Requirements

- **GPU**: 8-10 GB VRAM (T4 16GB, RTX 3060 12GB)
- **4-bit quantization**: Enabled by default
- **Generation time**: ~2-5 seconds per window

### Output Format

```json
{
  "final_diagnosis": "Pipeline Blockage",
  "confidence": 0.87,
  "evidence": [
    "Pressure increased to 545 kPa (>20% above baseline)",
    "Flow decreased to 1.2 m³/s (<40% of normal)",
    "Density increased 8% indicating settling"
  ],
  "cross_checks": [
    "Mass balance within acceptable limits",
    "Pressure-flow correlation negative (-0.78)"
  ],
  "recommended_actions": [
    "Inspect pipeline for obstructions",
    "Consider flushing procedure"
  ]
}
```

## Evaluation Metrics

- **Macro F1**: Average F1 across all classes (handles imbalance)
- **Per-Class Recall**: Catch rare critical faults
- **PR-AUC**: Precision-recall curve area
- **Confusion Matrix**: Visual performance breakdown

## Robustness Testing

Automated tests for:

1. **Noise Injection**: SNR 30→20→10→5 dB
2. **Sensor Dropout**: Remove pressure/temp/DV individually
3. **Calibration Drift**: ±5-10% systematic shifts

```bash
python evaluation/robustness_tests.py \
    --model_path outputs/my_experiment/model.pkl \
    --test_data data/test.csv
```

## Performance

**Training (T4 16GB GPU):**
- MultiROCKET: 2-5 minutes (10k samples)
- Late Fusion: 5-10 minutes
- LLM QLoRA: 3-5 hours (1k examples)

**Inference:**
- Classifier: <1ms per window (CPU)
- LLM Explanation: ~2-5 seconds per window (GPU)

## Troubleshooting

### Out of Memory (GPU)

```python
# Use 4-bit quantization (default)
explainer = LLMExplainer(load_in_4bit=True)

# Or reduce batch size / use CPU for classifier
```

### Low Classification Accuracy

1. Check physical validation: Are many windows being rejected?
2. Verify sensor calibration: Run `physical_checks.validate_window()`
3. Increase training data or adjust kernels
4. Try late fusion for multi-rate sensors

### LLM Explanations Invalid

- Increase `max_attempts` in self-consistency
- Check prompt template formatting
- Verify JSON parsing logic

## Citation

If you use this code, please cite:

```bibtex
@software{fd-llm,
  title={FD-LLM: Hybrid Fault Diagnosis with Time-Series Classification and LLM Explanations},
  author={RedMeters},
  year={2025},
  version={0.1.0}
}
```

## License

[Specify your license]

## Google Colab Integration

**Recommended approach**: Train classifier locally (fast on CPU) + Generate explanations on Colab GPU (fast & cheap)

### Why Colab?
- MultiROCKET training is **fast on CPU** (5-10 min for 10k samples)
- LLM explanations are **slow on CPU** (30-60s per window)
- Colab Pro provides **GPU access** for $9.99/month
- **Best of both worlds**: Local control + Cloud GPU power

### Setup Steps

1. **Train locally** (your Mac, ~10 minutes):
   ```bash
   python training/train_rocket.py --data data.csv --run_name exp_001
   python scripts/export_for_colab.py --model outputs/exp_001/model.pkl --data data.csv
   ```

2. **Upload to Google Drive**:
   - `predictions_for_colab.parquet`
   - `explainer/` folder (Python scripts)

3. **Run on Colab** (GPU, ~5-10 minutes for 100 windows):
   - Open `FD_LLM_Colab_Explainer.ipynb` in Colab
   - Select GPU runtime (T4/V100/A100)
   - Follow the notebook cells

4. **Download results**:
   - `explanations.jsonl` saved to Drive
   - Download and analyze locally

**See `docs/COLAB_SETUP.md` for complete instructions with code examples.**

## Support

For issues and questions:
- **Project status**: See `DEMO_PROJECT_SUMMARY.md` (complete project overview)
- **Colab setup**: See `docs/COLAB_SETUP.md`
- **Architecture**: Check `docs/ARCHITECTURE.md`
- **Original spec**: See `docs/original_spec.md`
- **Implementation notes**: See `docs/IMPLEMENTATION_SUMMARY.md`
- Run tests: `pytest tests/ -v`
- Review logs in `outputs/<run_name>/`

---

**Version**: 0.1.0 | **Updated**: 2025-10-12
