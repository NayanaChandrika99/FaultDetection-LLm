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

## Colab Quick Steps

1. **Train locally**  
   `python training/train_rocket.py --config experiments/configs/baseline.yaml --data data/raw/your_data.csv --run_name exp_001`

2. **Export predictions**  
   `python scripts/export_for_colab.py --model outputs/exp_001/model.pkl --data data/raw/your_data.csv`

3. **Upload to Drive & run notebook**  
   Copy `explainer/`, `FD_LLM_Colab_Explainer.ipynb`, and `outputs/exp_001/predictions_for_colab.parquet`, then run the notebook in a GPU-enabled Colab session.

4. **Download explanations**  
   Retrieve `outputs/exp_001/explanations.jsonl` from Drive and analyze locally (`demo.py`, `evaluation/metrics.py`, etc.).

Need the full walkthrough? See `docs/COLAB_SETUP.md`.

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
