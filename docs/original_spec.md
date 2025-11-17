# FD-LLM: Multi-Sensor Slurry Fault Diagnosis

## Overview

Hybrid system combining specialized time-series classification with LLM explanations for slurry pipeline monitoring. **Architecture**: TS Classifier (label/confidence) → LLM Explainer (structured JSON with grounded evidence).

**Core Principle**: Use TS models for numeric classification, LLMs for natural language explanation. Never force LLMs to classify raw time series.

## Quick Reference

### Key Decisions
| Item | Choice | Rationale |
|------|--------|-----------|
| **Classifier** | MultiROCKET + Ridge | Fast, strong baseline, minimal tuning |
| **Enhancement** | +HYDRA features | 2-5% accuracy gain |
| **Fusion** | Late (default) | Better for multi-rate sensors |
| **LLM** | LLaMA-3-8B + QLoRA 4-bit | Fits T4 16GB, good instruction-following |
| **Window** | 60s / 15s stride | Captures transients, 75% overlap |
| **Min Evidence** | 3 numeric claims | Prevents unfaithful explanations |

### Sensor Groups (from your CSV)
```
Flow: Slurry Flow, Mass Flow, Solids Flow, Solids Mass Flow
Density: Slurry Density, SG, Percent Solids (mass/vol)
Process: Pressure, Temperature, DV Constant
Derived: DV, SG Shift (Raw/Target)
```

### Fault Classes
Normal | Pump Cavitation | Pipeline Blockage | Settling/Segregation | Air Entrainment | Dilution | Concentration | Valve Transient | Instrument Fault | Process Upset

## Tech Stack

```python
# Time Series
aeon-toolkit==0.9.0, sktime==0.25.0, numpy, scipy

# Deep Learning  
torch, transformers==4.35.0, peft==0.7.0, bitsandbytes==0.41.0

# Data
pandas>=2.0.0, pyarrow, polars (optional)

# Config & Eval
pyyaml, hydra-core, scikit-learn, matplotlib, seaborn
```

## Repository Structure

```
fd-llm/
├── data/
│   ├── raw/                    # CSV files (gitignored)
│   └── loaders/                # slurry_loader.py, window_creator.py
├── models/
│   ├── encoders/               # flow_features.py, density_features.py, etc.
│   ├── rocket_heads.py         # MultiROCKET + Ridge
│   ├── patchtst_head.py        # Deep alternative
│   └── fusion.py               # Late/Early fusion
├── training/                   # train_rocket.py, train_patchtst.py
├── evaluation/                 # metrics.py, robustness_tests.py
├── explainer/                  # prompt_templates.py, self_consistency.py
├── utils/                      # signal_processing.py, physical_checks.py
├── experiments/configs/        # YAML configs
└── tests/                      # Unit & integration tests
```

## Data Loading

### Parse Your CSV Format
```python
import pandas as pd

def load_slurry_csv(path):
    df = pd.read_csv(path)
    
    # Parse MM:SS.s timestamps
    df['timestamp'] = df['Timestamp'].apply(
        lambda x: pd.Timedelta(minutes=int(x.split(':')[0]), 
                               seconds=float(x.split(':')[1]))
    )
    df = df.drop('Timestamp', axis=1).set_index('timestamp')
    
    # Resample to uniform 1 Hz, interpolate gaps ≤3s
    df = df.resample('1S').mean().interpolate(method='linear', limit=3)
    
    # Flag zero-flow events
    df['zero_flow_flag'] = (df['Slurry Flow (m3/s)'] == 0).astype(int)
    
    return df

def create_windows(df, window_sec=60, stride_sec=15):
    """Create overlapping windows. Returns [n_windows, n_sensors, 60]"""
    windows, labels = [], []
    n_samples = len(df)
    
    for start in range(0, n_samples - window_sec + 1, stride_sec):
        window = df.iloc[start:start+window_sec].values
        if np.isnan(window).sum() / window.size < 0.1:  # <10% missing
            windows.append(window)
            # labels.append(get_label(window))  # Your labeling logic
    
    return np.array(windows), np.array(labels)
```

## Feature Engineering

### Quick Reference Table
| Group | Key Features | Thresholds |
|-------|-------------|------------|
| **Flow** | mean, std, CV, rate_of_change, n_zero_events, stability | CV > 0.15 (unstable), zero_flow threshold = 0.01 m³/s |
| **Density** | mean, std, trend (kg/m³ per 5min), SG deviation from target | SG deviation > ±0.03, density spike > 3σ |
| **Pressure** | mean, std, n_spikes, pressure-flow correlation | High variability > 0.1, spike > 3σ |
| **Temperature** | mean, change per 10min, states (normal/elevated/high) | Normal <30°C, elevated 30-40°C, high >40°C |
| **Particle Size** | mean DV, drift over window | Drift > 5 μm indicates segregation |
| **Physical** | Mass balance error, solids consistency ratio, density-SG alignment | Mass balance error >15%, density-SG diff >0.05 |

### Compact Feature Extraction
```python
def extract_features(window_df):
    """Compute all features for a 60s window."""
    features = {}
    
    # Flow
    flow = window_df['Slurry Flow (m3/s)']
    features.update({
        'flow_mean': flow.mean(), 'flow_std': flow.std(),
        'flow_cv': flow.std()/(flow.mean()+1e-6),
        'flow_roc': (flow.iloc[-1]-flow.iloc[0])/60,  # per min
        'flow_n_zero': (flow==0).sum(),
    })
    
    # Density
    density = window_df['Slurry Density (kg/m3)']
    sg = window_df['SG']
    features.update({
        'density_mean': density.mean(), 'density_std': density.std(),
        'sg_mean': sg.mean(),
        'sg_target_dev': sg.mean() - window_df['SG Shift: Target SG'].mean(),
        'density_trend': (density.iloc[-1]-density.iloc[0])/300,  # per 5min
    })
    
    # Physical consistency
    mass_flow_meas = window_df['Slurry Mass Flow (kg/s)'].mean()
    mass_flow_calc = flow.mean() * density.mean()
    features['mass_balance_error'] = abs(mass_flow_meas - mass_flow_calc) / mass_flow_meas
    
    return features
```

## Model Training

### MultiROCKET Baseline
```python
from aeon.transformations.collection.convolution_based import MultiRocket
from sklearn.linear_model import RidgeClassifierCV

# X: [n_windows, n_sensors, 60], y: [n_windows]
rocket = MultiRocket(n_kernels=6250, random_state=42)
X_transform = rocket.fit_transform(X_train, y_train)

clf = RidgeClassifierCV(alphas=np.logspace(-3,3,10))
clf.fit(X_transform, y_train)
```

### Late Fusion (Multi-Sensor)
```python
# Separate transforms per sensor group
rocket_flow = MultiRocket(n_kernels=3000).fit_transform(X_train[:,[0,1,2,3],:])
rocket_dens = MultiRocket(n_kernels=2000).fit_transform(X_train[:,[4,5,6,7],:])
rocket_proc = MultiRocket(n_kernels=1250).fit_transform(X_train[:,[8,9,10],:])

X_fused = np.hstack([rocket_flow, rocket_dens, rocket_proc])
clf = RidgeClassifierCV().fit(X_fused, y_train)
```

## LLM Explainer

### Prompt Template
```python
SYSTEM = """You are a process engineer. Analyze sensor data, think step-by-step internally, output ONLY valid JSON."""

def create_prompt(features, pred, conf):
    return f"""PRIMARY_CLASSIFIER: {pred}, confidence={conf:.3f}

SENSORS:
  Flow: {features['flow_mean']:.3f} m³/s (CV={features['flow_cv']:.3f}, zeros={features['flow_n_zero']})
  Density: {features['density_mean']:.1f} kg/m³ (trend={features['density_trend']:.2f}/5min)
  SG: {features['sg_mean']:.3f} (target_dev={features['sg_target_dev']:.3f})

OUTPUT (JSON only):
{{"final_diagnosis":"class", "confidence":0-1, "evidence":["claim1","claim2","claim3"], 
"cross_checks":["check1"], "recommended_actions":["action1"]}}"""
```

### Self-Consistency (5 samples, vote)
```python
from collections import Counter

def explain_with_self_consistency(features, pred, k=5):
    explanations = []
    for _ in range(k):
        resp = llm.generate(prompt, temperature=0.8)
        try:
            exp = json.loads(resp)
            if len(exp['evidence']) >= 3:  # Min 3 numeric claims
                explanations.append(exp)
        except: continue
    
    diagnoses = [e['final_diagnosis'] for e in explanations]
    final_diag = Counter(diagnoses).most_common(1)[0][0]
    
    agreeing = [e for e in explanations if e['final_diagnosis']==final_diag]
    return {
        'final_diagnosis': final_diag,
        'confidence': np.mean([e['confidence'] for e in agreeing]),
        'evidence': list(set([ev for e in agreeing for ev in e['evidence']]))[:5]
    }
```

## Configuration (YAML)

```yaml
data:
  window_sec: 60
  stride_sec: 15
  sample_rate: 1.0
  interpolate_gaps: true
  max_gap_sec: 3

model:
  family: "multirocket"  # "hydra" | "patchtst"
  fusion: "late"
  multirocket:
    n_kernels: 6250
  classifier:
    type: "ridge"
    alphas: [0.001, 0.01, 0.1, 1, 10, 100, 1000]

explainer:
  backbone: "meta-llama/Llama-3-8B-Instruct"
  lora_r: 16
  lora_alpha: 32
  load_in_4bit: true
  self_consistency_k: 5
  min_numeric_evidence: 3

training:
  epochs: 50
  batch_size: 64
  learning_rate: 0.0001
  early_stopping_patience: 10
```

## CLI Quick Start

```bash
# Train baseline
python training/train_rocket.py --config configs/baseline.yaml

# Evaluate
python evaluation/metrics.py --run_id exp_001 --robustness

# Generate explanations
python explainer/run_explainer.py --pred_file preds.parquet --output explains.jsonl

# Full ablation
bash experiments/run_ablation_matrix.sh
```

## Evaluation

### Metrics
- **Macro-F1**: Average across classes (handles imbalance)
- **Per-class Recall**: Catch rare critical faults
- **PR-AUC**: Precision-recall curve area
- **MTTD**: Mean time to detection

### Ablation Matrix
| Phase | Sensors | Model | Goal |
|-------|---------|-------|------|
| 1a | Flow only | MultiROCKET | Baseline |
| 2a | Flow+Density | MultiROCKET | Fusion benefit? |
| 3a | All sensors | MultiROCKET+HYDRA | Full system |
| 4 | All sensors | PatchTST | Deep vs transforms |

### Robustness Tests
- Cross-machine: Train Machine A, test Machine B
- Sensor dropout: Remove pressure/temp/DV individually
- Noise: SNR 30→20→10→5 dB
- Calibration drift: ±5-10% systematic shifts

## Critical Pitfalls & Solutions

### 1. Data Leakage
❌ **Wrong**: `scaler.fit(X_all)` then split  
✅ **Correct**: `scaler.fit(X_train)` only, transform test separately

### 2. Zero-Flow Events
❌ **Wrong**: Label all zero-flow as faults  
✅ **Correct**: Check if gradual decrease (planned) vs sudden (fault)

### 3. Physical Inconsistency
❌ **Wrong**: Train on all data including bad sensors  
✅ **Correct**: Validate mass balance, density-SG alignment, reject violations

```python
def validate_window(df):
    # Mass balance: measured ≈ flow × density
    mass_calc = df['Slurry Flow (m3/s)'].mean() * df['Slurry Density (kg/m3)'].mean()
    mass_meas = df['Slurry Mass Flow (kg/s)'].mean()
    if abs(mass_calc - mass_meas)/mass_meas > 0.15:
        return False, "mass_balance_violation"
    
    # Density-SG: SG ≈ Density/1000
    if abs(df['SG'].mean() - df['Slurry Density (kg/m3)'].mean()/1000) > 0.05:
        return False, "density_sg_mismatch"
    
    return True, "valid"
```

### 4. LLM Faithfulness
❌ **Wrong**: Accept any well-formed JSON  
✅ **Correct**: Enforce ≥3 numeric claims, validate against actual features

```python
def validate_explanation(exp, features):
    if len(exp['evidence']) < 3:
        return False
    
    numeric_claims = sum(1 for e in exp['evidence'] 
                        if any(c.isdigit() for c in e) and any(op in e for op in ['>', '<', '=']))
    return numeric_claims >= 3
```

## Testing

### Unit Tests
```python
# test_features.py
def test_flow_features():
    flow = np.ones(60) * 2.0
    flow[45:] = 0  # Drop to zero
    features = compute_flow_features(pd.DataFrame({'Slurry Flow (m3/s)': flow}))
    assert features['flow_n_zero'] == 15

# test_loaders.py  
def test_timestamp_parse():
    assert parse_timestamp("01:23.7") == pd.Timedelta(minutes=1, seconds=23.7)
```

### Integration Test
```python
def test_end_to_end():
    loader = SlurryDataLoader("test.csv")
    X, y = loader.load()
    clf = LateFusionClassifier().fit(X, y)
    preds = clf.predict(X)
    exp = LLMExplainer().explain(clf.extract_features(X[0]), preds[0], 0.85)
    assert 'final_diagnosis' in exp
```

### Smoke Test
```bash
python training/train_rocket.py --max_samples 100 --epochs 1 --run_name smoke
```

## Environment Setup

### Local
```bash
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

### Colab
```python
!pip install -q aeon-toolkit sktime transformers peft bitsandbytes
!git clone <repo> && %cd fd-llm && !pip install -e .
from google.colab import drive; drive.mount('/content/drive')
```

## Labeling Strategy (Semi-Supervised)

```python
# 1. Threshold heuristics
labels = np.where(df['Slurry Flow (m3/s)']==0, 'Valve_Transient', 'Normal')
density_z = (df['Slurry Density (kg/m3)'] - mean) / std
labels = np.where(density_z > 3, 'Density_Anomaly', labels)

# 2. From maintenance logs
for event in logs:
    mask = (df['timestamp'] >= event['start']) & (df['timestamp'] <= event['end'])
    labels[mask] = event['fault']

# 3. Active learning on uncertain predictions
uncertain_idx = np.argsort(model.decision_function(X).max(axis=1))[:100]
# Manually label these for next iteration
```

## Synthetic Fault Generation

```python
def inject_blockage(window):
    # Pressure ↑, Flow ↓, Density ↑ (settling)
    window[:, flow_col] *= np.linspace(1.0, 0.6, len(window))
    window[:, pressure_col] *= np.linspace(1.0, 1.3, len(window))
    window[:, density_col] *= np.linspace(1.0, 1.1, len(window))
    return window

def inject_air_entrainment(window):
    # Density ↓, SG ↓, Flow oscillations
    window[:, density_col] *= 0.90
    window[:, sg_col] *= 0.95
    window[:, flow_col] += np.random.normal(0, 0.05, len(window))
    return window
```

## Git Workflow

**Branches**: `main` (stable) | `dev` (integration) | `feature/*` | `experiment/*`

**Commits**: `feat(loader): add CSV parser` | `fix(features): correct mass balance` | `test: add flow feature tests`

**Pre-commit**: black, isort, flake8, pytest (quick)

## Performance Notes (T4 16GB)

- MultiROCKET: 2-5 min (10k samples, CPU/GPU)
- PatchTST: 15-45 min (batch=64, fp16)
- LLM QLoRA: 3-5 hours (1k examples, batch=1, grad_accum=8)

## Key Hyperparameters

```python
MultiROCKET: n_kernels=6250, max_dilations=32
Ridge: alphas=logspace(-3,3,10), cv=5
PatchTST: patch_len=16, d_model=256, layers=3
LLM: LoRA r=16, alpha=32, 4bit, temp=0.8 (generation)
```

## Resources

- CWRU Bearing: https://engineering.case.edu/bearingdatacenter
- Paderborn: https://mb.uni-paderborn.de/kat/datacenter
- MultiROCKET paper: Tan et al. (2022), code in aeon
- Self-consistent CoT: Wang et al. (2022)
- QLoRA: Dettmers et al. (2023)

---

**Version**: 2.1 | **Updated**: Oct 11, 2025 | **Target**: Claude Sonnet 4.5+