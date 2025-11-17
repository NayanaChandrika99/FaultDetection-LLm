# FD-LLM Architecture

## Purpose & Goals

- Detect slurry-pipeline faults from multi-sensor time-series while maintaining millisecond-level classifier latency.
- Pair numeric predictions with engineering-grade natural-language explanations that cite sensor evidence.
- Keep the full pipeline deployable on plant laptops while offloading the GPU-intensive LLM stage to Colab when needed.

## End-to-End Flow

```
 ┌────────────┐     ┌──────────────────┐     ┌────────────────────┐
 │ Raw CSVs   │ →   │ Data Load + QA   │ →   │ MultiROCKET Models │
 │ (11 sensors│     │ (windowing +     │     │ (baseline or late  │
 │ @ 1 Hz)    │     │ heuristics + physics) │ │ fusion)            │
 └────────────┘     └────────┬─────────┘     └────────┬────────────┘
                               │                          │
                               │ predictions + features   │
                               ▼                          │
                         ┌───────────────┐                │
                         │ Export/Drive  │───────────────▶│
                         │ (parquet)     │                │
                         └───────────────┘                │
                               │                          │
                               ▼                          │
                         ┌───────────────┐                │
                         │ LLM Explainer │◀───────────────┘
                         │ (Colab GPU or │
                         │ local CPU)    │
                         └───────────────┘
                               │
                               ▼
                         explanations.jsonl + reports
```

The canonical run (`outputs/exp_full_dataset`) contains 3,697 labeled 60 s windows (11 sensors, 10 fault classes) with a 2,957/740 train–test split and feeds every downstream component described below.

## Component Breakdown

| Stage | Repo Path(s) | Highlights |
|-------|--------------|------------|
| Ingestion & Windowing | `data/loaders/slurry_loader.py`, `data/loaders/window_creator.py` | Parses MM:SS.s timestamps, resamples to 1 Hz, creates 60 s windows w/15 s stride, applies maintenance-log and heuristic labels plus missing-data and outlier screening. |
| Physical Validation | `utils/physical_checks.py` | Mass balance, density–SG, and solids consistency checks discard windows that violate physics. |
| Feature Engineering | `models/encoders/*` | 25+ engineered stats for flow, density, and process signals; also powers prompt context for the LLM. |
| Classification | `models/rocket_heads.py`, `models/fusion.py`, `training/train_rocket.py` | MultiROCKET transform + RidgeClassifierCV baseline (<1 ms inference) plus optional late-fusion head when sensor groups differ. |
| Evaluation | `evaluation/metrics.py`, `docs/IMPLEMENTATION_SUMMARY.md` | Macro F1, recall, PR-AUC, and robustness suites (noise, dropout, calibration drift). |
| Explanation | `explainer/*`, `explainer/run_explainer.py` | Llama‑3‑8B Instruct (4-bit) with self-consistency voting (k=5) and strict JSON validation (≥3 numeric claims). |
| Deployment Assist | `scripts/export_for_colab.py`, `FD_LLM_Colab_Explainer.ipynb`, `docs/COLAB_SETUP.md` | Export predictions + features to Drive, run the explainer on Colab, download structured explanations. |

## Data & Label Model

- **Sensors**: 11 inputs (flow, density, SG, percent solids, particle size, pressure, temperature, mass and solids flow).
- **Windows**: 60 s segments sampled at 1 Hz with 75 % overlap; each becomes `[n_windows, n_sensors, 60]`.
- **Labels**: 10 discrete classes (Normal + 9 faults) derived from maintenance logs and heuristics; configurable thresholds live in `experiments/configs/baseline.yaml`.
- **Quality Gates**: windows must keep ≥90 % valid samples, pass physics checks, and avoid constant/outlier sensors before training.

## Modeling Stack & Current Performance

| Layer | Details | Reference Metrics |
|-------|---------|------------------|
| MultiROCKET Baseline | 6,250 kernels, RidgeClassifierCV (`alphas` 1e-3…1e3), stratified 80/20 split | Macro F1 0.66, mean confidence 0.63 on 740-window test set (see `outputs/exp_full_dataset/results.json`). |
| Late Fusion (optional) | Separate MultiROCKET per sensor group (flow/density/process) concatenated before Ridge head | Activate via `model.fusion: late` in config to better handle sensors with different scales/rates. |
| LLM Explainer | Llama‑3‑8B Instruct w/QLoRA-ready config, 4-bit quantization, temperature 0.8, self-consistency k=5 | 2–5 s per window on Colab T4 (≈10× faster than CPU), each explanation includes final diagnosis, confidence, top evidence, cross-checks, and recommended actions. |

## Execution Modes

| Mode | When to Use | Classifier Runtime | LLM Runtime | Notes |
|------|-------------|--------------------|-------------|-------|
| **All Local** | Quick smoke tests, <100 windows | 5–10 min training on M-series CPU | 30–60 s/window | No external dependencies but slow explanations. |
| **Hybrid (Recommended)** | Standard workflow | Same as above | 2–5 s/window on Colab Pro GPU | Train + evaluate locally, upload parquet + `explainer/` folder, run `FD_LLM_Colab_Explainer.ipynb`, download JSONL. |
| **Full Cloud** | Future realtime deployment | TBD | TBD | Would containerize ROCKET + LLM; not implemented yet. |

## Scalability & Bottlenecks

1. **Memory footprint** – 10k windows ≈ 2–3 GB RAM; mitigate with chunked loading/export or reduced kernels.  
2. **LLM throughput** – CPU inference dominates runtime; Colab GPU yields ~10× speedup, and lowering `k` from 5→3 is another lever.  
3. **Storage** – Raw CSVs + parquet exports can exceed 5 GB; archive older runs or compress exports when syncing to Drive.  
4. **Configuration drift** – All knobs sit in `experiments/configs/baseline.yaml`; keep overrides in run-specific YAMLs checked into `experiments/configs/`.

## Security & Operational Considerations

- Data stays on local disk or the user’s Google Drive; no third-party APIs are called.
- LLM weights load from HuggingFace at runtime and stay inside the Colab session (cleared via `llm_explainer.free_memory()` + Drive cleanup).
- For regulated OT environments, use local-only mode or host the quantized weights on-prem; prompts already scrub PII/asset identifiers.

## Roadmap

| Phase | Status | Focus |
|-------|--------|-------|
| Phase 1 | ✅ Complete | MultiROCKET baseline, feature engineering, physical validation, CLI tooling. |
| Phase 2 | 🚧 Planned | PatchTST baseline, richer hydrodynamic features, live data adapters. |
| Phase 3 | 🧭 Future | Real-time streaming inference, OT dashboarding, automated remediation suggestions, cross-plant transfer learning. |

---

**Architecture version**: 1.1  
**Last updated**: 2025-10-12  
**Source of truth**: `outputs/exp_full_dataset` experiment files  
