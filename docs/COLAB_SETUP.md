# Running the Explainer on Google Colab

This guide describes the recommended “hybrid” workflow: keep the classifier training and evaluation on your local machine (fast on CPU), then push only the prediction artifacts to Colab for GPU‑accelerated LLM explanations.

---

## 0. Prerequisites

| Requirement | Notes |
|-------------|-------|
| Local environment | Project installed via `pip install -e .`, data under `data/raw/`, and at least one completed training run in `outputs/<run_id>/`. |
| Google Drive | Enough space for parquet exports (typically 200–500 MB) and explanation outputs. |
| Colab runtime | “GPU” hardware accelerator (T4 or better). Colab Pro shortens queue times but is optional. |
| Notebook | Use the checked-in `FD_LLM_Colab_Explainer.ipynb` or follow the cell-by-cell instructions below. |

---

## 1. Train Locally & Export Predictions

```bash
# 1) Train (MultiROCKET baseline shown)
python training/train_rocket.py \
  --config experiments/configs/baseline.yaml \
  --data data/raw/your_data.csv \
  --run_name plant_loop_a

# 2) Export predictions + features for Colab
python scripts/export_for_colab.py \
  --model outputs/plant_loop_a/model.pkl \
  --data data/raw/your_data.csv \
  --output outputs/plant_loop_a/predictions_for_colab.parquet
```

The export script automatically fills missing samples, re-computes features with `models/encoders/feature_extractor.py`, and writes a parquet file containing `window_id`, `prediction`, `confidence`, and JSON-serialized feature blobs.

Artifacts to upload afterwards:

```
outputs/
└── plant_loop_a/
    ├── model.pkl                         # (optional but handy for reference)
    ├── predictions_for_colab.parquet     # REQUIRED
    └── results.json                      # metrics, for traceability
explainer/                                # copy the entire folder
FD_LLM_Colab_Explainer.ipynb              # notebook (or use local copy)
```

---

## 2. Sync Files to Google Drive

1. Create a folder such as `MyDrive/fd-llm`.
2. Upload `explainer/`, the notebook, and your run-specific `outputs/<run_id>/`.
3. (Optional) Add a `logs/` subfolder if you want Colab to emit progress logs there.

> 💡 Tip: Google Drive uploads are faster with zipped archives. Upload `explainer.zip`, then unzip directly inside Colab with `!unzip explainer.zip -d /content/drive/MyDrive/fd-llm`.

---

## 3. Execute the Notebook on Colab

Each step below corresponds to a cell in `FD_LLM_Colab_Explainer.ipynb`. You can also paste into a fresh notebook.

1. **Runtime setup**
   ```python
   !nvidia-smi
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Install dependencies** (mirrors `requirements.txt` subset)
   ```python
   !pip install -q torch transformers peft bitsandbytes accelerate
   !pip install -q pandas numpy scikit-learn pyarrow tqdm
   ```

3. **Configure paths/models**
   ```python
   import sys, pathlib, json
   BASE = pathlib.Path('/content/drive/MyDrive/fd-llm')
   sys.path.append(str(BASE))

   CONFIG = {
       'predictions_file': BASE / 'outputs/plant_loop_a/predictions_for_colab.parquet',
       'output_file': BASE / 'outputs/plant_loop_a/explanations.jsonl',
       'model_name': 'mistralai/Mistral-7B-Instruct-v0.2',
       'load_in_4bit': True,
       'self_consistency_k': 5,
       'temperature': 0.8,
       'max_samples': None,     # set to e.g. 200 for smoke tests
   }
   ```

4. **Load the LLM**
   ```python
   from explainer.llm_setup import LLMExplainer
   llm_explainer = LLMExplainer(
       model_name=CONFIG['model_name'],
       load_in_4bit=CONFIG['load_in_4bit'],
       temperature=CONFIG['temperature']
   )
   ```

5. **Load predictions and iterate**
   ```python
   import pandas as pd, json, tqdm
   from explainer.self_consistency import explain_with_self_consistency
   pred_df = pd.read_parquet(CONFIG['predictions_file'])
   if CONFIG['max_samples']:
       pred_df = pred_df.head(CONFIG['max_samples'])

   explanations = []
   CONFIG['output_file'].parent.mkdir(parents=True, exist_ok=True)

   with open(CONFIG['output_file'], 'w') as writer:
       for _, row in tqdm.tqdm(pred_df.iterrows(), total=len(pred_df)):
           features = json.loads(row['features'])
           explanation = explain_with_self_consistency(
               llm_explainer=llm_explainer,
               features=features,
               prediction=row['prediction'],
               confidence=row['confidence'],
               k=CONFIG['self_consistency_k'],
               temperature=CONFIG['temperature']
           )
           explanation.update({
               'window_id': int(row['window_id']),
               'timestamp': pd.Timestamp.utcnow().isoformat()
           })
           writer.write(json.dumps(explanation) + '\n')
           writer.flush()
           explanations.append(explanation)
   ```

6. **Summaries & cleanup**
   ```python
   from collections import Counter
   diag_counts = Counter(exp['final_diagnosis'] for exp in explanations)
   print(diag_counts.most_common())
   avg_agreement = sum(exp['meta']['voting_agreement'] for exp in explanations) / len(explanations)
   print(f'Avg agreement: {avg_agreement:.1%}')

   llm_explainer.free_memory()
   ```

---

## 4. Retrieve Results & Continue Analysis

- The notebook writes directly to `outputs/<run_id>/explanations.jsonl` on Drive, so you can download the file via the Drive web UI or `files.download(...)` in Colab.
- Back on your workstation, pair that JSONL with the classifier predictions to perform deeper agreement analysis:
  ```bash
  python demo.py             # interactive overview (uses exp_full_dataset by default)
  python evaluation/metrics.py --run_id plant_loop_a
  ```

---

## Common Issues & Fixes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError` for `explainer.*` | Notebook cannot find repo code | Double-check that `sys.path.append(str(BASE))` points to the folder that contains `explainer/`. |
| CUDA OOM when loading LLM | Runtime not set to GPU or default precision too high | Enable GPU runtime and keep `load_in_4bit=True`. For longer prompts lower `max_new_tokens` (default 512). |
| JSON parsing failures | Model outputs reasoning text before JSON | Stick with `mistralai/Mistral-7B-Instruct-v0.2` or another JSON-friendly model (see README “Documentation Guide”). |
| Slow upload/download | Large parquet files | Compress with `zip`/`tar`, or limit `--max_samples` during dry runs. |
| Colab timeout mid-run | Sessions idle >90 min | Use `tqdm` progress, save after each explanation, and rerun cell with `max_samples` set to remaining windows. |

---

## Model Choices

- **Recommended**: `mistralai/Mistral-7B-Instruct-v0.2` – best JSON adherence, fits comfortably in 4-bit on a T4.
- **Alternatives**: `Qwen/Qwen2.5-7B-Instruct` (similar behavior) or `meta-llama/Llama-3.1-8B-Instruct` (requires HF access token). Avoid DeepSeek-R1 unless you explicitly want reasoning traces—the extra “thinking” text complicates JSON parsing.

---

## Time & Cost Reference

| Stage | Local CPU | Colab T4 | Notes |
|-------|-----------|----------|-------|
| MultiROCKET training | 5–10 min (3,700 windows) | N/A | Fully local workflow. |
| Export predictions | 2–3 min | N/A | Also local; produces parquet + JSON features. |
| LLM inference | 30–60 s/window | **2–5 s/window** | 10× speedup with GPU; k=5 self-consistency already included. |
| Total 500 windows | ~7 hrs | ~45 min | With Colab Pro ($9.99/mo) you can run multiple batches/day. |

Use this document alongside the notebook comments—the two stay in sync whenever we update the workflow.
