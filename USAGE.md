# Usage Guide

This guide explains how to set up, run, train, test, and use **The Eyes of Truth** (DINO self-supervised learning demo on CIFAR-10 with FastAPI and Streamlit).

## Prerequisites

- Windows/PowerShell environment (project developed/tested in this context)
- Python 3.11+ recommended (a local `.venv` may already exist)
- Enough disk space for CIFAR-10 download (`data/`) and artifacts (`artifacts/`)
- Internet access on first run to download CIFAR-10 (and pretrained weights are **not** used by default)

> **Note:** Training runs on CPU by default with small epoch/batch settings to keep the demo tractable. It can still take noticeable time.

## Setup

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Run all commands from the project root so relative paths (`data/`, `artifacts/`, config defaults) resolve correctly.

## Configuration

Configuration is managed via Pydantic Settings in `configs/settings.py`.

- Defaults include: `device="cpu"`, CIFAR-10 paths, DINO/classifier hyperparameters, and artifact paths.
- MLflow tracking defaults to a local SQLite backend: `sqlite:///artifacts/mlflow.db`.
- Environment overrides use the `EOT_` prefix (and `.env` is supported if present). Examples:

```powershell
$env:EOT_DINO_EPOCHS = "1"
$env:EOT_FINETUNE_EPOCHS = "1"
$env:EOT_BATCH_SIZE = "16"
$env:EOT_MLFLOW_TRACKING_URI = "sqlite:///artifacts/mlflow.db"
```

Key defaults (see `configs/settings.py` for full list):
- Data/artifacts: `data_dir="data"`, `artifacts_dir="artifacts"`
- DINO: `dino_epochs=3`, `batch_size=32`, `global/local crops=24/16`
- Classifier: `finetune_epochs=3`, `num_classes=10`
- Model artifacts: `backbone_path="artifacts/dino_backbone.pt"`, `model_path="artifacts/classifier.pt"`

## Project workflow

Typical flow:
1. **Pretrain DINO backbone** (self-supervised) → saves `artifacts/dino_backbone.pt`
2. **Fine-tune classifier** (uses backbone if available) → saves `artifacts/classifier.pt`
3. **Run inference** via script, API, or Streamlit dashboard
4. **Explore evaluation/quality/embeddings/explainability** in Streamlit (requires trained model/artifacts and data)

## Training

### 1) Pretrain DINO backbone

```powershell
.\.venv\Scripts\python -m scripts.pretrain_dino
```

- Downloads CIFAR-10 to `data/` if needed.
- Logs MLflow run under `artifacts/mlflow.db` (SQLite) and prints epoch losses.
- Saves backbone to `artifacts/dino_backbone.pt`.

### 2) Fine-tune classifier

```powershell
.\.venv\Scripts\python -m scripts.finetune_classifier
```

Optional: specify a backbone checkpoint explicitly:

```powershell
.\.venv\Scripts\python -m scripts.finetune_classifier --backbone artifacts\dino_backbone.pt
```

- Trains/evaluates on CIFAR-10 and saves classifier checkpoint to `artifacts/classifier.pt`.
- MLflow may print: `Experiment with name 'classifier_finetune' does not exist. Creating a new experiment.` — this is informational on first run.

### 3) Scripted prediction

```powershell
.\.venv\Scripts\python -m scripts.predict --image path\to\image.png
```

- Requires `artifacts/classifier.pt` to exist.
- Outputs JSON with predicted class, confidence, and per-class probabilities.

## Running the app and API

### Streamlit dashboard

```powershell
.\.venv\Scripts\streamlit run app/Home.py
```

- Run from the **project root**.
- Pages: Overview, Training, Data Quality, Evaluation, Embeddings, Explainability, Inference.
- Some pages (Evaluation/Embeddings/Explainability/Inference) require a trained classifier at `artifacts/classifier.pt` and available CIFAR-10 data; they display warnings when prerequisites are missing.
- The app loads model artifacts directly (single-process/local demo).

### FastAPI inference service

```powershell
.\.venv\Scripts\python -m uvicorn api.main:app --reload
```

Endpoints:
- `GET /health` → `{ "status": "ok" }`
- `POST /predict` (multipart form file upload, field name `file`) → prediction JSON

Example:

```powershell
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@path\to\image.png"
```

`/predict` returns `503` if `artifacts/classifier.pt` is missing/unreadable.

## Testing and linting

Install dev dependencies if not already installed (included in `requirements.txt`):

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Run tests:

```powershell
.\.venv\Scripts\python -m pytest -q
```

Lint/format checks (as used in CI):

```powershell
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m black --check .
```

Quick syntax check:

```powershell
.\.venv\Scripts\python -m compileall -q app api src configs scripts tests
```

> **Note:** Some tests may download/use CIFAR-10 data depending on environment and cache state; run from project root with dependencies installed.

## Docker

A Dockerfile is provided at `docker/Dockerfile` (starts the FastAPI service by default).

```powershell
docker build -f docker/Dockerfile -t eyes-of-truth .
docker run --rm -p 8000:8000 eyes-of-truth
```

Mount or copy trained artifacts/data as needed if you want `/predict` to work inside the container.

## Artifacts and outputs

- `artifacts/dino_backbone.pt` — pretrained DINO backbone weights
- `artifacts/classifier.pt` — fine-tuned classifier checkpoint (includes class labels)
- `artifacts/mlflow.db` — local MLflow tracking database (default)
- `data/` — CIFAR-10 dataset (downloaded by torchvision)

## Troubleshooting

- **MLflow file-store error (`filesystem tracking backend ... in maintenance mode`)**: defaults now use `sqlite:///artifacts/mlflow.db`. If overridden, set `EOT_MLFLOW_TRACKING_URI` to a SQLite URI or use `MLFLOW_ALLOW_FILE_STORE=true` only if you intentionally need file-store behavior.
- **`ModuleNotFoundError: No module named 'configs'` / `'src'` in Streamlit**: run Streamlit from the project root (`.\.venv\Scripts\streamlit run app/Home.py`). The app includes a path bootstrap (`app/_path.py`) to add the project root to `sys.path`.
- **GitPython/MLflow message about git unavailable**: training scripts set `GIT_PYTHON_REFRESH=quiet` to avoid blocking/noisy behavior on machines without Git; install Git if you want git metadata logged.
- **`DLL load failed ... Application Control policy has blocked this file`**: indicates environment policy blocking compiled `.pyd` extensions in the venv. Resolve with IT/policy allow-listing or recreate/install packages in an allowed environment; as a local workaround, affected pure-Python-fallback packages may run if blocked extensions are disabled (environment-specific).
- **`/predict` 503 or Streamlit "model not found" warnings**: run pretraining/fine-tuning first to generate `artifacts/classifier.pt`.
- **Slow/unstable CPU training**: reduce epochs/batch size via `EOT_DINO_EPOCHS`, `EOT_FINETUNE_EPOCHS`, `EOT_BATCH_SIZE`.
