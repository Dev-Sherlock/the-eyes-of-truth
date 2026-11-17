# The Eyes of Truth

DINO self-supervised learning demo on CIFAR-10 with FastAPI inference and a Streamlit dashboard.

## Quick start (Windows/PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt

# train (CPU-friendly defaults)
.\.venv\Scripts\python -m scripts.pretrain_dino
.\.venv\Scripts\python -m scripts.finetune_classifier

# run
.\.venv\Scripts\python -m uvicorn api.main:app --reload
.\.venv\Scripts\streamlit run app/Home.py
```

Run commands from the project root. Artifacts and MLflow tracking data are written under `artifacts/`.

## Documentation

- **[USAGE.md](USAGE.md)** — detailed guide: setup, configuration (`EOT_*`), training/fine-tuning, inference (script/API/Streamlit), testing/linting, Docker, and troubleshooting.
