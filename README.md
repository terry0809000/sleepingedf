# Sleep-EDF Sleep Staging Pipeline

Repository-ready version of the Sleep-EDF sleep-stage classification notebook.

## Contents

- `notebooks/sleep_edf_pipeline_record_boundary_fixed.ipynb` — clean notebook version with outputs stripped for Git.
- `scripts/sleep_edf_pipeline_record_boundary_fixed.py` — linear Python export of the notebook code.
- `requirements.txt` — Python dependencies used by the notebook.
- `.gitignore` — prevents raw EDF data, caches, trained models, and generated outputs from being committed.
- `.github/workflows/syntax-check.yml` — lightweight syntax check for the exported Python script.

## Project summary

This pipeline builds a lightweight, subject-wise Sleep-EDF sleep staging workflow. It includes:

- Sleep-EDF local/Google Drive data loading.
- Subject-wise fold assignment to reduce leakage.
- Record-boundary-safe sequence construction.
- Classical feature extraction and Random Forest baseline.
- 1D CNN / CNN-LSTM neural models.
- Hidden Markov Model smoothing.
- Confusion matrices, macro-F1, Cohen's kappa, and interpretability-oriented outputs.
- Channel ablation and model persistence.

## Important data note

Do **not** commit raw Sleep-EDF EDF files, zip archives, cache folders, or trained model artefacts to GitHub.

The notebook expects the Sleep-EDF Expanded archive in Google Drive by default:

```text
/content/drive/MyDrive/sleep-edf-database-expanded-1.0.0.zip
```

or an extracted directory:

```text
/content/drive/MyDrive/sleep-edf-database-expanded-1.0.0/
```

For local execution, edit the paths in the global configuration cell or adapt the exported Python script.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Open:

```text
notebooks/sleep_edf_pipeline_record_boundary_fixed.ipynb
```

## Running as a script

The exported script is mainly for version control and reproducibility review:

```bash
python scripts/sleep_edf_pipeline_record_boundary_fixed.py
```

Interactive notebook execution is recommended, especially in Google Colab where Drive mounting and GPU availability are handled more naturally.

## Reproducibility notes

- Random seeds are set in Python, NumPy, and PyTorch.
- CuDNN benchmarking is disabled for more stable runs.
- Cross-validation is subject-wise rather than epoch-wise.
- Sequence windows are constrained within the same record/night to avoid record-boundary leakage.
- Generated results should be stored under local/Drive output directories and excluded from Git.
