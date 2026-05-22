# Sleep-EDF BSPC Final Strengthened Pipeline

This ZIP is a GitHub-ready replacement package for:

```text
https://github.com/terry0809000/sleepingedf
```

It is designed to replace the previous notebook with a BSPC-strengthened version while preserving the repository structure.

## Main files

```text
notebooks/sleep_edf_pipeline_record_boundary_fixed.ipynb
notebooks/sleep_edf_pipeline_BSPC_FINAL_STRENGTHENED.ipynb
scripts/sleep_edf_pipeline_record_boundary_fixed.py
requirements.txt
.gitignore
docs/SECTION_INDEX.md
docs/GITHUB_REPLACE_COMMANDS.md
replace_original_notebook.sh
```

## What this version adds

- Final forced preprocessing-cache rebuild option: `FORCE_REBUILD_CACHE = True`.
- Preprocessing code-version hash to reduce stale-cache reuse.
- Full cross-validation per-stage precision, recall, F1, and support export.
- Fold-wise train/validation/test class-distribution export.
- Computational-cost tracking across folds.
- Repeated-seed sensitivity for CNN-LSTM and Transformer.
- Cross-fold channel ablation for EEG-only, EOG-only, and EEG+EOG.
- HMM smoothing reported as a trade-off by metric/model.
- Cautious entropy-deferral and calibration wording.

## Expected output CSVs after running

```text
cv_results_record_aware.csv
cv_per_stage_metrics.csv
cv_split_stage_distribution.csv
cv_computational_cost.csv
cv_hmm_tradeoff.csv
repeated_seed_sequence_results.csv
repeated_seed_sequence_per_stage_metrics.csv
repeated_seed_sequence_cost.csv
repeated_seed_sequence_summary.csv
channel_ablation_cv_results.csv
channel_ablation_cv_summary.csv
```

## Data note

Do not commit raw Sleep-EDF EDF files, zipped datasets, cache folders, trained models, or generated result folders to GitHub.

The notebook expects the Sleep-EDF Expanded archive in Google Drive by default:

```text
/content/drive/MyDrive/sleep-edf-database-expanded-1.0.0.zip
```

For local execution, edit the configuration cell or set local paths in the notebook/script.

## Quick GitHub replacement

See:

```text
docs/GITHUB_REPLACE_COMMANDS.md
replace_original_notebook.sh
```

The replacement path is:

```text
notebooks/sleep_edf_pipeline_record_boundary_fixed.ipynb
```

## Recommended workflow

1. Unzip this package.
2. Review the cleaned notebook locally.
3. Run the notebook in Colab/GPU runtime.
4. Confirm the exported CSVs are generated.
5. Commit the notebook and script to GitHub.
6. Keep generated outputs and raw data out of Git.
