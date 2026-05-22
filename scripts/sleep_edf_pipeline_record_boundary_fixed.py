"""Sleep-EDF BSPC final strengthened pipeline.

Linear Python export of notebooks/sleep_edf_pipeline_record_boundary_fixed.ipynb.
For interactive execution and figures, prefer the notebook.
Do not commit raw Sleep-EDF data, caches, trained models, or generated outputs.
"""



# ==============================================================================
# Section: Sleep-EDF BSPC Final Strengthened Pipeline
# ==============================================================================



# ==============================================================================
# Section: Sleep-EDF Sleep Staging: Lightweight, Subject-Wise, Interpretable Pipeline
# ==============================================================================



# ==============================================================================
# Section: Critical fixes applied in this version
# ==============================================================================



# ==============================================================================
# Section: Fixing plan mapped to this notebook revision
# ==============================================================================



# ==============================================================================
# Section: Drive-loading adjustment
# ==============================================================================



# ==============================================================================
# Section: 0. Install Dependencies
# ==============================================================================

# %% [code] Cell 6
# Install required packages
# Run once; restart kernel if needed
import subprocess, sys

packages = [
    'mne', 'pyEDFlib', 'scikit-learn', 'shap',
    'torch', 'torchvision',
    'antropy', 'yasa',
    'seaborn', 'matplotlib', 'pandas', 'numpy',
    'scipy', 'tqdm', 'joblib'
]

for pkg in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

print('All dependencies installed.')



# ==============================================================================
# Section: 1. Imports & Global Configuration
# ==============================================================================

# %% [code] Cell 8
import os, re, glob, warnings, random, zipfile, shutil, time, gc, json, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import pickle

import mne
mne.set_log_level('WARNING')

import scipy.signal as signal
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, cohen_kappa_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
import shap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Custom record-aware Viterbi smoothing is implemented below; no hmmlearn dependency is required.

warnings.filterwarnings('ignore')

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Deterministic-ish settings: fixed seeds plus stable CuDNN choices where practical.
# benchmark=False improves reproducibility; set True only if speed matters more.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ── Execution profile ────────────────────────────────────────────────────────
# BSPC final-strengthening profile.
# This profile is intentionally heavier than a debugging run:
#   - it rebuilds preprocessing cache once after the final parser is frozen;
#   - it runs full subject-wise CV for all model families;
#   - it exports per-stage metrics, split distributions, cost, repeated-seed,
#     and channel-ablation evidence needed for a journal-style manuscript.
#
# After one successful final preprocessing rebuild, you may set
# FORCE_REBUILD_CACHE=False for later analysis-only reruns.
PAPER_RUN = True
PAPER_RUN_DEEP_CV = True
FORCE_REBUILD_CACHE = True
FILTER_CONTINUOUS_BEFORE_EPOCHING = True  # Preferred signal-processing path.

# Journal-strengthening extensions.
RUN_REPEATED_SEED_CONFIRMATION = True
REPEATED_SEEDS = [42, 123, 2026]
REPEATED_SEED_MODELS = ['CNN-LSTM', 'Transformer']
REPEATED_SEED_FOLDS = 'all'   # 'all' or a list such as [0] for a cheaper sensitivity check.
REPEATED_SEED_EPOCHS = 15
REPEATED_SEED_PATIENCE = 5

RUN_CHANNEL_ABLATION_CV = True
CHANNEL_ABLATION_MODES = ['EEG only', 'EOG only', 'EEG + EOG']
CHANNEL_ABLATION_MODELS = ['Random Forest', '1D CNN', 'CNN-LSTM']
CHANNEL_ABLATION_DEEP_EPOCHS = 10
CHANNEL_ABLATION_DEEP_PATIENCE = 4

# ── Google Drive + Sleep-EDF local runtime setup ─────────────────────────────
# This pipeline now uses the same Drive location as your previous notebook:
#   /content/drive/MyDrive/sleep-edf-database-expanded-1.0.0.zip
#
# For speed, the zip is copied/extracted to local /content during each Colab
# runtime. Final outputs are saved back to Google Drive.

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    IN_COLAB = True
except Exception as e:
    print('Google Drive mount unavailable; continuing outside Colab:', e)
    IN_COLAB = False

DRIVE_ZIP = Path('/content/drive/MyDrive/sleep-edf-database-expanded-1.0.0.zip')
DRIVE_EXTRACTED_ROOT = Path('/content/drive/MyDrive/sleep-edf-database-expanded-1.0.0')

LOCAL_PARENT = Path('/content')
LOCAL_ZIP = LOCAL_PARENT / 'sleep-edf-database-expanded-1.0.0.zip'
LOCAL_DATA_ROOT_CANDIDATE = LOCAL_PARENT / 'sleep-edf-database-expanded-1.0.0'

# Keep raw EDF and cache in local /content for speed.
# Keep final results/splits in Drive so they persist.
RUNTIME_ROOT = Path('/content/sleep_edf_pipeline_runtime')
OUTPUT_ROOT = Path('/content/drive/MyDrive/SleepEDF_pipeline_outputs') if IN_COLAB else Path('./SleepEDF_pipeline_outputs')

CACHE_DIR  = RUNTIME_ROOT / 'cache'
SPLITS_DIR = OUTPUT_ROOT / 'splits'
RESULTS_DIR= OUTPUT_ROOT / 'results'
for d in [RUNTIME_ROOT, CACHE_DIR, SPLITS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def has_sleepedf_subfolders(path: Path) -> bool:
    path = Path(path)
    return (path / 'sleep-cassette').exists() or (path / 'sleep-telemetry').exists()

def locate_sleepedf_root(search_parent: Path):
    """Find the folder directly containing sleep-cassette/ or sleep-telemetry/."""
    search_parent = Path(search_parent)
    candidates = [
        search_parent,
        search_parent / 'sleep-edf-database-expanded-1.0.0',
        search_parent / 'sleep-edf-database-expanded-1.0.0' / 'sleep-edf-database-expanded-1.0.0',
    ]
    for c in candidates:
        if c.exists() and has_sleepedf_subfolders(c):
            return c.resolve()

    if search_parent.exists():
        for dirpath, dirnames, filenames in os.walk(search_parent):
            current = Path(dirpath)
            if current.name in ['sleep-cassette', 'sleep-telemetry']:
                return current.parent.resolve()
    return None

def prepare_sleepedf_from_drive():
    """
    Use the previous-notebook Drive path. Prefer local /content extraction for speed.
    This cell is safe to rerun in the same runtime: copy/extract are skipped if local
    files already exist.
    """
    # 1) Already extracted locally
    local_root = locate_sleepedf_root(LOCAL_PARENT)
    if local_root is not None:
        print('Using existing local Sleep-EDF root:', local_root)
        return local_root

    # 2) Zip in Drive: copy to /content, then extract locally
    if DRIVE_ZIP.exists():
        print('Found Drive zip:', DRIVE_ZIP)
        if not LOCAL_ZIP.exists():
            print('Copying zip from Drive to local /content. This may take several minutes once...')
            start = time.time()
            shutil.copy2(DRIVE_ZIP, LOCAL_ZIP)
            print(f'Copy complete in {(time.time() - start) / 60:.2f} minutes.')
        else:
            print('Local zip already exists; skipping copy.')

        print('Extracting local zip if needed...')
        start = time.time()
        with zipfile.ZipFile(LOCAL_ZIP, 'r') as zf:
            zf.extractall(LOCAL_PARENT)
        print(f'Extraction/check complete in {(time.time() - start) / 60:.2f} minutes.')

        local_root = locate_sleepedf_root(LOCAL_PARENT)
        if local_root is not None:
            return local_root

    # 3) Extracted folder in Drive: use it directly as fallback
    drive_root = locate_sleepedf_root(DRIVE_EXTRACTED_ROOT)
    if drive_root is not None:
        print('Using extracted Sleep-EDF folder directly from Drive:', drive_root)
        print('Note: this is slower than local /content extraction.')
        return drive_root

    raise FileNotFoundError(
        'Could not find Sleep-EDF. Expected either:\n'
        f'  {DRIVE_ZIP}\n'
        f'or extracted folder:\n'
        f'  {DRIVE_EXTRACTED_ROOT}\n'
        'containing sleep-cassette/ and/or sleep-telemetry/.'
    )

DATA_ROOT = prepare_sleepedf_from_drive()
DATA_DIR = DATA_ROOT  # raw EDF root used by the manifest builder

print('\nDATA_ROOT:', DATA_ROOT)
print('BASE_CACHE_DIR:', CACHE_DIR)
print('RESULTS_DIR:', RESULTS_DIR)
print('SPLITS_DIR:', SPLITS_DIR)
print('Top-level folders:', [p.name for p in DATA_ROOT.iterdir() if p.is_dir()])

# ── Constants ────────────────────────────────────────────────────────────────
EPOCH_SEC    = 30          # 30-second epochs
FS           = 100         # target sampling rate (Hz)
N_FOLDS      = 5           # subject-wise cross-validation folds
WAKE_TRIM    = 30 * 60     # seconds to retain around sleep (30 min)
SEQ_LEN      = 21          # epochs in sequence models context window
EPOCH_SAMPLES= EPOCH_SEC * FS  # 3000 samples per epoch

STAGE_MAP = {
    # Valid 5-class Sleep-EDF mapping. Unknown/movement annotations are ignored
    # during event construction rather than being assigned a numeric class.
    'Sleep stage W':  0,
    'Sleep stage 1':  1,
    'Sleep stage 2':  2,
    'Sleep stage 3':  3,
    'Sleep stage 4':  3,   # N3/N4 merged → N3
    'Sleep stage R':  4,
}
DISCARDED_STAGE_LABELS = {'Sleep stage ?', 'Movement time'}
STAGE_NAMES = ['W', 'N1', 'N2', 'N3', 'REM']
N_CLASSES   = 5

# Cache versioning: a new preprocessing folder is created whenever these
# settings change. This prevents silent reuse of stale cached epochs.
# PREPROCESSING_CODE_VERSION should be bumped whenever load_and_epoch() or
# annotation/wake-trimming logic changes, even if the numeric configuration does not.
PREPROCESSING_CODE_VERSION = 'bspc-final-2026-05-record-aware-v2'
PREPROCESSING_CONFIG = {
    'epoch_sec': EPOCH_SEC,
    'fs': FS,
    'wake_trim_sec': WAKE_TRIM,
    'stage_map': STAGE_MAP,
    'discarded_stage_labels': sorted(DISCARDED_STAGE_LABELS),
    'filter_continuous_before_epoching': FILTER_CONTINUOUS_BEFORE_EPOCHING,
    'bandpass_hz': [0.4, 30.0],
    'epoch_samples': EPOCH_SAMPLES,
    'preprocessing_code_version': PREPROCESSING_CODE_VERSION,
}
PREPROCESSING_HASH = hashlib.sha256(
    json.dumps(PREPROCESSING_CONFIG, sort_keys=True).encode('utf-8')
).hexdigest()[:10]

BASE_CACHE_DIR = CACHE_DIR
CACHE_DIR = BASE_CACHE_DIR / f'preproc_{PREPROCESSING_HASH}'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print('\nConfiguration loaded.')
print(f'Epoch: {EPOCH_SEC}s | FS: {FS}Hz | Folds: {N_FOLDS} | SeqLen: {SEQ_LEN}')
print(f'Preprocessing hash: {PREPROCESSING_HASH}')
print(f'Versioned CACHE_DIR: {CACHE_DIR}')



# ==============================================================================
# Section: 2. Data Download & Manifest
# ==============================================================================

# %% [code] Cell 10
# ============================================================
# Build manifest from local/Drive Sleep-EDF files
# ============================================================
# No MNE download is used here. The notebook reads your existing
# Sleep-EDF Expanded v1.0.0 data from DATA_ROOT.

MAX_SUBJECTS = None      # Full available SC benchmark. Set to 20 only for a quick smoke test.
USE_SUBSETS = ['sleep-cassette']  # Recommended primary benchmark. Add 'sleep-telemetry' if needed.

def psg_record_id(psg_path: Path) -> str:
    return psg_path.name.replace('-PSG.edf', '')

def hyp_record_id(hyp_path: Path) -> str:
    return hyp_path.name.replace('-Hypnogram.edf', '')

def pairing_key_from_record_id(record_id: str) -> str:
    # SC4001E0-PSG pairs with SC4001EC-Hypnogram; first 6 chars identify night.
    # ST7011J0-PSG pairs with ST7011JP-Hypnogram; first 6 chars identify night.
    return record_id[:6]

def parse_subject_label_and_night(record_id: str, subset_name: str):
    """
    Returns a stable subject label and night/record index from a Sleep-EDF record id.
    For SC records, SC4001E0 and SC4002E0 become same subject label SC_00
    with nights 1 and 2.
    """
    prefix = record_id[:2]
    if prefix == 'SC' and len(record_id) >= 6:
        subject_label = f"SC_{record_id[3:5]}"
        try:
            night = int(record_id[5])
        except Exception:
            night = 0
    elif prefix == 'ST' and len(record_id) >= 6:
        # Keep telemetry subjects distinct from cassette subjects.
        subject_label = f"ST_{record_id[3:5]}"
        try:
            night = int(record_id[5])
        except Exception:
            night = 0
    else:
        # Fallback for unexpected local file names.
        subject_label = f"{subset_name}_{record_id[:6]}"
        night = 0
    return subject_label, night

def build_manifest_from_sleepedf_root(data_root: Path, use_subsets=None, max_subjects=None):
    data_root = Path(data_root)
    if use_subsets is None:
        use_subsets = ['sleep-cassette']

    rows = []
    for subset_name in use_subsets:
        subset_dir = data_root / subset_name
        if not subset_dir.exists():
            print(f'Subset folder not found, skipping: {subset_dir}')
            continue

        psg_files = sorted(subset_dir.rglob('*-PSG.edf'))
        hyp_files = sorted(subset_dir.rglob('*-Hypnogram.edf'))

        hyp_lookup = {}
        for h in hyp_files:
            key = pairing_key_from_record_id(hyp_record_id(h))
            hyp_lookup.setdefault(key, []).append(h)

        print(f'{subset_name}: PSG={len(psg_files)}, Hypnogram={len(hyp_files)}')

        for psg in psg_files:
            rid = psg_record_id(psg)
            key = pairing_key_from_record_id(rid)
            hyp_candidates = hyp_lookup.get(key, [])
            if not hyp_candidates:
                print(f'WARNING: no hypnogram found for {psg.name}')
                continue

            hyp = hyp_candidates[0]
            subject_label, night = parse_subject_label_and_night(rid, subset_name)

            rows.append({
                'record_id': rid,
                'subject_label': subject_label,
                'night': night,
                'psg_path': str(psg),
                'hyp_path': str(hyp),
                'subset': 'SC' if subset_name == 'sleep-cassette' else 'ST',
            })

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError(
            f'No paired PSG/Hypnogram EDF files found under {data_root}. '
            'Check that DATA_ROOT contains sleep-cassette/ and/or sleep-telemetry/.'
        )

    # Assign numeric subject IDs for the existing pipeline.
    subject_labels = sorted(manifest['subject_label'].unique())
    if max_subjects is not None:
        subject_labels = subject_labels[:max_subjects]
        manifest = manifest[manifest['subject_label'].isin(subject_labels)].copy()

    subject_map = {s: i for i, s in enumerate(subject_labels)}
    manifest['subject_id'] = manifest['subject_label'].map(subject_map).astype(int)

    manifest = manifest.sort_values(['subject_id', 'night', 'record_id']).reset_index(drop=True)
    return manifest

manifest = build_manifest_from_sleepedf_root(
    DATA_ROOT,
    use_subsets=USE_SUBSETS,
    max_subjects=MAX_SUBJECTS
)

manifest_path = RESULTS_DIR / 'manifest.csv'
manifest.to_csv(manifest_path, index=False)

print(f'\nManifest saved to: {manifest_path}')
print(f'Manifest: {len(manifest)} nights from {manifest.subject_id.nunique()} subjects')
display(manifest.head(10))



# ==============================================================================
# Section: 3. Subject-Wise Fold Assignment
# ==============================================================================

# %% [code] Cell 12
from sklearn.model_selection import KFold

subjects = manifest['subject_id'].unique()
rng = np.random.default_rng(SEED)
subjects = rng.permutation(subjects)

if len(subjects) < 2:
    raise ValueError('Need at least 2 subjects for subject-wise splitting.')

# Avoid KFold error in smoke-test mode if MAX_SUBJECTS < N_FOLDS.
N_FOLDS = min(N_FOLDS, len(subjects))
print(f'Using N_FOLDS={N_FOLDS} for {len(subjects)} subjects.')

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_assignment = {}   # subject_id → fold index

for fold_idx, (_, test_idx) in enumerate(kf.split(subjects)):
    for s in subjects[test_idx]:
        fold_assignment[int(s)] = int(fold_idx)

manifest['fold'] = manifest['subject_id'].map(fold_assignment).astype(int)
manifest.to_csv(RESULTS_DIR / 'manifest.csv', index=False)  # update with folds

# Save frozen split file
split_df = pd.DataFrame({
    'subject_id': list(fold_assignment.keys()),
    'fold': list(fold_assignment.values())
}).sort_values(['fold', 'subject_id'])
split_df.to_csv(SPLITS_DIR / 'subject_folds.csv', index=False)

print(f'Subject-wise {N_FOLDS}-fold split saved to {SPLITS_DIR / "subject_folds.csv"}')
for f in range(N_FOLDS):
    n = (split_df.fold == f).sum()
    print(f'  Fold {f}: {n} subjects')



# ==============================================================================
# Section: 4. Preprocessing & Epoch Caching
# ==============================================================================

# %% [code] Cell 14
def _npz_scalar_to_py(value):
    """Convert np.load scalar/object arrays to plain Python values."""
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return value


def load_and_epoch(psg_path, hyp_path, wake_trim_sec=WAKE_TRIM, fs=FS,
                   epoch_sec=EPOCH_SEC, stage_map=STAGE_MAP):
    """
    Load one PSG night, epoch into 30 s windows, and return arrays.

    Fixes retained here:
    - Only valid sleep-stage annotations are mapped.
    - Unknown / movement annotations are ignored.
    - Event mapping is callable, so MNE does not fail when a stage label is absent.
    - Filtering/resampling is performed on the continuous recording before epoching
      when FILTER_CONTINUOUS_BEFORE_EPOCHING=True, reducing artificial 30 s edge
      artefacts at every epoch boundary.
    """
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    ann = mne.read_annotations(hyp_path)
    raw.set_annotations(ann, emit_warning=False)

    # Prefer continuous preprocessing before epoch construction.
    # Annotations remain attached to raw and MNE converts them to events after
    # resampling, preserving sample-index alignment.
    if FILTER_CONTINUOUS_BEFORE_EPOCHING:
        if raw.info['sfreq'] != fs:
            raw.resample(fs, npad='auto', verbose=False)
        raw.filter(0.4, 30.0, picks=['eeg', 'eog'], method='iir', verbose=False)

    def _event_id_from_annotation(description):
        return stage_map.get(description, None)

    events, evt_id = mne.events_from_annotations(
        raw,
        event_id=_event_id_from_annotation,
        chunk_duration=epoch_sec,
        verbose=False
    )

    if len(events) == 0:
        raise RuntimeError(f'No valid sleep-stage events found in {Path(hyp_path).name}')

    tmax = epoch_sec - 1.0 / raw.info['sfreq']
    epochs = mne.Epochs(
        raw,
        events,
        event_id=None,           # use numeric event codes already assigned above
        tmin=0.,
        tmax=tmax,
        baseline=None,
        picks=['eeg', 'eog'],
        preload=True,
        verbose=False
    )

    y = epochs.events[:, 2].astype(np.int64)

    # Wake trimming is applied within each night only.
    if wake_trim_sec is not None:
        sleep_mask = (y > 0)
        if sleep_mask.any():
            first_sleep = np.argmax(sleep_mask)
            last_sleep = len(sleep_mask) - 1 - np.argmax(sleep_mask[::-1])
            trim_epochs = int(wake_trim_sec / epoch_sec)
            start = max(0, first_sleep - trim_epochs)
            end = min(len(y), last_sleep + trim_epochs + 1)
            epochs = epochs[start:end]
            y = y[start:end]

    if len(y) == 0:
        raise RuntimeError(f'No epochs retained after wake trimming for {Path(psg_path).name}')

    # Fallback path: retained only for reproducibility comparisons with older runs.
    if not FILTER_CONTINUOUS_BEFORE_EPOCHING:
        if raw.info['sfreq'] != fs:
            epochs.resample(fs, verbose=False)
        epochs.filter(0.4, 30.0, method='iir', verbose=False)

    data = epochs.get_data()  # (n_epochs, n_channels, n_times)

    ch_names = [ch.lower() for ch in epochs.ch_names]
    eeg_idx = next((i for i, c in enumerate(ch_names)
                    if 'fpz' in c or 'eeg fpz' in c or c.startswith('eeg')), 0)
    eog_idx = next((i for i, c in enumerate(ch_names) if 'eog' in c), -1)

    X_eeg = data[:, eeg_idx, :]
    X_eog = data[:, eog_idx, :] if eog_idx >= 0 else np.zeros_like(X_eeg)

    # Convert V → µV.
    X_eeg = X_eeg * 1e6
    X_eog = X_eog * 1e6

    # Defensive shape check for downstream CNNs.
    if X_eeg.shape[1] != EPOCH_SAMPLES:
        raise RuntimeError(
            f'Unexpected epoch length {X_eeg.shape[1]} samples in {Path(psg_path).name}; '
            f'expected {EPOCH_SAMPLES}.'
        )

    return X_eeg.astype(np.float32), X_eog.astype(np.float32), y.astype(np.int64)


def cache_metadata_for_row(row):
    """Metadata stored with each cached record for stale-cache detection."""
    return {
        'record_id': str(row.get('record_id')),
        'subject_id': int(row.get('subject_id')),
        'subject_label': str(row.get('subject_label', row.get('subject_id'))),
        'night': int(row.get('night')),
        'subset': str(row.get('subset', '')),
        'fold': int(row.get('fold')),
        'preprocessing_hash': PREPROCESSING_HASH,
        'preprocessing_config_json': json.dumps(PREPROCESSING_CONFIG, sort_keys=True),
    }


def cache_file_is_valid(cache_path, expected_metadata):
    """Return True only if the cache was produced by the current preprocessing config."""
    if not Path(cache_path).exists():
        return False
    try:
        d = np.load(cache_path, allow_pickle=False)
        if 'preprocessing_hash' not in d:
            return False
        cached_hash = str(_npz_scalar_to_py(d['preprocessing_hash']))
        if cached_hash != expected_metadata['preprocessing_hash']:
            return False

        # Basic structural checks catch truncated/stale/corrupted npz files.
        required = ['eeg', 'eog', 'labels', 'record_id', 'subject_id', 'n_epochs']
        if any(k not in d for k in required):
            return False
        n = len(d['labels'])
        if d['eeg'].shape[0] != n or d['eog'].shape[0] != n:
            return False
        if d['eeg'].shape[1] != EPOCH_SAMPLES or d['eog'].shape[1] != EPOCH_SAMPLES:
            return False
        if str(_npz_scalar_to_py(d['record_id'])) != expected_metadata['record_id']:
            return False
        return True
    except Exception:
        return False


# ── Cache all nights ──────────────────────────────────────────────────────
failed = []
processed = 0
reused = 0
invalidated = 0

for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc='Preprocessing'):
    sid, night = int(row['subject_id']), int(row['night'])
    record_id = row.get('record_id', f'S{sid:02d}_N{night}')
    cache_path = CACHE_DIR / f'{record_id}.npz'
    expected_metadata = cache_metadata_for_row(row)

    if FORCE_REBUILD_CACHE and cache_path.exists():
        cache_path.unlink()

    if cache_path.exists() and cache_file_is_valid(cache_path, expected_metadata):
        reused += 1
        continue
    elif cache_path.exists():
        invalidated += 1
        cache_path.unlink()

    try:
        X_eeg, X_eog, y = load_and_epoch(row['psg_path'], row['hyp_path'])
        np.savez_compressed(
            cache_path,
            eeg=X_eeg,
            eog=X_eog,
            labels=y,
            n_epochs=int(len(y)),
            **expected_metadata,
        )
        processed += 1
    except Exception as e:
        failed.append((record_id, sid, night, str(e)))
        print(f'  FAILED {record_id} | subject={sid} night={night}: {e}')

print(f'\nCaching complete. Newly processed: {processed}. Reused: {reused}. Invalidated: {invalidated}. Failed: {len(failed)}')

expected_records = set(manifest['record_id'].astype(str))
cached_records = {p.stem for p in CACHE_DIR.glob('*.npz')}
missing_records = sorted(expected_records - cached_records)

print(f'Cached manifest records: {len(expected_records) - len(missing_records)} / {len(expected_records)}')

if failed:
    failed_df = pd.DataFrame(failed, columns=['record_id', 'subject_id', 'night', 'error'])
    display(failed_df)
    failed_df.to_csv(RESULTS_DIR / 'preprocessing_failures.csv', index=False)
    raise RuntimeError(
        f'Preprocessing failed for {len(failed)} records. '
        'Fix these before trusting any downstream model results.'
    )

if missing_records:
    raise RuntimeError(
        f'{len(missing_records)} manifest records are still missing from cache. '
        f'Examples: {missing_records[:5]}'
    )

cached = sorted(CACHE_DIR.glob('*.npz'))
print(f'Total cached files in versioned cache directory: {len(cached)}')



# ==============================================================================
# Section: 5. Data Loading Utilities
# ==============================================================================

# %% [code] Cell 16
def _npz_scalar_to_py(value):
    """Convert np.load scalar/object arrays to plain Python values."""
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return value


def load_cached_record(cache_path):
    """Load one cached night as a record dictionary and validate the cache hash."""
    d = np.load(cache_path, allow_pickle=False)
    if 'preprocessing_hash' not in d:
        raise RuntimeError(f'Cache file lacks preprocessing metadata: {cache_path}')
    cached_hash = str(_npz_scalar_to_py(d['preprocessing_hash']))
    if cached_hash != PREPROCESSING_HASH:
        raise RuntimeError(
            f'Stale cache file {cache_path}: hash={cached_hash}, expected={PREPROCESSING_HASH}. '
            'Delete/rebuild the cache or check PREPROCESSING_CONFIG.'
        )

    record = {
        'eeg': d['eeg'].astype(np.float32),
        'eog': d['eog'].astype(np.float32),
        'labels': d['labels'].astype(np.int64),
        'subject_id': int(_npz_scalar_to_py(d['subject_id'])),
        'subject_label': str(_npz_scalar_to_py(d['subject_label'])) if 'subject_label' in d else '',
        'night': int(_npz_scalar_to_py(d['night'])) if 'night' in d else -1,
        'record_id': str(_npz_scalar_to_py(d['record_id'])) if 'record_id' in d else Path(cache_path).stem,
        'fold': int(_npz_scalar_to_py(d['fold'])) if 'fold' in d else -1,
        'preprocessing_hash': cached_hash,
        'cache_path': str(cache_path),
    }
    n = len(record['labels'])
    assert record['eeg'].shape[0] == n and record['eog'].shape[0] == n, f'Shape mismatch in {cache_path}'
    assert record['eeg'].shape[1] == EPOCH_SAMPLES and record['eog'].shape[1] == EPOCH_SAMPLES, f'Epoch length mismatch in {cache_path}'
    return record


def assert_no_subject_overlap(train_records, val_records, test_records):
    """Hard leakage check: subjects must be disjoint across train/validation/test."""
    tr = {r['subject_id'] for r in train_records}
    va = {r['subject_id'] for r in val_records}
    te = {r['subject_id'] for r in test_records}
    overlaps = {
        'train_val': tr & va,
        'train_test': tr & te,
        'val_test': va & te,
    }
    bad = {k: sorted(v) for k, v in overlaps.items() if v}
    if bad:
        raise RuntimeError(f'Subject leakage detected across splits: {bad}')
    return True


def load_fold_records(fold_test, manifest, cache_dir=CACHE_DIR):
    """
    Load train / validation / test as lists of record dictionaries.

    This is the boundary-preserving loader. Downstream sequence windows, HMM
    smoothing, transition analysis and hypnogram plotting should use these
    records so that no operation crosses night or subject boundaries.
    """
    if N_FOLDS < 3:
        raise ValueError('At least 3 subject-wise folds are required for train/validation/test splitting.')

    test_subs = set(manifest.loc[manifest.fold == fold_test, 'subject_id'].astype(int))
    val_fold = (fold_test + 1) % N_FOLDS
    val_subs = set(manifest.loc[manifest.fold == val_fold, 'subject_id'].astype(int))
    train_subs = set(manifest['subject_id'].astype(int)) - test_subs - val_subs

    split_subjects = {
        'train': train_subs,
        'val': val_subs,
        'test': test_subs,
    }

    records_by_split = {'train': [], 'val': [], 'test': []}
    cache_paths = {p.stem: p for p in Path(cache_dir).glob('*.npz')}

    for _, row in manifest.sort_values(['subject_id', 'night', 'record_id']).iterrows():
        sid = int(row['subject_id'])
        record_id = str(row['record_id'])
        if record_id not in cache_paths:
            raise FileNotFoundError(f'Cached file missing for manifest record {record_id}')

        split_name = None
        for name, subject_set in split_subjects.items():
            if sid in subject_set:
                split_name = name
                break

        if split_name is None:
            continue

        rec = load_cached_record(cache_paths[record_id])
        records_by_split[split_name].append(rec)

    for split_name, records in records_by_split.items():
        if len(records) == 0:
            raise RuntimeError(f'No cached records found for {split_name} split of fold {fold_test}.')

    assert_no_subject_overlap(records_by_split['train'], records_by_split['val'], records_by_split['test'])
    return records_by_split['train'], records_by_split['val'], records_by_split['test']


def concat_records(records):
    """Concatenate a list of night records into epoch arrays while preserving order."""
    if not records:
        return np.array([]), np.array([]), np.array([])
    eeg = np.concatenate([r['eeg'] for r in records], axis=0)
    eog = np.concatenate([r['eog'] for r in records], axis=0)
    y = np.concatenate([r['labels'] for r in records], axis=0)
    return eeg, eog, y


def record_lengths(records):
    return [len(r['labels']) for r in records]


def record_slices(records):
    lengths = record_lengths(records)
    out, start = [], 0
    for n in lengths:
        out.append(slice(start, start + n))
        start += n
    return out


def split_concat_by_records(arr, records):
    """Split a concatenated epoch-level array back into record-level chunks."""
    return [arr[s] for s in record_slices(records)]


def load_fold_data(fold_test, manifest, cache_dir=CACHE_DIR, mode='raw'):
    """
    Backward-compatible array loader.

    Returns:
        (train_eeg, train_eog, train_y),
        (val_eeg, val_eog, val_y),
        (test_eeg, test_eog, test_y)

    For sequence/time-series operations, prefer load_fold_records().
    """
    tr_records, va_records, te_records = load_fold_records(fold_test, manifest, cache_dir)
    return concat_records(tr_records), concat_records(va_records), concat_records(te_records)


def class_distribution(y, names=STAGE_NAMES):
    counts = pd.Series(y).value_counts().reindex(range(len(names)), fill_value=0)
    df = pd.DataFrame({
        'Stage': names,
        'Count': counts.values,
        'Pct': (counts.values / max(counts.sum(), 1) * 100).round(1)
    })
    return df


# Quick check on fold 0.
tr_records, va_records, te_records = load_fold_records(0, manifest)
(tr_eeg, tr_eog, tr_y) = concat_records(tr_records)
(va_eeg, va_eog, va_y) = concat_records(va_records)
(te_eeg, te_eog, te_y) = concat_records(te_records)

print('Fold 0 data shapes:')
print(f'  Train : {len(tr_records)} records | EEG {tr_eeg.shape}, labels {tr_y.shape}')
print(f'  Val   : {len(va_records)} records | EEG {va_eeg.shape}, labels {va_y.shape}')
print(f'  Test  : {len(te_records)} records | EEG {te_eeg.shape}, labels {te_y.shape}')

assert len(tr_y) > 0 and len(va_y) > 0 and len(te_y) > 0
assert sum(record_lengths(tr_records)) == len(tr_y)
assert sum(record_lengths(va_records)) == len(va_y)
assert sum(record_lengths(te_records)) == len(te_y)
assert_no_subject_overlap(tr_records, va_records, te_records)

print('\nClass distribution (train):')
display(class_distribution(tr_y))



# ==============================================================================
# Section: 6. Feature Extraction for Random Forest Baseline
# ==============================================================================

# %% [code] Cell 18
try:
    import antropy as ant
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'antropy'])
    import antropy as ant


BANDS = {
    'delta':  (0.5,  4.0),
    'theta':  (4.0,  8.0),
    'alpha':  (8.0, 13.0),
    'sigma': (12.0, 15.0),   # sleep spindles
    'beta':  (15.0, 30.0),
}

def bandpower(x, fs, fmin, fmax):
    f, pxx = signal.welch(x, fs=fs, nperseg=min(len(x), 4*fs))
    idx = np.logical_and(f >= fmin, f <= fmax)
    return np.trapz(pxx[idx], f[idx])


def extract_features_epoch(eeg, eog, fs=FS):
    feats = []

    # ── EEG absolute and relative band powers ────────────────────────────
    total_power = bandpower(eeg, fs, 0.5, 30.0) + 1e-10
    for band, (lo, hi) in BANDS.items():
        absp = bandpower(eeg, fs, lo, hi)
        feats += [absp, absp / total_power]   # absolute + relative

    # Power ratios
    d = bandpower(eeg, fs, 0.5, 4.0) + 1e-10
    t = bandpower(eeg, fs, 4.0, 8.0) + 1e-10
    a = bandpower(eeg, fs, 8.0, 13.0) + 1e-10
    b = bandpower(eeg, fs, 15.0, 30.0) + 1e-10
    feats += [d/t, d/a, (d+t)/a, a/b]

    # ── Statistical features ──────────────────────────────────────────────
    feats += [
        np.std(eeg),
        skew(eeg),
        kurtosis(eeg),
        np.mean(np.abs(np.diff(eeg))),   # mean absolute difference
        np.sum(np.abs(np.diff(np.sign(eeg)))) / 2,  # zero crossing rate
    ]

    # ── Hjorth parameters ────────────────────────────────────────────────
    d1  = np.diff(eeg)
    d2  = np.diff(d1)
    var0 = np.var(eeg)  + 1e-10
    var1 = np.var(d1)   + 1e-10
    var2 = np.var(d2)   + 1e-10
    mobility   = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / mobility
    feats += [mobility, complexity]

    # ── Nonlinear features ────────────────────────────────────────────────
    try:
        feats.append(ant.perm_entropy(eeg, normalize=True))
        feats.append(ant.higuchi_fd(eeg))
    except Exception:
        feats += [0.0, 0.0]

    # ── EOG features ──────────────────────────────────────────────────────
    eog_total = bandpower(eog, fs, 0.5, 4.0) + 1e-10  # slow eye movements
    feats += [
        bandpower(eog, fs, 0.5, 4.0),
        np.std(eog),
        np.mean(np.abs(np.diff(eog))),
    ]

    return np.array(feats, dtype=np.float32)


def extract_features_batch(X_eeg, X_eog, n_jobs=-1):
    if len(X_eeg) == 0:
        raise ValueError('Cannot extract features from an empty epoch array.')
    results = Parallel(n_jobs=n_jobs)(
        delayed(extract_features_epoch)(X_eeg[i], X_eog[i])
        for i in range(len(X_eeg))
    )
    return np.stack(results)


# Feature names for SHAP plots.
# Important: the order must exactly match extract_features_epoch().
FEATURE_NAMES = []
for b in BANDS:
    FEATURE_NAMES.extend([f'{b}_abs', f'{b}_rel'])
FEATURE_NAMES += [
    'd/t', 'd/a', '(d+t)/a', 'a/b',
    'std', 'skew', 'kurt', 'mad', 'zcr',
    'hjorth_mob', 'hjorth_cmp',
    'perm_ent', 'higuchi_fd',
    'eog_delta_pwr', 'eog_std', 'eog_mad'
]

_expected_feature_len = len(extract_features_epoch(np.zeros(EPOCH_SAMPLES), np.zeros(EPOCH_SAMPLES)))
assert len(FEATURE_NAMES) == _expected_feature_len, (
    f'FEATURE_NAMES length {len(FEATURE_NAMES)} does not match extracted feature length {_expected_feature_len}'
)

print(f'Feature vector length: {len(FEATURE_NAMES)}')
print('Extracting features for fold 0 train set (may take a minute)...')
tr_feat = extract_features_batch(tr_eeg, tr_eog)
va_feat = extract_features_batch(va_eeg, va_eog)
te_feat = extract_features_batch(te_eeg, te_eog)
print(f'Train features: {tr_feat.shape}, Test features: {te_feat.shape}')



# ==============================================================================
# Section: 7. Evaluation Utilities
# ==============================================================================

# %% [code] Cell 20
def evaluate(y_true, y_pred, model_name='Model', stage_names=STAGE_NAMES):
    """Compute and display macro-F1, kappa, per-class F1, and confusion matrix."""
    labels = list(range(len(stage_names)))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mf1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels)
    acc = (y_true == y_pred).mean() if len(y_true) else np.nan

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=stage_names,
        zero_division=0
    )

    print(f'\n══ {model_name} ══')
    print(f'  Macro-F1 : {mf1:.4f}  |  Cohen κ : {kappa:.4f}  |  Accuracy : {acc:.4f}')
    print(report)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=stage_names)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', model_name)
    ax.set_title(f'{model_name} — Confusion Matrix')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'cm_{safe_name}.png', dpi=120)
    plt.show()

    return {'model': model_name, 'macro_f1': mf1, 'kappa': kappa, 'accuracy': acc}


def make_class_weight_dict(y, n_classes=N_CLASSES):
    """Balanced-ish class-weight dictionary robust to absent classes."""
    y = np.asarray(y)
    counts = np.bincount(y[y >= 0], minlength=n_classes)
    max_count = max(counts.max(), 1)
    return {i: float(max_count / max(counts[i], 1)) for i in range(n_classes)}


def make_class_weight_vector(y, n_classes=N_CLASSES):
    weights = make_class_weight_dict(y, n_classes)
    return np.array([weights[i] for i in range(n_classes)], dtype=np.float32)


def append_metric_row(rows, fold, model, y_true, y_pred):
    labels = list(range(N_CLASSES))
    rows.append({
        'fold': fold,
        'model': model,
        'macro_f1': f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
        'kappa': cohen_kappa_score(y_true, y_pred, labels=labels),
        'accuracy': (np.asarray(y_true) == np.asarray(y_pred)).mean()
    })


def per_record_metrics(y_true_concat, y_pred_concat, records, model_name):
    """
    Compute one row per night/record. This prevents the final interpretation from
    depending only on epoch-pooled metrics, where longer records dominate.
    """
    rows = []
    y_true_concat = np.asarray(y_true_concat)
    y_pred_concat = np.asarray(y_pred_concat)

    for rec, sl in zip(records, record_slices(records)):
        y_t = y_true_concat[sl]
        y_p = y_pred_concat[sl]
        rows.append({
            'model': model_name,
            'record_id': rec.get('record_id', ''),
            'subject_id': rec.get('subject_id', np.nan),
            'subject_label': rec.get('subject_label', ''),
            'night': rec.get('night', np.nan),
            'n_epochs': int(len(y_t)),
            'macro_f1': f1_score(y_t, y_p, labels=list(range(N_CLASSES)), average='macro', zero_division=0),
            'kappa': cohen_kappa_score(y_t, y_p, labels=list(range(N_CLASSES))),
            'accuracy': float((y_t == y_p).mean()) if len(y_t) else np.nan,
        })
    return pd.DataFrame(rows)


def summarize_record_metrics(record_metric_df):
    """Summarise per-record metrics as mean/SD; useful alongside epoch-pooled metrics."""
    return (record_metric_df
            .groupby('model')
            .agg(record_macro_f1_mean=('macro_f1', 'mean'),
                 record_macro_f1_sd=('macro_f1', 'std'),
                 record_kappa_mean=('kappa', 'mean'),
                 record_kappa_sd=('kappa', 'std'),
                 n_records=('record_id', 'count'))
            .sort_values('record_macro_f1_mean', ascending=False)
            .reset_index())


def multiclass_brier_score(y_true, proba, n_classes=N_CLASSES):
    """Mean squared error between one-hot labels and predicted class probabilities."""
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    Y = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((proba - Y) ** 2, axis=1)))


def expected_calibration_error(y_true, proba, n_bins=15):
    """
    Confidence-based multiclass ECE.
    This is a diagnostic, not a definitive calibration proof.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi) if hi < 1.0 else (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = conf[mask].mean()
        weight = mask.mean()
        ece += weight * abs(bin_acc - bin_conf)
        rows.append({
            'bin_low': lo,
            'bin_high': hi,
            'n': int(mask.sum()),
            'accuracy': float(bin_acc),
            'confidence': float(bin_conf),
            'abs_gap': float(abs(bin_acc - bin_conf)),
        })
    return float(ece), pd.DataFrame(rows)


def align_class_proba(proba, classes, n_classes=N_CLASSES):
    """Ensure classifier probabilities have one column per sleep-stage class."""
    proba = np.asarray(proba)
    classes = np.asarray(classes, dtype=int)
    if proba.shape[1] == n_classes and np.array_equal(classes, np.arange(n_classes)):
        return proba
    out = np.zeros((proba.shape[0], n_classes), dtype=float)
    for j, cls in enumerate(classes):
        if 0 <= cls < n_classes:
            out[:, cls] = proba[:, j]
    row_sums = out.sum(axis=1, keepdims=True)
    out = np.divide(out, np.maximum(row_sums, 1e-12))
    return out


results_log = []   # accumulate fold-0 results across models

# %% [code] Cell 21
# ============================================================
# BSPC final-strengthening utilities
# ============================================================
# These utilities save the evidence needed for a journal-style manuscript:
# fold-wise class distribution, per-stage metrics, computational cost,
# repeated-seed sensitivity, and channel ablation.

from sklearn.metrics import precision_recall_fscore_support
import tempfile
import os
import time

def set_global_seed(seed):
    """Set Python, NumPy, and PyTorch seeds for a reproducible training attempt."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_generator_for(seed):
    """Return a seeded torch.Generator for deterministic DataLoader shuffling."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def model_num_parameters(model):
    """Number of trainable parameters."""
    if model is None:
        return np.nan
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def model_size_mb(model):
    """
    Approximate PyTorch model parameter size in megabytes.
    This is parameter tensor size, not full checkpoint size.
    """
    if model is None:
        return np.nan
    return float(sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6)


def sklearn_object_size_mb(obj):
    """Approximate serialized size of a scikit-learn object in megabytes."""
    try:
        return float(len(pickle.dumps(obj)) / 1e6)
    except Exception:
        return np.nan


def timed_predict_proba_torch(model, loader):
    """Run predict_proba and return probabilities plus wall-clock inference seconds."""
    start = time.perf_counter()
    proba, y = predict_proba(model, loader)
    infer_seconds = time.perf_counter() - start
    return proba, y, infer_seconds


def append_per_class_rows(rows, fold, model, y_true, y_pred, split='test', seed=SEED):
    """Append one precision/recall/F1/support row per sleep stage."""
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred,
        labels=list(range(N_CLASSES)),
        zero_division=0
    )
    for i, stage in enumerate(STAGE_NAMES):
        rows.append({
            'seed': int(seed),
            'fold': int(fold),
            'split': split,
            'model': model,
            'stage': stage,
            'precision': float(p[i]),
            'recall': float(r[i]),
            'f1': float(f1[i]),
            'support': int(support[i]),
        })


def append_split_distribution_rows(rows, fold, split_name, y, records=None):
    """
    Append stage counts and proportions for a split in a given fold.
    Records are optional but used to report subject/record counts.
    """
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y[y >= 0], minlength=N_CLASSES)
    n_epochs = int(counts.sum())
    n_records = len(records) if records is not None else np.nan
    n_subjects = len({str(r.get('subject_id')) for r in records}) if records is not None else np.nan
    for class_id, stage in enumerate(STAGE_NAMES):
        rows.append({
            'fold': int(fold),
            'split': split_name,
            'stage': stage,
            'class_id': int(class_id),
            'n_epochs': int(counts[class_id]),
            'proportion': float(counts[class_id] / n_epochs) if n_epochs else np.nan,
            'total_epochs_in_split': n_epochs,
            'n_records_in_split': int(n_records) if not pd.isna(n_records) else np.nan,
            'n_subjects_in_split': int(n_subjects) if not pd.isna(n_subjects) else np.nan,
        })


def append_cost_row(rows, fold, model, train_seconds=np.nan, inference_seconds=np.nan,
                    n_train_epochs=np.nan, n_test_epochs=np.nan, n_parameters=np.nan,
                    model_size_mb_value=np.nan, device=None, seed=SEED, notes=''):
    rows.append({
        'seed': int(seed),
        'fold': int(fold),
        'model': model,
        'train_seconds': float(train_seconds) if not pd.isna(train_seconds) else np.nan,
        'inference_seconds': float(inference_seconds) if not pd.isna(inference_seconds) else np.nan,
        'n_train_epochs': int(n_train_epochs) if not pd.isna(n_train_epochs) else np.nan,
        'n_test_epochs': int(n_test_epochs) if not pd.isna(n_test_epochs) else np.nan,
        'n_parameters': int(n_parameters) if not pd.isna(n_parameters) else np.nan,
        'model_size_mb': float(model_size_mb_value) if not pd.isna(model_size_mb_value) else np.nan,
        'device': str(device or DEVICE),
        'notes': notes,
    })


def append_hmm_tradeoff_row(rows, fold, base_model, y_true, base_pred, smoothed_pred, records, seed=SEED):
    """
    HMM is reported as a trade-off: classification metrics and large-jump counts
    are both saved. The manuscript should only claim improvement for the specific
    metric/model combination where the saved delta is positive.
    """
    base_mf1 = f1_score(y_true, base_pred, labels=list(range(N_CLASSES)), average='macro', zero_division=0)
    smooth_mf1 = f1_score(y_true, smoothed_pred, labels=list(range(N_CLASSES)), average='macro', zero_division=0)
    base_kappa = cohen_kappa_score(y_true, base_pred, labels=list(range(N_CLASSES)))
    smooth_kappa = cohen_kappa_score(y_true, smoothed_pred, labels=list(range(N_CLASSES)))
    base_acc = float((np.asarray(y_true) == np.asarray(base_pred)).mean())
    smooth_acc = float((np.asarray(y_true) == np.asarray(smoothed_pred)).mean())
    base_large_jumps = count_illegal_by_records(base_pred, records)
    smooth_large_jumps = count_illegal_by_records(smoothed_pred, records)
    rows.append({
        'seed': int(seed),
        'fold': int(fold),
        'base_model': base_model,
        'smoothed_model': f'{base_model} + HMM',
        'macro_f1_before': float(base_mf1),
        'macro_f1_after': float(smooth_mf1),
        'macro_f1_delta_after_minus_before': float(smooth_mf1 - base_mf1),
        'kappa_before': float(base_kappa),
        'kappa_after': float(smooth_kappa),
        'kappa_delta_after_minus_before': float(smooth_kappa - base_kappa),
        'accuracy_before': float(base_acc),
        'accuracy_after': float(smooth_acc),
        'accuracy_delta_after_minus_before': float(smooth_acc - base_acc),
        'large_jump_transitions_before': int(base_large_jumps),
        'large_jump_transitions_after': int(smooth_large_jumps),
        'large_jump_delta_after_minus_before': int(smooth_large_jumps - base_large_jumps),
        'interpretation_note': (
            'Large-jump transition count is a plausibility heuristic, not a formal clinical impossibility rule.'
        )
    })


def apply_channel_mode(eeg, eog, mode):
    """
    Return EEG/EOG arrays according to a channel-ablation mode.
    EEG-only and EOG-only keep the same two-input interface by zeroing the omitted channel.
    """
    eeg = np.asarray(eeg)
    eog = np.asarray(eog)
    zeros_eeg = np.zeros_like(eeg)
    zeros_eog = np.zeros_like(eog)
    if mode == 'EEG only':
        return eeg, zeros_eog
    if mode == 'EOG only':
        return zeros_eeg, eog
    if mode == 'EEG + EOG':
        return eeg, eog
    raise ValueError(f'Unknown channel mode: {mode}')


def records_with_channel_mode(records, mode):
    """Shallow-copy record dictionaries while replacing EEG/EOG arrays for ablation."""
    out = []
    for rec in records:
        rec2 = dict(rec)
        rec2['eeg'], rec2['eog'] = apply_channel_mode(rec['eeg'], rec['eog'], mode)
        out.append(rec2)
    return out


def save_many_csv(prefix, **named_frames):
    """Save multiple DataFrames with consistent filenames."""
    for name, df in named_frames.items():
        if df is not None and len(df) > 0:
            Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
            pd.DataFrame(df).to_csv(RESULTS_DIR / f'{prefix}_{name}.csv', index=False)



# ==============================================================================
# Section: 8. Random Forest Baseline + SHAP
# ==============================================================================

# %% [code] Cell 23
# ── Train RF ──────────────────────────────────────────────────────────────
scaler_rf = StandardScaler()
tr_feat_sc = scaler_rf.fit_transform(tr_feat)
va_feat_sc = scaler_rf.transform(va_feat)
te_feat_sc = scaler_rf.transform(te_feat)

class_weights = make_class_weight_dict(tr_y)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight=class_weights,
    n_jobs=-1,
    random_state=SEED
)
rf.fit(tr_feat_sc, tr_y)

rf_pred = rf.predict(te_feat_sc)
rf_proba = align_class_proba(rf.predict_proba(te_feat_sc), rf.classes_)
rf_val_proba = align_class_proba(rf.predict_proba(va_feat_sc), rf.classes_)

res_rf = evaluate(te_y, rf_pred, model_name='Random Forest')
results_log.append(res_rf)

# %% [code] Cell 24
import numpy as np

# ── SHAP Feature Importance ───────────────────────────────────────────────
print('Computing SHAP values (TreeExplainer)...')
explainer = shap.TreeExplainer(rf)

# Use a subsample for speed.
n_shap = min(500, len(te_feat_sc))
idx_shap = np.random.default_rng(SEED).choice(len(te_feat_sc), n_shap, replace=False)
shap_vals = explainer.shap_values(te_feat_sc[idx_shap])

def multiclass_mean_abs_shap(shap_values):
    """
    Return one mean-|SHAP| value per feature.

    Handles both common SHAP outputs:
    - list[class] of arrays shaped (n_samples, n_features)
    - array shaped (n_samples, n_features, n_classes)
    """
    if isinstance(shap_values, list):
        arr = np.stack([np.abs(sv) for sv in shap_values], axis=0)  # class, sample, feature
        return arr.mean(axis=(0, 1))
    arr = np.abs(np.asarray(shap_values))
    if arr.ndim == 3:
        return arr.mean(axis=(0, 2))  # sample and class
    if arr.ndim == 2:
        return arr.mean(axis=0)       # sample
    raise ValueError(f'Unexpected SHAP array shape: {arr.shape}')

shap_importance = multiclass_mean_abs_shap(shap_vals)
assert len(shap_importance) == len(FEATURE_NAMES), (
    f'SHAP importance length {len(shap_importance)} does not match feature names {len(FEATURE_NAMES)}'
)

top_k = np.argsort(shap_importance)[::-1][:15]

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh([FEATURE_NAMES[i] for i in top_k[::-1]],
         shap_importance[top_k[::-1]], color='steelblue')
ax.set_xlabel('Mean |SHAP| (across all classes)')
ax.set_title('Top 15 Feature Importances (SHAP)')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'shap_importance.png', dpi=120)
plt.show()



# ==============================================================================
# Section: 9. 1D CNN — Single-Epoch Deep Model
# ==============================================================================

# %% [code] Cell 26
# ── Dataset ───────────────────────────────────────────────────────────────
class EpochDataset(Dataset):
    """Single-epoch dataset: (EEG, EOG) concatenated as 2-channel input."""
    def __init__(self, eeg, eog, labels):
        self.X = torch.FloatTensor(
            np.stack([eeg, eog], axis=1) / 100.0   # rough amplitude norm
        )  # (N, 2, 3000)
        self.y = torch.LongTensor(labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Architecture ──────────────────────────────────────────────────────────
class CNN1D(nn.Module):
    """Lightweight 1D CNN for single-epoch classification."""
    def __init__(self, in_ch=2, n_classes=N_CLASSES):
        super().__init__()
        self.conv = nn.Sequential(
            # Block 1: large kernel to capture slow waves (0.5–4 Hz)
            nn.Conv1d(in_ch, 64, kernel_size=50, stride=6, padding=25),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.MaxPool1d(8, 8),

            # Block 2: medium kernel for spindles / alpha
            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.MaxPool1d(4, 4),

            # Block 3
            nn.Conv1d(128, 256, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4, 256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.conv(x))

    def embed(self, x):
        return self.conv(x).flatten(1)   # for CNN-LSTM


# ── Training helper ───────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, epochs=30, lr=1e-3,
                class_weights_tensor=None, patience=7):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    best_val_f1, best_state, wait = -np.inf, None, 0
    history = {'train_loss': [], 'val_f1': []}

    for ep in range(epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds.append(model(xb.to(DEVICE)).argmax(1).cpu())
                trues.append(yb)
        val_f1 = f1_score(torch.cat(trues), torch.cat(preds),
                           average='macro', zero_division=0)

        history['train_loss'].append(np.mean(losses))
        history['val_f1'].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f'  Early stop at epoch {ep+1}')
                break

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f'  Ep {ep+1:3d} | loss {np.mean(losses):.4f} | val MF1 {val_f1:.4f}')

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict_proba(model, loader):
    model.eval()
    probas, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(DEVICE))
            probas.append(F.softmax(logits, dim=-1).cpu().numpy())
            trues.append(yb.numpy())
    return np.concatenate(probas), np.concatenate(trues)


# ── Build data loaders ────────────────────────────────────────────────────
cw = torch.FloatTensor(make_class_weight_vector(tr_y)).to(DEVICE)

tr_ds = EpochDataset(tr_eeg, tr_eog, tr_y)
va_ds = EpochDataset(va_eeg, va_eog, va_y)
te_ds = EpochDataset(te_eeg, te_eog, te_y)

tr_ld = DataLoader(tr_ds, batch_size=256, shuffle=True,  num_workers=0)
va_ld = DataLoader(va_ds, batch_size=256, shuffle=False, num_workers=0)
te_ld = DataLoader(te_ds, batch_size=256, shuffle=False, num_workers=0)

# ── Train 1D CNN ──────────────────────────────────────────────────────────
print('Training 1D CNN...')
cnn_model = CNN1D().to(DEVICE)
n_params   = sum(p.numel() for p in cnn_model.parameters() if p.requires_grad)
print(f'  Parameters: {n_params:,}')

cnn_model, cnn_hist = train_model(cnn_model, tr_ld, va_ld, epochs=30,
                                   class_weights_tensor=cw)

cnn_val_proba, _ = predict_proba(cnn_model, va_ld)
cnn_proba, _ = predict_proba(cnn_model, te_ld)
cnn_pred     = cnn_proba.argmax(axis=1)
res_cnn = evaluate(te_y, cnn_pred, model_name='1D CNN')
results_log.append(res_cnn)



# ==============================================================================
# Section: 10. CNN-LSTM — Lightweight Sequence Model (TinySleepNet-style)
# ==============================================================================

# %% [code] Cell 28
class SeqEpochDataset(Dataset):
    """
    Boundary-safe sequence dataset.

    Each item is a context window centred on one epoch, but the window is built
    from the same night/record only. Edge context is padded by clipping within
    that record. This prevents the earlier error where windows crossed from one
    subject/night into the next after concatenation.
    """
    def __init__(self, records, seq_len=SEQ_LEN):
        self.records = records
        self.seq_len = seq_len
        self.half = seq_len // 2
        self.index = []

        for r_idx, rec in enumerate(records):
            n = len(rec['labels'])
            for epoch_idx in range(n):
                self.index.append((r_idx, epoch_idx))

        if len(self.index) == 0:
            raise ValueError('SeqEpochDataset received no epochs.')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        r_idx, center_idx = self.index[idx]
        rec = self.records[r_idx]
        n = len(rec['labels'])

        rel = np.arange(center_idx - self.half, center_idx + self.half + 1)
        rel = np.clip(rel, 0, n - 1)

        win = np.stack([rec['eeg'][rel], rec['eog'][rel]], axis=1)  # (seq_len, 2, samples)
        label = rec['labels'][center_idx]

        return torch.FloatTensor(win / 100.0), torch.LongTensor([label]).squeeze(0)


class CNNLSTM(nn.Module):
    """
    TinySleepNet-style: shared CNN encoder per epoch,
    then bidirectional LSTM over the sequence.
    """
    def __init__(self, seq_len=SEQ_LEN, n_classes=N_CLASSES,
                 lstm_hidden=128, lstm_layers=1):
        super().__init__()
        self.seq_len = seq_len

        # Epoch-level CNN encoder
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 64,  kernel_size=50, stride=6, padding=25),
            nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(8, 8),
            nn.Conv1d(64, 128, kernel_size=8, padding=4),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=8, padding=4),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),                  # → 512-dim
        )
        cnn_out = 128 * 4

        self.lstm = nn.LSTM(
            input_size=cnn_out, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            bidirectional=True, dropout=0.3 if lstm_layers > 1 else 0.
        )
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(lstm_hidden * 2, n_classes)
        )

    def forward(self, x):
        # x: (B, seq_len, 2, 3000)
        B, S, C, T = x.shape
        x = x.view(B * S, C, T)
        emb = self.cnn(x)          # (B*S, cnn_out)
        emb = emb.view(B, S, -1)   # (B, S, cnn_out)
        out, _ = self.lstm(emb)    # (B, S, 2*hidden)
        center = out[:, self.seq_len // 2, :]  # center epoch
        return self.head(center)


print('Building boundary-safe sequence datasets...')
tr_seq_ds = SeqEpochDataset(tr_records)
va_seq_ds = SeqEpochDataset(va_records)
te_seq_ds = SeqEpochDataset(te_records)

tr_seq_ld = DataLoader(tr_seq_ds, batch_size=64, shuffle=True,  num_workers=0)
va_seq_ld = DataLoader(va_seq_ds, batch_size=64, shuffle=False, num_workers=0)
te_seq_ld = DataLoader(te_seq_ds, batch_size=64, shuffle=False, num_workers=0)

print('Training CNN-LSTM...')
cnnlstm = CNNLSTM().to(DEVICE)
n_params = sum(p.numel() for p in cnnlstm.parameters() if p.requires_grad)
print(f'  Parameters: {n_params:,}')

cnnlstm, lstm_hist = train_model(
    cnnlstm, tr_seq_ld, va_seq_ld,
    epochs=30, lr=5e-4, class_weights_tensor=cw, patience=7
)

lstm_val_proba, _ = predict_proba(cnnlstm, va_seq_ld)
lstm_proba, _ = predict_proba(cnnlstm, te_seq_ld)
assert len(lstm_proba) == len(te_y), 'CNN-LSTM predictions do not align with test labels.'

lstm_pred = lstm_proba.argmax(axis=1)
res_lstm = evaluate(te_y, lstm_pred, model_name='CNN-LSTM')
results_log.append(res_lstm)



# ==============================================================================
# Section: 11. Compact Transformer Sequence Model + Entropy Uncertainty
# ==============================================================================

# %% [code] Cell 30
class SleepTransformerLite(nn.Module):
    """
    Lightweight Transformer for sleep staging.
    CNN epoch encoder → Transformer encoder → center epoch classification.

    Note: PyTorch's standard TransformerEncoderLayer does not expose attention
    weights here. Do not present this model as attention-interpretable unless a
    custom attention layer is added later.
    """
    def __init__(self, seq_len=SEQ_LEN, n_classes=N_CLASSES,
                 d_model=256, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len

        # Shared CNN encoder per epoch (same as CNN-LSTM)
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 64,  kernel_size=50, stride=6, padding=25),
            nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(8, 8),
            nn.Conv1d(64, 128, kernel_size=8, padding=4),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(4), nn.Flatten(),
        )
        cnn_out = 128 * 4  # 512

        self.proj = nn.Linear(cnn_out, d_model)

        # Positional encoding
        pos = torch.zeros(1, seq_len, d_model)
        pos[0, :, 0::2] = torch.sin(
            torch.arange(seq_len).unsqueeze(1).float() /
            (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        )
        pos[0, :, 1::2] = torch.cos(
            torch.arange(seq_len).unsqueeze(1).float() /
            (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        )
        self.register_buffer('pos_enc', pos)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes)
        )

    def forward(self, x):
        B, S, C, T = x.shape
        emb = self.cnn(x.view(B * S, C, T)).view(B, S, -1)
        emb = self.proj(emb) + self.pos_enc[:, :S, :]
        out = self.transformer(emb)          # (B, S, d_model)
        center = out[:, self.seq_len // 2, :]
        return self.head(center)


print('Training Compact Transformer...')
transformer = SleepTransformerLite().to(DEVICE)
n_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
print(f'  Parameters: {n_params:,}')

transformer, tf_hist = train_model(
    transformer, tr_seq_ld, va_seq_ld,
    epochs=30, lr=3e-4, class_weights_tensor=cw, patience=8
)

tf_val_proba, _ = predict_proba(transformer, va_seq_ld)
tf_proba, _ = predict_proba(transformer, te_seq_ld)
assert len(tf_proba) == len(te_y), 'Transformer predictions do not align with test labels.'
tf_pred     = tf_proba.argmax(axis=1)
res_tf = evaluate(te_y, tf_pred, model_name='Transformer')
results_log.append(res_tf)



# ==============================================================================
# Section: 12. Entropy-Based Uncertainty & Low-Confidence Deferral
# ==============================================================================

# %% [code] Cell 32
def entropy_from_proba(proba):
    """Shannon entropy of softmax distribution, normalized to [0, 1]."""
    proba = np.clip(np.asarray(proba), 1e-9, 1.0)
    H = -np.sum(proba * np.log(proba), axis=1)
    H_max = np.log(proba.shape[1])
    return H / H_max


def uncertainty_rejection_curve(y_true, proba, thresholds=None):
    """
    Diagnostic curve: Macro-F1 vs retained fraction after deferring high-entropy
    epochs. Threshold selection for a reported operating point should still be
    done on validation data, not this test curve.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    H = entropy_from_proba(proba)

    if thresholds is None:
        thresholds = np.unique(np.quantile(H, np.linspace(0.05, 1.00, 20)))

    rows = []
    for thr in thresholds:
        keep = H <= thr
        if keep.sum() < 10:
            continue
        rows.append({
            'threshold': float(thr),
            'retained': float(keep.mean()),
            'macro_f1': float(f1_score(
                y_true[keep],
                proba[keep].argmax(1),
                labels=list(range(N_CLASSES)),
                average='macro',
                zero_division=0
            ))
        })

    # Explicit keep-all baseline.
    rows.append({
        'threshold': float(H.max()),
        'retained': 1.0,
        'macro_f1': float(f1_score(
            y_true,
            proba.argmax(1),
            labels=list(range(N_CLASSES)),
            average='macro',
            zero_division=0
        ))
    })

    df = pd.DataFrame(rows).drop_duplicates(subset=['threshold', 'retained'])
    return df.sort_values('retained', ascending=False).reset_index(drop=True)


def choose_entropy_threshold_on_validation(val_proba, target_retained=0.80):
    """
    Select an entropy threshold using validation predictions only.
    Keeping H <= threshold should retain approximately target_retained of
    validation epochs.
    """
    H_val = entropy_from_proba(val_proba)
    return float(np.quantile(H_val, target_retained))


def evaluate_entropy_deferral(y_true, proba, threshold):
    H = entropy_from_proba(proba)
    keep = H <= threshold
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    if keep.sum() == 0:
        return {
            'threshold': threshold,
            'retained': 0.0,
            'deferred': 1.0,
            'macro_f1_retained': np.nan,
            'macro_f1_all': f1_score(
                y_true, proba.argmax(1), labels=list(range(N_CLASSES)),
                average='macro', zero_division=0
            ),
            'n_retained': 0,
            'n_deferred': int(len(y_true)),
        }

    return {
        'threshold': threshold,
        'retained': float(keep.mean()),
        'deferred': float(1.0 - keep.mean()),
        'macro_f1_retained': float(f1_score(
            y_true[keep],
            proba[keep].argmax(1),
            labels=list(range(N_CLASSES)),
            average='macro',
            zero_division=0
        )),
        'macro_f1_all': float(f1_score(
            y_true,
            proba.argmax(1),
            labels=list(range(N_CLASSES)),
            average='macro',
            zero_division=0
        )),
        'n_retained': int(keep.sum()),
        'n_deferred': int((~keep).sum()),
    }


# Validation-selected 80% retained operating point.
TARGET_RETAINED = 0.80

deferral_rows = []
for model_name, val_proba, test_proba in [
    ('Random Forest', rf_val_proba, rf_proba),
    ('CNN-LSTM', lstm_val_proba, lstm_proba),
    ('Transformer', tf_val_proba, tf_proba),
]:
    threshold = choose_entropy_threshold_on_validation(val_proba, TARGET_RETAINED)
    row = evaluate_entropy_deferral(te_y, test_proba, threshold)
    row['model'] = model_name
    row['target_retained_on_validation'] = TARGET_RETAINED
    deferral_rows.append(row)

deferral_df = pd.DataFrame(deferral_rows)[[
    'model', 'threshold', 'target_retained_on_validation',
    'retained', 'deferred', 'macro_f1_all', 'macro_f1_retained',
    'n_retained', 'n_deferred'
]]

print('\nValidation-selected entropy deferral results on test fold:')
display(deferral_df.style.format({
    'threshold': '{:.3f}',
    'target_retained_on_validation': '{:.2f}',
    'retained': '{:.3f}',
    'deferred': '{:.3f}',
    'macro_f1_all': '{:.4f}',
    'macro_f1_retained': '{:.4f}',
}))

deferral_df.to_csv(RESULTS_DIR / 'entropy_deferral_validation_selected.csv', index=False)

# Diagnostic test curves, not used for threshold selection.
tf_entropy = entropy_from_proba(tf_proba)
rf_entropy = entropy_from_proba(rf_proba)
lstm_entropy = entropy_from_proba(lstm_proba)

tf_rej = uncertainty_rejection_curve(te_y, tf_proba)
lstm_rej = uncertainty_rejection_curve(te_y, lstm_proba)
rf_rej = uncertainty_rejection_curve(te_y, rf_proba)

fig, ax = plt.subplots(figsize=(8, 5))
for df, label, color in [
    (tf_rej, 'Transformer', 'steelblue'),
    (lstm_rej, 'CNN-LSTM', 'darkorange'),
    (rf_rej, 'Random Forest', 'green'),
]:
    ax.plot(df['retained'], df['macro_f1'], marker='o', label=label, color=color)

ax.set_xlabel('Fraction of Test Epochs Retained (1 = keep all)')
ax.set_ylabel('Macro-F1 on Retained Epochs')
ax.set_title('Diagnostic Uncertainty-Rejection Curve\n(thresholds selected on test only for visualization)')
ax.legend()
ax.invert_xaxis()
ax.axvline(TARGET_RETAINED, linestyle='--', color='gray', alpha=0.5)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'uncertainty_rejection_curve_diagnostic.png', dpi=120)
plt.show()


# Probability-quality diagnostics for probability-dependent analyses.
# These do not change the selected model; they warn whether entropy/HMM emissions
# should be interpreted cautiously.
calibration_rows = []
calibration_bin_tables = {}
for model_name, proba in [
    ('Random Forest', rf_proba),
    ('CNN-LSTM', lstm_proba),
    ('Transformer', tf_proba),
]:
    ece, bins_df = expected_calibration_error(te_y, proba, n_bins=15)
    calibration_bin_tables[model_name] = bins_df
    calibration_rows.append({
        'model': model_name,
        'brier_score_multiclass': multiclass_brier_score(te_y, proba),
        'ece_15_bins': ece,
        'mean_confidence': float(np.asarray(proba).max(axis=1).mean()),
    })

calibration_df = pd.DataFrame(calibration_rows).sort_values('brier_score_multiclass')
print('\nProbability diagnostics on test fold:')
display(calibration_df.style.format({
    'brier_score_multiclass': '{:.4f}',
    'ece_15_bins': '{:.4f}',
    'mean_confidence': '{:.4f}',
}))
calibration_df.to_csv(RESULTS_DIR / 'probability_diagnostics_fold0.csv', index=False)



# ==============================================================================
# Section: 13. HMM Post-Processing (Viterbi Smoothing)
# ==============================================================================

# %% [code] Cell 34
def _label_sequences_from_records(records):
    return [np.asarray(r['labels'], dtype=np.int64) for r in records]


def fit_hmm_from_records(records, n_classes=N_CLASSES):
    """
    Estimate transition matrix and start probabilities from record-level labels.

    Transitions are counted within each night only. This avoids the earlier error
    where the final epoch of one night was treated as preceding the first epoch
    of the next night.
    """
    label_sequences = _label_sequences_from_records(records)

    A = np.zeros((n_classes, n_classes), dtype=float)
    pi = np.zeros(n_classes, dtype=float)

    for y_seq in label_sequences:
        if len(y_seq) == 0:
            continue
        pi[y_seq[0]] += 1
        for a, b in zip(y_seq[:-1], y_seq[1:]):
            if 0 <= a < n_classes and 0 <= b < n_classes:
                A[a, b] += 1

    A = (A + 0.01) / (A + 0.01).sum(axis=1, keepdims=True)
    pi = (pi + 1.0) / (pi + 1.0).sum()
    return A, pi


def fit_hmm_from_labels(y_train, n_classes=N_CLASSES):
    """
    Backward-compatible fallback for a single continuous sequence.
    Prefer fit_hmm_from_records(records) when record boundaries are available.
    """
    pseudo_records = [{'labels': np.asarray(y_train, dtype=np.int64)}]
    return fit_hmm_from_records(pseudo_records, n_classes=n_classes)


def viterbi_smooth(proba_seq, A, pi):
    """
    Viterbi decoding using per-epoch softmax probabilities as emissions.
    proba_seq: (T, n_classes)
    """
    proba_seq = np.asarray(proba_seq)
    T, K = proba_seq.shape

    log_A = np.log(A + 1e-12)
    log_B = np.log(proba_seq + 1e-12)
    log_pi = np.log(pi + 1e-12)

    delta = np.full((T, K), -np.inf)
    psi = np.zeros((T, K), dtype=int)

    delta[0] = log_pi + log_B[0]
    for t in range(1, T):
        for j in range(K):
            trans = delta[t - 1] + log_A[:, j]
            psi[t, j] = trans.argmax()
            delta[t, j] = trans.max() + log_B[t, j]

    path = np.zeros(T, dtype=int)
    path[-1] = delta[-1].argmax()
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    return path


def viterbi_smooth_by_record(proba_concat, records, A, pi):
    """Apply Viterbi smoothing independently to each night/record."""
    proba_concat = np.asarray(proba_concat)
    smoothed = []
    pos = 0

    for rec in records:
        n = len(rec['labels'])
        if n == 0:
            continue
        rec_proba = proba_concat[pos:pos + n]
        if len(rec_proba) != n:
            raise ValueError('Probability array length does not match record lengths.')
        smoothed.append(viterbi_smooth(rec_proba, A, pi))
        pos += n

    if pos != len(proba_concat):
        raise ValueError('Unused probabilities remain after record-level Viterbi smoothing.')

    return np.concatenate(smoothed)


# Fit HMM on training labels with record boundaries preserved.
A_hmm, pi_hmm = fit_hmm_from_records(tr_records)

print('Transition matrix (HMM, trained within records):')
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(pd.DataFrame(A_hmm, index=STAGE_NAMES, columns=STAGE_NAMES),
             annot=True, fmt='.2f', cmap='Blues', ax=ax)
ax.set_title('Estimated Transition Matrix')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'hmm_transition_matrix.png', dpi=120)
plt.show()

print('\nApplying per-record Viterbi smoothing...')
lstm_smooth = viterbi_smooth_by_record(lstm_proba, te_records, A_hmm, pi_hmm)
tf_smooth = viterbi_smooth_by_record(tf_proba, te_records, A_hmm, pi_hmm)

res_lstm_hmm = evaluate(te_y, lstm_smooth, model_name='CNN-LSTM + HMM')
res_tf_hmm = evaluate(te_y, tf_smooth, model_name='Transformer + HMM')
results_log += [res_lstm_hmm, res_tf_hmm]

# ── Illegal transition count, respecting record boundaries ───────────────
ILLEGAL_PAIRS = {(0, 3), (0, 4), (3, 4), (4, 3), (1, 3), (3, 1)}

def count_illegal(y_seq):
    count = 0
    y_seq = np.asarray(y_seq)
    for a, b in zip(y_seq[:-1], y_seq[1:]):
        if (int(a), int(b)) in ILLEGAL_PAIRS:
            count += 1
    return count


def count_illegal_by_records(y_concat, records):
    return sum(count_illegal(y_concat[s]) for s in record_slices(records))


print('\nIllegal transitions within records only (before vs after HMM):')
print(f'  CNN-LSTM   : {count_illegal_by_records(lstm_pred, te_records):4d} → {count_illegal_by_records(lstm_smooth, te_records):4d}')
print(f'  Transformer: {count_illegal_by_records(tf_pred, te_records):4d} → {count_illegal_by_records(tf_smooth, te_records):4d}')


# Explicitly quantify the trade-off instead of assuming HMM is always beneficial.
hmm_tradeoff_rows = []
for base_name, base_pred, smooth_name, smooth_pred in [
    ('CNN-LSTM', lstm_pred, 'CNN-LSTM + HMM', lstm_smooth),
    ('Transformer', tf_pred, 'Transformer + HMM', tf_smooth),
]:
    base_mf1 = f1_score(te_y, base_pred, labels=list(range(N_CLASSES)), average='macro', zero_division=0)
    smooth_mf1 = f1_score(te_y, smooth_pred, labels=list(range(N_CLASSES)), average='macro', zero_division=0)
    base_illegal = count_illegal_by_records(base_pred, te_records)
    smooth_illegal = count_illegal_by_records(smooth_pred, te_records)
    hmm_tradeoff_rows.append({
        'base_model': base_name,
        'smoothed_model': smooth_name,
        'macro_f1_before': base_mf1,
        'macro_f1_after': smooth_mf1,
        'macro_f1_delta_after_minus_before': smooth_mf1 - base_mf1,
        'illegal_transitions_before': base_illegal,
        'illegal_transitions_after': smooth_illegal,
        'illegal_transition_delta_after_minus_before': smooth_illegal - base_illegal,
    })

hmm_tradeoff_df = pd.DataFrame(hmm_tradeoff_rows)
print('\nHMM smoothing trade-off summary:')
display(hmm_tradeoff_df.style.format({
    'macro_f1_before': '{:.4f}',
    'macro_f1_after': '{:.4f}',
    'macro_f1_delta_after_minus_before': '{:+.4f}',
}))
hmm_tradeoff_df.to_csv(RESULTS_DIR / 'hmm_smoothing_tradeoff_fold0.csv', index=False)



# ==============================================================================
# Section: 14. Transition-Aware Error Analysis
# ==============================================================================

# %% [code] Cell 36
def transition_mask_for_sequence(y_seq, window=3):
    """Boolean mask for epochs within ±window of a true stage boundary in one record."""
    y_seq = np.asarray(y_seq)
    boundaries = np.where(np.diff(y_seq) != 0)[0] + 1
    mask = np.zeros(len(y_seq), dtype=bool)
    for b in boundaries:
        lo = max(0, b - window)
        hi = min(len(y_seq), b + window + 1)
        mask[lo:hi] = True
    return mask


def transition_mask_by_records(records, window=3):
    """Concatenated transition mask, computed independently within each record."""
    masks = [transition_mask_for_sequence(r['labels'], window=window) for r in records]
    return np.concatenate(masks) if masks else np.array([], dtype=bool)


def transition_mask(y_true, window=3):
    """
    Backward-compatible single-sequence mask.
    Prefer transition_mask_by_records(records) for Sleep-EDF evaluation.
    """
    return transition_mask_for_sequence(y_true, window=window)


def _safe_macro_f1(y_true, y_pred):
    if len(y_true) == 0:
        return np.nan
    return f1_score(
        y_true, y_pred,
        labels=list(range(N_CLASSES)),
        average='macro',
        zero_division=0
    )


def transition_analysis(y_true, y_pred, model_name='Model', window=3, records=None):
    if records is None:
        t_mask = transition_mask(y_true, window)
    else:
        t_mask = transition_mask_by_records(records, window)

    s_mask = ~t_mask
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    f1_stable = _safe_macro_f1(y_true[s_mask], y_pred[s_mask])
    f1_trans = _safe_macro_f1(y_true[t_mask], y_pred[t_mask])

    print(f'\n── {model_name} Transition Analysis (window=±{window} epochs; record-aware={records is not None}) ──')
    print(f'  Stable epochs    : {s_mask.sum():5d}  MF1 = {f1_stable:.4f}')
    print(f'  Transition epochs: {t_mask.sum():5d}  MF1 = {f1_trans:.4f}')
    print(f'  MF1 drop (transition vs stable): {f1_stable - f1_trans:.4f}')
    return f1_stable, f1_trans


for name, pred in [
    ('Random Forest', rf_pred),
    ('1D CNN', cnn_pred),
    ('CNN-LSTM', lstm_pred),
    ('CNN-LSTM + HMM', lstm_smooth),
    ('Transformer', tf_pred),
    ('Transformer + HMM', tf_smooth),
]:
    transition_analysis(te_y, pred, model_name=name, records=te_records)

# %% [code] Cell 37
# ── Hypnogram visualization ────────────────────────────────────────────────
def plot_hypnogram_record(y_true, y_pred_dict, records, record_index=None, title='Hypnogram comparison'):
    """
    Plot one test-night hypnogram. This avoids visualising concatenated nights as
    if they were one continuous sleep recording.
    """
    if record_index is None:
        lengths = record_lengths(records)
        record_index = int(np.argmax(lengths))

    rec = records[record_index]
    rec_slice = record_slices(records)[record_index]

    y_true_rec = np.asarray(y_true)[rec_slice]
    y_pred_rec_dict = {name: np.asarray(y)[rec_slice] for name, y in y_pred_dict.items()}

    n_epochs = len(y_true_rec)
    t = np.arange(n_epochs) * EPOCH_SEC / 3600

    n_models = len(y_pred_rec_dict) + 1
    fig, axes = plt.subplots(n_models, 1, figsize=(14, 2 * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]

    all_preds = {'Ground Truth': y_true_rec}
    all_preds.update(y_pred_rec_dict)

    for ax, (name, y) in zip(axes, all_preds.items()):
        ax.step(t, y, where='post', color='steelblue', linewidth=0.8)
        ax.set_yticks(range(N_CLASSES))
        ax.set_yticklabels(STAGE_NAMES)
        ax.set_ylabel(name, fontsize=8, rotation=0, labelpad=60, va='center')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

    axes[-1].set_xlabel('Time (hours)')
    rec_label = f"{rec.get('record_id', record_index)} | subject={rec.get('subject_label', rec.get('subject_id'))}"
    fig.suptitle(f'{title}\n{rec_label}', fontsize=11)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'hypnogram_comparison_record_aware.png', dpi=120)
    plt.show()


plot_hypnogram_record(
    te_y,
    {
        'Random Forest': rf_pred,
        'CNN-LSTM': lstm_pred,
        'CNN-LSTM+HMM': lstm_smooth,
        'Transformer': tf_pred,
    },
    records=te_records
)



# ==============================================================================
# Section: 15. Fold-0 Results Summary
# ==============================================================================

# %% [code] Cell 39
results_df = pd.DataFrame(results_log)
results_df = results_df.sort_values('macro_f1', ascending=False).reset_index(drop=True)

print('\n══════════════ FOLD-0 EPOCH-POOLED RESULTS ══════════════')
print('Interpretation note: this table is a fold-0 demonstration result. Use the full CV cell for final paper-level claims.')
display(results_df.style.format({'macro_f1': '{:.4f}', 'kappa': '{:.4f}', 'accuracy': '{:.4f}'})
        .highlight_max(subset=['macro_f1','kappa'], color='#c6efce'))

# Record/night-level metrics: avoids relying only on epoch-pooled metrics.
record_metric_dfs = []
for model_name, pred in [
    ('Random Forest', rf_pred),
    ('1D CNN', cnn_pred),
    ('CNN-LSTM', lstm_pred),
    ('CNN-LSTM + HMM', lstm_smooth),
    ('Transformer', tf_pred),
    ('Transformer + HMM', tf_smooth),
]:
    record_metric_dfs.append(per_record_metrics(te_y, pred, te_records, model_name))

record_metrics_fold0 = pd.concat(record_metric_dfs, ignore_index=True)
record_summary_fold0 = summarize_record_metrics(record_metrics_fold0)

print('\n══════════════ FOLD-0 RECORD/NIGHT-LEVEL SUMMARY ══════════════')
display(record_summary_fold0.style.format({
    'record_macro_f1_mean': '{:.4f}',
    'record_macro_f1_sd': '{:.4f}',
    'record_kappa_mean': '{:.4f}',
    'record_kappa_sd': '{:.4f}',
}))

# Bar chart: fold-0 epoch-pooled metric comparison.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

models = results_df['model'].values
x = np.arange(len(models))

ax1.bar(x, results_df['macro_f1'], color='steelblue', alpha=0.85)
ax1.set_xticks(x); ax1.set_xticklabels(models, rotation=30, ha='right')
ax1.set_ylabel('Macro-F1'); ax1.set_title('Fold-0 Epoch-Pooled Macro-F1')
ax1.set_ylim(max(0, results_df['macro_f1'].min() - 0.05), 1.0)

ax2.bar(x, results_df['kappa'], color='darkorange', alpha=0.85)
ax2.set_xticks(x); ax2.set_xticklabels(models, rotation=30, ha='right')
ax2.set_ylabel("Cohen's κ"); ax2.set_title("Fold-0 Epoch-Pooled Cohen's κ")
ax2.set_ylim(max(0, results_df['kappa'].min() - 0.05), 1.0)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'model_comparison_fold0_epoch_pooled.png', dpi=120)
plt.show()

results_df.to_csv(RESULTS_DIR / 'results_fold0_epoch_pooled.csv', index=False)
record_metrics_fold0.to_csv(RESULTS_DIR / 'record_metrics_fold0.csv', index=False)
record_summary_fold0.to_csv(RESULTS_DIR / 'record_summary_fold0.csv', index=False)
print(f'\nFold-0 result tables saved to {RESULTS_DIR}')



# ==============================================================================
# Section: 16. Full Cross-Validation Loop with BSPC Evidence Exports
# ==============================================================================

# %% [code] Cell 41
# ── Full CV with journal-strengthening exports ─────────────────────────────
# This cell is the primary BSPC evidence-generation loop.
#
# It exports:
#   1) cv_results_record_aware.csv
#   2) cv_per_stage_metrics.csv
#   3) cv_split_stage_distribution.csv
#   4) cv_computational_cost.csv
#   5) cv_hmm_tradeoff.csv
#   6) summaries for manuscript tables/figures
#
# HMM is deliberately reported as a trade-off: it may reduce large-jump
# transition heuristics without improving Macro-F1 for stronger sequence models.

RUN_FULL_CV = bool(PAPER_RUN)
CV_RUN_DEEP_MODELS = bool(PAPER_RUN_DEEP_CV)
CV_DEEP_EPOCHS = 15
CV_DEEP_PATIENCE = 5

def save_cv_artifacts(prefix='partial'):
    """Persist partial evidence after each fold to protect long Colab runs."""
    if cv_results:
        pd.DataFrame(cv_results).to_csv(RESULTS_DIR / f'cv_results_record_aware_{prefix}.csv', index=False)
    if cv_per_class_rows:
        pd.DataFrame(cv_per_class_rows).to_csv(RESULTS_DIR / f'cv_per_stage_metrics_{prefix}.csv', index=False)
    if cv_split_distribution_rows:
        pd.DataFrame(cv_split_distribution_rows).to_csv(RESULTS_DIR / f'cv_split_stage_distribution_{prefix}.csv', index=False)
    if cv_cost_rows:
        pd.DataFrame(cv_cost_rows).to_csv(RESULTS_DIR / f'cv_computational_cost_{prefix}.csv', index=False)
    if cv_hmm_tradeoff_rows:
        pd.DataFrame(cv_hmm_tradeoff_rows).to_csv(RESULTS_DIR / f'cv_hmm_tradeoff_{prefix}.csv', index=False)

if RUN_FULL_CV:
    cv_results = []
    cv_per_class_rows = []
    cv_split_distribution_rows = []
    cv_cost_rows = []
    cv_hmm_tradeoff_rows = []
    completed_folds = []

    for fold in range(N_FOLDS):
        print(f'\n══ Fold {fold+1}/{N_FOLDS} ══')
        set_global_seed(SEED + fold)

        tr_rec_f, va_rec_f, te_rec_f = load_fold_records(fold, manifest)
        (tr_eeg_f, tr_eog_f, tr_y_f) = concat_records(tr_rec_f)
        (va_eeg_f, va_eog_f, va_y_f) = concat_records(va_rec_f)
        (te_eeg_f, te_eog_f, te_y_f) = concat_records(te_rec_f)

        append_split_distribution_rows(cv_split_distribution_rows, fold, 'train', tr_y_f, tr_rec_f)
        append_split_distribution_rows(cv_split_distribution_rows, fold, 'validation', va_y_f, va_rec_f)
        append_split_distribution_rows(cv_split_distribution_rows, fold, 'test', te_y_f, te_rec_f)

        # HMM transition matrix is estimated from training records only.
        A_f, pi_f = fit_hmm_from_records(tr_rec_f)

        # ── RF baseline ──────────────────────────────────────────────────
        start = time.perf_counter()
        tr_f = extract_features_batch(tr_eeg_f, tr_eog_f)
        te_f = extract_features_batch(te_eeg_f, te_eog_f)
        scaler_f = StandardScaler().fit(tr_f)
        rf_f = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight=make_class_weight_dict(tr_y_f),
            n_jobs=-1,
            random_state=SEED + fold
        )
        rf_f.fit(scaler_f.transform(tr_f), tr_y_f)
        rf_train_seconds = time.perf_counter() - start

        start = time.perf_counter()
        rf_proba_f = align_class_proba(rf_f.predict_proba(scaler_f.transform(te_f)), rf_f.classes_)
        rf_infer_seconds = time.perf_counter() - start
        rf_pred_f = rf_proba_f.argmax(1)

        append_metric_row(cv_results, fold, 'Random Forest', te_y_f, rf_pred_f)
        append_per_class_rows(cv_per_class_rows, fold, 'Random Forest', te_y_f, rf_pred_f)
        append_cost_row(
            cv_cost_rows, fold, 'Random Forest',
            train_seconds=rf_train_seconds,
            inference_seconds=rf_infer_seconds,
            n_train_epochs=len(tr_y_f),
            n_test_epochs=len(te_y_f),
            n_parameters=np.nan,
            model_size_mb_value=sklearn_object_size_mb({'model': rf_f, 'scaler': scaler_f}),
            notes='Training time includes feature extraction, scaling, and RF fit.'
        )

        start = time.perf_counter()
        rf_smooth_f = viterbi_smooth_by_record(rf_proba_f, te_rec_f, A_f, pi_f)
        rf_hmm_seconds = time.perf_counter() - start
        append_metric_row(cv_results, fold, 'Random Forest + HMM', te_y_f, rf_smooth_f)
        append_per_class_rows(cv_per_class_rows, fold, 'Random Forest + HMM', te_y_f, rf_smooth_f)
        append_hmm_tradeoff_row(cv_hmm_tradeoff_rows, fold, 'Random Forest', te_y_f, rf_pred_f, rf_smooth_f, te_rec_f)
        append_cost_row(
            cv_cost_rows, fold, 'Random Forest + HMM',
            train_seconds=0.0,
            inference_seconds=rf_hmm_seconds,
            n_train_epochs=len(tr_y_f),
            n_test_epochs=len(te_y_f),
            notes='Post-processing only; HMM transition matrix estimated from training labels.'
        )

        if CV_RUN_DEEP_MODELS:
            cw_f = torch.FloatTensor(make_class_weight_vector(tr_y_f)).to(DEVICE)

            # Shared single-epoch loaders
            tr_ds_f = EpochDataset(tr_eeg_f, tr_eog_f, tr_y_f)
            va_ds_f = EpochDataset(va_eeg_f, va_eog_f, va_y_f)
            te_ds_f = EpochDataset(te_eeg_f, te_eog_f, te_y_f)

            tr_ld_f = DataLoader(
                tr_ds_f, batch_size=256, shuffle=True, num_workers=0,
                generator=torch_generator_for(SEED + 1000 + fold)
            )
            va_ld_f = DataLoader(va_ds_f, batch_size=256, shuffle=False, num_workers=0)
            te_ld_f = DataLoader(te_ds_f, batch_size=256, shuffle=False, num_workers=0)

            # 1D CNN
            set_global_seed(SEED + 10_000 + fold)
            cnn_f = CNN1D().to(DEVICE)
            start = time.perf_counter()
            cnn_f, _ = train_model(
                cnn_f, tr_ld_f, va_ld_f,
                epochs=CV_DEEP_EPOCHS,
                lr=1e-3,
                class_weights_tensor=cw_f,
                patience=CV_DEEP_PATIENCE
            )
            cnn_train_seconds = time.perf_counter() - start
            cnn_proba_f, _, cnn_infer_seconds = timed_predict_proba_torch(cnn_f, te_ld_f)
            cnn_pred_f = cnn_proba_f.argmax(1)

            append_metric_row(cv_results, fold, '1D CNN', te_y_f, cnn_pred_f)
            append_per_class_rows(cv_per_class_rows, fold, '1D CNN', te_y_f, cnn_pred_f)
            append_cost_row(
                cv_cost_rows, fold, '1D CNN',
                train_seconds=cnn_train_seconds,
                inference_seconds=cnn_infer_seconds,
                n_train_epochs=len(tr_y_f),
                n_test_epochs=len(te_y_f),
                n_parameters=model_num_parameters(cnn_f),
                model_size_mb_value=model_size_mb(cnn_f),
                notes=f'{CV_DEEP_EPOCHS} epoch maximum with early stopping.'
            )

            del cnn_f
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Boundary-safe sequence loaders
            tr_seq_ds_f = SeqEpochDataset(tr_rec_f)
            va_seq_ds_f = SeqEpochDataset(va_rec_f)
            te_seq_ds_f = SeqEpochDataset(te_rec_f)

            tr_seq_ld_f = DataLoader(
                tr_seq_ds_f, batch_size=64, shuffle=True, num_workers=0,
                generator=torch_generator_for(SEED + 20_000 + fold)
            )
            va_seq_ld_f = DataLoader(va_seq_ds_f, batch_size=64, shuffle=False, num_workers=0)
            te_seq_ld_f = DataLoader(te_seq_ds_f, batch_size=64, shuffle=False, num_workers=0)

            # CNN-LSTM
            set_global_seed(SEED + 30_000 + fold)
            lstm_f = CNNLSTM().to(DEVICE)
            start = time.perf_counter()
            lstm_f, _ = train_model(
                lstm_f, tr_seq_ld_f, va_seq_ld_f,
                epochs=CV_DEEP_EPOCHS,
                lr=5e-4,
                class_weights_tensor=cw_f,
                patience=CV_DEEP_PATIENCE
            )
            lstm_train_seconds = time.perf_counter() - start
            lstm_proba_f, _, lstm_infer_seconds = timed_predict_proba_torch(lstm_f, te_seq_ld_f)
            lstm_pred_f = lstm_proba_f.argmax(1)

            append_metric_row(cv_results, fold, 'CNN-LSTM', te_y_f, lstm_pred_f)
            append_per_class_rows(cv_per_class_rows, fold, 'CNN-LSTM', te_y_f, lstm_pred_f)
            append_cost_row(
                cv_cost_rows, fold, 'CNN-LSTM',
                train_seconds=lstm_train_seconds,
                inference_seconds=lstm_infer_seconds,
                n_train_epochs=len(tr_y_f),
                n_test_epochs=len(te_y_f),
                n_parameters=model_num_parameters(lstm_f),
                model_size_mb_value=model_size_mb(lstm_f),
                notes=f'Record-aware {SEQ_LEN}-epoch context; {CV_DEEP_EPOCHS} epoch maximum.'
            )

            start = time.perf_counter()
            lstm_smooth_f = viterbi_smooth_by_record(lstm_proba_f, te_rec_f, A_f, pi_f)
            lstm_hmm_seconds = time.perf_counter() - start
            append_metric_row(cv_results, fold, 'CNN-LSTM + HMM', te_y_f, lstm_smooth_f)
            append_per_class_rows(cv_per_class_rows, fold, 'CNN-LSTM + HMM', te_y_f, lstm_smooth_f)
            append_hmm_tradeoff_row(cv_hmm_tradeoff_rows, fold, 'CNN-LSTM', te_y_f, lstm_pred_f, lstm_smooth_f, te_rec_f)
            append_cost_row(
                cv_cost_rows, fold, 'CNN-LSTM + HMM',
                train_seconds=0.0,
                inference_seconds=lstm_hmm_seconds,
                n_train_epochs=len(tr_y_f),
                n_test_epochs=len(te_y_f),
                notes='Post-processing only; interpret as transition-plausibility trade-off.'
            )

            del lstm_f
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Transformer
            set_global_seed(SEED + 40_000 + fold)
            tf_f = SleepTransformerLite().to(DEVICE)
            start = time.perf_counter()
            tf_f, _ = train_model(
                tf_f, tr_seq_ld_f, va_seq_ld_f,
                epochs=CV_DEEP_EPOCHS,
                lr=3e-4,
                class_weights_tensor=cw_f,
                patience=CV_DEEP_PATIENCE
            )
            tf_train_seconds = time.perf_counter() - start
            tf_proba_f, _, tf_infer_seconds = timed_predict_proba_torch(tf_f, te_seq_ld_f)
            tf_pred_f = tf_proba_f.argmax(1)

            append_metric_row(cv_results, fold, 'Transformer', te_y_f, tf_pred_f)
            append_per_class_rows(cv_per_class_rows, fold, 'Transformer', te_y_f, tf_pred_f)
            append_cost_row(
                cv_cost_rows, fold, 'Transformer',
                train_seconds=tf_train_seconds,
                inference_seconds=tf_infer_seconds,
                n_train_epochs=len(tr_y_f),
                n_test_epochs=len(te_y_f),
                n_parameters=model_num_parameters(tf_f),
                model_size_mb_value=model_size_mb(tf_f),
                notes='Compact Transformer encoder; attention weights are not extracted.'
            )

            start = time.perf_counter()
            tf_smooth_f = viterbi_smooth_by_record(tf_proba_f, te_rec_f, A_f, pi_f)
            tf_hmm_seconds = time.perf_counter() - start
            append_metric_row(cv_results, fold, 'Transformer + HMM', te_y_f, tf_smooth_f)
            append_per_class_rows(cv_per_class_rows, fold, 'Transformer + HMM', te_y_f, tf_smooth_f)
            append_hmm_tradeoff_row(cv_hmm_tradeoff_rows, fold, 'Transformer', te_y_f, tf_pred_f, tf_smooth_f, te_rec_f)
            append_cost_row(
                cv_cost_rows, fold, 'Transformer + HMM',
                train_seconds=0.0,
                inference_seconds=tf_hmm_seconds,
                n_train_epochs=len(tr_y_f),
                n_test_epochs=len(te_y_f),
                notes='Post-processing only; interpret as transition-plausibility trade-off.'
            )

            del tf_f
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        completed_folds.append(fold)
        save_cv_artifacts(prefix='partial')

        # Free fold-local large arrays before the next fold.
        del tr_eeg_f, tr_eog_f, tr_y_f, va_eeg_f, va_eog_f, va_y_f, te_eeg_f, te_eog_f, te_y_f
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Final summaries ──────────────────────────────────────────────────
    cv_df = pd.DataFrame(cv_results)
    cv_per_class_df = pd.DataFrame(cv_per_class_rows)
    cv_split_distribution_df = pd.DataFrame(cv_split_distribution_rows)
    cv_cost_df = pd.DataFrame(cv_cost_rows)
    cv_hmm_tradeoff_df = pd.DataFrame(cv_hmm_tradeoff_rows)

    cv_summary = cv_df.groupby('model').agg(
        macro_f1_mean=('macro_f1', 'mean'),
        macro_f1_std=('macro_f1', 'std'),
        kappa_mean=('kappa', 'mean'),
        kappa_std=('kappa', 'std'),
        accuracy_mean=('accuracy', 'mean'),
        accuracy_std=('accuracy', 'std'),
        n_folds=('fold', 'nunique')
    ).sort_values('macro_f1_mean', ascending=False).round(4)

    per_class_summary = (cv_per_class_df
        .groupby(['model', 'stage'])
        .agg(
            precision_mean=('precision', 'mean'),
            precision_std=('precision', 'std'),
            recall_mean=('recall', 'mean'),
            recall_std=('recall', 'std'),
            f1_mean=('f1', 'mean'),
            f1_std=('f1', 'std'),
            support_total=('support', 'sum'),
            n_folds=('fold', 'nunique')
        )
        .reset_index()
        .sort_values(['model', 'stage'])
    )

    cost_summary = (cv_cost_df
        .groupby('model')
        .agg(
            train_seconds_mean=('train_seconds', 'mean'),
            train_seconds_std=('train_seconds', 'std'),
            inference_seconds_mean=('inference_seconds', 'mean'),
            inference_seconds_std=('inference_seconds', 'std'),
            n_parameters_mean=('n_parameters', 'mean'),
            model_size_mb_mean=('model_size_mb', 'mean'),
            n_folds=('fold', 'nunique')
        )
        .reset_index()
    )

    hmm_tradeoff_summary = (cv_hmm_tradeoff_df
        .groupby(['base_model', 'smoothed_model'])
        .agg(
            macro_f1_delta_mean=('macro_f1_delta_after_minus_before', 'mean'),
            macro_f1_delta_std=('macro_f1_delta_after_minus_before', 'std'),
            kappa_delta_mean=('kappa_delta_after_minus_before', 'mean'),
            large_jump_delta_mean=('large_jump_delta_after_minus_before', 'mean'),
            large_jump_before_mean=('large_jump_transitions_before', 'mean'),
            large_jump_after_mean=('large_jump_transitions_after', 'mean'),
            n_folds=('fold', 'nunique')
        )
        .reset_index()
    )

    print('\n── Cross-Validation Summary ──')
    display(cv_summary)

    print('\n── Per-stage F1 summary ──')
    display(per_class_summary.pivot(index='model', columns='stage', values='f1_mean').round(4))

    print('\n── Computational-cost summary ──')
    display(cost_summary.round(4))

    print('\n── HMM trade-off summary ──')
    display(hmm_tradeoff_summary.round(4))

    cv_df.to_csv(RESULTS_DIR / 'cv_results_record_aware.csv', index=False)
    cv_summary.to_csv(RESULTS_DIR / 'cv_summary_record_aware.csv')
    cv_per_class_df.to_csv(RESULTS_DIR / 'cv_per_stage_metrics.csv', index=False)
    per_class_summary.to_csv(RESULTS_DIR / 'cv_per_stage_summary.csv', index=False)
    cv_split_distribution_df.to_csv(RESULTS_DIR / 'cv_split_stage_distribution.csv', index=False)
    cv_cost_df.to_csv(RESULTS_DIR / 'cv_computational_cost.csv', index=False)
    cost_summary.to_csv(RESULTS_DIR / 'cv_computational_cost_summary.csv', index=False)
    cv_hmm_tradeoff_df.to_csv(RESULTS_DIR / 'cv_hmm_tradeoff.csv', index=False)
    hmm_tradeoff_summary.to_csv(RESULTS_DIR / 'cv_hmm_tradeoff_summary.csv', index=False)

else:
    print('Full CV skipped because PAPER_RUN = False.')
    print('For final BSPC paper-quality fold-wise estimates, set:')
    print('  PAPER_RUN = True')
    print('  PAPER_RUN_DEEP_CV = True')
    print('  FORCE_REBUILD_CACHE = True for one final preprocessing rebuild.')

# %% [code] Cell 42
# ── Repeated-seed confirmation for top sequence models ─────────────────────
# This is a journal-strengthening sensitivity analysis for BSPC.
# It evaluates whether the near-tie between CNN-LSTM and Transformer is stable
# across random initialization and DataLoader shuffling.
#
# Outputs:
#   repeated_seed_sequence_results.csv
#   repeated_seed_sequence_per_stage_metrics.csv
#   repeated_seed_sequence_cost.csv
#   repeated_seed_sequence_summary.csv

if RUN_REPEATED_SEED_CONFIRMATION:
    if REPEATED_SEED_FOLDS == 'all':
        seed_folds = list(range(N_FOLDS))
    else:
        seed_folds = list(REPEATED_SEED_FOLDS)

    repeated_rows = []
    repeated_per_class_rows = []
    repeated_cost_rows = []

    for seed in REPEATED_SEEDS:
        print(f'\n════ Repeated-seed run: seed={seed} ════')
        for fold in seed_folds:
            print(f'\n── Seed {seed} | Fold {fold+1}/{N_FOLDS} ──')
            set_global_seed(seed + fold)

            tr_rec_f, va_rec_f, te_rec_f = load_fold_records(fold, manifest)
            tr_eeg_f, tr_eog_f, tr_y_f = concat_records(tr_rec_f)
            va_eeg_f, va_eog_f, va_y_f = concat_records(va_rec_f)
            te_eeg_f, te_eog_f, te_y_f = concat_records(te_rec_f)

            cw_f = torch.FloatTensor(make_class_weight_vector(tr_y_f)).to(DEVICE)

            tr_seq_ds_f = SeqEpochDataset(tr_rec_f)
            va_seq_ds_f = SeqEpochDataset(va_rec_f)
            te_seq_ds_f = SeqEpochDataset(te_rec_f)

            tr_seq_ld_f = DataLoader(
                tr_seq_ds_f, batch_size=64, shuffle=True, num_workers=0,
                generator=torch_generator_for(seed + 50_000 + fold)
            )
            va_seq_ld_f = DataLoader(va_seq_ds_f, batch_size=64, shuffle=False, num_workers=0)
            te_seq_ld_f = DataLoader(te_seq_ds_f, batch_size=64, shuffle=False, num_workers=0)

            if 'CNN-LSTM' in REPEATED_SEED_MODELS:
                set_global_seed(seed + 60_000 + fold)
                lstm_rs = CNNLSTM().to(DEVICE)
                start = time.perf_counter()
                lstm_rs, _ = train_model(
                    lstm_rs, tr_seq_ld_f, va_seq_ld_f,
                    epochs=REPEATED_SEED_EPOCHS,
                    lr=5e-4,
                    class_weights_tensor=cw_f,
                    patience=REPEATED_SEED_PATIENCE
                )
                train_seconds = time.perf_counter() - start
                proba, _, infer_seconds = timed_predict_proba_torch(lstm_rs, te_seq_ld_f)
                pred = proba.argmax(1)

                append_metric_row(repeated_rows, fold, 'CNN-LSTM', te_y_f, pred)
                repeated_rows[-1]['seed'] = int(seed)
                append_per_class_rows(repeated_per_class_rows, fold, 'CNN-LSTM', te_y_f, pred, seed=seed)
                append_cost_row(
                    repeated_cost_rows, fold, 'CNN-LSTM',
                    train_seconds=train_seconds,
                    inference_seconds=infer_seconds,
                    n_train_epochs=len(tr_y_f),
                    n_test_epochs=len(te_y_f),
                    n_parameters=model_num_parameters(lstm_rs),
                    model_size_mb_value=model_size_mb(lstm_rs),
                    seed=seed,
                    notes='Repeated-seed confirmation run.'
                )
                del lstm_rs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if 'Transformer' in REPEATED_SEED_MODELS:
                set_global_seed(seed + 70_000 + fold)
                tf_rs = SleepTransformerLite().to(DEVICE)
                start = time.perf_counter()
                tf_rs, _ = train_model(
                    tf_rs, tr_seq_ld_f, va_seq_ld_f,
                    epochs=REPEATED_SEED_EPOCHS,
                    lr=3e-4,
                    class_weights_tensor=cw_f,
                    patience=REPEATED_SEED_PATIENCE
                )
                train_seconds = time.perf_counter() - start
                proba, _, infer_seconds = timed_predict_proba_torch(tf_rs, te_seq_ld_f)
                pred = proba.argmax(1)

                append_metric_row(repeated_rows, fold, 'Transformer', te_y_f, pred)
                repeated_rows[-1]['seed'] = int(seed)
                append_per_class_rows(repeated_per_class_rows, fold, 'Transformer', te_y_f, pred, seed=seed)
                append_cost_row(
                    repeated_cost_rows, fold, 'Transformer',
                    train_seconds=train_seconds,
                    inference_seconds=infer_seconds,
                    n_train_epochs=len(tr_y_f),
                    n_test_epochs=len(te_y_f),
                    n_parameters=model_num_parameters(tf_rs),
                    model_size_mb_value=model_size_mb(tf_rs),
                    seed=seed,
                    notes='Repeated-seed confirmation run.'
                )
                del tf_rs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Save partial after each fold to make long runs recoverable.
            pd.DataFrame(repeated_rows).to_csv(RESULTS_DIR / 'repeated_seed_sequence_results_partial.csv', index=False)
            pd.DataFrame(repeated_per_class_rows).to_csv(RESULTS_DIR / 'repeated_seed_sequence_per_stage_metrics_partial.csv', index=False)
            pd.DataFrame(repeated_cost_rows).to_csv(RESULTS_DIR / 'repeated_seed_sequence_cost_partial.csv', index=False)

            del tr_eeg_f, tr_eog_f, tr_y_f, va_eeg_f, va_eog_f, va_y_f, te_eeg_f, te_eog_f, te_y_f
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    repeated_df = pd.DataFrame(repeated_rows)
    repeated_per_class_df = pd.DataFrame(repeated_per_class_rows)
    repeated_cost_df = pd.DataFrame(repeated_cost_rows)

    repeated_summary = (repeated_df
        .groupby('model')
        .agg(
            macro_f1_mean=('macro_f1', 'mean'),
            macro_f1_std=('macro_f1', 'std'),
            kappa_mean=('kappa', 'mean'),
            kappa_std=('kappa', 'std'),
            accuracy_mean=('accuracy', 'mean'),
            accuracy_std=('accuracy', 'std'),
            n_seeds=('seed', 'nunique'),
            n_folds=('fold', 'nunique')
        )
        .sort_values('macro_f1_mean', ascending=False)
        .reset_index()
    )

    print('\n── Repeated-seed summary ──')
    display(repeated_summary.round(4))

    repeated_df.to_csv(RESULTS_DIR / 'repeated_seed_sequence_results.csv', index=False)
    repeated_per_class_df.to_csv(RESULTS_DIR / 'repeated_seed_sequence_per_stage_metrics.csv', index=False)
    repeated_cost_df.to_csv(RESULTS_DIR / 'repeated_seed_sequence_cost.csv', index=False)
    repeated_summary.to_csv(RESULTS_DIR / 'repeated_seed_sequence_summary.csv', index=False)

else:
    print('Repeated-seed confirmation skipped because RUN_REPEATED_SEED_CONFIRMATION=False.')



# ==============================================================================
# Section: 17. Per-Class Error Analysis & N1 Deep-Dive
# ==============================================================================

# %% [code] Cell 44
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
model_pairs = [
    ('CNN-LSTM', lstm_pred, 'darkorange'),
    ('CNN-LSTM + HMM', lstm_smooth, 'chocolate'),
    ('Transformer + HMM', tf_smooth, 'steelblue'),
]

# Per-class F1 bars grouped by stage
per_class_data = {}
for name, pred, _ in model_pairs:
    per_class_data[name] = f1_score(
        te_y,
        pred,
        labels=list(range(N_CLASSES)),
        average=None,
        zero_division=0
    )

x = np.arange(N_CLASSES)
width = 0.25
for i, (name, _, color) in enumerate(model_pairs):
    axes[0].bar(x + i * width, per_class_data[name], width, label=name, color=color, alpha=0.85)
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(STAGE_NAMES)
axes[0].set_ylabel('F1 Score')
axes[0].set_title('Per-Class F1')
axes[0].legend(fontsize=7)

# N1 confusion: what does N1 get confused with?
cm_lstm = confusion_matrix(te_y, lstm_pred, labels=list(range(N_CLASSES)))
cm_tf_hmm = confusion_matrix(te_y, tf_smooth, labels=list(range(N_CLASSES)))

def row_pct(cm, row_idx):
    denom = cm[row_idx].sum()
    if denom == 0:
        return np.zeros(cm.shape[1])
    return cm[row_idx] / denom * 100

n1_row_lstm = row_pct(cm_lstm, 1)
n1_row_tf_hmm = row_pct(cm_tf_hmm, 1)

axes[1].bar(np.arange(N_CLASSES) - 0.2, n1_row_lstm, 0.4, label='CNN-LSTM', color='darkorange', alpha=0.85)
axes[1].bar(np.arange(N_CLASSES) + 0.2, n1_row_tf_hmm, 0.4, label='Transformer+HMM', color='steelblue', alpha=0.85)
axes[1].set_xticks(range(N_CLASSES))
axes[1].set_xticklabels(STAGE_NAMES)
axes[1].set_ylabel('%')
axes[1].set_title('N1 Confusion Breakdown\n(row = true N1)')
axes[1].legend(fontsize=8)

# Entropy distribution by class
for cls, color in zip(range(N_CLASSES), ['C0', 'C1', 'C2', 'C3', 'C4']):
    mask = te_y == cls
    if mask.sum() > 0:
        axes[2].hist(tf_entropy[mask], bins=30, alpha=0.5,
                     label=STAGE_NAMES[cls], color=color, density=True)
axes[2].set_xlabel('Normalized Entropy')
axes[2].set_ylabel('Density')
axes[2].set_title('Prediction Entropy by Stage\n(Transformer)')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'per_class_analysis.png', dpi=120)
plt.show()



# ==============================================================================
# Section: 18. Cross-Fold Channel Ablation Study
# ==============================================================================

# %% [code] Cell 46
# ── Cross-fold channel ablation study ───────────────────────────────────────
# The earlier fold-0 RF-only ablation is replaced with a journal-strengthened
# cross-fold ablation. This keeps the analysis out of "notebook demo" territory.
#
# Modes:
#   EEG only  : EOG channel is zeroed.
#   EOG only  : EEG channel is zeroed.
#   EEG + EOG : both channels retained.
#
# Models:
#   Random Forest, 1D CNN, CNN-LSTM by default.
#
# Outputs:
#   channel_ablation_cv_results.csv
#   channel_ablation_cv_per_stage_metrics.csv
#   channel_ablation_cv_cost.csv
#   channel_ablation_cv_summary.csv

if RUN_CHANNEL_ABLATION_CV:
    channel_rows = []
    channel_per_class_rows = []
    channel_cost_rows = []

    for fold in range(N_FOLDS):
        print(f'\n════ Channel ablation fold {fold+1}/{N_FOLDS} ════')
        tr_rec_base, va_rec_base, te_rec_base = load_fold_records(fold, manifest)

        for mode in CHANNEL_ABLATION_MODES:
            print(f'\n── {mode} ──')
            tr_rec_f = records_with_channel_mode(tr_rec_base, mode)
            va_rec_f = records_with_channel_mode(va_rec_base, mode)
            te_rec_f = records_with_channel_mode(te_rec_base, mode)

            tr_eeg_f, tr_eog_f, tr_y_f = concat_records(tr_rec_f)
            va_eeg_f, va_eog_f, va_y_f = concat_records(va_rec_f)
            te_eeg_f, te_eog_f, te_y_f = concat_records(te_rec_f)

            if 'Random Forest' in CHANNEL_ABLATION_MODELS:
                start = time.perf_counter()
                tr_f = extract_features_batch(tr_eeg_f, tr_eog_f)
                te_f = extract_features_batch(te_eeg_f, te_eog_f)
                scaler_ab = StandardScaler().fit(tr_f)
                rf_ab = RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight=make_class_weight_dict(tr_y_f),
                    n_jobs=-1,
                    random_state=SEED + fold
                )
                rf_ab.fit(scaler_ab.transform(tr_f), tr_y_f)
                train_seconds = time.perf_counter() - start

                start = time.perf_counter()
                proba = align_class_proba(rf_ab.predict_proba(scaler_ab.transform(te_f)), rf_ab.classes_)
                infer_seconds = time.perf_counter() - start
                pred = proba.argmax(1)

                append_metric_row(channel_rows, fold, 'Random Forest', te_y_f, pred)
                channel_rows[-1]['channel_mode'] = mode
                append_per_class_rows(channel_per_class_rows, fold, 'Random Forest', te_y_f, pred)
                channel_per_class_rows[-N_CLASSES:] = [
                    dict(row, channel_mode=mode) for row in channel_per_class_rows[-N_CLASSES:]
                ]
                append_cost_row(
                    channel_cost_rows, fold, 'Random Forest',
                    train_seconds=train_seconds,
                    inference_seconds=infer_seconds,
                    n_train_epochs=len(tr_y_f),
                    n_test_epochs=len(te_y_f),
                    model_size_mb_value=sklearn_object_size_mb({'model': rf_ab, 'scaler': scaler_ab}),
                    notes=f'Channel ablation: {mode}'
                )
                channel_cost_rows[-1]['channel_mode'] = mode

            if '1D CNN' in CHANNEL_ABLATION_MODELS or 'CNN-LSTM' in CHANNEL_ABLATION_MODELS:
                cw_f = torch.FloatTensor(make_class_weight_vector(tr_y_f)).to(DEVICE)

            if '1D CNN' in CHANNEL_ABLATION_MODELS:
                set_global_seed(SEED + 80_000 + fold)
                tr_ds_f = EpochDataset(tr_eeg_f, tr_eog_f, tr_y_f)
                va_ds_f = EpochDataset(va_eeg_f, va_eog_f, va_y_f)
                te_ds_f = EpochDataset(te_eeg_f, te_eog_f, te_y_f)
                tr_ld_f = DataLoader(
                    tr_ds_f, batch_size=256, shuffle=True, num_workers=0,
                    generator=torch_generator_for(SEED + 81_000 + fold)
                )
                va_ld_f = DataLoader(va_ds_f, batch_size=256, shuffle=False, num_workers=0)
                te_ld_f = DataLoader(te_ds_f, batch_size=256, shuffle=False, num_workers=0)

                cnn_ab = CNN1D().to(DEVICE)
                start = time.perf_counter()
                cnn_ab, _ = train_model(
                    cnn_ab, tr_ld_f, va_ld_f,
                    epochs=CHANNEL_ABLATION_DEEP_EPOCHS,
                    lr=1e-3,
                    class_weights_tensor=cw_f,
                    patience=CHANNEL_ABLATION_DEEP_PATIENCE
                )
                train_seconds = time.perf_counter() - start
                proba, _, infer_seconds = timed_predict_proba_torch(cnn_ab, te_ld_f)
                pred = proba.argmax(1)

                append_metric_row(channel_rows, fold, '1D CNN', te_y_f, pred)
                channel_rows[-1]['channel_mode'] = mode
                append_per_class_rows(channel_per_class_rows, fold, '1D CNN', te_y_f, pred)
                channel_per_class_rows[-N_CLASSES:] = [
                    dict(row, channel_mode=mode) for row in channel_per_class_rows[-N_CLASSES:]
                ]
                append_cost_row(
                    channel_cost_rows, fold, '1D CNN',
                    train_seconds=train_seconds,
                    inference_seconds=infer_seconds,
                    n_train_epochs=len(tr_y_f),
                    n_test_epochs=len(te_y_f),
                    n_parameters=model_num_parameters(cnn_ab),
                    model_size_mb_value=model_size_mb(cnn_ab),
                    notes=f'Channel ablation: {mode}'
                )
                channel_cost_rows[-1]['channel_mode'] = mode

                del cnn_ab
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if 'CNN-LSTM' in CHANNEL_ABLATION_MODELS:
                set_global_seed(SEED + 90_000 + fold)
                tr_seq_ds_f = SeqEpochDataset(tr_rec_f)
                va_seq_ds_f = SeqEpochDataset(va_rec_f)
                te_seq_ds_f = SeqEpochDataset(te_rec_f)
                tr_seq_ld_f = DataLoader(
                    tr_seq_ds_f, batch_size=64, shuffle=True, num_workers=0,
                    generator=torch_generator_for(SEED + 91_000 + fold)
                )
                va_seq_ld_f = DataLoader(va_seq_ds_f, batch_size=64, shuffle=False, num_workers=0)
                te_seq_ld_f = DataLoader(te_seq_ds_f, batch_size=64, shuffle=False, num_workers=0)

                lstm_ab = CNNLSTM().to(DEVICE)
                start = time.perf_counter()
                lstm_ab, _ = train_model(
                    lstm_ab, tr_seq_ld_f, va_seq_ld_f,
                    epochs=CHANNEL_ABLATION_DEEP_EPOCHS,
                    lr=5e-4,
                    class_weights_tensor=cw_f,
                    patience=CHANNEL_ABLATION_DEEP_PATIENCE
                )
                train_seconds = time.perf_counter() - start
                proba, _, infer_seconds = timed_predict_proba_torch(lstm_ab, te_seq_ld_f)
                pred = proba.argmax(1)

                append_metric_row(channel_rows, fold, 'CNN-LSTM', te_y_f, pred)
                channel_rows[-1]['channel_mode'] = mode
                append_per_class_rows(channel_per_class_rows, fold, 'CNN-LSTM', te_y_f, pred)
                channel_per_class_rows[-N_CLASSES:] = [
                    dict(row, channel_mode=mode) for row in channel_per_class_rows[-N_CLASSES:]
                ]
                append_cost_row(
                    channel_cost_rows, fold, 'CNN-LSTM',
                    train_seconds=train_seconds,
                    inference_seconds=infer_seconds,
                    n_train_epochs=len(tr_y_f),
                    n_test_epochs=len(te_y_f),
                    n_parameters=model_num_parameters(lstm_ab),
                    model_size_mb_value=model_size_mb(lstm_ab),
                    notes=f'Channel ablation: {mode}'
                )
                channel_cost_rows[-1]['channel_mode'] = mode

                del lstm_ab
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            pd.DataFrame(channel_rows).to_csv(RESULTS_DIR / 'channel_ablation_cv_results_partial.csv', index=False)
            pd.DataFrame(channel_per_class_rows).to_csv(RESULTS_DIR / 'channel_ablation_cv_per_stage_metrics_partial.csv', index=False)
            pd.DataFrame(channel_cost_rows).to_csv(RESULTS_DIR / 'channel_ablation_cv_cost_partial.csv', index=False)

            del tr_eeg_f, tr_eog_f, tr_y_f, va_eeg_f, va_eog_f, va_y_f, te_eeg_f, te_eog_f, te_y_f
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    channel_df = pd.DataFrame(channel_rows)
    channel_per_class_df = pd.DataFrame(channel_per_class_rows)
    channel_cost_df = pd.DataFrame(channel_cost_rows)

    channel_summary = (channel_df
        .groupby(['model', 'channel_mode'])
        .agg(
            macro_f1_mean=('macro_f1', 'mean'),
            macro_f1_std=('macro_f1', 'std'),
            kappa_mean=('kappa', 'mean'),
            kappa_std=('kappa', 'std'),
            accuracy_mean=('accuracy', 'mean'),
            accuracy_std=('accuracy', 'std'),
            n_folds=('fold', 'nunique')
        )
        .reset_index()
        .sort_values(['model', 'macro_f1_mean'], ascending=[True, False])
    )

    print('\n── Channel-ablation CV summary ──')
    display(channel_summary.round(4))

    channel_df.to_csv(RESULTS_DIR / 'channel_ablation_cv_results.csv', index=False)
    channel_per_class_df.to_csv(RESULTS_DIR / 'channel_ablation_cv_per_stage_metrics.csv', index=False)
    channel_cost_df.to_csv(RESULTS_DIR / 'channel_ablation_cv_cost.csv', index=False)
    channel_summary.to_csv(RESULTS_DIR / 'channel_ablation_cv_summary.csv', index=False)

else:
    print('Channel ablation skipped because RUN_CHANNEL_ABLATION_CV=False.')



# ==============================================================================
# Section: 19. Save All Models
# ==============================================================================

# %% [code] Cell 48
# ── Persist trained fold-0 models and metadata ─────────────────────────────
torch.save(cnn_model.state_dict(), RESULTS_DIR / 'cnn1d.pt')
torch.save(cnnlstm.state_dict(), RESULTS_DIR / 'cnn_lstm.pt')
torch.save(transformer.state_dict(), RESULTS_DIR / 'transformer.pt')

with open(RESULTS_DIR / 'rf_model.pkl', 'wb') as f:
    pickle.dump({
        'model': rf,
        'scaler': scaler_rf,
        'feature_names': FEATURE_NAMES,
        'hmm_A': A_hmm,
        'hmm_pi': pi_hmm,
        'stage_names': STAGE_NAMES,
        'preprocessing_hash': PREPROCESSING_HASH,
        'preprocessing_config': PREPROCESSING_CONFIG,
        'notes': (
            'Fold-0 models. HMM transition matrix was fitted within training records; '
            'Viterbi smoothing should be applied per record. Attention weights are not '
            'extracted from the Transformer in this version.'
        )
    }, f)

run_metadata = {
    'seed': SEED,
    'data_root': str(DATA_ROOT),
    'max_subjects': MAX_SUBJECTS,
    'use_subsets': USE_SUBSETS,
    'n_manifest_records': int(len(manifest)),
    'n_subjects': int(manifest.subject_id.nunique()),
    'n_folds': int(N_FOLDS),
    'epoch_sec': int(EPOCH_SEC),
    'fs': int(FS),
    'seq_len': int(SEQ_LEN),
    'wake_trim_sec': int(WAKE_TRIM),
    'preprocessing_hash': PREPROCESSING_HASH,
    'preprocessing_config': PREPROCESSING_CONFIG,
    'filter_continuous_before_epoching': FILTER_CONTINUOUS_BEFORE_EPOCHING,
    'cache_dir': str(CACHE_DIR),
    'paper_run': bool(PAPER_RUN),
    'paper_run_deep_cv': bool(PAPER_RUN_DEEP_CV),
    'record_boundary_safe_sequences': True,
    'record_boundary_safe_hmm': True,
    'record_boundary_safe_transition_analysis': True,
    'validation_selected_entropy_thresholds': True,
    'record_level_metrics_saved': True,
    'probability_diagnostics_saved': True,
    'force_rebuild_cache': bool(FORCE_REBUILD_CACHE),
    'run_repeated_seed_confirmation': bool(RUN_REPEATED_SEED_CONFIRMATION),
    'repeated_seeds': REPEATED_SEEDS,
    'repeated_seed_models': REPEATED_SEED_MODELS,
    'repeated_seed_folds': REPEATED_SEED_FOLDS,
    'run_channel_ablation_cv': bool(RUN_CHANNEL_ABLATION_CV),
    'channel_ablation_modes': CHANNEL_ABLATION_MODES,
    'channel_ablation_models': CHANNEL_ABLATION_MODELS,
    'full_cv_per_stage_metrics_saved': True,
    'cv_computational_cost_saved': True,
    'split_stage_distributions_saved': True,
    'hmm_reported_as_tradeoff': True,
    'entropy_deferral_reported_as_selective_prediction': True,
}
with open(RESULTS_DIR / 'run_metadata.pkl', 'wb') as f:
    pickle.dump(run_metadata, f)

with open(RESULTS_DIR / 'run_metadata.json', 'w') as f:
    json.dump(run_metadata, f, indent=2, default=str)

print('All fold-0 models and metadata saved to', RESULTS_DIR)
print('\n══ Pipeline complete ══')
print(f'Results directory: {RESULTS_DIR.resolve()}')



# ==============================================================================
# Section: Summary
# ==============================================================================

