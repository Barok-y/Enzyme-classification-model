"""Central configuration for the enzyme-classification pipeline.

Every path, seed and tunable constant used by scripts 01-07 lives here so that
all stages agree on the same numbers. Anything that changes the data itself
(``ENZYME_CAP``) is persisted into ``data/processed/meta.json`` by stage 01 and
read back by every later stage, so the splits are guaranteed to be identical.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
PREDICTIONS_DIR = ARTIFACTS / "predictions"
METRICS_DIR = ARTIFACTS / "metrics"
FEATURE_PIPELINE_DIR = ARTIFACTS / "feature_pipeline"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

META_PATH = DATA_PROCESSED / "meta.json"
SPLIT_PATHS = {
    "train": DATA_PROCESSED / "train.parquet",
    "val": DATA_PROCESSED / "val.parquet",
    "test": DATA_PROCESSED / "test.parquet",
}
SPARSE_REPRESENTATIONS = frozenset({"tripeptide_full"})


def feature_path(rep: str, split: str) -> Path:
    suffix = "npz" if rep in SPARSE_REPRESENTATIONS else "npy"
    return DATA_PROCESSED / f"X_{rep}_{split}.{suffix}"

HF_DATASET = os.environ.get("ENZYME_HF_DATASET", "DanielHesslow/SwissProt-EC")
HF_SPLIT = os.environ.get("ENZYME_HF_SPLIT", "train")
HF_FILE = "train-00000-of-00001.parquet"

SEED = int(os.environ.get("ENZYME_SEED", 42))
N_JOBS = int(os.environ.get("ENZYME_N_JOBS", "-1"))

TEST_SIZE_HOLDOUT = float(os.environ.get("ENZYME_TEST_SIZE", 0.30))
VAL_FRACTION_OF_HOLDOUT = 0.50

CAP = os.environ.get("ENZYME_CAP", "none")
CAP = None if CAP.lower() in {"none", "", "null", "full"} else int(CAP)

AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
VALID_AA = frozenset(AMINO_ACIDS)
CLASSES = (1, 2, 3, 4, 5, 6)
EC_NAMES = {
    1: "Oxidoreductases",
    2: "Transferases",
    3: "Hydrolases",
    4: "Lyases",
    5: "Isomerases",
    6: "Ligases",
}

MIN_SEQ_LEN = int(os.environ.get("ENZYME_MIN_SEQ_LEN", 30))

KMER_MAX_FEATURES = int(os.environ.get("ENZYME_KMER_FEATURES", 8000))
SVD_COMPONENTS = int(os.environ.get("ENZYME_SVD_COMPONENTS", 128))
SELECT_K = int(os.environ.get("ENZYME_SELECT_K", 500))
K_SWEEP = tuple(
    int(v) for v in os.environ.get("ENZYME_K_SWEEP", "50,100,250,500,1000,2000,4000,8000").split(",")
)
MI_FIT_MAX_ROWS = int(os.environ.get("ENZYME_MI_MAX_ROWS", 40000))

SCREEN_CV_FOLDS = int(os.environ.get("ENZYME_SCREEN_CV", 3))
SCREEN_MAX_ROWS = int(os.environ.get("ENZYME_SCREEN_MAX_ROWS", 60000))
SCREEN_N_ESTIMATORS = int(os.environ.get("ENZYME_SCREEN_N_ESTIMATORS", 150))
TUNE_CV_FOLDS = int(os.environ.get("ENZYME_TUNE_CV", 5))
TUNE_MAX_ROWS = int(os.environ.get("ENZYME_TUNE_MAX_ROWS", 0))

SVM_KERNEL = os.environ.get("ENZYME_SVM_KERNEL", "rbf")
SVM_CACHE_MB = int(os.environ.get("ENZYME_SVM_CACHE_MB", 4000))
SVM_PROBABILITY = os.environ.get("ENZYME_SVM_PROBABILITY", "0") == "1"

STACK_CV_FOLDS = int(os.environ.get("ENZYME_STACK_CV", 5))
SMOTE_ENABLED = os.environ.get("ENZYME_SMOTE", "0") == "1"
SMOTE_MAX_ROWS = int(os.environ.get("ENZYME_SMOTE_MAX_ROWS", 30000))

# An ensemble needs every base model on the *same* feature space, so the
# representation is resolved from the tuned stages 03-05 (or forced with
# ENZYME_ENSEMBLE_REP). ENSEMBLE_MAX_ROWS 0 means "no row cap".
ENSEMBLE_REP = os.environ.get("ENZYME_ENSEMBLE_REP", "") or None
ENSEMBLE_MAX_ROWS = int(os.environ.get("ENZYME_ENSEMBLE_MAX_ROWS", 0))
ENSEMBLE_SVM_PROBABILITY = os.environ.get("ENZYME_ENSEMBLE_SVM_PROB", "1") == "1"
ENSEMBLE_META_MAX_ITER = int(os.environ.get("ENZYME_ENSEMBLE_META_ITER", 2000))
ENSEMBLE_ESTIMATOR_WEIGHTS = os.environ.get("ENZYME_ENSEMBLE_WEIGHTS", "") or None

REFIT_ON_TRAIN_VAL = os.environ.get("ENZYME_REFIT_ON_TRAIN_VAL", "1") == "1"

PRIMARY_METRIC = "macro_f1"
SCORE_TOLERANCE = 1e-9

REPRESENTATIONS = ("aac", "dipeptide", "tripeptide", "svd", "concat")
REQUIRED_REPRESENTATIONS = ("aac", "dipeptide", "tripeptide")
OPTIONAL_REPRESENTATIONS = ("svd", "concat")

RAW_INPUT = {
    "aac": "aac",
    "dipeptide": "dipeptide",
    "tripeptide": "tripeptide_full",
    "svd": "svd",
    "concat": "concat",
}

REP_LABELS = {
    "aac": "AAC (20)",
    "dipeptide": "Dipeptide (400)",
    "tripeptide": "Tripeptide + SelectKBest",
    "svd": f"SVD embedding ({SVD_COMPONENTS})",
    "concat": "AAC + Dipeptide + length + SVD",
}


def representation_label(rep: str) -> str:
    """``REP_LABELS`` with the *resolved* SelectKBest budget filled in."""
    label = REP_LABELS[rep]
    if rep == "tripeptide":
        return f"Tripeptide + SelectKBest (k={select_k()})"
    return label


MODEL_KEYS = {"rf": "RandomForest", "lgbm": "LightGBM", "svm": "SVM"}
MODEL_ORDER = ("rf", "lgbm", "svm")

for _directory in (
    DATA_RAW,
    DATA_PROCESSED,
    ARTIFACTS,
    MODELS_DIR,
    PREDICTIONS_DIR,
    METRICS_DIR,
    FEATURE_PIPELINE_DIR,
    REPORTS,
    FIGURES,
):
    _directory.mkdir(parents=True, exist_ok=True)


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def meta() -> dict:
    """Dataset-level metadata written by stage 01 and consumed by every other stage."""
    if not META_PATH.exists():
        raise FileNotFoundError(
            f"{META_PATH} not found. Run `python src/01_eda_and_cleaning.py` first."
        )
    with META_PATH.open() as handle:
        return json.load(handle)


def sample_cap() -> int | None:
    """Effective stratified subsample size (None means the full cleaned dataset)."""
    stored = meta().get("sample_cap")
    return None if stored is None else int(stored)


def select_k() -> int:
    stored = meta().get("select_k")
    return SELECT_K if stored is None else int(stored)


def apply_cap_overrides(cap: int | None) -> int | None:
    """Apply a CLI ``--cap`` override to the cached metadata.

    Stages 03-06 cannot re-split the data, so a cap given on their command line
    is written into ``meta.json`` and picked up by :func:`sample_cap` from then
    on. Passing ``None`` or ``0`` clears an existing cap.
    """
    if cap is None:
        return sample_cap()
    effective = int(cap) if cap > 0 else None
    stored = sample_cap()
    if effective != stored:
        update_meta(sample_cap=effective)
    return effective


def metrics_paths_for_family(family: str) -> list[Path]:
    """All stage metrics files belonging to one model family (``rf``/``lgbm``/``svm``)."""
    return sorted(METRICS_DIR.glob(f"{safe_name(family)}_*.json"))


def update_meta(**updates) -> dict:
    payload = meta() if META_PATH.exists() else {}
    payload.update(updates)
    with META_PATH.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return payload


def safe_name(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
