"""Shared helpers for stages 01-07: data loading, leakage-safe pipelines, metrics, IO.

Two rules are enforced here rather than repeated in every script:

1. Every fitted transform (vectoriser, feature selector, scaler, SVD) lives
   inside a :class:`sklearn.pipeline.Pipeline` so that cross-validation refits it
   on the training folds only.
2. Every prediction/scores file is stored with a fixed column order
   (``config.CLASSES``) so evaluation can overlay models without alignment bugs.
"""

from __future__ import annotations

import functools
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402

from scipy import sparse  # noqa: E402
from sklearn.base import BaseEstimator, clone  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402
from sklearn.feature_selection import SelectKBest, mutual_info_classif  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

# ``SVC(probability=True)`` is removed in sklearn 1.11; 1.9+ needs CalibratedClassifierCV.
_SKLEARN_VERSION = tuple(int(part) for part in sklearn.__version__.split(".")[:2])
_SVC_PROBABILITY_REMOVED = _SKLEARN_VERSION >= (1, 9)

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - lightgbm is a hard requirement
    raise SystemExit("lightgbm is required: pip install lightgbm") from exc


def banner(title: str, width: int = 78) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


@contextmanager
def stage_timer(label: str):
    start = time.perf_counter()
    print(f"[{label}] start", flush=True)
    yield
    print(f"[{label}] done in {time.perf_counter() - start:.1f}s", flush=True)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def relative_label(path: Path) -> str:
    """Project-relative path when possible, absolute otherwise.

    Keeps log lines readable when an output directory is not nested inside
    ``config.ROOT`` (for example a remapped working directory on Kaggle).
    """
    try:
        return str(Path(path).relative_to(C.ROOT))
    except ValueError:
        return str(path)


def load_splits() -> dict[str, pd.DataFrame]:
    """Load the canonical splits produced by stage 01."""
    frames: dict[str, pd.DataFrame] = {}
    for split, path in C.SPLIT_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run `python src/01_eda_and_cleaning.py`.")
        frames[split] = pd.read_parquet(path)
    return frames


def labels(frames: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    return {split: frame["target"].to_numpy(dtype=np.int64) for split, frame in frames.items()}


def matrix(rep: str, split: str) -> np.ndarray | sparse.csr_matrix:
    """Load one cached feature matrix (dense float32 or sparse CSR)."""
    path = C.feature_path(rep, split)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python src/02_feature_extract.py` first."
        )
    if rep in C.SPARSE_REPRESENTATIONS:
        return sparse.load_npz(path).tocsr()
    return np.load(path)


def check_artifact_consistency() -> list[str]:
    """Check the cached features against the ``k`` recorded in ``meta.json``.

    Stage 02 writes the selected tripeptide matrix and the ``k`` it used
    separately, so an interrupted or re-run stage 02 can leave them describing
    different feature spaces - screening would then score a different feature
    set from the model that is eventually fitted. Returns a list of warnings,
    empty when everything agrees.
    """
    warnings: list[str] = []
    meta = C.meta()

    k = meta.get("select_k")
    if k is not None:
        cached = C.feature_path("tripeptide", "train")
        if not cached.exists():
            warnings.append("cached 'tripeptide' matrix is missing; re-run stage 02")
        else:
            cols = matrix("tripeptide", "train").shape[1]
            if cols != int(k):
                warnings.append(
                    f"select_k={k} in meta.json but the cached 'tripeptide' matrix has {cols} "
                    f"columns; the final pipeline would select {k} of "
                    f"{matrix(C.RAW_INPUT['tripeptide'], 'train').shape[1]} instead. "
                    "Re-run `python src/02_feature_extract.py`."
                )

    names_path = C.DATA_PROCESSED / "feature_names.json"
    if names_path.exists():
        names = load_json(names_path)
        for rep, rep_names in names.items():
            try:
                cols = matrix(rep, "train").shape[1]
            except FileNotFoundError:
                continue
            if len(rep_names) != cols:
                warnings.append(
                    f"feature_names.json lists {len(rep_names)} names for '{rep}' "
                    f"but the cached matrix has {cols} columns"
                )

    for split, path in C.SPLIT_PATHS.items():
        if not path.exists():
            warnings.append(f"missing split file for '{split}': {path.name}")

    return warnings


def representation(rep: str) -> dict[str, Any]:
    return {split: matrix(rep, split) for split in ("train", "val", "test")}


def feature_names(rep: str) -> list[str]:
    """Human-readable column names for a representation (length must match)."""
    names_path = C.DATA_PROCESSED / "feature_names.json"
    if not names_path.exists():
        raise FileNotFoundError(
            f"Missing {names_path}. Run `python src/02_feature_extract.py` first."
        )
    names = load_json(names_path)[rep]
    expected = matrix(rep, "train").shape[1]
    if len(names) != expected:
        raise ValueError(
            f"feature_names.json has {len(names)} names for {rep} but the matrix has {expected} columns"
        )
    return names


def vectoriser(n: int, max_features: int | None = None) -> CountVectorizer:
    """Character n-gram vectoriser over amino-acid k-mers (fast, C-level counting)."""
    return CountVectorizer(
        analyzer="char",
        ngram_range=(n, n),
        lowercase=False,
        min_df=2,
        max_features=max_features,
        dtype=np.float32,
    )


def normalise_rows(mat: sparse.csr_matrix, per_sequence: np.ndarray) -> sparse.csr_matrix:
    """Turn k-mer counts into within-sequence frequencies (length-independent)."""
    out = mat.tocsr(copy=True).astype(np.float32)
    scale = np.asarray(per_sequence, dtype=np.float32).reshape(-1, 1)
    scale[scale <= 0] = 1.0
    return sparse.diags((1.0 / scale).ravel()).dot(out).tocsr()


def supervised_selector(k: int, random_state: int = C.SEED) -> SelectKBest:
    score_func = functools.partial(mutual_info_classif, random_state=random_state, n_neighbors=3)
    return SelectKBest(score_func=score_func, k=k)


class SupervisedSelector(BaseEstimator):
    """SelectKBest wrapper that subsamples for a stable, faster MI estimate."""

    def __init__(self, k: int = 500, max_rows: int = C.MI_FIT_MAX_ROWS, random_state: int = C.SEED):
        self.k = k
        self.max_rows = max_rows
        self.random_state = random_state

    def fit(self, X, y):
        rows = np.arange(X.shape[0])
        if self.max_rows and X.shape[0] > self.max_rows:
            keep, _ = _stratified_subsample(y, self.max_rows, self.random_state)
            rows = np.sort(keep)
        selector = supervised_selector(self.k, self.random_state)
        selector.fit(X[rows], np.asarray(y)[rows])
        self.selector_ = selector
        self.support_ = selector.get_support()
        self.scores_ = selector.scores_
        self.pvalues_ = selector.pvalues_
        return self

    def transform(self, X):
        if not hasattr(self, "selector_"):
            raise RuntimeError("SupervisedSelector must be fitted before transform")
        return self.selector_.transform(X)

    def get_support(self, indices: bool = False):
        return self.selector_.get_support(indices=indices)

    def get_feature_names_out(self, input_features=None):
        return self.selector_.get_feature_names_out(input_features)


class SvdEmbedding(BaseEstimator):
    """Thin wrapper around TruncatedSVD that records the explained variance."""

    def __init__(self, n_components: int = C.SVD_COMPONENTS, random_state: int = C.SEED):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        from sklearn.decomposition import TruncatedSVD

        min_dim = min(X.shape) - 1
        self.n_components_ = max(2, min(self.n_components, min_dim))
        self.svd_ = TruncatedSVD(
            n_components=self.n_components_, random_state=self.random_state, algorithm="randomized"
        )
        self.svd_.fit(X)
        self.explained_variance_ratio_ = self.svd_.explained_variance_ratio_
        self.components_ = self.svd_.components_
        return self

    def transform(self, X):
        if not hasattr(self, "svd_"):
            raise RuntimeError("SvdEmbedding must be fitted before transform")
        return self.svd_.transform(X).astype(np.float32)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"SVD_{i:03d}" for i in range(self.n_components_)])


def _stratified_subsample(y: Sequence[int], n: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (kept_indices, dropped_indices) preserving class proportions."""
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    keep_parts, drop_parts = [], []
    for cls, cnt in zip(classes, counts):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        take = int(round(cnt * min(1.0, n / len(y))))
        keep_parts.append(idx[:take])
        drop_parts.append(idx[take:])
    return np.sort(np.concatenate(keep_parts)), np.sort(np.concatenate(drop_parts))


def stratified_subsample_indices(y: Sequence[int], n: int, random_state: int = C.SEED):
    return _stratified_subsample(y, n, random_state)


def base_model(kind: str, params: dict | None = None, probability: bool | None = None):
    """Instantiate one of the three required models with sane defaults."""
    params = dict(params or {})
    if kind == "rf":
        defaults = dict(
            n_estimators=500,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=C.SEED,
            n_jobs=C.N_JOBS,
        )
        return RandomForestClassifier(**{**defaults, **params})
    if kind == "lgbm":
        defaults = dict(
            n_estimators=500,
            learning_rate=0.1,
            num_leaves=31,
            max_depth=-1,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=0.0,
            class_weight="balanced",
            random_state=C.SEED,
            n_jobs=C.N_JOBS,
            verbose=-1,
        )
        return LGBMClassifier(**{**defaults, **params})
    if kind == "svm":
        prob = C.SVM_PROBABILITY if probability is None else probability
        prob = bool(params.pop("probability", prob))
        defaults = dict(
            kernel=C.SVM_KERNEL,
            C=10.0,
            gamma="scale",
            class_weight="balanced",
            cache_size=C.SVM_CACHE_MB,
            decision_function_shape="ovr",
            random_state=C.SEED,
        )
        model = SVC(**{**defaults, **params})
        return calibrated_svc(model, prob)
    raise KeyError(kind)


def calibrated_svc(model: SVC, prob: bool):
    """Attach probabilities to an :class:`SVC` the way the installed sklearn wants.

    ``SVC(probability=True)`` is deprecated from sklearn 1.9 and removed in 1.11;
    the supported replacement is ``CalibratedClassifierCV(SVC(), ensemble=False)``,
    which performs the same internal 5-fold Platt scaling.

    A plain :class:`SVC` is returned whenever probabilities are not needed, so
    stages 03-05 keep the fast, uncalibrated path and their ``decision_function``
    handling stays valid.
    """
    if not prob:
        return model
    if _SVC_PROBABILITY_REMOVED:
        from sklearn.calibration import CalibratedClassifierCV

        return CalibratedClassifierCV(model, method="sigmoid", cv=5, ensemble=False)
    model.set_params(probability=True)
    return model


def LGBMClassifier(**kwargs):
    return lgb.LGBMClassifier(**kwargs)


def needs_scaling(kind: str) -> bool:
    return kind == "svm"


def build_pipeline(rep: str, kind: str, params: dict | None = None, probability: bool | None = None):
    """Leakage-safe pipeline for one representation.

    ``tripeptide`` keeps :class:`SupervisedSelector` inside the pipeline because
    feature selection consumes ``y``; refitting it per fold is what keeps the
    cross-validated score honest. ``svd`` is unsupervised and therefore safe to
    fit once on the training split only.
    """
    model = base_model(kind, params, probability=probability)
    steps: list[tuple[str, Any]] = []
    if rep == "tripeptide":
        steps.append(("select", SupervisedSelector(k=C.select_k())))
    if needs_scaling(kind):
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", model))
    from sklearn.pipeline import Pipeline

    return Pipeline(steps)


def build_base_pipeline(rep: str, kind: str, params: dict | None = None, probability: bool | None = None):
    """Pipeline used by stage 02 representation screening (no inner selector)."""
    model = base_model(kind, params, probability=probability)
    steps: list[tuple[str, Any]] = []
    if needs_scaling(kind):
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", model))
    from sklearn.pipeline import Pipeline

    return Pipeline(steps)


def aligned_scores(model, X) -> np.ndarray | None:
    """Return a (n, 6) score matrix whose columns follow ``config.CLASSES``."""
    if not hasattr(model, "predict_proba"):
        if hasattr(model, "decision_function"):
            raw = np.asarray(model.decision_function(X))
            if raw.ndim == 1:
                raw = np.column_stack([-raw, raw])
            return _softmax_ovr(raw)
        return None
    proba = np.asarray(model.predict_proba(X))
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    return _reorder_columns(proba, getattr(model, "classes_", None))


def _reorder_columns(proba: np.ndarray, classes) -> np.ndarray:
    if classes is None:
        return proba
    order = [int(np.flatnonzero(np.asarray(classes) == c)[0]) if c in classes else -1 for c in C.CLASSES]
    if -1 in order:
        filled = np.zeros((proba.shape[0], len(C.CLASSES)), dtype=proba.dtype)
        for col, cls in enumerate(classes):
            idx = C.CLASSES.index(int(cls))
            filled[:, idx] = proba[:, col]
        return filled
    return proba[:, order]


def _softmax_ovr(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def classification_metrics(y_true, y_pred, y_score=None) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        if y_score.ndim == 2 and y_score.shape[1] == len(C.CLASSES):
            onehot = np.zeros_like(y_score)
            for col, cls in enumerate(C.CLASSES):
                onehot[:, col] = (y_true == cls).astype(float)
            metrics["auprc_macro"] = float(
                average_precision_score(onehot, y_score, average="macro")
            )
    return metrics


def per_class_table(y_true, y_pred) -> pd.DataFrame:
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(C.CLASSES), zero_division=0
    )
    rows = []
    for idx, cls in enumerate(C.CLASSES):
        mask = np.asarray(y_true) == cls
        correct = int((np.asarray(y_pred)[mask] == cls).sum())
        rows.append(
            {
                "EC Class": cls,
                "Family": C.EC_NAMES[cls],
                "Precision": precision[idx],
                "Recall": recall[idx],
                "F1": f1[idx],
                "Support": int(support[idx]),
                "Accuracy": (correct / support[idx]) if support[idx] else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def prediction_path(model_tag: str, split: str, suffix: str = "") -> Path:
    return C.PREDICTIONS_DIR / f"{C.safe_name(model_tag)}__{split}{suffix}.npy"


def prediction_meta_path(model_tag: str, split: str) -> Path:
    return C.PREDICTIONS_DIR / f"{C.safe_name(model_tag)}__{split}__meta.json"


def save_predictions(model_tag: str, split: str, y_pred, y_score=None) -> dict:
    np.save(prediction_path(model_tag, split), np.asarray(y_pred, dtype=np.int64))
    meta = {"classes": list(C.CLASSES), "n": int(len(y_pred)), "has_score": y_score is not None}
    if y_score is not None:
        np.save(prediction_path(model_tag, split, "__score"), np.asarray(y_score, dtype=np.float32))
    save_json(prediction_meta_path(model_tag, split), meta)
    return meta


def load_predictions(model_tag: str, split: str) -> tuple[np.ndarray, np.ndarray | None]:
    y_pred = np.load(prediction_path(model_tag, split))
    score_path = prediction_path(model_tag, split, "__score")
    y_score = np.load(score_path) if score_path.exists() else None
    return y_pred, y_score


def available_model_tags() -> list[str]:
    tags = set()
    for path in C.PREDICTIONS_DIR.glob("*__test__meta.json"):
        tags.add(path.name.split("__test")[0])
    return sorted(tags)


def metrics_path(model_tag: str) -> Path:
    return C.METRICS_DIR / f"{C.safe_name(model_tag)}.json"


def save_model_metrics(model_tag: str, payload: dict) -> None:
    payload = dict(payload)
    payload["model_tag"] = model_tag
    save_json(metrics_path(model_tag), payload)
    print(f"saved metrics -> {relative_label(metrics_path(model_tag))}")


def load_model_metrics(model_tag: str) -> dict:
    path = metrics_path(model_tag)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return load_json(path)


def markdown_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    def fmt(value):
        if isinstance(value, float):
            return floatfmt.format(value)
        return str(value)

    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def subsample_for_search(X, y, max_rows: int, random_state: int = C.SEED):
    """Stratified row subsample used to keep hyperparameter search affordable."""
    y = np.asarray(y)
    if not max_rows or X.shape[0] <= max_rows:
        return X, y
    keep, _ = _stratified_subsample(y, max_rows, random_state)
    return X[keep], y[keep]


def to_dense(X):
    return X.toarray() if sparse.issparse(X) else X


def clone_safe(estimator):
    return clone(estimator)


def report_memory(X) -> str:
    if sparse.issparse(X):
        return f"{X.shape} sparse={X.nnz} nnz, {C.human_bytes(X.data.nbytes + X.indices.nbytes)}"
    return f"{X.shape} dense, {C.human_bytes(X.nbytes)}"
