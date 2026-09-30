"""Shared training protocol for the three single-model stages (03, 04, 05).

Protocol implemented here (identical for Random Forest, LightGBM and SVM so the
comparison is fair):

1. **Representation screening** - the model is cross-validated on every cached
   representation with fixed hyper-parameters, so only the representation
   changes.
2. **Hyper-parameter tuning** - 5-fold cross-validation *inside the training
   split only*, with every fitted transform inside the pipeline. For the
   tripeptide representation that means ``SelectKBest`` is refitted on each
   training fold, which is the step that most easily leaks labels.
3. **Validation** - the winning configuration is fitted on train and scored on
   the untouched validation split.
4. **Test** - the same configuration is refitted on train+validation (a decision
   already taken) and scored once on the test split, which is never used for any
   decision.
5. **Imbalance experiment** - balanced class weights are compared against no
   weighting and against SMOTE applied inside the cross-validation folds only.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
import common as K

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score


def screening_matrix(rep: str, split: str):
    """Matrix used for the cheap representation screening."""
    return K.matrix("tripeptide" if rep == "tripeptide" else rep, split)


def pipeline_input(rep: str, split: str):
    """Matrix fed to the tuning/final pipeline (raw input, transforms live inside)."""
    return K.matrix(C.RAW_INPUT[rep], split)


def evaluate_split(model, X, y, tag: str, split: str) -> dict:
    y_pred = model.predict(X)
    y_score = K.aligned_scores(model, X)
    metrics = K.classification_metrics(y, y_pred, y_score)
    metrics.update({"n": int(len(y_pred)), "tag": tag, "split": split})
    print(
        f"  {split:5s} {tag:34s} macro_f1={metrics['macro_f1']:.4f} "
        f"acc={metrics['accuracy']:.4f} mcc={metrics['mcc']:.4f}"
        + (f" auprc={metrics['auprc_macro']:.4f}" if "auprc_macro" in metrics else "")
    )
    K.save_predictions(tag, split, y_pred, y_score)
    return metrics


def screen_representations(kind: str, reps, frames, y, params: dict) -> pd.DataFrame:
    rows = []
    cv = StratifiedKFold(n_splits=C.SCREEN_CV_FOLDS, shuffle=True, random_state=C.SEED)
    for rep in reps:
        X = screening_matrix(rep, "train")
        Xs, ys = K.subsample_for_search(X, y["train"], C.SCREEN_MAX_ROWS, C.SEED)
        model = K.build_base_pipeline(rep, kind, params)
        if "n_estimators" in model.named_steps["clf"].get_params():
            model.set_params(clf__n_estimators=C.SCREEN_N_ESTIMATORS)
        started = time.perf_counter()
        scores = cross_val_score(model, Xs, ys, cv=cv, scoring="f1_macro", n_jobs=1)
        rows.append(
            {
                "representation": rep,
                "label": C.representation_label(rep),
                "n_features": int(X.shape[1]),
                "cv_macro_f1_mean": scores.mean(),
                "cv_macro_f1_std": scores.std(),
                "seconds": time.perf_counter() - started,
            }
        )
        print(
            f"  {C.representation_label(rep):38s} dims={X.shape[1]:5d} "
            f"cv macro-F1={scores.mean():.4f} +/- {scores.std():.4f} ({rows[-1]['seconds']:.0f}s)"
        )
        del X, Xs
        gc.collect()
    table = pd.DataFrame(rows).sort_values("cv_macro_f1_mean", ascending=False).reset_index(drop=True)
    return table


def tune(kind: str, rep: str, grid: dict, frames, y, folds: int) -> tuple[dict, dict]:
    X = pipeline_input(rep, "train")
    Xs, ys = K.subsample_for_search(X, y["train"], C.TUNE_MAX_ROWS, C.SEED)
    params = {}
    if kind in {"rf", "lgbm"}:
        params["n_jobs"] = 1
    pipeline = K.build_pipeline(rep, kind, params)
    search = GridSearchCV(
        pipeline,
        dict(grid),
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=C.SEED),
        n_jobs=C.N_JOBS,
        refit=True,
        return_train_score=False,
        verbose=1,
    )
    print(f"grid search: {len(grid)} parameter sets x {folds} folds on {Xs.shape[0]:,} rows x {X.shape[1]} features")
    search.fit(Xs, ys)
    best = dict(search.best_params_)
    for name in list(best):
        if name.startswith("clf__"):
            best[name[len("clf__") :]] = best.pop(name)
    best_index = int(search.best_index_)
    fold_scores = np.array(
        [search.cv_results_[f"split{i}_test_score"][best_index] for i in range(folds)], dtype=float
    )
    cv_info = {
        "folds": folds,
        "macro_f1_mean": float(fold_scores.mean()),
        "macro_f1_std": float(fold_scores.std()),
        "rows_used": int(Xs.shape[0]),
        "features_used": int(X.shape[1]),
        "n_candidates": int(search.n_candidates_),
    }
    return best, cv_info


def imbalance_experiment(kind: str, rep: str, best_params: dict, frames, y, folds: int) -> list[dict]:
    """Compare no weighting, balanced class weights and SMOTE.

    All three variants read the *same* already-selected matrix, so the only thing
    that changes between them is the resampling strategy. SMOTE is wrapped in an
    ``imblearn`` pipeline so that it is refitted inside every cross-validation
    fold: synthetic neighbours are never derived from the validation or test rows.

    ``subsample_for_search`` is applied to ``X`` *and* ``y`` together, because a
    stratified subsample keeps different rows per class - slicing ``y`` by length
    would silently pair features with the wrong labels.
    """
    if not C.SMOTE_ENABLED:
        print("SMOTE experiment skipped (set ENZYME_SMOTE=1 to enable)")
        return []

    X, ys = K.subsample_for_search(screening_matrix(rep, "train"), y["train"], C.SMOTE_MAX_ROWS, C.SEED)
    X = K.to_dense(X)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=C.SEED)
    print(f"imbalance experiment on {X.shape[0]:,} rows x {X.shape[1]} features, {folds}-fold CV")

    def make_estimator(strategy):
        params = dict(best_params)
        params["class_weight"] = "balanced" if strategy == "balanced" else None
        if strategy == "smote":
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
            from sklearn.preprocessing import StandardScaler

            steps = []
            if kind == "svm":
                steps.append(("scale", StandardScaler()))
            steps.append(("clf", K.base_model(kind, params)))
            steps.insert(0, ("smote", SMOTE(random_state=C.SEED, k_neighbors=5)))
            return ImbPipeline(steps)
        return K.build_base_pipeline(rep, kind, params)

    rows = []
    for label, strategy in (("unweighted", None), ("class_weight=balanced", "balanced"), ("SMOTE", "smote")):
        estimator = make_estimator(strategy)
        started = time.perf_counter()
        result = cross_val_score(estimator, X, ys, cv=cv, scoring="f1_macro", n_jobs=1)
        rows.append(
            {
                "strategy": label,
                "cv_macro_f1_mean": result.mean(),
                "cv_macro_f1_std": result.std(),
                "seconds": time.perf_counter() - started,
            }
        )
        print(f"  {label:22s} cv macro-F1 = {result.mean():.4f} +/- {result.std():.4f}")
    return rows


def _aligned(X, y_all: np.ndarray) -> np.ndarray:
    return y_all[: X.shape[0]]


def fit_and_evaluate(kind: str, rep: str, best_params: dict, frames, y, tag: str) -> tuple[dict, dict]:
    """Fit on train -> validation, then refit on train+validation -> test."""
    results: dict[str, dict] = {}

    with K.stage_timer(f"fit on train ({tag})"):
        model = K.build_pipeline(rep, kind, best_params)
        model.fit(pipeline_input(rep, "train"), y["train"])
        results["validation"] = evaluate_split(model, pipeline_input(rep, "val"), y["val"], tag, "val")

    fit_scope = "train"
    if C.REFIT_ON_TRAIN_VAL:
        fit_scope = "train+val"
        with K.stage_timer(f"refit on train+validation ({tag})"):
            X_full = _stack([pipeline_input(rep, "train"), pipeline_input(rep, "val")])
            y_full = np.concatenate([y["train"], y["val"]])
            model = K.build_pipeline(rep, kind, best_params)
            model.fit(X_full, y_full)
            del X_full, y_full
            gc.collect()
    results["test"] = evaluate_split(model, pipeline_input(rep, "test"), y["test"], tag, "test")

    joblib.dump(model, C.MODELS_DIR / f"{C.safe_name(tag)}.joblib", compress=3)
    print(f"model -> {C.MODELS_DIR.name}/{C.safe_name(tag)}.joblib")
    results["fit_scope"] = fit_scope
    return model, results


def _stack(matrices: list):
    from scipy import sparse

    if any(sparse.issparse(m) for m in matrices):
        return sparse.vstack([sparse.csr_matrix(m) for m in matrices]).tocsr()
    return np.vstack(matrices)


def run_single_model(kind: str, tag: str, grid: dict, reps=None, tune_folds: int | None = None, run_imbalance: bool = True):
    """Full training protocol for one model family."""
    K.banner(f"STAGE {tag.upper()} - {C.MODEL_KEYS[kind]}")
    for warning in K.check_artifact_consistency():
        print(f"warning: {warning}")
    frames = K.load_splits()
    y = K.labels(frames)
    folds = tune_folds or C.TUNE_CV_FOLDS
    reps = list(reps or C.REPRESENTATIONS)
    meta = C.meta()

    screen_params = grid.get("screen_params", {})
    with K.stage_timer(f"{tag}: representation screening ({len(reps)} representations)"):
        screen = screen_representations(kind, reps, frames, y, screen_params)
    screen.to_csv(C.METRICS_DIR / f"{tag}_representation_screening.csv", index=False)
    best_rep = screen.iloc[0]["representation"]
    print(f"\nbest representation by screening CV: {C.representation_label(best_rep)}")

    search_grid = {k: v for k, v in grid.items() if k != "screen_params"}
    with K.stage_timer(f"{tag}: 5-fold grid search on the training split"):
        best_params, cv_info = tune(kind, best_rep, search_grid, frames, y, folds)
    print(f"best params: {best_params}")
    print(f"best CV macro-F1: {cv_info['macro_f1_mean']:.4f}")

    imbalance = imbalance_experiment(kind, best_rep, best_params, frames, y, min(folds, 3)) if run_imbalance else []

    model_tag = f"{tag}_{best_rep}"
    _, results = fit_and_evaluate(kind, best_rep, best_params, frames, y, model_tag)

    payload = {
        "model_tag": model_tag,
        "model_family": tag,
        "model_name": C.MODEL_KEYS[kind],
        "representation": best_rep,
        "representation_label": C.representation_label(best_rep),
        "best_params": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in best_params.items()},
        "cross_validation": cv_info,
        "validation": results["validation"],
        "test": results["test"],
        "fit_scope": results["fit_scope"],
        "representation_screening": screen.to_dict("records"),
        "imbalance_experiment": imbalance,
        "select_k": meta.get("select_k"),
        "data": {
            "n_train": int(len(y["train"])),
            "n_val": int(len(y["val"])),
            "n_test": int(len(y["test"])),
            "sample_cap": meta.get("sample_cap"),
        },
    }
    K.save_model_metrics(model_tag, payload)

    pd.DataFrame(
        [{"stage": "validation", **results["validation"]}, {"stage": "test", **results["test"]}]
    ).to_csv(C.METRICS_DIR / f"{tag}_{best_rep}_scores.csv", index=False)
    if imbalance:
        pd.DataFrame(imbalance).to_csv(C.METRICS_DIR / f"{tag}_imbalance.csv", index=False)
    return payload


def model_argparser(stage: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"{stage} - train and evaluate one model family")
    parser.add_argument("--reps", default=",".join(C.REPRESENTATIONS), help="comma separated representations to screen")
    parser.add_argument("--folds", type=int, default=C.TUNE_CV_FOLDS, help="folds for hyper-parameter tuning")
    parser.add_argument("--tune-max-rows", type=int, default=None, help="subsample the training split before grid search")
    parser.add_argument("--skip-imbalance", action="store_true", help="skip the class-weight / SMOTE comparison")
    parser.add_argument("--cap", type=int, default=None, help="override the training subsample cap recorded by stage 01")
    return parser


def run_from_cli(stage: str, kind: str, grid: dict) -> dict:
    args = model_argparser(stage).parse_args()
    C.apply_cap_overrides(args.cap)
    if args.tune_max_rows:
        C.TUNE_MAX_ROWS = args.tune_max_rows
    reps = [r.strip() for r in args.reps.split(",") if r.strip()]
    payload = run_single_model(kind, kind, grid, reps=reps, tune_folds=args.folds, run_imbalance=not args.skip_imbalance)
    print(f"\n{tags_line(payload)}")
    return payload


def tags_line(payload: dict) -> str:
    test = payload["test"]
    val = payload["validation"]
    return (
        f"[{payload['model_tag']}] representation={payload['representation']} "
        f"val macro-F1={val['macro_f1']:.4f} / test macro-F1={test['macro_f1']:.4f} "
        f"/ test MCC={test['mcc']:.4f} / test acc={test['accuracy']:.4f}"
    )
