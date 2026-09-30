"""Stage 06 - hard voting, soft voting and stacking over the three tuned models.

Run: ``python src/06_ensembles.py [--rep tripeptide] [--max-rows 0]``

Same protocol as stages 03-05: train -> validation, then refit on train+val and
score the test split once at the end.

An ensemble cannot mix feature spaces, so all three base models are re-fitted on
one shared representation - by default the one chosen most often by stages 03-05
(ties broken by cross-validated macro-F1), or whatever ``--rep`` forces. Those
re-fitted baselines are saved next to the ensembles so the comparison isolates
the combination rule rather than the feature set.

Stacking uses ``LogisticRegression(class_weight="balanced")`` over internal
out-of-fold probabilities. Soft voting and stacking both need ``predict_proba``,
so the SVM is rebuilt with probability calibration; ``--max-rows`` caps the
fitting rows if that does not fit in RAM, and the cap is recorded in the metrics.
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

from scipy import sparse
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hard voting, soft voting and stacking")
    parser.add_argument(
        "--rep",
        default=C.ENSEMBLE_REP,
        help=f"shared representation (default: resolved from stages 03-05); one of {C.REPRESENTATIONS}",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=C.ENSEMBLE_MAX_ROWS,
        help="cap the rows used to fit the ensembles (0 = use the whole training split)",
    )
    parser.add_argument("--families", default=",".join(C.MODEL_ORDER), help="base model families to combine")
    parser.add_argument("--skip-stacking", action="store_true", help="skip the stacking ensemble")
    parser.add_argument("--no-refit", action="store_true", help="score the test split with the train-only fit")
    return parser.parse_args()


def tuned_spec(family: str) -> dict | None:
    """Read the hyper-parameters chosen by the stage that trained ``family``.

    If a family was trained on several representations, the run with the best
    cross-validated macro-F1 wins.
    """
    best: tuple[float, dict] | None = None
    for path in C.metrics_paths_for_family(family):
        payload = K.load_json(path)
        score = payload.get("cross_validation", {}).get("macro_f1_mean")
        score = float(score) if score is not None else -1.0
        if best is None or score > best[0]:
            best = (score, payload)
    return best[1] if best else None


def resolve_representation(families: list[str]) -> tuple[str, dict]:
    """Pick the representation every base model will share.

    Preference order: ``--rep`` if given, otherwise the representation chosen by
    most families (ties -> highest mean cross-validated macro-F1), otherwise the
    first required representation so the stage still runs standalone.
    """
    specs = {family: tuned_spec(family) for family in families}
    for family in families:
        if not specs[family]:
            print(f"warning: no tuned metrics for '{family}', using library defaults")

    rows = []
    for rep in C.REPRESENTATIONS:
        users = [f for f, s in specs.items() if s and s.get("representation") == rep]
        scores = [
            float(specs[f]["cross_validation"]["macro_f1_mean"])
            for f in users
            if specs[f].get("cross_validation", {}).get("macro_f1_mean") is not None
        ]
        rows.append(
            {
                "representation": rep,
                "label": C.representation_label(rep),
                "families": ",".join(users) or "-",
                "n": len(users),
                "mean_cv_macro_f1": float(np.mean(scores)) if scores else float("nan"),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["n", "mean_cv_macro_f1"], ascending=[False, False], na_position="last"
    ).reset_index(drop=True)
    print("\nrepresentation votes across the tuned stages:")
    print(ranking.to_string(index=False))

    if ranking.iloc[0]["n"] == 0:
        chosen = C.REQUIRED_REPRESENTATIONS[0]
        print(f"\nno stage metrics found, falling back to {C.representation_label(chosen)}")
    else:
        chosen = str(ranking.iloc[0]["representation"])
    return chosen, specs


def pipeline_matrix(rep: str, split: str):
    """Feature matrix for a pipeline (raw input; fitted transforms live inside)."""
    return K.matrix(C.RAW_INPUT[rep], split)


def stack_matrices(matrices: list):
    if any(sparse.issparse(m) for m in matrices):
        return sparse.vstack([sparse.csr_matrix(m) for m in matrices]).tocsr()
    return np.vstack(matrices)


def clean_params(raw: dict | None) -> dict:
    """Grid-search parameters round-trip through JSON in the metrics files."""
    return dict(raw) if raw else {}


def estimator_for(family: str, rep: str, params: dict, probability: bool):
    """One base estimator as a pipeline, identical in shape to stages 03-05."""
    return K.build_pipeline(rep, family, params, probability=probability)


def base_estimators(families, rep, specs, probability) -> list[tuple[str, object]]:
    estimators = []
    for family in families:
        params = clean_params((specs.get(family) or {}).get("best_params"))
        estimators.append((family, estimator_for(family, rep, params, probability)))
        tuned = ", ".join(f"{k}={v}" for k, v in params.items()) or "defaults"
        print(f"  {family:5s} {C.MODEL_KEYS[family]:12s} params: {tuned}")
    return estimators


def weights_for(families) -> list | None:
    """Optional per-model voting weights (same order as ``families``)."""
    if not C.ENSEMBLE_ESTIMATOR_WEIGHTS:
        return None
    raw = [float(v) for v in C.ENSEMBLE_ESTIMATOR_WEIGHTS.split(",")]
    if len(raw) != len(families):
        raise ValueError(f"ENZYME_ENSEMBLE_WEIGHTS needs {len(families)} values, got {len(raw)}")
    return raw


def fitting_data(rep: str, y: dict, splits: tuple[str, ...], max_rows: int):
    X = stack_matrices([pipeline_matrix(rep, split) for split in splits])
    yy = np.concatenate([y[split] for split in splits])
    if max_rows and X.shape[0] > max_rows:
        keep, _ = K.stratified_subsample_indices(yy, max_rows, C.SEED)
        print(f"  row cap active: {len(keep):,} / {X.shape[0]:,} fitting rows (--max-rows)")
        X, yy = X[keep], yy[keep]
    return X, yy


def fit_model(estimator, rep: str, y: dict, tag: str, fit_splits: tuple[str, ...]):
    X_fit, y_fit = fitting_data(rep, y, fit_splits, C.ENSEMBLE_MAX_ROWS)
    started = time.perf_counter()
    estimator.fit(X_fit, y_fit)
    print(f"  fitted on {len(y_fit):,} rows in {time.perf_counter() - started:.0f}s")
    del X_fit, y_fit
    gc.collect()
    return estimator


def score_model(estimator, rep: str, y: dict, tag: str, split: str, persist: bool) -> dict:
    X = pipeline_matrix(rep, split)
    y_pred = estimator.predict(X)
    y_score = K.aligned_scores(estimator, X)
    metrics = K.classification_metrics(y[split], y_pred, y_score)
    metrics.update({"n": int(len(y_pred)), "tag": tag, "split": split})
    print(
        f"  {split:5s} {tag:24s} macro_f1={metrics['macro_f1']:.4f} "
        f"mcc={metrics['mcc']:.4f} acc={metrics['accuracy']:.4f}"
        + (f" auprc={metrics['auprc_macro']:.4f}" if "auprc_macro" in metrics else "")
    )
    K.save_predictions(tag, split, y_pred, y_score)
    if persist:
        joblib.dump(estimator, C.MODELS_DIR / f"{C.safe_name(tag)}.joblib", compress=3)
        print(f"  model -> {C.MODELS_DIR.name}/{C.safe_name(tag)}.joblib")
    del X
    gc.collect()
    return metrics


def fit_score(estimator, rep: str, y: dict, tag: str, fit_splits: tuple[str, ...], score_split: str, persist: bool) -> dict:
    """Fit on ``fit_splits`` and score ``score_split``, saving predictions."""
    fit_model(estimator, rep, y, tag, fit_splits)
    return score_model(estimator, rep, y, tag, score_split, persist and score_split == "test")


def run_estimator(make, rep: str, y: dict, tag: str, refit: bool) -> dict:
    """Train -> validation, then (by default) refit on train+val -> test.

    With ``refit=False`` the single train-only fit is scored on both splits and
    ``fit_scope`` records it, so a cheaper run is never mistaken for the protocol
    used everywhere else.
    """
    if refit:
        out = {
            "validation": fit_score(make(), rep, y, tag, ("train",), "val", persist=False),
            "test": fit_score(make(), rep, y, tag, ("train", "val"), "test", persist=True),
            "fit_scope": "train+val",
        }
        return out
    model = fit_model(make(), rep, y, tag, ("train",))
    out = {
        "validation": score_model(model, rep, y, tag, "val", persist=False),
        "test": score_model(model, rep, y, tag, "test", persist=True),
        "fit_scope": "train (no refit)",
    }
    return out


def comparison_table(families: list[str], results: dict, rep: str) -> pd.DataFrame:
    rows = []
    for family in families:
        entry = results[f"base_{family}"]
        rows.append(
            {
                "model": f"base_{family}",
                "kind": "single model",
                "representation": C.representation_label(rep),
                "val_macro_f1": entry["validation"]["macro_f1"],
                "test_macro_f1": entry["test"]["macro_f1"],
                "test_mcc": entry["test"]["mcc"],
                "test_auprc_macro": entry["test"].get("auprc_macro", float("nan")),
                "test_accuracy": entry["test"]["accuracy"],
            }
        )
    for name, tag in (
        ("hard_voting", "ensemble_hard"),
        ("soft_voting", "ensemble_soft"),
        ("stacking", "ensemble_stack"),
    ):
        if name not in results:
            continue
        entry = results[name]
        rows.append(
            {
                "model": tag,
                "kind": "ensemble",
                "representation": C.representation_label(rep),
                "val_macro_f1": entry["validation"]["macro_f1"],
                "test_macro_f1": entry["test"]["macro_f1"],
                "test_mcc": entry["test"]["mcc"],
                "test_auprc_macro": entry["test"].get("auprc_macro", float("nan")),
                "test_accuracy": entry["test"]["accuracy"],
            }
        )
    return pd.DataFrame(rows).sort_values("test_macro_f1", ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    K.banner("STAGE 06 - VOTING AND STACKING ENSEMBLES")

    if not C.SPLIT_PATHS["train"].exists():
        raise SystemExit("run `python src/01_eda_and_cleaning.py` first")
    for warning in K.check_artifact_consistency():
        print(f"warning: {warning}")
    if args.max_rows:
        C.ENSEMBLE_MAX_ROWS = int(args.max_rows)
    refit = not args.no_refit and bool(C.REFIT_ON_TRAIN_VAL)

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [f for f in families if f not in C.MODEL_KEYS]
    if unknown:
        raise SystemExit(f"unknown model families: {unknown}")
    frames = K.load_splits()
    y = K.labels(frames)

    if args.rep:
        if args.rep not in C.REPRESENTATIONS:
            raise SystemExit(f"unknown representation '{args.rep}'; choose from {C.REPRESENTATIONS}")
        rep = args.rep
        specs = {family: tuned_spec(family) for family in families}
        print(f"shared representation forced from --rep: {C.representation_label(rep)}")
    else:
        rep, specs = resolve_representation(families)
    print(f"\nshared representation for every base model: {C.representation_label(rep)}")
    for family in families:
        spec = specs.get(family)
        if spec and spec.get("cross_validation", {}).get("macro_f1_mean") is not None:
            print(
                f"  {family:5s} tuned on {C.representation_label(spec['representation'])} "
                f"(CV macro-F1 {spec['cross_validation']['macro_f1_mean']:.4f})"
            )

    results: dict[str, dict] = {}

    with K.stage_timer(f"base models on {C.representation_label(rep)}"):
        base = base_estimators(families, rep, specs, probability=C.ENSEMBLE_SVM_PROBABILITY)
        for family, _ in base:
            params = clean_params((specs.get(family) or {}).get("best_params"))
            results[f"base_{family}"] = run_estimator(
                lambda fam=family, par=params: estimator_for(
                    fam, rep, par, probability=C.ENSEMBLE_SVM_PROBABILITY
                ),
                rep,
                y,
                f"base_{family}",
                refit,
            )

    weights = weights_for(families)

    with K.stage_timer("hard voting (majority vote of the predicted classes)"):
        results["hard_voting"] = run_estimator(
            lambda: VotingClassifier(estimators=base, voting="hard", n_jobs=1, weights=weights),
            rep, y, "ensemble_hard", refit,
        )

    with K.stage_timer("soft voting (mean of the class probabilities)"):
        results["soft_voting"] = run_estimator(
            lambda: VotingClassifier(estimators=base, voting="soft", n_jobs=1, weights=weights),
            rep, y, "ensemble_soft", refit,
        )

    if not args.skip_stacking:
        with K.stage_timer(f"stacking (cv={C.STACK_CV_FOLDS}, logistic-regression meta-classifier)"):
            results["stacking"] = run_estimator(
                lambda: StackingClassifier(
                    estimators=base,
                    final_estimator=LogisticRegression(
                        class_weight="balanced",
                        max_iter=C.ENSEMBLE_META_MAX_ITER,
                        random_state=C.SEED,
                    ),
                    cv=StratifiedKFold(n_splits=C.STACK_CV_FOLDS, shuffle=True, random_state=C.SEED),
                    stack_method="predict_proba",
                    passthrough=False,
                    n_jobs=1,
                ),
                rep, y, "ensemble_stack", refit,
            )

    table = comparison_table(families, results, rep)
    print("\n" + K.markdown_table(table))
    table.to_csv(C.METRICS_DIR / "ensemble_comparison.csv", index=False)

    singles = table[table["kind"] == "single model"]["test_macro_f1"]
    ensembles = table[table["kind"] == "ensemble"]["test_macro_f1"]
    best_single = float(singles.max()) if len(singles) else float("nan")
    best_ensemble = float(ensembles.max()) if len(ensembles) else float("nan")
    print(
        f"\nensemble gain over the best single model on the same features: "
        f"{best_ensemble - best_single:+.4f} macro-F1"
    )

    payload = {
        "stage": "06_ensembles",
        "shared_representation": rep,
        "shared_representation_label": C.representation_label(rep),
        "families": families,
        "max_rows": C.ENSEMBLE_MAX_ROWS,
        "svm_probability": C.ENSEMBLE_SVM_PROBABILITY,
        "voting_weights": weights,
        "stack_cv_folds": C.STACK_CV_FOLDS,
        "base_parameters": {
            family: clean_params((specs.get(family) or {}).get("best_params")) for family in families
        },
        "results": results,
        "comparison": table.to_dict("records"),
        "ensemble_gain_macro_f1": best_ensemble - best_single,
        "fit_scope": "train+val" if refit else "train",
    }
    K.save_json(C.METRICS_DIR / "06_ensembles.json", payload)
    print(f"\nmetrics -> {K.relative_label(C.METRICS_DIR / '06_ensembles.json')}")
    print("stage 06 complete")


if __name__ == "__main__":
    main()