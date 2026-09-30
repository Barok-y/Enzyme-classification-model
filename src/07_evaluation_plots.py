"""Stage 07 - confusion matrices, per-class metrics, ROC curves, feature importance and error analysis.

Run: ``python src/07_evaluation_plots.py [--model ensemble_stack] [--model rf_tripeptide]``

Reads predictions only - the ``.npy`` files written by stages 03-06 plus their
``__score.npy`` probabilities - so the report can be regenerated without
retraining. With no ``--model`` flags every model that has predictions for the
chosen split is analysed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
import common as K

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final evaluation and error analysis")
    parser.add_argument("--model", action="append", help="model tag to evaluate (can be given multiple times)")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def model_list(explicit: list[str] | None, split: str) -> list[str]:
    """Models to analyse: the requested ones, else everything with predictions."""
    if explicit:
        return list(dict.fromkeys(explicit))
    tags = {
        path.name.split(f"__{split}")[0]
        for path in C.PREDICTIONS_DIR.glob(f"*__{split}__meta.json")
    }
    return sorted(tags)


def confusion_matrix_normalized(y_true, y_pred, tag: str, split: str) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(C.CLASSES), normalize="true")
    df_cm = pd.DataFrame(cm, index=[f"EC {c}" for c in C.CLASSES], columns=[f"Pred {c}" for c in C.CLASSES])
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(
        df_cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar_kws={"label": "Fraction of true class"},
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{tag} - confusion matrix (normalized by true class) - {split}")
    fig.tight_layout()
    path = C.FIGURES / f"cm_{C.safe_name(tag)}_{split}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {K.relative_label(path)}")
    return df_cm


def per_class_metrics(y_true, y_pred, tag: str, split: str) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(C.CLASSES), zero_division=0
    )
    rows = []
    for idx, cls in enumerate(C.CLASSES):
        mask = np.asarray(y_true) == cls
        correct = int((np.asarray(y_pred)[mask] == cls).sum())
        rows.append(
            {
                "Model": tag,
                "Split": split,
                "EC Class": cls,
                "Family": C.EC_NAMES[cls],
                "Precision": float(precision[idx]),
                "Recall": float(recall[idx]),
                "F1": float(f1[idx]),
                "Support": int(support[idx]),
                "Accuracy (within class)": (correct / support[idx]) if support[idx] else float("nan"),
            }
        )
    table = pd.DataFrame(rows)
    path = C.METRICS_DIR / f"per_class_{C.safe_name(tag)}_{split}.csv"
    table.to_csv(path, index=False)
    print(f"table  -> {K.relative_label(path)}")
    print("\n" + K.markdown_table(table))
    return table


def plot_roc(y_true, y_score, tag: str, split: str) -> dict:
    """One-vs-rest ROC curves for the model.

    If ``y_score`` is missing (e.g. an SVM was stored without probabilities),
    this function returns immediately with an empty dict.
    """
    if y_score is None:
        print(f"skipping ROC for {tag} ({split}): probabilities not available")
        return {}
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score, dtype=float)
    if y_score_arr.shape[1] != len(C.CLASSES):
        print(f"skipping ROC for {tag} ({split}): unexpected score shape {y_score_arr.shape}")
        return {}

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    aucs = {}
    for cls in C.CLASSES:
        y_bin = (y_true_arr == cls).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_score_arr[:, C.CLASSES.index(cls)])
        roc_auc = auc(fpr, tpr)
        aucs[cls] = float(roc_auc)
        ax.plot(fpr, tpr, lw=2, label=f"EC {cls} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{tag} - one-vs-rest ROC curves - {split}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = C.FIGURES / f"roc_{C.safe_name(tag)}_{split}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {K.relative_label(path)}")
    return aucs


def _column_names(pipeline, tag: str, n_expected: int) -> list[str]:
    """Best available column names for a fitted pipeline's final estimator."""
    selector = pipeline.named_steps.get("select") if hasattr(pipeline, "named_steps") else None
    if selector is not None and hasattr(selector, "get_feature_names_out"):
        try:
            names = [str(n) for n in selector.get_feature_names_out()]
            if len(names) == n_expected:
                return names
        except Exception:  # noqa: BLE001 - a naming hint must never break the plot
            pass

    rep = tag.rsplit("_", 1)[-1]
    try:
        names = K.feature_names(rep)
        if len(names) == n_expected:
            return names
    except Exception:  # noqa: BLE001 - fall through to positional labels
        pass
    return [f"f_{i}" for i in range(n_expected)]


def plot_feature_importance(tag: str) -> pd.DataFrame | None:
    """Plot the top-25 features for tree-based models.

    The pipeline was saved as a ``joblib`` file with a ``Pipeline`` object in
    ``artifacts/models``. Tree importances are taken from the ``clf`` step (and
    after the ``select`` step for ``tripeptide``, so the names already match).
    """
    model_path = C.MODELS_DIR / f"{C.safe_name(tag)}.joblib"
    if not model_path.exists():
        print(f"skipping feature importance for {tag}: model not found at {model_path.name}")
        return None

    import joblib

    pipeline = joblib.load(model_path)
    clf = pipeline.named_steps.get("clf")
    if clf is None or not hasattr(clf, "feature_importances_"):
        print(f"skipping feature importance for {tag}: {type(clf)} has no feature_importances_")
        return None

    importances = np.asarray(clf.feature_importances_).ravel()
    if importances.ndim != 1:
        importances = importances[0]

    feature_names = _column_names(pipeline, tag, len(importances))
    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)
    top = df_imp.head(25)

    fig, ax = plt.subplots(figsize=(9, 10))
    sns.barplot(data=top, x="importance", y="feature", ax=ax, color="#1f77b4")
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Feature")
    ax.set_title(f"{tag} - top 25 feature importances")
    fig.tight_layout()
    path = C.FIGURES / f"fi_{C.safe_name(tag)}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {K.relative_label(path)}")
    return df_imp


def error_analysis(y_true, y_pred, tag: str, split: str) -> pd.DataFrame:
    df = pd.DataFrame({"true": np.asarray(y_true), "pred": np.asarray(y_pred)})
    errors = df[df["true"] != df["pred"]].copy()
    confusion_pairs = (
        errors.groupby(["true", "pred"]).size().rename("count").reset_index().sort_values("count", ascending=False)
    )
    confusion_pairs["share_errors"] = confusion_pairs["count"] / max(len(errors), 1)
    path = C.METRICS_DIR / f"errors_pairs_{C.safe_name(tag)}_{split}.csv"
    confusion_pairs.to_csv(path, index=False)
    print(f"table  -> {K.relative_label(path)}")
    print("\nTop confused class pairs (true -> predicted, by count):")
    print(confusion_pairs.head(15).to_string(index=False))
    return confusion_pairs


def summarize_model(tag: str, split: str, y_true) -> dict:
    y_pred, y_score = K.load_predictions(tag, split)
    if len(y_pred) == 0:
        return {}
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"{tag}/{split}: {len(y_pred)} predictions but {len(y_true)} labels in the split"
        )

    cm = confusion_matrix_normalized(y_true, y_pred, tag, split)
    per = per_class_metrics(y_true, y_pred, tag, split)
    roc_auc = plot_roc(y_true, y_score, tag, split)
    errors = error_analysis(y_true, y_pred, tag, split)

    metrics_all = K.classification_metrics(y_true, y_pred, y_score)
    metrics_all["tag"] = tag
    metrics_all["split"] = split
    if roc_auc:
        metrics_all["roc_auc_macro"] = float(np.mean(list(roc_auc.values())))
        metrics_all["roc_auc_per_class"] = {str(k): float(v) for k, v in roc_auc.items()}

    K.save_model_metrics(f"{tag}_{split}", metrics_all)
    plot_feature_importance(tag)

    summary = {
        "tag": tag,
        "split": split,
        "n": int(len(y_true)),
        "metrics": metrics_all,
        "roc_auc_per_class": roc_auc,
        "most_confused": errors.head(20).to_dict("records"),
    }
    K.save_json(C.METRICS_DIR / f"analysis_{C.safe_name(tag)}_{split}.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    K.banner("STAGE 07 - EVALUATION, ANALYSIS AND PLOTS")
    models = model_list(args.model, args.split)
    if not models:
        raise SystemExit("no model predictions found; run stages 03-06 first")

    frames = K.load_splits()
    y_by_split = K.labels(frames)

    summaries = []
    failures = []
    for tag in models:
        try:
            summary = summarize_model(tag, args.split, y_by_split[args.split])
        except FileNotFoundError as exc:
            print(f"skipping {tag}: {exc}")
            failures.append((tag, f"missing predictions ({exc})"))
            continue
        except Exception as exc:  # noqa: BLE001 - one bad tag must not kill the report
            print(f"failed on {tag}: {type(exc).__name__}: {exc}")
            failures.append((tag, f"{type(exc).__name__}: {exc}"))
            continue
        if summary:
            summaries.append(summary)

    if failures:
        print("\n=== MODELS THAT COULD NOT BE ANALYSED ===")
        print(K.markdown_table(pd.DataFrame(failures, columns=["model", "reason"])))

    if not summaries:
        print("no summaries produced")
        return

    overview = pd.DataFrame(
        [
            {
                "model": s["tag"],
                "split": s["split"],
                "n": s["n"],
                "accuracy": s["metrics"]["accuracy"],
                "macro_f1": s["metrics"]["macro_f1"],
                "mcc": s["metrics"]["mcc"],
                "auprc_macro": s["metrics"].get("auprc_macro", float("nan")),
                "roc_auc_macro": s["metrics"].get("roc_auc_macro", float("nan")),
            }
            for s in summaries
            if s
        ]
    ).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    print("\n=== OVERALL COMPARISON (on the selected split) ===")
    print(K.markdown_table(overview))
    overview.to_csv(C.METRICS_DIR / f"analysis_overview_{args.split}.csv", index=False)

    hardest = []
    for s in summaries:
        per = pd.read_csv(C.METRICS_DIR / f"per_class_{C.safe_name(s['tag'])}_{s['split']}.csv")
        hardest.extend(per[["Model", "EC Class", "Family", "F1", "Recall", "Support"]].to_dict("records"))
    if hardest:
        h = pd.DataFrame(hardest)
        worst = h.sort_values("F1").head(20).reset_index(drop=True)
        print("\n=== HARDEST-TO-PREDICT CLASSES ACROSS ALL MODELS ===")
        print(K.markdown_table(worst))
        worst.to_csv(C.METRICS_DIR / f"analysis_hardest_{args.split}.csv", index=False)

    print("\nstage 07 complete")
    print(f"figures -> {K.relative_label(C.FIGURES)}")
    print(f"tables  -> {K.relative_label(C.METRICS_DIR)}")


if __name__ == "__main__":
    main()