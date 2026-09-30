#!/usr/bin/env python3
"""Generate the Colab/Kaggle notebooks from the canonical ``src/`` pipeline.

Run: ``python colab_notebooks/build_notebooks.py``

Why a generator instead of hand-written notebooks: the scripts in ``src/`` are
the source of truth, and a notebook that quietly embeds a *copy* of that logic
will drift the moment a stage is fixed. Every notebook here therefore

1. installs dependencies,
2. copies ``src/`` into the runtime's working directory, and
3. runs the stage(s) with ``runpy``.

so the executed code is always the current code. Only the surrounding glue
(arguments, ordering notes, result rendering) lives in the notebook.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The project may sit next to this directory (``Task-01/enzyme_classification``)
# or inside it, depending on how the package was copied.
CANDIDATES = [HERE.parent / "enzyme_classification", HERE.parent, HERE]
PROJECT = next(
    (c for c in CANDIDATES if (c / "src" / "config.py").exists()),
    CANDIDATES[0],
)
SRC = PROJECT / "src"

SETUP = r'''# %% [markdown]
# Setup - dependencies and the project source
#
# This notebook runs the real pipeline from `src/`; it does not reimplement it.
# The cell below copies the project into the working directory so the stage can
# be imported and run unchanged.

import os, sys, subprocess, shutil, textwrap, zipfile, urllib.request
from pathlib import Path

# Kaggle ships most of these already. `datasets` is the one that is often missing.
QUIET = "-q"
subprocess.run([sys.executable, "-m", "pip", "install", QUIET,
                "datasets", "lightgbm", "imbalanced-learn", "joblib", "pyarrow"],
               check=False)

HERE = Path.cwd()          # a notebook has no __file__, so the cwd is the anchor
WORK = HERE / "enzyme_classification"
REPO_URL = os.environ.get("ENZYME_REPO_URL", "").strip()

def fetch_project(dest: Path) -> Path:
    """Materialise the project at `dest`.

    Resolution order:
      1. an already-extracted copy next to the notebook
      2. a zip uploaded as a Kaggle dataset / Colab file in /kaggle/input or /content
      3. a git clone from ENZYME_REPO_URL
    """
    if (dest / "src" / "config.py").exists():
        print(f"project already present at {dest}")
        return dest
    if dest.exists():
        shutil.rmtree(dest)

    roots = [p for p in (Path("/kaggle/input"), Path("/content"), HERE) if p.exists()]
    for root in roots:
        for zip_path in root.rglob("enzyme_classification*.zip"):
            print(f"extracting {zip_path}")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(zip_path.parent)
            inner = zip_path.parent / "enzyme_classification"
            if (inner / "src" / "config.py").exists():
                if inner.resolve() != dest.resolve():
                    shutil.move(str(inner), str(dest))
                return dest

    if REPO_URL:
        print(f"cloning {REPO_URL}")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(dest)], check=True)
        return dest

    raise SystemExit(
        "Could not find the project. Either\n"
        "  * zip the enzyme_classification folder (src/, config.py) and upload it as a "
        "Kaggle dataset / Colab file, or\n"
        "  * set ENZYME_REPO_URL to a git URL before running this cell."
    )

PROJECT = fetch_project(WORK)
sys.path.insert(0, str(PROJECT / "src"))

import runpy
def run_stage(script: str, *argv: str):
    """Execute src/<script> with the given argv, as if typed on the command line."""
    argv_backup = sys.argv
    sys.argv = [str(PROJECT / "src" / script), *argv]
    try:
        return runpy.run_path(str(PROJECT / "src" / script), run_name="__main__")
    finally:
        sys.argv = argv_backup

print("project:", PROJECT)
print("stages :", sorted(p.name for p in (PROJECT / "src").glob("*.py") if p.name[0].isdigit()))'''

STAGE_CELL = 'run_stage("{script}"{args})'

SHOW_METRICS = r'''# %% [markdown]
# Results
# Any stage that writes `artifacts/metrics/*.json` is summarised here.

import json, glob, os

METRICS = PROJECT / "artifacts" / "metrics"
if not METRICS.exists():
    print("no metrics yet")
else:
    rows = []
    for path in sorted(METRICS.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        test = payload.get("test") or (payload.get("results") or {}).get("test") or {}
        if isinstance(test, dict) and "macro_f1" in test:
            rows.append({
                "run": path.stem,
                "representation": payload.get("representation") or payload.get("shared_representation", "-"),
                "val macro-F1": round(payload.get("validation", payload.get("results", {}).get("validation", {})).get("macro_f1", float("nan")), 4),
                "test macro-F1": round(test["macro_f1"], 4),
                "test MCC": round(test.get("mcc", float("nan")), 4),
                "test acc": round(test.get("accuracy", float("nan")), 4),
            })
    import pandas as pd
    df = pd.DataFrame(rows)
    display(df if len(df) else "no scored runs found")

figs = sorted((PROJECT / "reports" / "figures").glob("*.png"))
if figs:
    print("\nfigures written:")
    for f in figs:
        print("  ", f.relative_to(PROJECT))'''


def nb(cells: list[tuple[str, str]]) -> dict:
    """Build an nbformat 4.5 notebook from (cell_type, source) pairs."""
    out = []
    for kind, src in cells:
        src = src.strip("\n") + "\n"
        if kind == "markdown":
            out.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})
        else:
            out.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": src.splitlines(keepends=True),
                }
            )
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def stage_notebook(number: int, title: str, stage_no: str, blurb: str, argv: list[str]) -> dict:
    """One notebook that runs a single stage, then prints its metrics."""
    return nb(
        [
            ("markdown", f"# {number}. {title}\n\n{blurb}"),
            ("code", SETUP),
            ("markdown", f"## Run stage {stage_no}\n\n`python src/{stage_no}`"),
            ("code", STAGE_CELL.format(script=f"{stage_no}.py", args="".join(f", {a!r}" for a in argv))),
            ("markdown", "## Results"),
            ("code", SHOW_METRICS),
        ]
    )


def combined_data_notebook() -> dict:
    """Stages 01 and 02 in one notebook: the shared setup only pays off once."""
    return nb(
        [
            (
                "markdown",
                "# 1. Data, EDA and feature engineering\n\n"
                "Runs stage 01 (download, clean, stratified 70/15/15 split, EDA figures) and\n"
                "stage 02 (AAC / dipeptide / tripeptide / SVD / concat matrices, plus the\n"
                "`SelectKBest` sweep that picks `k`).\n\n"
                "Both stages must run before any model stage. They are combined here because\n"
                "the setup cell - by far the slowest part of a cold start - is shared.",
            ),
            ("code", SETUP),
            (
                "markdown",
                "## Stage 01 - dataset, cleaning and the canonical split\n\n"
                "`python src/01_eda_and_cleaning.py [--cap N]`\n\n"
                "The 70/15/15 split written here is the single source of truth for every\n"
                "later stage. Pass `--cap` for a cheaper run; the cap is recorded in\n"
                "`data/processed/meta.json` so a reduced run is never mistaken for a full one.",
            ),
            ("code", 'run_stage("01_eda_and_cleaning.py")'),
            (
                "markdown",
                "## Stage 02 - feature matrices\n\n"
                "`python src/02_feature_extract.py [--skip-sweep]`\n\n"
                "Every fitted transform (vectorisers, `SelectKBest`, `TruncatedSVD`) is fitted on\n"
                "the **training split only** and then applied with `transform` to validation and\n"
                "test. Drop `--skip-sweep` to reuse the recorded `k` instead of re-running the\n"
                "cross-validated budget sweep.",
            ),
            ("code", 'run_stage("02_feature_extract.py")'),
            (
                "markdown",
                "## Inspect the split and the chosen `k`\n\n"
                "Worth checking on every fresh run: the class distribution, the split sizes and\n"
                "the `k` chosen by the sweep.",
            ),
            (
                "code",
                r'''import json, pandas as pd

meta = json.loads((PROJECT / "data" / "processed" / "meta.json").read_text())
print("split sizes   :", meta.get("split_sizes"))
print("class counts  :", meta.get("class_counts"))
print("imbalance     :", round(meta.get("imbalance_ratio", float("nan")), 2))
print("select_k      :", meta.get("select_k"), "(chosen by the stage-02 sweep)")
print("feature dims  :", meta.get("feature_dims") or meta.get("feature_names"))

sweep = PROJECT / "data" / "processed" / "select_k_sweep.csv"
if sweep.exists():
    display(pd.read_csv(sweep))''',
            ),
            ("markdown", "## Results"),
            ("code", SHOW_METRICS),
        ]
    )


NOTEBOOKS = {
    "01_data_and_features.ipynb": combined_data_notebook,
    "02_random_forest.ipynb": lambda: stage_notebook(
        2,
        "Random Forest",
        "03_model_rf",
        "Representation screening -> 5-fold grid search on the training split -> imbalance\n"
        "study -> refit on train+val -> one test evaluation.\n\n"
        "Every stage 03-05 notebook is independent; each one is a full experiment on its own\n"
        "model family, and all of them write into the same `artifacts/` tree.",
        ["--skip-imbalance"],
    ),
    "03_lightgbm.ipynb": lambda: stage_notebook(
        3,
        "LightGBM",
        "04_model_lgbm",
        "Same protocol as the Random Forest stage, on gradient-boosted trees. LightGBM is the\n"
        "strongest single model here on imbalanced tabular features, so its representation\n"
        "choice also drives what stage 06 evaluates.",
        ["--skip-imbalance"],
    ),
    "04_svm.ipynb": lambda: stage_notebook(
        4,
        "Support Vector Machine",
        "05_model_svm",
        "RBF-kernel SVM. It is by far the slowest of the three - the feature matrix must be\n"
        "dense, so expect long runtimes and a large `cache_size`.\n\n"
        "SVM probabilities come from `CalibratedClassifierCV` on sklearn >= 1.9 (the old\n"
        "`SVC(probability=True)` is deprecated and removed in 1.11); `common.py` picks the\n"
        "right route automatically.",
        ["--skip-imbalance"],
    ),
    "05_ensembles.ipynb": lambda: stage_notebook(
        5,
        "Voting and stacking",
        "06_ensembles",
        "Hard voting, soft voting and stacking over the three tuned models.\n\n"
        "**Requires stages 03-05 to have run** - the tuned hyper-parameters are read from their\n"
        "metrics files. All three base models are re-fitted on a single shared representation,\n"
        "because you cannot vote across different feature spaces. The re-fitted baselines are\n"
        "saved alongside the ensembles so the comparison isolates the combination rule rather\n"
        "than the feature set.\n\n"
        "Stacking uses a `LogisticRegression(class_weight='balanced')` meta-classifier over\n"
        "internal out-of-fold probabilities. No neural meta-model.",
        [],
    ),
    "06_evaluation.ipynb": lambda: stage_notebook(
        6,
        "Evaluation, plots and error analysis",
        "07_evaluation_plots",
        "Reads saved **predictions only** - never the raw models - so the whole report can be\n"
        "regenerated without retraining.\n\n"
        "Produces confusion matrices, per-class precision/recall/F1, one-vs-rest ROC curves,\n"
        "feature importances (tree models) and the top confused class pairs.\n\n"
        "Add `--split val` to analyse the validation split instead of the test split.",
        [],
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the notebook package")
    parser.add_argument("--output", default=str(HERE), help="directory to write the notebooks into")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if not SRC.exists():
        raise SystemExit(f"cannot find {SRC}")

    for name, builder in NOTEBOOKS.items():
        path = out / name
        path.write_text(json.dumps(builder(), indent=1) + "\n")
        print(f"wrote {path.name}")

    print(f"\n{len(NOTEBOOKS)} notebooks in {out}")


if __name__ == "__main__":
    main()