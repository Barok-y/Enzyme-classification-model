# Enzyme Function Classification

Predict the **top-level EC number** (`1`–`6`) of a protein from its amino-acid
sequence. Six imbalanced classes, 45k Swiss-Prot sequences, five feature
representations, three model families and three ensemble rules — scored under one
fixed, leakage-safe protocol.

```
Oxidoreductases · Transferases · Hydrolases · Lyases · Isomerases · Ligases
```

## Pipeline

| Stage | Script | What it does |
| --- | --- | --- |
| 01 | `src/01_eda_and_cleaning.py` | Download `DanielHesslow/SwissProt-EC`, clean sequences, stratified 70/15/15 split, EDA figures |
| 02 | `src/02_feature_extract.py` | Build AAC, dipeptide, tripeptide, SVD and concat matrices; pick `SelectKBest` `k` |
| 03 | `src/03_model_rf.py` | Random Forest: screen representations, grid search, fit, evaluate |
| 04 | `src/04_model_lgbm.py` | LightGBM, same protocol |
| 05 | `src/05_model_svm.py` | SVM (RBF), same protocol |
| 06 | `src/06_ensembles.py` | Hard voting, soft voting, stacking over the three tuned models |
| 07 | `src/07_evaluation_plots.py` | Confusion matrices, per-class metrics, OvR ROC, feature importance, error analysis |

`src/config.py` holds every path and tunable; `src/common.py` is the shared
library (pipelines, caching, metrics, IO); `src/model_runner.py` is the driver
that stages 03–05 share.

## Setup

```bash
pip install numpy pandas pyarrow scikit-learn lightgbm joblib \
            matplotlib seaborn imbalanced-learn datasets
```

Python 3.10+. On Kaggle/Colab, `scikit-learn ≥ 1.9` is supported: probabilities
for the SVM come from `CalibratedClassifierCV` instead of the deprecated
`SVC(probability=True)`, handled transparently in `common.calibrated_svc`.

## Run

Stages must run **in order** — each consumes the previous one's artifacts.

```bash
python src/01_eda_and_cleaning.py
python src/02_feature_extract.py
python src/03_model_rf.py
python src/04_model_lgbm.py
python src/05_model_svm.py
python src/06_ensembles.py
python src/07_evaluation_plots.py
```

Useful flags:

```bash
python src/01_eda_and_cleaning.py --cap 20000     # cheaper run, recorded in meta.json
python src/02_feature_extract.py --skip-sweep     # reuse the recorded k
python src/03_model_rf.py --reps tripeptide,aac  # limit representation screening
python src/03_model_rf.py --tune-max-rows 12000  # subsample for the grid search only
python src/06_ensembles.py --rep tripeptide      # force the shared representation
python src/06_ensembles.py --max-rows 20000      # cap ensemble fitting rows
python src/07_evaluation_plots.py --split val     # analyse the validation split
```

## Protocol

The rules that keep the numbers honest:

- **Splits are fixed once** in stage 01 and reused everywhere.
- **Every transform is fitted on the training split only.** `tripeptide`
  feature selection lives *inside* the pipeline, so it is refitted per CV fold.
- **Selection never sees the test split.** The grid search and the imbalance
  study both run on the training split, with the validation split used only to
  report.
- **The test split is scored once, at the end**, after refitting on train+val.
- **Ensembles share one representation.** You cannot vote across different
  feature spaces, so stage 06 re-fits all three base models on a single
  representation and saves those fitted baselines too, so the comparison
  isolates the combination rule.
- **Stacking uses a logistic-regression meta-classifier** over internal
  out-of-fold probabilities (`LogisticRegression(class_weight="balanced")`).
  No neural meta-model.
- **`SelectKBest` and `TruncatedSVD` are fitted on train only**; SVD is
  unsupervised so it is fitted once, outside the CV loop.
- Every score table records the `fit_scope` (`train` or `train+val`) and the
  row cap, so a reduced run is never mistaken for a full one.

## Representations

| Key | Description | Raw columns |
| --- | --- | --- |
| `aac` | Amino-acid composition | 20 |
| `dipeptide` | Dipeptide frequencies | 400 |
| `tripeptide` | Tripeptide frequencies + `SelectKBest` | 8000 → swept `k` |
| `svd` | `TruncatedSVD` embedding of the sparse profile | 128 |
| `concat` | AAC + dipeptide + length + SVD | 549 |

`k` is chosen by stage 02 with a cross-validated sweep, not hard-coded.

## Outputs

```
data/processed/          splits, cached feature matrices, feature_names.json, meta.json
artifacts/models/        fitted pipelines (.joblib)
artifacts/predictions/   per-model val/test predictions and score matrices (.npy)
artifacts/metrics/       per-stage metrics (.json) and comparison tables (.csv)
artifacts/feature_pipeline/  fitted vectorisers and selectors
reports/figures/         EDA, confusion matrices, ROC curves, feature importances
```

## Configuration

`src/config.py` reads environment variables, so nothing needs editing to change
a run:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ENZYME_CAP` | — | Stratified subsample size (low-RAM run) |
| `ENZYME_SEED` | `42` | Global seed |
| `ENZYME_N_JOBS` | `-1` | Parallelism for RF/LightGBM |
| `ENZYME_TUNE_MAX_ROWS` | — | Subsample used for grid search only |
| `ENZYME_SELECT_K` | `500` | `SelectKBest` `k` fallback (stage 02 overrides it from the sweep) |
| `ENZYME_SVM_PROBABILITY` | `0` (off) | Calibrate the SVM in stage 05 |
| `ENZYME_SMOTE` | `0` (off) | Also try SMOTE in the imbalance study |
| `ENZYME_ENSEMBLE_REP` | auto | Shared representation for stage 06 |
| `ENZYME_ENSEMBLE_MAX_ROWS` | — | Cap ensemble fitting rows |
| `ENZYME_ENSEMBLE_WEIGHTS` | — | Voting weights, comma separated |
| `ENZYME_ENSEMBLE_SVM_PROB` | `1` (on) | Calibrate the SVM inside the ensembles |
| `ENZYME_REFIT_ON_TRAIN_VAL` | `1` (on) | Refit on train+val before the test split |

## Notebooks

`colab_notebooks/` contains the same pipeline as separate Kaggle/Colab
notebooks — see its `README.md`.