# Colab / Kaggle notebooks

Six notebooks that run the same pipeline as `../src`, one stage group per
notebook, so you can run them in separate Kaggle/Colab sessions and still
accumulate the results in a shared output directory.

| Notebook | Runs | Needs |
| --- | --- | --- |
| `01_data_and_features.ipynb` | stages 01 + 02 | nothing |
| `02_random_forest.ipynb` | stage 03 | 01 |
| `03_lightgbm.ipynb` | stage 04 | 01 |
| `04_svm.ipynb` | stage 05 | 01 |
| `05_ensembles.ipynb` | stage 06 | 02, 03, 04 |
| `06_evaluation.ipynb` | stage 07 | any model stage |

## How these notebooks avoid duplicating logic

Each notebook installs dependencies, copies the project's `src/` into the working
directory, and then executes the stage with `runpy`:

```python
run_stage("03_model_rf.py", "--skip-imbalance")
```

The executed code is always the current code in `src/` — there is no second copy
to drift out of sync. Regenerate the notebooks after editing the build script:

```bash
python build_notebooks.py
```

## Getting the code into the runtime

The setup cell resolves the project in this order:

1. **already extracted** next to the notebook,
2. **an uploaded zip** — zip the project so the archive is named
   `enzyme_classification.zip` (top-level folder `enzyme_classification/`), then
   attach it as a Kaggle dataset or upload it to Colab,
3. **a git clone** — set `ENZYME_REPO_URL` before running the cell.

```bash
# from the directory containing enzyme_classification/
zip -r enzyme_classification.zip enzyme_classification \
    -x '*/__pycache__/*' '*/data/*' '*/artifacts/*'
```

## Sharing results between sessions

Stages 03–07 read what the earlier stages wrote, so a later notebook needs those
artifacts. Two options:

- **Re-run notebook 01 in the same session** (cheapest, but 01+02 is the slow part),
- **carry `data/processed/` and `artifacts/` forward** as a saved version or
  dataset, and unzip it into the working directory at the top of the setup cell.

At minimum stage 06 needs the three metrics files written by stages 03–05, and
stage 07 needs the prediction `.npy` files.

## Notes

- **SVM is slow.** It needs a dense feature matrix; expect long runtimes. Use
  `--tune-max-rows` to subsample for the grid search.
- **Probability handling.** On sklearn ≥ 1.9 the SVM's probabilities come from
  `CalibratedClassifierCV` (`SVC(probability=True)` is removed in 1.11).
  `common.py` picks the right route for the installed version.
- **Cheap smoke run.** `python src/01_eda_and_cleaning.py --cap 20000` then
  `--tune-max-rows 8000` exercises the whole pipeline end to end quickly. The
  cap is recorded in `meta.json`, so a reduced run is never mistaken for a full
  one.
- **Check `select_k`.** If `meta.json` disagrees with the cached `tripeptide`
  matrix, stages 03–06 print a warning — re-run notebook 01 to resynchronise.