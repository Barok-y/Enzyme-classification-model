"""Stage 02 - turn protein sequences into numeric representations.

Run: ``python src/02_feature_extract.py``

Produces one cached matrix per representation under ``data/processed``:

===========  ======  ===================================================
representation dims    content
===========  ======  ===================================================
aac                20  amino-acid composition (relative frequencies)
dipeptide         400  dipeptide composition (relative frequencies)
tripeptide      8000->k tripeptide composition + ``SelectKBest``
svd               128  TruncatedSVD embedding of the 3-mer frequencies
concat            549  AAC + dipeptide + length + SVD (extra ablation)
===========  ======  ===================================================

Every fitted object (vectorisers, selector, SVD) is fitted on the *training*
split only and then applied with ``transform`` to validation/test, which is what
the leakage rule in the task requires.  The ``k`` of ``SelectKBest`` is chosen by
an explicit experiment at the end of this stage rather than by guesswork.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
import common as K

from scipy import sparse
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cached feature matrices")
    parser.add_argument("--skip-sweep", action="store_true", help="skip the SelectKBest justification sweep")
    parser.add_argument("--sweep-folds", type=int, default=C.SCREEN_CV_FOLDS)
    return parser.parse_args()


def lengths_per_split(frames: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    return {split: frame["seq"].str.len().to_numpy(dtype=np.int32) for split, frame in frames.items()}


def verify_dipeptide(vectoriser, dipeptide_matrix, seqs: list[str], n_check: int = 25) -> None:
    """Cross-check the vectorised dipeptide frequencies against a naive implementation."""
    tokens = [f"{a}{b}" for a in C.AMINO_ACIDS for b in C.AMINO_ACIDS]

    def naive(seq: str) -> np.ndarray:
        pairs = [seq[i : i + 2] for i in range(len(seq) - 1)]
        counts = pd.Series(pairs).value_counts()
        return np.asarray([counts.get(token, 0) / max(len(pairs), 1) for token in tokens])

    order = {name: i for i, name in enumerate(vectoriser.get_feature_names_out())}
    columns = [order[token] for token in tokens]
    checked = np.asarray(dipeptide_matrix[:n_check].toarray())[:, columns]
    reference = np.asarray([naive(s) for s in seqs])
    print(
        f"dipeptide self-check on {n_check} sequences: "
        f"max abs deviation = {np.abs(reference - checked).max():.2e}"
    )


def build_composition(frames, lengths):
    with K.stage_timer("dipeptide composition"):
        vec2 = K.vectoriser(2)
        counts = vec2.fit_transform(frames["train"]["seq"])
        print(f"dipeptide vocabulary: {len(vec2.vocabulary_)} columns (expected 400)")
        train = K.normalise_rows(counts, lengths["train"] - 1)
        val = K.normalise_rows(vec2.transform(frames["val"]["seq"]), lengths["val"] - 1)
        test = K.normalise_rows(vec2.transform(frames["test"]["seq"]), lengths["test"] - 1)
        verify_dipeptide(vec2, train, list(frames["train"]["seq"].head(25)))

        joblib.dump(vec2, C.FEATURE_PIPELINE_DIR / "vectoriser_dipeptide.joblib")
        names = [f"DIP_{n}" for n in vec2.get_feature_names_out()]
        dense = {
            "train": train.toarray().astype(np.float32),
            "val": val.toarray().astype(np.float32),
            "test": test.toarray().astype(np.float32),
        }
        gc.collect()
        return dense, names


def build_aac(frames):
    with K.stage_timer("amino-acid composition (20)"):
        def comp(seqs: pd.Series) -> np.ndarray:
            counts = np.column_stack([seqs.str.count(aa).to_numpy(dtype=np.float32) for aa in C.AMINO_ACIDS])
            return counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)

        mats = {split: comp(frame["seq"]).astype(np.float32) for split, frame in frames.items()}
        print(f"AAC matrix: {mats['train'].shape}  row sums={mats['train'].sum(axis=1)[:3]}")
        return mats, [f"AA_{aa}" for aa in C.AMINO_ACIDS]


def build_tripeptide_counts(frames, lengths):
    with K.stage_timer("tripeptide composition (3-mer counts)"):
        vec3 = K.vectoriser(3, max_features=C.KMER_MAX_FEATURES)
        train_counts = vec3.fit_transform(frames["train"]["seq"])
        val_counts = vec3.transform(frames["val"]["seq"])
        test_counts = vec3.transform(frames["test"]["seq"])
        print(f"3-mer vocabulary: {train_counts.shape[1]} columns (expected 8000)")
        print(f"train 3-mer matrix: {K.report_memory(train_counts)}")

        train = K.normalise_rows(train_counts, lengths["train"] - 2)
        val = K.normalise_rows(val_counts, lengths["val"] - 2)
        test = K.normalise_rows(test_counts, lengths["test"] - 2)
        names = [f"TRI_{n}" for n in vec3.get_feature_names_out()]
        joblib.dump(vec3, C.FEATURE_PIPELINE_DIR / "vectoriser_tripeptide.joblib")
        gc.collect()
        return {"train": train, "val": val, "test": test}, names


def score_tripeptides(X, y, names: list[str]) -> np.ndarray:
    """Mutual information of every 3-mer with the target (independent of ``k``)."""
    with K.stage_timer("mutual information for all 8000 3-mers"):
        scores = mutual_info_classif(X, y, random_state=C.SEED, n_neighbors=3)
        frame = pd.DataFrame(
            {
                "tripeptide": [n.replace("TRI_", "") for n in names],
                "mutual_information": scores,
            }
        ).sort_values("mutual_information", ascending=False)
        frame.to_csv(C.DATA_PROCESSED / "tripeptide_mi_scores.csv", index=False)
        print(frame.head(10).to_string(index=False))
        return scores


def sweep_select_k(tripeptide_full, y, folds: int, names: list[str]) -> tuple[pd.DataFrame, int]:
    """Justify the SelectKBest budget experimentally instead of by assumption.

    ``SelectKBest(mutual_info_classif, k)`` keeps the ``k`` highest scoring
    columns, so the mutual information is computed once and re-used for every
    candidate ``k`` instead of being recomputed inside each fold.
    """
    Xs, ys = K.subsample_for_search(tripeptide_full["train"], y["train"], C.SCREEN_MAX_ROWS, C.SEED)
    print(f"sweep on {Xs.shape[0]:,} training rows x {Xs.shape[1]} 3-mer columns, {folds}-fold CV")
    scores = score_tripeptides(Xs, ys, names)
    order = np.argsort(-np.nan_to_num(scores))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=C.SEED)

    rows = []
    for k in C.K_SWEEP:
        if k >= Xs.shape[1]:
            Xk = Xs
            label = f"all ({Xs.shape[1]})"
        else:
            Xk = Xs[:, np.sort(order[:k])]
            label = str(k)
        pipeline = K.build_base_pipeline("tripeptide", "lgbm")
        pipeline.set_params(clf__n_estimators=C.SCREEN_N_ESTIMATORS)
        result = cross_val_score(pipeline, Xk, ys, cv=cv, scoring="f1_macro", n_jobs=1)
        rows.append({"k": k, "cv_macro_f1_mean": result.mean(), "cv_macro_f1_std": result.std()})
        print(f"k={label:>6s}  cv macro-F1 = {result.mean():.4f} +/- {result.std():.4f}")
    table = pd.DataFrame(rows)
    chosen = int(table.loc[table["cv_macro_f1_mean"].idxmax(), "k"])
    table["selected"] = table["k"] == chosen
    return table, chosen


def select_and_cache(tripeptide_full, y, names: list[str], k: int):
    with K.stage_timer(f"SelectKBest(mutual_info_classif, k={k})"):
        selector = K.SupervisedSelector(k=k)
        selector.fit(tripeptide_full["train"], y["train"])
        selected = {
            split: np.asarray(selector.transform(tripeptide_full[split]).todense(), dtype=np.float32)
            for split in ("train", "val", "test")
        }
        joblib.dump(selector, C.FEATURE_PIPELINE_DIR / "selector_tripeptide.joblib")
    keep = np.flatnonzero(selector.get_support())
    selected_names = [names[i] for i in keep]
    mi = selector.scores_[keep]
    print(
        f"selected {len(selected_names)} tripeptides; MI of selected features "
        f"{np.nanmin(mi):.4f} .. {np.nanmax(mi):.4f}"
    )
    print(f"top selected tripeptides: {[n.replace('TRI_', '') for n in selected_names[:10]]}")
    return selected, selected_names



def build_svd(tripeptide_full, n_components: int):
    with K.stage_timer(f"TruncatedSVD embedding ({n_components} components)"):
        embedder = K.SvdEmbedding(n_components=n_components)
        embedder.fit(tripeptide_full["train"])
        mats = {split: embedder.transform(tripeptide_full[split]).astype(np.float32)
                for split in ("train", "val", "test")}
        joblib.dump(embedder, C.FEATURE_PIPELINE_DIR / "svd_embedding.joblib")
        print(f"explained variance (cumulative): {embedder.explained_variance_ratio_.sum():.4f}")
        return mats, [f"SVD_{i:03d}" for i in range(mats['train'].shape[1])]


def save_rep(rep: str, mats: dict) -> None:
    for split, mat in mats.items():
        path = C.feature_path(rep, split)
        if rep in C.SPARSE_REPRESENTATIONS:
            sparse.save_npz(path, sparse.csr_matrix(mat))
        else:
            np.save(path, np.asarray(mat, dtype=np.float32))
        print(f"  {rep:16s} {split:5s} {str(mat.shape):16s} {C.human_bytes(path.stat().st_size):>10s} -> {path.name}")


def main() -> None:
    args = parse_args()
    K.banner("STAGE 02 - FEATURE ENGINEERING")
    frames = K.load_splits()
    y = K.labels(frames)
    lengths = lengths_per_split(frames)
    print(f"train/val/test: {len(frames['train']):,}/{len(frames['val']):,}/{len(frames['test']):,}")
    print(f"length medians: train={np.median(lengths['train']):.0f} "
          f"val={np.median(lengths['val']):.0f} test={np.median(lengths['test']):.0f}")

    names: dict[str, list[str]] = {}

    aac, aac_names = build_aac(frames)
    save_rep("aac", aac)
    names["aac"] = aac_names

    dip, dip_names = build_composition(frames, lengths)
    save_rep("dipeptide", dip)
    names["dipeptide"] = dip_names

    tri_full, tri_names = build_tripeptide_counts(frames, lengths)
    save_rep("tripeptide_full", tri_full)
    names["tripeptide_full"] = tri_names

    if args.skip_sweep:
        sweep_table = pd.DataFrame(columns=["k", "cv_macro_f1_mean", "cv_macro_f1_std", "selected"])
        chosen_k = C.SELECT_K
        print(f"skipping sweep, using k={chosen_k}")
    else:
        with K.stage_timer("SelectKBest budget sweep (justifies k experimentally)"):
            sweep_table, chosen_k = sweep_select_k(tri_full, y, args.sweep_folds, tri_names)
        sweep_table.to_csv(C.DATA_PROCESSED / "select_k_sweep.csv", index=False)

    tri_sel, tri_sel_names = select_and_cache(tri_full, y, tri_names, chosen_k)
    save_rep("tripeptide", tri_sel)
    names["tripeptide"] = tri_sel_names
    del tri_sel
    gc.collect()

    svd, svd_names = build_svd(tri_full, C.SVD_COMPONENTS)
    save_rep("svd", svd)
    names["svd"] = svd_names

    with K.stage_timer("concat ablation (AAC + dipeptide + length + SVD)"):
        concat = {}
        for split in ("train", "val", "test"):
            length_feature = (lengths[split] / 1000.0).astype(np.float32).reshape(-1, 1)
            concat[split] = np.hstack(
                [aac[split], dip[split], length_feature, svd[split]]
            ).astype(np.float32)
        save_rep("concat", concat)
        names["concat"] = names["aac"] + names["dipeptide"] + ["SEQ_LENGTH_1000AA"] + names["svd"]

    K.save_json(C.DATA_PROCESSED / "feature_names.json", names)
    for rep, rep_names in names.items():
        dims = K.matrix(rep, "train").shape[1]
        assert len(rep_names) == dims, f"{rep}: {len(rep_names)} names vs {dims} columns"
        print(f"names ok: {rep:16s} {dims:5d} columns")

    C.update_meta(
        select_k=chosen_k,
        feature_dims={rep: int(K.matrix(rep, "train").shape[1]) for rep in names},
        select_k_sweep=sweep_table.to_dict("records"),
    )
    print(f"\nchosen k = {chosen_k}, recorded in {C.META_PATH.name}")
    print("stage 02 complete")


if __name__ == "__main__":
    main()
