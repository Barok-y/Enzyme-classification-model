"""Stage 01 - dataset, exploratory analysis, cleaning and the canonical 70/15/15 split.

Run: ``python src/01_eda_and_cleaning.py [--cap N]``

The split written here is the single source of truth for stages 02-07: it is the
first artefact that exists, and everything downstream loads it instead of
splitting again (see ``data/processed/meta.json``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
import common as K

from collections import Counter
from sklearn.model_selection import train_test_split

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")

EC_PATTERN = re.compile(r"EC:([1-6])")
ANY_EC_PATTERN = re.compile(r"EC:(\d)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA, cleaning and stratified split")
    parser.add_argument("--cap", type=int, default=C.CAP, help="stratified subsample size for low-RAM runs")
    parser.add_argument("--force-download", action="store_true", help="ignore a cached raw parquet file")
    return parser.parse_args()


def load_raw(force_download: bool = False) -> pd.DataFrame:
    """Load the SwissProt-EC records, preferring the cached parquet file."""
    local = C.DATA_RAW / C.HF_FILE
    if local.exists() and not force_download:
        print(f"reading cached raw file: {local}")
        df = pd.read_parquet(local)
    else:
        from datasets import load_dataset

        print(f"downloading {C.HF_DATASET} split={C.HF_SPLIT} from Hugging Face ...")
        dataset = load_dataset(C.HF_DATASET, split=C.HF_SPLIT)
        df = pd.DataFrame(dataset)
        df.to_parquet(local, index=False)
        print(f"cached raw file: {local}")
    print(f"raw records: {len(df)}  columns: {df.columns.tolist()}")
    return df


def extract_primary_ec(labels_entry) -> float:
    """First EC class 1-6 of a record, or NaN when the record carries none.

    ``labels_str`` is stored as the *string* representation of a list such as
    ``"['EC:6.-.-.-', 'EC:6.1.-.-', ...]"``. Matching the regex over the whole
    string keeps wildcard parents (``EC:6.-.-.-``) and multi-label records
    working, and rejects classes outside 1-6 (e.g. EC 7 translocases).
    """
    if isinstance(labels_entry, (list, tuple)):
        text = " ".join(str(item) for item in labels_entry)
    else:
        text = str(labels_entry)
    match = EC_PATTERN.search(text)
    return float(match.group(1)) if match else float("nan")


def top_level_classes(labels_entry) -> tuple[int, ...]:
    if isinstance(labels_entry, (list, tuple)):
        text = " ".join(str(item) for item in labels_entry)
    else:
        text = str(labels_entry)
    return tuple(sorted({int(g) for g in ANY_EC_PATTERN.findall(text)}))


def build_clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats: dict = {"raw_records": int(len(df))}

    missing_seq = int(df["seq"].isna().sum())
    stats["missing_sequence"] = missing_seq

    df = df[df["seq"].notna()].copy()
    df["target"] = df["labels_str"].apply(extract_primary_ec)

    stats["unparsed_label"] = int(df["target"].isna().sum())

    all_digits = Counter()
    for text in df["labels_str"]:
        all_digits.update(int(g) for g in ANY_EC_PATTERN.findall(str(text)))
    stats["top_level_class_counts"] = {str(k): int(v) for k, v in sorted(all_digits.items())}

    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)
    df = df[df["target"].isin(C.CLASSES)].copy()

    ambiguous = int(df["labels_str"].apply(lambda s: len(top_level_classes(s)) > 1).sum())
    stats["records_with_multiple_top_level_classes"] = ambiguous

    seqs = df["seq"].astype(str)
    invalid_mask = seqs.apply(lambda s: set(s) - C.VALID_AA)
    invalid_counts = Counter()
    for chars in invalid_mask:
        invalid_counts.update(chars)
    stats["non_standard_characters"] = {k: int(v) for k, v in sorted(invalid_counts.items())}
    stats["records_with_non_standard_aa"] = int(sum(1 for chars in invalid_mask if chars))

    df["is_standard_aa"] = invalid_mask.apply(lambda c: not c)
    stats["records_dropped_non_standard_aa"] = int((~df["is_standard_aa"]).sum())
    df = df[df["is_standard_aa"]].copy()

    df["seq"] = df["seq"].astype(str).str.upper().str.strip()
    lengths = df["seq"].str.len()
    stats["records_dropped_too_short"] = int((lengths < C.MIN_SEQ_LEN).sum())
    df = df[df["seq"].str.len() >= C.MIN_SEQ_LEN].copy()

    dup_mask = df["seq"].duplicated(keep=False)
    stats["duplicate_sequence_records"] = int(df["seq"].duplicated().sum())
    conflicting_seqs = (
        set(df.loc[dup_mask].groupby("seq")["target"].nunique().loc[lambda s: s > 1].index)
        if dup_mask.any()
        else set()
    )
    stats["duplicate_sequences_with_conflicting_labels"] = len(conflicting_seqs)
    if conflicting_seqs:
        modal = (
            df[df["seq"].isin(conflicting_seqs)]
            .groupby(["seq", "target"])
            .size()
            .rename("votes")
            .reset_index()
            .sort_values(["seq", "votes", "target"], ascending=[True, False, True])
            .drop_duplicates("seq")
            .set_index("seq")["target"]
        )
        df.loc[df["seq"].isin(conflicting_seqs), "target"] = df.loc[
            df["seq"].isin(conflicting_seqs), "seq"
        ].map(modal)
    df = (
        df.sort_values("target")
        .drop_duplicates(subset=["seq"], keep="first")
        .sort_values("id")
        .reset_index(drop=True)
    )

    stats["clean_records"] = int(len(df))
    stats["class_counts"] = {str(k): int(v) for k, v in sorted(df["target"].value_counts().items())}
    return df[["id", "seq", "target"]], stats


def apply_cap(df: pd.DataFrame, cap: int | None) -> pd.DataFrame:
    if not cap or len(df) <= cap:
        return df
    keep, _ = K.stratified_subsample_indices(df["target"].to_numpy(), cap, C.SEED)
    print(f"cap active: stratified subsample {len(keep)} / {len(df)} records (ENZYME_CAP)")
    return df.iloc[np.sort(keep)].reset_index(drop=True)


def make_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    target = df["target"].to_numpy()
    train_df, holdout = train_test_split(
        df,
        test_size=C.TEST_SIZE_HOLDOUT,
        stratify=target,
        random_state=C.SEED,
    )
    val_df, test_df = train_test_split(
        holdout,
        test_size=C.VAL_FRACTION_OF_HOLDOUT,
        stratify=holdout["target"],
        random_state=C.SEED,
    )
    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def composition_matrix(seqs: pd.Series) -> pd.DataFrame:
    counts = np.column_stack([seqs.str.count(aa).to_numpy(dtype=np.float32) for aa in C.AMINO_ACIDS])
    lengths = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    return pd.DataFrame(counts / lengths, columns=list(C.AMINO_ACIDS))


def plot_overview(df: pd.DataFrame) -> None:
    counts = df["target"].value_counts().reindex(C.CLASSES).fillna(0)
    lengths = df["seq"].str.len()

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    axes[0, 0].bar([str(c) for c in C.CLASSES], counts.to_numpy(), color="steelblue", edgecolor="black")
    for i, v in enumerate(counts.to_numpy()):
        axes[0, 0].text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9)
    axes[0, 0].set_xlabel("EC class")
    axes[0, 0].set_ylabel("Number of sequences")
    axes[0, 0].set_title("Class distribution (log scale)")
    axes[0, 0].set_yscale("log")

    axes[0, 1].bar([str(c) for c in C.CLASSES], counts.to_numpy(), color="indianred", edgecolor="black")
    axes[0, 1].set_xlabel("EC class")
    axes[0, 1].set_ylabel("Number of sequences")
    axes[0, 1].set_title("Class distribution (linear scale)")
    ratio = counts.max() / max(counts.min(), 1)
    axes[0, 1].set_title(f"Class distribution - majority/minority ratio = {ratio:.0f}x")

    axes[1, 0].hist(lengths, bins=60, color="coral", edgecolor="black", alpha=0.75)
    axes[1, 0].set_xlabel("Sequence length (aa)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Sequence length distribution")
    axes[1, 0].set_yscale("log")

    sample_rows = []
    for cls in C.CLASSES:
        block = lengths[df["target"].to_numpy() == cls]
        take = min(3000, len(block))
        block = block.sample(take, random_state=C.SEED)
        sample_rows.append(pd.DataFrame({"seq_length": block.to_numpy(), "target": cls}))
    sample = pd.concat(sample_rows, ignore_index=True)
    sns.boxplot(data=sample, x="target", y="seq_length", ax=axes[1, 1], color="#E5989B")
    axes[1, 1].set_xlabel("EC class")
    axes[1, 1].set_ylabel("Sequence length (aa)")
    axes[1, 1].set_title("Sequence length by class (log y)")
    axes[1, 1].set_yscale("log")

    fig.suptitle("EDA - SwissProt-EC first-level enzyme class", fontsize=15, y=0.995)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "eda_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_composition(df: pd.DataFrame) -> None:
    comp = composition_matrix(df["seq"])
    comp["target"] = df["target"].to_numpy()
    profile = comp.groupby("target").mean().reindex(C.CLASSES)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    sns.heatmap(
        profile,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        ax=axes[0],
        cbar_kws={"label": "mean frequency"},
    )
    axes[0].set_xlabel("Amino acid")
    axes[0].set_ylabel("EC class")
    axes[0].set_title("Mean amino-acid composition per EC class")

    stacked = df["target"].value_counts().reindex(C.CLASSES).fillna(0)
    bottom = 0.0
    colours = sns.color_palette("tab10", n_colors=len(C.CLASSES))
    for cls, colour in zip(C.CLASSES, colours):
        share = stacked.loc[cls] / stacked.sum()
        axes[1].bar([0], [share], bottom=bottom, color=colour, edgecolor="white", label=f"EC {cls}")
        bottom += share
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(["all records"])
    axes[1].set_ylabel("Share of dataset")
    axes[1].set_title("Class proportions (drives the imbalance strategy)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(C.FIGURES / "eda_composition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_splits(splits: dict[str, pd.DataFrame]) -> None:
    table = pd.DataFrame(
        {split: frame["target"].value_counts().reindex(C.CLASSES).fillna(0).astype(int)
         for split, frame in splits.items()}
    )
    shares = table.div(table.sum(axis=0), axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    table.plot(kind="bar", ax=axes[0], color=["#4C72B0", "#DD8452", "#C44E52"], edgecolor="black")
    axes[0].set_xlabel("EC class")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Split sizes per class (stratified 70/15/15)")
    axes[0].set_xticklabels([str(c) for c in C.CLASSES], rotation=0)
    axes[0].legend(title="split")

    shares.plot(kind="bar", ax=axes[1], color=["#4C72B0", "#DD8452", "#C44E52"], edgecolor="black")
    axes[1].set_xlabel("EC class")
    axes[1].set_ylabel("Share within split")
    axes[1].set_title("Class proportions are preserved across splits")
    axes[1].set_xticklabels([str(c) for c in C.CLASSES], rotation=0)
    axes[1].axhline(1 / len(C.CLASSES), ls="--", c="grey", lw=1, label="balanced reference")
    axes[1].legend(title="split")

    fig.tight_layout()
    fig.savefig(C.FIGURES / "eda_splits.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    K.banner("STAGE 01 - DATA, EDA, CLEANING AND STRATIFIED SPLIT")

    raw = load_raw(force_download=args.force_download)
    with K.stage_timer("raw schema"):
        print(raw.head(3).to_string(max_colwidth=60))
        lengths = raw["seq"].astype(str).str.len()
        print(
            f"length: min={lengths.min()} q25={int(lengths.quantile(0.25))} "
            f"median={int(lengths.median())} q75={int(lengths.quantile(0.75))} "
            f"p99={int(lengths.quantile(0.99))} max={lengths.max()}"
        )

    with K.stage_timer("target extraction and cleaning"):
        clean, stats = build_clean_dataset(raw)
        print(json.dumps(stats, indent=2))

    with K.stage_timer("optional stratified cap"):
        clean = apply_cap(clean, args.cap)

    with K.stage_timer("stratified 70/15/15 split"):
        splits = make_splits(clean)
        for split, frame in splits.items():
            frame.to_parquet(C.SPLIT_PATHS[split], index=False)
            counts = frame["target"].value_counts().reindex(C.CLASSES).fillna(0).astype(int)
            print(f"{split:5s} n={len(frame):7,d}  {dict(counts.to_dict())}")

    with K.stage_timer("EDA figures"):
        plot_overview(clean)
        plot_composition(clean)
        plot_splits(splits)
        print(f"figures -> {C.FIGURES.relative_to(C.ROOT)}")

    summary = {
        **stats,
        "capped": bool(args.cap and args.cap < stats["clean_records"]),
        "sample_cap": args.cap if (args.cap and args.cap < stats["clean_records"]) else None,
        "split_sizes": {split: int(len(frame)) for split, frame in splits.items()},
        "split_class_counts": {
            split: {str(k): int(v) for k, v in frame["target"].value_counts().reindex(C.CLASSES).fillna(0).to_dict().items()}
            for split, frame in splits.items()
        },
        "select_k": None,
        "seed": C.SEED,
        "min_seq_len": C.MIN_SEQ_LEN,
        "hf_dataset": C.HF_DATASET,
        "hf_split": C.HF_SPLIT,
        "imbalance_ratio": (
            float(max(stats["class_counts"].values()) / max(min(stats["class_counts"].values()), 1))
        ),
    }
    C.update_meta(**summary)
    K.save_json(C.DATA_PROCESSED / "dataset_summary.json", summary)
    print(f"\nmeta -> {C.META_PATH.relative_to(C.ROOT)}")
    print("stage 01 complete")


if __name__ == "__main__":
    main()
