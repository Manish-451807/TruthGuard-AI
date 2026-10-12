"""
scripts/prepare_dataset.py

Turns raw data into clean CSV files that the training scripts can read.

Usage (run from the project root):
    python scripts/prepare_dataset.py              # real LIAR files
    python scripts/prepare_dataset.py --use-sample # tiny synthetic demo data

Output (in data/processed/):
    train.csv, test.csv, (valid.csv for LIAR), dataset_info.json
Every CSV has the columns: statement, label, original_label
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Make `import config` work when running this file directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

# LIAR files have no header row, so we name the 14 columns ourselves.
LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "speaker_job",
    "state", "party", "barely_true_count", "false_count", "half_true_count",
    "mostly_true_count", "pants_on_fire_count", "context",
]


def map_label(original_label: str) -> str:
    """Apply the label mode from config. Six-class keeps labels unchanged."""
    if config.LABEL_MODE == "grouped_two_class":
        return config.GROUPED_TWO_CLASS[original_label]
    return original_label


def clean_frame(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Validate labels, drop empty/duplicate statements, apply label mapping."""
    df = df.copy()
    df["statement"] = df["statement"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    unknown = sorted(set(df["label"]) - set(config.LIAR_LABELS))
    if unknown:
        raise ValueError(
            f"{source_name}: unexpected labels {unknown}. "
            f"Expected only {config.LIAR_LABELS}. Is this the right file?"
        )

    before = len(df)
    df = df[df["statement"] != ""]
    df = df.drop_duplicates(subset="statement")
    if df.empty:
        raise ValueError(f"{source_name}: no usable rows after cleaning.")
    print(f"  {source_name}: kept {len(df)} of {before} rows")

    df["original_label"] = df["label"]
    df["label"] = df["original_label"].map(map_label)
    return df[["statement", "label", "original_label"]].reset_index(drop=True)


def load_liar_split(path: Path) -> pd.DataFrame:
    """Read one LIAR .tsv file with checks for missing/empty/malformed files."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. See scripts/download_instructions.md "
            "or run with --use-sample for the demo data."
        )
    if path.stat().st_size == 0:
        raise ValueError(f"{path} is empty.")
    df = pd.read_csv(
        path, sep="\t", header=None, names=LIAR_COLUMNS, index_col=False,
        quoting=3,               # 3 = QUOTE_NONE: LIAR text contains quote marks
        dtype=str, keep_default_na=False,
    )
    return clean_frame(df, path.name)


def remove_leakage(train: pd.DataFrame, others: dict) -> pd.DataFrame:
    """Drop training rows whose statement also appears in valid/test data."""
    held_out = set()
    for frame in others.values():
        held_out.update(frame["statement"])
    leaked = train["statement"].isin(held_out)
    if leaked.any():
        print(f"  Removed {int(leaked.sum())} training rows that also appear "
              "in validation/test data (data leakage).")
    return train[~leaked].reset_index(drop=True)


def prepare_liar() -> tuple:
    print(f"Reading LIAR files from {config.RAW_DATA_DIR}")
    splits = {name: load_liar_split(config.RAW_DATA_DIR / f"{name}.tsv")
              for name in ("train", "valid", "test")}
    splits["train"] = remove_leakage(
        splits["train"], {k: v for k, v in splits.items() if k != "train"})
    info = {"source": "LIAR dataset (Wang, ACL 2017)", "is_demo": False}
    return splits, info


def prepare_sample() -> tuple:
    print(f"Reading DEMO sample from {config.SAMPLE_DATA_PATH}")
    path = config.SAMPLE_DATA_PATH
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Sample file missing or empty: {path}")
    df = pd.read_csv(path)
    missing = {"statement", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"sample_news.csv is missing columns: {sorted(missing)}")
    df = clean_frame(df, path.name)

    # Stratified split keeps every class in both train and test.
    train, test = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED,
        stratify=df["original_label"])
    splits = {"train": train.reset_index(drop=True),
              "test": test.reset_index(drop=True)}
    info = {"source": "Synthetic demonstration data (NOT real research data)",
            "is_demo": True}
    return splits, info


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TruthGuard AI data.")
    parser.add_argument("--use-sample", action="store_true",
                        help="use the tiny synthetic demo dataset")
    args = parser.parse_args()

    try:
        splits, info = prepare_sample() if args.use_sample else prepare_liar()
    except (FileNotFoundError, ValueError) as err:
        print(f"ERROR: {err}")
        sys.exit(1)

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Remove stale split files so an old valid.csv never mixes with a new run.
    # (Only these three names: sentiment_*.csv files are left alone.)
    for name in ("train", "valid", "test"):
        stale = config.PROCESSED_DATA_DIR / f"{name}.csv"
        if stale.exists():
            stale.unlink()
    for name, frame in splits.items():
        frame.to_csv(config.PROCESSED_DATA_DIR / f"{name}.csv", index=False)

    info.update({
        "label_mode": config.LABEL_MODE,
        "labels": sorted(set().union(*[set(f["label"]) for f in splits.values()])),
        "rows": {name: int(len(frame)) for name, frame in splits.items()},
        "random_seed": config.RANDOM_SEED,
    })
    with open(config.PROCESSED_DATA_DIR / "dataset_info.json", "w",
              encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)

    print(f"Done. Files written to {config.PROCESSED_DATA_DIR}")
    if info["is_demo"]:
        print("NOTE: this is DEMO data only. Models trained on it are for "
              "testing the app, not for real conclusions.")


if __name__ == "__main__":
    main()
